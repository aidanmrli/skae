"""Outcome-blind GPU-utilization adjudication for residual forecast jobs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.neurips_2026.global_k_residual_forecast.protocol import (
    DEFAULT_CARD,
    DEFAULT_SOURCES,
    DEFAULT_TASKS,
    authenticate_checkpoint_roster,
    authenticate_v2_inputs,
    atomic_json,
    load_frozen_protocol,
    load_json,
)


TRACE_FIELDS = [
    "epoch_seconds",
    "gpu_uuid",
    "gpu_name",
    "utilization_gpu",
    "memory_used_mib",
    "memory_total_mib",
]


def _read_trace(path: Path) -> dict[str, Any]:
    epochs, utilization, memory_used, memory_total = [], [], [], []
    uuids, names = [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != TRACE_FIELDS:
            raise RuntimeError(f"Telemetry schema drift in {path}: {reader.fieldnames}")
        for row in reader:
            epochs.append(float(row["epoch_seconds"]))
            utilization.append(float(row["utilization_gpu"]))
            memory_used.append(float(row["memory_used_mib"]))
            memory_total.append(float(row["memory_total_mib"]))
            uuids.append(row["gpu_uuid"].strip())
            names.append(row["gpu_name"].strip())
    if len(epochs) < 2:
        raise RuntimeError(f"Telemetry trace needs at least two samples: {path}")
    arrays = {
        "epoch": np.asarray(epochs, dtype=np.float64),
        "utilization": np.asarray(utilization, dtype=np.float64),
        "memory_used": np.asarray(memory_used, dtype=np.float64),
        "memory_total": np.asarray(memory_total, dtype=np.float64),
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError(f"Non-finite telemetry value in {path}")
    if not np.all(np.diff(arrays["epoch"]) > 0):
        raise RuntimeError(f"Telemetry timestamps are not strictly increasing: {path}")
    if not np.all((arrays["utilization"] >= 0) & (arrays["utilization"] <= 100)):
        raise RuntimeError(f"GPU utilization outside [0,100]: {path}")
    if not np.all(
        (arrays["memory_used"] >= 0)
        & (arrays["memory_total"] > 0)
        & (arrays["memory_used"] <= arrays["memory_total"])
    ):
        raise RuntimeError(f"GPU memory telemetry is outside its physical range: {path}")
    if len(set(uuids)) != 1 or len(set(names)) != 1 or not uuids[0] or not names[0]:
        raise RuntimeError(f"Telemetry does not identify exactly one stable GPU: {path}")
    if "A100" not in names[0] or float(arrays["memory_total"][0]) < 75 * 1024:
        raise RuntimeError(f"Telemetry is not from an A100 with at least 75 GiB: {path}")
    if not np.all(arrays["memory_total"] == arrays["memory_total"][0]):
        raise RuntimeError(f"GPU memory capacity changed within telemetry trace: {path}")
    return {**arrays, "gpu_uuid": uuids[0], "gpu_name": names[0]}


def _minimum_time_rolling(
    epoch: np.ndarray, values: np.ndarray, window_seconds: float,
) -> float:
    if epoch.size != values.size or epoch.size < 2 or epoch[-1] - epoch[0] < window_seconds:
        return float("nan")
    means = []
    for start_index, start in enumerate(epoch):
        target = start + float(window_seconds)
        if target > epoch[-1]:
            break
        end_index = int(np.searchsorted(epoch, target, side="right"))
        if end_index - start_index >= 2:
            means.append(float(values[start_index:end_index].mean()))
    return min(means) if means else float("nan")


def _assess_one(
    trace_path: Path,
    window_path: Path,
    thresholds: dict[str, Any],
    *,
    protocol_id: str,
    mode: str,
    task_id: int,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    window = load_json(window_path)
    expected_window_keys = {
        "schema_version",
        "protocol_id",
        "artifact_role",
        "mode",
        "task_id",
        "start_epoch_seconds",
        "end_epoch_seconds",
        "elapsed_seconds",
    }
    if (
        set(window) != expected_window_keys
        or window.get("schema_version") != 1
        or window.get("protocol_id") != protocol_id
        or window.get("artifact_role") != "forecast_compute_window"
        or window.get("mode") != mode
        or int(window.get("task_id", -1)) != int(task_id)
    ):
        raise RuntimeError(f"Compute-window identity or schema drifted: {window_path}")
    start = float(window["start_epoch_seconds"])
    end = float(window["end_epoch_seconds"])
    elapsed = float(window["elapsed_seconds"])
    if (
        not np.isfinite([start, end, elapsed]).all()
        or not start < end
        or elapsed <= 0
        or not np.isclose(elapsed, end - start, rtol=0.0, atol=1e-6)
    ):
        raise RuntimeError(f"Compute-window endpoints are invalid: {window_path}")
    trace = _read_trace(trace_path)
    epoch = trace["epoch"]
    utilization = trace["utilization"]
    before = np.flatnonzero(epoch <= start)
    after = np.flatnonzero(epoch >= end)
    if before.size == 0 or after.size == 0:
        raise RuntimeError(f"Telemetry does not bracket the compute interval: {trace_path}")
    leading_bracket_gap = float(start - epoch[before[-1]])
    trailing_bracket_gap = float(epoch[after[0]] - end)
    inside = (epoch >= start) & (epoch <= end)
    retained = utilization[inside]
    if retained.size < 2:
        raise RuntimeError(f"Fewer than two telemetry samples inside {window_path}")
    retained_epoch = epoch[inside]
    intervals = np.diff(epoch)
    retained_intervals = np.diff(retained_epoch)
    rolling_seconds = float(thresholds["rolling_window_seconds"])
    rolling_minimum = _minimum_time_rolling(
        retained_epoch, retained, rolling_seconds
    )
    maximum_gap = float(thresholds["maximum_sampling_gap_seconds"])
    checks = {
        "minimum_compute_window_samples": retained.size
        >= int(thresholds["minimum_compute_window_samples"]),
        "minimum_compute_window_duration": elapsed
        >= float(thresholds["minimum_compute_window_duration_seconds"]),
        "trace_brackets_compute_start": leading_bracket_gap <= maximum_gap,
        "trace_brackets_compute_end": trailing_bracket_gap <= maximum_gap,
        "allocation_sampling_cadence": float(np.median(intervals))
        >= float(thresholds["minimum_median_sampling_interval_seconds"])
        and float(np.median(intervals))
        <= float(thresholds["maximum_median_sampling_interval_seconds"]),
        "compute_sampling_cadence": retained_intervals.size > 0
        and float(np.median(retained_intervals))
        >= float(thresholds["minimum_median_sampling_interval_seconds"])
        and float(np.median(retained_intervals))
        <= float(thresholds["maximum_median_sampling_interval_seconds"]),
        "allocation_maximum_sampling_gap": float(intervals.max())
        <= maximum_gap,
        "compute_maximum_sampling_gap": retained_intervals.size > 0
        and float(retained_intervals.max())
        <= maximum_gap,
        "minimum_compute_window_mean_utilization": float(retained.mean())
        >= float(thresholds["minimum_compute_window_mean_utilization_percent"]),
        "minimum_compute_window_p10_utilization": float(np.quantile(retained, 0.10))
        >= float(thresholds["minimum_compute_window_p10_utilization_percent"]),
        "minimum_rolling_utilization": np.isfinite(rolling_minimum)
        and rolling_minimum
        >= float(thresholds["minimum_rolling_utilization_percent"]),
        "minimum_allocation_wide_mean_utilization": float(utilization.mean())
        >= float(thresholds["minimum_allocation_wide_mean_utilization_percent"]),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "artifact_role": "outcome_blind_gpu_task_assessment",
        "mode": mode,
        "task_id": int(task_id),
        "freeze": freeze,
        "trace_path": str(trace_path),
        "compute_window_path": str(window_path),
        "allocation_sample_count": int(utilization.size),
        "compute_window_sample_count": int(retained.size),
        "gpu_uuid": trace["gpu_uuid"],
        "gpu_name": trace["gpu_name"],
        "allocation_duration_seconds": float(epoch[-1] - epoch[0]),
        "compute_window_duration_seconds": elapsed,
        "trace_to_compute_start_gap_seconds": leading_bracket_gap,
        "trace_after_compute_end_gap_seconds": trailing_bracket_gap,
        "allocation_median_sampling_interval_seconds": float(np.median(intervals)),
        "compute_median_sampling_interval_seconds": float(np.median(retained_intervals)),
        "allocation_maximum_sampling_gap_seconds": float(intervals.max()),
        "compute_maximum_sampling_gap_seconds": float(retained_intervals.max()),
        "allocation_mean_utilization_percent": float(utilization.mean()),
        "compute_window_mean_utilization_percent": float(retained.mean()),
        "compute_window_p10_utilization_percent": float(np.quantile(retained, 0.10)),
        "minimum_rolling_utilization_percent": (
            rolling_minimum if np.isfinite(rolling_minimum) else None
        ),
        "peak_memory_mib": float(trace["memory_used"].max()),
        "memory_total_mib": float(trace["memory_total"][0]),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_smoke_shard(
    path: Path, *, protocol_id: str, freeze: dict[str, Any]
) -> dict[str, bool]:
    smoke = load_json(path)
    expected_keys = {
        "schema_version",
        "protocol_id",
        "artifact_role",
        "task_id",
        "all_required_predictions_finite",
        "exact_method_count",
        "route_fit_completed",
        "route_audit_completed",
        "null_scale_matching_completed",
        "forecast_metrics_labels_and_alignment_values_persisted",
        "outcomes_inspected",
        "elapsed_seconds",
        "freeze",
    }
    elapsed = smoke.get("elapsed_seconds")
    return {
        "exact_schema": set(smoke) == expected_keys,
        "schema_version": smoke.get("schema_version") == 1,
        "protocol_id": smoke.get("protocol_id") == protocol_id,
        "artifact_role": smoke.get("artifact_role") == "outcome_blind_gpu_smoke",
        "task_id": smoke.get("task_id") == 0,
        "freeze": smoke.get("freeze") == freeze,
        "all_required_predictions_finite": smoke.get(
            "all_required_predictions_finite"
        ) is True,
        "exact_method_count": smoke.get("exact_method_count") == 41,
        "route_fit_completed": smoke.get("route_fit_completed") is True,
        "route_audit_completed": smoke.get("route_audit_completed") is True,
        "null_scale_matching_completed": smoke.get("null_scale_matching_completed")
        is True,
        "no_metrics_labels_or_alignment_persisted": smoke.get(
            "forecast_metrics_labels_and_alignment_values_persisted"
        ) is False,
        "outcomes_not_inspected": smoke.get("outcomes_inspected") is False,
        "finite_positive_elapsed_seconds": isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and np.isfinite(elapsed)
        and elapsed > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "scientific"), required=True)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--expected-task-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    card, tasks, freeze = load_frozen_protocol(
        card_path=args.card,
        task_path=args.tasks,
        source_manifest_path=args.sources,
        expected_card_sha256=args.expected_card_sha256,
        expected_task_sha256=args.expected_task_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
    )
    authenticate_v2_inputs(card)
    authenticate_checkpoint_roster(tasks)
    task_ids = [0] if args.mode == "smoke" else [row["task_id"] for row in tasks["tasks"]]
    rows = []
    for task_id in task_ids:
        directory = args.output_root / args.mode
        rows.append(
            _assess_one(
                directory / "telemetry" / f"task_{int(task_id):02d}.csv",
                directory / "compute_windows" / f"task_{int(task_id):02d}.json",
                card["gpu_utilization_gate"],
                protocol_id=card["protocol_id"],
                mode=args.mode,
                task_id=int(task_id),
                freeze=freeze,
            )
        )
    smoke_checks = None
    if args.mode == "smoke":
        smoke_path = args.output_root / "smoke" / "shards" / "task_00.json"
        smoke_checks = _validate_smoke_shard(
            smoke_path, protocol_id=card["protocol_id"], freeze=freeze
        )
    payload = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": f"outcome_blind_{args.mode}_gpu_assessment",
        "mode": args.mode,
        "freeze": freeze,
        "rows": rows,
        "smoke_checks": smoke_checks,
        "passed": (
            len(rows) == len(task_ids)
            and all(row["passed"] for row in rows)
            and (smoke_checks is None or all(smoke_checks.values()))
        ),
        "forecast_outcomes_read": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "passed" if payload["passed"] else "failed"}))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
