"""Authenticate marker-bounded GPU telemetry without reading forecast outcomes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    duplicate_safe_json,
    sha256_path,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.telemetry_bindings import (
    authenticate_runtime_bindings,
)
RAW_TELEMETRY_NAME = "raw_gpu_telemetry.csv"
RUNTIME_LINEAGE_NAME = "runtime_lineage.json"
GUARD_NAME = "outcome_guard_receipt.json"
AUDIT_NAME = "telemetry_audit.json"
WINDOW_MARKERS = {
    "selection_validity": ("selection_start", "selection_end"),
    "evaluation_validity": ("evaluation_start", "evaluation_end"),
}

REQUIRED_HARDWARE_PLAN: dict[str, int | float | str | bool] = {
    "device_name": "NVIDIA A100L 80GB",
    "boundary_samples_excluded_per_side": 1,
    "minimum_all_window_samples_before_boundary_exclusion": 12,
    "minimum_retained_all_window_samples": 10,
    "minimum_mean_retained_all_window_gpu_utilization_percent": 90.0,
    "strict_p10_retained_all_window_gpu_utilization_percent_above": 80.0,
    "minimum_median_sample_cadence_seconds": 0.5,
    "maximum_median_sample_cadence_seconds": 1.5,
    "maximum_sample_gap_seconds": 2.0,
    "maximum_marker_edge_gap_seconds": 2.0,
    "maximum_peak_memory_fraction": 0.8,
    "no_padding": True,
}
def _parse_epoch(timestamp: str) -> float:
    value = timestamp.strip()
    for format_string in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value, format_string).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse nvidia-smi timestamp {timestamp!r}")


def _number(value: str) -> float:
    cleaned = "".join(character for character in value if character in "0123456789.-")
    if not cleaned:
        raise ValueError(f"Cannot parse telemetry number {value!r}")
    number = float(cleaned)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite telemetry number {value!r}")
    return number


def read_samples(path: Path) -> list[dict[str, Any]]:
    """Read the six-field raw nvidia-smi stream and validate one stable GPU."""

    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if not row:
                continue
            if row[0].strip().lower() == "timestamp":
                continue
            if len(row) != 6:
                raise ValueError(f"Telemetry row does not have six fields: {row}")
            rows.append(row)
    if not rows:
        raise ValueError("Raw GPU telemetry is empty")

    samples = [
        {
            "epoch_seconds": _parse_epoch(row[0]),
            "gpu_uuid": row[1].strip(),
            "gpu_name": row[2].strip(),
            "utilization_percent": _number(row[3]),
            "memory_used_mib": _number(row[4]),
            "memory_total_mib": _number(row[5]),
        }
        for row in rows
    ]
    epochs = np.asarray([row["epoch_seconds"] for row in samples], dtype=np.float64)
    utilization = np.asarray(
        [row["utilization_percent"] for row in samples], dtype=np.float64
    )
    used = np.asarray([row["memory_used_mib"] for row in samples], dtype=np.float64)
    total = np.asarray([row["memory_total_mib"] for row in samples], dtype=np.float64)
    uuids = {str(row["gpu_uuid"]) for row in samples}
    names = {str(row["gpu_name"]) for row in samples}
    if len(uuids) != 1 or not next(iter(uuids)).startswith("GPU-"):
        raise ValueError("Telemetry must bind exactly one valid GPU UUID")
    if len(names) != 1 or not next(iter(names)):
        raise ValueError("Telemetry must bind exactly one nonempty GPU name")
    if epochs.size > 1 and not np.all(np.diff(epochs) > 0):
        raise ValueError("Telemetry timestamps must be strictly increasing")
    if not np.all((utilization >= 0.0) & (utilization <= 100.0)):
        raise ValueError("GPU utilization lies outside [0,100]")
    if not np.all((total > 0.0) & (used >= 0.0) & (used <= total)):
        raise ValueError("GPU memory telemetry is physically invalid")
    if not np.all(total == total[0]):
        raise ValueError("GPU memory capacity changed within one telemetry stream")
    return samples


def hardware_plan(card: dict[str, Any]) -> dict[str, Any]:
    plan = card.get("hardware_plan")
    if not isinstance(plan, dict):
        raise RuntimeError("Prediction card lacks the frozen hardware_plan")
    for key, expected in REQUIRED_HARDWARE_PLAN.items():
        observed = plan.get(key)
        if isinstance(expected, float):
            matches = (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and float(observed) == expected
            )
        else:
            matches = observed == expected
        if not matches:
            raise RuntimeError(
                f"Frozen hardware gate drifted for {key}: {observed!r} != {expected!r}"
            )
    return plan


def window_statistics(
    samples: list[dict[str, Any]], *, start: float, end: float
) -> dict[str, Any]:
    """Summarize the marker interval after exactly one boundary drop per side."""

    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("Evaluation marker interval is invalid")
    selected = [
        row for row in samples if start <= float(row["epoch_seconds"]) <= end
    ]
    if not selected:
        raise ValueError("No telemetry samples fall inside the evaluation interval")
    epochs = np.asarray([row["epoch_seconds"] for row in selected], dtype=np.float64)
    if epochs.size > 1 and not np.all(np.diff(epochs) > 0):
        raise ValueError("In-window telemetry timestamps must be strictly increasing")

    # This exclusion is unconditional: it never depends on utilization or memory.
    retained = selected[1:-1]
    utilization = np.asarray(
        [row["utilization_percent"] for row in retained], dtype=np.float64
    )
    memory_fraction = np.asarray(
        [row["memory_used_mib"] / row["memory_total_mib"] for row in retained],
        dtype=np.float64,
    )
    gaps = np.diff(epochs)
    return {
        "start_epoch_seconds": float(start),
        "end_epoch_seconds": float(end),
        "duration_seconds": float(end - start),
        "all_window_samples": int(len(selected)),
        "boundary_samples_excluded_per_side": 1,
        "retained_all_window_samples": int(len(retained)),
        "zero_utilization_retained_samples": int(np.sum(utilization == 0.0)),
        "mean_retained_all_window_gpu_utilization_percent": (
            float(utilization.mean()) if utilization.size else 0.0
        ),
        "p10_retained_all_window_gpu_utilization_percent": (
            float(np.quantile(utilization, 0.10)) if utilization.size else 0.0
        ),
        "median_sample_cadence_seconds": (
            float(np.median(gaps)) if gaps.size else float("inf")
        ),
        "maximum_sample_gap_seconds": (
            float(gaps.max()) if gaps.size else float("inf")
        ),
        "leading_marker_edge_gap_seconds": float(epochs[0] - start),
        "trailing_marker_edge_gap_seconds": float(end - epochs[-1]),
        "peak_memory_fraction": (
            float(memory_fraction.max()) if memory_fraction.size else float("inf")
        ),
        "utilization_filter_applied": False,
    }


def gate_checks(window: dict[str, Any], plan: dict[str, Any]) -> dict[str, bool]:
    cadence = float(window["median_sample_cadence_seconds"])
    maximum_edge = float(plan["maximum_marker_edge_gap_seconds"])
    checks = {
        "exact_unconditional_boundary_exclusion": (
            window["boundary_samples_excluded_per_side"]
            == plan["boundary_samples_excluded_per_side"]
            == 1
        ),
        "no_utilization_filter": window["utilization_filter_applied"] is False,
        "minimum_all_window_samples": window["all_window_samples"]
        >= int(plan["minimum_all_window_samples_before_boundary_exclusion"]),
        "minimum_retained_all_window_samples": window["retained_all_window_samples"]
        >= int(plan["minimum_retained_all_window_samples"]),
        "minimum_mean_retained_utilization": window[
            "mean_retained_all_window_gpu_utilization_percent"
        ]
        >= float(plan["minimum_mean_retained_all_window_gpu_utilization_percent"]),
        "strict_p10_retained_utilization": window[
            "p10_retained_all_window_gpu_utilization_percent"
        ]
        > float(plan["strict_p10_retained_all_window_gpu_utilization_percent_above"]),
        "median_sample_cadence": float(
            plan["minimum_median_sample_cadence_seconds"]
        )
        <= cadence
        <= float(plan["maximum_median_sample_cadence_seconds"]),
        "maximum_sample_gap": window["maximum_sample_gap_seconds"]
        <= float(plan["maximum_sample_gap_seconds"]),
        "leading_marker_edge_coverage": 0.0
        <= window["leading_marker_edge_gap_seconds"]
        <= maximum_edge,
        "trailing_marker_edge_coverage": 0.0
        <= window["trailing_marker_edge_gap_seconds"]
        <= maximum_edge,
        "strict_peak_memory_fraction": window["peak_memory_fraction"]
        < float(plan["maximum_peak_memory_fraction"]),
    }
    return {name: bool(passed) for name, passed in checks.items()}


def _marker(
    root: Path,
    relative_path: Path,
    *,
    expected_stage: str,
    card_hash: str,
    source_hash: str,
    slurm_job_id: str,
) -> tuple[dict[str, Any], Path]:
    path = root / relative_path
    marker = duplicate_safe_json(path)
    if (
        marker.get("stage") != expected_stage
        or marker.get("card_sha256") != card_hash
        or marker.get("source_manifest_sha256") != source_hash
        or str(marker.get("slurm_job_id")) != slurm_job_id
    ):
        raise RuntimeError(f"Marker lineage failed for {expected_stage}")
    epoch = marker.get("epoch_seconds")
    if (
        not isinstance(epoch, (int, float))
        or isinstance(epoch, bool)
        or not math.isfinite(float(epoch))
    ):
        raise RuntimeError(f"Marker epoch is invalid for {expected_stage}")
    return marker, path


def _frozen_output_root(card: dict[str, Any]) -> Path:
    candidates = [
        card.get("output_root"),
        card.get("execution", {}).get("output_root")
        if isinstance(card.get("execution"), dict)
        else None,
        card.get("prospective_datasets", {}).get("output_root")
        if isinstance(card.get("prospective_datasets"), dict)
        else None,
    ]
    roots = {str(value) for value in candidates if isinstance(value, str) and value}
    if len(roots) != 1:
        raise RuntimeError("Prediction card must declare exactly one frozen output root")
    return Path(next(iter(roots)))
def audit_and_issue_guard(
    *,
    root: Path,
    card_path: Path,
    expected_card_sha256: str,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    raw_telemetry: Path,
) -> dict[str, Any]:
    """Issue a summary guard after hash-only inspection of scientific artifacts."""

    if sha256_path(card_path) != expected_card_sha256:
        raise RuntimeError("Prediction-card hash differs from the authorized freeze")
    if sha256_path(source_manifest) != expected_source_manifest_sha256:
        raise RuntimeError("Source-manifest hash differs from the authorized freeze")
    card = duplicate_safe_json(card_path)
    plan = hardware_plan(card)
    if root != _frozen_output_root(card):
        raise RuntimeError("Output root differs from the frozen prediction card")
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    if not slurm_job_id:
        raise RuntimeError("Telemetry authentication requires the scientific SLURM job")

    runtime_path = root / RUNTIME_LINEAGE_NAME
    runtime = duplicate_safe_json(runtime_path)
    required_runtime = {
        "status": "scientific_payload_written_but_not_authorized_for_summary",
        "card_sha256": expected_card_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "slurm_job_id": slurm_job_id,
        "scientific_metrics_printed": False,
    }
    environment = runtime.get("environment")
    if (
        any(runtime.get(key) != value for key, value in required_runtime.items())
        or not isinstance(environment, dict)
        or str(environment.get("slurm_job_id")) != slurm_job_id
    ):
        raise RuntimeError("Metric-free runtime lineage does not match the scientific job")
    bindings = authenticate_runtime_bindings(
        card,
        card_path,
        runtime,
        root,
        card_hash=expected_card_sha256,
        source_hash=expected_source_manifest_sha256,
    )
    samples = read_samples(raw_telemetry)
    first = samples[0]
    if (
        "A100" not in str(first["gpu_name"])
        or float(first["memory_total_mib"]) < float(75 * 1024)
    ):
        raise RuntimeError("Telemetry is not from the frozen A100L 80 GB device class")
    environment_gpu_name = str(environment.get("gpu_name", ""))
    if environment_gpu_name != str(first["gpu_name"]):
        raise RuntimeError("Runtime and telemetry GPU names differ")
    markers: dict[str, dict[str, Any]] = {}
    marker_epochs: dict[str, float] = {}
    windows: dict[str, dict[str, Any]] = {}
    checks: dict[str, dict[str, bool]] = {}
    for window_name, (start_stage, end_stage) in WINDOW_MARKERS.items():
        start, start_path = _marker(
            root, Path(f"markers/{start_stage}.json"),
            expected_stage=start_stage, card_hash=expected_card_sha256,
            source_hash=expected_source_manifest_sha256, slurm_job_id=slurm_job_id,
        )
        end, end_path = _marker(
            root, Path(f"markers/{end_stage}.json"),
            expected_stage=end_stage, card_hash=expected_card_sha256,
            source_hash=expected_source_manifest_sha256, slurm_job_id=slurm_job_id,
        )
        markers[start_stage] = {"path": str(start_path), "sha256": sha256_path(start_path)}
        markers[end_stage] = {"path": str(end_path), "sha256": sha256_path(end_path)}
        marker_epochs[start_stage] = float(start["epoch_seconds"])
        marker_epochs[end_stage] = float(end["epoch_seconds"])
        windows[window_name] = window_statistics(
            samples, start=float(start["epoch_seconds"]), end=float(end["epoch_seconds"])
        )
        checks[window_name] = gate_checks(windows[window_name], plan)
        if end_stage == "evaluation_end" and end.get(
            "scientific_payload_sha256"
        ) != bindings["scientific_payload_sha256"]:
            raise RuntimeError("Evaluation-end marker lacks the scientific payload binding")
    compute_end, compute_end_path = _marker(
        root, Path("markers/evaluation_compute_end.json"),
        expected_stage="evaluation_compute_end", card_hash=expected_card_sha256,
        source_hash=expected_source_manifest_sha256, slurm_job_id=slurm_job_id,
    )
    markers["evaluation_compute_end"] = {
        "path": str(compute_end_path), "sha256": sha256_path(compute_end_path)
    }
    if float(compute_end["epoch_seconds"]) != marker_epochs["evaluation_end"]:
        raise RuntimeError("Evaluation compute-end and sealed end epochs differ")
    if not marker_epochs["selection_end"] < marker_epochs["evaluation_start"]:
        raise RuntimeError("Selection and evaluation telemetry windows are not disjoint")
    all_window_checks_passed = all(all(row.values()) for row in checks.values())
    utilization = np.asarray(
        [sample["utilization_percent"] for sample in samples], dtype=np.float64
    )
    allocation = {
        "sample_count": int(utilization.size),
        "duration_seconds": float(samples[-1]["epoch_seconds"] - samples[0]["epoch_seconds"]),
        "mean_gpu_utilization_percent_descriptive": float(utilization.mean()),
    }

    raw_copy = root / RAW_TELEMETRY_NAME
    if raw_copy.exists():
        raise FileExistsError(raw_copy)
    with raw_telemetry.open("rb") as source, raw_copy.open("xb") as target:
        shutil.copyfileobj(source, target)
    audit_path = root / AUDIT_NAME
    audit = {
        "schema_version": 1,
        "protocol_id": card.get("protocol_id"),
        "status": "passed" if all_window_checks_passed else "failed",
        "card_sha256": expected_card_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "slurm_job_id": slurm_job_id,
        "gpu_uuid": first["gpu_uuid"],
        "gpu_name": first["gpu_name"],
        "raw_telemetry_path": str(raw_copy),
        "raw_telemetry_sha256": sha256_path(raw_copy),
        "validity_windows": windows,
        "validity_checks": checks,
        "marker_bindings": markers,
        "whole_allocation_descriptive": allocation,
        "both_disjoint_windows_required": True,
        "disjoint_window_ordering_verified": True,
        "evaluation_end_epoch_identity_verified": True,
        "gpu_identity_binding": {
            "runtime_gpu_name": environment_gpu_name,
            "telemetry_gpu_name": first["gpu_name"],
            "same_name": True,
            "same_slurm_job_id": True,
            "one_telemetry_uuid_stream": True,
            "runtime_uuid": str(environment.get("gpu_uuid", "not_recorded")),
            "telemetry_uuid": first["gpu_uuid"],
            "uuid_equality_required": False,
            "rationale": "PyTorch and nvidia-smi UUID renderings may differ; one visible GPU, one telemetry UUID, an exact name match, and one SLURM job bind identity.",
        },
        "every_retained_sample_including_zero_used": True,
        "scientific_payload_opened": False,
        "forecast_outcomes_accessed": False,
    }
    write_json_once(audit_path, audit)
    if not all_window_checks_passed:
        raise RuntimeError("A disjoint GPU-utilization window failed; no guard issued")

    guard_path = root / GUARD_NAME
    guard = {
        "schema_version": 1,
        "protocol_id": card.get("protocol_id"),
        "status": "authorized_for_dependent_cpu_summary",
        "card_sha256": expected_card_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "slurm_job_id": slurm_job_id,
        "runtime_lineage_path": str(runtime_path),
        "runtime_lineage_sha256": sha256_path(runtime_path),
        **bindings,
        "marker_bindings": markers,
        "validity_windows": windows,
        "validity_checks": checks,
        "whole_allocation_descriptive": allocation,
        "both_disjoint_windows_required": True,
        "disjoint_window_ordering_verified": True,
        "evaluation_end_epoch_identity_verified": True,
        "gpu_identity_binding": audit["gpu_identity_binding"],
        "raw_telemetry_path": str(raw_copy),
        "raw_telemetry_sha256": sha256_path(raw_copy),
        "telemetry_audit_path": str(audit_path),
        "telemetry_audit_sha256": sha256_path(audit_path),
        "gpu_uuid": first["gpu_uuid"],
        "scientific_payload_opened": False,
        "forecast_outcomes_accessed": False,
    }
    write_json_once(guard_path, guard)
    return {
        "status": "passed",
        "outcome_guard_receipt_path": str(guard_path),
        "outcome_guard_receipt_sha256": sha256_path(guard_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--raw-telemetry", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_telemetry = args.raw_telemetry or args.root / "raw_gpu_telemetry_unverified.csv"
    result = audit_and_issue_guard(
        root=args.root,
        card_path=args.card,
        expected_card_sha256=args.expected_card_sha256,
        source_manifest=args.source_manifest,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        raw_telemetry=raw_telemetry,
    )
    # Metric-free receipt lineage only; scientific outcomes remain sealed.
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
