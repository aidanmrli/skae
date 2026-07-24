#!/usr/bin/env python3
"""Assess only lifecycle and GPU telemetry from the quarantined mixed V2 smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments.neurips_2026.global_k_distinct_laws_v2_source_lock import (
    verify_source_lock,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    build_rows,
    load_card,
    sha256_path,
)


START_RE = re.compile(r"^\[gpu-guard\] (\S+) :: paper benchmark training start task_id=(\d+)\s*$")
END_RE = re.compile(r"^\[gpu-guard\] (\S+) :: paper benchmark training end task_id=(\d+) exit_code=(\d+)\s*$")


def _phase_epoch(raw: str) -> float:
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def _telemetry_epoch(raw: str) -> float:
    return datetime.strptime(raw.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def _read_lifecycle(log_dir: Path, expected: int) -> tuple[dict[int, float], dict[int, tuple[float, int]]]:
    starts: dict[int, float] = {}
    ends: dict[int, tuple[float, int]] = {}
    for task_id in range(expected):
        path = log_dir / f"task_{task_id}.log"
        with path.open(errors="replace") as handle:
            for line in handle:
                # Deliberately ignore every non-lifecycle line; smoke outcomes stay closed.
                if not line.startswith("[gpu-guard]"):
                    continue
                start = START_RE.match(line)
                if start:
                    observed = int(start.group(2))
                    if observed in starts:
                        raise RuntimeError(f"Duplicate training-start marker for task {observed}")
                    starts[observed] = _phase_epoch(start.group(1))
                    continue
                end = END_RE.match(line)
                if end:
                    observed = int(end.group(2))
                    if observed in ends:
                        raise RuntimeError(f"Duplicate training-end marker for task {observed}")
                    ends[observed] = (_phase_epoch(end.group(1)), int(end.group(3)))
    return starts, ends


def _read_statuses(status_dir: Path, expected: int) -> list[dict[str, Any]]:
    records = []
    for task_id in range(expected):
        path = status_dir / f"task_{task_id}.tsv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != 1 or int(rows[0]["task_id"]) != task_id:
            raise RuntimeError(f"Malformed status record: {path}")
        records.append(
            {
                "task_id": task_id,
                "start": float(rows[0]["start_epoch_seconds"]),
                "end": float(rows[0]["end_epoch_seconds"]),
                "exit_code": int(rows[0]["exit_code"]),
            }
        )
    return records


def _read_timing(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    timing = {row["event"]: float(row["epoch_seconds"]) for row in rows}
    if set(timing) != {"pack_start", "pack_end"}:
        raise RuntimeError(f"Malformed pack timing: {timing}")
    return timing


def _read_telemetry(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                records.append(
                    {
                        "epoch": _telemetry_epoch(row["timestamp"]),
                        "uuid": row["uuid"].strip(),
                        "name": row["name"].strip(),
                        "utilization": float(row["gpu_utilization_percent"]),
                        "memory_used": float(row["memory_used_mib"]),
                        "memory_total": float(row["memory_total_mib"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not records:
        raise RuntimeError("No valid GPU telemetry samples")
    if len({record["uuid"] for record in records}) != 1:
        raise RuntimeError("Smoke telemetry contains more than one GPU UUID")
    return sorted(records, key=lambda record: record["epoch"])


def assess(
    card: dict[str, Any], statuses: list[dict[str, Any]], starts: dict[int, float],
    ends: dict[int, tuple[float, int]], timing: dict[str, float],
    telemetry: list[dict[str, Any]],
) -> dict[str, Any]:
    smoke = card["gpu_utilization_and_schedule"]["smoke"]
    expected = int(smoke["pack_size"])
    roster = set(range(expected))
    lifecycle_complete = set(starts) == roster and set(ends) == roster
    all_zero = bool(
        lifecycle_complete
        and all(code == 0 for _stamp, code in ends.values())
        and all(record["exit_code"] == 0 for record in statuses)
    )
    latest_start = max(starts.values()) if starts else float("inf")
    latest_end = max(value[0] for value in ends.values()) if ends else float("-inf")
    active = [record for record in telemetry if latest_start <= record["epoch"] <= latest_end]
    epochs = np.asarray([record["epoch"] for record in active], dtype=np.float64)
    maximum_gap = float(np.max(np.diff(epochs))) if epochs.size > 1 else float("inf")
    interval = float(smoke["telemetry_interval_seconds"])
    expected_samples = max(1.0, (latest_end - latest_start) / interval)
    coverage_fraction = len(active) / expected_samples
    gap_limit = float(smoke["maximum_boundary_or_internal_telemetry_gap_seconds"])
    telemetry_covers_window = bool(
        epochs.size
        and epochs[0] <= latest_start + gap_limit
        and epochs[-1] >= latest_end - gap_limit
        and maximum_gap <= gap_limit
        and coverage_fraction >= float(smoke["minimum_expected_interval_coverage_fraction"])
    )
    utilization = np.asarray([record["utilization"] for record in active], dtype=np.float64)
    mean_active = float(np.mean(utilization)) if utilization.size else float("nan")
    p10_active = float(np.quantile(utilization, 0.1)) if utilization.size else float("nan")
    peak_memory = max(record["memory_used"] for record in telemetry)
    total_memory = max(record["memory_total"] for record in telemetry)
    peak_fraction = peak_memory / total_memory
    training_durations = [ends[index][0] - starts[index] for index in roster] if lifecycle_complete else []
    scale = float(card["training_arms"]["sparse"]["num_steps"]) / float(smoke["num_steps"])
    startup = max(0.0, latest_start - timing["pack_start"]) if math.isfinite(latest_start) else float("inf")
    projected = 1.25 * (startup + max(training_durations, default=float("inf")) * scale)
    checks = {
        "all_processes_exit_zero": all_zero,
        "minimum_active_samples": len(active) >= int(smoke["minimum_active_samples"]),
        "telemetry_covers_complete_active_window": telemetry_covers_window,
        "mean_active_gpu_utilization": math.isfinite(mean_active) and mean_active >= float(smoke["minimum_mean_active_gpu_utilization_percent"]),
        "p10_active_gpu_utilization": math.isfinite(p10_active) and p10_active >= float(smoke["minimum_p10_active_gpu_utilization_percent"]),
        "peak_memory": peak_fraction <= float(smoke["maximum_peak_memory_fraction"]),
        "projected_full_pack_wall_time": projected / 3600.0 <= float(smoke["maximum_projected_full_pack_wall_hours_with_25pct_safety"]),
        "one_a100_80gb": len({record["uuid"] for record in telemetry}) == 1
        and all("A100" in record["name"] for record in telemetry)
        and total_memory >= 80000.0,
    }
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "outcomes_inspected": False,
        "allowed_inputs_only": ["exit_status", "wall_time", "step_throughput", "GPU_utilization", "GPU_memory"],
        "passed": all(checks.values()),
        "checks": checks,
        "task_process_count": len(statuses),
        "lifecycle_start_count": len(starts),
        "lifecycle_end_count": len(ends),
        "active_sample_count": len(active),
        "maximum_active_telemetry_gap_seconds": maximum_gap if math.isfinite(maximum_gap) else None,
        "active_telemetry_expected_interval_coverage_fraction": coverage_fraction,
        "mean_active_gpu_utilization_percent": mean_active if math.isfinite(mean_active) else None,
        "p10_active_gpu_utilization_percent": p10_active if math.isfinite(p10_active) else None,
        "peak_memory_used_mib": peak_memory,
        "memory_total_mib": total_memory,
        "peak_memory_fraction": peak_fraction,
        "pack_wall_seconds": timing["pack_end"] - timing["pack_start"],
        "maximum_training_process_seconds": max(training_durations) if training_durations else None,
        "aggregate_training_steps_per_second": (
            expected * float(smoke["num_steps"]) / max(training_durations)
            if training_durations and max(training_durations) > 0 else None
        ),
        "projected_full_pack_wall_seconds_with_25pct_safety": projected if math.isfinite(projected) else None,
        "projected_full_pack_wall_hours_with_25pct_safety": projected / 3600.0 if math.isfinite(projected) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--source_lock", type=Path, required=True)
    parser.add_argument("--expected_source_lock_sha", required=True)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--task_manifest", type=Path, required=True)
    parser.add_argument("--status_dir", type=Path, required=True)
    parser.add_argument("--task_log_dir", type=Path, required=True)
    parser.add_argument("--telemetry_csv", type=Path, required=True)
    parser.add_argument("--pack_timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_path(args.source_lock) != args.expected_source_lock_sha:
        raise RuntimeError("Expected source-lock hash mismatch")
    lock = verify_source_lock(args.source_lock)
    card, card_hash = load_card(args.card)
    manifest = json.loads(args.task_manifest.read_text())
    rows = build_rows(card, "smoke")
    expected = len(rows)
    if manifest.get("mode") != "smoke" or manifest.get("card_sha256") != card_hash:
        raise RuntimeError("Smoke manifest/card mismatch")
    task_hash = sha256_path(args.task_tsv)
    if task_hash != manifest.get("task_tsv_sha256") or task_hash != lock["external_inputs"]["smoke_task_tsv"]["sha256"]:
        raise RuntimeError("Smoke task-table hash mismatch")
    payload = assess(
        card,
        _read_statuses(args.status_dir, expected),
        *_read_lifecycle(args.task_log_dir, expected),
        _read_timing(args.pack_timing),
        _read_telemetry(args.telemetry_csv),
    )
    payload["provenance"] = {
        "card_sha256": card_hash,
        "task_tsv_sha256": task_hash,
        "source_lock_sha256": args.expected_source_lock_sha,
        "telemetry_csv_sha256": sha256_path(args.telemetry_csv),
        "assessor_sha256": sha256_path(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "passed": payload["passed"], "outcomes_inspected": False}, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 3)


if __name__ == "__main__":
    main()
