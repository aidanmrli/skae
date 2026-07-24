"""Audit raw and marker-bounded GPU telemetry without dropping idle samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from experiments.neurips_2026.allen_cahn_direct_baseline.core import (
    duplicate_safe_json,
    sha256_path,
    write_json_once,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--phase-start", type=Path, required=True)
    parser.add_argument("--phase-end", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-core-samples", type=int, default=5)
    parser.add_argument("--task-lock-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--artifact-role",
        choices=[
            "non_scientific_gpu_smoke",
            "scientific_training",
            "scientific_evaluation",
        ],
        required=True,
    )
    parser.add_argument("--slurm-job-id", required=True)
    return parser.parse_args()


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    utilization = [float(row["utilization_gpu_percent"]) for row in rows]
    memory = [float(row["memory_used_mib"]) for row in rows]
    return {
        "samples": len(rows),
        "mean_gpu_utilization_percent": mean(utilization) if utilization else None,
        "p10_gpu_utilization_percent": percentile(utilization, 0.10)
        if utilization
        else None,
        "zero_utilization_samples": sum(value == 0.0 for value in utilization),
        "peak_memory_used_mib": max(memory) if memory else None,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows: list[dict[str, object]] = []
    with args.telemetry.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "unix_time_seconds",
            "gpu_index",
            "gpu_uuid",
            "gpu_name",
            "utilization_gpu_percent",
            "utilization_memory_percent",
            "memory_used_mib",
            "memory_total_mib",
            "power_draw_w",
            "power_limit_w",
        }
        if set(reader.fieldnames or []) != expected:
            raise RuntimeError(f"Telemetry columns drifted: {reader.fieldnames}")
        for raw in reader:
            row: dict[str, object] = {
                "unix_time_seconds": float(raw["unix_time_seconds"]),
                "gpu_index": int(raw["gpu_index"]),
                "gpu_uuid": raw["gpu_uuid"],
                "gpu_name": raw["gpu_name"],
                "utilization_gpu_percent": float(raw["utilization_gpu_percent"]),
                "utilization_memory_percent": float(
                    raw["utilization_memory_percent"]
                ),
                "memory_used_mib": float(raw["memory_used_mib"]),
                "memory_total_mib": float(raw["memory_total_mib"]),
                "power_draw_w": float(raw["power_draw_w"]),
                "power_limit_w": float(raw["power_limit_w"]),
            }
            rows.append(row)
    if not rows:
        raise RuntimeError("Telemetry is empty")
    if len({str(row["gpu_uuid"]) for row in rows}) != 1:
        raise RuntimeError("Telemetry spans multiple GPU UUIDs")
    timestamps = [float(row["unix_time_seconds"]) for row in rows]
    gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
    start_marker = duplicate_safe_json(args.phase_start)
    end_marker = duplicate_safe_json(args.phase_end)
    if start_marker.get("phase") not in {
        "optimizer_loop_start",
        "evaluation_loop_start",
    }:
        raise RuntimeError("Unexpected phase-start marker")
    if end_marker.get("phase") not in {
        "optimizer_loop_end",
        "evaluation_loop_end",
    }:
        raise RuntimeError("Unexpected phase-end marker")
    if str(start_marker["phase"]).replace("start", "end") != str(
        end_marker["phase"]
    ):
        raise RuntimeError("Phase marker roles do not match")
    start = float(start_marker["unix_time_seconds"])
    end = float(end_marker["unix_time_seconds"])
    if not start < end:
        raise RuntimeError("Invalid phase-marker ordering")
    startup = [row for row in rows if float(row["unix_time_seconds"]) < start]
    core = [
        row
        for row in rows
        if start <= float(row["unix_time_seconds"]) <= end
    ]
    tail = [row for row in rows if float(row["unix_time_seconds"]) > end]
    raw_summary = summarize(rows)
    core_summary = summarize(core)
    startup_summary = summarize(startup)
    tail_summary = summarize(tail)
    core_mean = core_summary["mean_gpu_utilization_percent"]
    core_p10 = core_summary["p10_gpu_utilization_percent"]
    checks = {
        "minimum_core_samples": len(core) >= int(args.minimum_core_samples),
        "core_mean_at_least_85_percent": core_mean is not None
        and float(core_mean) >= 85.0,
        "core_p10_at_least_80_percent": core_p10 is not None
        and float(core_p10) >= 80.0,
        "one_gpu_uuid": True,
        "one_gpu_index": len({int(row["gpu_index"]) for row in rows}) == 1,
        "one_gpu_name": len({str(row["gpu_name"]) for row in rows}) == 1,
        "all_values_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "unix_time_seconds",
                "utilization_gpu_percent",
                "utilization_memory_percent",
                "memory_used_mib",
                "memory_total_mib",
                "power_draw_w",
                "power_limit_w",
            )
        ),
        "utilization_ranges_valid": all(
            0.0 <= float(row["utilization_gpu_percent"]) <= 100.0
            and 0.0 <= float(row["utilization_memory_percent"]) <= 100.0
            for row in rows
        ),
        "memory_ranges_valid": all(
            0.0 <= float(row["memory_used_mib"]) <= float(row["memory_total_mib"])
            and float(row["memory_total_mib"]) > 0.0
            for row in rows
        ),
        "power_ranges_valid": all(
            0.0 <= float(row["power_draw_w"])
            and 0.0 < float(row["power_limit_w"])
            and float(row["power_draw_w"]) <= 1.25 * float(row["power_limit_w"])
            for row in rows
        ),
        "timestamps_strictly_increasing": all(gap > 0.0 for gap in gaps),
        "maximum_sampling_gap_at_most_3_seconds": bool(gaps)
        and max(gaps) <= 3.0,
        "marker_interval_covered_on_both_sides": timestamps[0] <= start
        and timestamps[-1] >= end,
        "startup_sample_present": len(startup) >= 1,
        "tail_sample_present": len(tail) >= 1,
        "marker_boundary_distance_at_most_3_seconds": min(
            abs(value - start) for value in timestamps
        )
        <= 3.0
        and min(abs(value - end) for value in timestamps) <= 3.0,
        "raw_partition_is_exact": len(rows) == len(startup) + len(core) + len(tail),
        "zero_utilization_samples_retained": (
            raw_summary["zero_utilization_samples"]
            == startup_summary["zero_utilization_samples"]
            + core_summary["zero_utilization_samples"]
            + tail_summary["zero_utilization_samples"]
        ),
        "startup_and_tail_reported": True,
    }
    payload = {
        "schema_version": 1,
        "protocol_id": "allen_cahn_matched_direct_baseline_v1",
        "artifact_role": args.artifact_role,
        "task_lock_sha256": args.task_lock_sha256,
        "model_seed": int(args.seed),
        "slurm_job_id": str(args.slurm_job_id),
        "telemetry_sha256": sha256_path(args.telemetry),
        "phase_start_sha256": sha256_path(args.phase_start),
        "phase_end_sha256": sha256_path(args.phase_end),
        "gpu_uuid": str(rows[0]["gpu_uuid"]),
        "gpu_name": str(rows[0]["gpu_name"]),
        "raw_all_samples": raw_summary,
        "startup_before_optimizer_loop": startup_summary,
        "marker_bounded_optimizer_loop": core_summary,
        "tail_after_optimizer_loop": tail_summary,
        "maximum_sampling_gap_seconds": max(gaps) if gaps else None,
        "phase_start_unix_time_seconds": start,
        "phase_end_unix_time_seconds": end,
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json_once(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
