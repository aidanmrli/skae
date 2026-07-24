#!/usr/bin/env python3
"""Assess only throughput and GPU telemetry from the quarantined dense smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.neurips_2026.global_k_dense_zero_wd_tasks import load_card


START_RE = re.compile(
    r"\[gpu-guard\] (\S+) :: paper benchmark training start task_id=(\d+)"
)
END_RE = re.compile(
    r"\[gpu-guard\] (\S+) :: paper benchmark training end "
    r"task_id=(\d+) exit_code=(\d+)"
)


def _parse_phase_time(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)


def _parse_telemetry_time(raw: str) -> datetime:
    return datetime.strptime(raw.strip(), "%Y/%m/%d %H:%M:%S.%f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--telemetry_csv", type=Path, required=True)
    parser.add_argument("--training_log", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_json}")
    card, card_hash = load_card(args.card)
    smoke = card["gpu_smoke"]

    log_text = args.training_log.read_text(errors="replace")
    starts = {int(task): _parse_phase_time(stamp) for stamp, task in START_RE.findall(log_text)}
    ends = {
        int(task): (_parse_phase_time(stamp), int(code))
        for stamp, task, code in END_RE.findall(log_text)
    }
    expected = len(smoke["seeds"])
    process_complete = (
        len(starts) == expected
        and len(ends) == expected
        and all(code == int(smoke["required_exit_status"]) for _, code in ends.values())
    )

    telemetry = []
    with args.telemetry_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                telemetry.append(
                    {
                        "time": _parse_telemetry_time(row["timestamp"]),
                        "gpu": float(row["gpu_utilization_percent"]),
                        "memory": float(row["memory_used_mib"]),
                        "total": float(row["memory_total_mib"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not telemetry:
        raise RuntimeError("No valid nvidia-smi telemetry rows")
    peak_memory = max(item["memory"] for item in telemetry)
    total_memory = max(item["total"] for item in telemetry)
    latest_start = max(starts.values()) if starts else telemetry[0]["time"]
    earliest_end = min((value[0] for value in ends.values()), default=telemetry[-1]["time"])
    memory_floor = 0.9 * peak_memory
    active = [
        item for item in telemetry
        if latest_start <= item["time"] <= earliest_end and item["memory"] >= memory_floor
    ]
    active_util = [item["gpu"] for item in active]
    active_count = len(active_util)
    mean_active = float(np.mean(active_util)) if active_util else float("nan")
    peak_fraction = peak_memory / total_memory
    active_start = active[0]["time"] if active else latest_start
    initialization_seconds = max(0.0, (active_start - latest_start).total_seconds())
    steady_seconds = max(0.0, (earliest_end - active_start).total_seconds())
    scale = float(card["training"]["num_steps"]) / float(smoke["num_steps"])
    projected_pack_seconds = 1.25 * (initialization_seconds + steady_seconds * scale)

    checks = {
        "all_processes_completed_zero": process_complete,
        "minimum_active_samples": active_count >= int(smoke["minimum_active_samples"]),
        "mean_active_gpu_utilization": math.isfinite(mean_active)
        and mean_active >= float(smoke["minimum_mean_active_gpu_utilization_percent"]),
        "peak_memory": peak_fraction <= float(smoke["maximum_peak_memory_fraction"]),
    }
    payload = {
        "schema_version": 1,
        "card_sha256": card_hash,
        "smoke_outcomes_inspected": False,
        "passed": all(checks.values()),
        "checks": checks,
        "task_process_count": len(starts),
        "completed_process_count": len(ends),
        "telemetry_sample_count": len(telemetry),
        "active_sample_count": active_count,
        "mean_active_gpu_utilization_percent": mean_active if math.isfinite(mean_active) else None,
        "active_gpu_utilization_p10_percent": (
            float(np.quantile(active_util, 0.1)) if active_util else None
        ),
        "peak_memory_used_mib": peak_memory,
        "memory_total_mib": total_memory,
        "peak_memory_fraction": peak_fraction,
        "latest_training_start": latest_start.isoformat(),
        "earliest_training_end": earliest_end.isoformat(),
        "initialization_seconds": initialization_seconds,
        "steady_window_seconds": steady_seconds,
        "projected_full_pack_wall_seconds_with_25pct_safety": projected_pack_seconds,
        "projected_full_pack_wall_hours_with_25pct_safety": projected_pack_seconds / 3600.0,
        "telemetry_csv": str(args.telemetry_csv),
        "training_log": str(args.training_log),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if payload["passed"] else 3)


if __name__ == "__main__":
    main()
