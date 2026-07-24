#!/usr/bin/env python3
"""Outcome-blind post-pack GPU assessment for distinct-law V2 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.neurips_2026.assess_global_k_distinct_laws_v2_smoke import (
    _read_lifecycle,
    _read_statuses,
    _read_telemetry,
    _read_timing,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_source_lock import (
    verify_source_lock,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    load_card,
    sha256_path,
)


def assess_scientific_gpu(
    card: dict[str, Any], statuses: list[dict[str, Any]], starts: dict[int, float],
    ends: dict[int, tuple[float, int]], timing: dict[str, float],
    telemetry: list[dict[str, Any]],
) -> dict[str, Any]:
    protocol = card["gpu_utilization_and_schedule"]["scientific_training_after_smoke_pass"]
    expected = int(protocol["pack_size"])
    roster = set(range(expected))
    if set(starts) != roster or set(ends) != roster or len(statuses) != expected:
        raise RuntimeError("Incomplete scientific lifecycle/status roster")
    if any(code != 0 for _stamp, code in ends.values()) or any(
        row["exit_code"] != 0 for row in statuses
    ):
        raise RuntimeError("Scientific training process failed")
    latest_start = max(starts.values())
    latest_end = max(stamp for stamp, _code in ends.values())
    active = [row for row in telemetry if latest_start <= row["epoch"] <= latest_end]
    epochs = np.asarray([row["epoch"] for row in active], dtype=np.float64)
    gap_limit = float(protocol["maximum_boundary_or_internal_telemetry_gap_seconds"])
    interval = float(protocol["telemetry_interval_seconds"])
    maximum_gap = float(np.max(np.diff(epochs))) if epochs.size > 1 else float("inf")
    expected_samples = max(1.0, (latest_end - latest_start) / interval)
    coverage_fraction = len(active) / expected_samples
    if (
        not active or active[0]["epoch"] > latest_start + gap_limit
        or active[-1]["epoch"] < latest_end - gap_limit
        or maximum_gap > gap_limit
        or coverage_fraction < float(protocol["minimum_expected_interval_coverage_fraction"])
    ):
        raise RuntimeError("Scientific telemetry does not densely cover the full active window")
    rolling = []
    for record in active:
        if record["epoch"] < latest_start + 600.0:
            continue
        values = [
            item["utilization"] for item in active
            if record["epoch"] - 600.0 <= item["epoch"] <= record["epoch"]
        ]
        if values:
            rolling.append(float(np.mean(values)))
    if not rolling:
        raise RuntimeError("No complete ten-minute scientific utilization window")
    minimum = min(rolling)
    threshold = 85.0
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "assessment_complete": True,
        "outcomes_inspected": False,
        "allowed_inputs_only": ["exit_status", "wall_time", "GPU_utilization", "GPU_memory"],
        "tail_included_through_final_training_end": True,
        "latest_training_start_epoch": latest_start,
        "latest_training_end_epoch": latest_end,
        "active_sample_count": len(active),
        "maximum_active_telemetry_gap_seconds": maximum_gap,
        "active_telemetry_expected_interval_coverage_fraction": coverage_fraction,
        "complete_rolling_ten_minute_window_count": len(rolling),
        "minimum_rolling_ten_minute_gpu_utilization_percent": minimum,
        "utilization_alert_threshold_percent": threshold,
        "flagged_low_utilization": minimum < threshold,
        "alert_is_diagnostic_not_a_relaunch_or_tuning_trigger": True,
        "pack_wall_seconds": timing["pack_end"] - timing["pack_start"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--source_lock", type=Path, required=True)
    parser.add_argument("--expected_source_lock_sha", required=True)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--task_manifest", type=Path, required=True)
    parser.add_argument("--pack_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_path(args.source_lock) != args.expected_source_lock_sha:
        raise RuntimeError("Expected source-lock hash mismatch")
    lock = verify_source_lock(args.source_lock)
    card, card_hash = load_card(args.card)
    manifest = json.loads(args.task_manifest.read_text())
    task_hash = sha256_path(args.task_tsv)
    if (
        manifest.get("mode") != "full"
        or manifest.get("card_sha256") != card_hash
        or manifest.get("task_tsv_sha256") != task_hash
        or lock["external_inputs"]["full_task_tsv"]["sha256"] != task_hash
    ):
        raise RuntimeError("Scientific task/card/source-lock mismatch")
    expected = int(card["task_table_contract"]["full_task_count"])
    payload = assess_scientific_gpu(
        card,
        _read_statuses(args.pack_root / "status", expected),
        *_read_lifecycle(args.pack_root / "quarantined_task_logs", expected),
        _read_timing(args.pack_root / "pack_timing.tsv"),
        _read_telemetry(args.pack_root / "gpu_telemetry.csv"),
    )
    payload["provenance"] = {
        "card_sha256": card_hash,
        "task_tsv_sha256": task_hash,
        "source_lock_sha256": args.expected_source_lock_sha,
        "telemetry_csv_sha256": sha256_path(args.pack_root / "gpu_telemetry.csv"),
        "assessor_sha256": sha256_path(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "flagged_low_utilization": payload["flagged_low_utilization"],
        "outcomes_inspected": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
