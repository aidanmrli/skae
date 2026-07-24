"""Authenticate profile and scientific GPU telemetry scopes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import shutil
import statistics as py_statistics

import numpy as np

from .io import (
    duplicate_safe_json,
    load_card,
    load_task_manifest,
    sha256_path,
    verify_source_manifest,
    write_json_once,
)


TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"


def read_telemetry(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().lower() == "timestamp":
                continue
            if len(row) != 6:
                raise ValueError(f"Unexpected telemetry row: {row}")
            timestamp = row[0].strip()
            try:
                moment = datetime.strptime(timestamp, TIME_FORMAT)
            except ValueError:
                moment = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S")
            records.append(
                {
                    "epoch": moment.timestamp(),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "utilization": float(row[3].strip()),
                    "memory_used": float(row[4].strip()),
                    "memory_total": float(row[5].strip()),
                }
            )
    if not records:
        raise ValueError("No GPU telemetry samples")
    if len({str(item["uuid"]) for item in records}) != 1:
        raise ValueError("Telemetry contains more than one GPU UUID")
    return records


def summarize_scope(
    records: list[dict[str, object]], started_at: float, completed_at: float
) -> dict[str, object]:
    selected = [
        row
        for row in records
        if float(row["epoch"]) >= float(started_at)
        and float(row["epoch"]) <= float(completed_at)
    ]
    if not selected:
        raise ValueError("Telemetry has no samples inside the declared scope")
    utilizations = np.asarray([row["utilization"] for row in selected], dtype=float)
    active = utilizations[utilizations > 0]
    epochs = [float(row["epoch"]) for row in selected]
    intervals = np.diff(epochs)
    return {
        "sample_count": len(selected),
        "active_sample_count": int(active.size),
        "mean_all_gpu_utilization_percent": float(utilizations.mean()),
        "mean_active_gpu_utilization_percent": float(active.mean()) if active.size else 0.0,
        "zero_utilization_fraction": float(np.mean(utilizations == 0)),
        "peak_memory_fraction": float(
            max(float(row["memory_used"]) / float(row["memory_total"]) for row in selected)
        ),
        "median_sample_interval_seconds": (
            float(py_statistics.median(intervals)) if intervals.size else 0.0
        ),
        "started_at_epoch": float(started_at),
        "completed_at_epoch": float(completed_at),
        "duration_seconds": float(completed_at - started_at),
    }


def _profile(
    card: dict[str, object],
    roots: dict[str, str],
    output_root: Path,
    records: list[dict[str, object]],
    raw_path: Path,
) -> None:
    workload_path = output_root / "profile" / "workload.json"
    workload = duplicate_safe_json(workload_path)
    for key, value in roots.items():
        if workload.get(key) != value:
            raise RuntimeError(f"Profile workload root mismatch for {key}")
    all_scope = summarize_scope(
        records, records[0]["epoch"], records[-1]["epoch"]
    )
    candidates = []
    for item in workload["candidates"]:
        telemetry = summarize_scope(
            records, item["started_at_epoch"], item["completed_at_epoch"]
        )
        memory_fraction = max(
            telemetry["peak_memory_fraction"],
            item["torch_peak_reserved_bytes"]
            / (records[0]["memory_total"] * 1024 * 1024),
        )
        telemetry["peak_memory_fraction"] = float(memory_fraction)
        passed = (
            telemetry["active_sample_count"]
            >= card["hardware_profile"]["minimum_active_samples"]
            and telemetry["mean_active_gpu_utilization_percent"]
            >= card["hardware_profile"]["minimum_mean_active_gpu_utilization_percent"]
            and telemetry["mean_all_gpu_utilization_percent"]
            >= card["hardware_profile"]["minimum_mean_all_gpu_utilization_percent"]
            and memory_fraction
            <= card["hardware_profile"]["maximum_peak_memory_fraction"]
            and telemetry["duration_seconds"]
            >= card["hardware_profile"]["minimum_seconds_each"]
        )
        candidates.append({**item, "telemetry": telemetry, "passed": bool(passed)})
    passing = [item for item in candidates if item["passed"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda item: (item["encoded_states_per_second"], item["batch_size"]),
        )
    copied_raw = output_root / "profile" / "raw_telemetry.csv"
    if copied_raw.exists():
        raise FileExistsError(copied_raw)
    shutil.copyfile(raw_path, copied_raw)
    decision = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "passed" if selected is not None else "invalid_no_profile_candidate_passed",
        **roots,
        "workload_sha256": sha256_path(workload_path),
        "raw_telemetry_sha256": sha256_path(copied_raw),
        "gpu_uuid": records[0]["uuid"],
        "gpu_name": records[0]["name"],
        "allocation_window": all_scope,
        "candidates": candidates,
        "selected_batch_size": None if selected is None else selected["batch_size"],
    }
    write_json_once(output_root / "profile" / "decision.json", decision)
    if selected is None:
        raise RuntimeError("No batch-size candidate met the frozen GPU profile gates")


def _scientific(
    card: dict[str, object],
    roots: dict[str, str],
    output_root: Path,
    records: list[dict[str, object]],
    raw_path: Path,
) -> None:
    markers_path = output_root / "field_only" / "markers.json"
    features_path = output_root / "field_only" / "features.pt"
    markers = duplicate_safe_json(markers_path)
    for key, value in roots.items():
        if markers.get(key) != value:
            raise RuntimeError(f"Scientific marker root mismatch for {key}")
    profile_path = output_root / "profile" / "decision.json"
    profile = duplicate_safe_json(profile_path)
    if profile.get("status") != "passed":
        raise RuntimeError("Synthetic profile did not pass")
    scope = summarize_scope(
        records, markers["started_at_epoch"], markers["completed_at_epoch"]
    )
    all_scope = summarize_scope(records, records[0]["epoch"], records[-1]["epoch"])
    gates = card["scientific_hardware"]
    passed = (
        scope["active_sample_count"] >= gates["minimum_active_samples"]
        and scope["mean_active_gpu_utilization_percent"]
        >= gates["minimum_mean_active_gpu_utilization_percent"]
        and scope["mean_all_gpu_utilization_percent"]
        >= gates["minimum_mean_all_gpu_utilization_percent"]
        and scope["peak_memory_fraction"] <= gates["maximum_peak_memory_fraction"]
        and scope["median_sample_interval_seconds"]
        <= gates["maximum_median_sample_interval_seconds"]
        and scope["duration_seconds"] >= gates["minimum_gpu_scope_seconds"]
    )
    copied_raw = output_root / "field_only" / "raw_telemetry.csv"
    if copied_raw.exists():
        raise FileExistsError(copied_raw)
    shutil.copyfile(raw_path, copied_raw)
    receipt = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "authenticated" if passed else "invalid_gpu_telemetry",
        **roots,
        "features_sha256": sha256_path(features_path),
        "markers_sha256": sha256_path(markers_path),
        "profile_decision_sha256": sha256_path(profile_path),
        "raw_telemetry_sha256": sha256_path(copied_raw),
        "gpu_uuid": records[0]["uuid"],
        "gpu_name": records[0]["name"],
        "kernel_scope": scope,
        "allocation_window": all_scope,
    }
    write_json_once(output_root / "field_only" / "telemetry_receipt.json", receipt)
    if not passed:
        raise RuntimeError("Scientific encoder scope failed the frozen GPU gates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("profile", "scientific"))
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-task-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-telemetry", type=Path, required=True)
    args = parser.parse_args()

    card, card_sha = load_card(expected_sha256=args.expected_card_sha256)
    source_sha = verify_source_manifest(
        card, expected_sha256=args.expected_source_manifest_sha256
    )
    task, task_sha = load_task_manifest(
        card, expected_sha256=args.expected_task_manifest_sha256
    )
    if Path(task["output_root"]) != args.output_root:
        raise RuntimeError("Telemetry output root differs from task")
    roots = {
        "card_sha256": card_sha,
        "source_manifest_sha256": source_sha,
        "task_manifest_sha256": task_sha,
    }
    records = read_telemetry(args.raw_telemetry)
    required_name = (
        card["hardware_profile"]["required_device_name_fragment"]
        if args.mode == "profile"
        else card["scientific_hardware"]["required_device_name_fragment"]
    )
    if required_name not in str(records[0]["name"]):
        raise RuntimeError("Telemetry GPU name failed the frozen device gate")
    if args.mode == "profile":
        _profile(card, roots, args.output_root, records, args.raw_telemetry)
    else:
        _scientific(card, roots, args.output_root, records, args.raw_telemetry)


if __name__ == "__main__":
    main()
