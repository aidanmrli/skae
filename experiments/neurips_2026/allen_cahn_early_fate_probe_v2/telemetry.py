"""Authenticate prospective generation and field-only encoding telemetry."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
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


def read_telemetry(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().lower() == "timestamp":
                continue
            if len(row) != 6:
                raise ValueError(f"Unexpected telemetry row: {row}")
            stamp = row[0].strip()
            try:
                moment = datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f")
            except ValueError:
                moment = datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S")
            records.append(
                {
                    "epoch": moment.timestamp(),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "utilization": float(row[3]),
                    "memory_used": float(row[4]),
                    "memory_total": float(row[5]),
                }
            )
    if not records or len({str(row["uuid"]) for row in records}) != 1:
        raise RuntimeError("Telemetry must contain one nonempty GPU-UUID stream")
    return records


def summarize_scope(
    records: list[dict[str, object]], started_at: float, completed_at: float
) -> dict[str, object]:
    chosen = [
        row
        for row in records
        if float(started_at) <= float(row["epoch"]) <= float(completed_at)
    ]
    if not chosen:
        raise RuntimeError("No telemetry sample falls inside a declared scope")
    values = np.asarray([row["utilization"] for row in chosen], dtype=float)
    active = values[values > 0]
    gaps = np.diff([float(row["epoch"]) for row in chosen])
    return {
        "sample_count": len(chosen),
        "active_sample_count": int(active.size),
        "mean_all_gpu_utilization_percent": float(values.mean()),
        "mean_active_gpu_utilization_percent": float(active.mean()) if active.size else 0.0,
        "zero_utilization_fraction": float(np.mean(values == 0)),
        "peak_memory_fraction": float(
            max(float(row["memory_used"]) / float(row["memory_total"]) for row in chosen)
        ),
        "median_sample_interval_seconds": (
            float(py_statistics.median(gaps)) if gaps.size else 0.0
        ),
        "maximum_sample_gap_seconds": float(gaps.max()) if gaps.size else 0.0,
        "started_at_epoch": float(started_at),
        "completed_at_epoch": float(completed_at),
        "duration_seconds": float(completed_at - started_at),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-task-manifest-sha256", required=True)
    parser.add_argument("--expected-gpu-slurm-job-id", required=True)
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
        raise RuntimeError("Telemetry output root differs from frozen task")
    roots = {
        "card_sha256": card_sha,
        "source_manifest_sha256": source_sha,
        "task_manifest_sha256": task_sha,
    }
    markers_path = args.output_root / "field_only" / "markers.json"
    markers = duplicate_safe_json(markers_path)
    if any(markers.get(key) != value for key, value in roots.items()):
        raise RuntimeError("Marker roots differ from launch roots")
    gpu_slurm_job_id = str(args.expected_gpu_slurm_job_id)
    telemetry_audit_slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if markers.get("slurm_job_id") != gpu_slurm_job_id:
        raise RuntimeError("GPU marker lineage differs from the queued GPU job")
    if not telemetry_audit_slurm_job_id:
        raise RuntimeError("Telemetry authentication must run as a SLURM job")
    if telemetry_audit_slurm_job_id == gpu_slurm_job_id:
        raise RuntimeError("Telemetry authentication must use a dependent CPU job")

    features_path = args.output_root / "field_only" / "features.pt"
    manifest_path = args.output_root / "dataset_manifest.json"
    if sha256_path(features_path) != markers["features_sha256"]:
        raise RuntimeError("Feature artifact changed before telemetry audit")
    if sha256_path(manifest_path) != markers["dataset_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed before telemetry audit")
    manifest = duplicate_safe_json(manifest_path)
    if any(manifest.get(key) != value for key, value in roots.items()):
        raise RuntimeError("Dataset manifest roots differ from launch roots")

    records = read_telemetry(args.raw_telemetry)
    if card["hardware"]["required_device_name_fragment"] not in str(records[0]["name"]):
        raise RuntimeError("Telemetry did not come from the frozen GPU type")
    generation = summarize_scope(
        records,
        markers["generation_started_at_epoch"],
        markers["generation_completed_at_epoch"],
    )
    extraction = summarize_scope(
        records,
        markers["extraction_started_at_epoch"],
        markers["extraction_completed_at_epoch"],
    )
    gpu_runtime = summarize_scope(records, records[0]["epoch"], records[-1]["epoch"])
    torch_memory_fraction = markers["torch_peak_reserved_bytes"] / (
        float(records[0]["memory_total"]) * 1024 * 1024
    )
    extraction["peak_memory_fraction"] = max(
        extraction["peak_memory_fraction"], torch_memory_fraction
    )
    hardware = card["hardware"]
    passed = (
        extraction["active_sample_count"] >= hardware["encoding_minimum_active_samples"]
        and extraction["mean_active_gpu_utilization_percent"]
        >= hardware["encoding_minimum_mean_active_gpu_utilization_percent"]
        and extraction["mean_all_gpu_utilization_percent"]
        >= hardware["encoding_minimum_mean_all_gpu_utilization_percent"]
        and extraction["peak_memory_fraction"] <= hardware["maximum_peak_memory_fraction"]
        and extraction["median_sample_interval_seconds"]
        <= hardware["maximum_median_sample_interval_seconds"]
        and extraction["duration_seconds"] >= hardware["encoding_minimum_scope_seconds"]
    )
    raw_copy = args.output_root / "field_only" / "raw_telemetry.csv"
    if raw_copy.exists():
        raise FileExistsError(raw_copy)
    shutil.copyfile(args.raw_telemetry, raw_copy)
    receipt = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "authenticated" if passed else "invalid_gpu_telemetry",
        **roots,
        "features_sha256": sha256_path(features_path),
        "dataset_manifest_sha256": sha256_path(manifest_path),
        "markers_sha256": sha256_path(markers_path),
        "raw_telemetry_sha256": sha256_path(raw_copy),
        "gpu_uuid": records[0]["uuid"],
        "gpu_name": records[0]["name"],
        "gpu_slurm_job_id": gpu_slurm_job_id,
        "telemetry_audit_slurm_job_id": telemetry_audit_slurm_job_id,
        "generation_scope_descriptive": generation,
        "encoding_scope_validity": extraction,
        "gpu_runtime_window_descriptive": gpu_runtime,
        "semantic_outcomes_accessed": 0,
    }
    write_json_once(args.output_root / "field_only" / "telemetry_receipt.json", receipt)
    if not passed:
        raise RuntimeError("V2 field-only encoding failed the frozen GPU telemetry gates")


if __name__ == "__main__":
    main()
