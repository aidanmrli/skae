"""Audit GPU windows and issue a metric-free outcome-guard receipt."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from experiments.neurips_2026.allen_cahn_forecast_replication.io import (
    CARD_PATH,
    MANIFEST_PATH,
    assert_runtime_values_safe,
    duplicate_safe_json,
    load_card,
    sha256_path,
    verify_source_manifest,
    write_json_once,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--raw-telemetry", type=Path, required=True)
    return parser.parse_args()


def _number(value: str) -> float:
    cleaned = "".join(char for char in value if char.isdigit() or char in ".-")
    if not cleaned:
        raise ValueError(f"Cannot parse telemetry number {value!r}")
    return float(cleaned)


def _epoch(value: str) -> float:
    text = value.strip()
    formats = ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S")
    for format_string in formats:
        try:
            return datetime.strptime(text, format_string).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Cannot parse nvidia-smi timestamp {value!r}")


def parse_samples(path: Path) -> list[dict[str, Any]]:
    samples = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        required = {
            "timestamp",
            "uuid",
            "name",
            "utilization.gpu [%]",
            "memory.used [MiB]",
            "memory.total [MiB]",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Telemetry columns drifted: {reader.fieldnames}")
        for row in reader:
            samples.append(
                {
                    "epoch_seconds": _epoch(row["timestamp"]),
                    "gpu_uuid": row["uuid"].strip(),
                    "gpu_name": row["name"].strip(),
                    "utilization_percent": _number(row["utilization.gpu [%]"]),
                    "memory_used_mib": _number(row["memory.used [MiB]"]),
                    "memory_total_mib": _number(row["memory.total [MiB]"]),
                }
            )
    if not samples:
        raise ValueError("Raw GPU telemetry is empty")
    for sample in samples:
        if not 0.0 <= sample["utilization_percent"] <= 100.0:
            raise ValueError("GPU utilization lies outside [0,100]")
        if (
            sample["memory_total_mib"] <= 0.0
            or sample["memory_used_mib"] < 0.0
            or sample["memory_used_mib"] > sample["memory_total_mib"]
        ):
            raise ValueError("GPU memory telemetry is physically invalid")
    if len({sample["gpu_name"] for sample in samples}) != 1:
        raise ValueError("Telemetry contains more than one GPU name")
    uuids = {sample["gpu_uuid"] for sample in samples}
    if len(uuids) != 1 or not next(iter(uuids)).startswith("GPU-"):
        raise ValueError("Telemetry must bind exactly one valid GPU UUID")
    if "A100" not in samples[0]["gpu_name"]:
        raise ValueError("Telemetry is not from the frozen A100 device class")
    timestamps = np.asarray([sample["epoch_seconds"] for sample in samples])
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("Telemetry timestamps must be strictly increasing")
    return samples


def _checkpoint_roster_rows(card: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "checkpoint_step": int(row["checkpoint_step"]),
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in card["checkpoint_roster"]["runs"]
    ]


def _checkpoint_roster_digest(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bind_slurm_job(runtime: dict[str, Any]) -> str:
    runtime_value = str(runtime.get("environment", {}).get("slurm_job_id", "not_recorded"))
    current_value = str(os.environ.get("SLURM_JOB_ID", "not_recorded"))
    missing = {"", "None", "not_recorded"}
    runtime_available = runtime_value not in missing
    current_available = current_value not in missing
    if runtime_available != current_available:
        raise RuntimeError("SLURM job lineage is available on only one side of telemetry")
    if runtime_available and runtime_value != current_value:
        raise RuntimeError("Telemetry and runtime SLURM job lineage differ")
    return runtime_value if runtime_available else "not_recorded"


def _marker_epoch(root: Path, stage: str, card_hash: str, source_hash: str) -> float:
    path = root / "markers" / f"{stage}.json"
    payload = duplicate_safe_json(path)
    if (
        payload.get("stage") != stage
        or payload.get("card_sha256") != card_hash
        or payload.get("source_manifest_sha256") != source_hash
    ):
        raise RuntimeError(f"Marker lineage failed for {stage}")
    return float(payload["epoch_seconds"])


def window_statistics(
    samples: list[dict[str, Any]],
    *,
    start: float,
    end: float,
    boundary_exclusion_per_side: int = 0,
) -> dict[str, Any]:
    if not start < end:
        raise ValueError("Telemetry window is not ordered")
    if int(boundary_exclusion_per_side) not in (0, 1):
        raise ValueError("Boundary exclusion must be exactly zero or one sample per side")
    selected = [sample for sample in samples if start <= sample["epoch_seconds"] <= end]
    if not selected:
        raise ValueError("Telemetry window contains no samples")
    timestamps = np.asarray([sample["epoch_seconds"] for sample in selected])
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("Window telemetry timestamps must be strictly increasing")
    exclusion = int(boundary_exclusion_per_side)
    retained = selected[exclusion : len(selected) - exclusion] if exclusion else selected
    selected_utilization = np.asarray(
        [sample["utilization_percent"] for sample in selected], dtype=np.float64
    )
    retained_utilization = np.asarray(
        [sample["utilization_percent"] for sample in retained], dtype=np.float64
    )
    active = retained_utilization[retained_utilization > 0]
    memory_fraction = np.asarray(
        [sample["memory_used_mib"] / sample["memory_total_mib"] for sample in retained],
        dtype=np.float64,
    )
    gaps = np.diff(timestamps)
    return {
        "start_epoch_seconds": float(start),
        "end_epoch_seconds": float(end),
        "duration_seconds": float(end - start),
        "all_window_samples": int(selected_utilization.size),
        "boundary_samples_excluded_per_side": exclusion,
        "retained_all_window_samples": int(retained_utilization.size),
        "active_retained_samples_descriptive": int(active.size),
        "zero_utilization_retained_samples_descriptive": int(
            np.sum(retained_utilization == 0)
        ),
        "mean_all_window_gpu_utilization_percent_descriptive": float(
            selected_utilization.mean()
        ),
        "mean_retained_all_window_gpu_utilization_percent": float(
            retained_utilization.mean()
        )
        if retained_utilization.size
        else 0.0,
        "p10_retained_all_window_gpu_utilization_percent": float(
            np.quantile(retained_utilization, 0.10)
        )
        if retained_utilization.size
        else 0.0,
        "mean_active_gpu_utilization_percent_descriptive": float(active.mean())
        if active.size
        else 0.0,
        "median_sample_cadence_seconds": float(np.median(gaps))
        if gaps.size
        else 1.0e9,
        "maximum_sample_gap_seconds": float(gaps.max())
        if gaps.size
        else 1.0e9,
        "leading_marker_edge_gap_seconds": float(timestamps[0] - start),
        "trailing_marker_edge_gap_seconds": float(end - timestamps[-1]),
        "peak_memory_fraction": float(memory_fraction.max())
        if memory_fraction.size
        else 1.0e9,
        "utilization_filter_applied": False,
    }


def evaluation_gate_checks(
    evaluation: dict[str, Any], hardware: dict[str, Any]
) -> dict[str, bool]:
    expected_exclusion = int(hardware["boundary_samples_excluded_per_side"])
    minimum_cadence = float(hardware["minimum_median_sample_cadence_seconds"])
    maximum_cadence = float(hardware["maximum_median_sample_cadence_seconds"])
    maximum_gap = float(hardware["maximum_sample_gap_seconds"])
    maximum_edge_gap = float(hardware["maximum_marker_edge_gap_seconds"])
    return {
        "exact_unconditional_boundary_exclusion": evaluation[
            "boundary_samples_excluded_per_side"
        ]
        == expected_exclusion
        == 1,
        "no_utilization_filter": evaluation["utilization_filter_applied"] is False,
        "minimum_all_window_samples": evaluation["all_window_samples"]
        >= int(hardware["minimum_all_window_samples_before_boundary_exclusion"]),
        "minimum_retained_all_window_samples": evaluation[
            "retained_all_window_samples"
        ]
        >= int(hardware["minimum_retained_all_window_samples"]),
        "mean_retained_all_window_gpu_utilization": evaluation[
            "mean_retained_all_window_gpu_utilization_percent"
        ]
        >= float(hardware["minimum_mean_retained_all_window_gpu_utilization_percent"]),
        "strict_p10_retained_all_window_gpu_utilization": evaluation[
            "p10_retained_all_window_gpu_utilization_percent"
        ]
        > float(
            hardware[
                "strict_p10_retained_all_window_gpu_utilization_percent_above"
            ]
        ),
        "median_sample_cadence": minimum_cadence
        <= evaluation["median_sample_cadence_seconds"]
        <= maximum_cadence,
        "maximum_sample_gap": evaluation["maximum_sample_gap_seconds"]
        <= maximum_gap,
        "leading_marker_edge_coverage": evaluation[
            "leading_marker_edge_gap_seconds"
        ]
        <= maximum_edge_gap,
        "trailing_marker_edge_coverage": evaluation[
            "trailing_marker_edge_gap_seconds"
        ]
        <= maximum_edge_gap,
        "peak_memory_fraction": evaluation["peak_memory_fraction"]
        <= float(hardware["maximum_peak_memory_fraction"]),
    }


def audit_and_receipt(
    card: dict[str, Any],
    *,
    card_hash: str,
    source_hash: str,
    output_root: Path,
    raw_telemetry: Path,
) -> dict[str, Any]:
    runtime_path = output_root / "runtime_lineage.json"
    runtime = duplicate_safe_json(runtime_path)
    expected_roster = _checkpoint_roster_rows(card)
    if (
        runtime.get("status") != "scientific_payload_written_but_not_authorized_for_summary"
        or runtime.get("card_sha256") != card_hash
        or runtime.get("source_manifest_sha256") != source_hash
        or runtime.get("crossed_cells") != 60
        or runtime.get("scientific_metrics_printed") is not False
        or runtime.get("checkpoint_roster") != expected_roster
        or runtime.get("checkpoint_roster_sha256")
        != _checkpoint_roster_digest(expected_roster)
    ):
        raise RuntimeError("Runtime lineage failed before telemetry audit")
    slurm_job_id = _bind_slurm_job(runtime)
    scientific_path = Path(str(runtime["scientific_payload_path"]))
    dataset_manifest_path = Path(str(runtime["dataset_manifest_path"]))
    if sha256_path(scientific_path) != runtime["scientific_payload_sha256"]:
        raise RuntimeError("Scientific payload hash differs from metric-free lineage")
    if sha256_path(dataset_manifest_path) != runtime["dataset_manifest_sha256"]:
        raise RuntimeError("Dataset manifest hash differs from metric-free lineage")

    samples = parse_samples(raw_telemetry)
    gpu_uuid = str(samples[0]["gpu_uuid"])
    windows = {}
    for name, start_stage, end_stage, boundary_exclusion in (
        ("generation_descriptive", "generation_start", "generation_end", 0),
        (
            "evaluation_validity",
            "evaluation_start",
            "evaluation_end",
            int(card["hardware_plan"]["boundary_samples_excluded_per_side"]),
        ),
        ("end_to_end_descriptive", "job_start", "job_end", 0),
    ):
        windows[name] = window_statistics(
            samples,
            start=_marker_epoch(output_root, start_stage, card_hash, source_hash),
            end=_marker_epoch(output_root, end_stage, card_hash, source_hash),
            boundary_exclusion_per_side=boundary_exclusion,
        )
    hardware = card["hardware_plan"]
    evaluation = windows["evaluation_validity"]
    checks = evaluation_gate_checks(evaluation, hardware)
    raw_copy = output_root / "raw_gpu_telemetry.csv"
    if raw_copy.exists():
        raise FileExistsError(raw_copy)
    with raw_telemetry.open("rb") as source, raw_copy.open("xb") as target:
        shutil.copyfileobj(source, target)
    report_path = output_root / "telemetry_audit.json"
    report = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "raw_telemetry_path": str(raw_copy),
        "raw_telemetry_sha256": sha256_path(raw_copy),
        "gpu_uuid": gpu_uuid,
        "slurm_job_id": slurm_job_id,
        "windows": windows,
        "evaluation_checks": checks,
        "evaluation_gates_every_retained_sample_including_zeros": True,
        "generation_and_end_to_end_are_descriptive_only": True,
        "no_padding_policy": hardware["no_padding"],
        "scientific_payload_opened": False,
    }
    write_json_once(report_path, report)
    if not all(checks.values()):
        raise RuntimeError("Evaluation GPU-utilization gate failed; no outcome receipt issued")
    receipt_path = output_root / "outcome_guard_receipt.json"
    receipt = {
        "schema_version": 1,
        "status": "authorized_for_dependent_cpu_summary",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "runtime_lineage_path": str(runtime_path),
        "runtime_lineage_sha256": sha256_path(runtime_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": runtime["dataset_manifest_sha256"],
        "scientific_payload_path": str(scientific_path),
        "scientific_payload_sha256": runtime["scientific_payload_sha256"],
        "checkpoint_roster_sha256": runtime["checkpoint_roster_sha256"],
        "checkpoint_roster": runtime["checkpoint_roster"],
        "gpu_uuid": gpu_uuid,
        "slurm_job_id": slurm_job_id,
        "telemetry_audit_path": str(report_path),
        "telemetry_audit_sha256": sha256_path(report_path),
        "crossed_cells": 60,
        "scientific_payload_opened": False,
    }
    write_json_once(receipt_path, receipt)
    return {
        "telemetry_audit_path": str(report_path),
        "telemetry_audit_sha256": sha256_path(report_path),
        "outcome_guard_receipt_path": str(receipt_path),
        "outcome_guard_receipt_sha256": sha256_path(receipt_path),
    }


def main() -> None:
    args = parse_args()
    assert_runtime_values_safe(
        [
            args.card,
            args.source_manifest,
            args.output_root,
            args.raw_telemetry,
            args.expected_card_sha256,
            args.expected_source_manifest_sha256,
        ]
    )
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    if args.output_root != Path(card["prospective_datasets"]["output_root"]):
        raise RuntimeError("Telemetry output root differs from the frozen card")
    result = audit_and_receipt(
        card,
        card_hash=card_hash,
        source_hash=source_hash,
        output_root=args.output_root,
        raw_telemetry=args.raw_telemetry,
    )
    print(json.dumps({"status": "passed", **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
