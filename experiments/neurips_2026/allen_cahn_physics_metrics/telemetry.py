"""Audit the physics-scoring GPU window and issue a metric-free receipt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Any

from experiments.neurips_2026.allen_cahn_forecast_replication.telemetry import (
    evaluation_gate_checks,
    parse_samples,
    window_statistics,
)
from experiments.neurips_2026.allen_cahn_physics_metrics.io import (
    CARD_PATH,
    MANIFEST_PATH,
    assert_paths_sealed,
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


def _marker_epoch(root: Path, stage: str, card_hash: str, source_hash: str) -> float:
    payload = duplicate_safe_json(root / "markers" / f"{stage}.json")
    if (
        payload.get("stage") != stage
        or payload.get("card_sha256") != card_hash
        or payload.get("source_manifest_sha256") != source_hash
    ):
        raise RuntimeError(f"Marker lineage failed for {stage}")
    return float(payload["epoch_seconds"])


def _bind_job(runtime: dict[str, Any]) -> str:
    runtime_job = str(runtime["environment"].get("slurm_job_id", "not_recorded"))
    current_job = str(os.environ.get("SLURM_JOB_ID", "not_recorded"))
    if runtime_job != current_job or runtime_job in {"", "None", "not_recorded"}:
        raise RuntimeError("Scientific evaluation and telemetry lack one exact SLURM job ID")
    return runtime_job


def main() -> None:
    args = parse_args()
    assert_paths_sealed(
        [args.card, args.source_manifest, args.output_root, args.raw_telemetry]
    )
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        args.source_manifest, expected_sha256=args.expected_source_manifest_sha256
    )
    if args.output_root != Path(card["execution"]["output_root"]):
        raise RuntimeError("Telemetry root differs from the frozen card")
    runtime_path = args.output_root / "runtime_lineage.json"
    runtime = duplicate_safe_json(runtime_path)
    if (
        runtime.get("status") != "physics_payload_written_but_not_authorized_for_summary"
        or runtime.get("card_sha256") != card_hash
        or runtime.get("source_manifest_sha256") != source_hash
        or runtime.get("row_count") != 63
        or runtime.get("scientific_metrics_printed") is not False
    ):
        raise RuntimeError("Metric-free runtime lineage failed")
    job_id = _bind_job(runtime)
    payload_path = Path(runtime["scientific_payload_path"])
    snapshot_path = Path(runtime["snapshot_path"])
    if sha256_path(payload_path) != runtime["scientific_payload_sha256"]:
        raise RuntimeError("Physics payload hash changed")
    if sha256_path(snapshot_path) != runtime["snapshot_sha256"]:
        raise RuntimeError("Snapshot payload hash changed")
    samples = parse_samples(args.raw_telemetry)
    start = _marker_epoch(args.output_root, "evaluation_start", card_hash, source_hash)
    end = _marker_epoch(args.output_root, "evaluation_end", card_hash, source_hash)
    hardware = card["hardware_plan"]
    window = window_statistics(
        samples,
        start=start,
        end=end,
        boundary_exclusion_per_side=int(hardware["boundary_samples_excluded_per_side"]),
    )
    checks = evaluation_gate_checks(window, hardware)
    raw_copy = args.output_root / "raw_gpu_telemetry.csv"
    if raw_copy.exists():
        raise FileExistsError(raw_copy)
    with args.raw_telemetry.open("rb") as source, raw_copy.open("xb") as target:
        shutil.copyfileobj(source, target)
    report_path = args.output_root / "telemetry_audit.json"
    report = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "raw_telemetry_sha256": sha256_path(raw_copy),
        "gpu_uuid": samples[0]["gpu_uuid"],
        "slurm_job_id": job_id,
        "evaluation_window": window,
        "evaluation_checks": checks,
        "every_retained_sample_including_zero_used": True,
        "scientific_payload_opened": False,
        "no_padding_policy": hardware["no_padding"],
    }
    write_json_once(report_path, report)
    if not all(checks.values()):
        raise RuntimeError("GPU-utilization gate failed; no outcome receipt issued")
    receipt_path = args.output_root / "outcome_guard_receipt.json"
    write_json_once(
        receipt_path,
        {
            "schema_version": 1,
            "status": "authorized_for_dependent_cpu_summary",
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "runtime_lineage_path": str(runtime_path),
            "runtime_lineage_sha256": sha256_path(runtime_path),
            "scientific_payload_path": str(payload_path),
            "scientific_payload_sha256": runtime["scientific_payload_sha256"],
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": runtime["snapshot_sha256"],
            "prior_receipt_sha256": runtime["prior_receipt_sha256"],
            "checkpoint_roster_sha256": runtime["checkpoint_roster_sha256"],
            "telemetry_audit_path": str(report_path),
            "telemetry_audit_sha256": sha256_path(report_path),
            "gpu_uuid": samples[0]["gpu_uuid"],
            "slurm_job_id": job_id,
            "row_count": 63,
            "scientific_payload_opened": False,
        },
    )


if __name__ == "__main__":
    main()
