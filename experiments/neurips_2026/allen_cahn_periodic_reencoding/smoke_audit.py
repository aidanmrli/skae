"""Audit outcome-free GPU smoke telemetry and issue a launch receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    CARD_PATH,
    MANIFEST_PATH,
    duplicate_safe_json,
    load_card,
    sha256_path,
    verify_file,
    verify_source_manifest,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.telemetry import (
    gate_checks,
    hardware_plan,
    read_samples,
    window_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--raw-telemetry", type=Path, required=True)
    return parser.parse_args()


def _marker(
    root: Path,
    name: str,
    *,
    card_hash: str,
    source_hash: str,
    slurm_job_id: str,
) -> tuple[dict, Path]:
    path = root / "markers" / f"{name}.json"
    marker = duplicate_safe_json(path)
    expected = {
        "stage": name,
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "slurm_job_id": slurm_job_id,
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Smoke marker lineage failed for {name}")
    return marker, path


def main() -> None:
    args = parse_args()
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    if args.root != Path(card["outcome_free_smoke"]["output_root"]):
        raise RuntimeError("Smoke root differs from the frozen card")
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    if not slurm_job_id:
        raise RuntimeError("Smoke audit requires the smoke SLURM job")
    runtime_path = args.root / "smoke_runtime.json"
    runtime = duplicate_safe_json(runtime_path)
    expected_runtime = {
        "status": "outcome_free_workload_complete",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "slurm_job_id": slurm_job_id,
        "trained_checkpoints_loaded": 0,
        "physical_datasets_loaded_or_generated": 0,
        "synthetic_models": 2,
        "synthetic_evaluator_calls": 20,
        "scientific_outcomes_accessed": False,
    }
    if any(runtime.get(key) != value for key, value in expected_runtime.items()):
        raise RuntimeError("Smoke runtime violates the outcome-free contract")
    start, start_path = _marker(
        args.root,
        "smoke_start",
        card_hash=card_hash,
        source_hash=source_hash,
        slurm_job_id=slurm_job_id,
    )
    end, end_path = _marker(
        args.root,
        "smoke_end",
        card_hash=card_hash,
        source_hash=source_hash,
        slurm_job_id=slurm_job_id,
    )
    samples = read_samples(args.raw_telemetry)
    if "A100" not in str(samples[0]["gpu_name"]):
        raise RuntimeError("Smoke telemetry is not from an A100")
    window = window_statistics(
        samples,
        start=float(start["epoch_seconds"]),
        end=float(end["epoch_seconds"]),
    )
    checks = gate_checks(window, hardware_plan(card))
    raw_path = args.root / "raw_gpu_telemetry.csv"
    if raw_path.exists():
        raise FileExistsError(raw_path)
    with args.raw_telemetry.open("rb") as source, raw_path.open("xb") as target:
        shutil.copyfileobj(source, target)
    audit_path = args.root / "smoke_telemetry_audit.json"
    write_json_once(
        audit_path,
        {
            "schema_version": 1,
            "status": "passed" if all(checks.values()) else "failed",
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "slurm_job_id": slurm_job_id,
            "runtime_path": str(runtime_path),
            "runtime_sha256": sha256_path(runtime_path),
            "raw_telemetry_path": str(raw_path),
            "raw_telemetry_sha256": sha256_path(raw_path),
            "evaluation_window": window,
            "checks": checks,
            "scientific_outcomes_accessed": False,
        },
    )
    if not all(checks.values()):
        raise RuntimeError("Outcome-free GPU smoke failed its utilization gate")
    receipt_path = args.root / "smoke_receipt.json"
    write_json_once(
        receipt_path,
        {
            "schema_version": 1,
            "status": "passed_outcome_free_gpu_smoke",
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "slurm_job_id": slurm_job_id,
            "runtime_path": str(runtime_path),
            "runtime_sha256": sha256_path(runtime_path),
            "start_marker_path": str(start_path),
            "start_marker_sha256": sha256_path(start_path),
            "end_marker_path": str(end_path),
            "end_marker_sha256": sha256_path(end_path),
            "telemetry_audit_path": str(audit_path),
            "telemetry_audit_sha256": sha256_path(audit_path),
            "raw_telemetry_path": str(raw_path),
            "raw_telemetry_sha256": sha256_path(raw_path),
            "scientific_outcomes_accessed": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "smoke_receipt_path": str(receipt_path),
                "smoke_receipt_sha256": sha256_path(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
