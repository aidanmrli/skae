"""Write authenticated valid or invalid V2 reduction packets."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch

from .io import finite_tree, sha256_path, write_json_once


def _finish_packet(
    summary_dir: Path,
    *,
    card: dict[str, Any],
    roots: dict[str, str],
    receipt_path: Path,
    features_path: Path,
    decision_path: Path,
    rows_path: Path | None,
) -> None:
    provenance = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "authenticated_complete",
        **roots,
        "telemetry_receipt_sha256": sha256_path(receipt_path),
        "features_sha256": sha256_path(features_path),
        "decision_sha256": sha256_path(decision_path),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_available"),
        "cpu_summary_requested_gpu": False,
    }
    if rows_path is not None:
        provenance["rows_sha256"] = sha256_path(rows_path)
    provenance_path = summary_dir / "provenance.json"
    write_json_once(provenance_path, provenance)
    artifacts = {
        "features.pt": sha256_path(features_path),
        "telemetry_receipt.json": sha256_path(receipt_path),
        "decision.json": sha256_path(decision_path),
        "provenance.json": sha256_path(provenance_path),
    }
    if rows_path is not None:
        artifacts["rows.csv"] = sha256_path(rows_path)
    evidence = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "authenticated_packet_complete",
        **roots,
        "artifacts": artifacts,
    }
    write_json_once(summary_dir / "evidence_manifest.json", evidence)


def write_invalid_packet(
    output_root: Path,
    *,
    card: dict[str, Any],
    roots: dict[str, str],
    receipt_path: Path,
    features_path: Path,
    reasons: list[str],
    validity: dict[str, Any],
) -> None:
    summary_dir = output_root / "summary"
    if summary_dir.exists():
        raise FileExistsError(summary_dir)
    summary_dir.mkdir(parents=True)
    decision = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": "invalid",
        "decision_branch": "invalid",
        "invalid_reasons": reasons,
        "validity": validity,
        "no_v3": True,
        "claim_tiers": {
            name: False for name in card["primary_gate"]["claim_tiers"]
        },
        "attestation": "No probe fit, prediction, score, permutation, contrast, or scientific claim decision was computed because target validity failed first.",
        "claim_boundary": card["claim_boundary"],
    }
    if not finite_tree(decision):
        raise RuntimeError("Invalid decision contains nonfinite/null values")
    decision_path = summary_dir / "decision.json"
    write_json_once(decision_path, decision)
    _finish_packet(
        summary_dir,
        card=card,
        roots=roots,
        receipt_path=receipt_path,
        features_path=features_path,
        decision_path=decision_path,
        rows_path=None,
    )


def write_valid_packet(
    output_root: Path,
    *,
    card: dict[str, Any],
    roots: dict[str, str],
    receipt_path: Path,
    features_path: Path,
    decision: dict[str, Any],
    rows: list[dict[str, object]],
) -> None:
    summary_dir = output_root / "summary"
    if summary_dir.exists():
        raise FileExistsError(summary_dir)
    summary_dir.mkdir(parents=True)
    rows_path = summary_dir / "rows.csv"
    fieldnames = [
        "feature", "observation_index", "observation_time", "model_seed",
        "dataset_seed", "eligible_rows", "alpha", "cv_balanced_accuracy",
        "balanced_accuracy", "macro_f1", "accuracy",
    ]
    with rows_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if not finite_tree(decision):
        raise RuntimeError("Decision contains nonfinite/null values")
    decision_path = summary_dir / "decision.json"
    write_json_once(decision_path, decision)
    _finish_packet(
        summary_dir,
        card=card,
        roots=roots,
        receipt_path=receipt_path,
        features_path=features_path,
        decision_path=decision_path,
        rows_path=rows_path,
    )
