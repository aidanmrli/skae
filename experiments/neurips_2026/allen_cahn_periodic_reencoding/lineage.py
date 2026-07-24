"""Metric-free runtime lineage helpers for the periodic packet."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import write_json_once


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_runtime_lineage(
    *,
    root: Path,
    card: dict[str, Any],
    card_hash: str,
    source_hash: str,
    smoke_receipt: Path,
    smoke_hash: str,
    selection_path: Path,
    selection_hash: str,
    validation_manifest_path: Path,
    validation_manifest_hash: str,
    test_manifest_path: Path,
    test_manifest_hash: str,
    scientific_path: Path,
    scientific_hash: str,
    specs_and_models: Iterable[tuple[Any, torch.nn.Module]],
    rng_proof: dict[str, Any],
    precision: dict[str, Any],
    device: torch.device,
) -> Path:
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    properties = torch.cuda.get_device_properties(device)
    total_memory = int(properties.total_memory)
    if peak_memory / total_memory >= float(
        card["hardware_plan"]["maximum_peak_memory_fraction"]
    ):
        raise RuntimeError("Peak GPU memory violated the frozen strict upper bound")
    checkpoint_roster = [
        {
            "arm": spec.arm,
            "seed": int(spec.seed),
            "checkpoint_step": int(spec.checkpoint_step),
            "path": str(spec.path),
            "sha256": spec.sha256,
        }
        for spec, _model in specs_and_models
    ]
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "not_recorded")
    path = root / "runtime_lineage.json"
    write_json_once(
        path,
        {
            "schema_version": 1,
            "status": "scientific_payload_written_but_not_authorized_for_summary",
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "smoke_receipt_path": str(smoke_receipt),
            "smoke_receipt_sha256": smoke_hash,
            "slurm_job_id": slurm_job_id,
            "scientific_metrics_printed": False,
            "selection_decision_path": str(selection_path),
            "selection_decision_sha256": selection_hash,
            "validation_data_manifest_path": str(validation_manifest_path),
            "validation_data_manifest_sha256": validation_manifest_hash,
            "test_data_manifest_path": str(test_manifest_path),
            "test_data_manifest_sha256": test_manifest_hash,
            "scientific_payload_path": str(scientific_path),
            "scientific_payload_sha256": scientific_hash,
            "scientific_hash": scientific_hash,
            "checkpoint_roster": checkpoint_roster,
            "checkpoint_roster_sha256": canonical_digest(checkpoint_roster),
            "rng_stream_proof": rng_proof,
            "environment": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_uuid": getattr(properties, "uuid", "not_recorded"),
                "slurm_job_id": slurm_job_id,
                "git_commit": os.environ.get("SKAE_GIT_COMMIT", "not_recorded"),
                "peak_gpu_memory_bytes": peak_memory,
                "total_gpu_memory_bytes": total_memory,
                **precision,
            },
        },
    )
    return path
