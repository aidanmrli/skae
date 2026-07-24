"""One fail-closed A100 job: generate three field sets, then score 20 checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

from experiments.neurips_2026.allen_cahn_forecast_replication.core import (
    crossed_rows,
    generate_all_fields,
    realized_rng_streams,
)
from experiments.neurips_2026.allen_cahn_forecast_replication.io import (
    CARD_PATH,
    MANIFEST_PATH,
    assert_runtime_values_safe,
    checkpoint_specs,
    field_payload,
    load_card,
    load_checkpoint_model,
    load_fields_only,
    sha256_path,
    torch_save_once,
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
    return parser.parse_args()


def _marker(root: Path, stage: str, *, card_hash: str, source_hash: str) -> Path:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    path = root / "markers" / f"{stage}.json"
    write_json_once(
        path,
        {
            "schema_version": 1,
            "stage": stage,
            "epoch_seconds": time.time(),
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
        },
    )
    return path


def _checkpoint_roster_rows(card: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "checkpoint_step": int(row["checkpoint_step"]),
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in card["checkpoint_roster"]["runs"]
    ]
    if len(rows) != 20 or len({(row["arm"], row["seed"]) for row in rows}) != 20:
        raise RuntimeError("Runtime checkpoint hash roster is not the exact 20 rows")
    return rows


def _checkpoint_roster_digest(card: dict[str, Any]) -> str:
    rows = _checkpoint_roster_rows(card)
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_a100() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The frozen job requires exactly one visible CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name:
        raise RuntimeError(f"The frozen job requires an A100, got {name}")
    return torch.device("cuda:0")


def assert_precision_contract(card: dict[str, Any]) -> dict[str, Any]:
    numerics = card["evaluation"]["numerics"]
    observed = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    expected = {
        "float32_matmul_precision": str(numerics["float32_matmul_precision"]),
        "cuda_matmul_allow_tf32": bool(numerics["expected_cuda_matmul_allow_tf32"]),
        "cudnn_allow_tf32": bool(numerics["expected_cudnn_allow_tf32"]),
    }
    if observed != expected:
        raise RuntimeError(f"Float32 precision contract drifted: {observed} != {expected}")
    return observed


def configure_precision(card: dict[str, Any]) -> dict[str, Any]:
    torch.set_float32_matmul_precision(
        str(card["evaluation"]["numerics"]["float32_matmul_precision"])
    )
    return assert_precision_contract(card)


def main() -> None:
    args = parse_args()
    assert_runtime_values_safe(
        [
            args.card,
            args.source_manifest,
            args.output_root,
            args.expected_card_sha256,
            args.expected_source_manifest_sha256,
            *os.environ.values(),
        ]
    )
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    expected_root = Path(card["prospective_datasets"]["output_root"])
    if args.output_root != expected_root:
        raise RuntimeError(f"Output root differs from card: {args.output_root} != {expected_root}")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing pre-existing experiment root {args.output_root}")
    args.output_root.mkdir(parents=False, exist_ok=False)
    device = _require_a100()
    torch.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats(device)
    _marker(args.output_root, "job_start", card_hash=card_hash, source_hash=source_hash)

    rng_proof = realized_rng_streams(card)
    _marker(args.output_root, "generation_start", card_hash=card_hash, source_hash=source_hash)
    generated = generate_all_fields(card, device=device)
    _marker(args.output_root, "generation_end", card_hash=card_hash, source_hash=source_hash)

    dataset_records = []
    loaded_fields = []
    for dataset_index, (seed, relative) in enumerate(
        zip(
            card["prospective_datasets"]["seeds"],
            card["prospective_datasets"]["paths"],
            strict=True,
        )
    ):
        path = args.output_root / relative
        payload = field_payload(
            generated[dataset_index],
            card,
            dataset_index=dataset_index,
            seed=int(seed),
        )
        torch_save_once(path, payload)
        digest = sha256_path(path)
        loaded = load_fields_only(
            path,
            card,
            expected_sha256=digest,
            dataset_index=dataset_index,
            seed=int(seed),
        )
        loaded_fields.append(loaded)
        dataset_records.append(
            {
                "dataset_index": dataset_index,
                "dataset_seed": int(seed),
                "path": str(path),
                "sha256": digest,
                "field_shape": list(payload["fields"].shape),
                "top_level_keys": sorted(payload),
                "metadata_keys": sorted(payload["metadata"]),
            }
        )
    del generated
    dataset_manifest_path = args.output_root / "dataset_manifest.json"
    dataset_manifest = {
        "schema_version": 1,
        "status": "field_only_validated_and_hashed_before_evaluation",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "rng_stream_proof": rng_proof,
        "datasets": dataset_records,
    }
    write_json_once(dataset_manifest_path, dataset_manifest)
    dataset_manifest_hash = sha256_path(dataset_manifest_path)
    packed_fields = torch.stack(loaded_fields).to(device=device, dtype=torch.float32)
    if tuple(packed_fields.shape) != (3, 256, 201, 512):
        raise AssertionError(f"Unexpected evaluator field tensor {tuple(packed_fields.shape)}")

    precision = configure_precision(card)
    specs_and_models = []
    for spec in checkpoint_specs(card):
        model = load_checkpoint_model(spec, card, device=device)
        specs_and_models.append((spec, model))
    _marker(args.output_root, "evaluation_start", card_hash=card_hash, source_hash=source_hash)
    rows = crossed_rows(specs_and_models, packed_fields, card)
    if assert_precision_contract(card) != precision:
        raise RuntimeError("Float32 precision flags changed during crossed evaluation")
    _marker(args.output_root, "evaluation_end", card_hash=card_hash, source_hash=source_hash)

    peak_memory = int(torch.cuda.max_memory_allocated(device))
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    if peak_memory / total_memory > float(card["hardware_plan"]["maximum_peak_memory_fraction"]):
        raise RuntimeError("Frozen evaluation exceeded the peak-memory fraction guard")
    scientific_path = args.output_root / "scientific_curves.json"
    scientific_payload = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "checkpoint_roster_sha256": _checkpoint_roster_digest(card),
        "crossed_cells": 60,
        "rows": rows,
    }
    write_json_once(scientific_path, scientific_payload)
    scientific_hash = sha256_path(scientific_path)
    runtime_lineage_path = args.output_root / "runtime_lineage.json"
    runtime_lineage = {
        "schema_version": 1,
        "status": "scientific_payload_written_but_not_authorized_for_summary",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": dataset_manifest_hash,
        "scientific_payload_path": str(scientific_path),
        "scientific_payload_sha256": scientific_hash,
        "checkpoint_roster_sha256": _checkpoint_roster_digest(card),
        "checkpoint_roster": _checkpoint_roster_rows(card),
        "crossed_cells": 60,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
            "git_commit": os.environ.get("SKAE_GIT_COMMIT", "not_recorded"),
            "peak_gpu_memory_bytes": peak_memory,
            "total_gpu_memory_bytes": total_memory,
        },
        "scientific_metrics_printed": False,
    }
    write_json_once(runtime_lineage_path, runtime_lineage)
    _marker(args.output_root, "job_end", card_hash=card_hash, source_hash=source_hash)
    print(
        json.dumps(
            {
                "status": runtime_lineage["status"],
                "scientific_payload_sha256": scientific_hash,
                "runtime_lineage_sha256": sha256_path(runtime_lineage_path),
                "scientific_metrics_printed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
