"""Outcome-free exact-shape GPU workload for the periodic evaluator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

import torch

from experiments.neurips_2026.allen_cahn_forecast_replication import io as parent_io
from experiments.neurips_2026.allen_cahn_periodic_reencoding.core import (
    evaluate_model_packed,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    CARD_PATH,
    MANIFEST_PATH,
    load_card,
    load_parent_card,
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


def _marker(
    root: Path,
    stage: str,
    *,
    card_hash: str,
    source_hash: str,
) -> Path:
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
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
        },
    )
    return path


def _device() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Smoke requires exactly one visible CUDA GPU")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    if "A100" not in name or total < 70 * 2**30:
        raise RuntimeError(f"Smoke requires an A100 80GB-class GPU, got {name}")
    return torch.device("cuda:0")


def _models(card: dict[str, Any], device: torch.device) -> list[torch.nn.Module]:
    parent = load_parent_card(card)
    module = parent_io.load_pinned_module(
        parent_io.pinned_source(parent, "checkpoint_model")
    )
    common = {
        "grid_size": 16,
        "channels": 2,
        "z_dim": 2048,
        "hidden_channels": 32,
        "num_blocks": 2,
        "lista_loops": 1,
        "decoder_kind": "upsample",
        "dense_activation": "tanh",
        "conv_activation": "tanh",
        "padding_mode": "circular",
    }
    configurations = [
        {**common, "encoder_kind": "dense", "lista_alpha": 0.0},
        {**common, "encoder_kind": "lista", "lista_alpha": 0.15},
    ]
    models = []
    for index, values in enumerate(configurations):
        torch.manual_seed(17_291 + index)
        config = module.SpatialConvKoopmanConfig.from_mapping(values)
        model = module.SpatialConvKoopman(config).float().to(device).eval()
        with torch.no_grad():
            model.kmat.copy_(
                0.995 * torch.eye(2048, dtype=torch.float32, device=device)
            )
        models.append(model)
    return models


def _synthetic_fields(device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(82_771)
    initial = 0.2 * torch.randn(
        (3, 256, 512), generator=generator, device=device, dtype=torch.float32
    )
    times = torch.arange(401, device=device, dtype=torch.float32)
    decay = torch.exp(-0.002 * times).view(1, 1, 401, 1)
    fields = (initial[:, :, None, :] * decay).contiguous()
    if tuple(fields.shape) != (3, 256, 401, 512):
        raise AssertionError("Synthetic field workload shape drifted")
    if not bool(torch.isfinite(fields).all()):
        raise FloatingPointError("Synthetic smoke fields are nonfinite")
    return fields


def main() -> None:
    args = parse_args()
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    expected_root = Path(card["outcome_free_smoke"]["output_root"])
    if args.output_root != expected_root:
        raise RuntimeError("Smoke root differs from the frozen card")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=False, exist_ok=False)
    device = _device()
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats(device)
    models = _models(card, device)
    fields = _synthetic_fields(device)
    grid = [*card["cadence_selection"]["cadence_grid"], 200]
    _marker(
        args.output_root,
        "smoke_start",
        card_hash=card_hash,
        source_hash=source_hash,
    )
    start = time.time()
    calls = 0
    for model in models:
        for cadence in grid:
            records = evaluate_model_packed(
                model,
                fields,
                horizon=400,
                period=None if cadence == "direct" else int(cadence),
                batch_size=768,
                max_decode_segment=100,
            )
            if len(records) != 3 or any(record["horizon"] != 400 for record in records):
                raise RuntimeError("Synthetic evaluator roster drifted")
            calls += 1
            del records
    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    _marker(
        args.output_root,
        "smoke_end",
        card_hash=card_hash,
        source_hash=source_hash,
    )
    peak = int(torch.cuda.max_memory_allocated(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    if peak / total >= float(card["hardware_plan"]["maximum_peak_memory_fraction"]):
        raise RuntimeError("Synthetic smoke exceeded the strict peak-memory bound")
    write_json_once(
        args.output_root / "smoke_runtime.json",
        {
            "schema_version": 1,
            "status": "outcome_free_workload_complete",
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
            "trained_checkpoints_loaded": 0,
            "physical_datasets_loaded_or_generated": 0,
            "synthetic_models": 2,
            "synthetic_evaluator_calls": calls,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": peak,
            "total_gpu_memory_bytes": total,
            "gpu_name": torch.cuda.get_device_name(device),
            "scientific_outcomes_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
