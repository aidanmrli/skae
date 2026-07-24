"""GPU evaluation of frozen physics metrics on authenticated Allen--Cahn fields."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch

from experiments.neurips_2026.allen_cahn_forecast_replication.core import direct_rollout
from experiments.neurips_2026.allen_cahn_forecast_replication.io import (
    CheckpointSpec,
    load_checkpoint_model,
    load_fields_only,
    load_pinned_module,
    pinned_source,
)
from experiments.neurips_2026.allen_cahn_physics_metrics.core import (
    CHANNELS,
    GRID_SIZE,
    HORIZON,
    field_features,
    score_candidate,
    validate_score_record,
)
from experiments.neurips_2026.allen_cahn_physics_metrics.io import (
    CARD_PATH,
    MANIFEST_PATH,
    assert_paths_sealed,
    authenticated_prior,
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
    return parser.parse_args()


def _require_a100() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Physics evaluation requires exactly one visible CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "A100" not in name:
        raise RuntimeError(f"Physics evaluation requires an A100, got {name}")
    return torch.device("cuda:0")


def _marker(root: Path, stage: str, card_hash: str, source_hash: str) -> None:
    torch.cuda.synchronize()
    write_json_once(
        root / "markers" / f"{stage}.json",
        {
            "schema_version": 1,
            "stage": stage,
            "epoch_seconds": time.time(),
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
        },
    )


def _known_centers(source_card: dict[str, Any], device: torch.device) -> torch.Tensor:
    module = load_pinned_module(pinned_source(source_card, "physics_and_initial_conditions"))
    config = source_card["system_and_generator"]
    source_config = module.SpatialReactionDiffusionConfig(
        source_system="allen_cahn_4",
        allen_cahn_beta=float(config["allen_cahn_beta"]),
        allen_cahn_reaction_strength=float(config["allen_cahn_reaction_strength"]),
        allen_cahn_center_radius=float(config["allen_cahn_center_radius"]),
    )
    centers = module.extract_attractor_centers(
        module.get_source_system("allen_cahn_4", source_config)
    ).to(device=device, dtype=torch.float32)
    if tuple(centers.shape) != (4, 2):
        raise AssertionError("Pinned scoring system does not have exactly four 2D wells")
    return centers


def _load_fields(
    prior: dict[str, Any], device: torch.device
) -> list[torch.Tensor]:
    source_card = prior["source_card"]
    result = []
    for row in prior["datasets"]:
        fields = load_fields_only(
            Path(row["path"]),
            source_card,
            expected_sha256=row["sha256"],
            dataset_index=int(row["dataset_index"]),
            seed=int(row["dataset_seed"]),
        )
        expected = (256, HORIZON + 1, GRID_SIZE * GRID_SIZE * CHANNELS)
        if tuple(fields.shape) != expected:
            raise AssertionError(f"Authenticated field panel changed shape: {tuple(fields.shape)}")
        result.append(fields.to(device=device, dtype=torch.float32).reshape(256, HORIZON + 1, 16, 16, 2))
    return result


def _load_models(
    prior: dict[str, Any], device: torch.device
) -> list[tuple[CheckpointSpec, torch.nn.Module]]:
    source_card = prior["source_card"]
    models = []
    for row in prior["checkpoint_roster"]:
        spec = CheckpointSpec(
            arm=row["arm"],
            seed=int(row["seed"]),
            checkpoint_step=int(row["checkpoint_step"]),
            path=Path(row["path"]),
            sha256=row["sha256"],
        )
        models.append((spec, load_checkpoint_model(spec, source_card, device=device)))
    if len(models) != 20:
        raise AssertionError("Did not preload the exact twenty-checkpoint roster")
    return models


def _write_snapshots_once(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return sha256_path(path)


def main() -> None:
    args = parse_args()
    assert_paths_sealed(
        [args.card, args.source_manifest, args.output_root, *os.environ.values()]
    )
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        args.source_manifest, expected_sha256=args.expected_source_manifest_sha256
    )
    expected_root = Path(card["execution"]["output_root"])
    if args.output_root != expected_root or args.output_root.exists():
        raise RuntimeError("Output root differs from the card or already exists")
    prior = authenticated_prior(card)
    device = _require_a100()
    torch.set_float32_matmul_precision("high")
    if not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
        raise RuntimeError("Pinned direct-rollout TF32 contract is not active")
    fields = _load_fields(prior, device)
    centers = _known_centers(prior["source_card"], device)
    coefficients = card["metric_contract"]["energy_coefficients"]
    truth_features = [
        field_features(
            panel[:, 1:],
            centers,
            beta=float(coefficients["beta"]),
            reaction_strength=float(coefficients["reaction_strength"]),
            diffusion=float(coefficients["diffusion"]),
        )
        for panel in fields
    ]
    models = _load_models(prior, device)
    rows: list[dict[str, Any]] = []
    snapshots: dict[str, Any] = {
        "schema_version": 1,
        "contract": card["visualization_contract"],
        "truth_and_persistence": [],
        "model_predictions": [],
    }
    visual_trajectories = torch.tensor(
        card["visualization_contract"]["trajectory_indices"], device=device
    )
    visual_observations = card["visualization_contract"]["observation_indices"]
    for dataset_index, panel in enumerate(fields):
        persistence = panel[:, :1].expand(-1, HORIZON, -1, -1, -1)
        score = score_candidate(
            persistence,
            panel[:, 1:],
            centers,
            beta=float(coefficients["beta"]),
            reaction_strength=float(coefficients["reaction_strength"]),
            diffusion=float(coefficients["diffusion"]),
            truth_features=truth_features[dataset_index],
        )
        rows.append(
            {
                "arm": "persistence",
                "model_seed": None,
                "dataset_index": dataset_index,
                "dataset_seed": int(prior["datasets"][dataset_index]["dataset_seed"]),
                **score,
            }
        )
        selected = panel.index_select(0, visual_trajectories)
        snapshots["truth_and_persistence"].append(
            {
                "dataset_index": dataset_index,
                "truth": selected[:, visual_observations].detach().cpu(),
                "persistence": selected[:, :1]
                .expand(-1, len(visual_observations), -1, -1, -1)
                .detach()
                .cpu(),
            }
        )

    _marker(args.output_root, "evaluation_start", card_hash, source_hash)
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for spec, model in models:
            for dataset_index, panel in enumerate(fields):
                predictions = direct_rollout(
                    model, panel[:, 0].reshape(256, -1), horizon=HORIZON
                ).reshape(256, HORIZON, GRID_SIZE, GRID_SIZE, CHANNELS)
                score = score_candidate(
                    predictions,
                    panel[:, 1:],
                    centers,
                    beta=float(coefficients["beta"]),
                    reaction_strength=float(coefficients["reaction_strength"]),
                    diffusion=float(coefficients["diffusion"]),
                    truth_features=truth_features[dataset_index],
                )
                rows.append(
                    {
                        "arm": spec.arm,
                        "model_seed": int(spec.seed),
                        "checkpoint_step": int(spec.checkpoint_step),
                        "checkpoint_sha256": spec.sha256,
                        "dataset_index": dataset_index,
                        "dataset_seed": int(prior["datasets"][dataset_index]["dataset_seed"]),
                        **score,
                    }
                )
                if int(spec.seed) == int(card["visualization_contract"]["model_seed"]):
                    chosen = predictions.index_select(0, visual_trajectories)
                    prediction_indices = [observation - 1 for observation in visual_observations[1:]]
                    model_frames = torch.cat(
                        (panel.index_select(0, visual_trajectories)[:, :1], chosen[:, prediction_indices]),
                        dim=1,
                    )
                    snapshots["model_predictions"].append(
                        {
                            "arm": spec.arm,
                            "model_seed": int(spec.seed),
                            "dataset_index": dataset_index,
                            "fields": model_frames.detach().cpu(),
                        }
                    )
                del predictions
    _marker(args.output_root, "evaluation_end", card_hash, source_hash)

    expected_model_cells = {
        (arm, seed, dataset)
        for arm in ("dense", "sparse")
        for seed in range(64, 74)
        for dataset in range(3)
    }
    actual_model_cells = {
        (row["arm"], row["model_seed"], row["dataset_index"])
        for row in rows
        if row["arm"] != "persistence"
    }
    persistence_cells = [row for row in rows if row["arm"] == "persistence"]
    if actual_model_cells != expected_model_cells or len(rows) != 63 or len(persistence_cells) != 3:
        raise AssertionError("Physics packet is not the exact 20x3 panel plus three controls")
    for row in rows:
        validate_score_record(row)
    payload = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "prior_receipt_sha256": prior["prior_receipt_sha256"],
        "checkpoint_roster_sha256": prior["checkpoint_roster_sha256"],
        "rows": rows,
        "outcomes_printed": False,
    }
    payload_path = args.output_root / "scientific_physics_curves.json"
    write_json_once(payload_path, payload)
    snapshot_path = args.output_root / "visualization_snapshots.pt"
    snapshot_hash = _write_snapshots_once(snapshot_path, snapshots)
    runtime = {
        "schema_version": 1,
        "status": "physics_payload_written_but_not_authorized_for_summary",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "prior_receipt_sha256": prior["prior_receipt_sha256"],
        "checkpoint_roster_sha256": prior["checkpoint_roster_sha256"],
        "scientific_payload_path": str(payload_path),
        "scientific_payload_sha256": sha256_path(payload_path),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot_hash,
        "row_count": len(rows),
        "scientific_metrics_printed": False,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "total_gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
    }
    write_json_once(args.output_root / "runtime_lineage.json", runtime)


if __name__ == "__main__":
    main()
