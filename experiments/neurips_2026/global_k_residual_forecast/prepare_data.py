"""Generate outcome-free whole trajectories before any GPU forecast evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.neurips_2026.global_k_residual_forecast.protocol import (
    DEFAULT_CARD,
    DEFAULT_SOURCES,
    DEFAULT_TASKS,
    authenticate_checkpoint_roster,
    authenticate_v2_inputs,
    atomic_json,
    load_frozen_protocol,
    load_torch_payload,
    read_verified_bytes,
    sha256_path,
    task_by_index,
)
from skae.config import Config
from skae.data import VectorWrapper, make_env


def _trajectory_tensor(env, *, count: int, horizon: int, seed: int) -> torch.Tensor:
    with torch.no_grad():
        value = VectorWrapper(env, count).generate_sequence_batch(
            rng=torch.Generator().manual_seed(seed),
            window_length=horizon,
        ).contiguous()
    if value.shape != (count, horizon + 1, 2):
        raise RuntimeError(f"Unexpected trajectory shape: {value.shape}")
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise RuntimeError("Generated trajectories must be finite float32")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--expected-task-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    card, tasks, freeze = load_frozen_protocol(
        card_path=args.card,
        task_path=args.tasks,
        source_manifest_path=args.sources,
        expected_card_sha256=args.expected_card_sha256,
        expected_task_sha256=args.expected_task_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
    )
    authenticate_v2_inputs(card)
    authenticate_checkpoint_roster(tasks)
    first = task_by_index(tasks, 0)
    checkpoint_path = Path(first["sparse_checkpoint"]["path"])
    checkpoint_bytes = read_verified_bytes(
        checkpoint_path,
        first["sparse_checkpoint"]["sha256"],
        "frozen sparse checkpoint used for environment config",
    )
    checkpoint = load_torch_payload(checkpoint_bytes, map_location="cpu")
    cfg = Config.from_dict(checkpoint["config"])
    env = make_env(cfg)
    if str(cfg.ENV.ENV_NAME) != card["benchmark"]["system"]:
        raise RuntimeError("Checkpoint environment does not match the card")
    if float(env.unwrapped.dt) != float(card["benchmark"]["dt"]):
        raise RuntimeError("Environment dt does not match the card")

    data_dir = args.output_root / "outcome_free_data"
    if data_dir.exists():
        raise FileExistsError(f"Refusing pre-existing data directory: {data_dir}")
    data_dir.mkdir(parents=True)
    data_dir.chmod(0o700)

    corpora = card["outcome_free_trajectory_corpora"]
    rows = []
    for role in ("route_fit", "route_audit"):
        spec = corpora[role]
        tensor = _trajectory_tensor(
            env,
            count=int(spec["trajectory_count"]),
            horizon=int(spec["horizon_steps"]),
            seed=int(spec["seed"]),
        )
        path = data_dir / f"{role}.pt"
        torch.save(
            {
                "trajectories": tensor,
                "metadata": {
                    "protocol_id": card["protocol_id"],
                    "role": role,
                    "seed": int(spec["seed"]),
                    "shape": list(tensor.shape),
                    "contains_forecast_or_representation_outcomes": False,
                },
            },
            path,
        )
        rows.append(
            {
                "role": role,
                "seed": int(spec["seed"]),
                "path": str(path),
                "sha256": sha256_path(path),
            }
        )

    evaluation = corpora["evaluation"]
    for dataset_index, seed in enumerate(evaluation["seeds"]):
        tensor = _trajectory_tensor(
            env,
            count=int(evaluation["trajectory_count_each"]),
            horizon=int(evaluation["horizon_steps"]),
            seed=int(seed),
        )
        path = data_dir / f"evaluation_{dataset_index}.pt"
        torch.save(
            {
                "trajectories": tensor,
                "metadata": {
                    "protocol_id": card["protocol_id"],
                    "role": "evaluation",
                    "dataset_index": dataset_index,
                    "seed": int(seed),
                    "shape": list(tensor.shape),
                    "contains_forecast_or_representation_outcomes": False,
                },
            },
            path,
        )
        rows.append(
            {
                "role": "evaluation",
                "dataset_index": dataset_index,
                "seed": int(seed),
                "path": str(path),
                "sha256": sha256_path(path),
            }
        )

    smoke = corpora["smoke_evaluation"]
    tensor = _trajectory_tensor(
        env,
        count=int(smoke["trajectory_count"]),
        horizon=int(smoke["horizon_steps"]),
        seed=int(smoke["seed"]),
    )
    path = data_dir / "smoke_evaluation_0.pt"
    torch.save(
        {
            "trajectories": tensor,
            "metadata": {
                "protocol_id": card["protocol_id"],
                "role": "smoke_evaluation",
                "dataset_index": int(smoke["dataset_index"]),
                "seed": int(smoke["seed"]),
                "shape": list(tensor.shape),
                "contains_forecast_or_representation_outcomes": False,
            },
        },
        path,
    )
    rows.append(
        {
            "role": "smoke_evaluation",
            "dataset_index": int(smoke["dataset_index"]),
            "seed": int(smoke["seed"]),
            "path": str(path),
            "sha256": sha256_path(path),
        }
    )

    manifest = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": "outcome_free_physical_trajectory_manifest",
        "freeze": freeze,
        "rows": rows,
        "forbidden_content_assertion": (
            "No model forecast, representation, support, family, basin label, "
            "comparison metric, selector, or scientific decision was computed."
        ),
    }
    manifest_path = data_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "artifact_role": manifest["artifact_role"],
                "row_count": len(rows),
                "manifest_sha256": sha256_path(manifest_path),
                "outcomes_inspected": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
