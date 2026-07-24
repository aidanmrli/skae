"""GPU orchestration for a residualized support-routed unchanged global K."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from experiments.neurips_2026.global_k_distinct_laws_v2_checkpoint_audit import (
    TrainedRun,
    trainable_parameter_counts,
)
from experiments.neurips_2026.global_k_residual_forecast.checkpoint_audit import (
    audit_authenticated_checkpoint,
)
from experiments.neurips_2026.global_k_residual_forecast.protocol import (
    DEFAULT_CARD,
    DEFAULT_SOURCES,
    DEFAULT_TASKS,
    authenticate_checkpoint_roster,
    authenticate_v2_inputs,
    atomic_json,
    load_frozen_protocol,
    load_json,
    load_torch_payload,
    read_verified_bytes,
    sha256_array,
    sha256_path,
    task_by_index,
)
from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import make_env
from skae.model import make_model
from experiments.neurips_2026.global_k_residual_forecast.rollout import (
    dense_forecasts,
    sparse_forecasts,
    stratify_after_forecasting,
)
from experiments.neurips_2026.global_k_residual_forecast.routing import (
    audit_routing,
    fit_codebook,
    matched_null_projectors,
)


def _load_model(
    row: dict[str, Any], arm: str, v2_card: dict[str, Any], device: str,
    representative_raw: dict[str, Any],
):
    item = row[f"{arm}_checkpoint"]
    path = Path(item["path"])
    checkpoint_bytes = read_verified_bytes(path, item["sha256"], f"{arm} checkpoint")
    checkpoint = load_torch_payload(checkpoint_bytes, map_location=device)
    cfg = Config.from_dict(checkpoint["config"])
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    load_model_state_dict_compat(model, checkpoint["model_state_dict"])
    model = model.to(device).eval()
    task_id = int(row["task_id"]) + (0 if arm == "sparse" else 10)
    spec = TrainedRun(
        task_id=task_id,
        arm=arm,
        seed=int(row["model_seed"]),
        system_key="gated_local_linear",
        run_dir=path.parent,
        attempt_count=1,
        incomplete_attempt_count=0,
    )
    audit = audit_authenticated_checkpoint(
        cfg, model, checkpoint, v2_card, spec, representative_raw
    )
    return cfg, env, model, {
        "checkpoint_path": str(path),
        "checkpoint_sha256": item["sha256"],
        "v2_exact_checkpoint_audit_passed": True,
        "checkpoint_step": int(audit["checkpoint_step"]),
        "trainable_parameter_counts": trainable_parameter_counts(model),
    }


def _load_data_manifest(
    output_root: Path, card: dict[str, Any], freeze: dict[str, Any],
) -> dict[str, Any]:
    path = output_root / "outcome_free_data" / "manifest.json"
    manifest = load_json(path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != card["protocol_id"]
        or manifest.get("artifact_role") != "outcome_free_physical_trajectory_manifest"
        or manifest.get("freeze") != freeze
        or len(manifest.get("rows", [])) != 6
    ):
        raise RuntimeError("Outcome-free data manifest does not match the protocol")
    corpora = card["outcome_free_trajectory_corpora"]
    expected = [
        (
            "route_fit", None, int(corpora["route_fit"]["seed"]),
            int(corpora["route_fit"]["trajectory_count"]),
            int(corpora["route_fit"]["horizon_steps"]), "route_fit.pt",
        ),
        (
            "route_audit", None, int(corpora["route_audit"]["seed"]),
            int(corpora["route_audit"]["trajectory_count"]),
            int(corpora["route_audit"]["horizon_steps"]), "route_audit.pt",
        ),
    ]
    expected.extend(
        (
            "evaluation", index, int(seed),
            int(corpora["evaluation"]["trajectory_count_each"]),
            int(corpora["evaluation"]["horizon_steps"]), f"evaluation_{index}.pt",
        )
        for index, seed in enumerate(corpora["evaluation"]["seeds"])
    )
    smoke = corpora["smoke_evaluation"]
    expected.append(
        (
            "smoke_evaluation", int(smoke["dataset_index"]), int(smoke["seed"]),
            int(smoke["trajectory_count"]), int(smoke["horizon_steps"]),
            "smoke_evaluation_0.pt",
        )
    )
    for row, (role, index, seed, count, horizon, filename) in zip(
        manifest["rows"], expected
    ):
        expected_path = output_root / "outcome_free_data" / filename
        expected_row_keys = {"role", "seed", "path", "sha256"}
        if index is not None:
            expected_row_keys.add("dataset_index")
        if (
            set(row) != expected_row_keys
            or row["role"] != role
            or int(row["seed"]) != seed
            or (index is not None and int(row["dataset_index"]) != index)
            or Path(row["path"]) != expected_path
        ):
            raise RuntimeError(f"Trajectory manifest row semantics drifted: {row}")
        artifact_bytes = read_verified_bytes(
            expected_path, row["sha256"], f"{role} trajectory corpus"
        )
        payload = load_torch_payload(artifact_bytes, map_location="cpu")
        if set(payload) != {"trajectories", "metadata"}:
            raise RuntimeError(f"Unexpected top-level trajectory keys: {expected_path}")
        trajectories = payload["trajectories"]
        metadata = payload["metadata"]
        expected_metadata_keys = {
            "protocol_id", "role", "seed", "shape",
            "contains_forecast_or_representation_outcomes",
        }
        if index is not None:
            expected_metadata_keys.add("dataset_index")
        if (
            set(metadata) != expected_metadata_keys
            or metadata["protocol_id"] != card["protocol_id"]
            or metadata["role"] != role
            or int(metadata["seed"]) != seed
            or (index is not None and int(metadata["dataset_index"]) != index)
            or metadata["contains_forecast_or_representation_outcomes"] is not False
            or list(metadata["shape"]) != [count, horizon + 1, 2]
            or tuple(trajectories.shape) != (count, horizon + 1, 2)
            or trajectories.dtype != torch.float32
            or not torch.isfinite(trajectories).all()
        ):
            raise RuntimeError(f"Trajectory payload semantics drifted: {expected_path}")
        manifest.setdefault("_authenticated_trajectories", {})[(role, index)] = (
            trajectories.contiguous()
        )
    return manifest


def _load_corpus(manifest: dict[str, Any], role: str, index: int | None = None) -> torch.Tensor:
    rows = [row for row in manifest["rows"] if row["role"] == role]
    if index is not None:
        rows = [row for row in rows if int(row.get("dataset_index", -1)) == index]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one corpus for {role}/{index}, found {len(rows)}")
    trajectories = manifest["_authenticated_trajectories"].get((role, index))
    if trajectories is None:
        raise RuntimeError(f"Authenticated corpus snapshot is missing: {role}/{index}")
    if trajectories.dtype != torch.float32 or not torch.isfinite(trajectories).all():
        raise RuntimeError(f"Non-finite or wrong-dtype corpus: {role}/{index}")
    return trajectories.contiguous()


def _evaluate(
    *,
    mode: str,
    task_index: int,
    card: dict[str, Any],
    tasks: dict[str, Any],
    freeze: dict[str, Any],
    output_root: Path,
    compute_window_path: Path,
) -> dict[str, Any]:
    v2_bundle = authenticate_v2_inputs(card)
    authenticate_checkpoint_roster(tasks)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Exactly one visible GPU is required, found {torch.cuda.device_count()}")
    gpu = torch.cuda.get_device_properties(0)
    minimum_a100_capacity_bytes = 75 * 1024**3
    if "A100" not in gpu.name or int(gpu.total_memory) < minimum_a100_capacity_bytes:
        raise RuntimeError(
            "The frozen evaluator requires one A100-class GPU with at least 75 GiB; "
            f"observed {gpu.name!r} with {int(gpu.total_memory)} bytes"
        )
    device = "cuda"
    task = task_by_index(tasks, task_index)
    audit_card = v2_bundle["card"]
    representative_raw = v2_bundle["representative_config"]
    sparse_cfg, sparse_env, sparse_model, sparse_provenance = _load_model(
        task, "sparse", audit_card, device, representative_raw
    )
    dense_cfg, dense_env, dense_model, dense_provenance = _load_model(
        task, "dense", audit_card, device, representative_raw
    )
    parameter_contract = tasks.get("provenance_contract", {})
    if (
        sparse_provenance["trainable_parameter_counts"]
        != parameter_contract.get("sparse_trainable_parameter_counts")
        or dense_provenance["trainable_parameter_counts"]
        != parameter_contract.get("dense_trainable_parameter_counts")
    ):
        raise RuntimeError("Paired model trainable-parameter provenance drifted")
    if (
        str(sparse_cfg.ENV.ENV_NAME) != str(dense_cfg.ENV.ENV_NAME)
        or float(sparse_env.unwrapped.dt) != float(dense_env.unwrapped.dt)
    ):
        raise RuntimeError("Paired sparse/dense environment mismatch")
    manifest = _load_data_manifest(output_root, card, freeze)
    fit = _load_corpus(manifest, "route_fit")
    audit = _load_corpus(manifest, "route_audit")
    representatives, fit_diagnostics, fit_latent = fit_codebook(sparse_model, fit, card)
    audit_diagnostics = audit_routing(
        sparse_model, representatives, audit, sparse_env, card
    )
    null_banks, null_diagnostics = matched_null_projectors(
        sparse_model, representatives, fit_latent, card
    )

    if mode == "smoke":
        dataset_indices = [
            int(card["outcome_free_trajectory_corpora"]["smoke_evaluation"]["dataset_index"])
        ]
        role = "smoke_evaluation"
    else:
        dataset_indices = list(
            range(len(card["outcome_free_trajectory_corpora"]["evaluation"]["seeds"]))
        )
        role = "evaluation"
    torch.cuda.synchronize()
    started = time.time()
    atomic_json(
        compute_window_path,
        {
            "schema_version": 1,
            "protocol_id": card["protocol_id"],
            "artifact_role": "forecast_compute_window",
            "mode": mode,
            "task_id": task_index,
            "start_epoch_seconds": started,
            "end_epoch_seconds": None,
        },
    )
    dataset_rows = []
    smoke_all_finite = True
    observed_method_count: int | None = None
    for dataset_index in dataset_indices:
        truth = _load_corpus(manifest, role, dataset_index)
        sparse, sparse_hidden = sparse_forecasts(
            sparse_model, representatives, null_banks, truth, card
        )
        dense, dense_hidden = dense_forecasts(dense_model, truth, card)
        all_hidden = {**sparse_hidden, **dense_hidden}
        all_methods = {**sparse["methods"], **dense["methods"]}
        observed_method_count = len(all_methods)
        if observed_method_count != 41:
            raise RuntimeError(f"Expected exactly 41 forecast methods, found {observed_method_count}")
        all_finite = all(
            row["finite_through_h200_for_every_trajectory"]
            and row.get("finite_through_h500_for_every_trajectory", True)
            for name, row in all_methods.items()
            if not name.endswith("pure_k")
        )
        smoke_all_finite = smoke_all_finite and all_finite
        if mode == "scientific":
            dataset_rows.append(
                {
                    "dataset_index": dataset_index,
                    "dataset_seed": int(
                        card["outcome_free_trajectory_corpora"]["evaluation"]["seeds"][dataset_index]
                    ),
                    "trajectory_sha256": sha256_array(truth.numpy()),
                    "trajectory_count": int(truth.shape[0]),
                    "sparse": sparse,
                    "dense": dense,
                    "evaluation_only_basin_stratification": stratify_after_forecasting(
                        sparse_env, truth, all_hidden
                    ),
                }
            )
    torch.cuda.synchronize()
    ended = time.time()
    atomic_json(
        compute_window_path,
        {
            "schema_version": 1,
            "protocol_id": card["protocol_id"],
            "artifact_role": "forecast_compute_window",
            "mode": mode,
            "task_id": task_index,
            "start_epoch_seconds": started,
            "end_epoch_seconds": ended,
            "elapsed_seconds": ended - started,
        },
    )
    if mode == "smoke":
        return {
            "schema_version": 1,
            "protocol_id": card["protocol_id"],
            "artifact_role": "outcome_blind_gpu_smoke",
            "task_id": task_index,
            "all_required_predictions_finite": smoke_all_finite,
            "exact_method_count": observed_method_count,
            "route_fit_completed": True,
            "route_audit_completed": True,
            "null_scale_matching_completed": True,
            "forecast_metrics_labels_and_alignment_values_persisted": False,
            "outcomes_inspected": False,
            "elapsed_seconds": ended - started,
            "freeze": freeze,
        }
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": "quarantined_scientific_seed_shard",
        "task_id": task_index,
        "model_seed": int(task["model_seed"]),
        "freeze": freeze,
        "provenance": {
            "sparse": sparse_provenance,
            "dense": dense_provenance,
            "data_manifest_path": str(output_root / "outcome_free_data" / "manifest.json"),
            "data_manifest_sha256": sha256_path(
                output_root / "outcome_free_data" / "manifest.json"
            ),
            "evaluator_sha256": sha256_path(Path(__file__)),
            "git_commit": os.environ.get("SKAE_GIT_COMMIT", "launcher_not_recorded"),
            "gpu": {
                "name": gpu.name,
                "total_memory_bytes": int(gpu.total_memory),
            },
        },
        "predictor_assertions": {
            "one_unchanged_global_k_per_checkpoint": True,
            "no_latent_dynamics_or_local_operator_fit": True,
            "support_family_fit_and_assignment_use_no_labels_or_basin_count": True,
            "every_predicted_physical_state_is_reencoded_at_every_step": True,
            "no_teacher_forcing_truth_reset_or_periodic_refresh": True,
            "support_routed_predictor_is_autonomous_nonlinear_not_pure_k_power": True,
        },
        "label_free_family_fit": fit_diagnostics,
        "held_out_route_audit": audit_diagnostics,
        "matched_coordinate_null": null_diagnostics,
        "dataset_rows": dataset_rows,
        "compute_elapsed_seconds": ended - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "scientific"), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--expected-task-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compute-window", type=Path, required=True)
    args = parser.parse_args()
    card, tasks, freeze = load_frozen_protocol(
        card_path=args.card,
        task_path=args.tasks,
        source_manifest_path=args.sources,
        expected_card_sha256=args.expected_card_sha256,
        expected_task_sha256=args.expected_task_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
    )
    payload = _evaluate(
        mode=args.mode,
        task_index=args.task_index,
        card=card,
        tasks=tasks,
        freeze=freeze,
        output_root=args.output_root,
        compute_window_path=args.compute_window,
    )
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "mode": args.mode,
                "task_id": args.task_index,
                "outcomes_quarantined": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
