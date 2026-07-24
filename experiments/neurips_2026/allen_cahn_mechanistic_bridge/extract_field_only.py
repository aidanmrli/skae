"""Field-only feature, family, and forecast extraction for one crossed cell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.conditional_guard import (
    load_and_validate,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.families import jaccard_rows
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.integrity import (
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import (
    CARD_PATH,
    load_card,
    load_dataset_manifest,
    load_fields_only,
    sha256_path,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.rollouts import (
    initial_projection_controls,
    rollout_full,
    rollout_projected_modes,
    rollout_support_contrast,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.wrong_supports import (
    build_wrong_support_codebook,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.evaluation_helpers import (
    encode_states,
    load_profile_decision,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.io import (
    checkpoint_roster,
    load_card as load_mechanism_card,
    load_model,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.metrics import (
    assign_codebook,
    fit_codebook,
    matched_topk_masks,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")
MECHANISM_CARD = Path(
    "experiments/neurips_2026/allen_cahn_support_subspaces/prediction_card.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--dataset_manifest", type=Path, required=True)
    parser.add_argument("--expected_dataset_manifest_sha256", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected_decision_sha256", required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--profile_decision", type=Path, required=True)
    parser.add_argument("--expected_profile_decision_sha256", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--encode_batch_size", type=int, default=4096)
    parser.add_argument("--ready_file", type=Path, required=True)
    parser.add_argument("--release_file", type=Path, required=True)
    parser.add_argument("--start_file", type=Path, required=True)
    parser.add_argument("--done_file", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    return parser.parse_args()


def _family_bundle(
    train_masks: np.ndarray, new_masks: np.ndarray, card: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    cfg = card["support"]
    codebook = fit_codebook(
        train_masks,
        min_jaccard=float(cfg["family_jaccard"]),
        max_representatives=int(cfg["max_representatives"]),
        min_fit_count=int(cfg["min_train_count"]),
    )
    assignments, similarities = assign_codebook(
        new_masks,
        codebook.representatives,
        min_jaccard=float(cfg["family_jaccard"]),
    )
    train_assignments, train_similarities = assign_codebook(
        train_masks,
        codebook.representatives,
        min_jaccard=float(cfg["family_jaccard"]),
    )
    return (
        {
            "representatives": torch.from_numpy(codebook.representatives),
            "fit_counts": torch.from_numpy(codebook.fit_counts),
            "train_assignments": torch.from_numpy(train_assignments),
            "train_similarities": torch.from_numpy(train_similarities),
            "new_assignments": torch.from_numpy(assignments),
            "new_similarities": torch.from_numpy(similarities),
            "new_coverage": float(np.mean(assignments >= 0)),
        },
        assignments,
        codebook.representatives,
    )


def _route_masks(assignments: np.ndarray, representatives: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    covered = assignments >= 0
    if not np.any(covered):
        return torch.empty((0, representatives.shape[1]), dtype=torch.bool), covered
    return torch.from_numpy(representatives[assignments[covered]]), covered


def _support_stability(
    models: dict[str, torch.nn.Module],
    fields: torch.Tensor,
    initial_masks: dict[str, np.ndarray],
    codebooks: dict[str, dict[str, Any]],
    *,
    times: list[int],
    threshold: float,
    encode_batch_size: int,
    card: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    sparse_cardinality = initial_masks["sparse"].sum(1)
    for time_index in times:
        if time_index == 0:
            sparse = initial_masks["sparse"]
            dense = initial_masks["dense"]
        else:
            sparse_z = encode_states(
                models["sparse"], fields[:, time_index], batch_size=encode_batch_size
            ).numpy()
            dense_z = encode_states(
                models["dense"], fields[:, time_index], batch_size=encode_batch_size
            ).numpy()
            sparse = np.abs(sparse_z) > threshold
            dense = matched_topk_masks(dense_z, sparse)
        record: dict[str, Any] = {}
        for arm, masks in (("sparse", sparse), ("dense", dense)):
            representatives = codebooks[arm]["representatives"].numpy()
            assignments, similarities = assign_codebook(
                masks,
                representatives,
                min_jaccard=float(card["support"]["family_jaccard"]),
            )
            record[arm] = {
                "jaccard_to_x0": torch.from_numpy(jaccard_rows(initial_masks[arm], masks)),
                "family_assignments": torch.from_numpy(assignments),
                "family_similarities": torch.from_numpy(similarities),
                "active_cardinality": torch.from_numpy(masks.sum(1).astype(np.int64)),
            }
        if not np.array_equal(initial_masks["dense"].sum(1), sparse_cardinality):
            raise AssertionError("Initial dense top-k cardinality lost pairing")
        result[str(time_index)] = record
    return result


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for field-only extraction")
    if args.batch_size not in (128, 256):
        raise ValueError("Batch size must come from the frozen profile roster")
    torch.set_float32_matmul_precision("high")
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Bridge card differs from launcher root")
    source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Bridge source manifest differs from launcher root")
    _, decision_hash, _ = load_and_validate(
        args.decision,
        expected_sha256=args.expected_decision_sha256,
        card=card,
    )
    if sha256_path(args.dataset_manifest) != args.expected_dataset_manifest_sha256:
        raise RuntimeError("Dataset manifest differs from launcher root")
    dataset_manifest = load_dataset_manifest(args.dataset_manifest, card)
    model_seeds = [int(value) for value in card["roster"]["model_seeds"]]
    dataset_seeds = [int(value) for value in card["new_datasets"]["seeds"]]
    task_count = len(model_seeds) * len(dataset_seeds)
    if args.task_index < 0 or args.task_index >= task_count:
        raise ValueError("Extraction task is outside frozen crossed roster")
    model_seed = model_seeds[args.task_index // len(dataset_seeds)]
    dataset_index = args.task_index % len(dataset_seeds)
    dataset_seed = dataset_seeds[dataset_index]
    output = args.output_root / "field_artifacts" / f"model_{model_seed}_data_{dataset_seed}.pt"
    sidecar = output.with_suffix(".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(output)

    mechanism_card, mechanism_hash = load_mechanism_card(MECHANISM_CARD)
    if mechanism_hash != card["conditional_launch"]["required_card_sha256"]:
        raise RuntimeError("Mechanism model card drifted")
    profile, profile_hash = load_profile_decision(
        args.profile_decision,
        args.batch_size,
        card=mechanism_card,
        card_hash=mechanism_hash,
        source_manifest_hash=card["conditional_launch"]["required_source_manifest_sha256"],
    )
    if (
        profile_hash != args.expected_profile_decision_sha256
        or profile_hash != card["hardware"]["required_profile_decision_sha256"]
        or Path(card["hardware"]["profile_decision"]) != args.profile_decision
    ):
        raise RuntimeError("Hardware profile decision differs from launcher root")
    roster = checkpoint_roster(mechanism_card)
    models = {
        arm: load_model(roster[(arm, model_seed)], mechanism_card, "cuda")[0]
        for arm in ("sparse", "dense")
    }
    training = card["inputs"]["training_dataset"]
    train_fields = load_fields_only(
        Path(training["path"]),
        split=str(training["field_only_split"]),
        card=card,
        expected_sha256=str(training["sha256"]),
        expected_count=int(training["expected_trajectories"]),
        expected_horizon=200,
    )
    dataset_record = dataset_manifest["datasets"][dataset_index]
    new_fields = load_fields_only(
        Path(dataset_record["path"]),
        split="val",
        card=card,
        expected_sha256=str(dataset_record["sha256"]),
        expected_count=256,
        expected_horizon=400,
    )
    train_z = {
        arm: encode_states(models[arm], train_fields[:, 0], batch_size=args.encode_batch_size).numpy()
        for arm in ("sparse", "dense")
    }
    new_z = {
        arm: encode_states(models[arm], new_fields[:, 0], batch_size=args.encode_batch_size).numpy()
        for arm in ("sparse", "dense")
    }
    threshold = float(card["support"]["threshold"])
    train_masks = {"sparse": np.abs(train_z["sparse"]) > threshold}
    new_masks = {"sparse": np.abs(new_z["sparse"]) > threshold}
    train_masks["dense"] = matched_topk_masks(train_z["dense"], train_masks["sparse"])
    new_masks["dense"] = matched_topk_masks(new_z["dense"], new_masks["sparse"])
    codebooks, assignments, representatives = {}, {}, {}
    for arm in ("sparse", "dense"):
        codebooks[arm], assignments[arm], representatives[arm] = _family_bundle(
            train_masks[arm], new_masks[arm], card
        )
    sparse_wrong = build_wrong_support_codebook(
        train_masks["sparse"],
        codebooks["sparse"]["train_assignments"].numpy(),
        representatives["sparse"],
        codebooks["sparse"]["fit_counts"].numpy(),
    )
    codebooks["sparse"]["wrong_support_control"] = sparse_wrong
    features = {
        "train": {
            "raw_x0": train_fields[:, 0],
            "sparse_support": torch.from_numpy(train_masks["sparse"]),
            "sparse_values": torch.from_numpy(train_z["sparse"]),
            "dense_values": torch.from_numpy(train_z["dense"]),
            "dense_topk": torch.from_numpy(train_masks["dense"]),
        },
        "new": {
            "raw_x0": new_fields[:, 0],
            "sparse_support": torch.from_numpy(new_masks["sparse"]),
            "sparse_values": torch.from_numpy(new_z["sparse"]),
            "dense_values": torch.from_numpy(new_z["dense"]),
            "dense_topk": torch.from_numpy(new_masks["dense"]),
        },
    }
    route_lock = {
        "locked": True,
        "fit_times": [0],
        "score_times": [0],
        "future_states_used": False,
        "dataset_payloads_deserialized": True,
        "future_field_tensors_deserialized_before_route_lock": True,
        "future_values_used_for_routing": False,
        "x0_probe_features_materialized_before_future_encoding": True,
        "label_tensors_may_have_been_deserialized": True,
        "label_keys_accessed": False,
        "label_values_used": False,
    }

    stability = _support_stability(
        models,
        new_fields,
        new_masks,
        codebooks,
        times=[int(value) for value in card["support"]["stability_times"]],
        threshold=threshold,
        encode_batch_size=args.encode_batch_size,
        card=card,
    )
    marker_paths = (
        args.ready_file, args.release_file, args.start_file, args.done_file,
    )
    if any(path.exists() for path in marker_paths):
        raise FileExistsError("Telemetry handshake file already exists")
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.write_text("ready\n")
    deadline = time.monotonic() + 60.0
    while not args.release_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for forecast telemetry")
        time.sleep(0.05)
    torch.cuda.synchronize()
    write_json_once(args.start_file, {
        "event": "gpu_compute_start",
        "unix_time": time.time(),
        "model_seed": model_seed,
        "dataset_seed": dataset_seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
    })
    horizons = [160, 200, 400]
    ordinary = {
        arm: rollout_full(models[arm], new_fields, horizons=horizons, batch_size=args.batch_size)
        for arm in ("sparse", "dense")
    }
    correct_masks, covered = _route_masks(assignments["sparse"], representatives["sparse"])
    covered_tensor = torch.from_numpy(covered)
    routed = {
        "trajectory_indices": torch.from_numpy(np.flatnonzero(covered)),
        "correct": (
            rollout_projected_modes(
                models["sparse"], new_fields[covered_tensor], correct_masks,
                horizons=horizons, batch_size=args.batch_size,
            )
            if np.any(covered) else {}
        ),
    }
    valid_families = sparse_wrong["valid_and_distinct"].numpy()
    comparable = covered.copy()
    comparable[covered] &= valid_families[assignments["sparse"][covered]]
    if np.any(comparable):
        comparable_tensor = torch.from_numpy(comparable)
        family_indices = assignments["sparse"][comparable]
        correct_control = torch.from_numpy(representatives["sparse"][family_indices])
        wrong_control = sparse_wrong["representatives"][family_indices]
        if not torch.equal(correct_control.sum(1), wrong_control.sum(1)):
            raise AssertionError("Wrong-support control lost exact paired cardinality")
        if bool(torch.any(torch.all(correct_control == wrong_control, dim=1))):
            raise AssertionError("Wrong-support control is not distinct")
        contrast = rollout_support_contrast(
            models["sparse"], new_fields[comparable_tensor],
            correct_control, wrong_control,
            horizons=horizons, batch_size=args.batch_size,
        )
        routed["wrong_control"] = {
            "trajectory_indices": torch.from_numpy(np.flatnonzero(comparable)),
            "correct": {
                "mask_once": contrast["correct_mask_once"],
                "restricted": contrast["correct_restricted"],
            },
            "wrong": {
                "mask_once": contrast["wrong_mask_once"],
                "restricted": contrast["wrong_restricted"],
            },
            "initial_projection": initial_projection_controls(
                models["sparse"], new_fields[comparable_tensor],
                correct_control, wrong_control, batch_size=args.batch_size,
            ),
            "same_subset_for_all_modes": True,
            "paired_cardinality_exact": True,
            "train_frozen_before_new_trajectory_routing": True,
        }
    else:
        routed["wrong_control"] = {
            "trajectory_indices": torch.empty(0, dtype=torch.int64),
            "same_subset_for_all_modes": True,
            "paired_cardinality_exact": True,
            "train_frozen_before_new_trajectory_routing": True,
        }
    torch.cuda.synchronize()
    write_json_once(args.done_file, {
        "event": "gpu_compute_done",
        "unix_time": time.time(),
        "model_seed": model_seed,
        "dataset_seed": dataset_seed,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
    })

    telemetry_scope = {
        "evaluator_owned_start_marker": str(args.start_file),
        "evaluator_owned_done_marker": str(args.done_file),
        "start_marker_sha256": sha256_path(args.start_file),
        "done_marker_sha256": sha256_path(args.done_file),
        "preload_and_serialization_excluded": True,
    }

    artifact = {
        "schema_version": 1,
        "status": "field_only_complete",
        "model_seed": model_seed,
        "dataset_seed": dataset_seed,
        "task_index": int(args.task_index),
        "features": features,
        "codebooks": codebooks,
        "support_stability": stability,
        "ordinary_forecast": ordinary,
        "routed_forecast": routed,
        "truth_fields": {str(time): new_fields[:, time] for time in (0, 160, 200, 400)},
        "route_lock": route_lock,
        "dataset_sha256": dataset_record["sha256"],
        "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
        "mechanism_decision_sha256": decision_hash,
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "profile_decision_sha256": profile_hash,
        "profile_selected_batch_size": int(profile["selected_batch_size"]),
        "gpu_telemetry_scope": telemetry_scope,
        "checkpoint_sha256": {
            arm: roster[(arm, model_seed)].sha256 for arm in ("sparse", "dense")
        },
        "requested_dataset_keys": ["fields", "split_indices"],
        "dataset_payloads_deserialized": True,
        "label_tensors_may_have_been_deserialized": True,
        "label_keys_accessed": False,
        "label_values_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    record = {
        "status": "field_only_complete",
        "model_seed": model_seed,
        "dataset_seed": dataset_seed,
        "artifact": str(output),
        "artifact_sha256": sha256_path(output),
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "profile_decision_sha256": profile_hash,
        "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
        "mechanism_decision_sha256": decision_hash,
        "requested_dataset_keys": ["fields", "split_indices"],
        "future_encoding_after_route_lock": True,
        "x0_probe_features_materialized_before_future_encoding": True,
        "gpu_telemetry_scope": telemetry_scope,
        "dataset_payloads_deserialized": True,
        "label_tensors_may_have_been_deserialized": True,
        "label_keys_accessed": False,
        "label_values_used": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
    }
    write_json_once(sidecar, record)
    print(json.dumps({"status": record["status"], "artifact_sha256": record["artifact_sha256"]}))


if __name__ == "__main__":
    main()
