"""Evaluate fixed initial-support restrictions of Allen--Cahn global Koopman matrices."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.neurips_2026.allen_cahn_support_subspaces.io import (
    CARD_PATH,
    checkpoint_roster,
    load_card,
    load_fields,
    load_model,
    load_reference_forecasts,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.evaluation_helpers import (
    closure_bundle,
    encode_states,
    family_summary,
    forecast_kernel_discrepancy,
    forecast_ratios,
    historical_forecast_reproduction_metrics,
    initial_projection_diagnostics,
    load_profile_decision,
    masks_for_labels,
    matrix_for_row_vectors,
    verify_forecast_reproduction,
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.metrics import (
    closure_metrics,
    decoded_rollout_metrics,
    matched_topk_masks,
    matrix_leakage_metrics,
    operator_distance,
    operator_signature_distance,
    ordinary_permutations,
    restricted_operator_signature,
    summarize_null,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--profile_decision", type=Path, required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--expected_profile_decision_sha256", required=True)
    parser.add_argument("--ready_file", type=Path, required=True)
    parser.add_argument("--release_file", type=Path, required=True)
    parser.add_argument("--gpu_start_file", type=Path, required=True)
    parser.add_argument("--gpu_done_file", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--encode_batch_size", type=int, default=4096)
    parser.add_argument("--closure_state_batch_size", type=int, default=8192)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Card differs from launcher root of trust")
    declared_dataset_paths = [
        str(card["inputs"][name]["path"])
        for name in ("training_dataset", "score_dataset")
    ]
    if any("20260725" in path for path in declared_dataset_paths):
        raise AssertionError("The forbidden conditional holdout was referenced")
    source_manifest_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_manifest_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Source manifest differs from launcher root of trust")
    profile, profile_hash = load_profile_decision(
        args.profile_decision, args.batch_size, card=card, card_hash=card_hash,
        source_manifest_hash=source_manifest_hash,
    )
    if profile_hash != args.expected_profile_decision_sha256:
        raise RuntimeError("Profile decision differs from launcher root of trust")
    seeds = [int(value) for value in card["roster"]["model_seeds"]]
    if args.task_index < 0 or args.task_index >= len(seeds):
        raise ValueError("task_index outside frozen roster")
    seed = seeds[args.task_index]
    shard = args.output_root / "shards" / f"seed_{seed}.json"
    lineage = args.output_root / "lineage" / f"seed_{seed}.json"
    if shard.exists() or lineage.exists():
        raise FileExistsError(f"Shard or lineage receipt already exists for seed {seed}")
    started = time.time()
    roster = checkpoint_roster(card)
    reference_forecasts = load_reference_forecasts(card)
    models, checkpoints = {}, {}
    for arm in ("sparse", "dense"):
        models[arm], checkpoints[arm] = load_model(roster[(arm, seed)], card, args.device)
    train_fields = load_fields(card, "training_dataset")
    score_fields = load_fields(card, "score_dataset")
    horizons = [int(value) for value in card["roster"]["horizons"]]
    if score_fields.shape[1] != max(horizons) + 1:
        raise AssertionError("The score set must contain exactly H200 plus its initial state")
    handshake_paths = (
        args.ready_file, args.release_file, args.gpu_start_file, args.gpu_done_file
    )
    if any(path.exists() for path in handshake_paths):
        raise FileExistsError("Scientific telemetry handshake file already exists")
    args.ready_file.write_text("ready\n")
    deadline = time.monotonic() + 60.0
    while not args.release_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for scientific telemetry start signal")
        time.sleep(0.05)
    torch.cuda.synchronize()
    args.gpu_start_file.write_text(json.dumps({
        "event": "gpu_compute_start",
        "unix_time": time.time(),
        "seed": seed,
    }, sort_keys=True) + "\n")

    # BEGIN FROZEN V3 SCIENTIFIC COMPUTATION (v4 changes are outside this block).
    # Provenance-only fail-fast guard: exactly replay the historical evaluator
    # before encoding any support or computing any mechanism outcome.
    reproduction_batch = int(
        card["inputs"]["ordinary_forecast_seed_rows"][
            "historical_reproduction_batch_size"
        ]
    )
    reproduction_horizons = [
        int(value) for value in card["inputs"]["ordinary_forecast_seed_rows"][
            "historical_evaluator_horizon_sequence"
        ]
    ]
    historical_forecasts = {
        arm: historical_forecast_reproduction_metrics(
            models[arm], score_fields, horizons=reproduction_horizons,
            batch_size=reproduction_batch,
        )
        for arm in ("sparse", "dense")
    }
    forecast_reproduction = verify_forecast_reproduction(
        historical_forecasts, reference_forecasts, seed=seed, card=card
    )

    train_z0 = {
        arm: encode_states(models[arm], train_fields[:, 0], batch_size=args.encode_batch_size).numpy()
        for arm in ("sparse", "dense")
    }
    score_z0 = {
        arm: encode_states(models[arm], score_fields[:, 0], batch_size=args.encode_batch_size)
        for arm in ("sparse", "dense")
    }
    threshold = float(card["support"]["primary_threshold"])
    sparse_train_masks = np.abs(train_z0["sparse"]) > threshold
    sparse_score_masks = np.abs(score_z0["sparse"].numpy()) > threshold
    dense_train_masks = matched_topk_masks(train_z0["dense"], sparse_train_masks)
    dense_score_masks = matched_topk_masks(score_z0["dense"].numpy(), sparse_score_masks)
    family, labels, representatives, qualified_families = {}, {}, {}, {}
    for arm, fit_masks, scored_masks in (
        ("sparse", sparse_train_masks, sparse_score_masks),
        ("dense", dense_train_masks, dense_score_masks),
    ):
        (family[arm], labels[arm], representatives[arm], qualified_families[arm]) = family_summary(
            fit_masks, scored_masks, card
        )

    # The only routing objects are now locked from training x0 and score x0.
    # Future score states are encoded below solely as diagnostic outcomes.
    score_latents = {}
    for arm in ("sparse", "dense"):
        future = encode_states(
            models[arm], score_fields[:, 1:], batch_size=args.encode_batch_size
        )
        score_latents[arm] = torch.cat((score_z0[arm][:, None], future), dim=1)

    permutations = ordinary_permutations(
        int(card["roster"]["latent_dim"]),
        int(card["null"]["replicates"]),
        int(card["null"]["seed"]),
    )
    matrices = {arm: matrix_for_row_vectors(models[arm], card) for arm in ("sparse", "dense")}
    closure, forecasts, initial_projection = {}, {}, {}
    for arm, masks in (("sparse", sparse_score_masks), ("dense", dense_score_masks)):
        matrix = matrices[arm]
        closure[arm] = closure_bundle(
            score_latents[arm], masks, matrix, permutations, horizons,
            args.closure_state_batch_size,
        )
        forecasts[arm] = decoded_rollout_metrics(
            models[arm], score_fields, torch.from_numpy(masks),
            horizons=horizons, batch_size=args.batch_size,
        )
        forecasts[arm]["ratios"] = forecast_ratios(forecasts[arm])
        initial_projection[arm] = initial_projection_diagnostics(
            models[arm], score_fields, score_z0[arm], masks, batch_size=args.batch_size
        )
    forecast_kernel_differences = {
        arm: forecast_kernel_discrepancy(forecasts[arm], historical_forecasts[arm])
        for arm in ("sparse", "dense")
    }

    family_masks, covered = masks_for_labels(labels["sparse"], representatives["sparse"])
    sparse_family: dict[str, Any] = {"score_count": int(covered.sum())}
    if int(covered.sum()) > 0 and bool(family["sparse"]["eligible"]):
        sparse_matrix = matrices["sparse"]
        sparse_family["closure"] = closure_bundle(
            score_latents["sparse"][covered], family_masks, sparse_matrix,
            permutations, horizons, args.closure_state_batch_size,
        )
        family_forecast = decoded_rollout_metrics(
            models["sparse"], score_fields[covered], torch.from_numpy(family_masks),
            horizons=horizons, batch_size=args.batch_size,
            family_labels=labels["sparse"][covered],
        )
        family_forecast["ratios"] = forecast_ratios(family_forecast)
        sparse_family["forecast"] = family_forecast
        sparse_family["initial_projection"] = initial_projection_diagnostics(
            models["sparse"], score_fields[covered], score_z0["sparse"][covered],
            family_masks, batch_size=args.batch_size,
        )
        sparse_family["aggregate_includes_all_independently_routed_score_initial_conditions"] = True
        sparse_family["qualified_family_indices"] = np.flatnonzero(
            qualified_families["sparse"]
        ).tolist()
        top_two = [int(value) for value in family["sparse"]["fit_frozen_top_two_family_indices"]]
        derangement_keep = np.isin(labels["sparse"], np.asarray(top_two))
        correct_masks = representatives["sparse"][labels["sparse"][derangement_keep]]
        wrong_labels = np.where(
            labels["sparse"][derangement_keep] == top_two[0], top_two[1], top_two[0]
        )
        wrong_masks = representatives["sparse"][wrong_labels]
        correct_forecast = decoded_rollout_metrics(
            models["sparse"], score_fields[derangement_keep], torch.from_numpy(correct_masks),
            horizons=horizons, batch_size=args.batch_size,
        )
        wrong_forecast = decoded_rollout_metrics(
            models["sparse"], score_fields[derangement_keep], torch.from_numpy(wrong_masks),
            horizons=horizons, batch_size=args.batch_size,
        )
        correct_forecast["ratios"] = forecast_ratios(correct_forecast)
        wrong_forecast["ratios"] = forecast_ratios(wrong_forecast)
        sparse_family["top_two_family_derangement"] = {
            "selection_rule": "two largest training-fit counts, lower codebook index tie-break",
            "family_indices": top_two,
            "trajectory_count": int(derangement_keep.sum()),
            "correct": correct_forecast,
            "wrong_swap": wrong_forecast,
            "correct_initial_projection": initial_projection_diagnostics(
                models["sparse"], score_fields[derangement_keep],
                score_z0["sparse"][derangement_keep], correct_masks,
                batch_size=args.batch_size,
            ),
            "wrong_initial_projection": initial_projection_diagnostics(
                models["sparse"], score_fields[derangement_keep],
                score_z0["sparse"][derangement_keep], wrong_masks,
                batch_size=args.batch_size,
            ),
        }
        sparse_family["qualified_family_matrix_closure"] = {}
        for family_index in np.flatnonzero(qualified_families["sparse"]):
            family_mask = torch.from_numpy(
                representatives["sparse"][int(family_index)][None]
            ).to(sparse_matrix.device)
            matrix_null = []
            for permutation in permutations:
                index = torch.as_tensor(permutation, device=sparse_matrix.device)
                matrix_null.append(matrix_leakage_metrics(
                    family_mask.index_select(-1, index), sparse_matrix
                ))
            sparse_family["qualified_family_matrix_closure"][str(int(family_index))] = {
                "score_x0_count": int(family["sparse"]["score_counts"][int(family_index)]),
                "true": matrix_leakage_metrics(family_mask, sparse_matrix),
                "null_median": summarize_null(matrix_null),
            }
        rep_tensor = torch.from_numpy(representatives["sparse"][top_two]).to(
            sparse_matrix.device
        )
        pair_numpy = representatives["sparse"][top_two]
        intersection = int(np.logical_and(pair_numpy[0], pair_numpy[1]).sum())
        union = int(np.logical_or(pair_numpy[0], pair_numpy[1]).sum())
        geometry = {
            "family_indices": top_two,
            "cardinalities": [int(mask.sum()) for mask in pair_numpy],
            "intersection": intersection,
            "union": union,
            "jaccard": float(intersection / max(1, union)),
            "joint_coordinate_permutation_preserves_geometry": True,
        }
        for permutation in permutations:
            permuted = pair_numpy[:, permutation]
            if [int(mask.sum()) for mask in permuted] != geometry["cardinalities"] \
                    or int(np.logical_and(permuted[0], permuted[1]).sum()) != intersection:
                raise AssertionError("Joint coordinate null changed pair support geometry")
        sparse_family["fit_frozen_pair_support_geometry"] = geometry
        observed_signature_distance = operator_signature_distance(rep_tensor, sparse_matrix)
        observed_coordinate_distance = operator_distance(rep_tensor, sparse_matrix)
        null_signature_distances, null_coordinate_distances = [], []
        for permutation in permutations:
            index = torch.as_tensor(permutation, device=sparse_matrix.device)
            permuted = rep_tensor.index_select(-1, index)
            null_signature_distances.append(operator_signature_distance(permuted, sparse_matrix))
            null_coordinate_distances.append(operator_distance(permuted, sparse_matrix))
        signature_null = [float(value) for value in null_signature_distances if value is not None]
        coordinate_null = [float(value) for value in null_coordinate_distances if value is not None]
        signature_null_median = float(np.median(signature_null)) if signature_null else None
        coordinate_null_median = float(np.median(coordinate_null)) if coordinate_null else None
        sparse_family["signature_differentiation"] = {
            "definition": "delta on [trace mean, centered-symmetric RMS, skew RMS]",
            "observed": observed_signature_distance,
            "null_median": signature_null_median,
            "observed_over_null": (
                None if observed_signature_distance is None or not signature_null_median
                else observed_signature_distance / signature_null_median
            ),
            "null_replicates": null_signature_distances,
            "family_signatures": [
                {
                    "family_index": int(family_index),
                    "cardinality": int(rep_tensor[row].sum().item()),
                    "components_mu_s_r": restricted_operator_signature(
                        rep_tensor[row], sparse_matrix
                    ).cpu().tolist(),
                }
                for row, family_index in enumerate(top_two)
            ],
        }
        sparse_family["coordinate_chart_distance_descriptive_only"] = {
            "observed": observed_coordinate_distance,
            "null_median": coordinate_null_median,
            "observed_over_null": (
                None if observed_coordinate_distance is None or not coordinate_null_median
                else observed_coordinate_distance / coordinate_null_median
            ),
            "null_replicates": null_coordinate_distances,
        }

    sensitivities: dict[str, Any] = {}
    for sensitivity in card["support"]["raw_only_sensitivity_thresholds"]:
        sparse_masks = np.abs(score_z0["sparse"].numpy()) > float(sensitivity)
        dense_masks = matched_topk_masks(score_z0["dense"].numpy(), sparse_masks)
        record = {
            arm: closure_metrics(
                score_latents[arm].to(args.device),
                torch.from_numpy(mask).to(args.device),
                matrices[arm],
                horizon=max(horizons),
                state_batch_size=args.closure_state_batch_size,
            )
            for arm, mask in (("sparse", sparse_masks), ("dense", dense_masks))
        }
        record["mask_summary"] = {
            "sparse_active_density": float(sparse_masks.mean()),
            "sparse_cardinality_mean": float(sparse_masks.sum(axis=1).mean()),
            "paired_dense_cardinality_exact": bool(np.array_equal(
                sparse_masks.sum(axis=1), dense_masks.sum(axis=1)
            )),
        }
        sensitivities[str(sensitivity)] = record

    # END FROZEN V3 SCIENTIFIC COMPUTATION.
    torch.cuda.synchronize()
    args.gpu_done_file.write_text(json.dumps({
        "event": "gpu_compute_done",
        "unix_time": time.time(),
        "seed": seed,
    }, sort_keys=True) + "\n")
    telemetry_scope = {
        "evaluator_owned_start_marker": str(args.gpu_start_file),
        "evaluator_owned_done_marker": str(args.gpu_done_file),
        "start_marker_sha256": sha256_path(args.gpu_start_file),
        "done_marker_sha256": sha256_path(args.gpu_done_file),
        "scope": "historical replay through final scientific CUDA synchronization",
        "preload_and_serialization_excluded": True,
    }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "seed": seed,
        "task_index": int(args.task_index),
        "card_sha256": card_hash,
        "source_manifest_sha256": source_manifest_hash,
        "source_manifest_path": str(SOURCE_MANIFEST),
        "profile_decision_sha256": profile_hash,
        "profile_decision_path": str(args.profile_decision),
        "profile_selected_batch_size": int(profile["selected_batch_size"]),
        "gpu_telemetry_scope": telemetry_scope,
        "information_firewall": {
            "requested_dataset_keys": ["fields", "split_indices"],
            "requested_dataset_key_name_firewall_passed": True,
            "support_fit_time_indices": [0],
            "support_score_time_indices": [0],
            "future_encoding_began_only_after_support_lock": True,
            "future_states_used_for_routing": False,
            "periodic_reencoding": False,
        },
        "validity_audits": {
            "score_trajectory_count": int(score_fields.shape[0]),
            "architecture_and_treatment_audit_passed": True,
            "row_operator_orientation_audit_passed": True,
            "full_K_ordinary_forecast_reproduction": forecast_reproduction,
            "historical_reproduction_batch_size": reproduction_batch,
            "historical_evaluator_horizon_sequence": reproduction_horizons,
            "historical_kernel_is_provenance_only": True,
            "scientific_vs_historical_kernel_discrepancy": forecast_kernel_differences,
        },
        "provenance": {
            arm: {
                "checkpoint_path": str(roster[(arm, seed)].path),
                "checkpoint_sha256": roster[(arm, seed)].sha256,
                "checkpoint_step": int(checkpoints[arm].get("step", -1)),
                "launch_git_commit": roster[(arm, seed)].git_commit,
            }
            for arm in ("sparse", "dense")
        },
        "mask_summary": {
            "sparse_initial_active_density": float(sparse_score_masks.mean()),
            "sparse_initial_cardinality_mean": float(sparse_score_masks.sum(axis=1).mean()),
            "dense_mask_terminology": "sparse-cardinality-matched top-k coordinate mask",
            "paired_cardinality_exact": bool(np.array_equal(
                sparse_score_masks.sum(axis=1), dense_score_masks.sum(axis=1)
            )),
        },
        "family": family,
        "initial_projection": initial_projection,
        "closure": closure,
        "forecast": forecasts,
        "sparse_family": sparse_family,
        "raw_threshold_sensitivities_descriptive_only": sensitivities,
        "elapsed_seconds": time.time() - started,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_recorded"),
    }
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lineage_payload = {
        "schema_version": 1,
        "status": "lineage_complete",
        "semantic_scope": "non-mechanism lineage, replay, firewall, and provenance only",
        "mechanism_metric_keys_included": False,
        "seed": seed,
        "task_index": int(args.task_index),
        "scientific_shard": str(shard),
        "scientific_shard_sha256": sha256_path(shard),
        "card_sha256": card_hash,
        "source_manifest_sha256": source_manifest_hash,
        "source_manifest_path": str(SOURCE_MANIFEST),
        "profile_decision_sha256": profile_hash,
        "profile_decision_path": str(args.profile_decision),
        "profile_selected_batch_size": int(profile["selected_batch_size"]),
        "gpu_telemetry_scope": telemetry_scope,
        "historical_reproduction_passed": bool(forecast_reproduction["passed"]),
        "historical_reproduction_batch_size": reproduction_batch,
        "historical_evaluator_horizon_sequence": reproduction_horizons,
        "information_firewall": payload["information_firewall"],
        "score_trajectory_count": int(score_fields.shape[0]),
        "provenance": payload["provenance"],
        "slurm_job_id": payload["slurm_job_id"],
    }
    lineage.parent.mkdir(parents=True, exist_ok=True)
    lineage.write_text(
        json.dumps(lineage_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "completed", "seed": seed, "shard": str(shard)}), flush=True)


if __name__ == "__main__":
    main()
