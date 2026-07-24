#!/usr/bin/env python3
"""Prospective autograd test of support-selected laws in one unchanged global K."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.neurips_2026.global_k_distinct_laws_v2_checkpoint_audit import (
    TrainedRun,
    discover_trained_runs,
    load_trained_model,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_math import (
    EPS,
    antithetic_directions,
    authenticate_local_geometry,
    autograd_jacobian,
    center_forecast_metrics,
    central_difference_jacobian,
    direct_latent_closure,
    finite_radius_sweep,
    law_cost_summary,
    rk4_step_matrix,
    true_update_rms,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_routing import (
    dense_center_projectors,
    discover_sparse_families,
    match_sparse_routing,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_source_lock import (
    verify_source_lock,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    load_card,
    sha256_path,
)
from experiments.neurips_2026.global_k_support_invariance import (
    sign_pair_permutations,
)


def _geometry(
    env, offsets: np.ndarray, card: dict[str, Any],
) -> tuple[np.ndarray, list[float]]:
    matrices = env.unwrapped.basin_matrices.detach().cpu().numpy()
    step = np.stack([rk4_step_matrix(matrix, float(card["benchmark"]["dt"])) for matrix in matrices])
    truth = step - np.eye(2)
    rms = [true_update_rms(offsets, truth[basin]) for basin in range(3)]
    return truth, rms


def _authenticate_mechanism_geometry(
    env, centers: np.ndarray, sparse_routing: dict[str, Any],
    card: dict[str, Any],
) -> dict[str, Any]:
    kink = card["autograd_differentiability_kink_guard"]
    robustness = card["finite_radius_robustness_not_selection"]
    directions = antithetic_directions(
        int(robustness["direction_count"]), int(robustness["direction_seed"])
    )
    kink_points, radius_points = [], []
    for basin in range(3):
        by_kink = []
        for epsilon in kink["coordinate_symmetric_epsilons"]:
            for coordinate in range(centers.shape[1]):
                offset = np.zeros(centers.shape[1], dtype=np.float32)
                offset[coordinate] = float(epsilon)
                by_kink.extend((centers[basin] + offset, centers[basin] - offset))
        kink_points.append(np.asarray(by_kink, dtype=np.float32))
        radius_points.append(np.concatenate([
            centers[basin][None, :] + float(radius) * directions
            for radius in robustness["radii"]
        ]).astype(np.float32))
    threshold = float(card["validity"]["max_analytic_environment_step_disagreement"])
    categories = {
        "calibration": sparse_routing["geometry_authentication"]["calibration"],
        "verification": sparse_routing["geometry_authentication"]["verification"],
        "centers": authenticate_local_geometry(
            env, [center[None, :] for center in centers], category="centers",
            dt=float(card["benchmark"]["dt"]), max_abs_error=threshold,
        ),
        "kink": authenticate_local_geometry(
            env, kink_points, category="kink", dt=float(card["benchmark"]["dt"]),
            max_abs_error=threshold,
        ),
        "finite_radius": authenticate_local_geometry(
            env, radius_points, category="finite_radius",
            dt=float(card["benchmark"]["dt"]), max_abs_error=threshold,
        ),
    }
    expected = card["benchmark"]["geometry_authentication"][
        "expected_point_counts"
    ]
    counts_match = all(
        categories[name]["total_point_count"] == int(expected[name])
        for name in categories
    )
    maxima = [
        item["analytic_rk4_max_abs_error"] for item in categories.values()
        if item["analytic_rk4_max_abs_error"] is not None
    ]
    return {
        "categories": categories,
        "expected_point_counts": expected,
        "observed_total_point_count": sum(
            item["total_point_count"] for item in categories.values()
        ),
        "expected_total_point_count": int(expected["total"]),
        "point_counts_exact": counts_match and sum(
            item["total_point_count"] for item in categories.values()
        ) == int(expected["total"]),
        "maximum_analytic_rk4_abs_error": max(maxima) if maxima else None,
        "all_points_in_intended_regions": all(
            item["all_points_in_intended_region"] for item in categories.values()
        ),
        "passed": bool(
            counts_match
            and sum(item["total_point_count"] for item in categories.values())
            == int(expected["total"])
            and all(item["passed"] for item in categories.values())
        ),
    }


def _null_ratio(observed: float, null_median: float) -> float:
    """Conservatively make a zero-null tie fail a strict improvement gate."""
    return (
        float(observed / null_median)
        if null_median > EPS else float(max(1.0, observed / EPS))
    )


def _numerical_failure(error: BaseException) -> dict[str, Any]:
    return {
        "status": "ineligible_numerical", "law_valid": False,
        "direct_closure_valid": False,
        "failure_reasons": [
            f"numerical_scoring_error:{type(error).__name__}:{error}"
        ],
    }


def score_mechanism(
    model, masks: np.ndarray, centers: np.ndarray, verification_offsets: np.ndarray,
    matched_verification_points: list[np.ndarray], env, card: dict[str, Any], arm: str,
    geometry_authentication: dict[str, Any],
) -> dict[str, Any]:
    truth, rms = _geometry(env, verification_offsets, card)
    h_block = np.stack([autograd_jacobian(model, centers[b], masks[b], "h_block") for b in range(3)])
    g_block = np.stack([autograd_jacobian(model, centers[b], masks[b], "g_block") for b in range(3)])
    g_source = np.stack([autograd_jacobian(model, centers[b], masks[b], "g_source") for b in range(3)])
    h_global = np.stack([autograd_jacobian(model, centers[b], None, "h_global") for b in range(3)])
    g_global = np.stack([autograd_jacobian(model, centers[b], None, "g_global") for b in range(3)])
    finite = bool(all(np.isfinite(value).all() for value in (h_block, g_block, g_source, h_global, g_global)))
    if not finite:
        raise FloatingPointError("nonfinite autograd H/G Jacobian")
    h_block_laws = law_cost_summary(h_block, truth)
    g_block_laws = law_cost_summary(g_block, truth)
    g_source_laws = law_cost_summary(g_source, truth)
    h_global_laws = law_cost_summary(h_global, truth)
    g_global_laws = law_cost_summary(g_global, truth)
    closure = [
        float(np.linalg.norm(g_block[b] - g_source[b], ord="fro") / max(np.linalg.norm(g_source[b], ord="fro"), EPS))
        for b in range(3)
    ]
    kink = card["autograd_differentiability_kink_guard"]
    kink_rows = []
    for basin in range(3):
        values: dict[str, dict[str, float]] = {"H": {}, "G": {}}
        for epsilon in kink["coordinate_symmetric_epsilons"]:
            h_fd = central_difference_jacobian(model, centers[basin], masks[basin], "h_block", float(epsilon))
            g_fd = central_difference_jacobian(model, centers[basin], masks[basin], "g_block", float(epsilon))
            values["H"][str(epsilon)] = float(
                np.linalg.norm(h_fd - h_block[basin], ord="fro")
                / max(np.linalg.norm(truth[basin], ord="fro"), EPS)
            )
            values["G"][str(epsilon)] = float(
                np.linalg.norm(g_fd - g_block[basin], ord="fro")
                / max(np.linalg.norm(truth[basin], ord="fro"), EPS)
            )
        passed = all(
            value <= float(kink["maximum_disagreement_each_epsilon"])
            for estimand in values.values() for value in estimand.values()
        )
        kink_rows.append({"basin": basin, "disagreement_by_estimand_and_epsilon": values, "passed_both_estimands_both_epsilons": passed})
    center_rows = [center_forecast_metrics(model, centers[b], masks[b], rms[b]) for b in range(3)]
    directions = antithetic_directions(
        int(card["finite_radius_robustness_not_selection"]["direction_count"]),
        int(card["finite_radius_robustness_not_selection"]["direction_seed"]),
    )
    radii = [float(value) for value in card["finite_radius_robustness_not_selection"]["radii"]]
    radius_by_basin = [
        {
            "basin": b,
            "H": finite_radius_sweep(model, centers[b], masks[b], h_block[b], truth, radii, directions, b, "h_block"),
            "G": finite_radius_sweep(model, centers[b], masks[b], g_block[b], truth, radii, directions, b, "g_block"),
        }
        for b in range(3)
    ]
    closure_rows = [
        direct_latent_closure(model, matched_verification_points[b], masks[b])
        for b in range(3)
    ]
    closure_denominator_valid = all(
        np.isfinite(list(row.values())).all()
        and row["change_denominator_rms"] > 1e-8
        for row in closure_rows
    )
    observed_closure_max = max(row["change_normalized_leakage"] for row in closure_rows)
    null_h, null_g, null_closure_max = [], [], []
    permutations = sign_pair_permutations(
        masks.shape[-1] // 2,
        int(card["closure_and_coordinate_null"]["coordinate_null_replicates"]),
        int(card["closure_and_coordinate_null"]["coordinate_null_seed"]),
    )
    for permutation in permutations:
        h_null_matrix = np.stack([
            autograd_jacobian(model, centers[b], masks[b], "h_block", permutation)
            for b in range(3)
        ])
        g_null_matrix = np.stack([
            autograd_jacobian(model, centers[b], masks[b], "g_block", permutation)
            for b in range(3)
        ])
        null_h.append(law_cost_summary(h_null_matrix, truth)["identity_over_best_nonidentity"])
        null_g.append(law_cost_summary(g_null_matrix, truth)["identity_over_best_nonidentity"])
        null_closure_max.append(
            max(
                direct_latent_closure(
                    model, matched_verification_points[b], masks[b], permutation
                )["change_normalized_leakage"]
                for b in range(3)
            )
        )
    h_null_median, g_null_median = float(np.median(null_h)), float(np.median(null_g))
    h_null_ratio = _null_ratio(
        h_block_laws["identity_over_best_nonidentity"], h_null_median
    )
    g_null_ratio = _null_ratio(
        g_block_laws["identity_over_best_nonidentity"], g_null_median
    )
    closure_null_median = float(np.median(null_closure_max))
    closure_null_ratio = _null_ratio(observed_closure_max, closure_null_median)
    gates = card["per_seed_sparse_gate"]
    direct_closure_pass = bool(
        closure_denominator_valid
        and observed_closure_max <= float(gates["max_active_code_cloud_change_leakage"])
        and closure_null_ratio <= float(gates["max_active_code_cloud_observed_over_median_null"])
    )
    h_global_pass = bool(
        h_global_laws["max_own_relative_error"] <= float(gates["max_h_global_own_relative_error"])
        and h_global_laws["max_own_over_nearest_wrong"] <= float(gates["max_h_global_row_identification"])
        and h_global_laws["identity_over_best_nonidentity"] <= float(gates["max_h_global_assignment"])
        and h_global_laws["identity_is_unique_optimum"]
    )
    kink_pass = all(row["passed_both_estimands_both_epsilons"] for row in kink_rows)
    h_pass = bool(
        finite and kink_pass and h_global_pass
        and h_block_laws["max_own_relative_error"] <= float(gates["max_h_block_own_relative_error"])
        and h_block_laws["max_own_over_nearest_wrong"] <= float(gates["max_h_block_row_identification"])
        and h_block_laws["identity_over_best_nonidentity"] <= float(gates["max_h_block_assignment"])
        and h_block_laws["identity_is_unique_optimum"]
        and h_null_ratio <= float(gates["max_h_observed_over_median_coordinate_null_assignment"])
    )
    g_pass = bool(
        finite and kink_pass and h_global_pass and direct_closure_pass
        and g_block_laws["max_own_relative_error"] <= float(gates["max_g_block_own_relative_error"])
        and g_block_laws["max_own_over_nearest_wrong"] <= float(gates["max_g_block_row_identification"])
        and g_block_laws["identity_over_best_nonidentity"] <= float(gates["max_g_block_assignment"])
        and g_block_laws["identity_is_unique_optimum"]
        and max(closure) <= float(gates["max_block_source_closure"])
        and g_null_ratio <= float(gates["max_g_observed_over_median_coordinate_null_assignment"])
    )
    payload = {
        "geometry": {
            "true_update_matrices": truth.tolist(),
            "true_update_rms_by_basin": rms,
            "authentication": geometry_authentication,
        },
        "H_block": {"jacobians": h_block.tolist(), "law_identification": h_block_laws},
        "G_block": {"jacobians": g_block.tolist(), "law_identification": g_block_laws},
        "G_source_only": {"jacobians": g_source.tolist(), "law_identification": g_source_laws},
        "H_global": {"jacobians": h_global.tolist(), "law_identification": h_global_laws, "positive_control_pass": h_global_pass},
        "G_global_diagnostic": {"jacobians": g_global.tolist(), "law_identification": g_global_laws},
        "closure": {"block_over_source_discrepancy_by_basin": closure, "maximum": max(closure)},
        "kink_guard": {"rows": kink_rows, "complete_seed_pass": kink_pass},
        "center_forecast_guards": center_rows,
        "finite_radius_robustness": {"directions_are_fixed_antithetic": True, "by_basin": radius_by_basin},
        "coordinate_null": {
            "H": {"assignment_replicates": null_h, "median_assignment": h_null_median, "observed_over_median": h_null_ratio},
            "G": {"assignment_replicates": null_g, "median_assignment": g_null_median, "observed_over_median": g_null_ratio},
        },
        "direct_active_code_cloud_closure": {
            "rows": [
                {"basin": basin, "point_count": int(matched_verification_points[basin].shape[0]), **row}
                for basin, row in enumerate(closure_rows)
            ],
            "denominators_finite_and_above_1e-8": closure_denominator_valid,
            "observed_max_change_normalized_leakage": observed_closure_max,
            "null_max_replicates": null_closure_max,
            "median_null_max": closure_null_median,
            "observed_over_median_null": closure_null_ratio,
            "per_seed_pass": direct_closure_pass,
            "scope": "fixed current-family-assigned active verification code clouds, not whole coordinate subspaces",
        },
        "finite_autograd_jacobians": finite,
        "per_seed_joint_h_g_pass": bool(arm == "sparse" and h_pass and g_pass),
        "per_seed_g_only_pass": bool(arm == "sparse" and g_pass),
        "per_seed_h_only_pass": bool(arm == "sparse" and h_pass),
    }
    try:
        json.dumps(payload, allow_nan=False)
    except ValueError as error:
        raise FloatingPointError("nonfinite downstream mechanism metric") from error
    return payload


def _authenticate_audit(
    spec: TrainedRun, audit_dir: Path, card: dict[str, Any], card_hash: str,
    task_tsv_hash: str,
) -> dict[str, Any]:
    path = audit_dir / "shards" / f"task_{spec.task_id:02d}.json"
    payload = json.loads(path.read_text())
    if payload.get("status") != "passed" or payload.get("protocol_id") != card["protocol_id"]:
        raise RuntimeError(f"Checkpoint audit failed authentication: {path}")
    exact_fields = {
        "card_sha256": card_hash,
        "task_tsv_sha256": task_tsv_hash,
        "task_id": spec.task_id,
        "arm": spec.arm,
        "seed": spec.seed,
        "run_dir": str(spec.run_dir),
        "checkpoint_sha256": sha256_path(spec.run_dir / "checkpoint.pt"),
    }
    if any(payload.get(key) != value for key, value in exact_fields.items()):
        raise RuntimeError(f"Checkpoint audit hash mismatch: {path}")
    return {"path": str(path), "sha256": sha256_path(path)}


def evaluate_task(
    task_index: int, task_tsv: Path, base_out: Path, audit_dir: Path,
    source_lock: Path, card: dict[str, Any], card_hash: str, batch_size: int,
) -> dict[str, Any]:
    started = time.time()
    lock = verify_source_lock(source_lock)
    locked_task = lock["external_inputs"]["full_task_tsv"]
    task_tsv_hash = sha256_path(task_tsv)
    if locked_task["sha256"] != task_tsv_hash:
        raise RuntimeError("Evaluation task table does not match the source lock")
    roster = discover_trained_runs(task_tsv, base_out, card)
    spec = roster[task_index]
    paired = {(item.arm, item.seed): item for item in roster}
    sparse_spec = spec if spec.arm == "sparse" else paired[("sparse", spec.seed)]
    sparse_audit = _authenticate_audit(
        sparse_spec, audit_dir, card, card_hash, task_tsv_hash
    )
    selected_audit = (
        sparse_audit
        if spec.arm == "sparse"
        else _authenticate_audit(spec, audit_dir, card, card_hash, task_tsv_hash)
    )
    _sparse_cfg, sparse_env, sparse_model, _sparse_checkpoint, sparse_path = load_trained_model(sparse_spec)
    if spec.arm == "sparse":
        model, env, selected_path = sparse_model, sparse_env, sparse_path
    else:
        _cfg, env, model, _checkpoint, selected_path = load_trained_model(spec)
        if env.observation_size != sparse_env.observation_size:
            raise RuntimeError("Paired sparse/dense state dimensions differ")
    codebook, retained, discovery = discover_sparse_families(
        sparse_model, sparse_env, card, batch_size
    )
    sparse_routing, sparse_masks, centers, offsets, matched_points = match_sparse_routing(
        sparse_model, sparse_env, codebook, retained, card, batch_size
    )
    try:
        geometry_authentication = _authenticate_mechanism_geometry(
            sparse_env, centers, sparse_routing, card
        )
    except (ArithmeticError, KeyError, ValueError, RuntimeError) as error:
        geometry_authentication = {
            "passed": False,
            "error": f"{type(error).__name__}:{error}",
        }
    if spec.arm == "sparse":
        masks = sparse_masks
        routing = sparse_routing
        routing["mechanism_projector_source"] = "matched_sparse_family_representative"
    elif sparse_routing["family_valid"]:
        masks, dense_projector = dense_center_projectors(
            model, centers, sparse_masks, batch_size
        )
        routing = {
            "family_valid": dense_projector["center_projectors_valid"],
            "paired_sparse_routing": sparse_routing,
            **dense_projector,
            "identical_sparse_assigned_physical_points_used": True,
        }
    else:
        masks = np.empty((0, 0), dtype=bool)
        routing = {
            "family_valid": False,
            "paired_sparse_routing": sparse_routing,
            "center_projectors_valid": False,
            "failure": "paired_sparse_family_routing_invalid",
        }
    result: dict[str, Any] = {
        "status": "ineligible_family_mapping", "law_valid": False,
        "direct_closure_valid": False, "discovery": discovery, "routing": routing,
        "geometry": {"authentication": geometry_authentication},
        "failure_reasons": ["routing_or_dense_center_projector_invalid"],
    }
    if routing["family_valid"] and not geometry_authentication.get("passed", False):
        result.update({
            "status": "ineligible_geometry",
            "failure_reasons": ["evaluation_geometry_authentication_failed"],
        })
    elif routing["family_valid"]:
        try:
            scored = score_mechanism(
                model, masks, centers, offsets, matched_points,
                sparse_env, card, spec.arm, geometry_authentication,
            )
        except (ArithmeticError, ValueError, RuntimeError) as error:
            result.update(_numerical_failure(error))
        else:
            result.update(scored)
            geometry_valid = bool(
                result["geometry"]["authentication"]["passed"]
            )
            direct_valid = bool(result["direct_active_code_cloud_closure"][
                "denominators_finite_and_above_1e-8"
            ])
            law_valid = bool(geometry_valid and result["finite_autograd_jacobians"])
            reasons = []
            if not geometry_valid:
                reasons.append("evaluation_geometry_authentication_failed")
            if not result["finite_autograd_jacobians"]:
                reasons.append("nonfinite_autograd_jacobian")
            if not direct_valid:
                reasons.append("direct_closure_denominator_or_value_invalid")
            result.update({
                "status": "eligible" if law_valid else "ineligible_numerical",
                "law_valid": law_valid,
                "direct_closure_valid": direct_valid,
                "dense_secondary_closure_valid": (
                    direct_valid if spec.arm == "dense" else None
                ),
                "failure_reasons": reasons,
            })
    return {
        "schema_version": 1, "protocol_id": card["protocol_id"], "card_sha256": card_hash,
        "task_tsv_sha256": sha256_path(task_tsv), "task_id": spec.task_id, "arm": spec.arm, "seed": spec.seed,
        "result": result,
        "assertions": {"global_K_unmodified": True, "no_latent_dynamics_fit": True, "training_uses_basin_labels_or_count": False, "all_evaluation_geometry_authenticated": bool(geometry_authentication.get("passed", False)), "dense_center_projectors_exactly_match_paired_sparse_family_cardinalities": spec.arm != "dense" or bool(routing.get("center_projectors_valid", False)), "dense_closure_uses_identical_paired_sparse_physical_points": spec.arm != "dense" or bool(routing.get("identical_sparse_assigned_physical_points_used", False)), "physical_state_estimand": True},
        "provenance": {"run_dir": str(spec.run_dir), "selected_checkpoint_path": str(selected_path), "selected_checkpoint_sha256": sha256_path(selected_path), "paired_sparse_checkpoint_sha256": sha256_path(sparse_path), "checkpoint_audit": selected_audit, "paired_sparse_checkpoint_audit": sparse_audit, "source_lock": {"path": str(source_lock), "sha256": sha256_path(source_lock)}, "source_lock_protocol": lock["protocol_id"], "evaluator_sha256": sha256_path(Path(__file__)), "git_commit": os.environ.get("SKAE_GIT_COMMIT", "launcher_not_recorded")},
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--source_lock", type=Path, required=True)
    parser.add_argument("--expected_source_lock_sha", required=True)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--base_out", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--task_index", type=int, required=True)
    parser.add_argument("--encode_batch_size", type=int, default=4096)
    args = parser.parse_args()
    if sha256_path(args.source_lock) != args.expected_source_lock_sha:
        raise RuntimeError("Expected source-lock hash mismatch")
    card, card_hash = load_card(args.card)
    expected = int(card["task_table_contract"]["full_task_count"])
    if not 0 <= args.task_index < expected:
        raise IndexError(args.task_index)
    output = args.output_dir / "shards" / f"task_{args.task_index:02d}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    payload = evaluate_task(args.task_index, args.task_tsv, args.base_out, args.audit_dir, args.source_lock, card, card_hash, args.encode_batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": payload["result"]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
