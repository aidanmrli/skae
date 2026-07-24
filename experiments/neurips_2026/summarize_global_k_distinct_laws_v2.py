#!/usr/bin/env python3
"""Fail-closed H/G adjudication for prospective distinct-law V2 shards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.neurips_2026.global_k_distinct_laws_v2_source_lock import (
    verify_source_lock,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    load_card,
    sha256_path,
)


BOOTSTRAP_SEED = 20260732
BOOTSTRAP_REPLICATES = 20000


def _load_shards(
    input_dir: Path, card: dict[str, Any], card_hash: str,
    task_hash: str, source_lock_hash: str,
) -> list[dict[str, Any]]:
    expected = int(card["task_table_contract"]["full_task_count"])
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted((input_dir / "shards").glob("task_*.json")):
        payload = json.loads(path.read_text())
        checks = (
            payload.get("protocol_id") == card["protocol_id"],
            payload.get("card_sha256") == card_hash,
            payload.get("task_tsv_sha256") == task_hash,
            payload.get("provenance", {}).get("source_lock", {}).get("sha256") == source_lock_hash,
        )
        if not all(checks):
            raise RuntimeError(f"Shard authentication failed: {path}")
        task_id = int(payload["task_id"])
        if task_id in rows:
            raise RuntimeError(f"Duplicate task shard {task_id}")
        rows[task_id] = payload
    if set(rows) != set(range(expected)):
        raise RuntimeError(
            f"Refusing partial summary: missing={sorted(set(range(expected)) - set(rows))}, "
            f"extra={sorted(set(rows) - set(range(expected)))}"
        )
    ordered = [rows[index] for index in range(expected)]
    seeds = [int(seed) for seed in card["new_seed_contract"]["scientific_seeds"]]
    expected_roster = [("sparse", seed) for seed in seeds] + [
        ("dense", seed) for seed in seeds
    ]
    observed_roster = [(row["arm"], int(row["seed"])) for row in ordered]
    if observed_roster != expected_roster:
        raise RuntimeError(f"Evaluation task roster drift: {observed_roster}")
    evaluator_hashes = {row["provenance"]["evaluator_sha256"] for row in ordered}
    if len(evaluator_hashes) != 1:
        raise RuntimeError("Evaluator drift across shards")
    return ordered


def _law(row: dict[str, Any], estimand: str) -> dict[str, Any]:
    return row["result"][estimand]["law_identification"]


def _row_counts(rows: list[dict[str, Any]], estimand: str) -> tuple[int, int]:
    nearest = ratio = 0
    for row in rows:
        if row["result"]["status"] != "eligible":
            continue
        law = _law(row, estimand)
        costs = np.asarray(law["cost_matrix"])
        kink = row["result"]["kink_guard"]["rows"]
        for basin in range(3):
            if not kink[basin]["passed_both_estimands_both_epsilons"]:
                continue
            nearest += int(np.argmin(costs[basin]) == basin)
            ratio += int(law["own_over_nearest_wrong_by_basin"][basin] <= 0.8)
    return nearest, ratio


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "q25": None, "q75": None, "bootstrap_median_95_ci": None}
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(array, size=(BOOTSTRAP_REPLICATES, len(array)), replace=True)
    medians = np.median(samples, axis=1)
    return {
        "count": len(values),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "bootstrap_median_95_ci": [float(value) for value in np.quantile(medians, [0.025, 0.975])],
    }


def _exact_sign_p(successes: int, trials: int) -> float:
    return float(sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (2**trials))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > 1e-12:
        return float(numerator / denominator)
    return 1.0 if numerator <= 1e-12 else float(numerator / 1e-12)


def _compact_seed(row: dict[str, Any]) -> dict[str, Any]:
    result = row["result"]
    compact: dict[str, Any] = {
        "arm": row["arm"], "seed": row["seed"], "status": result["status"],
        "checkpoint_sha256": row["provenance"]["selected_checkpoint_sha256"],
        "law_valid": result.get("law_valid", False),
        "direct_closure_valid": result.get("direct_closure_valid", False),
        "dense_secondary_closure_valid": result.get("dense_secondary_closure_valid"),
        "failure_reasons": result.get("failure_reasons", []),
        "routing": result.get("routing"),
        "geometry_authentication": result.get("geometry", {}).get("authentication"),
    }
    if not result.get("law_valid", result["status"] == "eligible"):
        return compact
    for estimand in ("H_block", "G_block", "H_global", "G_global_diagnostic"):
        law = _law(row, estimand)
        compact[estimand] = {
            "max_own_relative_error": law["max_own_relative_error"],
            "max_own_over_nearest_wrong": law["max_own_over_nearest_wrong"],
            "identity_over_best_nonidentity": law["identity_over_best_nonidentity"],
            "identity_is_unique_optimum": law["identity_is_unique_optimum"],
        }
    compact.update(
        {
            "H_global_positive_control_pass": result["H_global"]["positive_control_pass"],
            "joint_h_g_seed_pass": result["per_seed_joint_h_g_pass"],
            "g_only_seed_pass": result["per_seed_g_only_pass"],
            "h_only_seed_pass": result["per_seed_h_only_pass"],
            "kink_complete_seed_pass": result["kink_guard"]["complete_seed_pass"],
            "max_g_block_source_closure": result["closure"]["maximum"],
            "H_observed_over_null_median": result["coordinate_null"]["H"]["observed_over_median"],
            "G_observed_over_null_median": result["coordinate_null"]["G"]["observed_over_median"],
            "active_code_cloud_closure": result["direct_active_code_cloud_closure"],
            "center_forecast_guards": result["center_forecast_guards"],
        }
    )
    return compact


def _finite_and_affine(
    rows: list[dict[str, Any]], card: dict[str, Any],
) -> dict[str, Any]:
    names = (
        "affine", "h_small", "g_small", "h_all_own", "g_all_own",
        "h_residual", "g_residual", "joint_finite", "considered",
    )
    counts = {name: 0 for name in names}
    residual_limit = float(
        card["finite_radius_robustness_not_selection"][
            "maximum_normalized_linear_fit_residual_each_radius"
        ]
    )
    for row in rows:
        result = row["result"]
        if result["status"] != "eligible":
            continue
        for basin_index, guard in enumerate(result["center_forecast_guards"]):
            if not result["kink_guard"]["rows"][basin_index]["passed_both_estimands_both_epsilons"]:
                continue
            counts["considered"] += 1
            basin = result["finite_radius_robustness"]["by_basin"][basin_index]
            h_records, g_records = basin["H"], basin["G"]
            row_passes = {
                "affine": guard["restricted_forecast"] <= 0.25
                and guard["k_induced_update"] <= 0.25,
                "h_small": all(record["autograd_agreement"] <= 0.25 for record in h_records[:2]),
                "g_small": all(record["autograd_agreement"] <= 0.25 for record in g_records[:2]),
                "h_all_own": all(record["own_law_is_nearest"] for record in h_records),
                "g_all_own": all(record["own_law_is_nearest"] for record in g_records),
                "h_residual": all(
                    math.isfinite(record["normalized_linear_fit_residual"])
                    and record["normalized_linear_fit_residual"] <= residual_limit
                    for record in h_records
                ),
                "g_residual": all(
                    math.isfinite(record["normalized_linear_fit_residual"])
                    and record["normalized_linear_fit_residual"] <= residual_limit
                    for record in g_records
                ),
            }
            for name, passed in row_passes.items():
                counts[name] += int(passed)
            counts["joint_finite"] += int(all(row_passes.values()))
    return {
        "kink_valid_rows_considered": counts["considered"],
        "affine_rows_passing_both_center_guards": counts["affine"],
        "H_rows_small_radius_agreement": counts["h_small"],
        "G_rows_small_radius_agreement": counts["g_small"],
        "H_rows_own_law_every_radius": counts["h_all_own"],
        "G_rows_own_law_every_radius": counts["g_all_own"],
        "H_rows_residual_at_most_0.25_every_radius": counts["h_residual"],
        "G_rows_residual_at_most_0.25_every_radius": counts["g_residual"],
        "joint_finite_neighborhood_rows_passing_every_gate": counts["joint_finite"],
    }


def _sparse_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {
        name: [] for name in (
            "H_max_own_relative_error", "H_max_own_over_nearest_wrong",
            "H_identity_over_best_nonidentity", "G_max_own_relative_error",
            "G_max_own_over_nearest_wrong", "G_identity_over_best_nonidentity",
            "decoded_G_block_source_closure_max", "active_code_cloud_closure_max",
            "H_observed_over_coordinate_null_median",
            "G_observed_over_coordinate_null_median",
            "active_code_cloud_observed_over_null_median",
        )
    }
    for row in rows:
        result = row["result"]
        h_law, g_law = _law(row, "H_block"), _law(row, "G_block")
        values["H_max_own_relative_error"].append(h_law["max_own_relative_error"])
        values["H_max_own_over_nearest_wrong"].append(h_law["max_own_over_nearest_wrong"])
        values["H_identity_over_best_nonidentity"].append(h_law["identity_over_best_nonidentity"])
        values["G_max_own_relative_error"].append(g_law["max_own_relative_error"])
        values["G_max_own_over_nearest_wrong"].append(g_law["max_own_over_nearest_wrong"])
        values["G_identity_over_best_nonidentity"].append(g_law["identity_over_best_nonidentity"])
        values["decoded_G_block_source_closure_max"].append(result["closure"]["maximum"])
        values["H_observed_over_coordinate_null_median"].append(result["coordinate_null"]["H"]["observed_over_median"])
        values["G_observed_over_coordinate_null_median"].append(result["coordinate_null"]["G"]["observed_over_median"])
        if result.get("direct_closure_valid", False):
            direct = result["direct_active_code_cloud_closure"]
            values["active_code_cloud_closure_max"].append(direct["observed_max_change_normalized_leakage"])
            values["active_code_cloud_observed_over_null_median"].append(direct["observed_over_median_null"])
    return {name: _distribution(metric) for name, metric in values.items()}


def adjudicate(
    rows: list[dict[str, Any]], card: dict[str, Any], audit_summary: dict[str, Any],
) -> dict[str, Any]:
    sparse = [row for row in rows if row["arm"] == "sparse"]
    dense = [row for row in rows if row["arm"] == "dense"]
    sparse_eligible = [row for row in sparse if row["result"]["status"] == "eligible"]
    dense_eligible = [row for row in dense if row["result"]["status"] == "eligible"]
    kink_pairs = sum(
        item["passed_both_estimands_both_epsilons"]
        for row in sparse_eligible for item in row["result"]["kink_guard"]["rows"]
    )
    kink_complete = sum(row["result"]["kink_guard"]["complete_seed_pass"] for row in sparse_eligible)
    audit_rows = audit_summary.get("rows", [])
    audit_valid = bool(
        audit_summary.get("status") == "passed"
        and audit_summary.get("protocol_id") == card["protocol_id"]
        and audit_summary.get("card_sha256") == rows[0]["card_sha256"]
        and audit_summary.get("task_tsv_sha256") == rows[0]["task_tsv_sha256"]
        and audit_summary.get("task_count") == 20
        and audit_summary.get("passed_count") == 20
        and audit_summary.get("arm_counts") == {"sparse": 10, "dense": 10}
        and len(audit_rows) == 20
        and all(
            int(audit["task_id"]) == int(row["task_id"])
            and audit["arm"] == row["arm"]
            and int(audit["seed"]) == int(row["seed"])
            and audit["checkpoint_sha256"] == row["provenance"]["selected_checkpoint_sha256"]
            for audit, row in zip(audit_rows, rows)
        )
    )
    validity = card["validity"]
    validity_pass = bool(
        audit_valid
        and len(sparse_eligible) >= int(validity["required_sparse_family_evaluable"])
        and kink_pairs >= int(validity["required_kink_guard_seed_basin_pairs"])
        and kink_complete >= int(validity["required_kink_guard_complete_seeds"])
    )
    h_nearest, h_ratio_rows = _row_counts(sparse, "H_block")
    g_nearest, g_ratio_rows = _row_counts(sparse, "G_block")
    aggregate = card["aggregate_sparse_gate"]
    joint_seed_count = sum(row["result"].get("per_seed_joint_h_g_pass", False) for row in sparse_eligible)
    g_seed_count = sum(row["result"].get("per_seed_g_only_pass", False) for row in sparse_eligible)
    h_seed_count = sum(row["result"].get("per_seed_h_only_pass", False) for row in sparse_eligible)
    kink_complete_sparse = [
        row for row in sparse_eligible
        if row["result"]["kink_guard"]["complete_seed_pass"]
    ]
    h_null_wins = sum(
        _law(row, "H_block")["identity_over_best_nonidentity"]
        < row["result"]["coordinate_null"]["H"]["median_assignment"]
        for row in kink_complete_sparse
    )
    g_null_wins = sum(
        _law(row, "G_block")["identity_over_best_nonidentity"]
        < row["result"]["coordinate_null"]["G"]["median_assignment"]
        for row in kink_complete_sparse
    )
    closure_seed_passes = sum(
        row["result"]["direct_active_code_cloud_closure"]["per_seed_pass"]
        for row in kink_complete_sparse
    )
    closure_evaluable = [
        row for row in sparse_eligible if row["result"].get("direct_closure_valid", False)
    ]
    closure_rows_pass = sum(
        item["change_normalized_leakage"] <= 0.50
        for row in closure_evaluable
        for basin, item in enumerate(row["result"]["direct_active_code_cloud_closure"]["rows"])
        if row["result"]["kink_guard"]["rows"][basin]["passed_both_estimands_both_epsilons"]
    )
    closure_null_wins = sum(
        row["result"]["direct_active_code_cloud_closure"]["observed_max_change_normalized_leakage"]
        < row["result"]["direct_active_code_cloud_closure"]["median_null_max"]
        for row in closure_evaluable
        if row["result"]["kink_guard"]["complete_seed_pass"]
    )
    closure_gate = bool(
        len(closure_evaluable) >= int(validity["required_sparse_direct_closure_evaluable_for_joint_or_g_only"])
        and
        closure_seed_passes >= int(aggregate["minimum_complete_active_code_cloud_closure_seed_passes"])
        and closure_rows_pass >= int(aggregate["minimum_of_30_active_code_cloud_rows_at_most_0.50"])
        and closure_null_wins >= int(aggregate["minimum_seeds_active_code_cloud_better_than_null"])
    )
    joint_gate = bool(
        validity_pass
        and joint_seed_count >= int(aggregate["minimum_joint_h_g_seed_passes"])
        and h_nearest >= int(aggregate["minimum_h_of_30_basin_rows_with_own_law_nearest"])
        and g_nearest >= int(aggregate["minimum_g_of_30_basin_rows_with_own_law_nearest"])
        and h_ratio_rows >= int(aggregate["minimum_h_of_30_basin_rows_with_own_over_nearest_wrong_at_most_0.8"])
        and g_ratio_rows >= int(aggregate["minimum_g_of_30_basin_rows_with_own_over_nearest_wrong_at_most_0.8"])
        and h_null_wins >= int(aggregate["minimum_seeds_h_assignment_better_than_coordinate_null_median"])
        and g_null_wins >= int(aggregate["minimum_seeds_g_assignment_better_than_coordinate_null_median"])
        and closure_gate
    )
    g_gate = bool(
        validity_pass and g_seed_count >= int(aggregate["minimum_g_only_seed_passes"])
        and g_nearest >= int(aggregate["minimum_g_of_30_basin_rows_with_own_law_nearest"])
        and g_ratio_rows >= int(aggregate["minimum_g_of_30_basin_rows_with_own_over_nearest_wrong_at_most_0.8"])
        and g_null_wins >= int(aggregate["minimum_seeds_g_assignment_better_than_coordinate_null_median"])
        and closure_gate
    )
    h_gate = bool(
        validity_pass and h_seed_count >= int(aggregate["minimum_h_only_seed_passes_for_reporting"])
        and h_nearest >= int(aggregate["minimum_h_of_30_basin_rows_with_own_law_nearest"])
        and h_ratio_rows >= int(aggregate["minimum_h_of_30_basin_rows_with_own_over_nearest_wrong_at_most_0.8"])
        and h_null_wins >= int(aggregate["minimum_seeds_h_assignment_better_than_coordinate_null_median"])
    )
    h_global_count = sum(row["result"]["H_global"]["positive_control_pass"] for row in sparse_eligible)
    geometry = _finite_and_affine(sparse, card)
    affine_gate = geometry["affine_rows_passing_both_center_guards"] >= 24
    finite_gate = bool(
        joint_gate
        and geometry["joint_finite_neighborhood_rows_passing_every_gate"] >= 24
    )
    if not validity_pass:
        mechanism = "invalid"
    elif joint_gate and affine_gate and finite_gate:
        mechanism = "finite_neighborhood_local_laws"
    elif joint_gate and affine_gate:
        mechanism = "affine_fixed_point_local_laws"
    elif joint_gate:
        mechanism = "distinct_predictor_and_k_induced_jacobians_only"
    elif g_gate:
        mechanism = "distinct_k_induced_update_jacobians_only"
    elif h_gate:
        mechanism = "restricted_predictor_jacobians_mechanism_unresolved"
    elif h_global_count >= 8:
        mechanism = "global_predictor_only"
    else:
        mechanism = "failed"

    sparse_by_seed = {
        int(row["seed"]): row for row in sparse_eligible
        if row["result"]["kink_guard"]["complete_seed_pass"]
    }
    dense_by_seed = {
        int(row["seed"]): row for row in dense_eligible
        if row["result"]["kink_guard"]["complete_seed_pass"]
    }
    paired_seeds = sorted(set(sparse_by_seed) & set(dense_by_seed))
    ratios: dict[str, list[float]] = {"H_row": [], "H_assignment": [], "G_row": [], "G_assignment": [], "active_code_cloud_closure": []}
    better_h = better_g = better_closure = 0
    for seed in paired_seeds:
        left, right = sparse_by_seed[seed], dense_by_seed[seed]
        h_row = _safe_ratio(_law(left, "H_block")["max_own_over_nearest_wrong"], _law(right, "H_block")["max_own_over_nearest_wrong"])
        h_assignment = _safe_ratio(_law(left, "H_block")["identity_over_best_nonidentity"], _law(right, "H_block")["identity_over_best_nonidentity"])
        g_row = _safe_ratio(_law(left, "G_block")["max_own_over_nearest_wrong"], _law(right, "G_block")["max_own_over_nearest_wrong"])
        g_assignment = _safe_ratio(_law(left, "G_block")["identity_over_best_nonidentity"], _law(right, "G_block")["identity_over_best_nonidentity"])
        ratios["H_row"].append(float(h_row)); ratios["H_assignment"].append(float(h_assignment))
        ratios["G_row"].append(float(g_row)); ratios["G_assignment"].append(float(g_assignment))
        better_h += int(
            _law(left, "H_block")["max_own_over_nearest_wrong"]
            < _law(right, "H_block")["max_own_over_nearest_wrong"]
            and _law(left, "H_block")["identity_over_best_nonidentity"]
            < _law(right, "H_block")["identity_over_best_nonidentity"]
        )
        better_g += int(
            _law(left, "G_block")["max_own_over_nearest_wrong"]
            < _law(right, "G_block")["max_own_over_nearest_wrong"]
            and _law(left, "G_block")["identity_over_best_nonidentity"]
            < _law(right, "G_block")["identity_over_best_nonidentity"]
        )
        if left["result"].get("direct_closure_valid", False) and right["result"].get("direct_closure_valid", False):
            closure_ratio = _safe_ratio(left["result"]["direct_active_code_cloud_closure"]["observed_max_change_normalized_leakage"], right["result"]["direct_active_code_cloud_closure"]["observed_max_change_normalized_leakage"])
            ratios["active_code_cloud_closure"].append(float(closure_ratio))
            better_closure += int(
                left["result"]["direct_active_code_cloud_closure"]["observed_max_change_normalized_leakage"]
                < right["result"]["direct_active_code_cloud_closure"]["observed_max_change_normalized_leakage"]
            )
    dense_global_count = sum(row["result"]["H_global"]["positive_control_pass"] for row in dense_eligible)
    specificity_card = card["dense_recipe_specificity_gate"]
    specificity_valid = bool(
        audit_valid
        and len(paired_seeds) >= int(validity["required_dense_center_projector_evaluable_for_specificity"])
        and dense_global_count >= 8
    )
    specificity_pass = bool(
        specificity_valid
        and h_gate
        and np.median(ratios["H_row"]) <= float(specificity_card["maximum_median_h_row_ratio"])
        and np.median(ratios["H_assignment"]) <= float(specificity_card["maximum_median_h_assignment_ratio"])
        and better_h >= int(specificity_card["minimum_paired_seeds_sparse_better_on_both_h_metrics"])
    )
    relative = (
        "dense_invalid_specificity_unresolved" if not specificity_valid
        else "sparse_recipe_support_basis_specific" if specificity_pass
        else "not_sparse_recipe_specific"
    )
    tiers = card["decision_structure"]
    return {
        "schema_version": 1, "protocol_id": card["protocol_id"],
        "mechanism_tier": mechanism, "mechanism_text": tiers["mechanism_claim_tiers"][mechanism],
        "relative_specificity_tier": relative, "relative_specificity_text": tiers["relative_specificity_tiers"][relative],
        "mandatory_caveat": tiers["mandatory_positive_claim_caveat"],
        "validity": {"passed": validity_pass, "checkpoint_audit": audit_valid, "sparse_law_evaluable": len(sparse_eligible), "sparse_direct_closure_evaluable": len(closure_evaluable), "dense_center_projector_law_evaluable": len(dense_eligible), "joint_H_G_kink_pairs": kink_pairs, "joint_H_G_kink_complete_seeds": kink_complete},
        "sparse_gates": {"joint_H_G": joint_gate, "G_only": g_gate, "H_only_reporting": h_gate, "joint_seed_passes": joint_seed_count, "G_seed_passes": g_seed_count, "H_seed_passes": h_seed_count, "H_nearest_rows": h_nearest, "G_nearest_rows": g_nearest, "H_ratio_rows": h_ratio_rows, "G_ratio_rows": g_ratio_rows, "H_G_row_kink_valid_denominator": kink_pairs, "H_null_wins": h_null_wins, "G_null_wins": g_null_wins, "H_G_null_complete_kink_seed_denominator": len(kink_complete_sparse), "H_null_exact_sign_p_10_trials": _exact_sign_p(h_null_wins, 10), "G_null_exact_sign_p_10_trials": _exact_sign_p(g_null_wins, 10), "active_code_cloud_closure_gate": closure_gate, "active_code_cloud_closure_seed_passes": closure_seed_passes, "active_code_cloud_rows_at_most_0.50": closure_rows_pass, "active_code_cloud_row_kink_and_closure_valid_denominator": sum(row["result"]["kink_guard"]["rows"][basin]["passed_both_estimands_both_epsilons"] for row in closure_evaluable for basin in range(3)), "active_code_cloud_null_wins": closure_null_wins, "active_code_cloud_null_complete_kink_seed_denominator": sum(row["result"]["kink_guard"]["complete_seed_pass"] for row in closure_evaluable), "active_code_cloud_null_exact_sign_p_10_trials": _exact_sign_p(closure_null_wins, 10), "H_global_positive_seeds": h_global_count, "affine_gate": affine_gate, "finite_gate": finite_gate, **geometry},
        "sparse_distributions": _sparse_distributions(sparse_eligible),
        "specificity": {"valid": specificity_valid, "passed": specificity_pass, "paired_seeds": paired_seeds, "missing_or_ineligible_pairs_counted_as_sign_failures": True, "dense_H_global_positive_seeds": dense_global_count, "H_sparse_better_both_count": better_h, "G_sparse_better_both_count_secondary": better_g, "active_code_cloud_sparse_better_count_secondary": better_closure, "H_exact_sign_p_10_trials": _exact_sign_p(better_h, 10), "G_exact_sign_p_secondary_10_trials": _exact_sign_p(better_g, 10), "active_code_cloud_exact_sign_p_secondary_10_trials": _exact_sign_p(better_closure, 10), "ratios_by_seed": ratios, "distributions": {name: _distribution(values) for name, values in ratios.items()}},
        "audited_parameter_counts_by_arm": audit_summary.get("parameter_counts_by_arm"),
        "seed_rows": [_compact_seed(row) for row in rows],
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "inference_unit": "paired model seed"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--source_lock", type=Path, required=True)
    parser.add_argument("--expected_source_lock_sha", required=True)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if sha256_path(args.source_lock) != args.expected_source_lock_sha:
        raise RuntimeError("Expected source-lock hash mismatch")
    lock = verify_source_lock(args.source_lock)
    card, card_hash = load_card(args.card)
    task_hash = sha256_path(args.task_tsv)
    if task_hash != lock["external_inputs"]["full_task_tsv"]["sha256"]:
        raise RuntimeError("Summary task-table/source-lock mismatch")
    rows = _load_shards(args.input_dir, card, card_hash, task_hash, args.expected_source_lock_sha)
    audit_summary = json.loads((args.audit_dir / "summary.json").read_text())
    decision = adjudicate(rows, card, audit_summary)
    decision["provenance"] = {"card_sha256": card_hash, "task_tsv_sha256": task_hash, "source_lock_sha256": args.expected_source_lock_sha, "summarizer_sha256": sha256_path(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "mechanism_tier": decision["mechanism_tier"], "relative_specificity_tier": decision["relative_specificity_tier"]}, sort_keys=True))


if __name__ == "__main__":
    main()
