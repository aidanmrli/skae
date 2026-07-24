"""Training-frozen family qualification and routing decisions."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from experiments.neurips_2026.allen_cahn_support_subspaces.reduction_statistics import (
    bootstrap_mean_interval,
    bootstrap_ratio_interval,
    median,
)


def family_decision(
    shards: list[dict[str, Any]],
    card: dict[str, Any],
    *,
    closure_reducer: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    gate = card["family_gates"]
    source_limit = float(gate["minimum_routed_initial_source_capture_rms"])
    reconstruction_limit = float(gate["maximum_projected_initial_reconstruction_over_full"])
    eligible, family_forecast_passes, family_matrix_passes = [], 0, 0
    signature_ratios, signature_differences = [], []
    qualification_rows = []
    for item in shards:
        if not item["family"]["sparse"]["eligible"]:
            qualification_rows.append({
                "seed": int(item["seed"]),
                "score_family_gate_passed": False,
                "source_capture_rms": None,
                "initial_reconstruction_ratio": None,
                "eligible": False,
                "exclusion_reasons": ["score_family_gate"],
            })
            continue
        projection = item["sparse_family"]["initial_projection"]
        full_reconstruction = float(projection["full_initial_reconstruction_mse"])
        projected_reconstruction = float(projection["projected_initial_reconstruction_mse"])
        reconstruction_ratio = projected_reconstruction / max(full_reconstruction, 1e-20)
        reasons = []
        if float(projection["source_capture_rms"]) < source_limit:
            reasons.append("source_capture")
        if reconstruction_ratio > reconstruction_limit:
            reasons.append("initial_reconstruction")
        qualification_rows.append({
            "seed": int(item["seed"]),
            "score_family_gate_passed": True,
            "source_capture_rms": float(projection["source_capture_rms"]),
            "initial_reconstruction_ratio": reconstruction_ratio,
            "eligible": not reasons,
            "exclusion_reasons": reasons,
        })
        if reasons:
            continue
        eligible.append(item)
        top_indices = [
            int(value)
            for value in item["family"]["sparse"]["fit_frozen_top_two_family_indices"]
        ]
        family_rows = item["sparse_family"]["forecast"]["families"]
        top_two = [family_rows[str(index)] for index in top_indices]
        aggregate = item["sparse_family"]["forecast"]
        utility_rows = top_two + [aggregate]
        utility_pass = len(top_two) == 2 and all(
            float(row[str(horizon)]["restricted"]["field_mse"])
            / float(row[str(horizon)]["mask_once"]["field_mse"])
            <= float(card["forecast_gates"]["family_maximum_restricted_over_mask_once"])
            and float(row[str(horizon)]["mask_once"]["field_mse"])
            / float(row[str(horizon)]["full"]["field_mse"])
            <= float(card["forecast_gates"]["family_maximum_mask_once_over_full"])
            and float(row[str(horizon)]["restricted"]["field_mse"])
            / float(row[str(horizon)]["full"]["field_mse"])
            <= float(card["forecast_gates"]["family_maximum_restricted_over_full"])
            for row in utility_rows for horizon in card["roster"]["horizons"]
        )
        if utility_pass:
            family_forecast_passes += 1
        matrix_rows = item["sparse_family"]["qualified_family_matrix_closure"]
        matrix_limits = {
            "matrix_k_leakage_fro": float(card["closure_gates"]["maximum_matrix_K_leakage"]),
            "matrix_kminusI_leakage_fro": float(
                card["closure_gates"]["maximum_matrix_KminusI_leakage"]
            ),
        }
        if len(top_indices) == 2 and all(
            float(matrix_rows[str(index)]["true"][metric]) <= limit
            and float(matrix_rows[str(index)]["true"][metric])
            / float(matrix_rows[str(index)]["null_median"][metric])
            <= float(card["closure_gates"]["maximum_observed_over_null_ratio"])
            for index in top_indices for metric, limit in matrix_limits.items()
        ):
            family_matrix_passes += 1
        signature = item["sparse_family"]["signature_differentiation"]
        signature_ratios.append(float(signature["observed_over_null"]))
        signature_differences.append(
            float(signature["observed"]) - float(signature["null_median"])
        )
    minimum = int(gate["minimum_qualifying_seeds"])
    family_checks = {
        "eligible_seeds": len(eligible) >= minimum,
        "two_family_forecast_seeds": family_forecast_passes
        >= int(card["forecast_gates"]["minimum_family_qualifying_seeds"]),
        "top_two_individual_matrix_closure_seeds": family_matrix_passes
        >= int(gate["minimum_top_two_individual_matrix_closure_seeds"]),
    }
    reps, seed = int(card["aggregation"]["bootstrap_replicates"]), int(card["aggregation"]["bootstrap_seed"])
    signature_interval = bootstrap_mean_interval(
        signature_ratios, replicates=reps, seed=seed + 3000
    ) if signature_ratios else [0.0, 0.0]
    signature_mean = float(np.mean(signature_ratios)) if signature_ratios else None
    signature_gate = card["signature_differentiation_gates"]
    signature_checks = {
        "mean_ratio": signature_mean is not None and signature_mean
        >= float(signature_gate["minimum_mean_observed_over_null_distance_ratio"]),
        "median_guard": bool(signature_ratios) and median(signature_ratios)
        >= float(signature_gate["minimum_median_observed_over_null_distance_ratio"]),
        "wins": sum(value > 0 for value in signature_differences)
        >= int(signature_gate["minimum_seed_wins"]),
        "interval": signature_interval[0]
        > float(signature_gate["bootstrap_lower_bound_above"]),
    }
    family_closure = closure_reducer(eligible, card, regime="family") if eligible else {"passed": False}
    routing_cells: dict[str, Any] = {}
    routing_gate = card["routing_specificity_gates"]
    for horizon in routing_gate["horizons"]:
        key = str(horizon)
        if not eligible:
            routing_cells[key] = {
                "correct_over_wrong_restricted_ratio_of_seed_means": None,
                "correct_over_wrong_restriction_factor_ratio_of_seed_means": None,
                "restricted_seed_wins": 0,
                "restriction_factor_seed_wins": 0,
                "restricted_ratio_bootstrap": None,
                "restriction_factor_ratio_bootstrap": None,
                "passed": False,
            }
            continue
        correct_restricted = [
            float(item["sparse_family"]["top_two_family_derangement"]["correct"][key]
                  ["restricted"]["field_mse"])
            for item in eligible
        ]
        wrong_restricted = [
            float(item["sparse_family"]["top_two_family_derangement"]["wrong_swap"][key]
                  ["restricted"]["field_mse"])
            for item in eligible
        ]
        correct_factor = [
            float(item["sparse_family"]["top_two_family_derangement"]["correct"]["ratios"]
                  [key]["mean_restricted_over_mask_once"])
            for item in eligible
        ]
        wrong_factor = [
            float(item["sparse_family"]["top_two_family_derangement"]["wrong_swap"]["ratios"]
                  [key]["mean_restricted_over_mask_once"])
            for item in eligible
        ]
        restricted_ci = bootstrap_ratio_interval(
            correct_restricted, wrong_restricted,
            replicates=int(card["aggregation"]["bootstrap_replicates"]),
            seed=int(card["aggregation"]["bootstrap_seed"]) + 4000 + int(horizon),
        )
        factor_ci = bootstrap_ratio_interval(
            correct_factor, wrong_factor,
            replicates=int(card["aggregation"]["bootstrap_replicates"]),
            seed=int(card["aggregation"]["bootstrap_seed"]) + 5000 + int(horizon),
        )
        restricted_wins = int(sum(a < b for a, b in zip(correct_restricted, wrong_restricted)))
        factor_wins = int(sum(a < b for a, b in zip(correct_factor, wrong_factor)))
        restricted_ratio = float(np.mean(correct_restricted) / np.mean(wrong_restricted))
        factor_ratio = float(np.mean(correct_factor) / np.mean(wrong_factor))
        routing_cells[key] = {
            "correct_over_wrong_restricted_ratio_of_seed_means": restricted_ratio,
            "correct_over_wrong_restriction_factor_ratio_of_seed_means": factor_ratio,
            "restricted_seed_wins": restricted_wins,
            "restriction_factor_seed_wins": factor_wins,
            "restricted_ratio_bootstrap": restricted_ci,
            "restriction_factor_ratio_bootstrap": factor_ci,
            "passed": restricted_ratio <= float(
                routing_gate["maximum_correct_over_wrong_restricted_ratio_of_seed_means"]
            ) and factor_ratio <= float(
                routing_gate["maximum_correct_over_wrong_restriction_factor_ratio_of_seed_means"]
            ) and min(restricted_wins, factor_wins) >= int(routing_gate["minimum_seed_wins"])
            and max(restricted_ci[1], factor_ci[1]) < float(
                routing_gate["paired_ratio_bootstrap_upper_below"]
            ),
        }
    routing_passed = len(eligible) >= minimum and all(
        cell["passed"] for cell in routing_cells.values()
    )
    return {
        "family_passed": all(family_checks.values()) and bool(family_closure["passed"]),
        "signature_differentiation_passed": all(signature_checks.values()),
        "routing_specificity_passed": routing_passed,
        "family_checks": family_checks,
        "family_closure": family_closure,
        "signature_checks": signature_checks,
        "signature_ratio_median": median(signature_ratios) if signature_ratios else None,
        "signature_ratio_mean": signature_mean,
        "signature_ratio_bootstrap": signature_interval,
        "routing_specificity": routing_cells,
        "qualification": {
            "source_capture_minimum": source_limit,
            "initial_reconstruction_ratio_maximum": reconstruction_limit,
            "eligible_seed_count": len(eligible),
            "top_two_utility_seed_count": family_forecast_passes,
            "top_two_individual_matrix_seed_count": family_matrix_passes,
            "eligible_seeds": [int(item["seed"]) for item in eligible],
            "per_seed": qualification_rows,
        },
    }
