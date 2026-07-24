"""Predeclared claim adjudication for periodic-reencoding outcomes."""

from __future__ import annotations

from typing import Any


def _strong(endpoint: dict[str, Any], *, minimum_reduction: float) -> dict[str, bool]:
    checks = {
        "minimum_reduction": float(endpoint["relative_reduction_of_arm_means"])
        >= minimum_reduction,
        "bootstrap_lower_above_zero": float(
            endpoint["paired_ratio_bootstrap"]["ci95_lower"]
        ) > 0.0,
        "arm_swap_p_at_most_0p05": float(
            endpoint["exact_one_sided_studentized_sign_flip"]["one_sided_exact_p"]
        ) <= 0.05,
        "at_least_8_of_10_seed_wins": int(endpoint["sparse_seed_wins"]) >= 8,
        "all_three_dataset_directions_positive": all(
            float(row["relative_reduction_of_arm_means"]) > 0.0
            for row in endpoint["per_dataset_effects"]
        ),
    }
    return {**checks, "passed": all(checks.values())}


def _policy_generalization(
    conditional: dict[str, Any], pipeline_arm: dict[str, Any]
) -> dict[str, Any]:
    cadence = pipeline_arm["selected_cadence"]
    point = float(pipeline_arm["heldout_point_relative_reduction"])
    if cadence == "direct":
        status = "direct_selected_no_periodic_policy_claim"
    elif point <= 0.0:
        status = "heldout_direction_reversal_selection_did_not_generalize"
    elif float(pipeline_arm["selection_aware_bootstrap_ci95_lower"]) > 0.0:
        status = "positive_heldout_gain_with_selection_aware_interval_above_zero"
    else:
        status = "positive_heldout_gain_but_selection_uncertainty_remains"
    return {
        "status": status,
        "selected_cadence": cadence,
        "heldout_point_relative_reduction": point,
        "selection_aware_pipeline_bootstrap": pipeline_arm,
        "conditional_fixed_cadence_diagnostic": conditional,
        "overfitting_boundary": (
            "A held-out direction reversal means cadence selection did not "
            "generalize. Periodic-only held-out success is instead consistent with "
            "decode--reencode stabilization or limited repeated-K durability; it "
            "does not establish parameter overfitting."
        ),
    }


def _conditional_policy_helped(endpoint: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "positive_reduction": float(endpoint["relative_reduction_of_arm_means"]) > 0.0,
        "bootstrap_lower_above_zero": float(
            endpoint["paired_ratio_bootstrap"]["ci95_lower"]
        ) > 0.0,
        "arm_swap_p_at_most_0p05": float(
            endpoint["exact_one_sided_studentized_sign_flip"]["one_sided_exact_p"]
        ) <= 0.05,
        "at_least_8_of_10_seed_wins": int(endpoint["selected_seed_wins"]) >= 8,
    }
    return {**checks, "passed": all(checks.values())}


def _truth_scope(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    if len(rows) != 3:
        raise RuntimeError("Truth-difficulty evidence must contain three datasets")
    ratios = [float(row["late_over_early_one_step_truth_change_ratio"]) for row in rows]
    all_near_stationary = all(value < threshold for value in ratios)
    any_near_stationary = any(value < threshold for value in ratios)
    if all_near_stationary:
        language = "attractor_or_pattern_retention_only"
    elif any_near_stationary:
        language = "heterogeneous_late_dynamics_dataset_specific_claims_only"
    else:
        language = "active_long_horizon_dynamics_forecasting"
    return {
        "late_over_early_one_step_truth_change_ratios": ratios,
        "mean_ratio_descriptive": sum(ratios) / len(ratios),
        "near_stationary_threshold": threshold,
        "all_three_near_stationary": all_near_stationary,
        "heterogeneous_near_stationarity": any_near_stationary and not all_near_stationary,
        "permitted_h400_language": language,
    }


def adjudicate(
    *,
    primary: dict[str, Any],
    direct_h200: dict[str, Any],
    policy_h200: dict[str, Any],
    absolute_skill_h200: dict[str, Any],
    absolute_skill_h400: dict[str, Any] | None,
    stress_summary: dict[str, Any] | None,
    stress_policy: dict[str, Any] | None,
    selection_aware_h400_tail: dict[str, Any] | None,
    stress_failures: list[dict[str, Any]],
    truth_difficulty: list[dict[str, Any]],
    truth_threshold: float,
    selection_aware_h400_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen H200, policy-generalization, and H400 language gates."""

    primary_checks = _strong(primary, minimum_reduction=0.05)
    h200_sparse_skill = absolute_skill_h200["endpoints"][
        "h200_cumulative_field_mse"
    ]["sparse"]
    primary_checks.update(
        {
            "sparse_beats_x0_persistence": float(
                h200_sparse_skill["model_over_x0_persistence"]
            ) < 1.0,
            "sparse_beats_x0_persistence_on_all_three_datasets": bool(
                h200_sparse_skill["all_three_dataset_ratios_below_one"]
            ),
        }
    )
    primary_checks["passed"] = all(
        value for key, value in primary_checks.items() if key != "passed"
    )
    direct_h200_checks = _strong(direct_h200, minimum_reduction=0.05)
    policy_status = {
        arm: _policy_generalization(
            policy_h200["conditional"][arm], policy_h200["selection_aware"][arm]
        )
        for arm in ("dense", "sparse")
    }
    tail = None
    tail_checks = None
    direct_tail_checks = None
    periodic_tail_support: dict[str, Any] = {}
    truth_scope = _truth_scope(truth_difficulty, truth_threshold)
    if stress_summary is not None and stress_policy is not None:
        tail = (
            selection_aware_h400_tail
            if selection_aware_h400_tail is not None
            else stress_summary["selected_recipe_comparison"]["endpoints"][
                "h201_h400_tail_field_mse"
            ]
        )
        direct = stress_summary["same_cadence_sensitivity"].get(
            "direct", stress_summary["selected_recipe_comparison"]
        )
        direct_tail = direct["endpoints"]["h201_h400_tail_field_mse"]
        tail_checks = _strong(tail, minimum_reduction=0.0)
        if absolute_skill_h400 is None:
            raise RuntimeError("A finite H400 tier lacks absolute-skill evidence")
        tail_sparse_skill = absolute_skill_h400["endpoints"][
            "h201_h400_tail_field_mse"
        ]["sparse"]
        tail_checks.update(
            {
                "sparse_beats_x0_persistence": float(
                    tail_sparse_skill["model_over_x0_persistence"]
                ) < 1.0,
                "sparse_beats_x0_persistence_on_all_three_datasets": bool(
                    tail_sparse_skill["all_three_dataset_ratios_below_one"]
                ),
            }
        )
        tail_checks["passed"] = all(
            value for key, value in tail_checks.items() if key != "passed"
        )
        direct_tail_checks = _strong(direct_tail, minimum_reduction=0.0)
        if selection_aware_h400_policy is not None:
            periodic_tail_support = {
                arm: {
                    "positive_reduction": float(
                        selection_aware_h400_policy[arm][
                            "heldout_point_relative_reduction"
                        ]
                    ) > 0.0,
                    "bootstrap_lower_above_zero": float(
                        selection_aware_h400_policy[arm][
                            "selection_aware_bootstrap_ci95_lower"
                        ]
                    ) > 0.0,
                    "at_least_8_of_10_seed_wins": int(
                        selection_aware_h400_policy[arm][
                            "heldout_point_selected_seed_wins"
                        ]
                    ) >= 8,
                }
                for arm in ("dense", "sparse")
                if policy_h200["selection_aware"][arm]["selected_cadence"]
                != "direct"
            }
            for checks in periodic_tail_support.values():
                checks["passed"] = all(checks.values())
        else:
            periodic_tail_support = {
                arm: _conditional_policy_helped(
                    stress_policy["arms"][arm]["endpoints"][
                        "h201_h400_tail_field_mse"
                    ]
                )
                for arm in ("dense", "sparse")
                if policy_h200["selection_aware"][arm]["selected_cadence"]
                != "direct"
            }
    periodic_arms = [
        arm
        for arm in ("dense", "sparse")
        if policy_h200["selection_aware"][arm]["selected_cadence"] != "direct"
    ]
    periodic_h200_supported = all(
        policy_status[arm]["status"]
        == "positive_heldout_gain_with_selection_aware_interval_above_zero"
        for arm in periodic_arms
    )
    periodic_tail_supported = all(
        periodic_tail_support.get(arm, {}).get("passed", False)
        for arm in periodic_arms
    )
    if not primary_checks["passed"]:
        if float(primary["relative_reduction_of_arm_means"]) <= 0.0:
            branch = "selected_sparse_recipe_no_heldout_h200_advantage"
        elif (
            not primary_checks["bootstrap_lower_above_zero"]
            or not primary_checks["arm_swap_p_at_most_0p05"]
        ):
            branch = "positive_but_inconclusive_h200_sparse_recipe_advantage"
        else:
            branch = "positive_h200_sparse_recipe_effect_below_full_strong_gate"
    elif stress_summary is None:
        branch = "strong_h200_sparse_recipe_h400_stress_unavailable"
    elif float(tail["relative_reduction_of_arm_means"]) <= 0.0:
        branch = "strong_h200_advantage_reverses_in_h201_h400"
    elif not tail_checks["passed"]:
        branch = (
            "positive_h201_h400_effect_but_"
            + (
                "selection_aware_durability_inconclusive"
                if selection_aware_h400_tail is not None
                else "conditional_durability_inconclusive"
            )
        )
    elif truth_scope["all_three_near_stationary"]:
        branch = "strong_h200_with_h400_pattern_retention_only"
    elif truth_scope["heterogeneous_near_stationarity"]:
        branch = "strong_h200_with_h400_dataset_specific_dynamics_only"
    elif not periodic_arms:
        branch = (
            "selection_aware_h400_durability"
            if selection_aware_h400_tail is not None
            else "conditional_h400_durability"
        ) + "_with_direct_selected_for_both_arms"
    elif periodic_h200_supported and periodic_tail_supported:
        branch = (
            "selection_aware_h400_durability"
            if selection_aware_h400_tail is not None
            else "conditional_h400_durability"
        ) + "_with_refresh_dependent_policies"
    else:
        branch = (
            ("selection_aware" if selection_aware_h400_tail is not None else "conditional")
            + "_h400_sparse_recipe_difference_but_periodic_policy_"
            "benefit_not_supported"
        )
    return {
        "branch": branch,
        "strong_selection_aware_h200_selected_policy": primary_checks,
        "unadjusted_descriptive_h200_direct_sensitivity": direct_h200_checks,
        "h201_h400_selected_policy": tail_checks,
        "h400_inference_role": (
            "selection_aware_selector_rerun"
            if selection_aware_h400_tail is not None
            else "conditional_fixed_validation_selection"
        ),
        "unadjusted_descriptive_h201_h400_direct_sensitivity": direct_tail_checks,
        "periodic_selection_generalization_by_arm": policy_status,
        "absolute_forecast_skill": {
            "h200": absolute_skill_h200,
            "h400": absolute_skill_h400,
        },
        "periodic_tail_support_by_arm": periodic_tail_support,
        "h400_within_arm_policy_inference_role": (
            "selection_aware_selector_rerun_bootstrap"
            if selection_aware_h400_policy is not None
            else "conditional_fixed_validation_selection"
        ),
        "h400_truth_dynamics_language_gate": truth_scope,
        "h400_stress_failures": stress_failures,
        "claim_boundary": (
            "Sparse-versus-dense differences identify the joint sparse training "
            "recipe. Direct rollout tests repeated-K dynamics; periodic rollout "
            "tests an autonomous nonlinear E-circ-D-cycled predictor. A benefit is "
            "consistent with stabilization but does not uniquely identify a mechanism."
        ),
    }
