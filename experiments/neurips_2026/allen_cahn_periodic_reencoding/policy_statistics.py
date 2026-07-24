"""Within-recipe selected-periodic versus direct forecast statistics."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    ARMS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DIRECT,
    ENDPOINTS,
    Cadence,
    Frozen,
    PreparedRow,
    RowKey,
    _test_rows,
    exact_one_sided_studentized_sign_flip,
    paired_ratio_bootstrap,
)


def _within_arm_seed_values(
    prepared: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    *,
    arm: str,
    selected_cadence: Cadence,
    endpoint: Callable[[PreparedRow], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired direct/selected values after within-seed dataset averaging."""

    result: dict[Cadence, np.ndarray] = {}
    for cadence in (DIRECT, selected_cadence):
        seed_values = [
            np.mean(
                [
                    endpoint(prepared[(arm, model, dataset, cadence)])
                    for dataset in frozen["test_seeds"]
                ]
            )
            for model in frozen["model_seeds"]
        ]
        result[cadence] = np.asarray(seed_values, dtype=np.float64)
    direct, selected = result[DIRECT], result[selected_cadence]
    if (
        direct.shape != (10,)
        or selected.shape != (10,)
        or not np.isfinite(direct).all()
        or not np.isfinite(selected).all()
    ):
        raise FloatingPointError("Within-arm aggregation is incomplete or nonfinite")
    return direct, selected


def _selected_vs_direct_endpoint(
    prepared: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    *,
    arm: str,
    selected_cadence: Cadence,
    endpoint: Callable[[PreparedRow], float],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    direct, selected = _within_arm_seed_values(
        prepared,
        frozen,
        arm=arm,
        selected_cadence=selected_cadence,
        endpoint=endpoint,
    )
    direct_mean, selected_mean = float(direct.mean()), float(selected.mean())
    if direct_mean <= 0.0:
        raise ValueError("Direct mean must be positive for a relative reduction")
    exact = exact_one_sided_studentized_sign_flip(direct - selected)
    bootstrap = paired_ratio_bootstrap(
        direct,
        selected,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    identity = selected_cadence == DIRECT
    if identity:
        # Do not interpret numerical cancellation as an estimated policy gain.
        exact["observed_studentized_statistic"] = 0.0
        exact["observed_studentized_statistic_status"] = "finite"
        exact["one_sided_exact_p"] = 1.0
        bootstrap["relative_reduction_of_arm_means"] = 0.0
        bootstrap["ci95_lower"] = 0.0
        bootstrap["ci95_upper"] = 0.0
    return {
        "direct_paired_seed_values": direct.tolist(),
        "selected_paired_seed_values": selected.tolist(),
        "direct_mean": direct_mean,
        "selected_mean": selected_mean,
        "selected_over_direct_ratio_of_arm_means": (
            1.0 if identity else selected_mean / direct_mean
        ),
        "relative_reduction_of_arm_means": (
            0.0 if identity else 1.0 - selected_mean / direct_mean
        ),
        "selected_seed_wins": int(np.sum(selected < direct)),
        "exact_one_sided_studentized_sign_flip": exact,
        "paired_ratio_bootstrap": bootstrap,
        "identity_comparison_selected_is_direct": identity,
    }


def summarize_selected_vs_direct(
    rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    selected_cadences: Mapping[str, Any],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compare each arm's validation-selected policy with its direct rollout."""

    prepared, frozen, selected, tested = _test_rows(rows, card, selected_cadences)
    arms: dict[str, Any] = {}
    for arm_index, arm in enumerate(ARMS):
        endpoints = {}
        for endpoint_index, (name, endpoint) in enumerate(ENDPOINTS):
            endpoints[name] = _selected_vs_direct_endpoint(
                prepared,
                frozen,
                arm=arm,
                selected_cadence=selected[arm],
                endpoint=endpoint,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 100 * arm_index + endpoint_index,
            )
            endpoints[name]["inference_role"] = (
                "primary_policy_generalization"
                if name == "h200_cumulative_field_mse"
                else "secondary_durability_diagnostic"
            )
        arms[arm] = {
            "baseline_cadence": DIRECT,
            "selected_cadence": selected[arm],
            "endpoints": endpoints,
        }
    return {
        "selected_cadences_from_validation": selected,
        "tested_cadences_equal_for_both_arms": list(tested),
        "arms": arms,
        "primary_endpoint": "h200_cumulative_field_mse",
        "effect": "one_minus_mean_selected_over_mean_direct",
        "aggregation_order": (
            "average_three_test_datasets_within_model_seed_then_compare_ten_"
            "paired_model_seeds"
        ),
        "selection_policy": "validation_frozen_recipe_level_cadence_no_test_selection",
    }
