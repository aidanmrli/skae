"""Optional full-grid H400 selection-aware durability inference.

The cadence selector in this module is always fit on the frozen H200
validation cross.  H400 outcomes are evaluated only after that choice and
are never allowed to alter it, including inside permutations and bootstrap
replicates.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_periodic_reencoding.pipeline_inference import (
    _point_effects,
    _point_selection,
    _risk_cube,
    _select_indices,
    selection_aware_exact_sign_flip,
    selection_aware_paired_bootstrap,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    ARMS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DIRECT,
    TEST_HORIZON,
    Frozen,
    PreparedRow,
    RowKey,
    _frozen_card,
    _integer,
    _prepare_rows,
    _validation_rows,
)


Endpoint = Callable[[PreparedRow], float]


def _h400_cumulative(row: PreparedRow) -> float:
    return float(row[1][TEST_HORIZON - 1])


def _h201_h400_tail(row: PreparedRow) -> float:
    return float(np.mean(row[0][200:TEST_HORIZON]))


H400_ENDPOINTS: tuple[tuple[str, Endpoint, str], ...] = (
    (
        "h400_cumulative_field_mse",
        _h400_cumulative,
        "mean decoded field MSE accumulated over forecast steps 1 through 400",
    ),
    (
        "h201_h400_tail_field_mse",
        _h201_h400_tail,
        "mean decoded field MSE over forecast steps 201 through 400 only",
    ),
)


def _full_grid_h400_rows(
    rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
) -> tuple[dict[RowKey, PreparedRow], Frozen]:
    frozen = _frozen_card(card)
    if len(frozen["cadence_grid"]) != 9:
        raise ValueError("The full-grid H400 tier requires nine frozen cadences")
    prepared = _prepare_rows(
        rows,
        model_seeds=frozen["model_seeds"],
        dataset_seeds=frozen["test_seeds"],
        cadences=frozen["cadence_grid"],
        horizon=TEST_HORIZON,
        allow_nonfinite=False,
    )
    return prepared, frozen


def validate_full_grid_h400_rows(
    rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
) -> None:
    """Require every finite arm x model x test-dataset x cadence H400 row."""

    _full_grid_h400_rows(rows, card)


def _endpoint_cube(
    prepared: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    endpoint: Endpoint,
) -> np.ndarray:
    shape = (
        2,
        len(frozen["model_seeds"]),
        len(frozen["test_seeds"]),
        len(frozen["cadence_grid"]),
    )
    cube = np.empty(shape, dtype=np.float64)
    for arm_index, arm in enumerate(ARMS):
        for model_index, model_seed in enumerate(frozen["model_seeds"]):
            for dataset_index, dataset_seed in enumerate(frozen["test_seeds"]):
                for cadence_index, cadence in enumerate(frozen["cadence_grid"]):
                    cube[arm_index, model_index, dataset_index, cadence_index] = (
                        endpoint(
                            prepared[(arm, model_seed, dataset_seed, cadence)]
                        )
                    )
    if not np.isfinite(cube).all() or np.any(cube < 0.0):
        raise FloatingPointError("A full-grid H400 endpoint cube is invalid")
    return cube


def _descriptive_accuracy_compute_frontier(
    endpoint_cube: np.ndarray,
    frozen: Frozen,
) -> list[dict[str, Any]]:
    """Describe accuracy versus exact encoder work without selecting a policy."""

    rows = []
    for cadence_index, cadence in enumerate(frozen["cadence_grid"]):
        dense_mean = float(endpoint_cube[0, :, :, cadence_index].mean())
        sparse_mean = float(endpoint_cube[1, :, :, cadence_index].mean())
        if dense_mean <= 0.0:
            raise ValueError("A frontier dense mean must be positive")
        refresh_count = (
            0
            if cadence == DIRECT
            else (TEST_HORIZON - 1) // int(cadence)
        )
        rows.append(
            {
                "cadence": cadence,
                "dense_arm_mean_mse": dense_mean,
                "sparse_arm_mean_mse": sparse_mean,
                "fixed_same_cadence_sparse_over_dense_ratio_of_arm_means": (
                    sparse_mean / dense_mean
                ),
                "fixed_same_cadence_relative_reduction_of_arm_means": (
                    1.0 - sparse_mean / dense_mean
                ),
                "refresh_count": refresh_count,
                "encoder_calls": 1 + refresh_count,
                "rollout_horizon_steps": TEST_HORIZON,
                "aggregation": (
                    "balanced_mean_over_ten_models_and_three_fixed_test_panels"
                ),
                "inference_role": "descriptive_only_not_used_for_selection",
            }
        )
    return rows


def _selection_aware_arm_swap(
    validation_seed_risks: np.ndarray,
    endpoint_seed_risks: np.ndarray,
    frozen: Frozen,
    *,
    permutation_chunk_size: int,
) -> dict[str, Any]:
    result = selection_aware_exact_sign_flip(
        validation_seed_risks,
        endpoint_seed_risks,
        frozen["cadence_grid"],
        chunk_size=permutation_chunk_size,
    )
    result.update(
        {
            "null_hypothesis": (
                "sharp_null_of_joint_dense_sparse_arm_exchangeability"
            ),
            "paired_exchangeability_assumption": (
                "within_each_paired_model_seed_dense_and_sparse_labels_are_"
                "exchangeable_for_the_complete_h200_validation_and_h400_test_"
                "cadence_vectors_under_the_sharp_null"
            ),
            "sharp_null_unit": "complete_paired_model_seed_pipeline",
            "fixed_three_test_panel_qualification": (
                "the_three_frozen_h400_test_panels_are_fixed_conditions_"
                "averaged_within_model_seed;_the_exact_p_value_is_conditional_"
                "on_them_and_does_not_infer_over_a_population_of_test_panels"
            ),
            "fixed_validation_panel_qualification": (
                "the_three_frozen_h200_validation_panels_are_also_held_fixed_"
                "inside_every_swapped_pipeline"
            ),
        }
    )
    return result


def _endpoint_summary(
    validation_seed_risks: np.ndarray,
    selected_indices: np.ndarray,
    prepared: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    endpoint: Endpoint,
    definition: str,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_chunk_size: int,
    permutation_chunk_size: int,
) -> dict[str, Any]:
    endpoint_cube = _endpoint_cube(prepared, frozen, endpoint)
    endpoint_seed_risks = endpoint_cube.mean(axis=2)
    return {
        "definition": definition,
        "point_test": _point_effects(endpoint_cube, selected_indices, frozen),
        "descriptive_accuracy_compute_frontier": (
            _descriptive_accuracy_compute_frontier(endpoint_cube, frozen)
        ),
        "selection_aware_pipeline_inference": {
            "exact_one_sided_studentized_arm_swap": (
                _selection_aware_arm_swap(
                    validation_seed_risks,
                    endpoint_seed_risks,
                    frozen,
                    permutation_chunk_size=permutation_chunk_size,
                )
            ),
            "paired_seed_bootstrap": selection_aware_paired_bootstrap(
                validation_seed_risks,
                endpoint_seed_risks,
                frozen["cadence_grid"],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
                chunk_size=bootstrap_chunk_size,
            ),
        },
        "selection_aware_within_arm_selected_vs_direct": (
            _within_arm_policy_bootstrap(
                validation_seed_risks,
                endpoint_seed_risks,
                frozen,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
                chunk_size=bootstrap_chunk_size,
            )
        ),
        "inference_role": "optional_full_grid_h400_durability_diagnostic",
    }


def _within_arm_policy_bootstrap(
    validation_seed_risks: np.ndarray,
    endpoint_seed_risks: np.ndarray,
    frozen: Frozen,
    *,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    repetitions = _integer(replicates, name="bootstrap replicates")
    random_seed = _integer(seed, name="bootstrap seed")
    chunk = _integer(chunk_size, name="bootstrap chunk size")
    if repetitions <= 0 or random_seed < 0 or chunk <= 0:
        raise ValueError("Bootstrap count/chunk must be positive and seed nonnegative")
    direct_index = frozen["cadence_grid"].index(DIRECT)
    report: dict[str, Any] = {}
    for arm_index, arm in enumerate(("dense", "sparse")):
        rng = np.random.default_rng(random_seed)
        point_index = int(
            _select_indices(
                validation_seed_risks[arm_index].mean(axis=0),
                frozen["cadence_grid"],
            )
        )
        direct = endpoint_seed_risks[arm_index, :, direct_index]
        selected = endpoint_seed_risks[arm_index, :, point_index]
        if direct.mean() <= 0.0:
            raise ValueError("A within-arm H400 direct mean is nonpositive")
        reductions = np.empty(repetitions, dtype=np.float64)
        counts = np.zeros(len(frozen["cadence_grid"]), dtype=np.int64)
        written = 0
        while written < repetitions:
            size = min(chunk, repetitions - written)
            indices = rng.integers(0, 10, size=(size, 10))
            choices = _select_indices(
                validation_seed_risks[arm_index][indices].mean(axis=1),
                frozen["cadence_grid"],
            )
            selected_values = endpoint_seed_risks[arm_index][
                indices, choices[:, None]
            ]
            direct_values = endpoint_seed_risks[arm_index][indices, direct_index]
            direct_means = direct_values.mean(axis=1)
            if np.any(direct_means <= 0.0):
                raise ValueError("A within-arm H400 bootstrap baseline is nonpositive")
            reductions[written:written + size] = (
                1.0 - selected_values.mean(axis=1) / direct_means
            )
            counts += np.bincount(choices, minlength=len(counts))
            written += size
        lower, upper = np.quantile(reductions, (0.025, 0.975))
        report[arm] = {
            "selected_cadence": frozen["cadence_grid"][point_index],
            "heldout_point_relative_reduction": float(
                1.0 - selected.mean() / direct.mean()
            ),
            "heldout_point_selected_seed_wins": int(np.sum(selected < direct)),
            "selection_aware_bootstrap_ci95_lower": float(lower),
            "selection_aware_bootstrap_ci95_upper": float(upper),
            "selection_frequencies": [
                {
                    "cadence": cadence,
                    "count": int(counts[index]),
                    "frequency": float(counts[index] / repetitions),
                }
                for index, cadence in enumerate(frozen["cadence_grid"])
            ],
            "replicates": repetitions,
            "seed": random_seed,
            "selector_rerun_for_every_replicate": True,
            "inference_role": "selection_aware_within_arm_h400_policy_bootstrap",
        }
    return report


def summarize_full_grid_h400_pipeline(
    validation_rows: Sequence[Mapping[str, Any]],
    h400_test_rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_chunk_size: int = 2_000,
    permutation_chunk_size: int = 128,
) -> dict[str, Any]:
    """Analyze both H400 endpoints with selection refit only on H200 data."""

    validation, frozen = _validation_rows(validation_rows, card)
    prepared, h400_frozen = _full_grid_h400_rows(h400_test_rows, card)
    if h400_frozen != frozen:
        raise AssertionError("H200 validation and H400 test rosters disagree")
    validation_cube = _risk_cube(
        validation,
        frozen,
        dataset_key="validation_seeds",
    )
    validation_seed_risks = validation_cube.mean(axis=2)
    selected_indices, selected, candidate_scores = _point_selection(
        validation_seed_risks,
        frozen["cadence_grid"],
    )
    endpoints = {}
    for offset, (name, endpoint, definition) in enumerate(H400_ENDPOINTS):
        endpoints[name] = _endpoint_summary(
            validation_seed_risks,
            selected_indices,
            prepared,
            frozen,
            endpoint,
            definition,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + offset,
            bootstrap_chunk_size=bootstrap_chunk_size,
            permutation_chunk_size=permutation_chunk_size,
        )
    return {
        "selected_cadences_from_h200_validation": selected,
        "h200_validation_candidate_scores": candidate_scores,
        "endpoints": endpoints,
        "validation_selector_horizon_steps": 200,
        "evaluated_h400_horizon_steps": TEST_HORIZON,
        "cadence_grid": list(frozen["cadence_grid"]),
        "required_full_grid_row_count": (
            2
            * len(frozen["model_seeds"])
            * len(frozen["test_seeds"])
            * len(frozen["cadence_grid"])
        ),
        "h400_outcomes_used_for_cadence_selection": False,
        "selector_rerun_for_every_arm_swap_and_bootstrap_replicate": True,
        "complete_finite_full_grid_required": True,
        "inference_unit": (
            "ten_paired_model_seeds_after_three_test_dataset_average"
        ),
        "selection_policy": (
            "recipe_level_cadence_selected_only_from_h200_validation;_"
            "h400_outcomes_never_select_cadence"
        ),
        "failure_policy": (
            "any_missing_or_nonfinite_cell_suppresses_this_full_grid_summary;_"
            "other_h400_tiers_follow_the_frozen_prediction_card"
        ),
    }
