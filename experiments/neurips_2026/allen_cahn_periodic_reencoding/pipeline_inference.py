"""Selection-aware inference for the prospective H200 periodic pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from experiments.neurips_2026.allen_cahn_periodic_reencoding.numeric_serialization import (
    json_safe_statistic,
)

from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    ARMS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DIRECT,
    VALIDATION_HORIZON,
    Cadence,
    Frozen,
    PreparedRow,
    RowKey,
    _frozen_card,
    _integer,
    _prepare_rows,
    _tie_key,
    exact_one_sided_studentized_sign_flip,
    paired_ratio_bootstrap,
)


def _full_h200_rows(
    validation_rows: Sequence[Mapping[str, Any]],
    primary_test_rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
) -> tuple[dict[RowKey, PreparedRow], dict[RowKey, PreparedRow], Frozen]:
    frozen = _frozen_card(card)
    common = {
        "model_seeds": frozen["model_seeds"],
        "cadences": frozen["cadence_grid"],
        "horizon": VALIDATION_HORIZON,
        "allow_nonfinite": False,
    }
    validation = _prepare_rows(
        validation_rows,
        dataset_seeds=frozen["validation_seeds"],
        **common,
    )
    test = _prepare_rows(
        primary_test_rows,
        dataset_seeds=frozen["test_seeds"],
        **common,
    )
    return validation, test, frozen


def _risk_cube(
    prepared: Mapping[RowKey, PreparedRow],
    frozen: Frozen,
    *,
    dataset_key: str,
) -> np.ndarray:
    datasets = frozen[dataset_key]
    cube = np.empty(
        (len(ARMS), len(frozen["model_seeds"]), len(datasets),
         len(frozen["cadence_grid"])),
        dtype=np.float64,
    )
    for arm_index, arm in enumerate(ARMS):
        for model_index, model in enumerate(frozen["model_seeds"]):
            for dataset_index, dataset in enumerate(datasets):
                for cadence_index, cadence in enumerate(frozen["cadence_grid"]):
                    cube[arm_index, model_index, dataset_index, cadence_index] = (
                        prepared[(arm, model, dataset, cadence)][1][-1]
                    )
    if not np.isfinite(cube).all() or np.any(cube < 0.0):
        raise FloatingPointError("A full-grid H200 risk cube is invalid")
    return cube


def _preference(grid: tuple[Cadence, ...]) -> np.ndarray:
    return np.asarray(sorted(range(len(grid)), key=lambda index: _tie_key(grid[index])))


def _select_indices(risks: np.ndarray, grid: tuple[Cadence, ...]) -> np.ndarray:
    """Select exact minima on the final axis using the frozen tie ordering."""

    values = np.asarray(risks, dtype=np.float64)
    if values.shape[-1] != len(grid) or not np.isfinite(values).all():
        raise ValueError("Cadence selection requires a finite complete risk vector")
    preference = _preference(grid)
    ordered_choice = np.argmin(values[..., preference], axis=-1)
    return preference[ordered_choice]


def _studentized(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=-1)
    scales = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
    result = np.zeros_like(means, dtype=np.float64)
    np.divide(means, scales, out=result, where=scales > 0)
    result[(scales == 0) & (means > 0)] = np.inf
    result[(scales == 0) & (means < 0)] = -np.inf
    return result


def _frequency_rows(counts: np.ndarray, grid: tuple[Cadence, ...], total: int
                    ) -> list[dict[str, Any]]:
    return [
        {
            "cadence": cadence,
            "count": int(counts[index]),
            "frequency": float(counts[index] / total),
        }
        for index, cadence in enumerate(grid)
    ]


def _point_selection(
    validation_seed_risks: np.ndarray,
    grid: tuple[Cadence, ...],
) -> tuple[np.ndarray, dict[str, Cadence], dict[str, list[dict[str, Any]]]]:
    arm_scores = validation_seed_risks.mean(axis=1)
    indices = _select_indices(arm_scores, grid)
    selected = {arm: grid[int(indices[index])] for index, arm in enumerate(ARMS)}
    scores = {
        arm: [
            {"cadence": cadence, "h200_cumulative_field_mse": float(value)}
            for cadence, value in zip(grid, arm_scores[arm_index])
        ]
        for arm_index, arm in enumerate(ARMS)
    }
    return indices, selected, scores


def _selected_seed_values(test_seed_risks: np.ndarray,
                          selected_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seeds = np.arange(test_seed_risks.shape[1])
    dense = test_seed_risks[0, seeds, int(selected_indices[0])]
    sparse = test_seed_risks[1, seeds, int(selected_indices[1])]
    return dense, sparse


def selection_aware_exact_sign_flip(
    validation_seed_risks: np.ndarray,
    test_seed_risks: np.ndarray,
    grid: tuple[Cadence, ...],
    *,
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Enumerate joint seed-wise arm swaps and rerun both selectors."""

    if validation_seed_risks.shape != (2, 10, len(grid)):
        raise ValueError("Validation risks must have shape [2,10,cadence]")
    if test_seed_risks.shape != validation_seed_risks.shape:
        raise ValueError("Test and validation seed-risk shapes must match")
    chunk = _integer(chunk_size, name="permutation chunk size")
    if chunk <= 0:
        raise ValueError("Permutation chunk size must be positive")
    point_indices, _, _ = _point_selection(validation_seed_risks, grid)
    point_dense, point_sparse = _selected_seed_values(test_seed_risks, point_indices)
    observed = float(_studentized((point_dense - point_sparse)[None, :])[0])
    exceedances = 0
    counts = np.zeros((2, len(grid)), dtype=np.int64)
    joint_counts = np.zeros((len(grid), len(grid)), dtype=np.int64)
    for start in range(0, 2**10, chunk):
        stop = min(2**10, start + chunk)
        integers = np.arange(start, stop, dtype=np.uint16)[:, None]
        swapped = ((integers >> np.arange(10, dtype=np.uint16)) & 1).astype(bool)
        mask = swapped[:, :, None]
        validation_dense = np.where(
            mask, validation_seed_risks[1], validation_seed_risks[0]
        )
        validation_sparse = np.where(
            mask, validation_seed_risks[0], validation_seed_risks[1]
        )
        dense_indices = _select_indices(validation_dense.mean(axis=1), grid)
        sparse_indices = _select_indices(validation_sparse.mean(axis=1), grid)
        test_dense = np.where(mask, test_seed_risks[1], test_seed_risks[0])
        test_sparse = np.where(mask, test_seed_risks[0], test_seed_risks[1])
        batch = np.arange(stop - start)[:, None]
        seeds = np.arange(10)[None, :]
        dense_values = test_dense[batch, seeds, dense_indices[:, None]]
        sparse_values = test_sparse[batch, seeds, sparse_indices[:, None]]
        statistics = _studentized(dense_values - sparse_values)
        exceedances += int(np.sum(statistics >= observed))
        counts[0] += np.bincount(dense_indices, minlength=len(grid))
        counts[1] += np.bincount(sparse_indices, minlength=len(grid))
        np.add.at(joint_counts, (dense_indices, sparse_indices), 1)
    observed_pair_count = int(joint_counts[int(point_indices[0]), int(point_indices[1])])
    return {
        **json_safe_statistic(observed, name="observed_studentized_statistic"),
        "one_sided_exact_p": float(exceedances / 2**10),
        "exceedances_literal_greater_equal": exceedances,
        "enumerated_seedwise_arm_swaps": 2**10,
        "comparison": "T_swapped >= T_observed_literal_no_tolerance",
        "swap_unit": "paired_seed_entire_validation_and_test_cadence_vectors",
        "null_and_assumption": (
            "sharp joint arm-exchangeability null for each paired checkpoint seed"
        ),
        "conditioning": (
            "conditional on the three fixed validation and three fixed test IC panels"
        ),
        "enumeration_exactness_boundary": (
            "all 2^10 swaps are enumerated; null calibration is not assumption-free"
        ),
        "selector_rerun_for_every_swap": True,
        "selection_frequencies": {
            arm: _frequency_rows(counts[index], grid, 2**10)
            for index, arm in enumerate(ARMS)
        },
        "joint_selection_pair_counts": [
            {
                "dense_cadence": grid[dense_index],
                "sparse_cadence": grid[sparse_index],
                "count": int(joint_counts[dense_index, sparse_index]),
            }
            for dense_index in range(len(grid))
            for sparse_index in range(len(grid))
            if joint_counts[dense_index, sparse_index] > 0
        ],
        "swaps_changing_point_selection": int(2**10 - observed_pair_count),
    }


def selection_aware_paired_bootstrap(
    validation_seed_risks: np.ndarray,
    test_seed_risks: np.ndarray,
    grid: tuple[Cadence, ...],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    chunk_size: int = 2_000,
) -> dict[str, Any]:
    """Resample paired seed pipelines, rerunning arm selectors each time."""

    if validation_seed_risks.shape != (2, 10, len(grid)):
        raise ValueError("Validation risks must have shape [2,10,cadence]")
    if test_seed_risks.shape != validation_seed_risks.shape:
        raise ValueError("Test and validation seed-risk shapes must match")
    repetitions = _integer(replicates, name="bootstrap replicates")
    random_seed = _integer(seed, name="bootstrap seed")
    chunk = _integer(chunk_size, name="bootstrap chunk size")
    if repetitions <= 0 or random_seed < 0 or chunk <= 0:
        raise ValueError("Bootstrap count/chunk must be positive and seed nonnegative")
    generator = np.random.default_rng(random_seed)
    reductions = np.empty(repetitions, dtype=np.float64)
    counts = np.zeros((2, len(grid)), dtype=np.int64)
    write = 0
    while write < repetitions:
        size = min(chunk, repetitions - write)
        indices = generator.integers(0, 10, size=(size, 10))
        dense_indices = _select_indices(
            validation_seed_risks[0][indices].mean(axis=1), grid
        )
        sparse_indices = _select_indices(
            validation_seed_risks[1][indices].mean(axis=1), grid
        )
        dense_values = test_seed_risks[0][indices, dense_indices[:, None]]
        sparse_values = test_seed_risks[1][indices, sparse_indices[:, None]]
        dense_means = dense_values.mean(axis=1)
        sparse_means = sparse_values.mean(axis=1)
        if np.any(dense_means <= 0.0) or not np.isfinite([dense_means, sparse_means]).all():
            raise FloatingPointError("A selection-aware bootstrap replicate is invalid")
        reductions[write:write + size] = 1.0 - sparse_means / dense_means
        counts[0] += np.bincount(dense_indices, minlength=len(grid))
        counts[1] += np.bincount(sparse_indices, minlength=len(grid))
        write += size
    point_indices, _, _ = _point_selection(validation_seed_risks, grid)
    dense, sparse = _selected_seed_values(test_seed_risks, point_indices)
    if dense.mean() <= 0.0:
        raise ValueError("Point dense test mean must be positive")
    lower, upper = np.quantile(reductions, (0.025, 0.975))
    return {
        "relative_reduction_of_arm_means": float(1.0 - sparse.mean() / dense.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "replicates": repetitions,
        "seed": random_seed,
        "resampling_unit": "paired_model_seed_entire_validation_and_test_pipeline",
        "selector_rerun_for_every_replicate": True,
        "selection_frequencies": {
            arm: _frequency_rows(counts[index], grid, repetitions)
            for index, arm in enumerate(ARMS)
        },
    }


def _point_effects(
    test_cube: np.ndarray,
    selected_indices: np.ndarray,
    frozen: Frozen,
) -> dict[str, Any]:
    test_seed = test_cube.mean(axis=2)
    dense, sparse = _selected_seed_values(test_seed, selected_indices)
    dense_mean, sparse_mean = float(dense.mean()), float(sparse.mean())
    if dense_mean <= 0.0:
        raise ValueError("Dense point-test mean must be positive")
    dataset_effects = []
    for dataset_index, dataset in enumerate(frozen["test_seeds"]):
        dense_dataset = test_cube[0, :, dataset_index, int(selected_indices[0])]
        sparse_dataset = test_cube[1, :, dataset_index, int(selected_indices[1])]
        baseline = float(dense_dataset.mean())
        if baseline <= 0.0:
            raise ValueError("A dense per-dataset mean must be positive")
        dataset_effects.append(
            {
                "dataset_seed": int(dataset),
                "dense_mean": baseline,
                "sparse_mean": float(sparse_dataset.mean()),
                "relative_reduction_of_arm_means": float(
                    1.0 - sparse_dataset.mean() / baseline
                ),
                "sparse_seed_wins": int(np.sum(sparse_dataset < dense_dataset)),
            }
        )
    return {
        "dense_paired_seed_values": dense.tolist(),
        "sparse_paired_seed_values": sparse.tolist(),
        "dense_mean": dense_mean,
        "sparse_mean": sparse_mean,
        "sparse_over_dense_ratio_of_arm_means": sparse_mean / dense_mean,
        "relative_reduction_of_arm_means": 1.0 - sparse_mean / dense_mean,
        "sparse_seed_wins": int(np.sum(sparse < dense)),
        "per_dataset_effects": dataset_effects,
    }


def _within_arm_policy(
    test_seed_risks: np.ndarray,
    selected_indices: np.ndarray,
    frozen: Frozen,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    direct_index = frozen["cadence_grid"].index(DIRECT)
    result = {}
    for arm_index, arm in enumerate(ARMS):
        selected_index = int(selected_indices[arm_index])
        direct = test_seed_risks[arm_index, :, direct_index]
        selected = test_seed_risks[arm_index, :, selected_index]
        direct_mean, selected_mean = float(direct.mean()), float(selected.mean())
        if direct_mean <= 0.0:
            raise ValueError("Within-arm direct mean must be positive")
        exact = exact_one_sided_studentized_sign_flip(direct - selected)
        bootstrap = paired_ratio_bootstrap(
            direct,
            selected,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + arm_index,
        )
        identity = selected_index == direct_index
        if identity:
            exact.update(
                observed_studentized_statistic=0.0,
                observed_studentized_statistic_status="finite",
                one_sided_exact_p=1.0,
            )
            bootstrap.update(
                relative_reduction_of_arm_means=0.0,
                ci95_lower=0.0,
                ci95_upper=0.0,
            )
        result[arm] = {
            "selected_cadence": frozen["cadence_grid"][selected_index],
            "direct_paired_seed_values": direct.tolist(),
            "selected_paired_seed_values": selected.tolist(),
            "direct_mean": direct_mean,
            "selected_mean": selected_mean,
            "relative_reduction_of_arm_means": (
                0.0 if identity else 1.0 - selected_mean / direct_mean
            ),
            "selected_seed_wins": int(np.sum(selected < direct)),
            "exact_one_sided_studentized_sign_flip": exact,
            "paired_ratio_bootstrap": bootstrap,
            "identity_comparison_selected_is_direct": identity,
        }
    return result


def summarize_pipeline_inference(
    validation_rows: Sequence[Mapping[str, Any]],
    primary_test_rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_chunk_size: int = 2_000,
    permutation_chunk_size: int = 128,
) -> dict[str, Any]:
    """Run point, conditional, and full selection-aware H200 inference."""

    validation, test, frozen = _full_h200_rows(
        validation_rows, primary_test_rows, card
    )
    validation_cube = _risk_cube(
        validation, frozen, dataset_key="validation_seeds"
    )
    test_cube = _risk_cube(test, frozen, dataset_key="test_seeds")
    validation_seed = validation_cube.mean(axis=2)
    test_seed = test_cube.mean(axis=2)
    indices, selected, candidate_scores = _point_selection(
        validation_seed, frozen["cadence_grid"]
    )
    point = _point_effects(test_cube, indices, frozen)
    dense = np.asarray(point["dense_paired_seed_values"], dtype=np.float64)
    sparse = np.asarray(point["sparse_paired_seed_values"], dtype=np.float64)
    return {
        "selected_cadences_from_validation": selected,
        "validation_candidate_scores": candidate_scores,
        "primary_point_test": point,
        "conditional_fixed_selection_inference": {
            "exact_one_sided_studentized_sign_flip": (
                exact_one_sided_studentized_sign_flip(dense - sparse)
            ),
            "paired_ratio_bootstrap": paired_ratio_bootstrap(
                dense,
                sparse,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
            "warning": "conditions_on_validation_selected_cadences",
        },
        "selection_aware_pipeline_inference": {
            "exact_one_sided_studentized_sign_flip": selection_aware_exact_sign_flip(
                validation_seed,
                test_seed,
                frozen["cadence_grid"],
                chunk_size=permutation_chunk_size,
            ),
            "paired_seed_bootstrap": selection_aware_paired_bootstrap(
                validation_seed,
                test_seed,
                frozen["cadence_grid"],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
                chunk_size=bootstrap_chunk_size,
            ),
        },
        "within_arm_selected_vs_direct_h200": _within_arm_policy(
            test_seed,
            indices,
            frozen,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 10_000,
        ),
        "inference_unit": "ten_paired_model_seeds_after_three_dataset_average",
        "selection_policy": "validation_only_recipe_level_exact_minimum",
        "test_selection_forbidden": True,
    }
