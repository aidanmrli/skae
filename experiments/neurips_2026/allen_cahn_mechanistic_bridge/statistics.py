"""Crossed model-seed and dataset-seed reductions for bridge evidence."""

from __future__ import annotations

from typing import Callable

import numpy as np


def _validate(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not 8 <= array.shape[0] <= 10:
        raise ValueError(
            f"Expected crossed [8--10 model, 3 dataset] cells, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise FloatingPointError("Crossed estimand contains nonfinite values")
    return array


def two_way_bootstrap(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    reducer: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    array = _validate(values)
    rng = np.random.default_rng(int(seed))
    output = np.empty(int(replicates), dtype=np.float64)
    reducer = reducer or (lambda sample: sample.mean(axis=(1, 2)))
    chunk = 5000
    for start in range(0, int(replicates), chunk):
        stop = min(int(replicates), start + chunk)
        count = stop - start
        model_index = rng.integers(0, array.shape[0], size=(count, array.shape[0]))
        data_index = rng.integers(0, array.shape[1], size=(count, array.shape[1]))
        sampled = array[model_index[:, :, None], data_index[:, None, :]]
        output[start:stop] = reducer(sampled)
    return output


def difference_summary(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    candidate_array, control_array = _validate(candidate), _validate(control)
    if candidate_array.shape != control_array.shape:
        raise ValueError("Candidate and control crossed axes differ")
    difference = candidate_array - control_array
    draws = two_way_bootstrap(difference, replicates=replicates, seed=seed)
    model_means = difference.mean(axis=1)
    dataset_means = difference.mean(axis=0)
    return {
        "difference_mean": float(difference.mean()),
        "bootstrap_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        "model_seed_candidate_wins": int(np.sum(model_means > 0.0)),
        "dataset_seed_candidate_wins": int(np.sum(dataset_means > 0.0)),
        "per_model_seed_difference": model_means.tolist(),
        "per_dataset_seed_difference": dataset_means.tolist(),
    }


def ratio_summary(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    replicates: int,
    seed: int,
    lower_is_better: bool = True,
) -> dict[str, object]:
    numerator_array, denominator_array = _validate(numerator), _validate(denominator)
    if numerator_array.shape != denominator_array.shape:
        raise ValueError("Numerator and denominator crossed axes differ")
    if np.any(denominator_array <= 0.0):
        raise ValueError("Ratio denominator must be positive")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    chunk = 5000
    for start in range(0, int(replicates), chunk):
        stop = min(int(replicates), start + chunk)
        count = stop - start
        model_count, dataset_count = numerator_array.shape
        models = rng.integers(0, model_count, size=(count, model_count))
        datasets = rng.integers(0, dataset_count, size=(count, dataset_count))
        sampled_numerator = numerator_array[models[:, :, None], datasets[:, None, :]]
        sampled_denominator = denominator_array[models[:, :, None], datasets[:, None, :]]
        draws[start:stop] = sampled_numerator.mean((1, 2)) / sampled_denominator.mean((1, 2))
    per_model = numerator_array.mean(1) / denominator_array.mean(1)
    per_dataset = numerator_array.mean(0) / denominator_array.mean(0)
    wins_model = per_model < 1.0 if lower_is_better else per_model > 1.0
    wins_dataset = per_dataset < 1.0 if lower_is_better else per_dataset > 1.0
    return {
        "ratio_of_cell_means": float(numerator_array.mean() / denominator_array.mean()),
        "bootstrap_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
        "model_seed_candidate_wins": int(wins_model.sum()),
        "dataset_seed_candidate_wins": int(wins_dataset.sum()),
        "per_model_seed_ratio": per_model.tolist(),
        "per_dataset_seed_ratio": per_dataset.tolist(),
    }
