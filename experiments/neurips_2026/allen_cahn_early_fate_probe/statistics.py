"""Frozen crossed-panel statistics for the Allen--Cahn early-fate probe."""

from __future__ import annotations

import itertools

import numpy as np

from .probes import classification_metrics


def two_way_bootstrap(
    differences: np.ndarray, *, replicates: int, seed: int
) -> tuple[float, float]:
    matrix = np.asarray(differences, dtype=np.float64)
    if matrix.shape != (10, 3) or not np.isfinite(matrix).all():
        raise ValueError("Frozen crossed contrast must have shape [10,3]")
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(replicates), dtype=np.float64)
    chunk_size = 5000
    for start in range(0, int(replicates), chunk_size):
        stop = min(start + chunk_size, int(replicates))
        count = stop - start
        model_draws = rng.integers(0, 10, size=(count, 10))
        dataset_draws = rng.integers(0, 3, size=(count, 3))
        sampled = matrix[model_draws[:, :, None], dataset_draws[:, None, :]]
        values[start:stop] = sampled.mean(axis=(1, 2))
    return tuple(float(item) for item in np.quantile(values, [0.025, 0.975]))


def exact_one_sided_sign_flip(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Sign-flip input must contain ten model-seed means")
    observed = float(values.mean())
    exceedances = 0
    for signs in itertools.product((-1.0, 1.0), repeat=10):
        statistic = float(np.mean(values * np.asarray(signs)))
        exceedances += statistic >= observed - 1e-15
    return float(exceedances / 1024.0)


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    total = values.size
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def contrast_summary(
    differences: np.ndarray, *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, object]:
    matrix = np.asarray(differences, dtype=np.float64)
    if matrix.shape != (10, 3):
        raise ValueError("Contrast panel must be [model_seed,dataset_seed]")
    model_means = matrix.mean(axis=1)
    dataset_means = matrix.mean(axis=0)
    interval = two_way_bootstrap(
        matrix, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    return {
        "mean_difference": float(matrix.mean()),
        "model_seed_differences": model_means.tolist(),
        "dataset_seed_differences": dataset_means.tolist(),
        "model_seed_wins": int(np.sum(model_means > 0.0)),
        "dataset_seed_wins": int(np.sum(dataset_means > 0.0)),
        "two_way_bootstrap_interval": list(interval),
        "one_sided_exact_sign_flip_p": exact_one_sided_sign_flip(model_means),
    }


def absolute_permutation_p(
    labels: list[np.ndarray],
    predictions: list[list[np.ndarray]],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Jointly permute each dataset target for all ten model predictions."""

    if len(labels) != 3 or len(predictions) != 10:
        raise ValueError("Absolute-null roster mismatch")
    for model_predictions in predictions:
        if len(model_predictions) != 3:
            raise ValueError("Absolute-null prediction roster mismatch")

    def statistic(targets: list[np.ndarray]) -> float:
        scores = []
        for model_predictions in predictions:
            for truth, pred in zip(targets, model_predictions):
                scores.append(classification_metrics(truth, pred)["balanced_accuracy"])
        return float(np.mean(scores))

    observed = statistic(labels)
    rng = np.random.default_rng(int(seed))
    exceedances = 0
    for _ in range(int(replicates)):
        permuted = [values[rng.permutation(values.size)] for values in labels]
        exceedances += statistic(permuted) >= observed - 1e-15
    return observed, float((exceedances + 1) / (int(replicates) + 1))
