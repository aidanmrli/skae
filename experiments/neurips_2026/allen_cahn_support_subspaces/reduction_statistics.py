"""Across-seed uncertainty helpers for the Allen--Cahn support-subspace audit."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np


def finite_tree(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def bootstrap_mean_interval(values: list[float], *, replicates: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, array.size, size=(int(replicates), array.size))
    statistics = array[samples].mean(axis=1)
    return [float(value) for value in np.quantile(statistics, (0.025, 0.975))]


def bootstrap_ratio_interval(
    numerator: list[float],
    denominator: list[float],
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    left = np.asarray(numerator, dtype=np.float64)
    right = np.asarray(denominator, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, left.size, size=(int(replicates), left.size))
    ratios = left[samples].mean(axis=1) / right[samples].mean(axis=1)
    return [float(value) for value in np.quantile(ratios, (0.025, 0.975))]


def _studentized(values: np.ndarray) -> float:
    scale = values.std(ddof=1) / math.sqrt(values.size)
    if scale <= 0:
        if values.mean() > 0:
            return math.inf
        if values.mean() < 0:
            return -math.inf
        return 0.0
    return float(values.mean() / scale)


def exact_max_t_adjusted_p(differences: dict[str, list[float]]) -> dict[str, float]:
    names = list(differences)
    matrix = np.stack([np.asarray(differences[name], dtype=np.float64) for name in names], axis=1)
    observed = np.asarray([_studentized(matrix[:, index]) for index in range(matrix.shape[1])])
    null_maxima = []
    for signs in itertools.product((-1.0, 1.0), repeat=matrix.shape[0]):
        signed = matrix * np.asarray(signs)[:, None]
        null_maxima.append(max(_studentized(signed[:, index]) for index in range(signed.shape[1])))
    null = np.asarray(null_maxima)
    return {
        name: float(np.sum(null >= observed[index]) / null.size)
        for index, name in enumerate(names)
    }


