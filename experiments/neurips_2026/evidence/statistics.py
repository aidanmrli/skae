"""Shared statistical summaries used by NeurIPS paper artifact builders."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats


IQM_PROPORTION_TO_CUT = 0.25
IQM_CONVENTION = "scipy.stats.trim_mean(proportiontocut=0.25)"


def interquartile_mean(values: Iterable[float]) -> float:
    """Return the Agarwal/rliable seed IQM using SciPy's trim convention.

    SciPy removes ``floor(0.25 * n)`` sorted observations from each tail.
    Thus a complete 15-seed cell averages its central nine observations.
    """

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(stats.trim_mean(array, proportiontocut=IQM_PROPORTION_TO_CUT))


def rowwise_interquartile_mean(values: np.ndarray) -> np.ndarray:
    """Vectorized SciPy-trim-mean equivalent for finite 2D bootstrap draws."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("rowwise_interquartile_mean expects a 2D array")
    if not np.isfinite(array).all():
        raise ValueError("rowwise_interquartile_mean requires finite values")
    tail = int(IQM_PROPORTION_TO_CUT * array.shape[1])
    ordered = np.sort(array, axis=1)
    retained = ordered[:, tail : array.shape[1] - tail if tail else None]
    return retained.mean(axis=1)
