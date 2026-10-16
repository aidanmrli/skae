"""Aggregation and significance tests used in the paper's tables."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats


def interquartile_mean(values: Iterable[float]) -> float:
    """Mean of the values between the 25th and 75th percentiles (inclusive).

    This is the estimator used for the paper's tables: percentiles use
    NumPy's default linear interpolation and values equal to a percentile are
    retained. Non-finite values are dropped; fewer than four values are averaged.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    if array.size < 4:
        return float(np.mean(array))
    low, high = np.percentile(array, [25, 75])
    return float(np.mean(array[(array >= low) & (array <= high)]))


def holm(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (NaNs are left untouched)."""
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    running = 0.0
    for rank, index in enumerate(finite[np.argsort(values[finite])]):
        running = max(running, min(1.0, (finite.size - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_wilcoxon_less(candidate: np.ndarray, baseline: np.ndarray) -> float:
    """One-sided signed-rank test that ``log10(candidate) < log10(baseline)`` (paired seeds).

    Pairs with a non-finite or non-positive value are dropped; at least four
    pairs are needed (otherwise NaN).
    """
    candidate, baseline = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    mask = np.isfinite(candidate) & np.isfinite(baseline) & (candidate > 0) & (baseline > 0)
    if mask.sum() < 4:
        return float("nan")
    diff = np.log10(candidate[mask]) - np.log10(baseline[mask])
    if np.all(diff == 0.0):
        return float("nan")
    return float(stats.wilcoxon(diff, alternative="less", zero_method="wilcox").pvalue)


def paired_wilcoxon_less_raw(candidate: np.ndarray, baseline: np.ndarray, zero_method: str = "zsplit") -> float:
    """One-sided signed-rank test that ``candidate < baseline`` on raw paired differences.

    For quantities that can be exactly zero, such as the support entropy
    ``H(B | F)``. Ties are kept: with ``zero_method="zsplit"`` their ranks are
    split evenly between the two rank sums, so every paired seed is used. At
    least four finite pairs are needed (otherwise NaN); if every difference is
    zero the p-value is 1.
    """
    candidate, baseline = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    mask = np.isfinite(candidate) & np.isfinite(baseline)
    if mask.sum() < 4:
        return float("nan")
    diff = candidate[mask] - baseline[mask]
    if np.all(diff == 0.0):
        return 1.0
    return float(stats.wilcoxon(diff, alternative="less", zero_method=zero_method).pvalue)
