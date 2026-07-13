"""Metric and aggregation helpers for benchmark results."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def nrmse(pred: np.ndarray, truth: np.ndarray, train_mean: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones_like(truth, dtype=np.float32)
    diff = (pred - truth) * mask
    centered = (truth - train_mean.reshape(*(1 for _ in range(truth.ndim - 1)), -1)) * mask
    num = float(np.sum(diff * diff))
    den = float(np.sum(centered * centered) + 1e-8)
    return float(np.sqrt(num / den))


def relative_l2(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones_like(truth, dtype=np.float32)
    diff = (pred - truth) * mask
    return float(np.linalg.norm(diff.reshape(-1)) / (np.linalg.norm((truth * mask).reshape(-1)) + 1e-8))


def temporal_corr(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        pred = pred * mask
        truth = truth * mask
    p = pred.reshape(-1)
    t = truth.reshape(-1)
    if p.size < 2 or np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return float("nan")
    return float(np.corrcoef(p, t)[0, 1])


def spectrum_error(pred: np.ndarray, truth: np.ndarray) -> float:
    p = np.abs(np.fft.rfft(pred.reshape(pred.shape[0], -1), axis=0)).mean(axis=1)
    t = np.abs(np.fft.rfft(truth.reshape(truth.shape[0], -1), axis=0)).mean(axis=1)
    return float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8))


def autocorr_error(pred: np.ndarray, truth: np.ndarray, max_lag: int = 10) -> float:
    def ac(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(x.shape[0], -1)
        flat = flat - flat.mean(axis=0, keepdims=True)
        denom = np.sum(flat * flat, axis=0) + 1e-8
        vals = []
        for lag in range(1, min(max_lag, flat.shape[0] - 1) + 1):
            vals.append(np.mean(np.sum(flat[:-lag] * flat[lag:], axis=0) / denom))
        return np.asarray(vals, dtype=np.float32)

    a = ac(pred)
    b = ac(truth)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))


def fit_percent(pred: np.ndarray, truth: np.ndarray) -> float:
    centered = truth - np.nanmean(truth)
    return float(100.0 * (1.0 - np.linalg.norm((truth - pred).reshape(-1)) / (np.linalg.norm(centered.reshape(-1)) + 1e-8)))


def valid_prediction_time(pred: np.ndarray, truth: np.ndarray, horizons: Sequence[int], threshold: float = 0.4) -> Tuple[int, bool]:
    for horizon in horizons:
        h = min(int(horizon), pred.shape[0])
        err = relative_l2(pred[:h], truth[:h])
        if err > threshold:
            return int(horizon), False
    return int(max(horizons)), True


def metric_rows_for_rollout(
    *,
    pred: np.ndarray,
    truth: np.ndarray,
    train_mean: np.ndarray,
    horizons: Sequence[int],
    mask: np.ndarray | None = None,
) -> List[Tuple[int, str, float]]:
    rows: List[Tuple[int, str, float]] = []
    for horizon in horizons:
        h = min(int(horizon), pred.shape[0], truth.shape[0])
        if h < 1:
            continue
        local_mask = None if mask is None else mask[:h]
        p = pred[:h]
        t = truth[:h]
        rows.extend(
            [
                (int(horizon), "nrmse", nrmse(p, t, train_mean, local_mask)),
                (int(horizon), "relative_l2", relative_l2(p, t, local_mask)),
                (int(horizon), "temporal_corr", temporal_corr(p, t, local_mask)),
                (int(horizon), "power_spectrum_error", spectrum_error(p, t)),
                (int(horizon), "autocorr_error", autocorr_error(p, t)),
                (int(horizon), "max_abs_error", float(np.nanmax(np.abs(p - t)))),
            ]
        )
    vpt, never_crossed = valid_prediction_time(pred, truth, horizons)
    rows.append((0, "valid_prediction_time", float(vpt)))
    rows.append((0, "valid_prediction_never_crossed", float(never_crossed)))
    return rows


def bootstrap_ci(values: Sequence[float], *, n_resamples: int, seed: int) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_resamples), dtype=np.float64)
    for i in range(int(n_resamples)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_rows(raw_rows: Iterable[Mapping[str, object]], *, n_resamples: int = 2000, seed: int = 0) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    keys = ["benchmark", "condition", "model", "split", "horizon", "latent_dim", "sparsity_coefficient", "metric_name"]
    for row in raw_rows:
        try:
            value = float(row["metric_value"])
        except (KeyError, TypeError, ValueError):
            continue
        groups[tuple(row.get(key) for key in keys)].append(value)

    summaries: List[Dict[str, object]] = []
    for key_values, values in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        lo, hi = bootstrap_ci(finite, n_resamples=n_resamples, seed=seed)
        summary = {key: value for key, value in zip(keys, key_values)}
        summary.update(
            {
                "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
                "std": float(np.nanstd(arr, ddof=1)) if finite.size > 1 else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
                "n": int(finite.size),
                "bootstrap_resamples": int(n_resamples),
            }
        )
        summaries.append(summary)
    return summaries


def paired_bootstrap_difference(
    rows: Iterable[Mapping[str, object]],
    *,
    model_a: str,
    model_b: str,
    metric_name: str,
    n_resamples: int,
    seed: int,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, object, object, object], Dict[str, Dict[object, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("metric_name") != metric_name:
            continue
        model = str(row.get("model"))
        if model not in {model_a, model_b}:
            continue
        key = (row.get("benchmark"), row.get("condition"), row.get("split"), row.get("horizon"))
        grouped[key][model][row.get("trajectory_identifier")] = float(row.get("metric_value"))

    out: List[Dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for key, per_model in grouped.items():
        ids = sorted(set(per_model.get(model_a, {})) & set(per_model.get(model_b, {})))
        if not ids:
            continue
        diff = np.asarray([per_model[model_a][i] - per_model[model_b][i] for i in ids], dtype=np.float64)
        boot = np.empty(int(n_resamples), dtype=np.float64)
        for j in range(int(n_resamples)):
            boot[j] = rng.choice(diff, size=diff.size, replace=True).mean()
        out.append(
            {
                "benchmark": key[0],
                "condition": key[1],
                "split": key[2],
                "horizon": key[3],
                "metric_name": f"{metric_name}_paired_difference_{model_a}_minus_{model_b}",
                "mean": float(diff.mean()),
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "n_pairs": int(diff.size),
            }
        )
    return out
