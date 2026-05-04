#!/usr/bin/env python3
"""Build IQM tables, within-system tests, and plots for the Dysts dt x30 packet."""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FORECASTING_CSV = (
    ROOT
    / "results"
    / "dysts_dt30_basinblock_p256_seq10_100k_20260430"
    / "long_horizon_eval"
    / "collect"
    / "forecasting_rows.csv"
)
DEFAULT_FIG_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026"
DEFAULT_TABLE_DIR = DEFAULT_FIG_DIR / "_tables"

ROOTS = OrderedDict(
    [
        ("lista", {"display": "LISTA", "color": "#7B3294", "linestyle": "-"}),
        ("lista_bd", {"display": "LISTA-BD", "color": "#0072B2", "linestyle": "-"}),
        ("lista_sb", {"display": "LISTA-SB", "color": "#56B4E9", "linestyle": "-"}),
        ("sparse_mlp", {"display": "Sparse MLP", "color": "#009E73", "linestyle": "--"}),
        ("sparse_mlp_bd", {"display": "Sparse MLP-BD", "color": "#44AA99", "linestyle": "--"}),
        ("dense_mlp_tanh", {"display": "Dense MLP", "color": "#D55E00", "linestyle": "--"}),
    ]
)
BASELINE_ROOT = "dense_mlp_tanh"
DEFAULT_HORIZONS = [100, 500, 1000, 1500, 2000, 3000, 4000, 5000]
DEFAULT_BOOTSTRAP_REPS = 10000
DEFAULT_BOOTSTRAP_SEED = 20260501
DEFAULT_EXCLUDED_SYSTEMS = ("dysts:LorenzCoupled", "dysts:MultiChua")
FORECAST_YLABEL_FONTSIZE = 11.5
FORECAST_YTICK_LABELSIZE = 9.5

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def forecasting_performance_title(n_systems: int) -> str:
    return f"{n_systems}-system Dysts forecasting performance"


def title_with_note(title: str, title_note: str) -> str:
    note = title_note.strip()
    if not note:
        return title
    return f"{title} {note}"


def iqm(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        return float(np.mean(arr))
    lo, hi = np.percentile(arr, [25, 75])
    keep = arr[(arr >= lo) & (arr <= hi)]
    if keep.size == 0:
        return float(np.mean(arr))
    return float(np.mean(keep))


def mean_finite(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def row_iqm(values: np.ndarray) -> np.ndarray:
    """Compute the same percentile-trimmed IQM for each row of a 2D array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("row_iqm expects a 2D array")
    if arr.shape[1] == 0:
        return np.full(arr.shape[0], np.nan)
    if arr.shape[1] < 4:
        return np.mean(arr, axis=1)
    lo = np.percentile(arr, 25, axis=1)
    hi = np.percentile(arr, 75, axis=1)
    keep = (arr >= lo[:, None]) & (arr <= hi[:, None])
    counts = keep.sum(axis=1)
    sums = np.where(keep, arr, 0.0).sum(axis=1)
    means = np.mean(arr, axis=1)
    return np.divide(sums, counts, out=means, where=counts > 0)


def finite_positive(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr) & (arr > 0.0)]


def fixed_system_seed_bootstrap_mean(
    system_seed_values: list[np.ndarray],
    *,
    rng: np.random.Generator,
    n_reps: int,
) -> tuple[float, float, float]:
    """Bootstrap uncertainty in the cross-system mean from finite training seeds only."""
    if n_reps <= 0:
        return float("nan"), float("nan"), float("nan")
    valid = [np.asarray(values, dtype=float) for values in system_seed_values if len(values) > 0]
    if not valid:
        return float("nan"), float("nan"), float("nan")

    per_system_draws = []
    for values in valid:
        indices = rng.integers(0, values.size, size=(n_reps, values.size))
        per_system_draws.append(row_iqm(values[indices]))
    draws = np.mean(np.column_stack(per_system_draws), axis=1)

    return (
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
        float(np.std(draws, ddof=1)),
    )


def system_bootstrap_mean(
    system_values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_reps: int,
) -> tuple[float, float, float]:
    """Bootstrap the mean by resampling fixed per-system estimates."""
    if n_reps <= 0:
        return float("nan"), float("nan"), float("nan")
    valid = np.asarray(system_values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan")
    indices = rng.integers(0, valid.size, size=(n_reps, valid.size))
    draws = np.mean(valid[indices], axis=1)
    return (
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
        float(np.std(draws, ddof=1)),
    )


def fixed_system_seed_bootstrap_log_mean(
    system_seed_values: list[np.ndarray],
    *,
    rng: np.random.Generator,
    n_reps: int,
) -> tuple[float, float, float]:
    """Bootstrap seed uncertainty after summarizing each system in log-MSE space."""
    if n_reps <= 0:
        return float("nan"), float("nan"), float("nan")
    valid = [
        np.log10(np.asarray(values, dtype=float))
        for values in system_seed_values
        if len(values) > 0 and np.all(np.asarray(values, dtype=float) > 0.0)
    ]
    if not valid:
        return float("nan"), float("nan"), float("nan")

    per_system_draws = []
    for values in valid:
        indices = rng.integers(0, values.size, size=(n_reps, values.size))
        per_system_draws.append(row_iqm(values[indices]))
    draws_log10 = np.mean(np.column_stack(per_system_draws), axis=1)

    return (
        float(10.0 ** np.percentile(draws_log10, 2.5)),
        float(10.0 ** np.percentile(draws_log10, 97.5)),
        float(np.std(draws_log10, ddof=1)),
    )


def holm_correct(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [float("nan")] * n
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p_values[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def tex_number(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value >= 1000.0 or abs_value < 1e-3:
        exponent = math.floor(math.log10(abs_value))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.2f}{{\times}}10^{{{exponent}}}"
    decimals = max(2 - math.floor(math.log10(abs_value)), 0)
    return f"{value:.{decimals}f}"


def normalize_system_key(system: str) -> str:
    system = system.strip()
    if not system:
        return system
    if system.startswith("dysts:"):
        return system
    return f"dysts:{system}"


def load_rows(path: Path, *, excluded_systems: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[df["status"] == "complete"].copy()
    df = df[df["root_label"].isin(ROOTS)].copy()
    if excluded_systems:
        df = df[~df["system_key"].isin(excluded_systems)].copy()
    if df.empty:
        raise RuntimeError(f"No complete Dysts dt30 rows found in {path}")
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    return df


def infer_system_count(summary: pd.DataFrame) -> int:
    counts = pd.to_numeric(summary["n_systems"], errors="coerce")
    counts = counts[np.isfinite(counts)]
    if counts.empty:
        return 0
    return int(counts.max())


def per_system_summary(
    df: pd.DataFrame,
    horizons: list[int],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_system_rows = []
    summary_rows = []
    seed_rng = np.random.default_rng(bootstrap_seed)
    system_rng = np.random.default_rng(bootstrap_seed + 104729)
    log_seed_rng = np.random.default_rng(bootstrap_seed + 130363)
    for root_label, meta in ROOTS.items():
        root_df = df[df["root_label"] == root_label]
        for h in horizons:
            mse_col = f"h{h}_best_periodic_mean"
            cov_col = f"h{h}_best_periodic_full_finite_fraction"
            system_seed_values = []
            for system, grp in root_df.groupby("system_key", sort=True):
                mse_values = finite_positive(grp[mse_col])
                coverage_values = pd.to_numeric(grp[cov_col], errors="coerce").to_numpy(dtype=float)
                coverage_values = coverage_values[np.isfinite(coverage_values)]
                if mse_values.size == 0:
                    continue
                system_seed_values.append(mse_values)
                per_system_rows.append(
                    {
                        "root_label": root_label,
                        "display": meta["display"],
                        "system_key": system,
                        "horizon": h,
                        "n_seeds": int(mse_values.size),
                        "mse_iqm": iqm(mse_values),
                        "mse_log10_iqm": iqm(np.log10(mse_values)),
                        "mse_median": float(np.median(mse_values)),
                        "mse_q25": float(np.percentile(mse_values, 25)),
                        "mse_q75": float(np.percentile(mse_values, 75)),
                        "full_finite_iqm": iqm(coverage_values),
                        "full_finite_min": float(np.min(coverage_values)) if coverage_values.size else float("nan"),
                    }
                )

            horizon_rows = [r for r in per_system_rows if r["root_label"] == root_label and r["horizon"] == h]
            system_iqms = np.asarray([r["mse_iqm"] for r in horizon_rows], dtype=float)
            system_log10_iqms = np.asarray([r["mse_log10_iqm"] for r in horizon_rows], dtype=float)
            finite_coverages = np.asarray([r["full_finite_iqm"] for r in horizon_rows], dtype=float)
            ci_low, ci_high, boot_se = fixed_system_seed_bootstrap_mean(
                system_seed_values,
                rng=seed_rng,
                n_reps=bootstrap_reps,
            )
            system_ci_low, system_ci_high, system_boot_se = system_bootstrap_mean(
                system_iqms,
                rng=system_rng,
                n_reps=bootstrap_reps,
            )
            log_ci_low, log_ci_high, log_boot_se = fixed_system_seed_bootstrap_log_mean(
                system_seed_values,
                rng=log_seed_rng,
                n_reps=bootstrap_reps,
            )
            system_mean = mean_finite(system_iqms)
            log10_system_mean = mean_finite(system_log10_iqms)
            log_space_center = 10.0**log10_system_mean if math.isfinite(log10_system_mean) else float("nan")
            summary_rows.append(
                {
                    "root_label": root_label,
                    "display": meta["display"],
                    "horizon": h,
                    "n_systems": int(np.isfinite(system_iqms).sum()),
                    "cross_system_mean": system_mean,
                    "cross_system_iqm": system_mean,
                    "cross_system_iqm_legacy": iqm(system_iqms),
                    "cross_system_log10_iqm_mean": log10_system_mean,
                    "cross_system_log_iqm_geomean": log_space_center,
                    "system_q25": float(np.nanpercentile(system_iqms, 25)) if system_iqms.size else float("nan"),
                    "system_q75": float(np.nanpercentile(system_iqms, 75)) if system_iqms.size else float("nan"),
                    "system_median": float(np.nanmedian(system_iqms)) if system_iqms.size else float("nan"),
                    "seed_bootstrap_ci95_low": ci_low,
                    "seed_bootstrap_ci95_high": ci_high,
                    "seed_bootstrap_se": boot_se,
                    "system_bootstrap_ci95_low": system_ci_low,
                    "system_bootstrap_ci95_high": system_ci_high,
                    "system_bootstrap_se": system_boot_se,
                    "log_seed_bootstrap_ci95_low": log_ci_low,
                    "log_seed_bootstrap_ci95_high": log_ci_high,
                    "log_seed_bootstrap_se_log10": log_boot_se,
                    "seed_bootstrap_reps": int(bootstrap_reps),
                    "full_finite_mean": mean_finite(finite_coverages),
                    "full_finite_iqm": mean_finite(finite_coverages),
                }
            )
    return pd.DataFrame(per_system_rows), pd.DataFrame(summary_rows)


def paired_tests(df: pd.DataFrame, per_system: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df[df["root_label"] == BASELINE_ROOT].copy()
    base_indexed = base.set_index(["system_key", "seed"], drop=False)
    systems = sorted(df["system_key"].dropna().unique().tolist())
    detailed_rows = []
    summary_rows = []

    ratio_lookup: dict[tuple[str, str, int], float] = {}
    for _, row in per_system.iterrows():
        ratio_lookup[(row["root_label"], row["system_key"], int(row["horizon"]))] = float(row["mse_iqm"])

    for root_label, meta in ROOTS.items():
        if root_label == BASELINE_ROOT:
            for h in horizons:
                summary_rows.append(
                    {
                        "root_label": root_label,
                        "display": meta["display"],
                        "horizon": h,
                        "K": np.nan,
                        "N": len(systems),
                        "systems_with_iqm_ratio_lt_1": np.nan,
                        "sign_test_p": np.nan,
                    }
                )
            continue

        cand = df[df["root_label"] == root_label].set_index(["system_key", "seed"], drop=False)
        for h in horizons:
            mse_col = f"h{h}_best_periodic_mean"
            test_rows = []
            for system in systems:
                deltas = []
                paired_seed_rows = []
                for seed in sorted(set(cand.index.get_level_values("seed")) | set(base_indexed.index.get_level_values("seed"))):
                    idx = (system, int(seed))
                    if idx not in cand.index or idx not in base_indexed.index:
                        continue
                    cand_val = float(cand.loc[idx, mse_col])
                    base_val = float(base_indexed.loc[idx, mse_col])
                    if not (math.isfinite(cand_val) and math.isfinite(base_val) and cand_val > 0.0 and base_val > 0.0):
                        continue
                    delta = math.log10(cand_val) - math.log10(base_val)
                    deltas.append(delta)
                    paired_seed_rows.append((int(seed), cand_val, base_val, delta))

                cand_iqm = ratio_lookup.get((root_label, system, h), float("nan"))
                base_iqm = ratio_lookup.get((BASELINE_ROOT, system, h), float("nan"))
                iqm_ratio = cand_iqm / base_iqm if math.isfinite(cand_iqm) and math.isfinite(base_iqm) and base_iqm > 0 else float("nan")

                deltas_arr = np.asarray(deltas, dtype=float)
                if deltas_arr.size < 2 or np.allclose(deltas_arr, deltas_arr[0]):
                    p_raw = float("nan")
                    reason = "too_few_or_constant"
                else:
                    try:
                        p_raw = float(stats.wilcoxon(deltas_arr, alternative="less", zero_method="wilcox").pvalue)
                        reason = ""
                    except ValueError:
                        p_raw = float("nan")
                        reason = "wilcoxon_error"

                test_rows.append(
                    {
                        "root_label": root_label,
                        "display": meta["display"],
                        "system_key": system,
                        "horizon": h,
                        "n_pairs": int(deltas_arr.size),
                        "log10_delta_iqm": iqm(deltas_arr),
                        "log10_delta_median": float(np.median(deltas_arr)) if deltas_arr.size else float("nan"),
                        "iqm_ratio_to_dense": iqm_ratio,
                        "candidate_system_iqm": cand_iqm,
                        "dense_system_iqm": base_iqm,
                        "p_raw": p_raw,
                        "p_holm": float("nan"),
                        "passes_holm_0p05": False,
                        "reason": reason,
                    }
                )

            valid_positions = [i for i, row in enumerate(test_rows) if math.isfinite(row["p_raw"])]
            if valid_positions:
                adjusted = holm_correct([test_rows[i]["p_raw"] for i in valid_positions])
                for pos, p_holm in zip(valid_positions, adjusted):
                    test_rows[pos]["p_holm"] = p_holm
                    test_rows[pos]["passes_holm_0p05"] = bool(p_holm < 0.05)

            detailed_rows.extend(test_rows)
            ratio_values = [row["iqm_ratio_to_dense"] for row in test_rows if math.isfinite(row["iqm_ratio_to_dense"])]
            n_better = sum(1 for value in ratio_values if value < 1.0)
            sign_test_p = (
                float(stats.binomtest(n_better, len(ratio_values), p=0.5, alternative="greater").pvalue)
                if ratio_values
                else float("nan")
            )
            summary_rows.append(
                {
                    "root_label": root_label,
                    "display": meta["display"],
                    "horizon": h,
                    "K": int(sum(1 for row in test_rows if row["passes_holm_0p05"])),
                    "N": len(systems),
                    "systems_with_iqm_ratio_lt_1": int(n_better),
                    "systems_with_iqm_ratio_n": int(len(ratio_values)),
                    "sign_test_p": sign_test_p,
                }
            )
    return pd.DataFrame(detailed_rows), pd.DataFrame(summary_rows)


def fixed_system_seed_bootstrap_ratio(
    system_seed_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    rng: np.random.Generator,
    n_reps: int,
) -> tuple[float, float, float]:
    if n_reps <= 0:
        return float("nan"), float("nan"), float("nan")
    valid = [
        (np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float))
        for candidate, baseline in system_seed_pairs
        if len(candidate) > 0 and len(candidate) == len(baseline)
    ]
    if not valid:
        return float("nan"), float("nan"), float("nan")

    system_ratio_draws = []
    for candidate, baseline in valid:
        indices = rng.integers(0, candidate.size, size=(n_reps, candidate.size))
        baseline_iqm = row_iqm(baseline[indices])
        candidate_iqm = row_iqm(candidate[indices])
        ratio = np.divide(
            candidate_iqm,
            baseline_iqm,
            out=np.full(n_reps, np.nan),
            where=np.isfinite(baseline_iqm) & (baseline_iqm > 0.0),
        )
        system_ratio_draws.append(ratio)
    draws = np.mean(np.column_stack(system_ratio_draws), axis=1)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan"), float("nan"), float("nan")

    return (
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
        float(np.std(draws, ddof=1)),
    )


def hierarchical_system_seed_bootstrap_ratio(
    system_seed_pairs: list[tuple[np.ndarray, np.ndarray]],
    *,
    rng: np.random.Generator,
    n_reps: int,
) -> tuple[float, float, float]:
    """Bootstrap the cross-system mean ratio while resampling systems and seeds.

    The fixed-system bootstrap used for the trend bands quantifies uncertainty
    from finite training seeds conditional on the selected Dysts systems. This
    hierarchical variant also resamples systems, so it is the more relevant
    interval when the question is whether a row is better than Dense MLP across
    the benchmark family rather than only within each fixed system.
    """
    if n_reps <= 0:
        return float("nan"), float("nan"), float("nan")
    valid = [
        (np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float))
        for candidate, baseline in system_seed_pairs
        if len(candidate) > 0 and len(candidate) == len(baseline)
    ]
    if not valid:
        return float("nan"), float("nan"), float("nan")

    n_systems = len(valid)

    per_system_ratio_draws = []
    for candidate, baseline in valid:
        seed_idx = rng.integers(0, candidate.size, size=(n_reps, candidate.size))
        candidate_iqm = row_iqm(candidate[seed_idx])
        baseline_iqm = row_iqm(baseline[seed_idx])
        ratios = np.divide(
            candidate_iqm,
            baseline_iqm,
            out=np.full(n_reps, np.nan),
            where=np.isfinite(baseline_iqm) & (baseline_iqm > 0.0),
        )
        per_system_ratio_draws.append(ratios)

    ratio_draws = np.column_stack(per_system_ratio_draws)
    system_idx = rng.integers(0, n_systems, size=(n_reps, n_systems))
    sampled = ratio_draws[np.arange(n_reps)[:, None], system_idx]
    draws = np.mean(sampled, axis=1)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan"), float("nan"), float("nan")

    return (
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
        float(np.std(draws, ddof=1)),
    )


def aggregate_tests_vs_dense(ratio_df: pd.DataFrame) -> pd.DataFrame:
    """Test each model against Dense using systems as the paired units."""
    rows = []
    for root_label, meta in ROOTS.items():
        if root_label == BASELINE_ROOT:
            continue
        for h, grp in ratio_df[ratio_df["root_label"] == root_label].groupby("horizon", sort=True):
            ratios = grp["ratio_to_dense"].to_numpy(dtype=float)
            ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
            log_ratios = np.log10(ratios)
            n_systems = int(log_ratios.size)
            n_better = int(np.sum(log_ratios < 0.0))

            if n_systems >= 2 and not np.allclose(log_ratios, 0.0):
                try:
                    wilcoxon_p = float(
                        stats.wilcoxon(log_ratios, alternative="less", zero_method="wilcox").pvalue
                    )
                except ValueError:
                    wilcoxon_p = float("nan")
                try:
                    ttest_p = float(stats.ttest_1samp(log_ratios, popmean=0.0, alternative="less").pvalue)
                except ValueError:
                    ttest_p = float("nan")
            else:
                wilcoxon_p = float("nan")
                ttest_p = float("nan")

            sign_p = (
                float(stats.binomtest(n_better, n_systems, p=0.5, alternative="greater").pvalue)
                if n_systems
                else float("nan")
            )

            rows.append(
                {
                    "root_label": root_label,
                    "display": meta["display"],
                    "horizon": int(h),
                    "n_systems": n_systems,
                    "systems_with_ratio_lt_1": n_better,
                    "ratio_iqm": mean_finite(ratios),
                    "ratio_median": float(np.median(ratios)) if ratios.size else float("nan"),
                    "ratio_mean": float(np.mean(ratios)) if ratios.size else float("nan"),
                    "ratio_sd_systems": float(np.std(ratios, ddof=1)) if ratios.size > 1 else float("nan"),
                    "ratio_q25": float(np.percentile(ratios, 25)) if ratios.size else float("nan"),
                    "ratio_q75": float(np.percentile(ratios, 75)) if ratios.size else float("nan"),
                    "log10_ratio_iqm": mean_finite(log_ratios),
                    "log10_ratio_median": float(np.median(log_ratios)) if log_ratios.size else float("nan"),
                    "log10_ratio_mean": float(np.mean(log_ratios)) if log_ratios.size else float("nan"),
                    "log10_ratio_sd_systems": (
                        float(np.std(log_ratios, ddof=1)) if log_ratios.size > 1 else float("nan")
                    ),
                    "p_system_wilcoxon_raw": wilcoxon_p,
                    "p_system_sign_raw": sign_p,
                    "p_system_ttest_raw": ttest_p,
                    "p_system_wilcoxon_holm_all": float("nan"),
                    "p_system_sign_holm_all": float("nan"),
                    "p_system_ttest_holm_all": float("nan"),
                    "p_system_wilcoxon_holm_by_horizon": float("nan"),
                    "p_system_sign_holm_by_horizon": float("nan"),
                    "p_system_ttest_holm_by_horizon": float("nan"),
                }
            )

    result = pd.DataFrame(rows)
    for raw_col, all_col, horizon_col in [
        ("p_system_wilcoxon_raw", "p_system_wilcoxon_holm_all", "p_system_wilcoxon_holm_by_horizon"),
        ("p_system_sign_raw", "p_system_sign_holm_all", "p_system_sign_holm_by_horizon"),
        ("p_system_ttest_raw", "p_system_ttest_holm_all", "p_system_ttest_holm_by_horizon"),
    ]:
        valid = result[raw_col].apply(math.isfinite).to_numpy()
        if np.any(valid):
            corrected = holm_correct(result.loc[valid, raw_col].tolist())
            result.loc[valid, all_col] = corrected
        for _, idx in result.groupby("horizon").groups.items():
            idx = list(idx)
            valid_idx = [i for i in idx if math.isfinite(float(result.at[i, raw_col]))]
            if valid_idx:
                corrected = holm_correct([float(result.at[i, raw_col]) for i in valid_idx])
                for i, p_holm in zip(valid_idx, corrected):
                    result.at[i, horizon_col] = p_holm
    return result


def significance_suffix(p_value: float) -> str:
    if math.isfinite(p_value) and p_value < 0.01:
        return r"^{\ast\ast}"
    if math.isfinite(p_value) and p_value < 0.05:
        return r"^{\ast}"
    return ""


def write_latex_table(
    summary: pd.DataFrame,
    aggregate_tests: pd.DataFrame,
    table_dir: Path,
    horizons: list[int],
) -> None:
    best_by_horizon = {
        h: summary[summary["horizon"] == h].sort_values("cross_system_iqm").iloc[0]["root_label"]
        for h in horizons
    }
    aggregate_lookup = {
        (row["root_label"], int(row["horizon"])): row
        for row in aggregate_tests.to_dict(orient="records")
    }

    lines = [
        r"\begin{tabular}{@{}l " + " ".join(["r"] * len(horizons)) + r"@{}}",
        r"\toprule",
        "Model & " + " & ".join([f"H{h}" for h in horizons]) + r" \\",
        r"\midrule",
    ]
    for root_label, meta in ROOTS.items():
        cells = []
        for h in horizons:
            row = summary[(summary["root_label"] == root_label) & (summary["horizon"] == h)].iloc[0]
            value = tex_number(float(row["cross_system_iqm"]))
            value_tex = value
            if root_label == best_by_horizon[h]:
                value_tex = rf"\mathbf{{{value}}}"
            if root_label != BASELINE_ROOT:
                aggregate_row = aggregate_lookup.get((root_label, h))
                if aggregate_row is not None:
                    p_holm = float(aggregate_row["p_system_wilcoxon_holm_all"])
                    suffix = significance_suffix(p_holm)
                    if suffix:
                        value_tex = rf"{{{value_tex}}}{suffix}"
                cell = rf"${value_tex}$"
            else:
                cell = rf"${value_tex}$"
            cells.append(cell)
        lines.append(f"{meta['display']} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (table_dir / "table4_dysts_dt30_iqm.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_iqm_horizon(
    summary: pd.DataFrame,
    fig_dir: Path,
    horizons: list[int],
    *,
    y_col: str = "cross_system_iqm",
    ci_low_col: str = "seed_bootstrap_ci95_low",
    ci_high_col: str = "seed_bootstrap_ci95_high",
    suffix: str = "",
    ylabel: str = "Best-periodic MSE (mean over system seed-IQMs)",
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    for root_label, meta in ROOTS.items():
        sub = summary[summary["root_label"] == root_label].sort_values("horizon")
        x = sub["horizon"].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        ylo = sub[ci_low_col].to_numpy(dtype=float)
        yhi = sub[ci_high_col].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=meta["display"],
            color=meta["color"],
            linestyle=meta["linestyle"],
        )
        ax.fill_between(x, ylo, yhi, color=meta["color"], alpha=0.13, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel(r"Rollout horizon $H$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons], rotation=35, ha="right")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    ax.set_title(
        forecasting_performance_title(infer_system_count(summary)),
        fontsize=11,
        pad=6,
    )
    fig.savefig(fig_dir / f"fig_dysts_dt30_iqm_horizon{suffix}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_dysts_dt30_iqm_horizon{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def _draw_forecasting_performance_panel(
    ax,
    summary: pd.DataFrame,
    horizons: list[int],
    *,
    log_scale: bool,
    root_order: list[str] | None = None,
    y_col: str = "cross_system_iqm",
    ci_low_col: str = "seed_bootstrap_ci95_low",
    ci_high_col: str = "seed_bootstrap_ci95_high",
    ylabel: str = "MSE (mean over system seed-IQMs)",
) -> None:
    style = {
        "lista": {"color": "#7B3294", "linestyle": "-", "label": "LISTA"},
        "lista_bd": {"color": "#0072B2", "linestyle": "-", "label": "LISTA-BD"},
        "lista_sb": {"color": "#56B4E9", "linestyle": "-", "label": "LISTA-SB"},
        "sparse_mlp_bd": {"color": "#44AA99", "linestyle": "--", "label": "Sparse MLP, BD"},
        "sparse_mlp": {"color": "#009E73", "linestyle": "--", "label": "Sparse MLP"},
        "dense_mlp_tanh": {"color": "#D55E00", "linestyle": "--", "label": "Dense MLP"},
    }

    for root_label in root_order or list(style.keys()):
        cfg = style[root_label]
        sub = summary[summary["root_label"] == root_label].sort_values("horizon")
        x = sub["horizon"].to_numpy(dtype=float)
        y = sub[y_col].to_numpy(dtype=float)
        ylo = sub[ci_low_col].to_numpy(dtype=float)
        yhi = sub[ci_high_col].to_numpy(dtype=float)
        ax.fill_between(x, ylo, yhi, color=cfg["color"], alpha=0.17, linewidth=0)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=5.5,
            linewidth=2.1,
            linestyle=cfg["linestyle"],
            color=cfg["color"],
            label=cfg["label"],
        )

    ax.set_xlabel(r"Rollout horizon $H$ (observation steps)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=FORECAST_YLABEL_FONTSIZE)
    if log_scale:
        ax.set_yscale("log")
        ax.set_title("Log MSE scale", fontsize=12)
    else:
        ax.set_ylim(bottom=0.0)
        ax.set_title("Linear MSE scale", fontsize=12)
    ax.set_xlim(min(horizons) - 70, max(horizons) + 180)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons], rotation=30, ha="right", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=FORECAST_YTICK_LABELSIZE)
    ax.grid(True, which="both", linewidth=0.45, alpha=0.38)
    ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=9.5)


def plot_forecasting_performance_style(
    summary: pd.DataFrame,
    fig_dir: Path,
    horizons: list[int],
    *,
    y_col: str = "cross_system_iqm",
    ci_low_col: str = "seed_bootstrap_ci95_low",
    ci_high_col: str = "seed_bootstrap_ci95_high",
    suffix: str = "",
    title_note: str = "",
    ylabel: str = "MSE (mean over system seed-IQMs)",
) -> None:
    """Dysts trend plots styled like the multibasin horizon curve."""
    n_systems = infer_system_count(summary)
    title = forecasting_performance_title(n_systems)
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    _draw_forecasting_performance_panel(
        ax,
        summary,
        horizons,
        log_scale=True,
        y_col=y_col,
        ci_low_col=ci_low_col,
        ci_high_col=ci_high_col,
        ylabel=ylabel,
    )
    ax.set_title(title_with_note(title, title_note), fontsize=11, pad=4)
    fig.savefig(fig_dir / f"fig_dysts_dt30_forecasting_performance{suffix}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_dysts_dt30_forecasting_performance{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    primary_roots = ["lista", "lista_bd", "sparse_mlp_bd", "sparse_mlp", "dense_mlp_tanh"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    _draw_forecasting_performance_panel(
        ax,
        summary,
        horizons,
        log_scale=True,
        root_order=primary_roots,
        y_col=y_col,
        ci_low_col=ci_low_col,
        ci_high_col=ci_high_col,
        ylabel=ylabel,
    )
    ax.set_title(title_with_note(title, title_note), fontsize=11, pad=4)
    fig.savefig(fig_dir / f"fig_dysts_dt30_forecasting_performance_no_lista_sb{suffix}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_dysts_dt30_forecasting_performance_no_lista_sb{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
    _draw_forecasting_performance_panel(
        ax,
        summary,
        horizons,
        log_scale=False,
        y_col=y_col,
        ci_low_col=ci_low_col,
        ci_high_col=ci_high_col,
        ylabel=ylabel,
    )
    ax.set_title(f"{n_systems}-system Dysts forecasting performance (linear scale)", fontsize=11, pad=6)
    fig.savefig(fig_dir / f"fig_dysts_dt30_forecasting_performance_linear{suffix}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_dysts_dt30_forecasting_performance_linear{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    _draw_forecasting_performance_panel(
        axes[0],
        summary,
        horizons,
        log_scale=False,
        y_col=y_col,
        ci_low_col=ci_low_col,
        ci_high_col=ci_high_col,
        ylabel=ylabel,
    )
    _draw_forecasting_performance_panel(
        axes[1],
        summary,
        horizons,
        log_scale=True,
        y_col=y_col,
        ci_low_col=ci_low_col,
        ci_high_col=ci_high_col,
        ylabel=ylabel,
    )
    axes[0].legend().remove()
    axes[1].legend(frameon=False, loc="lower right", ncol=1)
    fig.suptitle(f"{n_systems}-system Dysts forecasting performance: scale check", fontsize=14)
    fig.savefig(fig_dir / f"fig_dysts_dt30_forecasting_performance_scale_check{suffix}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"fig_dysts_dt30_forecasting_performance_scale_check{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def plot_ratio_to_dense(
    df: pd.DataFrame,
    per_system: pd.DataFrame,
    fig_dir: Path,
    horizons: list[int],
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_systems = int(df["system_key"].nunique())
    base = per_system[per_system["root_label"] == BASELINE_ROOT][
        ["system_key", "horizon", "mse_iqm"]
    ].rename(columns={"mse_iqm": "dense_system_iqm"})
    raw_base = df[df["root_label"] == BASELINE_ROOT].copy()
    rows = []
    per_system_ratio_rows = []
    rng = np.random.default_rng(bootstrap_seed + 7919)
    hierarchical_rng = np.random.default_rng(bootstrap_seed + 15485863)
    for root_label, meta in ROOTS.items():
        if root_label == BASELINE_ROOT:
            for h in horizons:
                rows.append(
                    {
                        "root_label": root_label,
                        "display": meta["display"],
                        "horizon": h,
                        "ratio_iqm": 1.0,
                        "ratio_q25": 1.0,
                        "ratio_q75": 1.0,
                        "ratio_seed_bootstrap_ci95_low": 1.0,
                        "ratio_seed_bootstrap_ci95_high": 1.0,
                        "ratio_seed_bootstrap_se": 0.0,
                        "ratio_hierarchical_bootstrap_ci95_low": 1.0,
                        "ratio_hierarchical_bootstrap_ci95_high": 1.0,
                        "ratio_hierarchical_bootstrap_se": 0.0,
                        "ratio_seed_bootstrap_reps": int(bootstrap_reps),
                        "n_systems": n_systems,
                    }
                )
            continue
        merged = per_system[per_system["root_label"] == root_label].merge(
            base,
            on=["system_key", "horizon"],
            how="inner",
        )
        merged["ratio_to_dense"] = merged["mse_iqm"] / merged["dense_system_iqm"]
        for _, ratio_row in merged.iterrows():
            per_system_ratio_rows.append(
                {
                    "root_label": root_label,
                    "display": meta["display"],
                    "system_key": ratio_row["system_key"],
                    "horizon": int(ratio_row["horizon"]),
                    "candidate_system_iqm": float(ratio_row["mse_iqm"]),
                    "dense_system_iqm": float(ratio_row["dense_system_iqm"]),
                    "ratio_to_dense": float(ratio_row["ratio_to_dense"]),
                    "log10_ratio_to_dense": math.log10(float(ratio_row["ratio_to_dense"]))
                    if math.isfinite(float(ratio_row["ratio_to_dense"])) and float(ratio_row["ratio_to_dense"]) > 0.0
                    else float("nan"),
                }
            )
        raw_candidate = df[df["root_label"] == root_label].copy()
        for h, grp in merged.groupby("horizon", sort=True):
            mse_col = f"h{h}_best_periodic_mean"
            system_seed_pairs = []
            for system in sorted(df["system_key"].dropna().unique()):
                candidate_system = raw_candidate[raw_candidate["system_key"] == system].set_index("seed", drop=False)
                baseline_system = raw_base[raw_base["system_key"] == system].set_index("seed", drop=False)
                candidate_values = []
                baseline_values = []
                for seed in sorted(set(candidate_system.index) & set(baseline_system.index)):
                    candidate_value = float(candidate_system.loc[seed, mse_col])
                    baseline_value = float(baseline_system.loc[seed, mse_col])
                    if (
                        math.isfinite(candidate_value)
                        and math.isfinite(baseline_value)
                        and candidate_value > 0.0
                        and baseline_value > 0.0
                    ):
                        candidate_values.append(candidate_value)
                        baseline_values.append(baseline_value)
                if candidate_values:
                    system_seed_pairs.append((np.asarray(candidate_values), np.asarray(baseline_values)))
            ci_low, ci_high, boot_se = fixed_system_seed_bootstrap_ratio(
                system_seed_pairs,
                rng=rng,
                n_reps=bootstrap_reps,
            )
            h_ci_low, h_ci_high, h_boot_se = hierarchical_system_seed_bootstrap_ratio(
                system_seed_pairs,
                rng=hierarchical_rng,
                n_reps=bootstrap_reps,
            )
            values = grp["ratio_to_dense"].to_numpy(dtype=float)
            values = values[np.isfinite(values) & (values > 0.0)]
            rows.append(
                {
                    "root_label": root_label,
                    "display": meta["display"],
                    "horizon": int(h),
                    "ratio_mean": mean_finite(values),
                    "ratio_iqm": mean_finite(values),
                    "ratio_q25": float(np.percentile(values, 25)) if values.size else float("nan"),
                    "ratio_q75": float(np.percentile(values, 75)) if values.size else float("nan"),
                    "ratio_seed_bootstrap_ci95_low": ci_low,
                    "ratio_seed_bootstrap_ci95_high": ci_high,
                    "ratio_seed_bootstrap_se": boot_se,
                    "ratio_hierarchical_bootstrap_ci95_low": h_ci_low,
                    "ratio_hierarchical_bootstrap_ci95_high": h_ci_high,
                    "ratio_hierarchical_bootstrap_se": h_boot_se,
                    "ratio_seed_bootstrap_reps": int(bootstrap_reps),
                    "n_systems": int(values.size),
                }
            )
    ratio_df = pd.DataFrame(rows)
    per_system_ratio_df = pd.DataFrame(per_system_ratio_rows)

    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    for root_label, meta in ROOTS.items():
        sub = ratio_df[ratio_df["root_label"] == root_label].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["ratio_iqm"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=meta["display"],
            color=meta["color"],
            linestyle=meta["linestyle"],
        )
        ax.fill_between(
            sub["horizon"].to_numpy(dtype=float),
            sub["ratio_seed_bootstrap_ci95_low"].to_numpy(dtype=float),
            sub["ratio_seed_bootstrap_ci95_high"].to_numpy(dtype=float),
            color=meta["color"],
            alpha=0.13,
            linewidth=0,
        )
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"Rollout horizon $H$")
    ax.set_ylabel("Ratio to Dense MLP (mean over system seed-IQMs)")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons], rotation=35, ha="right")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(fig_dir / "fig_dysts_dt30_ratio_to_dense.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_dysts_dt30_ratio_to_dense.png", bbox_inches="tight")
    plt.close(fig)
    return ratio_df, per_system_ratio_df


def plot_winner_counts(per_system: pd.DataFrame, fig_dir: Path, horizons: list[int]) -> pd.DataFrame:
    rows = []
    for h in horizons:
        sub = per_system[per_system["horizon"] == h]
        for system, grp in sub.groupby("system_key", sort=True):
            best = grp.sort_values("mse_iqm").iloc[0]
            rows.append(
                {
                    "horizon": h,
                    "system_key": system,
                    "best_root_label": best["root_label"],
                    "best_display": best["display"],
                    "best_mse_iqm": float(best["mse_iqm"]),
                }
            )
    winners = pd.DataFrame(rows)
    counts = (
        winners.groupby(["horizon", "best_root_label"])
        .size()
        .reset_index(name="n_systems")
    )

    fig, ax = plt.subplots(figsize=(6.6, 3.2), constrained_layout=True)
    bottom = np.zeros(len(horizons))
    x = np.arange(len(horizons))
    for root_label, meta in ROOTS.items():
        vals = []
        for h in horizons:
            match = counts[(counts["horizon"] == h) & (counts["best_root_label"] == root_label)]
            vals.append(int(match["n_systems"].iloc[0]) if not match.empty else 0)
        ax.bar(x, vals, bottom=bottom, color=meta["color"], edgecolor="white", label=meta["display"])
        bottom += np.asarray(vals, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in horizons], rotation=35, ha="right")
    ax.set_ylabel("# systems with lowest seed-IQM MSE")
    n_systems = int(per_system["system_key"].nunique())
    ax.set_ylim(0, n_systems)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(fig_dir / "fig_dysts_dt30_winner_counts.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "fig_dysts_dt30_winner_counts.png", bbox_inches="tight")
    plt.close(fig)
    return winners


def plot_perseed_histograms(df: pd.DataFrame, fig_dir: Path, horizons: list[int]) -> None:
    n_cols = 4
    n_rows = int(math.ceil(len(horizons) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.0, 2.6 * n_rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    bins = np.linspace(-5, 2, 42)
    for ax, h in zip(axes_arr, horizons):
        col = f"h{h}_best_periodic_mean"
        for root_label, meta in ROOTS.items():
            values = finite_positive(df[df["root_label"] == root_label][col])
            if values.size == 0:
                continue
            ax.hist(
                np.log10(values),
                bins=bins,
                color=meta["color"],
                alpha=0.38,
                edgecolor="white",
                linewidth=0.25,
                label=meta["display"],
            )
        ax.set_title(f"H{h}", fontsize=8)
        ax.set_xlabel(r"$\log_{10}$ MSE")
        ax.set_ylabel("# system-seed rows")
        ax.grid(axis="y", linewidth=0.3, alpha=0.25)
    for ax in axes_arr[len(horizons) :]:
        ax.axis("off")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.savefig(fig_dir / "appfig_dysts_dt30_perseed_histograms.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "appfig_dysts_dt30_perseed_histograms.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasting-csv", type=Path, default=DEFAULT_FORECASTING_CSV)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--exclude-systems",
        nargs="*",
        default=list(DEFAULT_EXCLUDED_SYSTEMS),
        help="Dysts systems to exclude from the paper-facing aggregation. "
        "Bare names are interpreted as dysts:<name>. Pass the flag with no values to include all systems.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    horizons = list(dict.fromkeys(int(h) for h in args.horizons))
    excluded_systems = {
        normalize_system_key(system)
        for system in args.exclude_systems
        if normalize_system_key(system)
    }

    df = load_rows(args.forecasting_csv, excluded_systems=excluded_systems)
    included_systems = sorted(df["system_key"].dropna().unique().tolist())
    per_system, summary = per_system_summary(
        df,
        horizons,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    tests_detailed, tests_summary = paired_tests(df, per_system, horizons)
    ratio_df, per_system_ratio_df = plot_ratio_to_dense(
        df,
        per_system,
        args.fig_dir,
        horizons,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    aggregate_tests = aggregate_tests_vs_dense(per_system_ratio_df)
    winners = plot_winner_counts(per_system, args.fig_dir, horizons)
    plot_iqm_horizon(summary, args.fig_dir, horizons)
    plot_iqm_horizon(
        summary,
        args.fig_dir,
        horizons,
        ci_low_col="system_bootstrap_ci95_low",
        ci_high_col="system_bootstrap_ci95_high",
        suffix="_system_ci",
        ylabel="Best-periodic MSE (system-bootstrap CI)",
    )
    plot_iqm_horizon(
        summary,
        args.fig_dir,
        horizons,
        y_col="cross_system_log_iqm_geomean",
        ci_low_col="log_seed_bootstrap_ci95_low",
        ci_high_col="log_seed_bootstrap_ci95_high",
        suffix="_log_seed_bootstrap",
        ylabel="Best-periodic MSE (log-space seed bootstrap)",
    )
    plot_forecasting_performance_style(summary, args.fig_dir, horizons)
    plot_forecasting_performance_style(
        summary,
        args.fig_dir,
        horizons,
        ci_low_col="system_bootstrap_ci95_low",
        ci_high_col="system_bootstrap_ci95_high",
        suffix="_system_ci",
        title_note="\n(system CI)",
        ylabel="MSE (system-bootstrap CI)",
    )
    plot_forecasting_performance_style(
        summary,
        args.fig_dir,
        horizons,
        y_col="cross_system_log_iqm_geomean",
        ci_low_col="log_seed_bootstrap_ci95_low",
        ci_high_col="log_seed_bootstrap_ci95_high",
        suffix="_log_seed_bootstrap",
        title_note="\n(log seed bootstrap)",
        ylabel="MSE (log-space seed bootstrap)",
    )
    plot_perseed_histograms(df, args.fig_dir, horizons)
    write_latex_table(summary, aggregate_tests, args.table_dir, horizons)

    per_system.to_csv(args.table_dir / "dysts_dt30_per_system_iqm.csv", index=False)
    summary.to_csv(args.table_dir / "dysts_dt30_iqm_summary.csv", index=False)
    tests_detailed.to_csv(args.table_dir / "dysts_dt30_per_system_wilcoxon_vs_dense.csv", index=False)
    tests_summary.to_csv(args.table_dir / "dysts_dt30_wilcoxon_summary_vs_dense.csv", index=False)
    ratio_df.to_csv(args.table_dir / "dysts_dt30_ratio_to_dense_summary.csv", index=False)
    per_system_ratio_df.to_csv(args.table_dir / "dysts_dt30_per_system_ratio_to_dense.csv", index=False)
    aggregate_tests.to_csv(args.table_dir / "dysts_dt30_aggregate_tests_vs_dense.csv", index=False)
    winners.to_csv(args.table_dir / "dysts_dt30_winner_systems.csv", index=False)

    payload = {
        "forecasting_csv": str(args.forecasting_csv),
        "horizons": horizons,
        "included_systems": included_systems,
        "excluded_systems": sorted(excluded_systems),
        "n_systems": len(included_systems),
        "baseline_root": BASELINE_ROOT,
        "bootstrap": {
            "scheme": "fixed_system_seed_resampling_raw_mse",
            "reps": int(args.bootstrap_reps),
            "seed": int(args.bootstrap_seed),
            "additional_schemes": [
                "system_resampling_over_fixed_per_system_seed_iqms",
                "fixed_system_seed_resampling_log_mse",
            ],
        },
        "roots": ROOTS,
        "summary": summary.to_dict(orient="records"),
        "tests_summary": tests_summary.to_dict(orient="records"),
        "aggregate_tests_vs_dense": aggregate_tests.to_dict(orient="records"),
    }
    (args.table_dir / "dysts_dt30_iqm_and_tests.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote tables to {args.table_dir}")
    print(f"Wrote figures to {args.fig_dir}")
    print(f"Included systems ({len(included_systems)}): {', '.join(included_systems)}")
    if excluded_systems:
        print(f"Excluded systems: {', '.join(sorted(excluded_systems))}")
    print("\nDysts dt30 mean over system seed-IQMs and Holm tests vs Dense MLP")
    merged = summary.merge(tests_summary, on=["root_label", "display", "horizon"], how="left")
    for root_label, meta in ROOTS.items():
        print(f"  {meta['display']}")
        for _, row in merged[merged["root_label"] == root_label].sort_values("horizon").iterrows():
            if root_label == BASELINE_ROOT:
                sig = "baseline"
            else:
                sig = f"K/N={int(row['K'])}/{int(row['N'])}, better-systems={int(row['systems_with_iqm_ratio_lt_1'])}/{int(row['systems_with_iqm_ratio_n'])}"
            print(f"    H{int(row['horizon'])}: mean={row['cross_system_mean']:.6g}; {sig}")
    print("\nAggregate system-level tests vs Dense MLP (per-system IQM ratios)")
    for root_label, meta in ROOTS.items():
        if root_label == BASELINE_ROOT:
            continue
        print(f"  {meta['display']}")
        sub = aggregate_tests[aggregate_tests["root_label"] == root_label].sort_values("horizon")
        for _, row in sub.iterrows():
            print(
                f"    H{int(row['horizon'])}: ratio-mean={row['ratio_mean']:.3g}; "
                f"systems={int(row['systems_with_ratio_lt_1'])}/{int(row['n_systems'])}; "
                f"p_w={row['p_system_wilcoxon_raw']:.4g}; "
                f"p_w_holm40={row['p_system_wilcoxon_holm_all']:.4g}"
            )


if __name__ == "__main__":
    main()
