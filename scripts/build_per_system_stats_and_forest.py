"""Per-system paired Wilcoxon, Holm correction, forest plots, and merged
boundary-only 15-system table.

Outputs
-------
docs/figures/neurips_paper_2026/fig_fixed17_forest.pdf  (forest plot, log-MSE diff per system)
docs/figures/neurips_paper_2026/_tables/persystem_wilcoxon_fixed17.json
docs/figures/neurips_paper_2026/_tables/table1_boundary_only.csv
docs/figures/neurips_paper_2026/_tables/table1_boundary_only.tex

Per-system paired testing design
--------------------------------
For each retained system and each (candidate, control) pair, we use the
common completed paired seeds, up to 15 per system in the current paper-facing
packet. The manuscript-facing confirmatory test is a one-sided paired Wilcoxon
signed-rank test on per-seed deltas. Forecasting uses paired
log10(best-periodic MSE) differences (candidate < control), while the
interpretability metrics below use the metric-specific directions documented in
docs/EXPERIMENTS.md and in the manuscript appendix. Holm step-down correction
is applied across the systems for each (candidate, control, metric/horizon)
combination at alpha=0.05, and we report:

- K: number of systems where Holm-corrected p < 0.05 (candidate < control).
- median_diff_log10: per-system median paired log-MSE difference, then median
  across the retained systems. This is in log10 raw-MSE units.
- per_system effect sizes (paired log-MSE diff with 95% bootstrap CI over the
  common completed seeds) for the forest plot.

Paired t-test p-values are still written as supporting diagnostics in the
supplementary CSV, but the K/N counts and appendix bolding use the Wilcoxon
Holm-corrected p-values.

The control is always the boundary-emphasized Dense MLP (no shrink) in this
analysis, so all rows live in the same sampling regime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/mila/l/lia/skae")
FIG_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026"
TBL_DIR = FIG_DIR / "_tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

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
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PALETTE = {
    "dense_lista": "#785EF0",
    "blockdiag_lista": "#0072B2",
    "softblock_lista": "#56B4E9",
    "sparse_mlp": "#009E73",
    "blockdiag_mlp": "#44AA99",
    "zero_mlp": "#D55E00",
}

HORIZONS = (100, 500, 1000)
ALPHA = 0.05  # significance level for Holm-corrected per-system tests
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260501
HORIZON_YLABEL_FONTSIZE = 11.5
HORIZON_YTICK_LABELSIZE = 9.5
EXCLUDED_SYSTEMS = {
    "multiwell_strong_transition",
    "claude_checkerboard_potential",
    "claude:checkerboard_potential",
}

FC_CSVS = [
    ROOT
    / "results"
    / "transition_rich_table2_5model_seed15_backfill_20260428"
    / "collect_pass0"
    / "forecasting_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428"
    / "collect_pass0"
    / "forecasting_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_dense_p256_hardinit_table123_20260430"
    / "collect_pass0"
    / "forecasting_rows.csv",
]
INTERP_CSVS = [
    ROOT
    / "results"
    / "transition_rich_table2_5model_seed15_backfill_20260428"
    / "interpretability_per_basin_deep_current_table1_pass0"
    / "interpretability_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428"
    / "interpretability_per_basin_deep_current_table1_pass0"
    / "interpretability_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_dense_p256_hardinit_table123_20260430"
    / "interpretability_per_basin_deep_current_table1_pass0"
    / "interpretability_rows.csv",
]

# Boundary-only roster: LISTA-BD; matched-dimension LISTA-SB d_z=256; Sparse
# MLP, Sparse MLP BD, Dense MLP no-shrink. The control for the paired Wilcoxon
# is the boundary Dense MLP no-shrink (`zero_mlp`).
ROOTS = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": {
        "label": "LISTA",
        "long": "Sparse Latent Koopman, dense-transition LISTA",
        "color": "dense_lista",
        "source": "main",
    },
    "lista_blockdiag_signsplit_hardinit_basin_partition": {
        "label": "LISTA-BD",
        "long": "Sparse Latent Koopman, block-diagonal LISTA",
        "color": "blockdiag_lista",
        "source": "main",
    },
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": {
        "label": "LISTA-SB",
        "long": "Sparse Latent Koopman, soft-block LISTA",
        "color": "softblock_lista",
        "source": "main",
    },
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": {
        "label": "Sparse MLP, BD",
        "long": "Sparse MLP control with block-diagonal $K$",
        "color": "blockdiag_mlp",
        "source": "hardinit",
    },
    "mlp_sparse_hardinit_basin_partition_control": {
        "label": "Sparse MLP",
        "long": "Sparse MLP control (dense $K$)",
        "color": "sparse_mlp",
        "source": "hardinit",
    },
    "mlp_zero_sparse_hardinit_basin_partition_control": {
        "label": "Dense MLP",
        "long": "Dense MLP control with no $\\ell_1$ shrinkage",
        "color": "zero_mlp",
        "source": "hardinit",
    },
}
BASELINE = "mlp_zero_sparse_hardinit_basin_partition_control"
STAGE2_LOCAL_MAP_SUMMARY_CSV = (
    ROOT
    / "results"
    / "routed_stage2_local_maps_20260506"
    / "combined_best_lista_multibasin_j040_50k_seed0_9"
    / "dense_mlp_comparison"
    / "stage2_vs_dense_summary.csv"
)

# --------------------------------------------------------------------------------------
# Load and merge
# --------------------------------------------------------------------------------------

print("Loading forecasting CSVs...")
fc = pd.concat([pd.read_csv(path, low_memory=False) for path in FC_CSVS], ignore_index=True)
fc = fc[fc["root_label"].isin(ROOTS.keys())].copy()

itp = pd.concat([pd.read_csv(path, low_memory=False) for path in INTERP_CSVS], ignore_index=True)
itp = itp[itp["root_label"].isin(ROOTS.keys())].copy()

for column in ("system_name", "system_key", "train_env_name"):
    if column in fc:
        fc = fc[~fc[column].isin(EXCLUDED_SYSTEMS)].copy()
    if column in itp:
        itp = itp[~itp[column].isin(EXCLUDED_SYSTEMS)].copy()
deep_mask = (
    (itp["support_scheme"] == "absolute:0.001")
    & (itp["subset"] == "deep")
    & (itp["family_jaccard_threshold"] == 0.5)
)
itp_deep = itp[deep_mask].copy()

systems = sorted(fc["system_name"].unique())
BENCHMARK_N = len(systems)
print(
    f"Found {BENCHMARK_N} retained systems after excluding "
    f"{sorted(EXCLUDED_SYSTEMS)}: {systems[:3]}... ({len(ROOTS)} roots)"
)

ALL_FORECAST_HORIZONS = tuple(
    sorted(
        {
            int(col[1 : -len("_best_periodic_mean")])
            for col in fc.columns
            if col.startswith("h") and col.endswith("_best_periodic_mean") and col[1 : -len("_best_periodic_mean")].isdigit()
        }
    )
)
print(f"Available retained-benchmark forecast horizons: {ALL_FORECAST_HORIZONS}")

# --------------------------------------------------------------------------------------
# IQM helpers
# --------------------------------------------------------------------------------------


def iqm(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    if v.size < 4:
        return float(np.mean(v))
    lo, hi = np.percentile(v, [25, 75])
    sel = v[(v >= lo) & (v <= hi)]
    if sel.size == 0:
        return float(np.median(v))
    return float(np.mean(sel))


def mean_finite(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.mean(v))


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
    """Bootstrap mean-over-systems uncertainty by resampling seeds within systems."""
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


def fixed_system_seed_bootstrap_log_relative_band(
    system_seed_values: list[np.ndarray],
    *,
    center: float,
    rng: np.random.Generator,
    n_reps: int,
) -> tuple[float, float, float]:
    """Bootstrap typical within-system seed uncertainty on a relative scale."""
    if n_reps <= 0 or not math.isfinite(center) or center <= 0.0:
        return float("nan"), float("nan"), float("nan")

    valid = []
    for values in system_seed_values:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        system_center = iqm(values)
        if values.size > 0 and math.isfinite(system_center) and system_center > 0.0:
            valid.append((values, system_center))
    if not valid:
        return float("nan"), float("nan"), float("nan")

    per_system_log_rel = []
    for values, system_center in valid:
        indices = rng.integers(0, values.size, size=(n_reps, values.size))
        draws = row_iqm(values[indices])
        draws = np.clip(draws, np.finfo(float).tiny, None)
        per_system_log_rel.append(np.log10(draws) - math.log10(system_center))

    log_rel_draws = np.mean(np.column_stack(per_system_log_rel), axis=1)
    return (
        float(center * (10.0 ** np.percentile(log_rel_draws, 2.5))),
        float(center * (10.0 ** np.percentile(log_rel_draws, 97.5))),
        float(np.std(log_rel_draws, ddof=1)),
    )


def std_log10(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size <= 1:
        return float("nan")
    return float(np.std(np.log10(v), ddof=1))


def iqr_log10(v: np.ndarray) -> float:
    """Inter-quartile range of log10 values: robust to catastrophic outliers.

    Better than std for heavy-tailed cross-system MSE distributions where a
    single blowup system can swamp the standard deviation.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size <= 1:
        return float("nan")
    lo, hi = np.percentile(np.log10(v), [25, 75])
    return float(hi - lo)


def holm(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        adj[idx] = max(running, min(1.0, (n - rank) * p[idx]))
        running = adj[idx]
    return adj


# --------------------------------------------------------------------------------------
# Per-system paired Wilcoxon (10 seeds per system) for each (root, horizon)
# --------------------------------------------------------------------------------------

print("Running per-system paired Wilcoxon...")

per_system_records: list[dict] = []

base_df = fc[fc["root_label"] == BASELINE]

for root_label, meta in ROOTS.items():
    if root_label == BASELINE:
        continue
    cand_df = fc[fc["root_label"] == root_label]
    for h in HORIZONS:
        col = f"h{h}_best_periodic_mean"
        # Per-system per-seed pairing
        cand_pivot = cand_df.pivot_table(
            index="system_name", columns="seed", values=col, aggfunc="first"
        )
        base_pivot = base_df.pivot_table(
            index="system_name", columns="seed", values=col, aggfunc="first"
        )
        common_systems = cand_pivot.index.intersection(base_pivot.index)
        per_sys_p = []
        per_sys_p_wilcoxon = []
        per_sys_mean = []
        per_sys_median = []
        per_sys_hl = []
        per_sys_lo = []
        per_sys_hi = []
        per_sys_n = []
        for sysname in common_systems:
            c = cand_pivot.loc[sysname].to_numpy(dtype=float)
            b = base_pivot.loc[sysname].to_numpy(dtype=float)
            mask = np.isfinite(c) & np.isfinite(b) & (c > 0) & (b > 0)
            c = c[mask]
            b = b[mask]
            if c.size < 4:
                per_sys_p.append(np.nan)
                per_sys_p_wilcoxon.append(np.nan)
                per_sys_mean.append(np.nan)
                per_sys_median.append(np.nan)
                per_sys_hl.append(np.nan)
                per_sys_lo.append(np.nan)
                per_sys_hi.append(np.nan)
                per_sys_n.append(int(c.size))
                continue
            log_c = np.log10(c)
            log_b = np.log10(b)
            paired = log_c - log_b
            # Supporting diagnostic only; manuscript K/N uses Wilcoxon below.
            try:
                t_res = stats.ttest_rel(log_c, log_b, alternative="less")
                p_val = float(t_res.pvalue)
            except ValueError:
                p_val = float("nan")
            try:
                w = stats.wilcoxon(
                    log_c, log_b, alternative="less", zero_method="wilcox"
                )
                p_val_w = float(w.pvalue)
            except ValueError:
                p_val_w = float("nan")
            mean_diff = float(np.mean(paired))
            median_diff = float(np.median(paired))
            # Hodges-Lehmann shift: median of all pairwise differences
            # (the natural location estimate paired with Wilcoxon)
            pairwise = (log_c[:, None] - log_b[None, :]).ravel()
            hl_diff = float(np.median(pairwise))
            # Bootstrap CI on per-seed paired log-diff over common seeds;
            # bootstrap the *median* to be consistent with the Wilcoxon test
            # (rank-based, robust to outlier seeds).
            rng = np.random.default_rng(abs(hash(("forest", root_label, h, sysname))) % 2**32)
            n_boot = 2000
            bs = np.empty(n_boot)
            for i in range(n_boot):
                idx = rng.integers(0, paired.size, size=paired.size)
                bs[i] = float(np.median(paired[idx]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            per_sys_p.append(p_val)
            per_sys_p_wilcoxon.append(p_val_w)
            per_sys_mean.append(mean_diff)
            per_sys_median.append(median_diff)
            per_sys_hl.append(hl_diff)
            per_sys_lo.append(float(lo))
            per_sys_hi.append(float(hi))
            per_sys_n.append(int(c.size))
        per_sys_p = np.array(per_sys_p)
        per_sys_p_wilcoxon = np.array(per_sys_p_wilcoxon)
        # Holm correction across systems. The manuscript-facing counts use
        # adj_w from Wilcoxon; adj is retained as a supporting t-test diagnostic.
        valid_mask = np.isfinite(per_sys_p)
        adj = np.full_like(per_sys_p, np.nan)
        if valid_mask.any():
            adj[valid_mask] = holm(per_sys_p[valid_mask])
        valid_w = np.isfinite(per_sys_p_wilcoxon)
        adj_w = np.full_like(per_sys_p_wilcoxon, np.nan)
        if valid_w.any():
            adj_w[valid_w] = holm(per_sys_p_wilcoxon[valid_w])
        for i, sysname in enumerate(common_systems):
            per_system_records.append(
                {
                    "root_label": root_label,
                    "label": meta["label"],
                    "horizon": h,
                    "system": sysname,
                    "p_raw_t": float(per_sys_p[i]) if np.isfinite(per_sys_p[i]) else None,
                    "p_holm_t": float(adj[i]) if np.isfinite(adj[i]) else None,
                    "p_raw_w": float(per_sys_p_wilcoxon[i]) if np.isfinite(per_sys_p_wilcoxon[i]) else None,
                    "p_holm_w": float(adj_w[i]) if np.isfinite(adj_w[i]) else None,
                    "mean_diff_log10": float(per_sys_mean[i]) if np.isfinite(per_sys_mean[i]) else None,
                    "median_diff_log10": float(per_sys_median[i]) if np.isfinite(per_sys_median[i]) else None,
                    "hl_diff_log10": float(per_sys_hl[i]) if np.isfinite(per_sys_hl[i]) else None,
                    "lo_log10": float(per_sys_lo[i]) if np.isfinite(per_sys_lo[i]) else None,
                    "hi_log10": float(per_sys_hi[i]) if np.isfinite(per_sys_hi[i]) else None,
                    "n_seeds": int(per_sys_n[i]),
                }
            )


per_system_df = pd.DataFrame(per_system_records)
per_system_df.to_csv(TBL_DIR / "persystem_wilcoxon_fixed17.csv", index=False)


# --------------------------------------------------------------------------------------
# Per-system Wilcoxon for the discriminative interpretability metrics on the
# deep-state slice. We test each candidate row against the same Dense MLP
# no-shrink baseline used for forecasting. H(S_abs|B) is "lower is better" so we
# use alternative='less'; U_exact is "higher is better" so we use 'greater'.
# H(B|S_abs) saturates at 0 for sparse rows on this slice; we skip it.
# --------------------------------------------------------------------------------------

print("Running per-system Wilcoxon on entropy metrics (deep slice)...")

ENTROPY_TESTS = {
    # legacy / supporting
    "h_support_given_basin": ("less", "HSgivenB"),
    "u_exact": ("greater", "U_exact"),
    # new headline interpretability metrics
    "mean_support_size": ("less", "MeanSupportSize"),
    "family_h_basin_given_family": ("less", "HBgivenF"),
    "support_freeze_wrong_over_base_h1": ("greater", "FreezeWrongH1"),
    "support_freeze_wrong_over_base_h20": ("greater", "FreezeWrongH20"),
}

entropy_records: list[dict] = []
itp_deep_baseline = itp_deep[itp_deep["root_label"] == BASELINE]

for metric_col, (alt, metric_label) in ENTROPY_TESTS.items():
    base_pivot = itp_deep_baseline.pivot_table(
        index="system_name", columns="seed", values=metric_col, aggfunc="first"
    )
    for root_label, meta in ROOTS.items():
        if root_label == BASELINE:
            continue
        cand_pivot = (
            itp_deep[itp_deep["root_label"] == root_label]
            .pivot_table(index="system_name", columns="seed", values=metric_col, aggfunc="first")
        )
        common = cand_pivot.index.intersection(base_pivot.index)
        per_sys_p = []
        per_sys_diff = []
        for sysname in common:
            c = cand_pivot.loc[sysname].to_numpy(dtype=float)
            b = base_pivot.loc[sysname].to_numpy(dtype=float)
            mask = np.isfinite(c) & np.isfinite(b)
            c = c[mask]
            b = b[mask]
            if c.size < 4:
                per_sys_p.append(np.nan)
                per_sys_diff.append(np.nan)
                continue
            paired = c - b
            try:
                w = stats.wilcoxon(c, b, alternative=alt, zero_method="wilcox")
                p_val = float(w.pvalue)
            except ValueError:
                p_val = float("nan")
            per_sys_p.append(p_val)
            per_sys_diff.append(float(np.median(paired)))
        per_sys_p = np.asarray(per_sys_p)
        valid = np.isfinite(per_sys_p)
        adj = np.full_like(per_sys_p, np.nan)
        if valid.any():
            adj[valid] = holm(per_sys_p[valid])
        for i, sysname in enumerate(common):
            entropy_records.append(
                {
                    "root_label": root_label,
                    "label": meta["label"],
                    "metric": metric_label,
                    "system": sysname,
                    "p_raw": float(per_sys_p[i]) if np.isfinite(per_sys_p[i]) else None,
                    "p_holm": float(adj[i]) if np.isfinite(adj[i]) else None,
                    "median_diff": float(per_sys_diff[i]) if np.isfinite(per_sys_diff[i]) else None,
                }
            )

entropy_df = pd.DataFrame(entropy_records)
entropy_df.to_csv(TBL_DIR / "persystem_wilcoxon_entropy_fixed17.csv", index=False)

# Aggregate K/N per (root, metric)
entropy_summary: dict[tuple[str, str], dict] = {}
for (root_label, metric_label), grp in entropy_df.groupby(["root_label", "metric"]):
    n = int(grp["p_holm"].notna().sum())
    k = int((grp["p_holm"] < ALPHA).sum())
    entropy_summary[(root_label, metric_label)] = {
        "K_systems_better": k,
        "N_systems_tested": n,
        "median_diff": float(grp["median_diff"].median()),
    }
print("Entropy K/N summary:")
for k, v in entropy_summary.items():
    print(f"  {k}: K={v['K_systems_better']}/{v['N_systems_tested']} median Δ={v['median_diff']:+.3f}")

# --------------------------------------------------------------------------------------
# Aggregate "K of N" summary table per (root, horizon)
# --------------------------------------------------------------------------------------

summary_rows: list[dict] = []
for root_label, meta in ROOTS.items():
    if root_label == BASELINE:
        continue
    for h in HORIZONS:
        sub = per_system_df[
            (per_system_df["root_label"] == root_label) & (per_system_df["horizon"] == h)
        ]
        n_total = int(sub["p_holm_w"].notna().sum())
        # Headline K of N: Wilcoxon (distribution-free) after Holm
        k_better_w = int((sub["p_holm_w"] < ALPHA).sum())
        # t-test (parametric, requires Gaussian paired diffs); supporting only
        k_better_t = int((sub["p_holm_t"] < ALPHA).sum())
        # Headline cross-system effect: median across systems of the per-system
        # median paired log10 difference (consistent with Wilcoxon-rank logic).
        med_diff = float(sub["median_diff_log10"].median())
        # Main table estimate: seed-IQM within each system, arithmetic mean
        # across systems so every benchmark system contributes.
        cand_per_sys_iqm = (
            fc[fc["root_label"] == root_label]
            .groupby("system_name")[f"h{h}_best_periodic_mean"]
            .apply(lambda v: iqm(v.to_numpy()))
        )
        mean_cell = mean_finite(cand_per_sys_iqm.to_numpy())
        std_cell = iqr_log10(cand_per_sys_iqm.to_numpy())
        summary_rows.append(
            {
                "root_label": root_label,
                "label": meta["label"],
                "horizon": h,
                "K_systems_better": k_better_w,
                "K_systems_better_ttest": k_better_t,
                "N_systems_tested": n_total,
                "median_diff_log10": med_diff,
                "mean_raw": mean_cell,
                "iqm_raw": mean_cell,
                "log10_iqr_systems": std_cell,
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TBL_DIR / "persystem_summary_fixed17.csv", index=False)
print(summary_df.to_string(index=False))

# Persist as JSON for the LaTeX automation
nested = {}
for row in summary_rows:
    nested.setdefault(row["root_label"], {})[f"H{row['horizon']}"] = {
        k: row[k]
        for k in (
            "K_systems_better",
            "K_systems_better_ttest",
            "N_systems_tested",
            "median_diff_log10",
            "iqm_raw",
            "log10_iqr_systems",
        )
    }
(TBL_DIR / "persystem_wilcoxon_fixed17.json").write_text(json.dumps(nested, indent=2))

# --------------------------------------------------------------------------------------
# Forest plot: rows = systems, x = log10 diff candidate vs Dense MLP (no shrink) at H=1000
# --------------------------------------------------------------------------------------

print("Building forest plot at H=1000...")

H_FOREST = 1000

# Stable system order: by median across all candidates (most-favorable first)
sys_order = (
    per_system_df[per_system_df["horizon"] == H_FOREST]
    .groupby("system")["median_diff_log10"]
    .median()
    .sort_values()
    .index.tolist()
)

candidate_roots = [r for r in ROOTS if r != BASELINE]
n_panels = len(candidate_roots)

# Compute a shared x-range across all panels (clip extreme CIs symmetrically so
# the plot is readable; outliers beyond the limits are drawn as arrows).
all_diffs = []
all_los = []
all_his = []
for root_label in candidate_roots:
    sub = per_system_df[
        (per_system_df["root_label"] == root_label)
        & (per_system_df["horizon"] == H_FOREST)
    ]
    all_diffs.extend(sub["mean_diff_log10"].dropna().tolist())
    all_los.extend(sub["lo_log10"].dropna().tolist())
    all_his.extend(sub["hi_log10"].dropna().tolist())
xmin = -4.0
xmax = 4.0  # symmetric so 0 sits in the middle and direction reads clearly

fig, axes = plt.subplots(
    1,
    n_panels,
    figsize=(2.0 + 1.5 * n_panels, max(4.0, 0.30 * len(sys_order))),
    sharey=True,
    sharex=True,
)
if n_panels == 1:
    axes = [axes]

# Pretty system labels: drop "claude_" prefix
def pretty_sys(s: str) -> str:
    s = s.replace("claude_", "").replace("_", " ")
    return s


for ax, root_label in zip(axes, candidate_roots):
    meta = ROOTS[root_label]
    sub = per_system_df[
        (per_system_df["root_label"] == root_label)
        & (per_system_df["horizon"] == H_FOREST)
    ].set_index("system")
    color = PALETTE[meta["color"]]
    ys = np.arange(len(sys_order))
    diffs = sub.loc[sys_order, "median_diff_log10"].to_numpy()
    los = sub.loc[sys_order, "lo_log10"].to_numpy()
    his = sub.loc[sys_order, "hi_log10"].to_numpy()
    p_holm = sub.loc[sys_order, "p_holm_w"].to_numpy()

    for y, d, lo, hi, p in zip(ys, diffs, los, his, p_holm):
        if not np.isfinite(d):
            continue
        # Clip CI ends to plot bounds; track if an end is truncated.
        lo_c = max(lo, xmin) if np.isfinite(lo) else d
        hi_c = min(hi, xmax) if np.isfinite(hi) else d
        ax.errorbar(
            d if (xmin <= d <= xmax) else np.clip(d, xmin, xmax),
            y,
            xerr=[[max(0.0, d - lo_c)], [max(0.0, hi_c - d)]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=2,
            mfc=color if (np.isfinite(p) and p < ALPHA) else "white",
            mec=color,
            ms=4,
            lw=0.0,
        )
    ax.axvline(0.0, color="grey", lw=0.7, alpha=0.6)
    ax.set_yticks(ys)
    K = int((sub["p_holm_w"] < ALPHA).sum())
    N = int(sub["p_holm_w"].notna().sum())
    ax.set_title(f"{meta['label']}\n{K}/{N} systems", fontsize=8)
    ax.grid(True, axis="x", lw=0.4, alpha=0.4)

# Y labels only on the leftmost panel, with adequate left margin
axes[0].set_yticklabels([pretty_sys(s) for s in sys_order], fontsize=6.5)
axes[0].invert_yaxis()
axes[0].set_xlim(xmin, xmax)

# One shared x-label
fig.supxlabel(
    r"$\log_{10}$(candidate MSE / Dense MLP no-shrink MSE) at $H{=}1000$",
    fontsize=8,
)
fig.suptitle(
    f"Per-system paired Wilcoxon effect sizes (10 seeds/system, Holm-corrected at $\\alpha={ALPHA}$ across {BENCHMARK_N} systems)",
    fontsize=8.5,
    y=1.0,
)
# Filled marker = significant; open = not significant after Holm
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="grey", mfc="grey", mec="grey",
               ls="", label=f"Holm $p<{ALPHA}$"),
    plt.Line2D([0], [0], marker="o", color="grey", mfc="white", mec="grey",
               ls="", label="not significant"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=7,
    bbox_to_anchor=(0.5, -0.02),
)
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig(FIG_DIR / "fig_fixed17_forest.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_fixed17_forest.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------------------
# Boundary-only main table: 5 rows, no sampling column
# --------------------------------------------------------------------------------------

print("Building boundary-only 15-system table...")


def collect_row(root_label: str, label: str) -> dict:
    sub_fc = fc[fc["root_label"] == root_label]
    row: dict = {"label": label, "root": root_label}
    for h in HORIZONS:
        col = f"h{h}_best_periodic_mean"
        per_sys = sub_fc.groupby("system_name")[col].apply(lambda v: iqm(v.to_numpy()))
        row[f"H{h}_mean"] = mean_finite(per_sys.to_numpy())
        row[f"H{h}_iqm"] = row[f"H{h}_mean"]
        row[f"H{h}_log10iqr"] = iqr_log10(per_sys.to_numpy())
    sub_itp = itp_deep[itp_deep["root_label"] == root_label]
    for metric, src in (
        ("HBgivenS", "h_basin_given_support"),
        ("HSgivenB", "h_support_given_basin"),
        ("U_exact", "u_exact"),
        ("MeanSupportSize", "mean_support_size"),
        ("UniqueSupportCount", "unique_support_count"),
        ("HBgivenF", "family_h_basin_given_family"),
        ("FamilyUniqueCount", "family_unique_count"),
        ("FreezeWrongH1", "support_freeze_wrong_over_base_h1"),
        ("FreezeWrongH20", "support_freeze_wrong_over_base_h20"),
    ):
        if src not in sub_itp.columns:
            row[metric] = float("nan")
            row[f"{metric}_std"] = float("nan")
            continue
        if metric == "FamilyUniqueCount":
            per_sys = sub_itp.groupby("system_name")[src].mean()
            row[metric] = float(np.mean(per_sys.to_numpy()))
        else:
            per_sys = sub_itp.groupby("system_name")[src].apply(lambda v: iqm(v.to_numpy()))
            row[metric] = mean_finite(per_sys.to_numpy())
        row[f"{metric}_std"] = float(np.std(per_sys.to_numpy(), ddof=1))
    return row


main_table_rows = [collect_row(rl, ROOTS[rl]["label"]) for rl in ROOTS.keys()]
main_table_df = pd.DataFrame(main_table_rows)
main_table_df.to_csv(TBL_DIR / "table1_boundary_only.csv", index=False)


# Helper: per-system summary lookup keyed by (root_label, horizon) for K/N + median
sys_summary_lookup = {}
for row in summary_rows:
    sys_summary_lookup[(row["root_label"], row["horizon"])] = row


# Convert to LaTeX-friendly numeric strings; mark "K/N" significance per cell.
def fmt_num(v: float, sig: int = 3) -> str:
    if not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    abs_v = abs(v)
    if abs_v >= 1000.0 or abs_v < 1e-3:
        exponent = math.floor(math.log10(abs_v))
        mantissa = v / (10.0**exponent)
        return rf"{mantissa:.{sig - 1}f}{{\times}}10^{{{exponent}}}"
    decimals = max(sig - 1 - math.floor(math.log10(abs_v)), 0)
    return f"{v:.{decimals}f}"


def fmt_kn(K: int, N: int) -> str:
    return f"{K}/{N}"


# Build the per-row sig stats keyed by (root, H)
sig_lookup = {}
for row in summary_rows:
    sig_lookup[(row["root_label"], row["horizon"])] = row

lines = []
lines.append(r"\begin{tabular}{@{}l rrr rrr@{}}")
lines.append(r"\toprule")
lines.append(
    r"Model & H100 & H500 & H1000 & $H(B\!\mid\!S)$ & $H(S\!\mid\!B)$ & $U_{\rm exact}$ \\"
)
lines.append(r"\midrule")

for row in main_table_rows:
    label = row["label"]
    cells = []
    for h in HORIZONS:
        iqm_v = row[f"H{h}_iqm"]
        iqr_v = row[f"H{h}_log10iqr"]
        if row["root"] == BASELINE:
            extra = "--"
        else:
            sig = sig_lookup[(row["root"], h)]
            extra = f"{sig['K_systems_better']}/{sig['N_systems_tested']}"
        cell = f"${fmt_num(iqm_v)}$"
        if np.isfinite(iqr_v):
            cell += f"\\,(${iqr_v:.2f}$)"
        cell += f"\\,[{extra}]"
        cells.append(cell)
    cells_main = " & ".join(cells)
    cells_intp = []
    for m in ("HBgivenS", "HSgivenB", "U_exact"):
        v = row[m]
        s = row[f"{m}_std"]
        if not np.isfinite(v):
            cells_intp.append("--")
        elif np.isfinite(s):
            cells_intp.append(f"${fmt_num(v, sig=2)}$\\,(${s:.2f}$)")
        else:
            cells_intp.append(f"${fmt_num(v, sig=2)}$")
    cells_intp = " & ".join(cells_intp)
    lines.append(f"{label} & {cells_main} & {cells_intp} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
(TBL_DIR / "table1_boundary_only.tex").write_text("\n".join(lines))


# --------------------------------------------------------------------------------------
# Option A: per-system K/N + median effect ONLY in forecasting cells.
# Absolute IQM scale carried by the Dense MLP no-shrink baseline row.
# --------------------------------------------------------------------------------------

linesA: list[str] = []
linesA.append(r"\begin{tabular}{@{}l rrr rrr@{}}")
linesA.append(r"\toprule")
linesA.append(
    rf"Model & H100 $K/{BENCHMARK_N}$ & H500 $K/{BENCHMARK_N}$ & H1000 $K/{BENCHMARK_N}$ & $H(B\!\mid\!S)$ & $H(S\!\mid\!B)$ & $U_{{\rm exact}}$ \\"
)
linesA.append(r"\midrule")
for row in main_table_rows:
    label = row["label"]
    cells: list[str] = []
    for h in HORIZONS:
        if row["root"] == BASELINE:
            iqm_v = row[f"H{h}_iqm"]
            cells.append(f"\\emph{{IQM ${fmt_num(iqm_v)}$}} (baseline)")
        else:
            sig = sig_lookup[(row["root"], h)]
            kn = f"{sig['K_systems_better']}/{sig['N_systems_tested']}"
            med = sig["median_diff_log10"]
            cells.append(f"$\\mathbf{{{kn}}}$\\, (${med:+.2f}$)")
    cells_main = " & ".join(cells)
    cells_intp = []
    for m in ("HBgivenS", "HSgivenB", "U_exact"):
        v = row[m]
        s = row[f"{m}_std"]
        if not np.isfinite(v):
            cells_intp.append("--")
        elif np.isfinite(s):
            cells_intp.append(f"${fmt_num(v, sig=2)}$\\,(${s:.2f}$)")
        else:
            cells_intp.append(f"${fmt_num(v, sig=2)}$")
    cells_intp = " & ".join(cells_intp)
    linesA.append(f"{label} & {cells_main} & {cells_intp} \\\\")
linesA.append(r"\bottomrule")
linesA.append(r"\end{tabular}")
(TBL_DIR / "table1_optionA.tex").write_text("\n".join(linesA))


# --------------------------------------------------------------------------------------
# Option B: K/N as primary (large), IQM as small secondary, on every cell.
# Entropy/U_exact columns get K/N as well; H(B|S_abs) is saturated so no K/N.
# Best value per column is underlined.
# --------------------------------------------------------------------------------------

# Determine "best" indices per column.
# For forecasting columns (H100, H500, H1000): higher K/N first, lower IQM as tiebreak.
# For H(B|S_abs): lower IQM (everyone is 0 for sparse rows so they tie).
# For H(S_abs|B): lower IQM is better.
# For U_exact: higher IQM is better.

candidate_indices = [
    i for i, r in enumerate(main_table_rows) if r["root"] != BASELINE
]


def best_idx_forecast(h: int) -> set[int]:
    best = []
    best_score = None  # (-K, IQM) — minimize lex order
    for i in candidate_indices:
        r = main_table_rows[i]
        sig = sig_lookup[(r["root"], h)]
        score = (-sig["K_systems_better"], r[f"H{h}_iqm"])
        if best_score is None or score < best_score:
            best_score = score
            best = [i]
        elif score == best_score:
            best.append(i)
    return set(best)


def best_idx_entropy(metric: str, lower_is_better: bool) -> set[int]:
    best: list[int] = []
    best_val = None
    for i in candidate_indices:
        v = main_table_rows[i][metric]
        if not np.isfinite(v):
            continue
        if best_val is None:
            best_val = v
            best = [i]
        else:
            if (lower_is_better and v < best_val - 1e-9) or (
                not lower_is_better and v > best_val + 1e-9
            ):
                best_val = v
                best = [i]
            elif abs(v - best_val) < 1e-9:
                best.append(i)
    return set(best)


best_h100 = best_idx_forecast(100)
best_h500 = best_idx_forecast(500)
best_h1000 = best_idx_forecast(1000)
best_HSgivenB = best_idx_entropy("HSgivenB", lower_is_better=True)
best_U_exact = best_idx_entropy("U_exact", lower_is_better=False)


def underline(s: str) -> str:
    return f"\\underline{{{s}}}"


# Best per column: |S_abs| (lower better), H(B|F_abs) (lower better),
# freeze-wrong h1 and h20 (higher better — wrong support hurts more = good).
best_MeanSupportSize = best_idx_entropy("MeanSupportSize", lower_is_better=True)
best_HBgivenF = best_idx_entropy("HBgivenF", lower_is_better=True)
best_FreezeWrongH1 = best_idx_entropy("FreezeWrongH1", lower_is_better=False)
best_FreezeWrongH20 = best_idx_entropy("FreezeWrongH20", lower_is_better=False)


def fmt_ratio(v: float) -> str:
    """Compact display for the wrong-over-base ratio (can span 0.5 to 10^4)."""
    if not np.isfinite(v):
        return "--"
    return f"${fmt_num(v)}$"


# Cell layout: metric value first, then [K/N] in brackets to the right.
# Up arrow next to column title = higher is better; down arrow = lower is better.
DOWN = r"$\downarrow$"  # lower is better
UP = r"$\uparrow$"      # higher is better


def kn_brackets(K: int, N: int) -> str:
    return f"\\,$\\bigl[\\,\\mathbf{{{K}/{N}}}\\,\\bigr]$"


linesB: list[str] = []
linesB.append(r"\begin{tabular}{@{}l rrr r r r rr@{}}")
linesB.append(r"\toprule")
linesB.append(
    rf"Model & H100\,{DOWN} & H500\,{DOWN} & H1000\,{DOWN} & $|S_{{\rm abs}}|$\,{DOWN} & $H(B\!\mid\!S_{{\rm abs}})$\,{DOWN} & $H(B\!\mid\!F_{{\rm abs}})$\,{DOWN} & wr.\,$h{{=}}1$\,{UP} & wr.\,$h{{=}}20$\,{UP} \\"
)
linesB.append(r"\midrule")
for i, row in enumerate(main_table_rows):
    label = row["label"]
    is_baseline = row["root"] == BASELINE
    cells = []
    for h, best_set in zip(HORIZONS, (best_h100, best_h500, best_h1000)):
        iqm_v = row[f"H{h}_iqm"]
        if is_baseline:
            cells.append(f"${fmt_num(iqm_v)}$\\,\\emph{{[baseline]}}")
        else:
            sig = sig_lookup[(row["root"], h)]
            primary = f"${fmt_num(iqm_v)}$"
            if i in best_set:
                primary = underline(primary)
            cells.append(f"{primary}{kn_brackets(sig['K_systems_better'], sig['N_systems_tested'])}")
    cells_main = " & ".join(cells)

    # Mean support size column
    size_v = row["MeanSupportSize"]
    unique_v = row["UniqueSupportCount"]
    if not np.isfinite(size_v):
        size_cell = "--"
    else:
        size_primary = f"${size_v:.0f}$"
        if i in best_MeanSupportSize:
            size_primary = underline(size_primary)
        if is_baseline:
            size_cell = f"{size_primary}\\,\\emph{{[baseline]}}"
        else:
            es = entropy_summary.get((row["root"], "MeanSupportSize"))
            kn_str = (
                kn_brackets(es["K_systems_better"], es["N_systems_tested"]) if es else ""
            )
            size_cell = f"{size_primary}{kn_str}"

    # H(B|S_abs) -- saturated, no K/N
    hbs_v = row["HBgivenS"]
    hbs_cell = f"${fmt_num(hbs_v, sig=2)}$" if np.isfinite(hbs_v) else "--"

    # H(B|F_abs) cell
    hbf_v = row["HBgivenF"]
    fam_uniq = row["FamilyUniqueCount"]
    if not np.isfinite(hbf_v):
        hbf_cell = "--"
    else:
        hbf_primary = f"${fmt_num(hbf_v, sig=2)}$"
        if i in best_HBgivenF:
            hbf_primary = underline(hbf_primary)
        fam_part = f"\\,{{\\scriptsize $|F_{{\\rm abs}}|{{=}}{fam_uniq:.0f}$}}" if np.isfinite(fam_uniq) else ""
        if is_baseline:
            hbf_cell = f"{hbf_primary}{fam_part}\\,\\emph{{[baseline]}}"
        else:
            es = entropy_summary.get((row["root"], "HBgivenF"))
            kn_str = (
                kn_brackets(es["K_systems_better"], es["N_systems_tested"]) if es else ""
            )
            hbf_cell = f"{hbf_primary}{fam_part}{kn_str}"

    # Freeze-wrong h1 and h20
    fw_cells = []
    for m, best_set in (("FreezeWrongH1", best_FreezeWrongH1), ("FreezeWrongH20", best_FreezeWrongH20)):
        v = row[m]
        if not np.isfinite(v):
            fw_cells.append("--")
            continue
        primary = fmt_ratio(v)
        if i in best_set:
            primary = underline(primary)
        if is_baseline:
            fw_cells.append(f"{primary}\\,\\emph{{[baseline]}}")
            continue
        es = entropy_summary.get((row["root"], m))
        kn_str = kn_brackets(es["K_systems_better"], es["N_systems_tested"]) if es else ""
        fw_cells.append(f"{primary}{kn_str}")

    linesB.append(f"{label} & {cells_main} & {size_cell} & {hbs_cell} & {hbf_cell} & {' & '.join(fw_cells)} \\\\")
linesB.append(r"\bottomrule")
linesB.append(r"\end{tabular}")
(TBL_DIR / "table1_optionB.tex").write_text("\n".join(linesB))


# --------------------------------------------------------------------------------------
# Main-text Table 1 fragment: current layout with dense-K LISTA included.
# --------------------------------------------------------------------------------------

def metric_cell(
    row: dict,
    metric: str,
    best_set: set[int],
    row_index: int,
    *,
    sig_key: str | None = None,
    is_ratio: bool = False,
    sig: int = 3,
) -> str:
    v = row[metric]
    if not np.isfinite(v):
        return "--"
    if is_ratio:
        primary = fmt_ratio(v)
    else:
        body = fmt_num(v, sig=sig)
        if row_index in best_set:
            body = rf"\mathbf{{{body}}}"
        primary = f"${body}$"
    if row["root"] == BASELINE:
        return primary + r"\,\emph{[baseline]}"
    if sig_key is None:
        return primary
    es = entropy_summary.get((row["root"], sig_key))
    if es is None:
        return primary
    return primary + rf"\,[{es['K_systems_better']}/{es['N_systems_tested']}]"


table1_lines: list[str] = [
    r"\begin{tabular}{@{}l rr rrr r@{}}",
    r"\toprule",
    r"& \multicolumn{2}{c}{Forecasting (all held-out)} & \multicolumn{4}{c}{Support diagnostics (global deep slice)} \\",
    r"\cmidrule(lr){2-3}\cmidrule(l){4-7}",
    r"Model & H100\,$\downarrow$ & H1000\,$\downarrow$ & $H(B\!\mid\!F_{\rm abs})$\,$\downarrow$ & wr.\,$h{=}1$\,$\uparrow$ & wr.\,$h{=}20$\,$\uparrow$ & $\overline{|F_{\rm abs}|}$ \\",
    r"\midrule",
]

stage2_summary_by_horizon: dict[int, dict] = {}
if STAGE2_LOCAL_MAP_SUMMARY_CSV.exists():
    stage2_summary = pd.read_csv(STAGE2_LOCAL_MAP_SUMMARY_CSV)
    for _, stage2_row in stage2_summary.iterrows():
        stage2_summary_by_horizon[int(stage2_row["horizon"])] = stage2_row.to_dict()

for i, row in enumerate(main_table_rows):
    label = row["label"]
    is_baseline = row["root"] == BASELINE
    hbf_cell = metric_cell(
        row,
        "HBgivenF",
        best_HBgivenF,
        i,
        sig_key="HBgivenF",
        sig=3,
    )
    fw1_cell = metric_cell(
        row,
        "FreezeWrongH1",
        best_FreezeWrongH1,
        i,
        sig_key="FreezeWrongH1",
        is_ratio=True,
    )
    fw20_cell = metric_cell(
        row,
        "FreezeWrongH20",
        best_FreezeWrongH20,
        i,
        sig_key="FreezeWrongH20",
        is_ratio=True,
    )
    family_count = row["FamilyUniqueCount"]
    if np.isfinite(family_count):
        family_count_cell = f"${family_count:.1f}$"
        if is_baseline:
            family_count_cell += r"\,\emph{[baseline]}"
    else:
        family_count_cell = "--"
    forecast_cells = []
    for h, best_set in ((100, best_h100), (1000, best_h1000)):
        v = row[f"H{h}_iqm"]
        if not np.isfinite(v):
            forecast_cells.append("--")
            continue
        body = fmt_num(v)
        if i in best_set and not is_baseline:
            body = rf"\mathbf{{{body}}}"
        cell = f"${body}$"
        if is_baseline:
            cell += r"\,\emph{[baseline]}"
        else:
            sig_row = sig_lookup[(row["root"], h)]
            cell += rf"\,[{sig_row['K_systems_better']}/{sig_row['N_systems_tested']}]"
        forecast_cells.append(cell)
    table1_lines.append(
        f"{label} & {' & '.join(forecast_cells)} & {hbf_cell} & "
        f"{fw1_cell} & {fw20_cell} & {family_count_cell} \\\\"
    )
    if row["root"] == "lista_dense_signsplit_p256_hardinit_basin_partition" and stage2_summary_by_horizon:
        stage2_cells = []
        for h in (100, 1000):
            stage2_row = stage2_summary_by_horizon.get(h)
            if stage2_row is None:
                stage2_cells.append("--")
                continue
            v = float(stage2_row["stage2_mean_over_system_iqm"])
            k = int(stage2_row["within_system_holm_passes"])
            n = int(stage2_row["within_system_holm_total"])
            stage2_cells.append(f"${fmt_num(v)}$\\,[{k}/{n}]")
        table1_lines.append(
            r"LISTA + local $K_c$ & "
            + " & ".join(stage2_cells)
            + r" & \multicolumn{4}{c@{}}{same frozen encoder/support diagnostics as LISTA} \\"
        )

table1_lines.extend([r"\bottomrule", r"\end{tabular}"])
(TBL_DIR / "table1_fixed17_alignment.tex").write_text("\n".join(table1_lines))


# --------------------------------------------------------------------------------------
# Regenerate retained-benchmark figures from the current Table 1 packet.
# --------------------------------------------------------------------------------------

print("Building 15-system horizon and diagnostic figures...")

DIAGNOSTIC_METRICS = [
    ("family_h_basin_given_family", r"$H(B\mid F_{\rm abs})$", "strip"),
    ("family_unique_count", r"$|F_{\rm abs}|$", "strip_mean"),
    ("support_freeze_wrong_over_base_h1", r"wrong-support ratio $h{=}1$", "strip_log"),
]


def plot_current_horizon_curve() -> None:
    horizons = ALL_FORECAST_HORIZONS or HORIZONS
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.2), constrained_layout=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED + 155921)
    for root_label, meta in ROOTS.items():
        sub = fc[fc["root_label"] == root_label]
        if sub.empty:
            continue
        ys, ylo, yhi = [], [], []
        for h in horizons:
            col = f"h{h}_best_periodic_mean"
            system_seed_values = [
                finite_positive(group[col])
                for _, group in sub.groupby("system_name", sort=True)
            ]
            system_seed_values = [values for values in system_seed_values if values.size > 0]
            per_system_iqms = np.asarray([iqm(values) for values in system_seed_values], dtype=float)
            center = mean_finite(per_system_iqms)
            ys.append(center)
            ci_low, ci_high, _ = fixed_system_seed_bootstrap_log_relative_band(
                system_seed_values,
                center=center,
                rng=rng,
                n_reps=BOOTSTRAP_REPS,
            )
            ylo.append(ci_low)
            yhi.append(ci_high)
        color = PALETTE[meta["color"]]
        linestyle = "-" if "LISTA" in meta["label"] else "--"
        ax.fill_between(horizons, ylo, yhi, color=color, alpha=0.17, lw=0)
        ax.plot(
            horizons,
            ys,
            marker="o",
            lw=2.1,
            ms=5.5,
            color=color,
            linestyle=linestyle,
            label=meta["label"],
        )
    ax.set_title(f"{BENCHMARK_N}-system multibasin forecasting\nperformance", fontsize=11, pad=4)
    ax.set_xlabel(r"Rollout horizon $H$ (observation steps)", fontsize=13)
    ax.set_ylabel("MSE (mean over system seed-IQMs)", fontsize=HORIZON_YLABEL_FONTSIZE)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)
    ax.set_xlim(min(horizons) - 70, max(horizons) + 180)
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons], rotation=30, ha="right", fontsize=10.5)
    ax.tick_params(axis="y", labelsize=HORIZON_YTICK_LABELSIZE)
    ax.grid(True, which="both", lw=0.45, alpha=0.38)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5, ncol=2)
    for stem in ("fig_fixed17_horizon_curves", "fig_fixed17_horizon_curves_boundary_only"):
        fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_seed_strips() -> None:
    horizon = 1000
    col = f"h{horizon}_best_periodic_per_dim_mean"
    if col not in fc.columns:
        col = f"h{horizon}_best_periodic_mean"
    labels, values, colors, iqm_points = [], [], [], []
    for root_label, meta in ROOTS.items():
        sub = fc[fc["root_label"] == root_label]
        vals = pd.to_numeric(sub[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        vals = vals[vals > 0]
        labels.append(meta["label"])
        values.append(vals)
        colors.append(PALETTE[meta["color"]])
        iqm_points.append(iqm(vals))
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.2))
    rng = np.random.default_rng(7)
    for i, vals in enumerate(values):
        if vals.size == 0:
            continue
        x = i + rng.uniform(-0.18, 0.18, size=vals.size)
        ax.scatter(x, vals, s=8, color=colors[i], alpha=0.32, linewidths=0)
        ax.hlines(iqm_points[i], i - 0.28, i + 0.28, color="black", lw=1.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(rf"H{horizon} MSE per dimension")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", lw=0.35, alpha=0.35)
    fig.tight_layout()
    for stem in ("fig_fixed17_seed_strips", "fig_fixed17_seed_strips_boundary_only"):
        fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_entropy_strips() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), constrained_layout=True)
    rng = np.random.default_rng(11)
    for ax, (metric, label, kind) in zip(axes, DIAGNOSTIC_METRICS):
        xlabels = []
        for i, (root_label, meta) in enumerate(ROOTS.items()):
            sub = itp_deep[itp_deep["root_label"] == root_label]
            vals = pd.to_numeric(sub[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            color = PALETTE[meta["color"]]
            if kind == "strip_log":
                vals = vals[vals > 0]
            cap = 6.0 if kind == "strip_mean" else None
            plot_vals = vals if cap is None else np.minimum(vals, cap)
            overflow = np.zeros(vals.shape, dtype=bool) if cap is None else vals > cap
            x = i + rng.uniform(-0.18, 0.18, size=vals.size)
            if vals.size:
                q1, q3 = np.percentile(vals, [25, 75])
                q1_plot = min(q1, cap) if cap is not None else q1
                q3_plot = min(q3, cap) if cap is not None else q3
                ax.vlines(i, q1_plot, q3_plot, color="0.25", alpha=0.9, lw=1.5, zorder=2.5)
                ax.hlines([q1_plot, q3_plot], i - 0.16, i + 0.16, color="0.25", alpha=0.9, lw=1.5, zorder=2.6)
                ax.scatter(x[~overflow], plot_vals[~overflow], s=8, color=color, alpha=0.32, linewidths=0, zorder=2)
                if np.any(overflow):
                    ax.scatter(
                        x[overflow],
                        plot_vals[overflow],
                        s=12,
                        color=color,
                        alpha=0.55,
                        marker="^",
                        linewidths=0,
                        zorder=2.2,
                        clip_on=False,
                    )
                summary = float(np.mean(vals)) if kind == "strip_mean" else iqm(vals)
                summary = min(summary, cap) if cap is not None and np.isfinite(summary) else summary
                ax.hlines(summary, i - 0.25, i + 0.25, color="black", lw=1.6, zorder=3)
            xlabels.append(meta["label"])
        if kind == "strip_log":
            ax.set_yscale("log")
        if kind == "strip_mean":
            ax.axhline(4.20, color="0.45", lw=0.9, ls="--", zorder=1)
            ax.set_ylim(0.75, 6.35)
            ax.set_yticks(np.arange(1, 7))
            ax.text(
                0.98,
                0.97,
                "triangles: >6",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="0.35",
            )
        ax.set_title(label)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right")
        ax.grid(axis="y", lw=0.35, alpha=0.35)
    for stem in ("fig_fixed17_entropy_strips", "fig_fixed17_entropy_strips_boundary_only", "fig_fixed17_entropy_strips_alt2"):
        fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_appendix_histograms() -> None:
    horizons = HORIZONS
    fig, axes = plt.subplots(1, len(horizons), figsize=(9.2, 2.8), constrained_layout=True)
    for ax, h in zip(np.ravel(axes), horizons):
        col = f"h{h}_best_periodic_mean"
        for root_label, meta in ROOTS.items():
            vals = pd.to_numeric(fc.loc[fc["root_label"] == root_label, col], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                ax.hist(np.log10(vals), bins=32, color=PALETTE[meta["color"]], alpha=0.35, label=meta["label"])
        ax.set_title(f"H{h}")
        ax.set_xlabel(r"$\log_{10}$ MSE")
        ax.grid(axis="y", lw=0.3, alpha=0.25)
    axes[0].set_ylabel("# system-seed pairs")
    axes[-1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(FIG_DIR / "appfig_fixed17_perseed_histograms.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "appfig_fixed17_perseed_histograms.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    for ax, (metric, label, _) in zip(axes, DIAGNOSTIC_METRICS):
        for root_label, meta in ROOTS.items():
            vals = pd.to_numeric(itp_deep.loc[itp_deep["root_label"] == root_label, metric], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ax.hist(vals, bins=30, color=PALETTE[meta["color"]], alpha=0.35, label=meta["label"])
        ax.set_title(label)
        ax.grid(axis="y", lw=0.3, alpha=0.25)
    axes[0].set_ylabel("# system-seed pairs")
    axes[-1].legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(FIG_DIR / "appfig_fixed17_entropy_histograms.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "appfig_fixed17_entropy_histograms.png", bbox_inches="tight")
    plt.close(fig)


plot_current_horizon_curve()
plot_seed_strips()
plot_entropy_strips()
plot_appendix_histograms()


# --------------------------------------------------------------------------------------
# Per-system appendix table at H=1000: retained systems × candidates × (effect, p_Holm)
# --------------------------------------------------------------------------------------

print("Building per-system appendix table at H=1000...")

H_PERSYS_TABLE = 1000

# One column per candidate; each column has 2 sub-columns: effect (mean log10
# paired diff) and Holm-corrected Wilcoxon p
candidate_roots_for_table = [r for r in ROOTS if r != BASELINE]


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "--"
    if p < 1e-3:
        return f"$<\\!10^{{-3}}$"
    return f"${p:.3f}$"


def fmt_eff(d: float, sig: bool) -> str:
    if not np.isfinite(d):
        return "--"
    val = f"${d:+.2f}$"
    if sig:
        val = "\\mathbf{" + val.strip("$") + "}"
        val = f"$\\mathbf{{{d:+.2f}}}$"
    return val


def fmt_signed_sig(d: float) -> str:
    if not np.isfinite(d):
        return "--"
    if d == 0:
        return "+0.00"
    sign = "+" if d > 0 else "-"
    return sign + fmt_num(abs(d))


# Build column header (4 candidates × 2 sub-cols)
n_cands = len(candidate_roots_for_table)
col_spec = "@{}l " + " ".join(["rr"] * n_cands) + "@{}"
header_top = (
    "System & "
    + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{ROOTS[r]['label']}}}"
        for r in candidate_roots_for_table
    )
    + " \\\\"
)
cmidrules = " ".join(
    f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
    for i in range(n_cands)
)
header_sub = (
    " & "
    + " & ".join(["$\\Delta$ & $p_{\\rm H}$"] * n_cands)
    + " \\\\"
)

lines: list[str] = []
lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
lines.append("\\toprule")
lines.append(header_top)
lines.append(cmidrules)
lines.append(header_sub)
lines.append("\\midrule")

# Sort systems by mean per-row effect (most-favorable first) for readability
sys_order_table = (
    per_system_df[per_system_df["horizon"] == H_PERSYS_TABLE]
    .groupby("system")["median_diff_log10"]
    .median()
    .sort_values()
    .index.tolist()
)


def pretty_sys_tex(s: str) -> str:
    return s.replace("claude_", "").replace("_", "\\_")


for sysname in sys_order_table:
    cells = [pretty_sys_tex(sysname)]
    for root_label in candidate_roots_for_table:
        rec = per_system_df[
            (per_system_df["root_label"] == root_label)
            & (per_system_df["horizon"] == H_PERSYS_TABLE)
            & (per_system_df["system"] == sysname)
        ]
        if rec.empty:
            cells.append("--")
            cells.append("--")
            continue
        d = rec["median_diff_log10"].iloc[0]
        p_h = rec["p_holm_w"].iloc[0]
        sig = (p_h is not None) and np.isfinite(p_h) and (p_h < ALPHA)
        if sig:
            cells.append(f"$\\mathbf{{{d:+.2f}}}$")
        else:
            cells.append(fmt_eff(d, False))
        cells.append(fmt_p(p_h if p_h is not None else float("nan")))
    lines.append(" & ".join(cells) + " \\\\")

# K/N summary row
lines.append("\\midrule")
summary_cells = [rf"\emph{{$K/{BENCHMARK_N}$ Holm $p<0.05$}}"]
for root_label in candidate_roots_for_table:
    sub = per_system_df[
        (per_system_df["root_label"] == root_label)
        & (per_system_df["horizon"] == H_PERSYS_TABLE)
    ]
    K = int((sub["p_holm_w"] < ALPHA).sum())
    N = int(sub["p_holm_w"].notna().sum())
    summary_cells.append(f"\\multicolumn{{2}}{{c}}{{$K{{=}}{K}\\,/\\,N{{=}}{N}$}}")
# Pad each multicolumn to span 2 sub-cols
summary_line = summary_cells[0]
for c in summary_cells[1:]:
    summary_line += " & " + c
lines.append(summary_line + " \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
(TBL_DIR / "table_persystem_h1000.tex").write_text("\n".join(lines))


# Same table at H=100, H=500 for completeness in the appendix
for H_VAL in (100, 500):
    lines2: list[str] = []
    lines2.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines2.append("\\toprule")
    lines2.append(header_top)
    lines2.append(cmidrules)
    lines2.append(header_sub)
    lines2.append("\\midrule")
    for sysname in sys_order_table:
        cells = [pretty_sys_tex(sysname)]
        for root_label in candidate_roots_for_table:
            rec = per_system_df[
                (per_system_df["root_label"] == root_label)
                & (per_system_df["horizon"] == H_VAL)
                & (per_system_df["system"] == sysname)
            ]
            if rec.empty:
                cells.append("--")
                cells.append("--")
                continue
            d = rec["mean_diff_log10"].iloc[0]
            p_h = rec["p_holm_w"].iloc[0]
            sig = (p_h is not None) and np.isfinite(p_h) and (p_h < ALPHA)
            if sig:
                cells.append(f"$\\mathbf{{{d:+.2f}}}$")
            else:
                cells.append(fmt_eff(d, False))
            cells.append(fmt_p(p_h if p_h is not None else float("nan")))
        lines2.append(" & ".join(cells) + " \\\\")
    lines2.append("\\midrule")
    summary_cells = [rf"\emph{{$K/{BENCHMARK_N}$ Holm $p<0.05$}}"]
    for root_label in candidate_roots_for_table:
        sub = per_system_df[
            (per_system_df["root_label"] == root_label)
            & (per_system_df["horizon"] == H_VAL)
        ]
        K = int((sub["p_holm_w"] < ALPHA).sum())
        N = int(sub["p_holm_w"].notna().sum())
        summary_cells.append(f"\\multicolumn{{2}}{{c}}{{$K{{=}}{K}\\,/\\,N{{=}}{N}$}}")
    summary_line = summary_cells[0]
    for c in summary_cells[1:]:
        summary_line += " & " + c
    lines2.append(summary_line + " \\\\")
    lines2.append("\\bottomrule")
    lines2.append("\\end{tabular}")
    (TBL_DIR / f"table_persystem_h{H_VAL}.tex").write_text("\n".join(lines2))


# --------------------------------------------------------------------------------------
# Per-system appendix tables for the new interpretability metrics
#  - mean_support_size (|S_abs|)
#  - family_h_basin_given_family (H(B|F_abs))
#  - support_freeze_wrong_over_base_h1
#  - support_freeze_wrong_over_base_h20
# Each table: retained systems × candidates, per-system median paired diff +
# Holm-corrected Wilcoxon p-value vs. the Dense MLP no-shrink baseline.
# We re-run the per-system Wilcoxon here so we have per-system records (the
# earlier entropy block only kept aggregate K/N, not per-system rows).
# --------------------------------------------------------------------------------------

print("Building per-system interpretability appendix tables...")

INTERP_APPENDIX_METRICS = {
    "MeanSupportSize": ("mean_support_size", "less", "$|S_{\\rm abs}|$ (mean active count)"),
    "HBgivenF": ("family_h_basin_given_family", "less", "$H(B\\mid F_{\\rm abs})$ (family-level)"),
    "FreezeWrongH1": ("support_freeze_wrong_over_base_h1", "greater", "wrong-support-freeze MSE ratio at $h{=}1$"),
    "FreezeWrongH20": ("support_freeze_wrong_over_base_h20", "greater", "wrong-support-freeze MSE ratio at $h{=}20$"),
}

# Re-collect per-system records for these metrics
appendix_persys: dict[str, list[dict]] = {k: [] for k in INTERP_APPENDIX_METRICS}
itp_deep_baseline = itp_deep[itp_deep["root_label"] == BASELINE]
for metric_label, (metric_col, alt, _name) in INTERP_APPENDIX_METRICS.items():
    base_pivot = itp_deep_baseline.pivot_table(
        index="system_name", columns="seed", values=metric_col, aggfunc="first"
    )
    for root_label, meta in ROOTS.items():
        if root_label == BASELINE:
            continue
        cand_pivot = (
            itp_deep[itp_deep["root_label"] == root_label]
            .pivot_table(index="system_name", columns="seed", values=metric_col, aggfunc="first")
        )
        common = cand_pivot.index.intersection(base_pivot.index)
        per_p = []
        per_diff = []
        for sysname in common:
            c = cand_pivot.loc[sysname].to_numpy(dtype=float)
            b = base_pivot.loc[sysname].to_numpy(dtype=float)
            mask = np.isfinite(c) & np.isfinite(b)
            c = c[mask]
            b = b[mask]
            if c.size < 4:
                per_p.append(np.nan)
                per_diff.append(np.nan)
                continue
            try:
                w = stats.wilcoxon(c, b, alternative=alt, zero_method="wilcox")
                per_p.append(float(w.pvalue))
            except ValueError:
                per_p.append(float("nan"))
            per_diff.append(float(np.median(c - b)))
        per_p = np.asarray(per_p)
        valid = np.isfinite(per_p)
        adj = np.full_like(per_p, np.nan)
        if valid.any():
            adj[valid] = holm(per_p[valid])
        for i, sysname in enumerate(common):
            appendix_persys[metric_label].append({
                "root_label": root_label,
                "label": meta["label"],
                "system": sysname,
                "p_holm": float(adj[i]) if np.isfinite(adj[i]) else None,
                "median_diff": float(per_diff[i]) if np.isfinite(per_diff[i]) else None,
            })


def write_persys_appendix_table(metric_label: str, alpha: float = ALPHA) -> Path:
    rows = appendix_persys[metric_label]
    df = pd.DataFrame(rows)
    candidate_roots_local = [r for r in ROOTS if r != BASELINE]
    sys_order_local = (
        df.groupby("system")["median_diff"].median().sort_values().index.tolist()
    )
    n_cands = len(candidate_roots_local)
    col_spec = "@{}l " + " ".join(["rr"] * n_cands) + "@{}"
    header_top = "System & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{ROOTS[r]['label']}}}" for r in candidate_roots_local
    ) + " \\\\"
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n_cands)
    )
    header_sub = " & " + " & ".join(["$\\Delta$ & $p_{\\rm H}$"] * n_cands) + " \\\\"

    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header_top,
        cmidrules,
        header_sub,
        "\\midrule",
    ]
    for sysname in sys_order_local:
        cells = [pretty_sys_tex(sysname)]
        for r in candidate_roots_local:
            rec = df[(df["root_label"] == r) & (df["system"] == sysname)]
            if rec.empty:
                cells.append("--")
                cells.append("--")
                continue
            d = rec["median_diff"].iloc[0]
            p = rec["p_holm"].iloc[0]
            sig = (p is not None) and np.isfinite(p) and (p < alpha)
            if metric_label.startswith("FreezeWrong"):
                body = fmt_signed_sig(d)
                cells.append(f"$\\mathbf{{{body}}}$" if sig else f"${body}$")
            else:
                cells.append(f"$\\mathbf{{{d:+.2f}}}$" if sig else fmt_eff(d, False))
            cells.append(fmt_p(p if p is not None else float("nan")))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    summary_cells = [rf"\emph{{$K/{BENCHMARK_N}$ Holm $p<0.05$}}"]
    for r in candidate_roots_local:
        sub = df[df["root_label"] == r]
        K = int((sub["p_holm"] < alpha).sum())
        N = int(sub["p_holm"].notna().sum())
        summary_cells.append(f"\\multicolumn{{2}}{{c}}{{$K{{=}}{K}\\,/\\,N{{=}}{N}$}}")
    summary_line = summary_cells[0]
    for c in summary_cells[1:]:
        summary_line += " & " + c
    lines.append(summary_line + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out = TBL_DIR / f"table_persystem_{metric_label}.tex"
    out.write_text("\n".join(lines))
    return out


for metric_label in INTERP_APPENDIX_METRICS:
    write_persys_appendix_table(metric_label)


print(f"Wrote per-system stats and boundary-only table.")
print(f"  - {TBL_DIR / 'persystem_wilcoxon_fixed17.json'}")
print(f"  - {TBL_DIR / 'persystem_wilcoxon_fixed17.csv'}")
print(f"  - {TBL_DIR / 'persystem_summary_fixed17.csv'}")
print(f"  - {TBL_DIR / 'table1_boundary_only.csv'}")
print(f"  - {TBL_DIR / 'table1_boundary_only.tex'}")
print(f"  - {TBL_DIR / 'table_persystem_h100.tex'}")
print(f"  - {TBL_DIR / 'table_persystem_h500.tex'}")
print(f"  - {TBL_DIR / 'table_persystem_h1000.tex'}")
for metric_label in INTERP_APPENDIX_METRICS:
    print(f"  - {TBL_DIR / f'table_persystem_{metric_label}.tex'}")
print(f"  - {FIG_DIR / 'fig_fixed17_forest.pdf'}")
