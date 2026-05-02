"""Compute paper-facing seed-stratified statistics and generate figures.

Outputs:
- docs/figures/neurips_paper_2026/*.pdf, *.png  (figures used in paper main text and appendix)
- docs/figures/neurips_paper_2026/_tables/*.json  (machine-readable tables for IQM±std cells)
- docs/figures/neurips_paper_2026/_tables/*.tex  (LaTeX-friendly numerical entries)

This script is read-only with respect to the experiment results: it only ingests CSVs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path("/home/mila/l/lia/skae")
FIG_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026"
TBL_DIR = FIG_DIR / "_tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------------------
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

# Colorblind-friendly palette
PALETTE = {
    "blockdiag_lista": "#0072B2",  # blue
    "softblock_lista": "#56B4E9",  # sky blue
    "sparse_mlp": "#009E73",        # green
    "zero_mlp": "#D55E00",          # vermilion
    "blockdiag_mlp": "#44AA99",     # teal
    "blockdiag_lista_sc6em3": "#882255",
    "blockdiag_lista_sc3em3": "#AA4499",
    "dense_lista": "#785EF0",
}

# --------------------------------------------------------------------------------------
# Statistical helpers
# --------------------------------------------------------------------------------------


def iqm(values: np.ndarray) -> float:
    """Interquartile mean: average of values within the 25-75 percentile range."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if values.size < 4:
        return float(np.mean(values))
    lo, hi = np.percentile(values, [25, 75])
    mask = (values >= lo) & (values <= hi)
    selected = values[mask]
    if selected.size == 0:
        return float(np.median(values))
    return float(np.mean(selected))


def iqm_std(values: np.ndarray) -> float:
    """Sample standard deviation of values used in the IQM (within IQR)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    lo, hi = np.percentile(values, [25, 75])
    selected = values[(values >= lo) & (values <= hi)]
    if selected.size <= 1:
        return 0.0
    return float(np.std(selected, ddof=1))


def std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def stratified_bootstrap_iqm(
    df: pd.DataFrame, value_col: str, n_resamples: int = 2000, rng_seed: int = 0
) -> tuple[float, float, float]:
    """Bootstrap IQM with confidence interval over (system, seed) pairs.

    Returns (iqm, lower 2.5%, upper 97.5%).
    """
    rng = np.random.default_rng(rng_seed)
    values = df[value_col].dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = iqm(values)
    bs = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=values.size, replace=True)
        bs[i] = iqm(sample)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def fmt(value: float, std_value: float | None = None, sig: int = 3) -> str:
    if not np.isfinite(value):
        return "--"
    if std_value is None or not np.isfinite(std_value):
        return f"{value:.{sig}g}"
    return f"{value:.{sig}g} \\pm {std_value:.{sig}g}"


def safe_log10(x: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x) & (x > 0)
    out[mask] = np.log10(x[mask])
    return out


# --------------------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------------------

FIXED17_FORECASTING_CSV = (
    ROOT
    / "results"
    / "transition_rich_basin_partition_final_seed10_20260409"
    / "collect_pass1"
    / "forecasting_rows.csv"
)

HARDINIT_FORECASTING_CSV = (
    ROOT
    / "results"
    / "transition_rich_hardinit_mlp_controls_seed10_20260416"
    / "collect_pass1"
    / "forecasting_rows.csv"
)

FIXED17_INTERP_CSV = (
    ROOT
    / "results"
    / "transition_rich_basin_partition_final_seed10_20260409"
    / "interpretability_final_pass1"
    / "interpretability_rows.csv"
)

HARDINIT_INTERP_CSV = (
    ROOT
    / "results"
    / "transition_rich_hardinit_mlp_controls_seed10_20260416"
    / "interpretability_final_pass1"
    / "interpretability_rows.csv"
)

SELF_ROUTED_CSV = (
    ROOT
    / "results"
    / "transition_rich_self_routed_forecasting_20260420"
    / "self_routed_forecasting_rows.csv"
)
SELF_ROUTED_CSVS = [
    SELF_ROUTED_CSV,
    ROOT
    / "results"
    / "transition_rich_self_routed_forecasting_hardinit_mlp_controls_seed0to9_20260428"
    / "self_routed_forecasting_rows.csv",
]
ROUTING_HORIZON = 1000

DYSTS_MAIN_CSV = (
    ROOT
    / "results"
    / "dysts_long_horizon_eval_20260414"
    / "collect"
    / "forecasting_rows.csv"
)

DYSTS_BLOCKDIAG_MLP_CSV = (
    ROOT
    / "results"
    / "dysts_long_horizon_eval_mlp_blockdiag_20260415"
    / "collect"
    / "forecasting_rows.csv"
)

PERIODIC_REFRESH_CSV = (
    ROOT
    / "results"
    / "periodic_support_refresh_fixed17_seed0_20260425"
    / "merged"
    / "periodic_support_refresh_rows.csv"
)


# Map raw root labels to (display_label, transition_structure, encoder, regime, color_key)
ROOT_LABELS_MAIN = {
    "lista_blockdiag_signsplit_hardinit_basin_partition": (
        "LISTA-BD",
        "block-diagonal",
        "LISTA",
        "boundary-emphasized",
        "blockdiag_lista",
    ),
    "lista_dense_softblock_signsplit_p64_hardinit_basin_partition": (
        "LISTA-SB",
        "soft-block",
        "LISTA",
        "boundary-emphasized",
        "softblock_lista",
    ),
    "mlp_sparse_basin_partition_control": (
        "Sparse MLP",
        "dense",
        "MLP",
        "standard",
        "sparse_mlp",
    ),
    "mlp_zero_sparse_basin_partition_control": (
        "Dense MLP",
        "dense",
        "MLP",
        "standard",
        "zero_mlp",
    ),
}

ROOT_LABELS_HARDINIT = {
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": (
        "Sparse MLP, BD",
        "block-diagonal",
        "MLP",
        "boundary-emphasized",
        "blockdiag_mlp",
    ),
    "mlp_sparse_hardinit_basin_partition_control": (
        "Sparse MLP",
        "dense",
        "MLP",
        "boundary-emphasized",
        "sparse_mlp",
    ),
    "mlp_zero_sparse_hardinit_basin_partition_control": (
        "Dense MLP",
        "dense",
        "MLP",
        "boundary-emphasized",
        "zero_mlp",
    ),
}

# Unified boundary-only roster: LISTA-BD + LISTA-SB (from main pass1) plus the
# three matched-boundary MLP controls (from hardinit pass1). All five rows are
# trained under the same boundary-emphasized sampling regime.
ROOT_LABELS_BOUNDARY_ONLY = {
    **{
        rl: meta
        for rl, meta in ROOT_LABELS_MAIN.items()
        if meta[2] == "LISTA"
    },
    **ROOT_LABELS_HARDINIT,
}

ROOT_LABELS_DYSTS = {
    "lista_dense_promoted_stage4": ("LISTA-D", "dense", "LISTA", "dense_lista"),
    "lista_blockdiag_ns200k_denseopt_sc3em3": (
        "LISTA-BD (low sp)",
        "block-diagonal",
        "LISTA",
        "blockdiag_lista_sc3em3",
    ),
    "lista_blockdiag_ns200k_denseopt_sc6em3": (
        "LISTA-BD (high sp)",
        "block-diagonal",
        "LISTA",
        "blockdiag_lista_sc6em3",
    ),
    "generic_sparse_ns200k_best": ("Sparse MLP", "dense", "MLP", "sparse_mlp"),
    "generic_sparse_sc0_ns200k_best": ("Dense MLP", "dense", "MLP", "zero_mlp"),
    "generic_sparse_blockdiag_ns200k_sc3em3": (
        "Sparse MLP, BD (low sp)",
        "block-diagonal",
        "MLP",
        "blockdiag_mlp",
    ),
    "generic_sparse_blockdiag_ns200k_sc6em3": (
        "Sparse MLP, BD (high sp)",
        "block-diagonal",
        "MLP",
        "blockdiag_lista_sc3em3",
    ),
}

# --------------------------------------------------------------------------------------
# Table 1: Fixed-17 alignment + forecasting (IQM ± std)
# --------------------------------------------------------------------------------------

print("Loading Fixed-17 forecasting + interpretability...")
fc_main = pd.read_csv(FIXED17_FORECASTING_CSV, low_memory=False)
fc_hardinit = pd.read_csv(HARDINIT_FORECASTING_CSV, low_memory=False)
itp_main = pd.read_csv(FIXED17_INTERP_CSV, low_memory=False)
itp_hardinit = pd.read_csv(HARDINIT_INTERP_CSV, low_memory=False)

# Restrict to selected support slice for entropy: absolute:0.001 / deep
HORIZONS_FIXED17 = [100, 500, 1000]
DEEP_INTERP_MASK = (itp_main["support_scheme"] == "absolute:0.001") & (
    itp_main["subset"] == "deep"
)
DEEP_INTERP_HARDINIT_MASK = (itp_hardinit["support_scheme"] == "absolute:0.001") & (
    itp_hardinit["subset"] == "deep"
)


def collect_fixed17_table(
    fc_df: pd.DataFrame,
    itp_df: pd.DataFrame,
    deep_mask: pd.Series,
    label_map: dict,
) -> pd.DataFrame:
    """Per-cell IQM with bootstrap 95% CI; std reported across system-level IQM-over-seeds."""
    rows = []
    interp_deep = itp_df[deep_mask].copy()
    for root_label, (
        display,
        transition,
        encoder,
        regime,
        color,
    ) in label_map.items():
        sub_fc = fc_df[fc_df["root_label"] == root_label]
        sub_itp = interp_deep[interp_deep["root_label"] == root_label]
        if sub_fc.empty:
            continue
        row = {
            "root_label": root_label,
            "display_label": display,
            "transition": transition,
            "encoder": encoder,
            "regime": regime,
            "color": color,
        }

        for h in HORIZONS_FIXED17:
            col = f"h{h}_best_periodic_mean"
            sys_iqm = sub_fc.groupby("system_key")[col].apply(
                lambda v: iqm(v.to_numpy())
            )
            sys_iqm = sys_iqm.replace([np.inf, -np.inf], np.nan).dropna()

            # Cross-system IQM (point estimate over per-system summaries)
            iqm_pt = iqm(sys_iqm.to_numpy())
            sd_sys = std(sys_iqm.to_numpy())
            # Bootstrap CI on per-system IQM aggregation (resample systems)
            point, lo, hi = stratified_bootstrap_iqm(
                pd.DataFrame({"v": sys_iqm.to_numpy()}),
                "v",
                n_resamples=2000,
                rng_seed=hash(root_label + str(h)) & 0xFFFF,
            )

            # Per-cell std across (system, seed): exclude any non-finite values
            cell_vals = sub_fc[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            row[f"H{h}_iqm"] = iqm_pt
            row[f"H{h}_iqm_ci_lo"] = lo
            row[f"H{h}_iqm_ci_hi"] = hi
            row[f"H{h}_std_systems"] = sd_sys
            # log-space std as a robust proxy when raw std is dominated by outliers
            row[f"H{h}_log10_std_systems"] = std(np.log10(np.clip(sys_iqm.to_numpy(), 1e-12, None)))
            row[f"H{h}_n_systems"] = int(sys_iqm.size)
            row[f"H{h}_n_seed_system_pairs"] = int(cell_vals.size)

        # Interpretability metrics on deep slice
        for col in ("h_basin_given_support", "h_support_given_basin", "u_exact"):
            sys_iqm = sub_itp.groupby("system_key")[col].apply(
                lambda v: iqm(v.to_numpy())
            )
            sys_iqm = sys_iqm.replace([np.inf, -np.inf], np.nan).dropna()
            iqm_pt = iqm(sys_iqm.to_numpy())
            sd_sys = std(sys_iqm.to_numpy())
            point, lo, hi = stratified_bootstrap_iqm(
                pd.DataFrame({"v": sys_iqm.to_numpy()}),
                "v",
                n_resamples=2000,
                rng_seed=hash(root_label + col) & 0xFFFF,
            )
            row[f"{col}_iqm"] = iqm_pt
            row[f"{col}_iqm_ci_lo"] = lo
            row[f"{col}_iqm_ci_hi"] = hi
            row[f"{col}_std_systems"] = sd_sys
            row[f"{col}_n_systems"] = int(sys_iqm.size)

        rows.append(row)
    return pd.DataFrame(rows)


tbl1_main = collect_fixed17_table(fc_main, itp_main, DEEP_INTERP_MASK, ROOT_LABELS_MAIN)
tbl1_hardinit = collect_fixed17_table(
    fc_hardinit, itp_hardinit, DEEP_INTERP_HARDINIT_MASK, ROOT_LABELS_HARDINIT
)

(TBL_DIR / "table1_main_iqm.csv").write_text(tbl1_main.to_csv(index=False))
(TBL_DIR / "table1_hardinit_iqm.csv").write_text(tbl1_hardinit.to_csv(index=False))


# --------------------------------------------------------------------------------------
# Paired Wilcoxon significance for forecasting (sparse-latent vs zero-sparse MLP control)
# --------------------------------------------------------------------------------------


def paired_wilcoxon_forecasting(fc_df: pd.DataFrame, label_map: dict, baseline_root: str) -> dict:
    """Per-system paired Wilcoxon: candidate vs baseline, paired across systems by IQM-over-seeds."""
    pairs: dict[str, dict[str, float]] = {}
    if baseline_root not in fc_df["root_label"].unique():
        return pairs
    for root_label in label_map:
        if root_label == baseline_root:
            continue
        out: dict[str, float] = {}
        for h in HORIZONS_FIXED17:
            col = f"h{h}_best_periodic_mean"
            cand = fc_df[fc_df["root_label"] == root_label].groupby("system_key")[col].apply(
                lambda v: iqm(v.to_numpy())
            )
            base = fc_df[fc_df["root_label"] == baseline_root].groupby("system_key")[col].apply(
                lambda v: iqm(v.to_numpy())
            )
            joined = pd.concat({"c": cand, "b": base}, axis=1).dropna()
            if joined.empty:
                continue
            try:
                stat = stats.wilcoxon(joined["c"], joined["b"], alternative="less", zero_method="wilcox")
                out[f"H{h}_pvalue"] = float(stat.pvalue)
                out[f"H{h}_n_systems"] = int(len(joined))
                out[f"H{h}_median_diff"] = float(np.median(joined["c"] - joined["b"]))
            except ValueError:
                out[f"H{h}_pvalue"] = float("nan")
                out[f"H{h}_n_systems"] = int(len(joined))
        pairs[root_label] = out
    return pairs


# We anchor against Dense MLP (zero-sparsity ReLU) in the locked packet
sig_main = paired_wilcoxon_forecasting(
    fc_main, ROOT_LABELS_MAIN, "mlp_zero_sparse_basin_partition_control"
)
sig_hardinit = paired_wilcoxon_forecasting(
    fc_hardinit,
    ROOT_LABELS_HARDINIT,
    "mlp_zero_sparse_hardinit_basin_partition_control",
)
(TBL_DIR / "wilcoxon_significance_main.json").write_text(json.dumps(sig_main, indent=2))
(TBL_DIR / "wilcoxon_significance_hardinit.json").write_text(json.dumps(sig_hardinit, indent=2))

# --------------------------------------------------------------------------------------
# Figure 1: Fixed-17 forecasting horizon curves with bootstrap IQM CI
# --------------------------------------------------------------------------------------

print("Building Figure: Fixed-17 forecasting curves...")


def build_horizon_curve(fc_df: pd.DataFrame, label_map: dict, ax, title: str):
    horizons = HORIZONS_FIXED17
    for root_label, (display, transition, encoder, regime, color) in label_map.items():
        sub = fc_df[fc_df["root_label"] == root_label]
        if sub.empty:
            continue
        iqms, lo_arr, hi_arr = [], [], []
        for h in horizons:
            col = f"h{h}_best_periodic_mean"
            sys_iqm = sub.groupby("system_key")[col].apply(lambda v: iqm(v.to_numpy()))
            sys_iqm = sys_iqm.replace([np.inf, -np.inf], np.nan).dropna()
            iqms.append(iqm(sys_iqm.to_numpy()))
            # Use 25-75 percentile band on the system-level distribution (robust to a few catastrophic systems).
            v = sys_iqm.to_numpy()
            v = v[np.isfinite(v) & (v > 0)]
            if v.size == 0:
                lo_arr.append(float("nan"))
                hi_arr.append(float("nan"))
            else:
                lo_arr.append(float(np.percentile(v, 25)))
                hi_arr.append(float(np.percentile(v, 75)))
        c = PALETTE[color]
        ls = "-" if encoder == "LISTA" else "--"
        ax.plot(horizons, iqms, marker="o", color=c, lw=1.6, ms=4, label=display, linestyle=ls)
        ax.fill_between(horizons, lo_arr, hi_arr, color=c, alpha=0.18, lw=0)
    ax.set_xlabel(r"Rollout horizon $H$ (observation steps)", fontsize=13)
    ax.set_ylabel("Raw MSE (IQM)", fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)
    ax.set_title(title, fontsize=11, pad=4)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(True, which="both", lw=0.4, alpha=0.4)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5, ncol=2)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
build_horizon_curve(fc_main, ROOT_LABELS_MAIN, axes[0], "Locked finalists vs. standard-sampling controls")
build_horizon_curve(fc_hardinit, ROOT_LABELS_HARDINIT, axes[1], "Boundary-emphasized matched controls")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_fixed17_horizon_curves.pdf")
fig.savefig(FIG_DIR / "fig_fixed17_horizon_curves.png")
plt.close(fig)

# Boundary-only single-panel horizon curve (all 5 rows trained under the same
# boundary-emphasized regime). Used as the main-text figure now that we drop
# the sampling column.
fc_boundary_only = pd.concat([fc_main, fc_hardinit], ignore_index=True)
fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.2))
build_horizon_curve(
    fc_boundary_only,
    ROOT_LABELS_BOUNDARY_ONLY,
    ax,
    "17-system multibasin forecasting\nperformance",
)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_fixed17_horizon_curves_boundary_only.pdf")
fig.savefig(FIG_DIR / "fig_fixed17_horizon_curves_boundary_only.png")
plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 2: Per-(system,seed) scatter of H1000 forecasting MSE on Fixed-17
# --------------------------------------------------------------------------------------

print("Building Figure: Fixed-17 per-(system,seed) distributions...")


def plot_per_seed_strip(
    fc_df: pd.DataFrame, label_map: dict, ax, horizon: int, title: str
):
    col = f"h{horizon}_best_periodic_per_dim_mean"
    keys = []
    distros = []
    colors = []
    iqm_pts = []
    n_seeds = []
    for root_label, (display, transition, encoder, regime, color) in label_map.items():
        sub = fc_df[fc_df["root_label"] == root_label]
        if sub.empty:
            continue
        vals = sub[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        keys.append(display)
        distros.append(vals)
        colors.append(PALETTE[color])
        iqm_pts.append(iqm(vals))
        n_seeds.append(len(vals))

    rng = np.random.default_rng(7)
    for idx, (vals, c, name) in enumerate(zip(distros, colors, keys)):
        if vals.size == 0:
            continue
        x_jitter = idx + (rng.random(vals.size) - 0.5) * 0.18
        ax.scatter(
            x_jitter,
            vals,
            color=c,
            alpha=0.45,
            s=11,
            edgecolor="none",
        )
        # IQM marker
        ax.hlines(iqm_pts[idx], idx - 0.25, idx + 0.25, color="black", lw=2.0)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(rf"H{horizon} raw MSE")
    ax.set_title(title, fontsize=8)
    ax.grid(axis="y", lw=0.3, alpha=0.4)


fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.6), sharey="row")
for j, h in enumerate(HORIZONS_FIXED17):
    plot_per_seed_strip(fc_main, ROOT_LABELS_MAIN, axes[0, j], h, rf"H{h} (locked finalists vs. controls)")
    plot_per_seed_strip(
        fc_hardinit, ROOT_LABELS_HARDINIT, axes[1, j], h, rf"H{h} (matched boundary-emphasized)"
    )
fig.suptitle("Fixed-17 multibasin forecasting per (system, seed)", fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_fixed17_seed_strips.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_fixed17_seed_strips.png", bbox_inches="tight")
plt.close(fig)

# Boundary-only single-row strip plot
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6), sharey=False)
for j, h in enumerate(HORIZONS_FIXED17):
    plot_per_seed_strip(
        fc_boundary_only,
        ROOT_LABELS_BOUNDARY_ONLY,
        axes[j],
        h,
        rf"H{h} (boundary-emphasized)",
    )
fig.suptitle(
    "Fixed-17 multibasin forecasting per (system, seed), boundary-emphasized rows",
    fontsize=10,
    y=1.02,
)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_fixed17_seed_strips_boundary_only.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_fixed17_seed_strips_boundary_only.png", bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 3: Support entropy distributions (deep slice, absolute:0.001)
# --------------------------------------------------------------------------------------

print("Building Figure: support entropy distributions...")


def plot_entropy_strip(
    itp_df: pd.DataFrame, label_map: dict, ax, metric: str, title: str,
    yscale: str = "linear",
    summary: str = "iqm",
):
    keys = []
    distros = []
    colors = []
    iqm_pts = []
    for root_label, (display, transition, encoder, regime, color) in label_map.items():
        sub = itp_df[(itp_df["root_label"] == root_label)]
        if sub.empty:
            continue
        vals = sub[metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if yscale == "log":
            vals = vals[vals > 0]
        keys.append(display)
        distros.append(vals)
        colors.append(PALETTE[color])
        if summary == "iqm":
            iqm_pts.append(iqm(vals))
        elif summary == "mean":
            iqm_pts.append(float(np.mean(vals)) if vals.size else float("nan"))
        else:
            raise ValueError(f"Unknown summary statistic: {summary}")

    rng = np.random.default_rng(11)
    for idx, (vals, c) in enumerate(zip(distros, colors)):
        if vals.size == 0:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        ax.vlines(idx, q1, q3, color="0.25", alpha=0.9, lw=1.5, zorder=2.5)
        ax.hlines([q1, q3], idx - 0.16, idx + 0.16, color="0.25", alpha=0.9, lw=1.5, zorder=2.6)
        x_jitter = idx + (rng.random(vals.size) - 0.5) * 0.18
        ax.scatter(x_jitter, vals, color=c, alpha=0.45, s=11, edgecolor="none", zorder=2)
        ax.hlines(iqm_pts[idx], idx - 0.25, idx + 0.25, color="black", lw=2.0, zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_title(title, fontsize=8)
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(axis="y", lw=0.3, alpha=0.4)


fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6))
deep_main = itp_main[DEEP_INTERP_MASK]
plot_entropy_strip(deep_main, ROOT_LABELS_MAIN, axes[0], "h_basin_given_support", r"$H(B\,|\,S)$ deep")
plot_entropy_strip(deep_main, ROOT_LABELS_MAIN, axes[1], "h_support_given_basin", r"$H(S\,|\,B)$ deep")
plot_entropy_strip(deep_main, ROOT_LABELS_MAIN, axes[2], "u_exact", r"$U_{\rm exact}$ deep")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_fixed17_entropy_strips.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_fixed17_entropy_strips.png", bbox_inches="tight")
plt.close(fig)

# Boundary-only entropy strip plot
deep_boundary = pd.concat([itp_main[DEEP_INTERP_MASK], itp_hardinit[DEEP_INTERP_HARDINIT_MASK]], ignore_index=True)
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6))
plot_entropy_strip(deep_boundary, ROOT_LABELS_BOUNDARY_ONLY, axes[0], "h_basin_given_support", r"$H(B\,|\,S)$ deep")
plot_entropy_strip(deep_boundary, ROOT_LABELS_BOUNDARY_ONLY, axes[1], "h_support_given_basin", r"$H(S\,|\,B)$ deep")
plot_entropy_strip(deep_boundary, ROOT_LABELS_BOUNDARY_ONLY, axes[2], "u_exact", r"$U_{\rm exact}$ deep")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_fixed17_entropy_strips_boundary_only.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_fixed17_entropy_strips_boundary_only.png", bbox_inches="tight")
plt.close(fig)

# Replacement for the original entropy-strip figure (Table 1-aligned panels).
# Use the same six-row packet as scripts/build_per_system_stats_and_forest.py,
# not the older five-row split loaded above, so this figure cannot regress if
# the all-figures builder is rerun.
table1_current_interp_csvs = [
    ROOT
    / "results"
    / "transition_rich_table2_5model_seed15_backfill_20260428"
    / "interpretability_pass0"
    / "interpretability_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428"
    / "interpretability_pass0"
    / "interpretability_rows.csv",
    ROOT
    / "results"
    / "transition_rich_lista_dense_p256_hardinit_table123_20260430"
    / "interpretability_pass0"
    / "interpretability_rows.csv",
]
root_labels_table1_current = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": (
        "LISTA",
        "dense",
        "LISTA",
        "boundary-emphasized",
        "dense_lista",
    ),
    "lista_blockdiag_signsplit_hardinit_basin_partition": (
        "LISTA-BD",
        "block-diagonal",
        "LISTA",
        "boundary-emphasized",
        "blockdiag_lista",
    ),
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": (
        "LISTA-SB",
        "soft-block",
        "LISTA",
        "boundary-emphasized",
        "softblock_lista",
    ),
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": (
        "Sparse MLP, BD",
        "block-diagonal",
        "MLP",
        "boundary-emphasized",
        "blockdiag_mlp",
    ),
    "mlp_sparse_hardinit_basin_partition_control": (
        "Sparse MLP",
        "dense",
        "MLP",
        "boundary-emphasized",
        "sparse_mlp",
    ),
    "mlp_zero_sparse_hardinit_basin_partition_control": (
        "Dense MLP",
        "dense",
        "MLP",
        "boundary-emphasized",
        "zero_mlp",
    ),
}
deep_table1_current = pd.concat(
    [pd.read_csv(path, low_memory=False) for path in table1_current_interp_csvs],
    ignore_index=True,
)
deep_table1_current = deep_table1_current[
    deep_table1_current["root_label"].isin(root_labels_table1_current.keys())
    & (deep_table1_current["support_scheme"] == "absolute:0.001")
    & (deep_table1_current["subset"] == "deep")
].copy()

# H(B|F_abs), |F_abs|, and the h=1 wrong-support/base ratio. The count panel
# uses a mean summary because family count is interpreted against basin counts;
# the wrong-support panel is log-scaled because the functional ablation spans
# orders of magnitude.
fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.6))
plot_entropy_strip(deep_table1_current, root_labels_table1_current, axes[0], "family_h_basin_given_family", r"$H(B\,|\,F_{\rm abs})$")
plot_entropy_strip(deep_table1_current, root_labels_table1_current, axes[1], "family_unique_count", r"$|F_{\rm abs}|$", summary="mean")
plot_entropy_strip(deep_table1_current, root_labels_table1_current, axes[2], "support_freeze_wrong_over_base_h1", r"wrong-support ratio $h{=}1$", yscale="log")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_fixed17_entropy_strips_alt2.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_fixed17_entropy_strips_alt2.png", bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------------------
# Table 2: Self-routed forecasting
# --------------------------------------------------------------------------------------

print("Loading self-routed forecasting...")
sr_frames = [pd.read_csv(path, low_memory=False) for path in SELF_ROUTED_CSVS if path.exists()]
if not sr_frames:
    raise FileNotFoundError("No self-routed forecasting CSVs found")
sr = pd.concat(sr_frames, ignore_index=True, sort=False)

# Restrict to topk:8 with all/deep depth and the routing modes of interest.
ROUTE_MODES = {
    "global_k": "Global $K$ baseline",
    "support_gated_k": "Support-gated $K$",
    "support_local_centered": "Centered support-local",
    "family_local_centered": "Centered family-local",
}
ROOTS_ROUTING = {
    "lista_dense_softblock_signsplit_p64_hardinit_basin_partition": "LISTA-SB",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
    "mlp_sparse_hardinit_basin_partition_control": "Sparse MLP",
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": "Sparse MLP, BD",
    "mlp_zero_sparse_hardinit_basin_partition_control": "Dense MLP",
}

ROOT_LABEL_NAME_FALLBACK = ROOTS_ROUTING

# Use the manuscript-facing routed/global ratio, restricted to topk:8.
sr = sr[sr["support_definition"] == "topk:8"].copy()


def build_routing_table(sr_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ratio_col = f"h{ROUTING_HORIZON}_over_global"
    for root_label, display in ROOTS_ROUTING.items():
        for depth_label, depth_key in [("all", "all"), ("deep", "q4")]:
            sub = sr_df[
                (sr_df["root_label"] == root_label)
                & (sr_df["depth_stratum"] == depth_key)
            ]
            if sub.empty:
                continue
            row = {"root": root_label, "display": display, "depth": depth_label}
            for mode_key, mode_display in ROUTE_MODES.items():
                if mode_key == "global_k":
                    continue
                seg = sub[sub["rollout_mode"] == mode_key]
                ratio = seg[ratio_col].to_numpy(dtype=float)
                ratio = ratio[np.isfinite(ratio)]
                wins = float(np.mean(ratio < 1.0)) if ratio.size > 0 else float("nan")
                row[f"{mode_key}_iqm"] = iqm(ratio)
                row[f"{mode_key}_std"] = std(ratio)
                row[f"{mode_key}_winrate"] = wins
                row[f"{mode_key}_n"] = int(ratio.size)
            rows.append(row)
    return pd.DataFrame(rows)


tbl2 = build_routing_table(sr)
(TBL_DIR / "table2_routing.csv").write_text(tbl2.to_csv(index=False))


# --------------------------------------------------------------------------------------
# Figure 4: Routing ratio bar chart on (all, deep)
# --------------------------------------------------------------------------------------

print("Building Figure: routing ratios...")


def plot_routing_bars(tbl: pd.DataFrame, ax, mode_keys: list[str], depth: str, title: str):
    bars = []
    labels = []
    colors = []
    win_rates = []
    for _, row in tbl[tbl["depth"] == depth].iterrows():
        for mode in mode_keys:
            ratio = row.get(f"{mode}_iqm", np.nan)
            if not np.isfinite(ratio):
                continue
            bars.append(ratio)
            labels.append(f"{row['display']} ({ROUTE_MODES[mode]})")
            display = row["display"]
            if "LISTA-BD" in display or "LISTA-BD" in display:
                color_key = "blockdiag_lista"
            elif "LISTA-SB" in display or "LISTA-SB" in display:
                color_key = "softblock_lista"
            elif "Sparse MLP, BD" in display:
                color_key = "blockdiag_mlp"
            elif "Sparse MLP" in display:
                color_key = "sparse_mlp"
            else:
                color_key = "zero_mlp"
            colors.append(PALETTE[color_key])
            win_rates.append(row.get(f"{mode}_winrate", np.nan))
    y = np.arange(len(bars))
    # Cap x-axis at 1.5 — values above are clipped with marker.
    bar_clipped = [min(b, 1.45) for b in bars]
    ax.barh(y, bar_clipped, color=colors, alpha=0.9)
    ax.axvline(1.0, color="black", ls="--", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Routed/global error ratio at H1000")
    ax.set_xlim(0, 1.55)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    for i, v in enumerate(bars):
        wr = win_rates[i]
        wr_text = "" if not np.isfinite(wr) else f"  win {wr*100:.0f}%"
        if v > 1.45:
            ax.text(1.46, i, f">{v:.0f}{wr_text}", va="center", fontsize=7)
        else:
            ax.text(min(v + 0.02, 1.46), i, f"{v:.2f}{wr_text}", va="center", fontsize=7)
    ax.invert_yaxis()


fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
plot_routing_bars(
    tbl2,
    axes[0],
    ["support_gated_k", "support_local_centered", "family_local_centered"],
    "all",
    "All states",
)
plot_routing_bars(
    tbl2,
    axes[1],
    ["support_gated_k", "support_local_centered", "family_local_centered"],
    "deep",
    "Deep states (far from boundaries)",
)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_routing_ratios.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_routing_ratios.png", bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 5: Periodic refresh — dominance/route-target/MSE-ratio vs period
# --------------------------------------------------------------------------------------

print("Building Figure: support refresh...")
pr = pd.read_csv(PERIODIC_REFRESH_CSV, low_memory=False)
print("Periodic-refresh columns sample:", list(pr.columns)[:30])


# --------------------------------------------------------------------------------------
# Table 3 + Figure 6: Dysts long-horizon
# --------------------------------------------------------------------------------------

print("Loading Dysts long-horizon...")
d_main = pd.read_csv(DYSTS_MAIN_CSV, low_memory=False)
d_blockdiag_mlp = pd.read_csv(DYSTS_BLOCKDIAG_MLP_CSV, low_memory=False)

DYSTS_HORIZONS = [5000, 10000, 20000, 30000]
DYSTS_ROOTS = list(ROOT_LABELS_DYSTS.keys())


def build_dysts_table(main_df: pd.DataFrame, mlp_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    combined = pd.concat([main_df, mlp_df], ignore_index=True)
    combined = combined[combined["root_label"].isin(DYSTS_ROOTS)]
    for root_label in DYSTS_ROOTS:
        if root_label not in combined["root_label"].unique():
            continue
        meta = ROOT_LABELS_DYSTS[root_label]
        display, transition, encoder, color = meta
        sub = combined[combined["root_label"] == root_label]
        row = {
            "root": root_label,
            "display": display,
            "transition": transition,
            "encoder": encoder,
            "color": color,
        }
        for h in DYSTS_HORIZONS:
            col = f"h{h}_best_periodic_mean"
            sys_iqm = sub.groupby("system_key")[col].apply(lambda v: iqm(v.to_numpy()))
            sys_iqm = sys_iqm.replace([np.inf, -np.inf], np.nan).dropna()
            iqm_pt = iqm(sys_iqm.to_numpy())
            sd_sys = std(sys_iqm.to_numpy())
            point, lo, hi = stratified_bootstrap_iqm(
                pd.DataFrame({"v": sys_iqm.to_numpy()}),
                "v",
                n_resamples=2000,
                rng_seed=hash(root_label + str(h)) & 0xFFFF,
            )
            row[f"H{h}_iqm"] = iqm_pt
            row[f"H{h}_iqm_ci_lo"] = lo
            row[f"H{h}_iqm_ci_hi"] = hi
            row[f"H{h}_std_systems"] = sd_sys
            row[f"H{h}_log10_std_systems"] = std(
                np.log10(np.clip(sys_iqm.to_numpy(), 1e-12, None))
            )
            row[f"H{h}_n_systems"] = int(sys_iqm.size)
        rows.append(row)
    return pd.DataFrame(rows)


tbl3 = build_dysts_table(d_main, d_blockdiag_mlp)
(TBL_DIR / "table3_dysts.csv").write_text(tbl3.to_csv(index=False))


def plot_dysts_horizon_curve(df_main, df_mlp, ax):
    horizons = DYSTS_HORIZONS
    combined = pd.concat([df_main, df_mlp], ignore_index=True)
    for root_label, meta in ROOT_LABELS_DYSTS.items():
        display, transition, encoder, color = meta
        sub = combined[combined["root_label"] == root_label]
        if sub.empty:
            continue
        ys, ylo, yhi = [], [], []
        for h in horizons:
            col = f"h{h}_best_periodic_mean"
            sys_iqm = sub.groupby("system_key")[col].apply(lambda v: iqm(v.to_numpy()))
            sys_iqm = sys_iqm.replace([np.inf, -np.inf], np.nan).dropna()
            ys.append(iqm(sys_iqm.to_numpy()))
            v = sys_iqm.to_numpy()
            v = v[np.isfinite(v) & (v > 0)]
            if v.size == 0:
                ylo.append(float("nan"))
                yhi.append(float("nan"))
            else:
                ylo.append(float(np.percentile(v, 25)))
                yhi.append(float(np.percentile(v, 75)))
        c = PALETTE[color]
        ls = "-" if encoder == "LISTA" else "--"
        ax.plot(horizons, ys, marker="o", color=c, lw=1.6, ms=4, label=display, linestyle=ls)
        ax.fill_between(horizons, ylo, yhi, color=c, alpha=0.15, lw=0)
    ax.set_xlabel(r"Rollout horizon $H$")
    ax.set_ylabel("MSE (IQM across 15 systems)")
    ax.set_yscale("log")
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)


def plot_dysts_winner_bars(df_main, df_mlp, ax):
    combined = pd.concat([df_main, df_mlp], ignore_index=True)
    horizons = DYSTS_HORIZONS
    counts = {root_label: [0] * len(horizons) for root_label in ROOT_LABELS_DYSTS}
    for j, h in enumerate(horizons):
        col = f"h{h}_best_periodic_mean"
        sys_iqm = (
            combined[combined["root_label"].isin(DYSTS_ROOTS)]
            .groupby(["root_label", "system_key"])[col]
            .apply(lambda v: iqm(v.to_numpy()))
            .reset_index()
        )
        for system_key, sgrp in sys_iqm.groupby("system_key"):
            sgrp = sgrp.dropna(subset=[col])
            if sgrp.empty:
                continue
            best = sgrp.loc[sgrp[col].idxmin(), "root_label"]
            if best in counts:
                counts[best][j] += 1

    bottom = np.zeros(len(horizons))
    for root_label, meta in ROOT_LABELS_DYSTS.items():
        display, transition, encoder, color = meta
        c = PALETTE[color]
        vals = counts[root_label]
        ax.bar(np.arange(len(horizons)), vals, bottom=bottom, color=c, label=display, edgecolor="white")
        bottom += np.asarray(vals, dtype=float)
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"H{h}" for h in horizons])
    ax.set_ylabel("# systems where root has lowest IQM")
    ax.legend(fontsize=6, loc="upper left", ncol=2, frameon=False)
    ax.set_title("Dysts winner counts per horizon", fontsize=9)


fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), gridspec_kw={"width_ratios": [1.4, 1.0]})
plot_dysts_horizon_curve(d_main, d_blockdiag_mlp, axes[0])
axes[0].set_title("Dysts long-horizon IQM (15 systems, IQR shaded)", fontsize=9)
axes[0].legend(frameon=False, fontsize=7, loc="upper left", ncol=1, bbox_to_anchor=(1.02, 1.0))
plot_dysts_winner_bars(d_main, d_blockdiag_mlp, axes[1])
axes[1].legend().remove()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_dysts_long_horizon.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_dysts_long_horizon.png", bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------------------
# Appendix: per-(system,seed) histograms for paper-facing benchmarks
# --------------------------------------------------------------------------------------

print("Building appendix per-seed distribution figures...")


def plot_log_hist(values_dict, title, x_label, ax):
    bins = np.linspace(-6, 4, 40)
    for label, (vals, color) in values_dict.items():
        v = np.asarray(vals, float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            continue
        ax.hist(np.log10(v), bins=bins, color=color, alpha=0.45, label=label, edgecolor="white")
    ax.set_xlabel(rf"$\log_{{10}}$({x_label})")
    ax.set_ylabel("# (system, seed) pairs")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6, frameon=False)


fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), constrained_layout=True)
for j, h in enumerate(HORIZONS_FIXED17):
    col = f"h{h}_best_periodic_mean"
    valdict = {}
    for root_label, (display, transition, encoder, regime, color) in ROOT_LABELS_MAIN.items():
        sub = fc_main[fc_main["root_label"] == root_label]
        valdict[display] = (sub[col].to_numpy(), PALETTE[color])
    plot_log_hist(valdict, f"H{h} per-coord MSE distribution (locked finalists)", f"H{h}", axes[0, j])

    valdict_h = {}
    for root_label, (display, transition, encoder, regime, color) in ROOT_LABELS_HARDINIT.items():
        sub = fc_hardinit[fc_hardinit["root_label"] == root_label]
        valdict_h[display] = (sub[col].to_numpy(), PALETTE[color])
    plot_log_hist(valdict_h, f"H{h} (matched boundary-emphasized)", f"H{h}", axes[1, j])

fig.suptitle(
    "Appendix: per-(system, seed) forecasting distributions on the 17-system multibasin benchmark",
    fontsize=10,
)
fig.savefig(FIG_DIR / "appfig_fixed17_perseed_histograms.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "appfig_fixed17_perseed_histograms.png", bbox_inches="tight")
plt.close(fig)


# Appendix: Dysts per-seed histograms
fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.8), constrained_layout=True)
combined_dysts = pd.concat([d_main, d_blockdiag_mlp], ignore_index=True)
for j, h in enumerate(DYSTS_HORIZONS):
    col = f"h{h}_best_periodic_mean"
    valdict = {}
    for root_label, meta in ROOT_LABELS_DYSTS.items():
        display, transition, encoder, color = meta
        sub = combined_dysts[combined_dysts["root_label"] == root_label]
        if sub.empty:
            continue
        valdict[display] = (sub[col].to_numpy(), PALETTE[color])
    plot_log_hist(valdict, f"H{h}", f"H{h}", axes[j])
fig.suptitle("Appendix: per-(system, seed) Dysts forecasting distributions", fontsize=10)
fig.savefig(FIG_DIR / "appfig_dysts_perseed_histograms.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "appfig_dysts_perseed_histograms.png", bbox_inches="tight")
plt.close(fig)


# Appendix: Fixed-17 entropy distributions across systems
fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2), constrained_layout=True)
for col_idx, metric in enumerate([
    ("h_basin_given_support", r"$H(B|S_{\rm abs})$"),
    ("h_support_given_basin", r"$H(S_{\rm abs}|B)$"),
    ("u_exact", r"$U_{\rm exact}$"),
]):
    metric_key, metric_name = metric
    deep_main = itp_main[DEEP_INTERP_MASK]
    valdict = {}
    for root_label, (display, transition, encoder, regime, color) in ROOT_LABELS_MAIN.items():
        sub = deep_main[deep_main["root_label"] == root_label]
        if sub.empty:
            continue
        v = sub[metric_key].to_numpy()
        v = v[np.isfinite(v)]
        valdict[display] = (v, PALETTE[color])
    bins = 30
    for label, (vals, c) in valdict.items():
        if vals.size == 0:
            continue
        axes[0, col_idx].hist(vals, bins=bins, color=c, alpha=0.45, label=label, edgecolor="white")
    axes[0, col_idx].set_xlabel(metric_name)
    axes[0, col_idx].set_ylabel("# (system, seed)")
    axes[0, col_idx].set_title(f"{metric_name} (locked finalists)", fontsize=8)
    axes[0, col_idx].legend(fontsize=6, frameon=False)

    deep_hardinit = itp_hardinit[DEEP_INTERP_HARDINIT_MASK]
    valdict_h = {}
    for root_label, (display, transition, encoder, regime, color) in ROOT_LABELS_HARDINIT.items():
        sub = deep_hardinit[deep_hardinit["root_label"] == root_label]
        if sub.empty:
            continue
        v = sub[metric_key].to_numpy()
        v = v[np.isfinite(v)]
        valdict_h[display] = (v, PALETTE[color])
    for label, (vals, c) in valdict_h.items():
        if vals.size == 0:
            continue
        axes[1, col_idx].hist(vals, bins=bins, color=c, alpha=0.45, label=label, edgecolor="white")
    axes[1, col_idx].set_xlabel(metric_name)
    axes[1, col_idx].set_ylabel("# (system, seed)")
    axes[1, col_idx].set_title(f"{metric_name} (matched boundary-emphasized)", fontsize=8)
    axes[1, col_idx].legend(fontsize=6, frameon=False)

fig.suptitle("Appendix: per-(system, seed) basin/support entropy distributions on deep slice", fontsize=10)
fig.savefig(FIG_DIR / "appfig_fixed17_entropy_histograms.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "appfig_fixed17_entropy_histograms.png", bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------------------------------------------
# Periodic refresh: route-target and refreshed/previous-support MSE
# --------------------------------------------------------------------------------------

print("Building periodic-refresh figure...")
candidate_cols = [c for c in pr.columns if c.lower() in {"period", "reencode_period", "re_encode_period"}]
print("candidate period cols:", candidate_cols)

period_col = None
for c in ["reencode_period", "re_encode_period", "period", "refresh_period", "support_refresh_period"]:
    if c in pr.columns:
        period_col = c
        break

if period_col is None:
    print("No period column found in periodic refresh CSV; skipping figure.")
    has_periodic_fig = False
else:
    has_periodic_fig = True


def safe_pick_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


route_target_col = safe_pick_col(pr, "route_target_fraction")
fallback_col = safe_pick_col(pr, "route_fallback_fraction")
mse_ratio_col = safe_pick_col(
    pr,
    "refreshed_gated_mse_vs_frozen_source_gated_ratio",
    "current_global_mse_vs_stale_source_ratio",
)

print("Found columns:")
print(" period:", period_col)
print(" route_target:", route_target_col)
print(" fallback:", fallback_col)
print(" mse_ratio:", mse_ratio_col)


def plot_refresh_panel(df: pd.DataFrame, ax_route, ax_mse):
    df = df.copy()
    df = df[df["status"].astype(str).str.lower() == "ok"]
    df = df[df["support_definition"] == "topk:8"]
    df = df[df["object_kind"] == "support"]
    df = df[df["transfer_success"] == True]  # noqa: E712
    df = df[df["start_mode"] == "post_start"]
    df = df[df["rollout_mode"] == "current_support_gated_periodic"]
    if df.empty:
        for ax, label in zip(
            (ax_route, ax_mse),
            ("route-target fraction", "refreshed/previous MSE ratio"),
        ):
            ax.set_visible(False)
        return

    roots = sorted(df["root_label"].unique())
    color_map = {
        "lista_dense_softblock_signsplit_p64_hardinit_basin_partition": (
            "LISTA-SB (dense)",
            PALETTE["softblock_lista"],
        ),
        "lista_blockdiag_signsplit_hardinit_basin_partition": (
            "LISTA-BD",
            PALETTE["blockdiag_lista"],
        ),
    }

    # We compare reencode_period in {1, 10}; the dataset includes refresh_period=0 = previous-support baseline
    df = df[df["rollout_mode"].astype(str) != "frozen_source_gated"]

    for root in roots:
        meta = color_map.get(root)
        if meta is None:
            continue
        display, color = meta
        sub = df[df["root_label"] == root]
        agg = sub.groupby("reencode_period").agg(
            route_target=(route_target_col, "mean"),
            route_target_std=(route_target_col, "std"),
            mse_ratio=(mse_ratio_col, "median"),
            mse_ratio_q25=(mse_ratio_col, lambda v: v.quantile(0.25)),
            mse_ratio_q75=(mse_ratio_col, lambda v: v.quantile(0.75)),
        )
        if agg.empty:
            continue
        x = agg.index.to_numpy()
        ax_route.plot(x, agg["route_target"], "-o", color=color, label=display, ms=5)
        ax_route.fill_between(
            x,
            agg["route_target"] - agg["route_target_std"],
            agg["route_target"] + agg["route_target_std"],
            color=color,
            alpha=0.15,
            lw=0,
        )
        ax_mse.plot(x, agg["mse_ratio"], "-o", color=color, label=display, ms=5)
        ax_mse.fill_between(
            x, agg["mse_ratio_q25"], agg["mse_ratio_q75"], color=color, alpha=0.15, lw=0
        )

    ax_route.set_xlabel("Re-encode period (steps)")
    ax_route.set_xscale("log")
    ax_route.grid(True, lw=0.4, alpha=0.4)
    ax_route.set_ylim(0, 1.05)
    ax_route.legend(fontsize=7, frameon=False)
    ax_route.set_ylabel("Route-target fraction after entry")
    ax_mse.set_xlabel("Re-encode period (steps)")
    ax_mse.set_xscale("log")
    ax_mse.set_yscale("log")
    ax_mse.set_ylabel("Refreshed/previous-support MSE ratio")
    ax_mse.grid(True, lw=0.4, alpha=0.4, which="both")
    ax_mse.legend(fontsize=7, frameon=False)


fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
if route_target_col is not None and mse_ratio_col is not None:
    plot_refresh_panel(pr, axes[0], axes[1])
fig.suptitle(
    "Support refresh after controlled basin entry (top-8 supports)", fontsize=10, y=1.02
)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_periodic_support_refresh.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_periodic_support_refresh.png", bbox_inches="tight")
plt.close(fig)


# Save a small CSV with the headline numbers for the table caption
def build_refresh_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["status"].astype(str).str.lower() == "ok"]
    df = df[df["support_definition"] == "topk:8"]
    df = df[df["object_kind"] == "support"]
    df = df[df["transfer_success"] == True]  # noqa: E712
    df = df[df["start_mode"] == "post_start"]
    df = df[df["rollout_mode"] == "current_support_gated_periodic"]
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (root, period), sub in df.groupby(["root_label", "reencode_period"]):
        rows.append(
            {
                "root": root,
                "reencode_period": period,
                "n_pairs": len(sub),
                "route_target_mean": sub[route_target_col].mean(),
                "route_target_std": sub[route_target_col].std(),
                "fallback_mean": sub[fallback_col].mean() if fallback_col else float("nan"),
                "mse_ratio_median": sub[mse_ratio_col].median(),
                "mse_ratio_q25": sub[mse_ratio_col].quantile(0.25),
                "mse_ratio_q75": sub[mse_ratio_col].quantile(0.75),
            }
        )
    return pd.DataFrame(rows)


tbl_refresh = build_refresh_table(pr)
(TBL_DIR / "table_refresh.csv").write_text(tbl_refresh.to_csv(index=False))


# --------------------------------------------------------------------------------------
# Persist a JSON manifest of all important figure files for the paper
# --------------------------------------------------------------------------------------

manifest = {
    "fixed17_horizon_curves": "fig_fixed17_horizon_curves.pdf",
    "fixed17_seed_strips": "fig_fixed17_seed_strips.pdf",
    "fixed17_entropy_strips": "fig_fixed17_entropy_strips.pdf",
    "routing_ratios": "fig_routing_ratios.pdf",
    "dysts_long_horizon": "fig_dysts_long_horizon.pdf",
    "appfig_fixed17_perseed_histograms": "appfig_fixed17_perseed_histograms.pdf",
    "appfig_dysts_perseed_histograms": "appfig_dysts_perseed_histograms.pdf",
    "appfig_fixed17_entropy_histograms": "appfig_fixed17_entropy_histograms.pdf",
    "tables": [str(p.name) for p in TBL_DIR.glob("*.csv")],
    "wilcoxon": [str(p.name) for p in TBL_DIR.glob("wilcoxon_*.json")],
}
(FIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

print("DONE. Outputs under:", FIG_DIR)
