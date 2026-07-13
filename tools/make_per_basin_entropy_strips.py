#!/usr/bin/env python3
"""Regenerate the Table 1-aligned entropy strip figure on the per-basin deep slice."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/mila/l/lia/skae")
FIG_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026"

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

EXCLUDED_SYSTEMS = {
    "multiwell_strong_transition",
    "claude_checkerboard_potential",
    "claude:checkerboard_potential",
}

ROOTS = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": ("LISTA", "dense_lista"),
    "lista_blockdiag_signsplit_hardinit_basin_partition": ("LISTA-BD", "blockdiag_lista"),
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": ("LISTA-SB", "softblock_lista"),
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": ("Sparse MLP, BD", "blockdiag_mlp"),
    "mlp_sparse_hardinit_basin_partition_control": ("Sparse MLP", "sparse_mlp"),
    "mlp_zero_sparse_hardinit_basin_partition_control": ("Dense MLP", "zero_mlp"),
}

PALETTE = {
    "dense_lista": "#785EF0",
    "blockdiag_lista": "#0072B2",
    "softblock_lista": "#56B4E9",
    "sparse_mlp": "#009E73",
    "blockdiag_mlp": "#44AA99",
    "zero_mlp": "#D55E00",
}

BASIN_COUNT_REFERENCE = 4.20
FAMILY_COUNT_CAP = 6.0


def iqm(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if values.size < 4:
        return float(np.mean(values))
    lo, hi = np.percentile(values, [25, 75])
    selected = values[(values >= lo) & (values <= hi)]
    if selected.size == 0:
        return float(np.median(values))
    return float(np.mean(selected))


def mean_finite(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def load_deep_rows() -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in INTERP_CSVS]
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["root_label"].isin(ROOTS)].copy()
    for column in ("system_name", "system_key", "train_env_name"):
        if column in rows:
            rows = rows[~rows[column].isin(EXCLUDED_SYSTEMS)].copy()
    return rows[
        (rows["support_scheme"] == "absolute:0.001")
        & (rows["subset"] == "deep")
        & (rows["family_jaccard_threshold"] == 0.5)
    ].copy()


def finite_metric_values(rows: pd.DataFrame, metric: str, *, positive: bool = False) -> np.ndarray:
    vals = pd.to_numeric(rows[metric], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if positive:
        vals = vals[vals > 0.0]
    return vals


def plot_strip(
    ax,
    rows: pd.DataFrame,
    *,
    metric: str,
    title: str,
    rng: np.random.Generator,
    yscale: str = "linear",
    summary: str = "iqm",
    cap: float | None = None,
) -> None:
    labels: list[str] = []
    for idx, (root_label, (label, color_key)) in enumerate(ROOTS.items()):
        sub = rows[rows["root_label"] == root_label]
        vals = finite_metric_values(sub, metric, positive=yscale == "log")
        labels.append(label)
        if vals.size == 0:
            continue

        q1, q3 = np.percentile(vals, [25, 75])
        yvals = vals if cap is None else np.minimum(vals, cap)
        overflow = np.zeros(vals.shape, dtype=bool) if cap is None else vals > cap
        x = idx + rng.uniform(-0.18, 0.18, size=vals.size)
        color = PALETTE[color_key]

        ax.vlines(idx, min(q1, cap or q1), min(q3, cap or q3), color="0.25", alpha=0.9, lw=1.5, zorder=2.5)
        ax.hlines(
            [min(q1, cap or q1), min(q3, cap or q3)],
            idx - 0.16,
            idx + 0.16,
            color="0.25",
            alpha=0.9,
            lw=1.5,
            zorder=2.6,
        )
        ax.scatter(x[~overflow], yvals[~overflow], s=8, color=color, alpha=0.32, linewidths=0, zorder=2)
        if np.any(overflow):
            ax.scatter(
                x[overflow],
                yvals[overflow],
                s=12,
                color=color,
                alpha=0.55,
                marker="^",
                linewidths=0,
                zorder=2.2,
                clip_on=False,
            )

        if summary == "mean":
            point = mean_finite(vals)
        elif summary == "iqm":
            point = iqm(vals)
        else:
            raise ValueError(f"Unknown summary: {summary}")
        point = min(point, cap) if cap is not None and np.isfinite(point) else point
        ax.hlines(point, idx - 0.25, idx + 0.25, color="black", lw=1.6, zorder=3)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(axis="y", lw=0.35, alpha=0.35)


def main() -> None:
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

    rows = load_deep_rows()
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), constrained_layout=True)

    plot_strip(
        axes[0],
        rows,
        metric="family_h_basin_given_family",
        title=r"$H(B\mid F_{\rm abs})$",
        rng=rng,
    )
    plot_strip(
        axes[1],
        rows,
        metric="family_unique_count",
        title=r"$|F_{\rm abs}|$",
        rng=rng,
        summary="mean",
        cap=FAMILY_COUNT_CAP,
    )
    axes[1].axhline(BASIN_COUNT_REFERENCE, color="0.45", lw=0.9, ls="--", zorder=1)
    axes[1].text(
        0.98,
        BASIN_COUNT_REFERENCE + 0.06,
        "basin mean 4.2",
        transform=axes[1].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
    )
    axes[1].set_ylim(0.75, FAMILY_COUNT_CAP + 0.35)
    axes[1].set_yticks(np.arange(1, int(FAMILY_COUNT_CAP) + 1))
    axes[1].text(
        0.98,
        0.97,
        "triangles: >6",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="0.35",
    )
    plot_strip(
        axes[2],
        rows,
        metric="support_freeze_wrong_over_base_h1",
        title=r"wrong-support ratio $h{=}1$",
        rng=rng,
        yscale="log",
    )

    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_fixed17_entropy_strips_alt2.{suffix}", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'fig_fixed17_entropy_strips_alt2.pdf'}")


if __name__ == "__main__":
    main()
