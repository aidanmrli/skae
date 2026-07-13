#!/usr/bin/env python3
"""Compare exact S_abs masks with merged F_abs families for basin identification."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/mila/l/lia/skae")
FIG_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026"
TABLE_DIR = FIG_DIR / "_tables"

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
    "lista_dense_signsplit_p256_hardinit_basin_partition": "LISTA",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition": "LISTA-SB",
    "mlp_sparse_blockdiag_hardinit_basin_partition_control": "Sparse MLP, BD",
    "mlp_sparse_hardinit_basin_partition_control": "Sparse MLP",
    "mlp_zero_sparse_hardinit_basin_partition_control": "Dense MLP",
}

OBJECTS = {
    "S_abs": {
        "label": r"exact $S_{\rm abs}$",
        "color": "#8A8A8A",
        "basin_entropy": "h_basin_given_support",
        "fragment_entropy": "h_support_given_basin",
        "count": "unique_support_count",
    },
    "F_abs": {
        "label": r"family $F_{\rm abs}$",
        "color": "#0072B2",
        "basin_entropy": "family_h_basin_given_family",
        "fragment_entropy": "family_h_family_given_basin",
        "count": "family_unique_count",
    },
}

BASIN_COUNT_REFERENCE = 4.20
SUPPORT_SCHEME = "absolute:0.001"
SUBSET = "deep"
FAMILY_JACCARD_THRESHOLD = 0.5


def iqm(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        return float(np.mean(arr))
    lo, hi = np.percentile(arr, [25, 75])
    keep = arr[(arr >= lo) & (arr <= hi)]
    if keep.size == 0:
        return float(np.median(arr))
    return float(np.mean(keep))


def mean_finite(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def load_rows() -> pd.DataFrame:
    frames = [pd.read_csv(path, low_memory=False) for path in INTERP_CSVS]
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["root_label"].isin(ROOTS)].copy()
    for column in ("system_name", "system_key", "train_env_name"):
        if column in rows:
            rows = rows[~rows[column].isin(EXCLUDED_SYSTEMS)].copy()
    rows = rows[
        (rows["support_scheme"] == SUPPORT_SCHEME)
        & (rows["subset"] == SUBSET)
        & (rows["family_jaccard_threshold"] == FAMILY_JACCARD_THRESHOLD)
    ].copy()
    for spec in OBJECTS.values():
        for column in (spec["basin_entropy"], spec["fragment_entropy"], spec["count"]):
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, float | str | int]] = []
    for root_label, label in ROOTS.items():
        root_rows = rows[rows["root_label"] == root_label]
        for object_key, spec in OBJECTS.items():
            metric_cols = {
                "basin_entropy": spec["basin_entropy"],
                "fragment_entropy": spec["fragment_entropy"],
                "count": spec["count"],
            }
            record: dict[str, float | str | int] = {
                "root_label": root_label,
                "model": label,
                "support_object": object_key,
                "num_rows": int(len(root_rows)),
                "num_systems": int(root_rows["system_name"].nunique()),
                "num_seeds": int(root_rows["seed"].nunique()),
            }
            for metric_name, column in metric_cols.items():
                if metric_name == "count":
                    per_system = root_rows.groupby("system_name")[column].mean()
                else:
                    per_system = root_rows.groupby("system_name")[column].apply(iqm)
                record[f"{metric_name}_mean_over_systems"] = mean_finite(per_system)
                record[f"{metric_name}_row_median"] = float(np.nanmedian(root_rows[column].to_numpy(dtype=float)))
            summary_rows.append(record)
    return pd.DataFrame(summary_rows)


def plot_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metric_name: str,
    title: str,
    ylabel: str,
    rng: np.random.Generator,
    yscale: str = "linear",
) -> None:
    model_labels = list(ROOTS.values())
    x_base = np.arange(len(model_labels), dtype=float)
    offsets = {"S_abs": -0.16, "F_abs": 0.16}
    for object_key, spec in OBJECTS.items():
        color = spec["color"]
        column = spec[metric_name]
        for idx, (root_label, _label) in enumerate(ROOTS.items()):
            sub = rows[rows["root_label"] == root_label]
            vals = sub[column].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if yscale == "log":
                vals = vals[vals > 0]
            if vals.size == 0:
                continue
            x_center = x_base[idx] + offsets[object_key]
            x_vals = x_center + rng.uniform(-0.055, 0.055, size=vals.size)
            ax.scatter(
                x_vals,
                vals,
                s=8,
                color=color,
                alpha=0.22 if object_key == "S_abs" else 0.28,
                linewidths=0,
                zorder=2,
            )
            summary_val = summary[
                (summary["root_label"] == root_label) & (summary["support_object"] == object_key)
            ][f"{metric_name}_mean_over_systems"].iloc[0]
            if np.isfinite(summary_val):
                ax.hlines(summary_val, x_center - 0.105, x_center + 0.105, color="black", lw=1.6, zorder=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_base)
    ax.set_xticklabels(model_labels, rotation=35, ha="right")
    ax.grid(axis="y", lw=0.35, alpha=0.35)
    if yscale == "log":
        ax.set_yscale("log")


def make_figure(rows: pd.DataFrame, summary: pd.DataFrame) -> None:
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
            "savefig.dpi": 250,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    rng = np.random.default_rng(17)
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.9), constrained_layout=True)
    plot_panel(
        axes[0],
        rows,
        summary,
        metric_name="basin_entropy",
        title=r"basin uncertainty",
        ylabel=r"$H(B\mid\mathrm{object})$",
        rng=rng,
    )
    plot_panel(
        axes[1],
        rows,
        summary,
        metric_name="fragment_entropy",
        title=r"within-basin fragmentation",
        ylabel=r"$H(\mathrm{object}\mid B)$",
        rng=rng,
    )
    plot_panel(
        axes[2],
        rows,
        summary,
        metric_name="count",
        title=r"number of objects",
        ylabel="unique objects",
        rng=rng,
        yscale="log",
    )
    axes[2].axhline(BASIN_COUNT_REFERENCE, color="0.35", lw=0.9, ls="--", zorder=1)
    axes[2].text(
        0.98,
        BASIN_COUNT_REFERENCE * 1.08,
        "basin mean 4.2",
        transform=axes[2].get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.30",
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=spec["color"], markersize=5, label=spec["label"])
        for spec in OBJECTS.values()
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=False, borderpad=0.0, handletextpad=0.4)
    fig.suptitle(r"Exact $S_{\rm abs}$ masks are pure but $F_{\rm abs}$ families are basin-scale", y=1.04)
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_fabs_vs_sabs_basin_identification.{suffix}", bbox_inches="tight")
    plt.close(fig)


def make_tradeoff_figure(summary: pd.DataFrame) -> None:
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
            "savefig.dpi": 250,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(3.7, 3.0), constrained_layout=True)
    ax.axhspan(0.5, 2.0, color="#E6F2F8", alpha=0.7, zorder=0)
    ax.axhline(1.0, color="0.35", lw=0.9, ls="--", zorder=1)
    ax.text(
        0.98,
        1.04,
        "basin-scale count",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.30",
    )

    for root_label, label in ROOTS.items():
        pair = summary[summary["root_label"] == root_label].set_index("support_object")
        if not {"S_abs", "F_abs"}.issubset(pair.index):
            continue
        xs = pair.loc[["S_abs", "F_abs"], "basin_entropy_mean_over_systems"].to_numpy(dtype=float)
        ys = (
            pair.loc[["S_abs", "F_abs"], "count_mean_over_systems"].to_numpy(dtype=float)
            / BASIN_COUNT_REFERENCE
        )
        ax.plot(xs, ys, color="0.70", lw=0.9, zorder=1.5)
        ax.scatter(
            xs[0],
            ys[0],
            s=30,
            color=OBJECTS["S_abs"]["color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.scatter(
            xs[1],
            ys[1],
            s=38,
            color=OBJECTS["F_abs"]["color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlim(-0.04, 1.42)
    ax.set_ylim(0.18, 260.0)
    ax.set_xlabel(r"basin uncertainty $H(B\mid\mathrm{object})$")
    ax.set_ylabel(r"unique objects / basin-count mean")
    ax.set_title(r"utility tradeoff for basin identification")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=spec["color"], markersize=5, label=spec["label"])
        for spec in OBJECTS.values()
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, borderpad=0.0, handletextpad=0.4)
    ax.text(0.04, 120.0, "exact masks:\nmany objects", fontsize=7, color="0.25", ha="left", va="center")
    ax.text(0.33, 0.72, "sparse rows:\nfamily count near basins", fontsize=7, color="0.20", ha="left", va="center")
    ax.text(0.98, 0.24, "Dense MLP\nfamily collapse", fontsize=7, color="0.20", ha="left", va="center")
    ax.grid(axis="y", lw=0.35, alpha=0.35)
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_fabs_vs_sabs_utility_tradeoff.{suffix}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if rows.empty:
        raise RuntimeError("No rows matched the F_abs/S_abs figure filters")
    summary = summarize(rows)
    summary_path = TABLE_DIR / "fabs_vs_sabs_basin_identification_summary.csv"
    metadata_path = TABLE_DIR / "fabs_vs_sabs_basin_identification_metadata.json"
    summary.to_csv(summary_path, index=False)
    metadata = {
        "source_csvs": [str(path) for path in INTERP_CSVS],
        "excluded_systems": sorted(EXCLUDED_SYSTEMS),
        "support_scheme": SUPPORT_SCHEME,
        "subset": SUBSET,
        "family_jaccard_threshold": FAMILY_JACCARD_THRESHOLD,
        "basin_count_reference": BASIN_COUNT_REFERENCE,
        "output_pdf": str(FIG_DIR / "fig_fabs_vs_sabs_basin_identification.pdf"),
        "output_png": str(FIG_DIR / "fig_fabs_vs_sabs_basin_identification.png"),
        "tradeoff_output_pdf": str(FIG_DIR / "fig_fabs_vs_sabs_utility_tradeoff.pdf"),
        "tradeoff_output_png": str(FIG_DIR / "fig_fabs_vs_sabs_utility_tradeoff.png"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    make_figure(rows, summary)
    make_tradeoff_figure(summary)
    print(f"Wrote {FIG_DIR / 'fig_fabs_vs_sabs_basin_identification.pdf'}")
    print(f"Wrote {FIG_DIR / 'fig_fabs_vs_sabs_utility_tradeoff.pdf'}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
