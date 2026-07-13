#!/usr/bin/env python3
"""Build appendix table and plot for Dysts IQM-over-IQM sensitivity."""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY_CSV = (
    ROOT
    / "docs"
    / "figures"
    / "neurips_paper_2026"
    / "_tables"
    / "dysts_dt30_iqm_summary.csv"
)
DEFAULT_FIG_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026"
DEFAULT_TABLE_DIR = DEFAULT_FIG_DIR / "_tables"
DEFAULT_HORIZONS = [100, 500, 1000, 1500, 2000, 3000, 4000, 5000]

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

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
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


def load_summary(path: Path, horizons: list[int]) -> pd.DataFrame:
    summary = pd.read_csv(path)
    required = {
        "root_label",
        "display",
        "horizon",
        "n_systems",
        "cross_system_mean",
        "system_q25",
        "system_q75",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")

    if "cross_system_iqm_over_iqm" in summary.columns:
        value_col = "cross_system_iqm_over_iqm"
    elif "cross_system_iqm_legacy" in summary.columns:
        value_col = "cross_system_iqm_legacy"
    else:
        raise RuntimeError(
            f"{path} needs cross_system_iqm_over_iqm or cross_system_iqm_legacy"
        )

    summary = summary[summary["root_label"].isin(ROOTS)].copy()
    summary = summary[summary["horizon"].isin(horizons)].copy()
    summary["iqm_over_system_seed_iqms"] = pd.to_numeric(
        summary[value_col],
        errors="coerce",
    )
    if summary.empty:
        raise RuntimeError(f"No retained rows found in {path}")
    return summary


def write_latex_table(summary: pd.DataFrame, table_dir: Path, horizons: list[int]) -> Path:
    best_by_horizon = {
        h: summary[summary["horizon"] == h]
        .sort_values("iqm_over_system_seed_iqms")
        .iloc[0]["root_label"]
        for h in horizons
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
            row = summary[
                (summary["root_label"] == root_label) & (summary["horizon"] == h)
            ].iloc[0]
            value = tex_number(float(row["iqm_over_system_seed_iqms"]))
            if root_label == best_by_horizon[h]:
                value = rf"\mathbf{{{value}}}"
            cells.append(rf"${value}$")
        lines.append(f"{meta['display']} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    out = table_dir / "table_dysts_dt30_iqm_over_iqm.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_summary_csv(summary: pd.DataFrame, table_dir: Path) -> Path:
    out = table_dir / "dysts_dt30_iqm_over_iqm_summary.csv"
    cols = [
        "root_label",
        "display",
        "horizon",
        "n_systems",
        "iqm_over_system_seed_iqms",
        "cross_system_mean",
        "system_median",
        "system_q25",
        "system_q75",
    ]
    keep = [col for col in cols if col in summary.columns]
    summary[keep].to_csv(out, index=False)
    return out


def plot_curves(summary: pd.DataFrame, fig_dir: Path, horizons: list[int]) -> Path:
    n_systems = int(pd.to_numeric(summary["n_systems"], errors="coerce").max())
    best_counts = (
        summary.sort_values("iqm_over_system_seed_iqms")
        .groupby("horizon", sort=False)
        .first()["root_label"]
        .value_counts()
    )
    highlight_root = str(best_counts.index[0]) if not best_counts.empty else ""
    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    for root_label, meta in ROOTS.items():
        sub = summary[summary["root_label"] == root_label].sort_values("horizon")
        x = sub["horizon"].to_numpy(dtype=float)
        y = sub["iqm_over_system_seed_iqms"].to_numpy(dtype=float)
        q25 = sub["system_q25"].to_numpy(dtype=float)
        q75 = sub["system_q75"].to_numpy(dtype=float)
        linewidth = 2.2 if root_label == highlight_root else 1.55
        zorder = 4 if root_label == highlight_root else 2
        ax.fill_between(x, q25, q75, color=meta["color"], alpha=0.10, linewidth=0)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=linewidth,
            markersize=4.2,
            label=meta["display"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            zorder=zorder,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Rollout horizon $H$")
    ax.set_ylabel("MSE (IQM over system seed-IQMs)")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons], rotation=35, ha="right")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    ax.set_title(f"{n_systems}-system Dysts robust aggregation sensitivity", pad=6)

    pdf = fig_dir / "fig_dysts_dt30_iqm_over_iqm_horizon.pdf"
    png = fig_dir / "fig_dysts_dt30_iqm_over_iqm_horizon.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return pdf


def update_manifest(fig_dir: Path, table_paths: list[Path], figure_path: Path) -> None:
    manifest_path = fig_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dysts_iqm_over_iqm"] = figure_path.name
    tables = list(manifest.get("tables", []))
    for path in table_paths:
        rel = path.name
        if rel not in tables:
            tables.append(rel)
    manifest["tables"] = tables
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    horizons = list(dict.fromkeys(int(h) for h in args.horizons))
    summary = load_summary(args.summary_csv, horizons)
    table_path = write_latex_table(summary, args.table_dir, horizons)
    csv_path = write_summary_csv(summary, args.table_dir)
    figure_path = plot_curves(summary, args.fig_dir, horizons)
    update_manifest(args.fig_dir, [table_path, csv_path], figure_path)
    print(f"Wrote {table_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
