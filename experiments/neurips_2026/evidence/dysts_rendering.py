"""Deterministic PDF rendering for the paper's Dysts evidence."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render_dysts_figure(
    robust: pd.DataFrame,
    methods: Mapping[str, tuple[str, str, str]],
    horizons: Sequence[int],
) -> bytes:
    style = {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    winners = (
        robust.sort_values("iqm_over_system_seed_iqms")
        .groupby("horizon", sort=False)
        .first()["root_label"]
        .value_counts()
    )
    highlight = str(winners.index[0])
    with plt.rc_context(style):
        fig, axis = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
        for root_label, (display, color, line_style) in methods.items():
            subset = robust[robust["root_label"] == root_label].sort_values(
                "horizon"
            )
            x = subset["horizon"].to_numpy(dtype=float)
            axis.fill_between(
                x,
                subset["system_q25"].to_numpy(dtype=float),
                subset["system_q75"].to_numpy(dtype=float),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            axis.plot(
                x,
                subset["iqm_over_system_seed_iqms"].to_numpy(dtype=float),
                marker="o",
                linewidth=2.2 if root_label == highlight else 1.55,
                markersize=4.2,
                label=display,
                color=color,
                linestyle=line_style,
                zorder=4 if root_label == highlight else 2,
            )
        axis.set_yscale("log")
        axis.set_xlabel(r"Rollout horizon $H$")
        axis.set_ylabel("MSE (IQM over system seed-IQMs)")
        axis.set_xticks(horizons)
        axis.set_xticklabels([str(h) for h in horizons], rotation=35, ha="right")
        axis.grid(True, which="both", linewidth=0.35, alpha=0.35)
        axis.legend(frameon=False, ncol=2)
        axis.set_title("10-system Dysts robust aggregation sensitivity", pad=6)
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="pdf",
            bbox_inches="tight",
            metadata={
                "Creator": "experiments.neurips_2026.evidence.dysts",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(fig)
    return buffer.getvalue()
