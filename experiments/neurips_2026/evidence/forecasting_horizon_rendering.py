"""Deterministic rendering for the submitted forecasting-horizon figure."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MEAN_COLUMN = "mean_over_system_seed_iqms"
LOW_COLUMN = "log_relative_seed_bootstrap_ci95_low"
HIGH_COLUMN = "log_relative_seed_bootstrap_ci95_high"

PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

PDF_METADATA = {
    "Creator": "experiments.neurips_2026.evidence.forecasting_horizons",
    "CreationDate": None,
    "ModDate": None,
}


def _draw_panel(
    axis,
    summary: pd.DataFrame,
    method_styles: Mapping[str, tuple[str, str, str]],
    horizons: Sequence[int],
    *,
    title: str,
) -> None:
    """Draw one benchmark without trimming any system-level observations."""

    for root_label, (display, color, line_style) in method_styles.items():
        subset = summary[summary["root_label"] == root_label].sort_values(
            "horizon"
        )
        if subset["horizon"].tolist() != list(horizons):
            raise ValueError(f"Incomplete horizon series for {root_label}")
        x = subset["horizon"].to_numpy(dtype=float)
        axis.fill_between(
            x,
            subset[LOW_COLUMN].to_numpy(dtype=float),
            subset[HIGH_COLUMN].to_numpy(dtype=float),
            color=color,
            alpha=0.17,
            linewidth=0,
        )
        axis.plot(
            x,
            subset[MEAN_COLUMN].to_numpy(dtype=float),
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            linestyle=line_style,
            color=color,
            label=display,
        )
    axis.set_yscale("log")
    axis.set_xlabel(r"Rollout horizon $H$ (observation steps)")
    axis.set_ylabel("MSE (mean over system seed-IQMs)")
    axis.set_xticks(horizons)
    rotation = 30 if len(horizons) > 4 else 0
    alignment = "right" if rotation else "center"
    axis.set_xticklabels(
        [str(horizon) for horizon in horizons],
        rotation=rotation,
        ha=alignment,
    )
    axis.grid(True, which="both", linewidth=0.4, alpha=0.35)
    axis.legend(frameon=False, loc="lower right", ncol=2)
    axis.set_title(title, pad=5)


def _save_pdf(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(figure)
    return buffer.getvalue()


def render_panel(
    summary: pd.DataFrame,
    method_styles: Mapping[str, tuple[str, str, str]],
    horizons: Sequence[int],
    *,
    title: str,
) -> bytes:
    """Render one legacy panel used by the archived submission source."""

    with plt.rc_context(PLOT_STYLE):
        figure, axis = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
        _draw_panel(axis, summary, method_styles, horizons, title=title)
        return _save_pdf(figure)


def render_composite(
    controlled_summary: pd.DataFrame,
    dysts_summary: pd.DataFrame,
    controlled_styles: Mapping[str, tuple[str, str, str]],
    dysts_styles: Mapping[str, tuple[str, str, str]],
    controlled_horizons: Sequence[int],
    dysts_horizons: Sequence[int],
) -> bytes:
    """Render the complete two-panel forecasting-horizon figure."""

    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(9.2, 3.35),
            constrained_layout=True,
        )
        _draw_panel(
            axes[0],
            controlled_summary,
            controlled_styles,
            controlled_horizons,
            title="15-system multibasin forecasting performance",
        )
        _draw_panel(
            axes[1],
            dysts_summary,
            dysts_styles,
            dysts_horizons,
            title=r"10-system Dysts $dt{\times}30$ forecasting performance",
        )
        axes[0].set_title("(a) 15-system multibasin benchmark", pad=5)
        axes[1].set_title(r"(b) 10-system Dysts $dt{\times}30$ benchmark", pad=5)
        return _save_pdf(figure)
