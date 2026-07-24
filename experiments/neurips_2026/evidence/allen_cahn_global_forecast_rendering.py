"""Deterministic rendering for the forecast-optimized Allen--Cahn packet."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.neurips_2026.evidence.highdimensional_rendering import (
    COLORS,
    MARKERS,
    PDF_METADATA,
    PNG_METADATA,
)


HORIZONS = (80, 120, 160, 200)
SEEDS = tuple(range(64, 74))


def _curves(
    rows: pd.DataFrame,
    arm: str,
    metric: str,
    persistence_metric: str,
) -> np.ndarray:
    values = []
    for seed in SEEDS:
        selected = rows.loc[
            (rows["arm"] == arm) & (rows["seed"] == seed)
        ].sort_values("horizon")
        if selected["horizon"].tolist() != list(HORIZONS):
            raise ValueError("Forecast curve is incomplete")
        values.append(
            selected[metric].to_numpy(dtype=np.float64)
            / selected[persistence_metric].to_numpy(dtype=np.float64)
        )
    return np.asarray(values)


def _style_axis(axis: plt.Axes, panel: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3)
    axis.text(
        -0.15,
        1.07,
        panel,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def _plot_curve(
    axis: plt.Axes,
    rows: pd.DataFrame,
    metric: str,
    persistence_metric: str,
    ylabel: str,
    panel: str,
) -> None:
    times = np.asarray(HORIZONS, dtype=np.float64) * 0.1
    for arm in ("dense", "sparse"):
        curves = _curves(rows, arm, metric, persistence_metric)
        for curve in curves:
            axis.plot(times, curve, color=COLORS[arm], alpha=0.14, linewidth=0.7)
        axis.plot(
            times,
            curves.mean(axis=0),
            color=COLORS[arm],
            marker=MARKERS[arm],
            markersize=3.8,
            linewidth=1.8,
            linestyle="--" if arm == "dense" else "-",
            label=(
                "Dense tanh KAE"
                if arm == "dense"
                else "Soft-thresholded sparse KAE"
            ),
        )
    axis.axhline(
        1.0,
        color=COLORS["persistence"],
        linestyle=":",
        linewidth=1.2,
        label="Persistence",
    )
    axis.set_yscale("log")
    axis.set_xticks(times)
    axis.set_xlabel("Forecast horizon (physical time)")
    axis.set_ylabel(ylabel)
    axis.set_title("No re-encoding; fresh 256-trajectory validation")
    axis.grid(alpha=0.18, linewidth=0.5, which="both")
    _style_axis(axis, panel)


def _plot_declared_ratios(
    axis: plt.Axes,
    rows: pd.DataFrame,
    statistics: Mapping[str, object],
) -> None:
    generator = np.random.default_rng(20_260_720)
    cells = statistics["comparison"]["cells"]
    specifications = (
        (160, "field_mse", "T=16\nMean\nPASS"),
        (160, "final_field_mse", "T=16\nTerminal\nMISS"),
        (200, "field_mse", "T=20\nMean\nPASS"),
        (200, "final_field_mse", "T=20\nTerminal\nMISS"),
    )
    positions = (0.0, 1.0, 2.35, 3.35)
    for position, (horizon, metric, _label) in zip(positions, specifications):
        dense = (
            rows.loc[(rows["arm"] == "dense") & (rows["horizon"] == horizon)]
            .sort_values("seed")[metric]
            .to_numpy(dtype=np.float64)
        )
        sparse = (
            rows.loc[(rows["arm"] == "sparse") & (rows["horizon"] == horizon)]
            .sort_values("seed")[metric]
            .to_numpy(dtype=np.float64)
        )
        jitter = generator.uniform(-0.08, 0.08, size=len(SEEDS))
        axis.scatter(
            np.full(len(SEEDS), position) + jitter,
            sparse / dense,
            s=19,
            color=COLORS["sparse"],
            marker=MARKERS["sparse"],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        cell = cells[f"h{horizon}_{metric}"]
        estimate = 1.0 - float(cell["relative_reduction_of_means"])
        lower = 1.0 - float(cell["ci95_upper"])
        upper = 1.0 - float(cell["ci95_lower"])
        axis.errorbar(
            [position],
            [estimate],
            yerr=[[estimate - lower], [upper - estimate]],
            fmt="D",
            markersize=4.5,
            color="black",
            capsize=3,
            linewidth=1.2,
            zorder=4,
        )
    axis.axhline(1.0, color=COLORS["persistence"], linestyle=":", linewidth=1.2)
    axis.axvline(1.675, color="#BDBDBD", linewidth=0.6, zorder=0)
    axis.set_xticks(positions, [label for _horizon, _metric, label in specifications])
    axis.set_ylabel("Sparse / dense MSE")
    axis.set_title(
        "Four-cell gate failed: both terminal CIs cross 1\n"
        "Squares: seed ratios; diamonds: ratio of means (95% CI)"
    )
    axis.set_xlim(-0.45, 3.8)
    _style_axis(axis, "c")


def _plot_sparsity(axis: plt.Axes, rows: pd.DataFrame) -> None:
    generator = np.random.default_rng(20_260_721)
    for index, arm in enumerate(("dense", "sparse")):
        values = (
            rows.loc[(rows["arm"] == arm) & (rows["horizon"] == 200)]
            .sort_values("seed")["near_zero_fraction_at_1e_minus_3"]
            .to_numpy(dtype=np.float64)
        )
        jitter = generator.uniform(-0.08, 0.08, size=values.size)
        axis.scatter(
            np.full(values.size, index) + jitter,
            values,
            s=20,
            color=COLORS[arm],
            marker=MARKERS[arm],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        axis.hlines(
            values.mean(), index - 0.18, index + 0.18, color="black", linewidth=1.5
        )
    axis.set_xticks((0, 1), ("Dense tanh", "Soft-thresholded"))
    axis.set_ylabel(r"Fraction with $|z_j|\leq10^{-3}$")
    axis.set_title("Fresh-state representation audit")
    axis.set_yscale("log")
    axis.set_ylim(8e-4, 0.7)
    _style_axis(axis, "d")


def render_global_forecast(
    rows: pd.DataFrame,
    statistics: Mapping[str, object],
    output_pdf: Path,
    output_png: Path,
) -> None:
    style = {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            2, 2, figsize=(7.0, 4.8), constrained_layout=True
        )
        _plot_curve(
            axes[0, 0],
            rows,
            "field_mse",
            "persistence_field_mse",
            "Through-horizon mean MSE\n/ persistence mean MSE",
            "a",
        )
        _plot_curve(
            axes[0, 1],
            rows,
            "final_field_mse",
            "persistence_final_field_mse",
            "Terminal MSE / persistence",
            "b",
        )
        axes[0, 0].legend(frameon=False, loc="upper right")
        _plot_declared_ratios(axes[1, 0], rows, statistics)
        _plot_sparsity(axes[1, 1], rows)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_pdf, bbox_inches="tight", metadata=PDF_METADATA)
        figure.savefig(output_png, dpi=300, bbox_inches="tight", metadata=PNG_METADATA)
        plt.close(figure)
