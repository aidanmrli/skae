"""Rendering helpers for the high-dimensional NeurIPS evidence figure."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "dense": "#0072B2",
    "dense_mlp_tanh_exact": "#0072B2",
    "sparse": "#D55E00",
    "lista_sign_split": "#D55E00",
    "direct": "#009E73",
    "persistence": "#666666",
    "dmd": "#CC79A7",
    "truncated_svd_dmd": "#E69F00",
}
MARKERS = {"dense": "o", "sparse": "s", "direct": "^"}
PDF_METADATA = {
    "Creator": "SKAE NeurIPS evidence builder",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {"Software": "SKAE NeurIPS evidence builder"}


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int,
    resamples: int = 100_000,
) -> Tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(array.mean()), float(low), float(high)


def plot_lorenz96(ax: plt.Axes, seed_rows: pd.DataFrame, summary: dict) -> None:
    names = {
        "dense_mlp_tanh_exact": "Dense tanh KAE",
        "lista_sign_split": "Sparse signed KAE",
    }
    for model_index, (model, label) in enumerate(names.items()):
        selected = seed_rows.loc[seed_rows["model"] == model]
        for _seed, group in selected.groupby("seed"):
            ax.plot(
                group["physical_time"],
                group["nrmse"],
                color=COLORS[model],
                alpha=0.14,
                linewidth=0.7,
            )
        times, means, lows, highs = [], [], [], []
        for horizon_index, (time, group) in enumerate(
            selected.groupby("physical_time")
        ):
            mean, low, high = bootstrap_mean_ci(
                group["nrmse"], seed=11_000 + 100 * model_index + horizon_index
            )
            times.append(time)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        ax.fill_between(times, lows, highs, color=COLORS[model], alpha=0.18)
        ax.plot(
            times,
            means,
            color=COLORS[model],
            marker="o" if model_index == 0 else "s",
            linewidth=2.0,
            markersize=4,
            label=label,
        )
    baselines = pd.DataFrame(summary["baselines"])
    styles = {
        "persistence": ("Persistence", "--"),
        "dmd": ("DMD", ":"),
        "truncated_svd_dmd": ("Truncated DMD", "-."),
    }
    for model, (label, linestyle) in styles.items():
        selected = baselines.loc[baselines["model"] == model].sort_values(
            "physical_time"
        )
        ax.plot(
            selected["physical_time"],
            selected["mean_nrmse"],
            color=COLORS[model],
            linestyle=linestyle,
            linewidth=1.25,
            label=label,
        )
    ax.set_title("a  Chaotic forecasting: Lorenz–96 ($d_x=128$)", loc="left")
    ax.set_xlabel("Forecast horizon (physical time)")
    ax.set_ylabel("Normalized RMSE")
    ax.set_xlim(0.0, 5.05)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, ncol=2)


def plot_allen_cahn_forecast(ax: plt.Axes, rows: pd.DataFrame) -> None:
    labels = {
        "direct": "Direct convolutional control",
        "dense": "Dense tanh KAE",
        "sparse": "Temporal sparse KAE",
    }
    for model_index, (model, label) in enumerate(labels.items()):
        selected = rows.loc[rows["model"] == model]
        for _seed, group in selected.groupby("seed"):
            ax.plot(
                group["physical_time"],
                group["final_relative_rmse"],
                color=COLORS[model],
                alpha=0.14,
                linewidth=0.7,
            )
        times, means, lows, highs = [], [], [], []
        for horizon_index, (time, group) in enumerate(
            selected.groupby("physical_time")
        ):
            mean, low, high = bootstrap_mean_ci(
                group["final_relative_rmse"],
                seed=22_000 + 100 * model_index + horizon_index,
            )
            times.append(time)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        ax.fill_between(times, lows, highs, color=COLORS[model], alpha=0.18)
        ax.plot(
            times,
            means,
            color=COLORS[model],
            marker=MARKERS[model],
            linewidth=2.0,
            markersize=4,
            label=label,
        )
    ax.axhline(
        1.0,
        color=COLORS["persistence"],
        linestyle="--",
        linewidth=1.3,
        label="Persistence",
    )
    ax.set_yscale("log")
    ax.set_title("b  Multibasin PDE forecasting: Allen–Cahn ($d_x=512$)", loc="left")
    ax.set_xlabel("Forecast horizon (physical time)")
    ax.set_ylabel("Final-time RMSE / persistence RMSE")
    maximum_time = float(rows["physical_time"].max())
    ax.set_xlim(0.0, maximum_time + 0.01 * max(maximum_time, 1.0))
    ax.grid(alpha=0.2, linewidth=0.6, which="both")
    ax.legend(frameon=False, fontsize=8)


def plot_support_alignment(ax: plt.Axes, rows: pd.DataFrame) -> None:
    selected = rows.loc[
        (rows["scope"] == "final") & (rows["slice"] == "all_test")
    ].copy()
    metrics = (
        ("trajectory_transfer_coverage", "Transfer\ncoverage"),
        ("normalized_h_basin_given_family", "Basin\ninformation"),
        ("normalized_h_family_given_basin", "Support\nuniqueness"),
        ("nmi", "NMI"),
        ("ari", "ARI"),
    )
    offsets = {"dense": -0.13, "sparse": 0.13}
    for metric_index, (metric, _label) in enumerate(metrics):
        dense_values = (
            selected.loc[selected["model"] == "dense"]
            .sort_values("seed")[metric]
            .to_numpy(dtype=np.float64)
        )
        sparse_values = (
            selected.loc[selected["model"] == "sparse"]
            .sort_values("seed")[metric]
            .to_numpy(dtype=np.float64)
        )
        if metric in {
            "normalized_h_basin_given_family",
            "normalized_h_family_given_basin",
        }:
            dense_values = 1.0 - dense_values
            sparse_values = 1.0 - sparse_values
        show_dense = metric != "normalized_h_family_given_basin"
        if show_dense:
            for dense_value, sparse_value in zip(dense_values, sparse_values):
                ax.plot(
                    [metric_index + offsets["dense"], metric_index + offsets["sparse"]],
                    [dense_value, sparse_value],
                    color="#999999",
                    alpha=0.28,
                    linewidth=0.65,
                    zorder=1,
                )
        values_by_model = [("sparse", sparse_values)]
        if show_dense:
            values_by_model.insert(0, ("dense", dense_values))
        for model_index, (model, values) in enumerate(values_by_model):
            x = metric_index + offsets[model]
            ax.scatter(
                np.full(values.size, x),
                values,
                color=COLORS[model],
                marker=MARKERS[model],
                s=15,
                alpha=0.35,
                zorder=2,
            )
            mean, low, high = bootstrap_mean_ci(
                values, seed=33_000 + 100 * metric_index + model_index
            )
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - low], [high - mean]],
                color=COLORS[model],
                marker=MARKERS[model],
                markersize=6,
                linewidth=1.8,
                capsize=3,
                zorder=3,
                label=(
                    "Dense tanh KAE" if model == "dense" else "Temporal sparse KAE"
                )
                if metric_index == 0
                else None,
            )
        if not show_dense:
            ax.text(
                metric_index + offsets["dense"],
                0.035,
                "n/a\n(1 family)",
                ha="center",
                va="bottom",
                color=COLORS["dense"],
                fontsize=6.5,
            )
    ax.set_xticks(range(len(metrics)), [label for _metric, label in metrics])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylim(-0.08, 1.04)
    ax.axhline(0.0, color="#777777", linewidth=0.7)
    ax.plot([1.72, 2.28], [0.65, 0.65], color="#444444", linestyle="--", linewidth=0.8)
    ax.text(1.68, 0.65, "0.65 gate", ha="right", va="center", fontsize=6.5)
    ax.set_ylabel("Alignment score (higher is better)")
    ax.set_title("c  Transferred PDE basin–support alignment", loc="left")
    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, loc="lower left")


def render_highdimensional(
    *,
    lorenz_summary: dict,
    lorenz_rows: pd.DataFrame,
    allen_forecast: pd.DataFrame,
    allen_support: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.35), constrained_layout=True)
    plot_lorenz96(axes[0], lorenz_rows, lorenz_summary)
    plot_allen_cahn_forecast(axes[1], allen_forecast)
    plot_support_alignment(axes[2], allen_support)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", metadata=PDF_METADATA)
    figure.savefig(output_png, dpi=300, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(figure)
