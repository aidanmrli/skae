"""Publication rendering for the Allen--Cahn new-IC replication packet."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"
LIGHT_GRAY = "#D9D9D9"
PDF_METADATA = {
    "Creator": "SKAE Allen-Cahn new-IC evidence builder",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {"Software": "SKAE Allen-Cahn new-IC evidence builder"}


def _style_axis(axis: plt.Axes, panel: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55, alpha=0.65, zorder=0)
    axis.text(
        -0.04,
        1.06,
        panel,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _prospective_effects(
    axis: plt.Axes,
    summary: Mapping[str, object],
    dataset_rows: Sequence[Mapping[str, object]],
) -> None:
    primary = summary["primary"]
    reductions = np.asarray(
        [float(row["relative_reduction"]) for row in dataset_rows],
        dtype=np.float64,
    )
    positions = np.arange(4, dtype=np.float64)
    axis.scatter(
        positions[:3],
        100.0 * reductions,
        color=ORANGE,
        marker="s",
        s=34,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Dataset reductions of arm means (descriptive)",
    )
    estimate = 100.0 * float(primary["relative_reduction_of_arm_means"])
    lower = 100.0 * float(primary["ci95_lower"])
    upper = 100.0 * float(primary["ci95_upper"])
    axis.errorbar(
        [positions[3]],
        [estimate],
        yerr=[[estimate - lower], [upper - estimate]],
        fmt="D",
        color="black",
        markerfacecolor=ORANGE,
        markeredgecolor="black",
        markersize=5.5,
        capsize=3,
        linewidth=1.3,
        zorder=4,
        label="Paired-seed aggregate (95% CI)",
    )
    for position, value in zip(positions[:3], 100.0 * reductions):
        axis.text(position, value + 0.35, f"{value:.2f}%", ha="center", fontsize=6.4)
    axis.text(positions[3], estimate + 0.35, f"{estimate:.2f}%", ha="center", fontsize=6.4)
    axis.axhline(0.0, color=GRAY, linewidth=0.9)
    axis.set_xticks(
        positions,
        ("New IC A", "New IC B", "New IC C", "3-set aggregate"),
    )
    axis.set_ylabel("Sparse reduction in mean MSE (%)")
    axis.set_ylim(0.0, max(10.0, upper + 1.0))
    axis.set_title("Prospective H200 through-horizon mean ($T=20$)", loc="left")
    axis.legend(frameon=False, fontsize=6.3, loc="upper left")
    axis.text(
        0.02,
        0.025,
        "A: seed 1775404171   B: 74732421   C: 293789188",
        transform=axis.transAxes,
        fontsize=5.7,
        color=GRAY,
    )
    _style_axis(axis, "a")


def _paired_endpoints(
    axis: plt.Axes,
    summary: Mapping[str, object],
    seed_rows: Sequence[Mapping[str, object]],
) -> None:
    rng = np.random.default_rng(20_260_722)
    specs = (
        (
            "cumulative",
            "h200_cumulative_sparse_over_dense",
            int(summary["primary"]["paired_model_seed_wins"]),
        ),
        (
            "terminal",
            "h200_terminal_sparse_over_dense",
            int(summary["secondary"]["h200_terminal"]["paired_model_seed_wins"]),
        ),
    )
    for position, (endpoint, column, wins) in enumerate(specs):
        values = np.asarray([float(row[column]) for row in seed_rows])
        jitter = rng.uniform(-0.075, 0.075, size=values.size)
        axis.scatter(
            np.full(values.size, position) + jitter,
            values,
            color=ORANGE if endpoint == "cumulative" else GRAY,
            marker="o" if endpoint == "cumulative" else "^",
            s=25,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        endpoint_summary = (
            summary["primary"]
            if endpoint == "cumulative"
            else summary["secondary"]["h200_terminal"]
        )
        ratio = 1.0 - float(endpoint_summary["relative_reduction_of_arm_means"])
        if endpoint == "cumulative":
            lower = 1.0 - float(endpoint_summary["ci95_upper"])
            upper = 1.0 - float(endpoint_summary["ci95_lower"])
            yerr = [[ratio - lower], [upper - ratio]]
        else:
            yerr = None
        axis.errorbar(
            [position],
            [ratio],
            yerr=yerr,
            fmt="D",
            color="black",
            markerfacecolor=ORANGE if endpoint == "cumulative" else "white",
            markersize=5.3,
            capsize=3,
            linewidth=1.3,
            zorder=4,
        )
        axis.text(position, 1.012, f"{wins}/10 wins", ha="center", va="bottom", fontsize=6.5)
    axis.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.1)
    axis.axvline(0.5, color=LIGHT_GRAY, linewidth=0.7)
    axis.set_xticks((0, 1), ("Mean over\nsteps 1--200\nprimary", "Terminal\ndescriptive"))
    axis.set_ylabel("Sparse / dense H200 field MSE")
    axis.set_ylim(0.82, 1.08)
    axis.set_xlim(-0.18, 1.12)
    axis.set_title("Paired model seeds; aggregate diamonds", loc="left")
    axis.text(
        0.02,
        0.03,
        "Below 1 favors sparse; CI shown only for primary",
        transform=axis.transAxes,
        fontsize=6.2,
        color=GRAY,
    )
    _style_axis(axis, "b")


def _development_comparison(axis: plt.Axes, summary: Mapping[str, object]) -> None:
    development = summary["development_context"]
    prospective = (
        float(summary["primary"]["relative_reduction_of_arm_means"]),
        float(summary["secondary"]["h200_terminal"]["relative_reduction_of_arm_means"]),
    )
    opened = (
        float(development["h200_cumulative_reduction_card_rounded"]),
        float(development["h200_terminal_reduction"]),
    )
    positions = np.asarray((0.0, 1.0))
    axis.scatter(
        positions - 0.04,
        100.0 * np.asarray(opened),
        color=BLUE,
        marker="o",
        facecolor="white",
        edgecolor=BLUE,
        linewidth=1.1,
        s=27,
        label="Opened development set",
    )
    axis.scatter(
        positions + 0.04,
        100.0 * np.asarray(prospective),
        color=ORANGE,
        marker="s",
        linewidth=1.1,
        s=27,
        label="3 new IC sets (aggregate)",
    )
    axis.axhline(0.0, color=GRAY, linewidth=0.9)
    axis.set_xticks(positions, ("Mean over\nsteps 1--200\nprimary", "Terminal\ndescriptive"))
    axis.set_ylabel("Sparse reduction in H200 MSE (%)")
    axis.set_ylim(0.0, 8.0)
    axis.set_title("Development vs prospective ($T=20$; no pooling)", loc="left")
    axis.legend(frameon=False, fontsize=6.4, loc="upper right")
    axis.text(
        0.02,
        0.04,
        "Development through-horizon mean is card-stated (5.48%, rounded)",
        transform=axis.transAxes,
        fontsize=6.1,
        color=GRAY,
    )
    _style_axis(axis, "c")


def render_replication_figure(
    summary: Mapping[str, object],
    seed_rows: Sequence[Mapping[str, object]],
    dataset_rows: Sequence[Mapping[str, object]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    """Render the rebuttal-ready H200 replication display."""

    style = {
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "legend.fontsize": 6.4,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure = plt.figure(figsize=(7.15, 5.25), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.92))
        _prospective_effects(figure.add_subplot(grid[0, 0]), summary, dataset_rows)
        _paired_endpoints(figure.add_subplot(grid[0, 1]), summary, seed_rows)
        _development_comparison(figure.add_subplot(grid[1, :]), summary)
        figure.suptitle(
            "Allen–Cahn direct rollout: same checkpoints, no retraining or reselection",
            fontsize=9.2,
            fontweight="bold",
        )
        figure.text(
            0.5,
            -0.012,
            "Three new 256-trajectory initial-condition datasets; ten paired model seeds; one encode and 200 repeated K steps (no re-encoding).",
            ha="center",
            fontsize=6.4,
            color=GRAY,
        )
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_pdf, bbox_inches="tight", metadata=PDF_METADATA)
        figure.savefig(output_png, dpi=300, bbox_inches="tight", metadata=PNG_METADATA)
        plt.close(figure)
