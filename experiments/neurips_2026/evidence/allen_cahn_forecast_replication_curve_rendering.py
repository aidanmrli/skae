"""Full-horizon rendering for the authenticated Allen--Cahn replication."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"
LIGHT_GRAY = "#D9D9D9"
PDF_METADATA = {
    "Creator": "SKAE Allen-Cahn new-IC full-horizon evidence builder",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {
    "Software": "SKAE Allen-Cahn new-IC full-horizon evidence builder"
}


def _style_axis(axis: plt.Axes, panel: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3)
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55, alpha=0.65, zorder=0)
    axis.text(
        -0.08,
        1.07,
        panel,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _arrays(rows: Sequence[Mapping[str, object]]) -> dict[str, np.ndarray]:
    fields = (
        "physical_time",
        "dense_mean_through_horizon_mean_field_mse",
        "dense_pointwise_ci95_lower",
        "dense_pointwise_ci95_upper",
        "sparse_mean_through_horizon_mean_field_mse",
        "sparse_pointwise_ci95_lower",
        "sparse_pointwise_ci95_upper",
        "relative_reduction_of_arm_means",
        "relative_reduction_pointwise_ci95_lower",
        "relative_reduction_pointwise_ci95_upper",
    )
    return {
        field: np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        for field in fields
    }


def _mse_panel(axis: plt.Axes, values: Mapping[str, np.ndarray]) -> None:
    time = values["physical_time"]
    dense = values["dense_mean_through_horizon_mean_field_mse"]
    sparse = values["sparse_mean_through_horizon_mean_field_mse"]
    axis.fill_between(
        time,
        values["dense_pointwise_ci95_lower"],
        values["dense_pointwise_ci95_upper"],
        color=BLUE,
        alpha=0.14,
        linewidth=0,
        zorder=1,
    )
    axis.fill_between(
        time,
        values["sparse_pointwise_ci95_lower"],
        values["sparse_pointwise_ci95_upper"],
        color=ORANGE,
        alpha=0.16,
        linewidth=0,
        zorder=1,
    )
    axis.plot(time, dense, color=BLUE, linestyle="--", linewidth=1.45, label="Dense tanh")
    axis.plot(time, sparse, color=ORANGE, linewidth=1.55, label="Joint sparse recipe")
    axis.scatter([time[-1]], [dense[-1]], color=BLUE, marker="o", s=20, zorder=4)
    axis.scatter([time[-1]], [sparse[-1]], color=ORANGE, marker="s", s=20, zorder=4)
    band = Patch(
        facecolor=GRAY,
        alpha=0.15,
        edgecolor="none",
        label="95% pointwise bootstrap intervals",
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles + [band], labels + [band.get_label()], frameon=False, loc="upper right")
    upper = max(
        float(values["dense_pointwise_ci95_upper"].max()),
        float(values["sparse_pointwise_ci95_upper"].max()),
    )
    axis.set_xlim(0.0, 20.0)
    axis.set_ylim(0.0, 1.08 * upper)
    axis.set_xlabel("Physical time")
    axis.set_ylabel(r"Mean field MSE over steps $1,\ldots,h$")
    axis.set_title("Through-horizon direct-rollout error", loc="left")
    _style_axis(axis, "a")


def _reduction_panel(
    axis: plt.Axes,
    values: Mapping[str, np.ndarray],
    summary: Mapping[str, object],
) -> None:
    time = values["physical_time"]
    reduction = 100.0 * values["relative_reduction_of_arm_means"]
    lower = 100.0 * values["relative_reduction_pointwise_ci95_lower"]
    upper = 100.0 * values["relative_reduction_pointwise_ci95_upper"]
    axis.fill_between(
        time,
        lower,
        upper,
        color=ORANGE,
        alpha=0.17,
        linewidth=0,
        zorder=1,
        label="95% pointwise interval (descriptive)",
    )
    axis.plot(time, reduction, color=ORANGE, linewidth=1.55, label="Relative reduction")
    primary = summary["primary_h200"]
    estimate = 100.0 * float(primary["relative_reduction_of_arm_means"])
    primary_lower = 100.0 * float(primary["ci95_lower"])
    primary_upper = 100.0 * float(primary["ci95_upper"])
    axis.errorbar(
        [20.0],
        [estimate],
        yerr=[[estimate - primary_lower], [primary_upper - estimate]],
        fmt="D",
        color="black",
        markerfacecolor=ORANGE,
        markeredgecolor="black",
        markersize=5.2,
        capsize=3,
        linewidth=1.25,
        zorder=5,
        label="H200 primary 95% CI (separate)",
    )
    axis.axhline(0.0, color=GRAY, linestyle=":", linewidth=1.0)
    axis.axvline(20.0, color=LIGHT_GRAY, linewidth=0.7, zorder=0)
    axis.set_xlim(0.0, 20.4)
    axis.set_ylim(0.0, 1.10 * float(upper.max()))
    axis.set_xlabel("Physical time")
    axis.set_ylabel("Sparse reduction in mean MSE (%)")
    axis.set_title("Relative reduction; positive favors sparse", loc="left")
    axis.legend(frameon=False, loc="upper right")
    axis.text(
        0.035,
        0.055,
        (
            f"H200 primary: {estimate:.2f}% "
            f"[{primary_lower:.2f}, {primary_upper:.2f}]\n"
            f"exact one-sided $p={float(primary['one_sided_exact_sign_flip_p']):.4f}$; "
            f"{int(primary['paired_model_seed_wins'])}/10 wins"
        ),
        transform=axis.transAxes,
        fontsize=6.2,
        color=GRAY,
        va="bottom",
    )
    _style_axis(axis, "b")


def render_full_horizon_figure(
    summary: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    """Render arm curves and the separately marked outcome-aware H200 result."""

    values = _arrays(rows)
    style = {
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "legend.fontsize": 6.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(7.15, 3.15),
            constrained_layout=True,
        )
        _mse_panel(axes[0], values)
        _reduction_panel(axes[1], values, summary)
        figure.suptitle(
            str(summary["figure_title"]).replace("--", "–"),
            fontsize=9.0,
            fontweight="bold",
        )
        figure.text(
            0.5,
            -0.075,
            (
                "Three datasets are averaged within each of ten paired model seeds. "
                "Shading is 95% pointwise bootstrap uncertainty (not simultaneous; no curve-wide test).\n"
                f"{summary['figure_disclosure']}"
            ),
            ha="center",
            fontsize=6.1,
            color=GRAY,
        )
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_pdf, bbox_inches="tight", metadata=PDF_METADATA)
        figure.savefig(output_png, dpi=300, bbox_inches="tight", metadata=PNG_METADATA)
        plt.close(figure)
