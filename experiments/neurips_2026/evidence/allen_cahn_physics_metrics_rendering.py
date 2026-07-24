"""Deterministic publication rendering for Allen--Cahn physics evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ARM_STYLE = {
    "sparse": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.8},
    "dense": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.6},
    "persistence": {"color": "#333333", "linestyle": ":", "linewidth": 1.5},
}

SHORT_LABELS = {
    "nearest_well_pixel_disagreement": "Pixel phase error",
    "modal_well_accuracy": "Modal-well accuracy",
    "well_area_fraction_tv_error": "Well-area TV error",
    "interface_edge_disagreement": "Interface-edge error",
    "free_energy_absolute_error": "Free-energy error",
    "potential_energy_absolute_error": "Potential-energy error",
    "gradient_energy_absolute_error": "Gradient-energy error",
}

CURVE_LABELS = {
    "nearest_well_pixel_disagreement": "Pixel phase",
    "modal_well_accuracy": "Modal accuracy",
    "well_area_fraction_tv_error": "Well-area TV",
    "interface_edge_disagreement": "Interface edges",
    "free_energy_absolute_error": "Free energy",
    "potential_energy_absolute_error": "Potential energy",
    "gradient_energy_absolute_error": "Gradient energy",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _effect_scale(metric: Mapping[str, Any]) -> float:
    """Scale frozen absolute effects to percent-like display units."""

    if metric["direction"] == "higher":
        return 100.0
    dense = float(metric["h200_cumulative"]["dense"])
    if dense <= 0.0:
        raise ValueError("Error-metric display requires a positive dense mean")
    return 100.0 / dense


def _plot_effect_panel(ax: plt.Axes, metrics: Sequence[Mapping[str, Any]]) -> None:
    positions = np.arange(len(metrics), dtype=float)
    all_display_values: list[float] = []
    for position, metric in zip(positions, metrics):
        scale = _effect_scale(metric)
        seed_effects = np.asarray(metric["paired_seed_oriented_absolute_effect"], dtype=float)
        point = float(metric["paired_bootstrap"]["oriented_absolute_improvement"])
        lower = float(metric["paired_bootstrap"]["ci95_lower"])
        upper = float(metric["paired_bootstrap"]["ci95_upper"])
        displayed = seed_effects * scale
        point_display = point * scale
        lower_display = lower * scale
        upper_display = upper * scale
        all_display_values.extend((float(displayed.min()), float(displayed.max()), lower_display, upper_display))
        ax.scatter(
            displayed,
            np.full(displayed.shape, position),
            s=10,
            color="#777777",
            alpha=0.55,
            edgecolors="none",
            zorder=2,
        )
        significant = bool(metric["holm_significant_0p05"])
        ax.errorbar(
            point_display,
            position,
            xerr=np.asarray([[point_display - lower_display], [upper_display - point_display]]),
            fmt="o",
            color="#0072B2",
            markerfacecolor="#0072B2" if significant else "white",
            markeredgewidth=1.1,
            markersize=5.2,
            capsize=2.5,
            linewidth=1.1,
            zorder=3,
        )
    ax.axvline(0.0, color="#666666", linewidth=0.8, zorder=1)
    span = max(all_display_values) - min(all_display_values)
    margin = max(0.5, 0.08 * span)
    ax.set_xlim(min(all_display_values) - margin, max(all_display_values) + margin)
    labels = [SHORT_LABELS[metric["name"]] for metric in metrics]
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Oriented sparse effect (error reduction vs dense, %; accuracy difference, pp)"
    )
    ax.set_title(
        "H200 effects: points = 10 paired seeds; bars = frozen 95% bootstrap CIs; "
        "filled = Holm q≤0.05",
        loc="left",
        pad=4,
    )
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(
        1.0,
        -0.27,
        "Error-metric CIs are frozen absolute-effect CIs divided by the observed dense mean.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        color="#555555",
    )


def _plot_metric_curve(ax: plt.Axes, metric: Mapping[str, Any]) -> None:
    time = np.asarray(metric["physical_time"], dtype=float)
    curves = metric["cumulative_curves"]
    for arm in ("sparse", "dense", "persistence"):
        values = np.asarray(curves[arm], dtype=float)
        style = ARM_STYLE[arm]
        ax.plot(time, values, label=arm.capitalize(), **style)
        ax.plot(time[-1], values[-1], marker="o", markersize=2.6, **style)
    if metric["direction"] == "lower":
        ax.set_yscale("log")
        arrow = "↓"
        scale_note = "; log y"
    else:
        arrow = "↑"
        scale_note = ""
    status = "Holm" if metric["holm_significant_0p05"] else "descriptive"
    ax.set_title(f"{CURVE_LABELS[metric['name']]} {arrow}{scale_note}", loc="left", pad=3)
    ax.text(
        1.0,
        1.025,
        f"{status} q={metric['holm_p']:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#444444",
    )
    ax.set_xlim(0.0, 20.0)
    ax.set_xticks((0.0, 10.0, 20.0))
    ax.grid(color="#E1E1E1", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def render_physics_figure(
    metrics: Sequence[Mapping[str, Any]],
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Render the frozen all-seven result with no metric selection."""

    if len(metrics) != 7 or set(metric["name"] for metric in metrics) != set(SHORT_LABELS):
        raise ValueError("The physics figure must contain every frozen metric exactly once")
    _configure_style()
    figure = plt.figure(figsize=(7.25, 9.0))
    grid = figure.add_gridspec(
        5,
        2,
        height_ratios=(1.48, 1.0, 1.0, 1.0, 1.0),
        hspace=0.62,
        wspace=0.30,
        left=0.155,
        right=0.975,
        top=0.895,
        bottom=0.06,
    )
    effect_ax = figure.add_subplot(grid[0, :])
    _plot_effect_panel(effect_ax, metrics)
    curve_axes = [
        figure.add_subplot(grid[row, column])
        for row in range(1, 5)
        for column in range(2)
    ]
    for ax, metric in zip(curve_axes, metrics):
        _plot_metric_curve(ax, metric)
    for ax in curve_axes[len(metrics) :]:
        ax.axis("off")
    handles, labels = curve_axes[0].get_legend_handles_labels()
    empty_ax = curve_axes[-1]
    empty_ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.82),
        frameon=False,
        ncol=3,
        columnspacing=1.2,
        handlelength=2.2,
    )
    empty_ax.text(
        0.5,
        0.22,
        "Outcome-aware, same-checkpoint\nsecondary analysis; n=10 paired\nmodel seeds, 3 datasets/seed.",
        transform=empty_ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        color="#444444",
    )
    figure.supxlabel("Physical time, T", y=0.016, fontsize=8.0)
    figure.suptitle(
        "Allen–Cahn direct-rollout physics\n"
        "all seven frozen metrics at the H200 cumulative endpoint",
        x=0.155,
        ha="left",
        fontsize=8.5,
        fontweight="bold",
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_metadata = {
        "Title": "Allen-Cahn: all seven frozen metrics at the H200 cumulative endpoint",
        "Author": "SKAE authors",
        "Creator": "experiments.neurips_2026.evidence.allen_cahn_physics_metrics_rendering",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(pdf_path, format="pdf", metadata=fixed_metadata)
    figure.savefig(
        png_path,
        format="png",
        dpi=300,
        metadata={"Title": fixed_metadata["Title"], "Software": fixed_metadata["Creator"]},
    )
    plt.close(figure)


def physics_table_bytes(summary: Mapping[str, Any]) -> bytes:
    """Return the deterministic all-seven H200 LaTeX table."""

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Outcome-aware, same-checkpoint secondary Allen--Cahn analysis showing all seven frozen metrics at the H200 cumulative endpoint. Effects are relative reductions for the six lower-is-better errors and percentage-point differences for modal accuracy. Confidence intervals are paired-seed 95\% intervals for the oriented absolute effect; $q$ is Holm-adjusted across all seven metrics. Bold $q$ values are at most 0.05.}",
        r"\label{tab:allen_cahn_physics_metrics}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Metric & Dense & Sparse & Oriented effect & Wins & Holm $q$ / abs. 95\% CI \\",
        r"\midrule",
    ]
    for metric in summary["metrics"]:
        dense = float(metric["h200_cumulative"]["dense"])
        sparse = float(metric["h200_cumulative"]["sparse"])
        effect = metric["arm_mean_effect"]
        if effect["label"] == "accuracy_percentage_point_difference":
            effect_text = f"{float(effect['value']):+.3f} pp"
        else:
            effect_text = f"{100.0 * float(effect['value']):.2f}\\%"
        bootstrap = metric["paired_bootstrap"]
        q_text = f"{float(metric['holm_p']):.4f}"
        if metric["holm_significant_0p05"]:
            q_text = rf"\textbf{{{q_text}}}"
        ci_text = (
            f"[{float(bootstrap['ci95_lower']):.5f}, "
            f"{float(bootstrap['ci95_upper']):.5f}]"
        )
        lines.append(
            f"{metric['display_name']} & {dense:.5f} & {sparse:.5f} & {effect_text} & "
            f"{int(metric['h200_cumulative_seed_wins'])}/10 & {q_text} / {ci_text} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""))
    return "\n".join(lines).encode()


__all__ = ["SHORT_LABELS", "physics_table_bytes", "render_physics_figure"]
