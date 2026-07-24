"""Deterministic four-panel and table rendering for periodic evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"
LIGHT_GRAY = "#D9D9D9"
ARM_COLOR = {"dense": ORANGE, "sparse": BLUE}
POLICY_STYLE = {"direct": "--", "selected": "-"}
PDF_METADATA = {
    "Title": "Allen-Cahn periodic-reencoding confirmation",
    "Author": "SKAE authors",
    "Creator": "SKAE periodic evidence builder",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {"Software": "SKAE periodic evidence builder"}


def _style_axis(axis: plt.Axes, label: str) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3)
    axis.grid(color=LIGHT_GRAY, linewidth=0.5, alpha=0.65, zorder=0)
    axis.text(
        -0.12, 1.08, label, transform=axis.transAxes,
        fontsize=10, fontweight="bold", va="top",
    )


def _cadence_label(value: str | int) -> str:
    return "Direct" if value == "direct" else f"p={value}"


def _validation_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    payload = summary["display_payload"]["validation_scores"]
    selected = summary["selected_cadences"]
    x = np.arange(9)
    for arm in ("dense", "sparse"):
        rows = payload[arm]
        values = np.asarray([row["mse"] for row in rows], dtype=float)
        axis.plot(
            x, values, color=ARM_COLOR[arm], marker="o", markersize=3.5,
            linewidth=1.35, label=arm.capitalize(), zorder=2,
        )
        selected_index = next(
            index for index, row in enumerate(rows)
            if row["cadence"] == selected[arm]
        )
        axis.scatter(
            [selected_index], [values[selected_index]], marker="*", s=82,
            color=ARM_COLOR[arm], edgecolor="black", linewidth=0.45, zorder=4,
        )
        axis.scatter(
            [0], [values[0]], marker="o", s=31, facecolor="white",
            edgecolor=ARM_COLOR[arm], linewidth=1.1, zorder=3,
        )
    axis.set_xticks(x, [_cadence_label(row["cadence"]) for row in payload["dense"]], rotation=35, ha="right")
    axis.set_ylabel("Validation H200 cumulative MSE")
    axis.set_title("Validation-only cadence selection (★ selected; ○ direct)", loc="left")
    axis.legend(frameon=False, ncol=2, loc="best")
    _style_axis(axis, "a")


def _heldout_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    curves = summary["display_payload"]["heldout_curves"]
    horizon, time_step = int(curves["horizon_steps"]), float(curves["time_step"])
    time = time_step * np.arange(1, horizon + 1)
    for record in curves["series"]:
        arm, policy = record["arm"], record["policy"]
        cadence = record["cadence"]
        axis.plot(
            time, record["mean_cumulative_field_mse"],
            color=ARM_COLOR[arm], linestyle=POLICY_STYLE[policy],
            linewidth=1.55 if policy == "selected" else 1.15,
            label=f"{arm.capitalize()} {policy} ({_cadence_label(cadence)})",
        )
    axis.axvline(20.0, color=GRAY, linestyle=":", linewidth=1.0)
    axis.text(
        20.0, 0.98, "trained horizon", transform=axis.get_xaxis_transform(),
        ha="right", va="top", rotation=90, fontsize=6.3, color=GRAY,
    )
    axis.set_xlim(0.0, max(20.0, horizon * time_step))
    axis.set_xlabel("Physical time, T")
    axis.set_ylabel("Held-out cumulative field MSE")
    title = "Direct and validation-selected autonomous forecasts"
    if not curves["h400_available"]:
        title += " (H400 tier unavailable)"
    axis.set_title(title, loc="left")
    axis.legend(frameon=False, fontsize=5.7, ncol=2, loc="best")
    _style_axis(axis, "b")


def _ratio_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    ratios = summary["display_payload"]["paired_sparse_over_dense_ratios"]
    specs = (("h200", "H200"), ("h400", "H400"), ("h201_h400", "H201–400"))
    offsets = np.linspace(-0.16, 0.16, 10)
    available_values: list[float] = []
    for index, (key, _label) in enumerate(specs):
        values = ratios[key]
        if values is None:
            axis.text(index, 1.02, "unavailable", ha="center", va="bottom", fontsize=6.1, color=GRAY, rotation=90)
            continue
        array = np.asarray(values, dtype=float)
        available_values.extend(array.tolist())
        axis.scatter(
            index + offsets, array, color=BLUE, s=18, alpha=0.72,
            edgecolor="white", linewidth=0.35, zorder=3,
        )
        axis.plot(
            [index - 0.20, index + 0.20], [float(array.mean())] * 2,
            color="black", linewidth=1.4, zorder=4,
        )
    axis.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.0)
    axis.set_xticks(range(3), [label for _key, label in specs])
    axis.set_ylabel("Paired-seed sparse / dense MSE")
    axis.set_title("Selected-policy paired ratios (n=10; bar = seed-ratio mean)", loc="left")
    if available_values:
        lower, upper = min(available_values), max(available_values)
        margin = max(0.04, 0.12 * max(upper - lower, abs(upper - 1), abs(lower - 1)))
        axis.set_ylim(min(lower, 1.0) - margin, max(upper, 1.0) + margin)
    _style_axis(axis, "c")


def _frontier_panel(axis: plt.Axes, summary: Mapping[str, Any]) -> None:
    payload = summary["display_payload"]
    rows = payload["accuracy_refresh_frontier_h400"]
    if not rows:
        axis.text(
            0.5, 0.52, "Complete H400 frontier unavailable\nunder the frozen failure policy",
            transform=axis.transAxes, ha="center", va="center", color=GRAY,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    else:
        selected = summary["selected_cadences"]
        plot_rows = sorted(rows, key=lambda row: int(row["refresh_count"]))
        for arm in ("dense", "sparse"):
            x = np.asarray([row["refresh_count"] for row in plot_rows], dtype=float)
            y = np.asarray(
                [row[f"{arm}_arm_mean_mse"] for row in plot_rows], dtype=float
            )
            axis.plot(x, y, color=ARM_COLOR[arm], marker="o", markersize=3.5, linewidth=1.25, label=arm.capitalize())
            selected_rows = [row for row in rows if row["cadence"] == selected[arm]]
            if selected_rows:
                row = selected_rows[0]
                axis.scatter(
                    [row["refresh_count"]], [row[f"{arm}_arm_mean_mse"]],
                    marker="*", s=80, color=ARM_COLOR[arm], edgecolor="black",
                    linewidth=0.45, zorder=4,
                )
        axis.set_xscale("symlog", linthresh=1.0, base=2)
        ticks = sorted({int(row["refresh_count"]) for row in rows})
        axis.set_xticks(ticks, [str(value) for value in ticks], rotation=35, ha="right")
        axis.legend(frameon=False, ncol=2, loc="best")
    axis.set_xlabel("Decode–reencode refreshes through H400")
    axis.set_ylabel("H400 cumulative field MSE")
    axis.set_title("Accuracy–refresh frontier (★ validation-selected)", loc="left")
    _style_axis(axis, "d")


def render_periodic_figure(
    summary: Mapping[str, Any], pdf_path: Path, png_path: Path
) -> None:
    """Always render the exact four panels frozen before outcome access."""

    panel_ids = summary["fixed_display_contract"]["panel_ids"]
    if panel_ids != [
        "validation_risk", "heldout_curves", "paired_ratios",
        "accuracy_refresh_frontier",
    ]:
        raise ValueError("Periodic figure panel roster drifted")
    style = {
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "axes.labelsize": 7.2, "axes.titlesize": 7.8,
        "legend.fontsize": 6.3, "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            2, 2, figsize=(7.25, 5.8), constrained_layout=True,
        )
        _validation_panel(axes[0, 0], summary)
        _heldout_panel(axes[0, 1], summary)
        _ratio_panel(axes[1, 0], summary)
        _frontier_panel(axes[1, 1], summary)
        branch = str(summary["decision"]["branch"]).replace("_", " ")
        figure.suptitle(
            "Allen–Cahn periodic reencoding: frozen confirmation display\n"
            f"Adjudication branch: {branch}",
            fontsize=9.0, fontweight="bold",
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf_path, format="pdf", metadata=PDF_METADATA)
        figure.savefig(png_path, format="png", dpi=300, metadata=PNG_METADATA)
        plt.close(figure)


def _latex_text(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def _short_role(value: str) -> str:
    mapping = {
        "confirmatory_selection_aware_primary": "Confirmatory",
        "selection_aware_h400_durability": "Selector-rerun secondary",
        "selection_aware_selector_rerun_bootstrap": "Selector-rerun secondary",
        "conditional_fixed_validation_selection": "Conditional secondary",
        "mandatory_unadjusted_descriptive_sensitivity": "Descriptive",
        "unavailable_no_finite_authorized_tier": "Unavailable",
    }
    return mapping[value]


def comparison_table_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Render all ten frozen rows, including unavailable failure branches."""

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\scriptsize",
        r"\caption{Frozen Allen--Cahn periodic-reencoding comparisons. Positive reduction favors the comparison policy. H200 selected sparse versus selected dense is the sole confirmatory row. H400 and within-arm rows are secondary; direct same-cadence rows are descriptive. Missing H400 tiers remain visible as unavailable rather than being dropped. Confidence intervals propagate validation cadence selection where labeled selector-rerun. No exact $p$-value is claimed for within-arm selector-rerun comparisons.}",
        r"\label{tab:allen_cahn_periodic_reencoding}",
        r"\begin{tabular}{lllrrrrl}", r"\toprule",
        r"Endpoint & Baseline & Comparison & Base MSE & Comp. MSE & Reduction [95\% CI] & Wins & $p$ / role \\",
        r"\midrule",
    ]
    for index, row in enumerate(rows):
        if index == 6:
            lines.extend((r"\midrule", r"\multicolumn{8}{l}{Within-arm validation-selected policy versus direct rollout} \\", r"\midrule"))
        if row["status"] == "unavailable_frozen_tier":
            base = comp = effect = wins = p_text = r"--"
        else:
            base = f"{float(row['baseline_mean_mse']):.5g}"
            comp = f"{float(row['comparison_mean_mse']):.5g}"
            effect = (
                f"{100.0 * float(row['relative_reduction']):+.2f}\\% "
                f"[{100.0 * float(row['ci95_lower']):+.2f}, "
                f"{100.0 * float(row['ci95_upper']):+.2f}]"
            )
            wins = f"{int(row['wins_out_of_10'])}/10"
            p_text = r"--" if row["one_sided_p"] is None else f"{float(row['one_sided_p']):.4g}"
        role = _short_role(str(row["inference_role"]))
        lines.append(
            f"{_latex_text(row['endpoint'])} & {_latex_text(row['baseline_policy'])} & "
            f"{_latex_text(row['comparison_policy'])} & {base} & {comp} & {effect} & "
            f"{wins} & {p_text} / {role} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""))
    return "\n".join(lines).encode()


__all__ = ["comparison_table_bytes", "render_periodic_figure"]
