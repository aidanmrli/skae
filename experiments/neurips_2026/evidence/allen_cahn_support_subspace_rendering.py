"""Outcome-agnostic rendering for the Allen--Cahn support-subspace audit.

This module deliberately consumes only the compact reducer outputs.  It does
not discover scratch results, choose a decision branch, or write an active
paper artifact unless an explicit output path is supplied by a caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SPARSE_COLOR = "#D55E00"
DENSE_COLOR = "#0072B2"
NEUTRAL_COLOR = "#555555"
PASS_COLOR = "#009E73"
FAIL_COLOR = "#CC79A7"
PDF_METADATA = {
    "Creator": "SKAE Allen--Cahn support-subspace evidence renderer",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {
    "Software": "SKAE Allen--Cahn support-subspace evidence renderer"
}

HORIZONS = (160, 200)
PHYSICAL_TIMES = {160: 16, 200: 20}
EXPECTED_SEEDS = tuple(range(64, 74))


def _required_columns() -> set[str]:
    columns = {
        "seed",
        "family_eligible",
        "signature_observed_over_null",
    }
    for horizon in HORIZONS:
        for arm in ("sparse", "dense"):
            columns.update(
                {
                    f"h{horizon}_{arm}_k_leakage",
                    f"h{horizon}_{arm}_k_null",
                    f"h{horizon}_{arm}_kminusI_leakage",
                    f"h{horizon}_{arm}_rho",
                }
            )
        columns.update(
            {
                f"h{horizon}_correct_family_rho",
                f"h{horizon}_wrong_family_rho",
            }
        )
    return columns


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Mechanism decision is missing {'.'.join(keys)}")
        value = value[key]
    return value


def validate_mechanism_display_inputs(
    rows: pd.DataFrame,
    decision: Mapping[str, Any],
    local_k_negative: Mapping[str, Any],
) -> None:
    """Fail closed on incomplete or orientation-ambiguous display inputs."""
    missing = _required_columns() - set(rows.columns)
    if missing:
        raise ValueError(f"Mechanism rows are missing columns: {sorted(missing)}")
    seeds = tuple(sorted(rows["seed"].astype(int).tolist()))
    if seeds != EXPECTED_SEEDS or rows["seed"].duplicated().any():
        raise ValueError("Mechanism display requires one row for each paired seed 64--73")

    positive_columns = [
        column
        for column in _required_columns()
        if column != "seed" and column != "family_eligible"
    ]
    family_columns = [
        column
        for column in positive_columns
        if "family" in column or column == "signature_observed_over_null"
    ]
    exact_columns = [column for column in positive_columns if column not in family_columns]
    exact = rows[exact_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(exact).all() or bool(np.any(exact <= 0.0)):
        raise ValueError("Exact-mask mechanism metrics must be finite and positive")
    family = rows.loc[rows["family_eligible"].astype(bool)]
    if family.empty:
        raise ValueError("No training-qualified family seed is available for display")
    family_values = family[family_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(family_values).all() or bool(np.any(family_values <= 0.0)):
        raise ValueError("Qualified-family display metrics must be finite and positive")

    _nested(decision, "validity", "passed")
    _nested(decision, "exact_fixed_P0_closure", "passed")
    _nested(decision, "decoded_forecast", "passed")
    _nested(decision, "decoded_forecast", "projected_vs_dense_full_passed")
    _nested(decision, "family", "family_passed")
    _nested(decision, "family", "signature_differentiation_passed")
    _nested(decision, "family", "routing_specificity_passed")
    _nested(decision, "family", "signature_ratio_mean")
    signature_interval = _nested(decision, "family", "signature_ratio_bootstrap")
    if len(signature_interval) != 2:
        raise ValueError("Signature bootstrap interval must have two endpoints")
    for horizon in HORIZONS:
        cell = _nested(decision, "family", "routing_specificity", str(horizon))
        if len(cell["restriction_factor_ratio_bootstrap"]) != 2:
            raise ValueError("Routing bootstrap interval must have two endpoints")

    required_negative = {
        "model_seed",
        "mean_local_over_global",
        "terminal_local_over_global",
        "route_coverage",
        "coverage_gate",
        "all_local_updates_zero",
        "different_recipe_from_positive_global_model",
    }
    if required_negative - set(local_k_negative):
        raise ValueError("Local-K negative disclosure is incomplete")


def _style_axis(axis: plt.Axes, panel: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3)
    axis.text(
        -0.14,
        1.08,
        panel,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def _paired_scatter(
    axis: plt.Axes,
    x: float,
    sparse: np.ndarray,
    dense: np.ndarray,
    *,
    label: bool,
) -> None:
    for sparse_value, dense_value in zip(sparse, dense):
        axis.plot(
            (x - 0.09, x + 0.09),
            (sparse_value, dense_value),
            color="#A0A0A0",
            alpha=0.45,
            linewidth=0.65,
            zorder=1,
        )
    axis.scatter(
        np.full(sparse.size, x - 0.09),
        sparse,
        color=SPARSE_COLOR,
        marker="s",
        s=17,
        zorder=2,
        label="Sparse support" if label else None,
    )
    axis.scatter(
        np.full(dense.size, x + 0.09),
        dense,
        facecolors="white",
        edgecolors=DENSE_COLOR,
        marker="o",
        linewidth=0.9,
        s=19,
        zorder=2,
        label="Dense top-$k$ (same $k$)" if label else None,
    )


def _plot_closure(axis: plt.Axes, rows: pd.DataFrame) -> None:
    specifications = (
        (160, r"$T=16$"),
        (200, r"$T=20$"),
    )
    for index, (horizon, _label) in enumerate(specifications):
        sparse = (
            rows[f"h{horizon}_sparse_k_leakage"].to_numpy(float)
            / rows[f"h{horizon}_sparse_k_null"].to_numpy(float)
        )
        dense = (
            rows[f"h{horizon}_dense_k_leakage"].to_numpy(float)
            / rows[f"h{horizon}_dense_k_null"].to_numpy(float)
        )
        _paired_scatter(axis, float(index), sparse, dense, label=index == 0)
    axis.axhline(0.8, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.0)
    axis.text(1.35, 0.8, "frozen gate 0.80", ha="right", va="bottom", fontsize=6.2)
    axis.set_yscale("log")
    axis.set_xticks(range(2), [label for _h, label in specifications])
    axis.set_ylabel(r"Observed raw-$K$ leakage / coordinate null")
    axis.set_title("Cardinality-matched closure specificity")
    axis.legend(frameon=False, fontsize=6.2, loc="best")
    axis.grid(axis="y", which="both", alpha=0.16, linewidth=0.5)
    _style_axis(axis, "a")


def _plot_change_closure(axis: plt.Axes, rows: pd.DataFrame) -> None:
    for index, horizon in enumerate(HORIZONS):
        _paired_scatter(
            axis,
            float(index),
            rows[f"h{horizon}_sparse_kminusI_leakage"].to_numpy(float),
            rows[f"h{horizon}_dense_kminusI_leakage"].to_numpy(float),
            label=index == 0,
        )
    axis.axhline(0.5, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.0)
    axis.text(1.35, 0.5, "absolute gate 0.50", ha="right", va="bottom", fontsize=6.2)
    axis.set_xticks((0, 1), (r"$T=16$", r"$T=20$"))
    axis.set_ylabel(r"$(K-I)$-normalized leakage")
    axis.set_title("Near-identity guard")
    axis.grid(axis="y", alpha=0.16, linewidth=0.5)
    _style_axis(axis, "b")


def _plot_restriction(axis: plt.Axes, rows: pd.DataFrame) -> None:
    for index, horizon in enumerate(HORIZONS):
        _paired_scatter(
            axis,
            float(index),
            rows[f"h{horizon}_sparse_rho"].to_numpy(float),
            rows[f"h{horizon}_dense_rho"].to_numpy(float),
            label=index == 0,
        )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.axhline(1.1, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.0)
    axis.text(1.35, 1.1, "retention gate 1.10", ha="right", va="bottom", fontsize=6.2)
    axis.set_xticks((0, 1), (r"$T=16$", r"$T=20$"))
    axis.set_ylabel("Repeated restriction / mask-once mean MSE")
    axis.set_title("Does the fixed subspace retain forecast utility?")
    axis.grid(axis="y", alpha=0.16, linewidth=0.5)
    _style_axis(axis, "c")


def _plot_routing(
    axis: plt.Axes,
    rows: pd.DataFrame,
    decision: Mapping[str, Any],
) -> None:
    eligible = rows.loc[rows["family_eligible"].astype(bool)]
    for index, horizon in enumerate(HORIZONS):
        ratios = (
            eligible[f"h{horizon}_correct_family_rho"].to_numpy(float)
            / eligible[f"h{horizon}_wrong_family_rho"].to_numpy(float)
        )
        axis.scatter(
            np.full(ratios.size, index),
            ratios,
            color=SPARSE_COLOR,
            marker="s",
            s=18,
            zorder=2,
        )
        cell = _nested(decision, "family", "routing_specificity", str(horizon))
        estimate = float(
            cell["correct_over_wrong_restriction_factor_ratio_of_seed_means"]
        )
        lower, upper = (float(value) for value in cell["restriction_factor_ratio_bootstrap"])
        axis.errorbar(
            [index],
            [estimate],
            yerr=[[estimate - lower], [upper - estimate]],
            fmt="D",
            color="black",
            markersize=4.2,
            linewidth=1.1,
            capsize=3,
            zorder=3,
        )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.axhline(0.9, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.0)
    axis.text(1.35, 0.9, "frozen gate 0.90", ha="right", va="top", fontsize=6.2)
    axis.set_xticks((0, 1), (r"$T=16$", r"$T=20$"))
    axis.set_ylabel("Correct / wrong-family restriction factor")
    axis.set_title("Outcome-blind routing derangement")
    axis.grid(axis="y", alpha=0.16, linewidth=0.5)
    _style_axis(axis, "d")


def _plot_signatures(
    axis: plt.Axes,
    rows: pd.DataFrame,
    decision: Mapping[str, Any],
) -> None:
    eligible = rows.loc[rows["family_eligible"].astype(bool)]
    values = eligible["signature_observed_over_null"].to_numpy(float)
    axis.scatter(
        np.zeros(values.size), values, color=SPARSE_COLOR, marker="s", s=19, zorder=2
    )
    estimate = float(_nested(decision, "family", "signature_ratio_mean"))
    lower, upper = (
        float(value)
        for value in _nested(decision, "family", "signature_ratio_bootstrap")
    )
    axis.errorbar(
        [0],
        [estimate],
        yerr=[[estimate - lower], [upper - estimate]],
        fmt="D",
        color="black",
        markersize=4.5,
        linewidth=1.2,
        capsize=3,
        zorder=3,
        label="Mean and paired-seed bootstrap CI",
    )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.axhline(1.1, color=NEUTRAL_COLOR, linestyle=":", linewidth=1.0)
    axis.set_xlim(-0.45, 0.45)
    axis.set_xticks((0,), ("Training-selected\nfamily pair",))
    axis.set_ylabel("Signature distance / joint-permutation null")
    axis.set_title("Family signature differentiation")
    axis.legend(frameon=False, fontsize=6.2, loc="best")
    axis.grid(axis="y", alpha=0.16, linewidth=0.5)
    _style_axis(axis, "e")


def _plot_gates(axis: plt.Axes, decision: Mapping[str, Any]) -> None:
    exact_checks = _nested(decision, "exact_fixed_P0_closure", "checks")
    dense_specificity = bool(exact_checks.get("dense_activity_specificity", False)) and bool(
        exact_checks.get("dense_matrix_specificity", False)
    )
    gates = (
        ("Packet validity", bool(_nested(decision, "validity", "passed"))),
        ("Exact-support closure", bool(_nested(decision, "exact_fixed_P0_closure", "passed"))),
        ("Sparse vs dense top-$k$", dense_specificity),
        ("Restriction retention", bool(_nested(decision, "decoded_forecast", "passed"))),
        ("Train-fit family utility", bool(_nested(decision, "family", "family_passed"))),
        ("Distinct signatures", bool(_nested(decision, "family", "signature_differentiation_passed"))),
        ("Correct vs wrong routing", bool(_nested(decision, "family", "routing_specificity_passed"))),
        ("Projected sparse vs dense full", bool(_nested(decision, "decoded_forecast", "projected_vs_dense_full_passed"))),
    )
    axis.axis("off")
    for row, (label, passed) in enumerate(gates):
        y = 0.91 - 0.095 * row
        axis.text(0.02, y, label, transform=axis.transAxes, va="center", fontsize=6.6)
        axis.text(
            0.98,
            y,
            "PASS" if passed else "MISS",
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=PASS_COLOR if passed else FAIL_COLOR,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": PASS_COLOR if passed else FAIL_COLOR,
                "linewidth": 0.8,
            },
        )
    branch = str(decision.get("decision", "missing decision")).replace("_", " ")
    axis.text(
        0.02,
        0.06,
        f"Frozen branch: {branch}",
        transform=axis.transAxes,
        fontsize=6.3,
        fontweight="bold",
        va="bottom",
        wrap=True,
    )
    axis.set_title("Predeclared decision accounting", pad=7)
    axis.text(-0.14, 1.08, "f", transform=axis.transAxes, fontsize=9,
              fontweight="bold", va="top")


def _plot_local_negative(axis: plt.Axes, local_k_negative: Mapping[str, Any]) -> None:
    axis.axis("off")
    mean_ratio = float(local_k_negative["mean_local_over_global"])
    terminal_ratio = float(local_k_negative["terminal_local_over_global"])
    coverage = float(local_k_negative["route_coverage"])
    coverage_gate = float(local_k_negative["coverage_gate"])
    lines = (
        f"Different recipe; one model seed ({int(local_k_negative['model_seed'])})",
        f"$T=20$ mean local/global MSE: {mean_ratio:.3f}  (worse)",
        f"$T=20$ terminal local/global MSE: {terminal_ratio:.3f}  (worse)",
        f"Route coverage: {coverage:.3f}  (< {coverage_gate:.2f} gate)",
        "Every local recipe selected zero update"
        if bool(local_k_negative["all_local_updates_zero"])
        else "At least one local recipe updated",
        "Not a causal global-vs-local test"
        if bool(local_k_negative["different_recipe_from_positive_global_model"])
        else "Same recipe as positive global model",
    )
    axis.text(
        0.02,
        0.55,
        "   ".join(lines),
        transform=axis.transAxes,
        va="center",
        fontsize=6.8,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#FAFAFA",
            "edgecolor": FAIL_COLOR,
            "linewidth": 1.1,
        },
    )
    axis.set_title("Required negative: half-global/half-local", pad=7)
    axis.text(-0.03, 1.08, "g", transform=axis.transAxes, fontsize=9,
              fontweight="bold", va="top")


def render_support_subspace_mechanism(
    rows: pd.DataFrame,
    decision: Mapping[str, Any],
    local_k_negative: Mapping[str, Any],
    *,
    output_pdf: Path,
    output_png: Path,
) -> None:
    """Render the frozen mechanism results without choosing favorable cells."""
    validate_mechanism_display_inputs(rows, decision, local_k_negative)
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
        figure = plt.figure(figsize=(7.0, 6.4), constrained_layout=False)
        grid = figure.add_gridspec(
            3,
            3,
            left=0.085,
            right=0.985,
            bottom=0.09,
            top=0.96,
            wspace=0.48,
            hspace=0.56,
            height_ratios=(1.0, 1.0, 0.38),
        )
        axes = [
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[0, 2]),
            figure.add_subplot(grid[1, 0]),
            figure.add_subplot(grid[1, 1]),
            figure.add_subplot(grid[1, 2]),
            figure.add_subplot(grid[2, :]),
        ]
        _plot_closure(axes[0], rows)
        _plot_change_closure(axes[1], rows)
        _plot_restriction(axes[2], rows)
        _plot_routing(axes[3], rows, decision)
        _plot_signatures(axes[4], rows, decision)
        _plot_gates(axes[5], decision)
        _plot_local_negative(axes[6], local_k_negative)
        handles = (
            Line2D([], [], marker="s", linestyle="none", color=SPARSE_COLOR,
                   label="Sparse support"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                   markeredgecolor=DENSE_COLOR, color=DENSE_COLOR,
                   label="Dense cardinality-matched top-$k$"),
            Line2D([], [], marker="D", linestyle="-", color="black",
                   label="Frozen aggregate and 95% seed bootstrap CI"),
        )
        figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                      bbox_to_anchor=(0.5, -0.02), fontsize=6.3)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_pdf, bbox_inches="tight", metadata=PDF_METADATA)
        figure.savefig(output_png, dpi=300, bbox_inches="tight", metadata=PNG_METADATA)
        plt.close(figure)
