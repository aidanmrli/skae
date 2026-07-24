"""Deterministic rendering for the global-K support-closure packet."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


TRUE_COLOR = "#D55E00"
NULL_COLOR = "#0072B2"
PAIR_COLOR = "#888888"
FAIL_COLOR = "#E69F00"
PDF_METADATA = {
    "Creator": "SKAE global-K support-closure evidence builder",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
PNG_METADATA = {
    "Software": "SKAE global-K support-closure evidence builder"
}
# At exactly 300 dpi, Agg places one title hyphen on a half-pixel boundary:
# Rome/Milan and Sapphire Rapids then differ by ten antialiased pixels.  A
# 320-dpi export avoids that CPU-specific rounding tie while retaining a
# publication-quality raster companion to the authoritative vector PDF.
PNG_DPI = 320


PANELS = (
    {
        "column": "activity_leakage",
        "title": r"Raw-$K$ activity leakage",
        "ylabel": r"$\|zPK(I-P)\|/\|zPK\|$",
        "ratio_gate": ("le", 0.80),
        "log": True,
    },
    {
        "column": "matrix_leakage",
        "title": r"Raw-$K$ matrix leakage",
        "ylabel": r"$\|PK(I-P)\|_F/\|PK\|_F$",
        "ratio_gate": None,
        "log": True,
    },
    {
        "column": "activity_change_leakage",
        "title": r"$(K-I)$-normalized leakage",
        "ylabel": r"$\|zPK(I-P)\|/\|zP(K-I)\|$",
        "ratio_gate": ("le", 0.80),
        "log": True,
    },
    {
        "column": "matrix_change_leakage",
        "title": r"$(K-I)$ matrix leakage",
        "ylabel": r"$\|PK(I-P)\|_F/\|P(K-I)\|_F$",
        "ratio_gate": None,
        "log": True,
    },
    {
        "column": "restricted_inside_residual",
        "title": r"Post-hoc $PKP$ inside residual",
        "ylabel": "Relative latent residual",
        "ratio_gate": ("le", 0.85),
        "log": True,
    },
    {
        "column": "operator_distance",
        "title": r"Restricted-operator distance",
        "ylabel": "Symmetric normalized Frobenius distance",
        "ratio_gate": ("ge", 1.10),
        "log": False,
    },
)


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


def _plot_panel(
    axis: plt.Axes,
    rows: pd.DataFrame,
    primary_rows: pd.DataFrame,
    specification: dict,
) -> None:
    column = str(specification["column"])
    true = rows[f"{column}_true"].to_numpy(dtype=np.float64)
    null = rows[f"{column}_null"].to_numpy(dtype=np.float64)
    for left, right in zip(true, null):
        axis.plot((0, 1), (left, right), color=PAIR_COLOR, alpha=0.45, linewidth=0.7)
    axis.scatter(
        np.zeros_like(true),
        true,
        color=TRUE_COLOR,
        marker="s",
        s=19,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    axis.scatter(
        np.ones_like(null),
        null,
        color=NULL_COLOR,
        marker="o",
        s=19,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    true_median = float(np.median(true))
    null_median = float(np.median(null))
    ratio = float(np.median(rows[f"{column}_true_over_null"]))
    primary_ratio = float(
        np.median(primary_rows[f"{column}_true_over_null"])
    )
    axis.plot(
        (0, 1),
        (true_median, null_median),
        color="black",
        linewidth=1.8,
        marker="D",
        markersize=4.2,
        zorder=4,
    )
    ratio_gate = specification["ratio_gate"]
    if ratio_gate is None:
        passed = True
        annotation = (
            f"all-current ratio={ratio:.3f}\n"
            f"frozen primary={primary_ratio:.3f}\n"
            "descriptive; no frozen gate"
        )
    else:
        direction, threshold = ratio_gate
        passed = ratio <= threshold if direction == "le" else ratio >= threshold
        relation = r"$\leq$" if direction == "le" else r"$\geq$"
        verdict = "MEETS" if passed else "MISSES"
        annotation = (
            f"all-current ratio={ratio:.3f}\n"
            f"frozen primary={primary_ratio:.3f}\n"
            f"reference {relation}{threshold:.2f}: {verdict}"
        )
    axis.text(
        0.04,
        0.96,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        fontweight="bold" if not passed else "normal",
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "#FFF2CC" if not passed else "white",
            "edgecolor": FAIL_COLOR if not passed else "#BBBBBB",
            "linewidth": 0.8,
            "alpha": 0.95,
        },
    )
    axis.set_xticks((0, 1), ("Observed support", "Matched null"))
    axis.set_xlim(-0.32, 1.32)
    if bool(specification["log"]):
        axis.set_yscale("log")
    axis.set_ylabel(str(specification["ylabel"]))
    axis.set_title(str(specification["title"]), pad=7)
    axis.grid(axis="y", which="both", alpha=0.18, linewidth=0.5)


def render_support_closure(
    all_current_rows: pd.DataFrame,
    primary_rows: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
) -> None:
    """Render the all-current paired diagnostic with frozen-primary context."""
    style = {
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(
            3, 2, figsize=(7.0, 7.8), constrained_layout=False
        )
        figure.subplots_adjust(
            left=0.10,
            right=0.985,
            bottom=0.11,
            top=0.89,
            wspace=0.27,
            hspace=0.42,
        )
        for panel, (axis, specification) in enumerate(
            zip(axes.flat, PANELS), start=1
        ):
            _plot_panel(axis, all_current_rows, primary_rows, specification)
            _style_axis(axis, chr(96 + panel))
        handles = (
            Line2D([], [], marker="s", linestyle="none", color=TRUE_COLOR,
                   label="Observed sparse support"),
            Line2D([], [], marker="o", linestyle="none", color=NULL_COLOR,
                   label="Sign-pair coordinate-permutation null"),
            Line2D([], [], marker="D", linestyle="-", color="black",
                   label="Cross-system medians"),
        )
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=3,
            frameon=False,
        )
        figure.text(
            0.5,
            0.015,
            r"Plotted: post-hoc all-current guard, 15 systems $\times$ 3 seeds; "
            "frozen persistent-primary ratios appear in annotations.\n"
            r"The guard adds 0.96 percentage points of transitions. $PKP$ is "
            "post-hoc; failed differentiation precludes invariant-subspace or "
            "distinct-law claims.",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_pdf, bbox_inches="tight", metadata=PDF_METADATA)
        figure.savefig(
            output_png,
            dpi=PNG_DPI,
            bbox_inches="tight",
            metadata=PNG_METADATA,
        )
        plt.close(figure)
