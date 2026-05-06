#!/usr/bin/env python
"""Create the Methods schematic for support-family construction and use."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
import numpy as np


OUT_DIR = Path("docs/figures/neurips_paper_2026")
PDF_PATH = OUT_DIR / "fig_methods_support_family_pipeline.pdf"
PNG_PATH = OUT_DIR / "fig_methods_support_family_pipeline.png"


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#756BB1"
INK = "#1f2933"
MID = "#6b7280"
LIGHT = "#eef2f6"
GRID = "#d8dee7"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 8.0,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_box(ax, x, y, w, h, label=None, fc="white", ec=GRID, lw=0.8, radius=0.045):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        facecolor=fc,
        edgecolor=ec,
        zorder=1,
    )
    ax.add_patch(patch)
    if label:
        ax.text(x + w / 2, y + h - 0.035, label, ha="center", va="top", color=INK, fontsize=8.0)
    return patch


def arrow(ax, start, end, color=INK, lw=0.85, dashed=False, rad=0.0, zorder=5):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        linestyle=(0, (3, 2)) if dashed else "solid",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def soft_curve(ax, start, end, color, lw=4.5, alpha=0.18, rad=0.0, zorder=2):
    sx, sy = start
    ex, ey = end
    mx = (sx + ex) / 2
    dy = ey - sy
    verts = [
        (sx, sy),
        (mx, sy + rad * dy),
        (mx, ey - rad * dy),
        (ex, ey),
    ]
    patch = PathPatch(
        MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]),
        facecolor="none",
        edgecolor=color,
        linewidth=lw,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_basin_panel(ax, x, y, w, h):
    add_box(ax, x, y, w, h, fc="#fbfcfd", ec=GRID, lw=0.75)
    ax.text(x + 0.055, y + h - 0.055, r"state $x_t$", ha="left", va="top", fontsize=8.4, color=INK)
    # Minimal basin field, intentionally schematic.
    bx, by, bw, bh = x + 0.06, y + 0.09, w - 0.12, h - 0.20
    clip = Rectangle((bx, by), bw, bh, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="white", edgecolor=GRID, linewidth=0.5))
    # background regions
    p1 = plt.Polygon([(bx, by + bh), (bx + 0.44 * bw, by + bh), (bx + 0.52 * bw, by + 0.48 * bh), (bx, by + 0.38 * bh)], color=BLUE, alpha=0.08, lw=0)
    p2 = plt.Polygon([(bx + 0.44 * bw, by + bh), (bx + bw, by + bh), (bx + bw, by + 0.37 * bh), (bx + 0.52 * bw, by + 0.48 * bh)], color=GREEN, alpha=0.08, lw=0)
    p3 = plt.Polygon([(bx, by), (bx + bw, by), (bx + bw, by + 0.37 * bh), (bx + 0.52 * bw, by + 0.48 * bh), (bx, by + 0.38 * bh)], color=ORANGE, alpha=0.09, lw=0)
    for p in (p1, p2, p3):
        p.set_clip_path(clip)
        ax.add_patch(p)
    # boundaries
    ax.plot([bx + 0.44 * bw, bx + 0.52 * bw], [by + bh, by + 0.48 * bh], color=MID, lw=0.55, ls=(0, (3, 2)))
    ax.plot([bx, bx + 0.52 * bw], [by + 0.38 * bh, by + 0.48 * bh], color=MID, lw=0.55, ls=(0, (3, 2)))
    ax.plot([bx + 0.52 * bw, bx + bw], [by + 0.48 * bh, by + 0.37 * bh], color=MID, lw=0.55, ls=(0, (3, 2)))
    # samples
    pts = [
        (bx + 0.24 * bw, by + 0.70 * bh, BLUE),
        (bx + 0.30 * bw, by + 0.78 * bh, BLUE),
        (bx + 0.74 * bw, by + 0.70 * bh, GREEN),
        (bx + 0.69 * bw, by + 0.79 * bh, GREEN),
        (bx + 0.50 * bw, by + 0.20 * bh, ORANGE),
        (bx + 0.57 * bw, by + 0.27 * bh, ORANGE),
    ]
    for px, py, c in pts:
        ax.scatter([px], [py], s=13, c=c, edgecolors="white", linewidths=0.35, zorder=4)
    ax.text(x + w / 2, y + 0.035, "basin labels withheld", ha="center", va="bottom", fontsize=7.2, color=MID)


def draw_sparse_code(ax, x, y, w, h):
    add_box(ax, x, y, w, h, fc="white", ec=GRID, lw=0.75)
    ax.text(x + 0.055, y + h - 0.055, r"sparse code $z_t=E_\theta(x_t)$", ha="left", va="top", fontsize=8.4, color=INK)
    n = 18
    rng_heights = np.array([0.08, 0.12, 0.72, 0.10, 0.28, 0.08, 0.58, 0.14, 0.10, 0.44, 0.09, 0.11, 0.78, 0.19, 0.08, 0.35, 0.10, 0.63])
    active = {2, 6, 9, 12, 15, 17}
    bx, by = x + 0.08, y + 0.23
    bw, bh = w - 0.16, h - 0.42
    bar_w = bw / n * 0.48
    for i in range(n):
        xx = bx + i * bw / n + bar_w * 0.25
        color = BLUE if i in active else "#dbe2ea"
        alpha = 0.85 if i in active else 0.95
        ax.add_patch(Rectangle((xx, by), bar_w, rng_heights[i] * bh, facecolor=color, edgecolor="none", alpha=alpha))
    ax.plot([bx, bx + bw], [by, by], color=GRID, lw=0.55)
    ax.text(x + w / 2, y + 0.070, "few nonzero coordinates", ha="center", va="bottom", fontsize=7.2, color=MID)


def mask_row(ax, x, y, pattern, color, cell=0.020, gap=0.006, alpha=0.92, outline=True):
    for i, bit in enumerate(pattern):
        xx = x + i * (cell + gap)
        fc = color if bit else "white"
        ec = color if bit else "#cfd7e3"
        ax.add_patch(
            Rectangle(
                (xx, y),
                cell,
                cell,
                facecolor=fc,
                edgecolor=ec if outline else "none",
                linewidth=0.45,
                alpha=alpha if bit else 1.0,
                zorder=4,
            )
        )


def draw_support_rules(ax, x, y, w, h):
    add_box(ax, x, y, w, h, fc="#fbfcfd", ec=GRID, lw=0.75)
    ax.text(x + 0.055, y + h - 0.055, "support rules", ha="left", va="top", fontsize=8.4, color=INK)
    ax.text(x + 0.07, y + 0.67 * h, r"$S_{\rm abs}$: threshold", ha="left", va="center", fontsize=8.0, color=INK)
    ax.text(x + 0.07, y + 0.30 * h, r"$S_{\rm top8}$: largest magnitudes", ha="left", va="center", fontsize=8.0, color=INK)
    base = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
    top8 = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    mask_row(ax, x + 0.50 * w, y + 0.64 * h, base, BLUE, cell=0.0185, gap=0.0045)
    mask_row(ax, x + 0.50 * w, y + 0.27 * h, top8, PURPLE, cell=0.0185, gap=0.0045)
    ax.text(x + 0.70 * w, y + 0.52 * h, "binary active-coordinate masks", ha="center", va="center", fontsize=7.1, color=MID)


def draw_family_merge(ax, x, y, w, h):
    add_box(ax, x, y, w, h, fc="white", ec=GRID, lw=0.75)
    ax.text(x + 0.055, y + h - 0.055, "Jaccard family merge", ha="left", va="top", fontsize=8.4, color=INK)
    ax.text(x + 0.055, y + h - 0.120, "frequency ordered; no basin labels", ha="left", va="top", fontsize=7.1, color=MID)
    patterns = [
        ([1, 0, 0, 1, 0, 1, 0, 0], BLUE, 0),
        ([1, 0, 0, 1, 0, 1, 1, 0], BLUE, 0),
        ([0, 1, 0, 0, 1, 0, 1, 0], GREEN, 1),
        ([0, 1, 0, 0, 1, 0, 0, 1], GREEN, 1),
        ([0, 0, 1, 0, 0, 1, 0, 1], ORANGE, 2),
        ([0, 0, 1, 0, 1, 1, 0, 1], ORANGE, 2),
    ]
    yy0 = y + 0.61 * h
    for k, (pat, col, fam) in enumerate(patterns):
        yy = yy0 - k * 0.055
        mask_row(ax, x + 0.08, yy, pat, col, cell=0.016, gap=0.004)
        arrow(ax, (x + 0.31, yy + 0.010), (x + 0.43, y + (0.69 - fam * 0.20) * h), color=col, lw=0.55, rad=0.05)
    for fam, (col, lab) in enumerate([(BLUE, r"$F_1$"), (GREEN, r"$F_2$"), (ORANGE, r"$F_3$")]):
        cy = y + (0.69 - fam * 0.20) * h
        add_box(ax, x + 0.45, cy - 0.035, 0.25, 0.070, label=None, fc=col, ec=col, lw=0.0, radius=0.020)
        ax.text(x + 0.575, cy, lab, ha="center", va="center", fontsize=8.2, color="white")
    ax.text(x + 0.50 * w, y + 0.050, r"families group nearby masks, not states by oracle labels", ha="center", va="bottom", fontsize=7.0, color=MID)


def draw_uses(ax, x, y, w, h):
    add_box(ax, x, y, w, h, fc="#fbfcfd", ec=GRID, lw=0.75)
    ax.text(x + 0.055, y + h - 0.055, "two downstream uses", ha="left", va="top", fontsize=8.4, color=INK)
    # Evaluation row.
    add_box(ax, x + 0.07, y + 0.52 * h, w - 0.14, 0.26 * h, fc="white", ec="#d9e1ec", lw=0.65, radius=0.030)
    ax.text(x + 0.12, y + 0.70 * h, r"$F_{\rm abs}$", ha="left", va="center", fontsize=9.0, color=BLUE)
    arrow(ax, (x + 0.33 * w, y + 0.70 * h), (x + 0.52 * w, y + 0.70 * h), color=MID, lw=0.65, dashed=True)
    ax.text(x + 0.60 * w, y + 0.70 * h, r"$H(B\,|\,F_{\rm abs})$", ha="left", va="center", fontsize=8.4, color=INK)
    ax.text(x + 0.12, y + 0.575 * h, "post-hoc evaluation only", ha="left", va="center", fontsize=7.0, color=MID)
    # Routing row.
    add_box(ax, x + 0.07, y + 0.16 * h, w - 0.14, 0.28 * h, fc="white", ec="#d9e1ec", lw=0.65, radius=0.030)
    ax.text(x + 0.12, y + 0.35 * h, r"$F_{\rm top8}$", ha="left", va="center", fontsize=9.0, color=PURPLE)
    arrow(ax, (x + 0.35 * w, y + 0.35 * h), (x + 0.49 * w, y + 0.35 * h), color=INK, lw=0.75)
    ax.text(x + 0.52 * w, y + 0.35 * h, r"$K_f$", ha="left", va="center", fontsize=9.0, color=INK)
    arrow(ax, (x + 0.61 * w, y + 0.35 * h), (x + 0.75 * w, y + 0.35 * h), color=INK, lw=0.75)
    ax.text(x + 0.78 * w, y + 0.35 * h, r"$\hat{x}_{t+1}$", ha="left", va="center", fontsize=9.0, color=INK)
    ax.text(x + 0.12, y + 0.22 * h, "label-free prediction route", ha="left", va="center", fontsize=7.0, color=MID)


def draw_figure() -> plt.Figure:
    configure()
    fig = plt.figure(figsize=(5.55, 2.82), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.028, 0.948, "A", ha="left", va="top", fontsize=9.3, color=INK)
    ax.text(0.056, 0.948, "two support-derived regime variables", ha="left", va="top", fontsize=8.8, color=INK)
    ax.plot([0.028, 0.972], [0.900, 0.900], color="#dce2eb", lw=0.75)

    # Shared encoder path: x -> E -> sparse code.
    bx, by, bw, bh = 0.044, 0.645, 0.060, 0.175
    ax.text(bx + bw / 2, by + bh + 0.030, r"$x_t$", ha="center", va="bottom", fontsize=9.8, color=INK)
    ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="white", edgecolor="#cbd5e1", lw=0.65))
    ax.add_patch(plt.Polygon([(bx, by + bh), (bx + bw, by + bh), (bx + 0.54 * bw, by + 0.50 * bh), (bx, by + 0.62 * bh)], color=BLUE, alpha=0.08, lw=0))
    ax.add_patch(plt.Polygon([(bx + bw, by + bh), (bx + bw, by), (bx + 0.54 * bw, by + 0.50 * bh)], color=GREEN, alpha=0.08, lw=0))
    ax.add_patch(plt.Polygon([(bx, by), (bx + bw, by), (bx + 0.54 * bw, by + 0.50 * bh), (bx, by + 0.62 * bh)], color=ORANGE, alpha=0.09, lw=0))
    ax.plot([bx, bx + 0.54 * bw, bx + bw], [by + 0.62 * bh, by + 0.50 * bh, by + 0.60 * bh], color=MID, lw=0.50, ls=(0, (3, 2)))
    for px, py, c in [
        (bx + 0.29 * bw, by + 0.72 * bh, BLUE),
        (bx + 0.69 * bw, by + 0.72 * bh, GREEN),
        (bx + 0.53 * bw, by + 0.23 * bh, ORANGE),
    ]:
        ax.scatter([px], [py], s=12, c=c, edgecolors="white", linewidths=0.35, zorder=6)
    ax.text(bx + bw / 2, by - 0.034, "labels hidden", ha="center", va="top", fontsize=6.4, color=MID)

    y_mid = 0.727
    arrow(ax, (0.116, y_mid), (0.147, y_mid), lw=0.90)
    add_box(ax, 0.156, y_mid - 0.036, 0.059, 0.072, fc="white", ec="#cfd8e3", lw=0.65, radius=0.012)
    ax.text(0.1855, y_mid, r"$E_\theta$", ha="center", va="center", fontsize=9.4, color=INK)
    arrow(ax, (0.220, y_mid), (0.252, y_mid), lw=0.90)

    bar_x, bar_y = 0.264, y_mid - 0.070
    heights = [0.15, 0.86, 0.17, 0.57, 0.12, 0.74, 0.20, 0.44, 0.13, 0.66]
    active = {1, 3, 5, 7, 9}
    for i, ht in enumerate(heights):
        ax.add_patch(Rectangle((bar_x + i * 0.0063, bar_y), 0.0040, ht * 0.145, facecolor=BLUE if i in active else "#dce3ec", edgecolor="none"))
    ax.plot([bar_x - 0.004, bar_x + 0.070], [bar_y, bar_y], color="#cfd8e3", lw=0.55)
    ax.text(bar_x + 0.034, by + bh + 0.030, r"$z_t$", ha="center", va="bottom", fontsize=9.8, color=INK)
    arrow(ax, (0.340, y_mid), (0.378, y_mid), lw=0.90)

    # Two explicit support lanes.
    lane_abs = 0.795
    lane_top = 0.638
    ax.plot([0.392, 0.948], [0.716, 0.716], color="#e6ebf2", lw=0.65)
    for y, label, rule, mask_color, family_label, family_color in [
        (lane_abs, r"$S_{\rm abs}$", r"$|z|>\tau$", BLUE, r"$F_{\rm abs}$", BLUE),
        (lane_top, r"$S_{\rm top8}$", "top 8", PURPLE, r"$F_{\rm top8}$", PURPLE),
    ]:
        add_box(ax, 0.395, y - 0.043, 0.150, 0.086, fc="#fbfcfd", ec="#d9e1ec", lw=0.65, radius=0.013)
        ax.text(0.408, y + 0.017, label, ha="left", va="center", fontsize=8.4, color=INK)
        ax.text(0.408, y - 0.021, rule, ha="left", va="center", fontsize=6.3, color=MID)
        pat = [1, 0, 0, 1, 0, 1] if y == lane_abs else [1, 1, 0, 1, 0, 1]
        mask_row(ax, 0.488, y - 0.005, pat, mask_color, cell=0.0098, gap=0.0032)
        arrow(ax, (0.548, y), (0.590, y), lw=0.80, color=MID if y == lane_abs else INK, dashed=(y == lane_abs))
        add_box(ax, 0.600, y - 0.036, 0.090, 0.072, fc="white", ec=family_color, lw=0.75, radius=0.012)
        ax.text(0.645, y, family_label, ha="center", va="center", fontsize=8.0, color=family_color)

    arrow(ax, (0.695, lane_abs), (0.753, lane_abs), color=MID, lw=0.75, dashed=True)
    ax.text(0.765, lane_abs, r"$H(B\,|\,F_{\rm abs})$", ha="left", va="center", fontsize=9.0, color=INK)
    ax.text(0.765, lane_abs - 0.055, "benchmark evaluation only", ha="left", va="center", fontsize=6.5, color=MID)
    arrow(ax, (0.695, lane_top), (0.745, lane_top), color=INK, lw=0.85)
    add_box(ax, 0.755, lane_top - 0.036, 0.056, 0.072, fc=GREEN, ec=GREEN, lw=0.0, radius=0.012)
    ax.text(0.783, lane_top, r"$K_f$", ha="center", va="center", fontsize=8.4, color="white")
    arrow(ax, (0.816, lane_top), (0.854, lane_top), lw=0.85)
    ax.text(0.864, lane_top, r"$\hat{x}_{t+1}$", ha="left", va="center", fontsize=9.5, color=INK)
    ax.text(0.755, lane_top - 0.090, "label-free local prediction", ha="left", va="center", fontsize=6.5, color=MID)

    # Bottom zoom-in: support-family construction.
    ax.text(0.028, 0.500, "B", ha="left", va="top", fontsize=9.3, color=INK)
    ax.text(0.056, 0.500, "label-free support-family construction", ha="left", va="top", fontsize=8.8, color=INK)
    ax.plot([0.028, 0.972], [0.452, 0.452], color="#dce2eb", lw=0.75)
    ax.text(0.044, 0.400, "exact masks", ha="left", va="center", fontsize=6.8, color=MID)
    ax.text(0.470, 0.400, "families", ha="left", va="center", fontsize=6.8, color=MID)
    ax.text(0.705, 0.400, "rule", ha="left", va="center", fontsize=6.8, color=MID)
    patterns = [
        ([1, 0, 1, 0, 0, 1, 0, 0], BLUE, 0),
        ([1, 0, 1, 0, 1, 1, 0, 0], BLUE, 0),
        ([0, 1, 0, 1, 0, 0, 1, 0], GREEN, 1),
        ([0, 1, 0, 1, 0, 0, 0, 1], GREEN, 1),
        ([0, 0, 1, 0, 1, 0, 0, 1], ORANGE, 2),
        ([0, 0, 1, 1, 1, 0, 0, 1], ORANGE, 2),
    ]
    base_y = 0.338
    row_step = 0.050
    family_y = [0.325, 0.225, 0.125]
    for k, (pat, col, fam) in enumerate(patterns):
        yy = base_y - k * row_step
        mask_row(ax, 0.045, yy, pat, col, cell=0.0132, gap=0.0042)
        soft_curve(ax, (0.188, yy + 0.007), (0.442, family_y[fam]), col, lw=5.0, alpha=0.15, rad=0.05)
        arrow(ax, (0.190, yy + 0.007), (0.431, family_y[fam]), color=col, lw=0.50, rad=0.04)
    for y, col, label in [(family_y[0], BLUE, r"$F_1$"), (family_y[1], GREEN, r"$F_2$"), (family_y[2], ORANGE, r"$F_3$")]:
        add_box(ax, 0.450, y - 0.026, 0.072, 0.052, fc=col, ec=col, lw=0.0, radius=0.011)
        ax.text(0.486, y, label, ha="center", va="center", fontsize=9.0, color="white")
    ax.text(0.705, 0.305, "greedy Jaccard merge", ha="left", va="center", fontsize=7.2, color=INK)
    ax.text(0.705, 0.240, "process masks by frequency", ha="left", va="center", fontsize=6.5, color=MID)
    ax.text(0.705, 0.175, "no basin labels or basin count", ha="left", va="center", fontsize=6.5, color=MID)

    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = draw_figure()
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(PNG_PATH, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    print(PDF_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
