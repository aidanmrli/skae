#!/usr/bin/env python3
"""Plot a design-map figure for the transition-rich shortlist."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt


DEFAULT_FORMATS = ("png", "svg", "pdf")


@dataclass(frozen=True)
class ShortlistRecord:
    """One shortlisted transition-rich system for design-space plotting."""

    name: str
    family: str
    basins: int
    priority: str
    tier: str
    role: str
    novelty: int
    calibration_risk: int
    paper_value: int
    stage: str
    tags: tuple[str, ...]


def parse_formats_arg(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return DEFAULT_FORMATS
    formats = []
    for item in value.split(","):
        cleaned = item.strip().lower().lstrip(".")
        if cleaned:
            formats.append(cleaned)
    allowed = {"png", "svg", "pdf"}
    unknown = sorted(set(formats) - allowed)
    if unknown:
        raise ValueError(f"Unknown formats: {unknown}. Expected subset of {sorted(allowed)}.")
    return tuple(formats)


def build_shortlist_records() -> list[ShortlistRecord]:
    """Return the current paper-facing shortlist as structured data.

    Scores are qualitative design scores, not empirical results:
    - novelty: how distinct the mechanism is from the current three-system suite
    - calibration_risk: expected tuning difficulty
    - paper_value: likely paper value if the system calibrates cleanly
    """

    return [
        ShortlistRecord(
            name="Triangle central gate and sectors",
            family="gated_transfer",
            basins=3,
            priority="High",
            tier="Elite",
            role="clean chart-switching positive",
            novelty=3,
            calibration_risk=2,
            paper_value=5,
            stage="Anchor",
            tags=("control", "shared_region"),
        ),
        ShortlistRecord(
            name="Four-way crossroads",
            family="bottleneck_transfer",
            basins=4,
            priority="High",
            tier="Elite",
            role="explicit bottleneck transfer",
            novelty=4,
            calibration_risk=2,
            paper_value=5,
            stage="Anchor",
            tags=("control", "routing", "shared_region"),
        ),
        ShortlistRecord(
            name="Sector relay-4",
            family="piecewise_affine",
            basins=4,
            priority="High",
            tier="Elite",
            role="piecewise-affine positive control",
            novelty=4,
            calibration_risk=3,
            paper_value=5,
            stage="Anchor",
            tags=("control", "piecewise"),
        ),
        ShortlistRecord(
            name="Rotating barrier-4",
            family="barrier_transport",
            basins=4,
            priority="High",
            tier="Elite",
            role="moving-separatrix stress test",
            novelty=5,
            calibration_risk=4,
            paper_value=5,
            stage="Hard stress",
            tags=("moving_boundary", "warped_geometry"),
        ),
        ShortlistRecord(
            name="Lens-warp triad",
            family="warped_geometry",
            basins=3,
            priority="High",
            tier="Elite",
            role="warped-geometry positive",
            novelty=5,
            calibration_risk=3,
            paper_value=5,
            stage="Geometry",
            tags=("warped_geometry",),
        ),
        ShortlistRecord(
            name="Braided diamond4",
            family="graph_routing",
            basins=4,
            priority="High",
            tier="Elite",
            role="route-choice topology benchmark",
            novelty=5,
            calibration_risk=4,
            paper_value=5,
            stage="Routing",
            tags=("routing", "multistage"),
        ),
        ShortlistRecord(
            name="Twin pinch bowtie",
            family="slow_fast",
            basins=4,
            priority="High",
            tier="Elite",
            role="two-stage slow-fast routing benchmark",
            novelty=5,
            calibration_risk=4,
            paper_value=5,
            stage="Hard stress",
            tags=("slow_fast", "multistage"),
        ),
        ShortlistRecord(
            name="Arc DAG4",
            family="graph_routing",
            basins=4,
            priority="High",
            tier="Elite",
            role="directed acyclic transition-graph benchmark",
            novelty=5,
            calibration_risk=3,
            paper_value=5,
            stage="Routing",
            tags=("routing", "causal_graph"),
        ),
        ShortlistRecord(
            name="Folded tri canard",
            family="slow_fast",
            basins=3,
            priority="Medium",
            tier="Reserve",
            role="compact slow-fast benchmark",
            novelty=4,
            calibration_risk=4,
            paper_value=4,
            stage="Hard stress",
            tags=("slow_fast",),
        ),
        ShortlistRecord(
            name="Heteroclinic lane-3",
            family="graph_routing",
            basins=3,
            priority="Medium",
            tier="Reserve",
            role="channel-network benchmark",
            novelty=4,
            calibration_risk=4,
            paper_value=4,
            stage="Routing",
            tags=("routing", "shared_region"),
        ),
        ShortlistRecord(
            name="Anisotropic-ridge quartet",
            family="warped_geometry",
            basins=4,
            priority="Medium",
            tier="Reserve",
            role="directional-resistance benchmark",
            novelty=5,
            calibration_risk=4,
            paper_value=4,
            stage="Geometry",
            tags=("warped_geometry", "shared_region"),
        ),
        ShortlistRecord(
            name="Sheet-to-sink hexad",
            family="slow_fast",
            basins=6,
            priority="Medium",
            tier="Reserve",
            role="folded-sheet transport benchmark",
            novelty=5,
            calibration_risk=5,
            paper_value=4,
            stage="Hard stress",
            tags=("slow_fast", "shared_region", "multistage"),
        ),
        ShortlistRecord(
            name="Triad fork graph3",
            family="graph_routing",
            basins=3,
            priority="Medium",
            tier="Reserve",
            role="minimal basin-graph benchmark",
            novelty=5,
            calibration_risk=3,
            paper_value=4,
            stage="Routing",
            tags=("routing", "causal_graph"),
        ),
        ShortlistRecord(
            name="Fan saddle6",
            family="graph_routing",
            basins=6,
            priority="Medium",
            tier="Reserve",
            role="hierarchical branching benchmark",
            novelty=5,
            calibration_risk=5,
            paper_value=4,
            stage="Routing",
            tags=("routing", "multistage"),
        ),
        ShortlistRecord(
            name="Six-basin bipartite bridge",
            family="modular_transfer",
            basins=6,
            priority="Medium",
            tier="Reserve",
            role="modular transfer benchmark",
            novelty=4,
            calibration_risk=4,
            paper_value=4,
            stage="Routing",
            tags=("routing", "shared_region"),
        ),
        ShortlistRecord(
            name="Oblique-trench hex",
            family="warped_geometry",
            basins=6,
            priority="Medium",
            tier="Reserve",
            role="shared-manifold transport benchmark",
            novelty=5,
            calibration_risk=4,
            paper_value=4,
            stage="Geometry",
            tags=("warped_geometry", "shared_region"),
        ),
    ]


def build_elite_shortlist_records() -> list[ShortlistRecord]:
    """Return the stricter top-tier shortlist used for the mechanism atlas."""

    return [record for record in build_shortlist_records() if record.tier == "Elite"]


def _family_palette(records: list[ShortlistRecord]) -> dict[str, tuple[float, float, float]]:
    families = list(dict.fromkeys(record.family for record in records))
    cmap = plt.get_cmap("tab20")
    colors = [cmap(index) for index in range(max(len(families), 3))]
    return {family: colors[index] for index, family in enumerate(families)}


def _save_formats(fig, stem: Path, formats: tuple[str, ...]) -> list[Path]:
    output_paths: list[Path] = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        output_paths.append(path)
    return output_paths


def _stage_groups(records: list[ShortlistRecord]) -> dict[str, list[ShortlistRecord]]:
    stage_order = ["Anchor", "Geometry", "Routing", "Hard stress"]
    groups = {stage: [] for stage in stage_order}
    for record in records:
        groups.setdefault(record.stage, []).append(record)
    return groups


def _card_summary(record: ShortlistRecord) -> tuple[str, str]:
    summaries = {
        "Triangle central gate and sectors": (
            "Local-linear cores plus one shared gate disk and angular exits.",
            "Best clean follow-up to the current chart-switching positive.",
        ),
        "Four-way crossroads": (
            "Orthogonal bottleneck corridors force rerouting at a single crossing.",
            "Easiest explicit transfer geometry to explain in one figure.",
        ),
        "Sector relay-4": (
            "Four stable affine sectors with one explicit transition strip.",
            "Strong piecewise-affine positive control for local charts.",
        ),
        "Lens-warp triad": (
            "A smooth coordinate lens bends trajectories through a warped strip.",
            "Separates visible geometry from transport geometry cleanly.",
        ),
        "Braided diamond4": (
            "Two saddles braid their manifolds before capture into four sinks.",
            "Gives real route choice without relying on a hub.",
        ),
        "Twin pinch bowtie": (
            "Two sequential pinch regions create a staged slow-fast routing web.",
            "Stronger two-stage transition story than a single bottleneck.",
        ),
        "Rotating barrier-4": (
            "Barrier orientation twists with radius, moving the effective separatrix.",
            "Best moving-boundary stress test in the inventory.",
        ),
        "Arc DAG4": (
            "Curved separatrix arcs encode a directed acyclic transition graph.",
            "Cleanest causal routing story for coauthor-facing discussion.",
        ),
    }
    return summaries.get(record.name, (record.role, ""))


def _risk_label(record: ShortlistRecord) -> str:
    if record.calibration_risk <= 2:
        return "Low tuning risk"
    if record.calibration_risk == 3:
        return "Moderate tuning risk"
    if record.calibration_risk == 4:
        return "Elevated tuning risk"
    return "High tuning risk"


def _draw_card_glyph(ax, record: ShortlistRecord, color) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if record.name == "Triangle central gate and sectors":
        centers = np.array([[0.22, 0.30], [0.78, 0.30], [0.50, 0.74]])
        for x, y in centers:
            ax.add_patch(Circle((x, y), 0.08, fill=False, lw=1.8, ec=color))
        ax.add_patch(Circle((0.50, 0.48), 0.11, fill=False, lw=1.6, ls="--", ec="black"))
        for x, y in centers:
            ax.add_patch(FancyArrowPatch((0.50, 0.48), (x, y), arrowstyle="-|>", mutation_scale=10, lw=1.2, color="black"))
        return

    if record.name == "Four-way crossroads":
        ax.plot([0.12, 0.88], [0.50, 0.50], color=color, lw=4)
        ax.plot([0.50, 0.50], [0.12, 0.88], color=color, lw=4)
        for x, y in [(0.18, 0.18), (0.82, 0.18), (0.18, 0.82), (0.82, 0.82)]:
            ax.add_patch(Circle((x, y), 0.07, fill=False, lw=1.6, ec="black"))
        return

    if record.name == "Sector relay-4":
        angles = np.linspace(0, 2 * np.pi, 5)
        for angle in angles[:-1]:
            ax.plot([0.50, 0.50 + 0.38 * np.cos(angle)], [0.50, 0.50 + 0.38 * np.sin(angle)], color=color, lw=2)
        ax.add_patch(Rectangle((0.43, 0.43), 0.14, 0.14, fill=False, lw=1.5, ec="black"))
        return

    if record.name == "Lens-warp triad":
        xs = np.linspace(0.12, 0.88, 100)
        for offset in [0.22, 0.50, 0.78]:
            ys = offset + 0.10 * np.exp(-((xs - 0.5) ** 2) / 0.02) - 0.05
            ax.plot(xs, ys, color=color, lw=1.4)
        ys = np.linspace(0.12, 0.88, 100)
        for offset in [0.24, 0.50, 0.76]:
            xs2 = offset + 0.10 * np.exp(-((ys - 0.5) ** 2) / 0.02) - 0.05
            ax.plot(xs2, ys, color="black", lw=1.1, alpha=0.7)
        return

    if record.name == "Braided diamond4":
        nodes = np.array([[0.50, 0.84], [0.84, 0.50], [0.50, 0.16], [0.16, 0.50]])
        for x, y in nodes:
            ax.add_patch(Circle((x, y), 0.065, fill=False, lw=1.5, ec="black"))
        t = np.linspace(0, 1, 120)
        ax.plot(0.18 + 0.64 * t, 0.72 - 0.40 * np.sin(np.pi * t) ** 2, color=color, lw=2.0)
        ax.plot(0.18 + 0.64 * t, 0.28 + 0.40 * np.sin(np.pi * t) ** 2, color=color, lw=2.0)
        return

    if record.name == "Twin pinch bowtie":
        x = np.linspace(0.12, 0.88, 220)
        y1 = 0.72 - 0.22 * np.exp(-((x - 0.38) ** 2) / 0.01) - 0.22 * np.exp(-((x - 0.62) ** 2) / 0.01)
        y2 = 0.28 + 0.22 * np.exp(-((x - 0.38) ** 2) / 0.01) + 0.22 * np.exp(-((x - 0.62) ** 2) / 0.01)
        ax.plot(x, y1, color=color, lw=2.1)
        ax.plot(x, y2, color=color, lw=2.1)
        ax.add_patch(FancyArrowPatch((0.22, 0.50), (0.78, 0.50), arrowstyle="-|>", mutation_scale=10, lw=1.2, color="black"))
        return

    if record.name == "Rotating barrier-4":
        theta = np.linspace(-0.8, 0.8, 120)
        for scale, alpha in [(0.22, 1.0), (0.30, 0.8), (0.38, 0.6)]:
            x = 0.50 + scale * np.cos(theta + 0.9 * scale * 3)
            y = 0.50 + 0.55 * scale * np.sin(theta)
            ax.plot(x, y, color=color, lw=2.0, alpha=alpha)
        for x0, y0 in [(0.22, 0.22), (0.78, 0.22), (0.22, 0.78), (0.78, 0.78)]:
            ax.add_patch(Circle((x0, y0), 0.06, fill=False, lw=1.5, ec="black"))
        return

    if record.name == "Arc DAG4":
        xs = [0.18, 0.45, 0.72, 0.84]
        ys = [0.72, 0.50, 0.32, 0.62]
        for x, y in zip(xs, ys):
            ax.add_patch(Circle((x, y), 0.055, fill=False, lw=1.5, ec="black"))
        arc_specs = [((0.18, 0.72), (0.45, 0.50), 0.2), ((0.45, 0.50), (0.72, 0.32), -0.1), ((0.45, 0.50), (0.84, 0.62), 0.2)]
        for start, end, rad in arc_specs:
            ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=10, lw=1.5, color=color))
        return

    ax.add_patch(Circle((0.50, 0.50), 0.28, fill=False, lw=2.0, ec=color))


def plot_shortlist_design_map(
    *,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    records = build_shortlist_records()
    palette = _family_palette(records)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    scatter_ax, bar_ax = axes

    for record in records:
        marker = "o" if record.priority == "High" else "s"
        alpha = 0.92 if record.tier == "Elite" else 0.72
        edge_width = 1.15 if record.tier == "Elite" else 0.7
        scatter_ax.scatter(
            record.novelty,
            record.calibration_risk,
            s=120 + 22 * record.basins,
            marker=marker,
            color=palette[record.family],
            edgecolor="black",
            linewidth=edge_width,
            alpha=alpha,
            zorder=3,
        )
        if record.tier == "Elite":
            scatter_ax.text(
                record.novelty + 0.06,
                record.calibration_risk + 0.03,
                f"{record.name} ({record.basins})",
                fontsize=8.6,
                ha="left",
                va="bottom",
            )

    scatter_ax.set_title("Transition-Rich Shortlist Design Map")
    scatter_ax.set_xlabel("Mechanism distinctiveness")
    scatter_ax.set_ylabel("Expected calibration risk")
    scatter_ax.set_xlim(2.6, 5.4)
    scatter_ax.set_ylim(1.6, 5.4)
    scatter_ax.set_xticks([3, 4, 5])
    scatter_ax.set_yticks([2, 3, 4, 5])
    scatter_ax.grid(alpha=0.25)

    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            label=family.replace("_", " "),
        )
        for family, color in palette.items()
    ]
    priority_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label="High priority / elite",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label="Medium priority / reserve",
        ),
    ]
    scatter_ax.legend(
        handles=family_handles + priority_handles,
        loc="upper left",
        fontsize=8,
        frameon=True,
        ncol=1,
    )

    ordered = sorted(records, key=lambda record: (record.paper_value, -record.calibration_risk, record.name))
    y_positions = list(range(len(ordered)))
    bar_colors = [palette[record.family] for record in ordered]
    bar_labels = [record.name for record in ordered]
    bar_values = [record.paper_value for record in ordered]

    bar_ax.barh(y_positions, bar_values, color=bar_colors, edgecolor="black", linewidth=0.7)
    for idx, record in enumerate(ordered):
        bar_ax.text(
            bar_values[idx] + 0.04,
            idx,
            record.role,
            fontsize=8.0,
            va="center",
        )
    bar_ax.set_yticks(y_positions)
    bar_ax.set_yticklabels(bar_labels, fontsize=8.4)
    bar_ax.set_xlabel("Estimated paper value")
    bar_ax.set_title("Shortlist Roles")
    bar_ax.set_xlim(0, 5.8)
    bar_ax.set_xticks([1, 2, 3, 4, 5])
    bar_ax.grid(axis="x", alpha=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "transition_rich_shortlist_design_map"
    return _save_formats(fig, stem, formats=formats)


def plot_mechanism_atlas(
    *,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    """Render a richer multi-panel atlas for the shortlist and elite tier."""

    records = build_shortlist_records()
    elite = build_elite_shortlist_records()
    palette = _family_palette(records)
    stage_groups = _stage_groups(records)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15.6, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.05, 1.0])
    scatter_ax = fig.add_subplot(grid[0, 0])
    matrix_ax = fig.add_subplot(grid[0, 1])
    strip_ax = fig.add_subplot(grid[1, 0])
    ladder_ax = fig.add_subplot(grid[1, 1])

    fig.suptitle("Transition-Rich Mechanism Atlas", fontsize=18, fontweight="bold")

    for record in records:
        marker = "o" if record.tier == "Elite" else "s"
        size = 105 + 26 * record.basins + (18 if record.tier == "Elite" else 0)
        scatter_ax.scatter(
            record.novelty,
            record.calibration_risk,
            s=size,
            marker=marker,
            color=palette[record.family],
            edgecolor="black",
            linewidth=1.05 if record.tier == "Elite" else 0.75,
            alpha=0.95 if record.tier == "Elite" else 0.65,
            zorder=3 if record.tier == "Elite" else 2,
        )
        if record.tier == "Elite":
            scatter_ax.text(
                record.novelty + 0.06,
                record.calibration_risk + 0.03,
                fill(record.name, width=16),
                fontsize=8.6,
                ha="left",
                va="bottom",
            )

    scatter_ax.set_title("A. Shortlist Landscape", loc="left", fontsize=13, fontweight="bold")
    scatter_ax.set_xlabel("Mechanism distinctiveness")
    scatter_ax.set_ylabel("Expected calibration risk")
    scatter_ax.set_xlim(2.6, 5.4)
    scatter_ax.set_ylim(1.6, 5.4)
    scatter_ax.set_xticks([3, 4, 5])
    scatter_ax.set_yticks([2, 3, 4, 5])
    scatter_ax.grid(alpha=0.25)

    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
            label=family.replace("_", " "),
        )
        for family, color in palette.items()
    ]
    tier_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label="Elite",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label="Reserve",
        ),
    ]
    scatter_ax.legend(
        handles=family_handles + tier_handles,
        loc="upper left",
        fontsize=7.6,
        frameon=True,
        ncol=1,
    )

    tag_order = [
        ("control", "Control"),
        ("routing", "Routing"),
        ("warped_geometry", "Warped"),
        ("slow_fast", "Slow-fast"),
        ("moving_boundary", "Moving"),
        ("shared_region", "Shared"),
        ("multistage", "Multi"),
        ("causal_graph", "Graph"),
        ("piecewise", "Piecewise"),
    ]
    matrix_ax.set_title("B. Elite Mechanism Coverage", loc="left", fontsize=13, fontweight="bold")
    matrix_ax.set_xlim(0, len(tag_order))
    matrix_ax.set_ylim(0, len(elite))
    matrix_ax.invert_yaxis()
    matrix_ax.set_xticks([idx + 0.5 for idx in range(len(tag_order))])
    matrix_ax.set_xticklabels([label for _, label in tag_order], rotation=35, ha="right", fontsize=8.2)
    matrix_ax.set_yticks([idx + 0.5 for idx in range(len(elite))])
    matrix_ax.set_yticklabels([f"{record.name} ({record.basins})" for record in elite], fontsize=8.0)
    matrix_ax.grid(False)

    for row, record in enumerate(elite):
        for col, (tag_key, _) in enumerate(tag_order):
            active = tag_key in record.tags
            face = palette[record.family] if active else (0.95, 0.95, 0.95, 1.0)
            rect = Rectangle(
                (col, row),
                1.0,
                1.0,
                facecolor=face,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9 if active else 1.0,
            )
            matrix_ax.add_patch(rect)
    for spine in matrix_ax.spines.values():
        spine.set_visible(False)
    matrix_ax.set_aspect("equal")

    families = list(dict.fromkeys(record.family for record in records))
    family_to_y = {family: idx for idx, family in enumerate(families)}
    for record in records:
        y = family_to_y[record.family]
        marker = "o" if record.tier == "Elite" else "s"
        strip_ax.scatter(
            record.basins,
            y,
            s=110 if record.tier == "Elite" else 80,
            marker=marker,
            color=palette[record.family],
            edgecolor="black",
            linewidth=1.0 if record.tier == "Elite" else 0.7,
            alpha=0.95 if record.tier == "Elite" else 0.7,
            zorder=3,
        )
        if record.tier == "Elite":
            strip_ax.text(
                record.basins + 0.08,
                y + 0.03,
                fill(record.name, width=18),
                fontsize=8.2,
                ha="left",
                va="bottom",
            )

    strip_ax.set_title("C. Family Coverage By Basin Count", loc="left", fontsize=13, fontweight="bold")
    strip_ax.set_xlabel("Intended endpoint basins")
    strip_ax.set_xlim(2.6, 6.7)
    strip_ax.set_xticks([3, 4, 5, 6])
    strip_ax.set_yticks(list(family_to_y.values()))
    strip_ax.set_yticklabels([family.replace("_", " ") for family in families], fontsize=8.3)
    strip_ax.grid(alpha=0.25)

    ladder_ax.set_title("D. Implementation Ladder", loc="left", fontsize=13, fontweight="bold")
    ladder_ax.axis("off")
    stage_specs = [
        ("Anchor", "#e2e8f0"),
        ("Geometry", "#dbeafe"),
        ("Routing", "#dcfce7"),
        ("Hard stress", "#fee2e2"),
    ]
    x_positions = [0.02, 0.27, 0.52, 0.77]
    for x, (stage_name, facecolor) in zip(x_positions, stage_specs):
        ladder_ax.add_patch(
            Rectangle(
                (x, 0.05),
                0.21,
                0.88,
                transform=ladder_ax.transAxes,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=0.9,
            )
        )
        ladder_ax.text(
            x + 0.105,
            0.9,
            stage_name,
            transform=ladder_ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
        )
        entries = stage_groups.get(stage_name, [])
        y = 0.84
        for record in entries:
            label = fill(record.name, width=16)
            ladder_ax.text(
                x + 0.02,
                y,
                f"{label}\n{record.role}",
                transform=ladder_ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
            )
            y -= 0.16 if record.tier == "Elite" else 0.13

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "transition_rich_mechanism_atlas"
    return _save_formats(fig, stem, formats=formats)


def plot_elite_cards(
    *,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    """Render a more polished card-style figure for the elite shortlist."""

    elite = build_elite_shortlist_records()
    palette = _family_palette(elite)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 10.4), constrained_layout=True)
    fig.suptitle("Elite Transition-Rich Benchmark Concepts", fontsize=19, fontweight="bold")

    for ax, record in zip(axes.flat, elite):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        color = palette[record.family]

        ax.add_patch(
            FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.96,
                boxstyle="round,pad=0.012,rounding_size=0.03",
                facecolor="white",
                edgecolor="black",
                linewidth=1.1,
            )
        )
        ax.add_patch(Rectangle((0.02, 0.87), 0.96, 0.11, facecolor=color, edgecolor="none", alpha=0.92))
        ax.text(0.05, 0.925, record.name, fontsize=11.1, fontweight="bold", color="black", ha="left", va="center")
        ax.text(0.95, 0.925, f"{record.basins} basins", fontsize=9.0, ha="right", va="center")

        glyph_ax = ax.inset_axes([0.06, 0.49, 0.38, 0.28])
        _draw_card_glyph(glyph_ax, record, color)

        summary, why = _card_summary(record)
        ax.text(0.50, 0.73, fill(summary, width=28), fontsize=8.7, ha="left", va="top")
        ax.text(0.50, 0.57, fill(why, width=28), fontsize=8.5, ha="left", va="top", color="#334155")

        ax.text(0.06, 0.40, "Mechanism", fontsize=8.4, fontweight="bold", ha="left")
        ax.text(0.06, 0.35, fill(record.role, width=26), fontsize=8.6, ha="left", va="top")

        ax.text(0.06, 0.23, "Tier / stage", fontsize=8.4, fontweight="bold", ha="left")
        ax.text(0.06, 0.18, f"{record.tier} / {record.stage}", fontsize=8.5, ha="left", va="top")

        ax.text(0.56, 0.23, "Calibration", fontsize=8.4, fontweight="bold", ha="left")
        ax.text(0.56, 0.18, _risk_label(record), fontsize=8.5, ha="left", va="top")

        ax.text(0.06, 0.08, "Tags:", fontsize=8.2, fontweight="bold", ha="left")
        ax.text(
            0.16,
            0.08,
            ", ".join(tag.replace("_", " ") for tag in record.tags),
            fontsize=8.2,
            ha="left",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "transition_rich_elite_cards"
    return _save_formats(fig, stem, formats=formats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot transition-rich shortlist figures.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("docs/figures/transition_rich_inventory_20260406"),
        help="Directory where the figure files will be written.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,svg,pdf",
        help="Comma-separated output formats. Supported: png, svg, pdf.",
    )
    args = parser.parse_args()
    formats = parse_formats_arg(args.formats)
    output_paths = []
    output_paths.extend(
        plot_shortlist_design_map(
            output_dir=args.output_dir,
            formats=formats,
        )
    )
    output_paths.extend(
        plot_mechanism_atlas(
            output_dir=args.output_dir,
            formats=formats,
        )
    )
    output_paths.extend(
        plot_elite_cards(
            output_dir=args.output_dir,
            formats=formats,
        )
    )
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
