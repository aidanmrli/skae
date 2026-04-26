#!/usr/bin/env python3
"""Plot a grounded audit figure for the implemented Claude catalog."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "claude_catalog_audit_20260407"
DEFAULT_SCREENING_PATH = (
    REPO_ROOT
    / "results"
    / "claude_catalog_priority_screen_20260407"
    / "combined_screening_results.json"
)
DEFAULT_FORMATS = ("png", "svg", "pdf")
CATALOG_MODULES = (
    "skae.claude_catalog.systems_gradient",
    "skae.claude_catalog.systems_bio_physical",
    "skae.claude_catalog.systems_creative",
    "skae.claude_catalog.systems_novel",
    "skae.claude_catalog.systems_tuned",
    "skae.claude_catalog.systems_variants",
    "skae.claude_catalog.systems_hybrid",
)


@dataclass(frozen=True)
class UnscreenedPriorityRecord:
    """Recommended next system to screen from the unscreened backlog."""

    name: str
    bucket: str
    rationale: str


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


def import_catalog_registry() -> tuple[list[str], dict[str, object]]:
    """Import all catalog modules and return registered names and classes."""

    sys.path.insert(0, str(REPO_ROOT))
    for module_name in CATALOG_MODULES:
        __import__(module_name)

    from skae.claude_catalog.registry import CATALOG_REGISTRY, list_systems

    return list_systems(), CATALOG_REGISTRY


def load_screening_records(screening_path: Path = DEFAULT_SCREENING_PATH) -> list[dict[str, object]]:
    rows = json.loads(screening_path.read_text())
    cleaned = [row for row in rows if "error" not in row]
    for row in cleaned:
        row["strict_crossing_pass"] = strict_crossing_pass(row)
        row["relaxed_crossing_pass"] = relaxed_crossing_pass(row)
        row["status"] = classify_screening_status(row)
        row["strict_crossing_gap"] = strict_crossing_gap(row)
    return cleaned


def load_combined_screening_records(
    screening_path: Path = DEFAULT_SCREENING_PATH,
    extra_screening_paths: tuple[Path, ...] = (),
) -> list[dict[str, object]]:
    """Load one or more screening packets and merge by system name."""

    merged: dict[str, dict[str, object]] = {}
    for path in (screening_path, *extra_screening_paths):
        for row in load_screening_records(path):
            merged[str(row["name"])] = row
    return [merged[name] for name in sorted(merged)]


def crossing_band_gap(crossing: float, lower: float = 0.30, upper: float = 0.70) -> float:
    if crossing < lower:
        return lower - crossing
    if crossing > upper:
        return crossing - upper
    return 0.0


def per_basin_crossing_values(row: dict[str, object]) -> list[float]:
    per_basin = row.get("per_basin_crossing") or {}
    return [float(value) for value in per_basin.values()]


def strict_crossing_pass(row: dict[str, object], lower: float = 0.30, upper: float = 0.70) -> bool:
    values = per_basin_crossing_values(row)
    return bool(values) and all(lower <= value <= upper for value in values)


def relaxed_crossing_pass(row: dict[str, object]) -> bool:
    values = per_basin_crossing_values(row)
    if not values:
        return False
    overall_crossing = float(row.get("overall_crossing", 0.0))
    return overall_crossing >= 0.30 and min(values) >= 0.25 and max(values) <= 0.75


def strict_crossing_gap(row: dict[str, object], lower: float = 0.30, upper: float = 0.70) -> float:
    values = per_basin_crossing_values(row)
    if not values:
        return float("inf")
    return max(crossing_band_gap(value, lower=lower, upper=upper) for value in values)


def classify_screening_status(row: dict[str, object]) -> str:
    if bool(row["all_pass"]) and strict_crossing_pass(row):
        return "strict_cross_pass"
    if bool(row["all_pass"]):
        return "accepted_relaxed_pass"
    if bool(row["basin_pass"]) and bool(row["occ_pass"]):
        return "retune_frontier"
    return "structure_fail"


def retune_note(row: dict[str, object]) -> str:
    status = str(row.get("status", ""))
    if status == "strict_cross_pass":
        return "accepted; every basin stays in the strict crossing band"
    if status == "accepted_relaxed_pass":
        return "accepted; one basin leaves the strict band but stays inside the relaxed gate"
    crossing = float(row["overall_crossing"])
    if 0.30 <= crossing <= 0.70:
        return "good global crossing, uneven per-basin routing"
    if crossing < 0.30:
        return "low crossing, needs stronger transport"
    return "crossing too high, needs damping"


def build_retune_frontier(records: list[dict[str, object]], limit: int = 8) -> list[dict[str, object]]:
    frontier = [
        record
        for record in records
        if record["status"] in {"strict_cross_pass", "accepted_relaxed_pass", "retune_frontier"}
    ]
    priority = {
        "strict_cross_pass": 0,
        "accepted_relaxed_pass": 1,
        "retune_frontier": 2,
    }
    frontier.sort(
        key=lambda record: (
            priority[str(record["status"])],
            float(record["strict_crossing_gap"]),
            abs(float(record["overall_crossing"]) - 0.45),
            -float(record["min_occupancy"]),
            abs(int(record["n_basins"]) - 4),
            str(record["name"]),
        )
    )
    return frontier[:limit]


def build_unscreened_priority_records() -> list[UnscreenedPriorityRecord]:
    return [
        UnscreenedPriorityRecord(
            name="cal_triangle_3",
            bucket="control",
            rationale="clean 3-basin control to anchor any richer benchmark packet",
        ),
        UnscreenedPriorityRecord(
            name="cal_square_4",
            bucket="control",
            rationale="simple 4-basin geometry for a cleaner bottleneck-free baseline",
        ),
        UnscreenedPriorityRecord(
            name="cal_pentagon_5",
            bucket="control",
            rationale="mid-count polygon sanity check already cited in the stale benchmark note",
        ),
        UnscreenedPriorityRecord(
            name="cal_octagon_8",
            bucket="control",
            rationale="high-basin scalability control with still-legible geometry",
        ),
        UnscreenedPriorityRecord(
            name="cal_hexagon_6",
            bucket="control",
            rationale="higher-basin polygon control that only missed on one weak basin-crossing read",
        ),
        UnscreenedPriorityRecord(
            name="cal_asymmetric_3",
            bucket="variant",
            rationale="breaks symmetry without adding visual complexity",
        ),
        UnscreenedPriorityRecord(
            name="cal_star_5",
            bucket="variant",
            rationale="radial topology is more interesting than another regular polygon",
        ),
        UnscreenedPriorityRecord(
            name="var_l_shape_5",
            bucket="variant",
            rationale="non-convex topology with clear paper-facing geometry",
        ),
        UnscreenedPriorityRecord(
            name="var_diamond_4",
            bucket="variant",
            rationale="rotated 4-basin separatrix test that should be easy to diagnose",
        ),
        UnscreenedPriorityRecord(
            name="var_depth_gradient_4",
            bucket="variant",
            rationale="direct occupancy-imbalance stress test with interpretable asymmetry",
        ),
        UnscreenedPriorityRecord(
            name="var_random_5a",
            bucket="variant",
            rationale="useful robustness read because it nearly works without any visible symmetry",
        ),
        UnscreenedPriorityRecord(
            name="var_mixed_widths_5",
            bucket="variant",
            rationale="width heterogeneity is a cleaner next variant than another geometric rearrangement",
        ),
        UnscreenedPriorityRecord(
            name="var_grid_2x2",
            bucket="variant",
            rationale="simple grid topology can clarify whether failures are geometric or purely tuning-driven",
        ),
        UnscreenedPriorityRecord(
            name="var_random_4a",
            bucket="variant",
            rationale="compact non-symmetric 4-basin variant worth checking after the stronger 5-basin random failure",
        ),
        UnscreenedPriorityRecord(
            name="transition_routes_4",
            bucket="hybrid",
            rationale="closest unscreened explicit route-choice benchmark in the implemented catalog",
        ),
        UnscreenedPriorityRecord(
            name="non_voronoi_basins",
            bucket="hybrid",
            rationale="interesting boundary-shape benchmark where geometry and routing should diverge",
        ),
        UnscreenedPriorityRecord(
            name="slow_fast_triple",
            bucket="hybrid",
            rationale="best unscreened slow-fast mechanism candidate for a richer benchmark",
        ),
        UnscreenedPriorityRecord(
            name="hybrid_state_dep_rot_5",
            bucket="hybrid",
            rationale="state-dependent crossing intensity is richer than fixed-omega controls",
        ),
        UnscreenedPriorityRecord(
            name="hybrid_rotating_centers_3",
            bucket="hybrid",
            rationale="rotational preference benchmark that could be cleaner than the slow-fast candidates",
        ),
        UnscreenedPriorityRecord(
            name="mixed_dynamics_triple",
            bucket="hybrid",
            rationale="local dynamics differ by basin, making support reuse more interesting",
        ),
    ]


def build_unscreened_backlog(
    registry_names: list[str], records: list[dict[str, object]], registry: dict[str, object]
) -> tuple[list[str], dict[str, int]]:
    screened_names = {str(record["name"]) for record in records}
    unscreened = [name for name in registry_names if name not in screened_names]
    counts = {"cal": 0, "var": 0, "hybrid_or_other": 0}
    for name in unscreened:
        category = getattr(registry[name], "category", "?")
        if name.startswith("cal_"):
            counts["cal"] += 1
        elif name.startswith("var_"):
            counts["var"] += 1
        elif category == "H":
            counts["hybrid_or_other"] += 1
        else:
            counts["hybrid_or_other"] += 1
    return unscreened, counts


def apply_base_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_status_scatter(
    ax: plt.Axes,
    records: list[dict[str, object]],
    frontier: list[dict[str, object]],
) -> None:
    colors = {
        "strict_cross_pass": "#2A9D8F",
        "accepted_relaxed_pass": "#3B82F6",
        "retune_frontier": "#F4A261",
        "structure_fail": "#7A8BA3",
    }
    labels = {
        "strict_cross_pass": "strict crossing pass",
        "accepted_relaxed_pass": "accepted via relaxed crossing",
        "retune_frontier": "retune frontier",
        "structure_fail": "fails basin/occupancy",
    }

    ax.axhspan(0.30, 0.70, color="#E8F4EA", alpha=0.9, zorder=0)
    ax.axhline(0.30, color="#93C5AA", linestyle="--", linewidth=1.0)
    ax.axhline(0.70, color="#93C5AA", linestyle="--", linewidth=1.0)

    for status in ("structure_fail", "retune_frontier", "accepted_relaxed_pass", "strict_cross_pass"):
        subset = [record for record in records if record["status"] == status]
        if not subset:
            continue
        x = [int(record["n_basins"]) for record in subset]
        y = [float(record["overall_crossing"]) for record in subset]
        sizes = [80 + 900 * float(record["min_occupancy"]) for record in subset]
        ax.scatter(
            x,
            y,
            s=sizes,
            color=colors[status],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
            label=labels[status],
        )

    for record in frontier[:6]:
        x = int(record["n_basins"])
        y = float(record["overall_crossing"])
        label = str(record["name"])
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#1F2933",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.9,
            },
        )

    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Detected basin count")
    ax.set_ylabel("Overall crossing fraction")
    ax.set_title("Combined Screened Systems: Strict Core, Relaxed Accepts, and Retune Frontier")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)
    ax.text(
        0.01,
        0.03,
        "Green band = nominal target crossing range [0.30, 0.70]",
        transform=ax.transAxes,
        fontsize=9,
        color="#375A4A",
    )


def plot_summary_panel(
    ax: plt.Axes,
    registry_names: list[str],
    records: list[dict[str, object]],
    unscreened: list[str],
    backlog_counts: dict[str, int],
) -> None:
    strict_cross_pass_count = sum(record["status"] == "strict_cross_pass" for record in records)
    accepted_pass_count = sum(
        record["status"] in {"strict_cross_pass", "accepted_relaxed_pass"} for record in records
    )
    basin_occ_pass = sum(
        bool(record["basin_pass"]) and bool(record["occ_pass"]) for record in records
    )

    ax.axis("off")
    ax.set_title("Coverage Audit")
    lines = [
        ("registered in registry", str(len(registry_names))),
        ("covered by combined grounded screen", str(len(records))),
        ("not yet screened in saved artifact", str(len(unscreened))),
        ("screened systems passing basin+occupancy", str(basin_occ_pass)),
        ("accepted under current fast-screen gates", str(accepted_pass_count)),
        ("strict-crossing subset of accepted systems", str(strict_cross_pass_count)),
    ]
    y = 0.95
    for label, value in lines:
        ax.text(0.02, y, value, fontsize=20, fontweight="bold", color="#1D3557", va="top")
        ax.text(0.26, y, label, fontsize=10, color="#334155", va="top")
        y -= 0.095

    ax.text(0.02, 0.32, "Unscreened backlog mix", fontsize=11, fontweight="bold", color="#1F2933")
    bar_y = [0.24, 0.14, 0.04]
    labels = ["calibrated controls", "topology variants", "hybrid / other"]
    values = [
        backlog_counts["cal"],
        backlog_counts["var"],
        backlog_counts["hybrid_or_other"],
    ]
    colors = ["#457B9D", "#E9C46A", "#8D6CAB"]
    max_value = max(values) if values else 1
    for row_y, label, value, color in zip(bar_y, labels, values, colors, strict=True):
        ax.add_patch(plt.Rectangle((0.02, row_y - 0.02), 0.72 * value / max_value, 0.05, color=color, alpha=0.95))
        ax.text(0.02, row_y + 0.045, f"{label} ({value})", fontsize=9, color="#334155", va="center")
    ax.text(
        0.02,
        -0.015,
        "Accepted systems can still clear the fast screen via the relaxed crossing gate; only the green strict-core subset keeps every basin inside [0.30, 0.70].",
        fontsize=8.5,
        color="#475569",
        wrap=True,
    )


def draw_text_table(
    ax: plt.Axes,
    title: str,
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    col_x: tuple[float, ...],
) -> None:
    ax.axis("off")
    ax.set_title(title)
    y = 0.96
    for header, x in zip(headers, col_x, strict=True):
        ax.text(x, y, header, transform=ax.transAxes, fontsize=9, fontweight="bold", color="#1F2933", va="top")
    ax.plot([0.0, 1.0], [0.91, 0.91], transform=ax.transAxes, color="#CBD5E1", linewidth=1.0)
    y = 0.87
    line_height = 0.10 if len(rows) <= 8 else 0.085
    for row in rows:
        for item, x in zip(row, col_x, strict=True):
            ax.text(x, y, item, transform=ax.transAxes, fontsize=8.5, color="#334155", va="top")
        y -= line_height


def plot_catalog_audit(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    screening_path: Path = DEFAULT_SCREENING_PATH,
    extra_screening_paths: tuple[Path, ...] = (),
) -> list[Path]:
    apply_base_style()
    registry_names, registry = import_catalog_registry()
    records = load_combined_screening_records(
        screening_path=screening_path,
        extra_screening_paths=extra_screening_paths,
    )
    frontier = build_retune_frontier(records)
    unscreened, backlog_counts = build_unscreened_backlog(registry_names, records, registry)
    unscreened_set = set(unscreened)
    next_queue = [
        record for record in build_unscreened_priority_records() if record.name in unscreened_set
    ]

    fig = plt.figure(figsize=(17, 11))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.6, 1.0),
        height_ratios=(1.15, 1.0),
        wspace=0.20,
        hspace=0.24,
    )
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_summary = fig.add_subplot(grid[0, 1])
    ax_frontier = fig.add_subplot(grid[1, 0])
    ax_queue = fig.add_subplot(grid[1, 1])

    plot_status_scatter(ax_scatter, records, frontier)
    plot_summary_panel(ax_summary, registry_names, records, unscreened, backlog_counts)

    frontier_rows = []
    for record in frontier:
        frontier_rows.append(
            (
                str(record["name"]),
                f"{int(record['n_basins'])}",
                f"{float(record['overall_crossing']):.3f}",
                f"{float(record['min_occupancy']):.3f}",
                retune_note(record),
            )
        )
    draw_text_table(
        ax_frontier,
        title="Best Screened Systems So Far",
        headers=("system", "B", "C", "occ", "interpretation"),
        rows=frontier_rows,
        col_x=(0.00, 0.45, 0.54, 0.64, 0.75),
    )

    queue_rows = []
    for record in next_queue:
        queue_rows.append((record.name, record.bucket, record.rationale))
    draw_text_table(
        ax_queue,
        title="Recommended Next-Screen Queue From The Unscreened Backlog",
        headers=("system", "track", "why now"),
        rows=queue_rows,
        col_x=(0.00, 0.40, 0.56),
    )

    fig.suptitle(
        "Claude Catalog Audit: Combined Screening Coverage, Acceptance Tiers, and Remaining Queue",
        y=0.99,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in formats:
        path = output_dir / f"claude_catalog_audit_atlas.{fmt}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the audit figure into.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated list of formats to save (png,svg,pdf).",
    )
    parser.add_argument(
        "--screening_path",
        type=Path,
        default=DEFAULT_SCREENING_PATH,
        help="Path to screening_results.json.",
    )
    parser.add_argument(
        "--extra_screening_path",
        type=Path,
        action="append",
        default=[],
        help="Optional extra screening JSON packet(s) to merge into the audit.",
    )
    args = parser.parse_args()

    plot_catalog_audit(
        output_dir=args.output_dir,
        formats=parse_formats_arg(args.formats),
        screening_path=args.screening_path,
        extra_screening_paths=tuple(args.extra_screening_path),
    )


if __name__ == "__main__":
    main()
