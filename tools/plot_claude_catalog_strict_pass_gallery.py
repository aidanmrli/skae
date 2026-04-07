#!/usr/bin/env python3
"""Render strict-core and accepted-pass galleries for the grounded Claude catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMBINED_RESULTS = (
    REPO_ROOT
    / "results"
    / "claude_catalog_priority_screen_20260407"
    / "combined_screening_results.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures" / "claude_catalog_audit_20260407"
DEFAULT_PORTRAIT_DIR = (
    REPO_ROOT / "results" / "claude_catalog_priority_screen_20260407" / "plots"
)
LEGACY_PORTRAIT_DIR = REPO_ROOT / "results" / "claude_catalog_validation" / "plots"
DEFAULT_FORMATS = ("png", "svg", "pdf")


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


def per_basin_crossing_values(row: dict[str, object]) -> list[float]:
    per_basin = row.get("per_basin_crossing") or {}
    return [float(value) for value in per_basin.values()]


def strict_crossing_pass(row: dict[str, object], lower: float = 0.30, upper: float = 0.70) -> bool:
    values = per_basin_crossing_values(row)
    return bool(values) and all(lower <= value <= upper for value in values)


def load_strict_pass_records(combined_results_path: Path = DEFAULT_COMBINED_RESULTS) -> list[dict[str, object]]:
    rows = json.loads(combined_results_path.read_text())
    strict = [
        row
        for row in rows
        if bool(row.get("all_pass")) and strict_crossing_pass(row)
    ]
    strict.sort(
        key=lambda row: (
            int(row["n_basins"]),
            str(row["category"]),
            -float(row["overall_crossing"]),
            str(row["name"]),
        )
    )
    return strict


def load_accepted_pass_records(combined_results_path: Path = DEFAULT_COMBINED_RESULTS) -> list[dict[str, object]]:
    rows = json.loads(combined_results_path.read_text())
    accepted = [row for row in rows if bool(row.get("all_pass"))]
    accepted.sort(
        key=lambda row: (
            int(row["n_basins"]),
            0 if strict_crossing_pass(row) else 1,
            str(row["category"]),
            -float(row["overall_crossing"]),
            str(row["name"]),
        )
    )
    return accepted


def ensure_portrait(record: dict[str, object], portrait_dir: Path) -> Path:
    """Return a portrait path, generating it if it does not already exist."""

    name = str(record["name"])
    portrait_dir.mkdir(parents=True, exist_ok=True)
    target = portrait_dir / f"{name}.png"
    if target.exists():
        return target

    legacy = LEGACY_PORTRAIT_DIR / f"{name}.png"
    if legacy.exists():
        return legacy

    from tools.fast_screen_catalog import get_system, plot_system, screen_system

    result, trajectories, assignments, centers = screen_system(
        name,
        n_traj=100,
        traj_len=500,
        extra_steps=2000,
        seed=42,
    )
    system = get_system(name)
    plot_system(system, trajectories, assignments, centers, result["n_basins"], str(target))
    return target


def plot_pass_gallery(
    records: list[dict[str, object]],
    title: str,
    subtitle: str,
    output_stem: str,
    output_dir: Path,
    portrait_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    portrait_paths = [ensure_portrait(record, portrait_dir) for record in records]

    plt.style.use("seaborn-v0_8-white")
    ncols = 3
    nrows = (len(records) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5.2 * nrows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, record, portrait_path in zip(axes_list, records, portrait_paths):
        image = mpimg.imread(portrait_path)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        is_strict = strict_crossing_pass(record)
        border_color = "#2A9D8F" if is_strict else "#3B82F6"
        badge_text = "strict" if is_strict else "relaxed"
        badge_face = "#2A9D8F" if is_strict else "#3B82F6"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(3.0)
        ax.set_title(
            f"{record['name']}\n"
            f"B={record['n_basins']}  C={float(record['overall_crossing']):.3f}  "
            f"occ={float(record['min_occupancy']):.3f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            0.02,
            0.98,
            badge_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": badge_face,
                "edgecolor": "white",
                "linewidth": 0.8,
                "alpha": 0.95,
            },
        )

    for ax in axes_list[len(records):]:
        ax.set_axis_off()

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.985,
        subtitle,
        ha="center",
        va="top",
        fontsize=11,
        color="#475569",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for fmt in formats:
        path = output_dir / f"{output_stem}.{fmt}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_strict_pass_gallery(
    combined_results_path: Path = DEFAULT_COMBINED_RESULTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    portrait_dir: Path = DEFAULT_PORTRAIT_DIR,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    records = load_strict_pass_records(combined_results_path)
    return plot_pass_gallery(
        records=records,
        title="Current Strict-Crossing Claude Catalog Systems",
        subtitle="Combined grounded fast screen: 8 strict-crossing passes across 12 accepted systems and 83 screened systems",
        output_stem="claude_catalog_strict_pass_gallery",
        output_dir=output_dir,
        portrait_dir=portrait_dir,
        formats=formats,
    )


def plot_accepted_pass_gallery(
    combined_results_path: Path = DEFAULT_COMBINED_RESULTS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    portrait_dir: Path = DEFAULT_PORTRAIT_DIR,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> list[Path]:
    records = load_accepted_pass_records(combined_results_path)
    return plot_pass_gallery(
        records=records,
        title="Current Accepted Claude Catalog Systems",
        subtitle="Combined grounded fast screen: 12 accepted systems, with an 8-system strict-crossing core and 4 relaxed-crossing passes",
        output_stem="claude_catalog_accepted_pass_gallery",
        output_dir=output_dir,
        portrait_dir=portrait_dir,
        formats=formats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined_results_path",
        type=Path,
        default=DEFAULT_COMBINED_RESULTS,
        help="Combined screening JSON containing the current accepted and strict-core systems.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the pass galleries into.",
    )
    parser.add_argument(
        "--portrait_dir",
        type=Path,
        default=DEFAULT_PORTRAIT_DIR,
        help="Directory for any newly generated portrait PNGs.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=",".join(DEFAULT_FORMATS),
        help="Comma-separated list of formats to save.",
    )
    args = parser.parse_args()

    plot_strict_pass_gallery(
        combined_results_path=args.combined_results_path,
        output_dir=args.output_dir,
        portrait_dir=args.portrait_dir,
        formats=parse_formats_arg(args.formats),
    )
    plot_accepted_pass_gallery(
        combined_results_path=args.combined_results_path,
        output_dir=args.output_dir,
        portrait_dir=args.portrait_dir,
        formats=parse_formats_arg(args.formats),
    )


if __name__ == "__main__":
    main()
