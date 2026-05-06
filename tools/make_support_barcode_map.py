#!/usr/bin/env python3
"""Build the NeurIPS support barcode map figure.

This script is intentionally separate from ``make_neurips_visual_drafts.py`` so
parallel figure iterations do not rewrite the alluvial/counterfactual drafts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.make_neurips_visual_drafts import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SYSTEM,
    SystemBundle,
    _basin_colors,
    _draw_streamlines,
    _family_dominant_basins,
    _load_bundle,
)
from tools.make_benchmark_support_dysts_composite import _topk_support_mask
from tools.make_support_family_index_codebook import _support_family_labels_and_prototypes


DEFAULT_INTERPRETABILITY_CSV = Path(
    "results/transition_rich_lista_dense_p256_hardinit_table123_20260430/"
    "interpretability_pass0/interpretability_rows.csv"
)
DEFAULT_ROOT_LABEL = "lista_dense_signsplit_p256_hardinit_basin_partition"
DEFAULT_OUTPUT_STEM = "fig_support_barcode_map_p256"


def _figure_setup() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _displayed_prototype_families(
    families: np.ndarray,
    basin_labels: np.ndarray,
    basin_count: int,
) -> tuple[list[int], dict[int, int], Counter[int]]:
    family_to_basin = _family_dominant_basins(families, basin_labels)
    family_counts = Counter(int(item) for item in families.tolist())
    displayed: list[int] = []
    for basin in range(basin_count):
        candidates = [
            (family, count)
            for family, count in family_counts.items()
            if family_to_basin.get(int(family), -1) == basin
        ]
        if candidates:
            displayed.append(int(max(candidates, key=lambda item: item[1])[0]))
    displayed = sorted(set(displayed), key=lambda fam: (family_to_basin[fam], -family_counts[fam]))
    return displayed, family_to_basin, family_counts


def _draw_composite_barcode(
    ax,
    prototypes: np.ndarray,
    displayed: list[int],
    family_to_basin: dict[int, int],
    family_counts: Counter[int],
    colors: list[str],
    total_count: int,
) -> list[dict[str, Any]]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    if not displayed:
        raise RuntimeError("No prototype support families were selected for display")
    latent_dim = int(np.asarray(prototypes[displayed[0]]).size)
    ax.set_xlim(-0.5, latent_dim - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("latent coordinate")
    ax.set_title("Composite prototype support", loc="left", fontweight="bold", pad=4)

    tick_step = 8 if latent_dim <= 96 else 32 if latent_dim <= 320 else 64
    for x in range(latent_dim):
        if x % tick_step == 0:
            ax.axvline(x - 0.5, color="#edf0f2", lw=0.45, zorder=0)

    track_y = 0.34
    track_height = 0.32
    metadata: list[dict[str, Any]] = []
    handles: list[Any] = []

    ax.add_patch(
        Rectangle(
            (-0.5, track_y),
            latent_dim,
            track_height,
            facecolor="#f8f9fa",
            edgecolor="#aeb4bd",
            linewidth=0.6,
            zorder=1,
        )
    )

    active_by_coord: dict[int, list[tuple[int, int, str]]] = {coord: [] for coord in range(latent_dim)}
    for idx, family in enumerate(displayed):
        basin = int(family_to_basin[family])
        color = colors[basin % len(colors)]
        active = np.flatnonzero(prototypes[family].astype(bool)).astype(int)
        for coord in active.tolist():
            active_by_coord[coord].append((idx, int(family), color))
        mass = 100.0 * family_counts[family] / max(1, total_count)
        label = f"B{basin} F{family} ({mass:.0f}%)"
        handles.append(Line2D([0], [0], color=color, lw=4.0, label=label))
        metadata.append(
            {
                "family": int(family),
                "dominant_basin": basin,
                "grid_fraction": float(family_counts[family] / max(1, total_count)),
                "prototype_indices": active.tolist(),
            }
        )

    for coord, active_items in active_by_coord.items():
        if not active_items:
            continue
        active_items = sorted(active_items, key=lambda item: item[0])
        segment_height = track_height / len(active_items)
        for segment, (_idx, _family, color) in enumerate(active_items):
            ax.add_patch(
                Rectangle(
                    (coord - 0.37, track_y + segment * segment_height),
                    0.74,
                    segment_height,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.88,
                    zorder=3,
                )
            )
        if len(active_items) > 1:
            ax.plot(
                [coord - 0.37, coord + 0.37],
                [track_y + track_height / 2.0, track_y + track_height / 2.0],
                color="white",
                lw=0.35,
                solid_capstyle="butt",
                zorder=4,
            )

    overlap_count = sum(1 for active_items in active_by_coord.values() if len(active_items) > 1)
    if overlap_count:
        ax.text(
            0.0,
            0.73,
            f"{overlap_count} shared coordinates split by color",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.7,
            color="#4b5563",
        )
    ax.set_xticks(np.arange(0, latent_dim, tick_step))
    ax.tick_params(axis="x", length=2.5, pad=2)
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        handlelength=1.1,
        borderaxespad=0.0,
        labelspacing=0.35,
    )
    return metadata


def _support_barcode_map_v2(bundle: SystemBundle, args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    basin_count = int(bundle.basin_labels.max()) + 1
    colors = _basin_colors(basin_count)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, basin_count + 0.5, 1), basin_count)

    support_mask = _topk_support_mask(bundle.latents, args.topk)
    families, prototypes = _support_family_labels_and_prototypes(
        support_mask,
        min_jaccard=args.family_jaccard,
    )
    displayed, family_to_basin, family_counts = _displayed_prototype_families(
        families,
        bundle.basin_labels,
        basin_count,
    )
    support_basin = np.asarray([family_to_basin[int(item)] for item in families], dtype=int)

    _figure_setup()
    fig = plt.figure(figsize=(6.35, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.13, 0.87], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_code = fig.add_subplot(gs[0, 1])

    basin_grid = bundle.basin_labels.reshape(args.grid_points, args.grid_points)
    ax.pcolormesh(
        bundle.xx,
        bundle.yy,
        basin_grid,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        alpha=0.11,
        rasterized=True,
    )
    ax.scatter(
        bundle.states[:, 0].numpy(),
        bundle.states[:, 1].numpy(),
        c=support_basin,
        cmap=cmap,
        norm=norm,
        s=2.0,
        alpha=0.62,
        linewidths=0,
        rasterized=True,
    )
    _draw_streamlines(ax, bundle, vector_points=28)

    ax.set_title("State-space support-family map", loc="left", fontweight="bold", pad=4)
    ax.set_xlim(bundle.xlim)
    ax.set_ylim(bundle.ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    displayed_metadata = _draw_composite_barcode(
        ax_code,
        prototypes,
        displayed,
        family_to_basin,
        family_counts,
        colors,
        families.size,
    )
    for spine in ("left", "right", "top"):
        ax_code.spines[spine].set_visible(False)
    ax_code.spines["bottom"].set_linewidth(0.6)

    fig.subplots_adjust(left=0.025, right=0.995, top=0.90, bottom=0.18)

    output_pdf = args.output_dir / f"{args.output_stem}.pdf"
    output_png = output_pdf.with_suffix(".png")
    output_json = output_pdf.with_suffix(".json")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": "support_barcode_map_p256",
        "output_pdf": str(output_pdf),
        "output_png": str(output_png),
        "support_definition": f"topk:{args.topk}",
        "family_jaccard": float(args.family_jaccard),
        "latent_dimension": int(support_mask.shape[1]),
        "support_family_count": int(len(family_counts)),
        "dominant_basin_agreement_on_grid": float(np.mean(support_basin == bundle.basin_labels)),
        "label_note": (
            "Basin colors and family-to-basin names are post-hoc evaluation mappings; "
            "training uses no basin labels."
        ),
        "displayed_families": displayed_metadata,
    }
    output_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpretability-csv", type=Path, default=DEFAULT_INTERPRETABILITY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root-label", default=DEFAULT_ROOT_LABEL)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-points", type=int, default=96)
    parser.add_argument("--endpoint-rollout-steps", type=int, default=5000)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--family-jaccard", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_bundle(args)
    figure_metadata = _support_barcode_map_v2(bundle, args)
    metadata = {
        "system": bundle.system,
        "title": bundle.title,
        "root_label": args.root_label,
        "model_family": "p256 dense LISTA" if args.root_label == DEFAULT_ROOT_LABEL else args.root_label,
        "seed": int(args.seed),
        "run_dir": str(bundle.run_dir),
        "p256_proof": {
            "root_label_contains_p256": "p256" in args.root_label,
            "root_label": args.root_label,
            "run_dir_contains_p256": "p256" in str(bundle.run_dir),
        },
        "grid_points": int(args.grid_points),
        "basin_label_source": bundle.basin_label_source,
        "figure": figure_metadata,
    }
    Path(figure_metadata["output_pdf"]).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
