#!/usr/bin/env python3
"""Build the NeurIPS basin -> exact support -> support-family alluvial figure.

This standalone generator intentionally does not modify the broader visual-draft
script because the support barcode map is being iterated independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable

import numpy as np

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    get_transition_rich_basin_count,
)
from tools.make_benchmark_support_dysts_composite import (
    OKABE_ITO,
    _find_run_dir,
    _load_model_and_env,
)
from tools.make_support_family_index_codebook import _support_family_labels_and_prototypes
from tools.reduce_transition_rich_interpretability_metrics import (
    _encode_trajectories,
    _generate_observation_trajectories,
    _label_sequences_and_centers,
    _margin_subsets,
    _support_mask,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERPRETABILITY_CSV = (
    ROOT
    / "results"
    / "transition_rich_lista_dense_p256_hardinit_table123_20260430"
    / "interpretability_pass0"
    / "interpretability_rows.csv"
)
DEFAULT_INTERPRETABILITY_CSVS = (
    DEFAULT_INTERPRETABILITY_CSV,
    ROOT
    / "results"
    / "transition_rich_table2_5model_seed15_backfill_20260428"
    / "interpretability_pass0"
    / "interpretability_rows.csv",
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "figures"
    / "neurips_paper_2026"
    / "fig_basin_support_family_alluvial_p256_deep.pdf"
)
DEFAULT_METADATA_OUTPUT = DEFAULT_OUTPUT.with_suffix(".json")
DEFAULT_SUPPORT_SCHEME = "absolute"
DEFAULT_SUPPORT_VALUE = 1e-3
DEFAULT_EXCLUDED_SYSTEMS = (
    "multiwell_strong_transition",
    "claude:checkerboard_potential",
)

ROOT_DISPLAY = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": "p256 LISTA",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "LISTA-BD",
}

ALLOWED_P256_ROOTS = {
    "lista_dense_signsplit_p256_hardinit_basin_partition",
    "lista_blockdiag_signsplit_hardinit_basin_partition",
}


@dataclass(frozen=True)
class Candidate:
    root_label: str
    system: str
    seed: int
    run_dir: Path
    subset: str
    source_csv: str
    basin_count: int
    csv_family_count: int
    csv_exact_count: int
    csv_family_nmi: float
    csv_family_u: float
    csv_h_basin_given_family: float
    csv_h_family_given_basin: float


@dataclass(frozen=True)
class AlluvialData:
    candidate: Candidate
    label_source: str
    basin_labels: np.ndarray
    families: np.ndarray
    support_keys: list[tuple[int, ...]]
    support_masks: dict[tuple[int, ...], np.ndarray]
    family_prototypes: list[np.ndarray]
    family_to_basin: dict[int, int]
    family_purity: dict[int, float]
    family_agreement: float
    latent_dim: int
    represented_basin_count: int
    represented_basins: tuple[int, ...]
    exact_support_count: int
    family_count: int
    selected_state_count: int


def _safe_float(value: str, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source_csv"] = str(path)
                rows.append(dict(row))
    return rows


def _support_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.flatnonzero(mask).tolist())


def _candidate_root_allowed(root_label: str) -> bool:
    return root_label in ALLOWED_P256_ROOTS


def _screen_candidates(
    rows: list[dict[str, str]],
    *,
    roots: set[str] | None,
    systems: set[str] | None,
    seeds: set[int] | None,
    excluded_systems: set[str],
    support_scheme: str,
    support_value: float,
    subset: str,
) -> list[Candidate]:
    support_name = f"{support_scheme}:{support_value:g}"
    by_key: dict[tuple[str, str, int], Candidate] = {}
    for row in rows:
        root_label = row.get("root_label", "")
        system = row.get("system_key", "")
        seed = _safe_int(row.get("seed", ""))
        if not _candidate_root_allowed(root_label):
            continue
        if "p64" in root_label:
            continue
        if roots is not None and root_label not in roots:
            continue
        if systems is not None and system not in systems:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if system in excluded_systems:
            continue
        if row.get("support_scheme") != support_name or row.get("subset") != subset:
            continue
        basin_count = int(get_transition_rich_basin_count(system))
        if basin_count < 2:
            continue
        run_dir = Path(row.get("run_dir", ""))
        if not run_dir:
            continue
        candidate = Candidate(
            root_label=root_label,
            system=system,
            seed=seed,
            run_dir=run_dir,
            subset=str(row.get("subset", "")),
            source_csv=str(row.get("_source_csv", "")),
            basin_count=basin_count,
            csv_family_count=_safe_int(row.get("family_unique_count", "")),
            csv_exact_count=_safe_int(row.get("unique_support_count", "")),
            csv_family_nmi=_safe_float(row.get("family_nmi", "")),
            csv_family_u=_safe_float(row.get("family_u", "")),
            csv_h_basin_given_family=_safe_float(row.get("family_h_basin_given_family", "")),
            csv_h_family_given_basin=_safe_float(row.get("family_h_family_given_basin", "")),
        )
        by_key[(candidate.root_label, candidate.system, candidate.seed)] = candidate

    candidates = list(by_key.values())

    def sort_key(item: Candidate) -> tuple[int, float, float, int, str, int]:
        count_gap = abs(item.csv_family_count - item.basin_count)
        return (
            count_gap,
            -item.csv_family_nmi,
            -item.csv_family_u,
            item.csv_exact_count,
            item.system,
            item.seed,
        )

    return sorted(candidates, key=sort_key)


def _dominant_basins(
    families: np.ndarray,
    basin_labels: np.ndarray,
) -> tuple[dict[int, int], dict[int, float]]:
    family_to_basin: dict[int, int] = {}
    purity: dict[int, float] = {}
    for family in sorted({int(item) for item in families.tolist()}):
        keep = families == family
        counts = Counter(int(item) for item in basin_labels[keep].tolist() if int(item) >= 0)
        if not counts:
            family_to_basin[family] = -1
            purity[family] = 0.0
            continue
        basin, count = counts.most_common(1)[0]
        family_to_basin[family] = int(basin)
        purity[family] = float(count / max(1, int(keep.sum())))
    return family_to_basin, purity


def _collect_alluvial_data(
    candidate: Candidate,
    *,
    rows: list[dict[str, str]],
    device: str,
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
    endpoint_rollout_steps: int,
    support_scheme: str,
    support_value: float,
    family_jaccard: float,
    subset: str,
    depth_slice_mode: str,
) -> AlluvialData:
    run_dir = candidate.run_dir
    if not (run_dir / "checkpoint.pt").exists():
        run_dir = _find_run_dir(
            rows,
            root_label=candidate.root_label,
            system=candidate.system,
            seed=candidate.seed,
        )
    model, env = _load_model_and_env(run_dir, candidate.system, device)
    trajectories = _generate_observation_trajectories(
        env,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        eval_seed=eval_seed,
    )
    basin_labels_t, _centers, label_source = _label_sequences_and_centers(
        env,
        trajectories,
        system_key=candidate.system,
        endpoint_rollout_steps=endpoint_rollout_steps,
    )
    latents = _encode_trajectories(model, trajectories, device)
    support_mask = _support_mask(latents, scheme=support_scheme, value=support_value)
    flat_support_mask = support_mask.reshape(-1, support_mask.shape[-1])
    flat_basin_labels = basin_labels_t.detach().cpu().numpy().reshape(-1)
    if subset == "all":
        subset_mask = np.ones_like(flat_basin_labels, dtype=bool)
    else:
        subsets = _margin_subsets(
            trajectories,
            _centers,
            basin_labels=basin_labels_t.detach().cpu().numpy(),
            depth_slice_mode=depth_slice_mode,
        )
        if subset not in subsets:
            raise ValueError(f"Unknown subset {subset!r}; available subsets are {sorted(subsets)}")
        subset_mask = subsets[subset]
    keep = (flat_basin_labels >= 0) & subset_mask
    flat_support_mask = flat_support_mask[keep]
    flat_basin_labels = flat_basin_labels[keep]
    families, prototypes = _support_family_labels_and_prototypes(
        flat_support_mask,
        min_jaccard=family_jaccard,
    )
    support_keys = [_support_key(mask) for mask in flat_support_mask]
    support_masks: dict[tuple[int, ...], np.ndarray] = {}
    for key, mask in zip(support_keys, flat_support_mask):
        support_masks.setdefault(key, mask.astype(bool, copy=True))
    family_to_basin, family_purity = _dominant_basins(families, flat_basin_labels)
    mapped = np.asarray([family_to_basin[int(item)] for item in families.tolist()], dtype=int)
    represented_basins = tuple(sorted({int(item) for item in flat_basin_labels.tolist() if int(item) >= 0}))
    return AlluvialData(
        candidate=candidate,
        label_source=label_source,
        basin_labels=flat_basin_labels.astype(int, copy=False),
        families=families.astype(int, copy=False),
        support_keys=support_keys,
        support_masks=support_masks,
        family_prototypes=prototypes,
        family_to_basin=family_to_basin,
        family_purity=family_purity,
        family_agreement=float(np.mean(mapped == flat_basin_labels)),
        latent_dim=int(flat_support_mask.shape[-1]),
        represented_basin_count=int(len(represented_basins)),
        represented_basins=represented_basins,
        exact_support_count=int(len(set(support_keys))),
        family_count=int(len(set(families.tolist()))),
        selected_state_count=int(flat_basin_labels.size),
    )


def _figure_setup() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.4,
            "font.weight": "normal",
            "axes.titlesize": 7.4,
            "axes.titleweight": "normal",
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _stack_positions(
    counts: dict[Hashable, int],
    order: list[Hashable],
    *,
    y0: float,
    y1: float,
    gap: float,
) -> dict[Hashable, tuple[float, float]]:
    total = max(1, sum(int(counts[item]) for item in order))
    available = max(0.001, (y1 - y0) - gap * max(0, len(order) - 1))
    cursor = y1
    out: dict[Hashable, tuple[float, float]] = {}
    for item in order:
        height = available * int(counts[item]) / total
        out[item] = (cursor - height, cursor)
        cursor -= height + gap
    return out


def _draw_ribbon(
    ax: Any,
    x0: float,
    ya: tuple[float, float],
    x1: float,
    yb: tuple[float, float],
    *,
    color: str,
    alpha: float,
) -> None:
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    y0a, y1a = ya
    y0b, y1b = yb
    dx = 0.20 * (x1 - x0)
    verts = [
        (x0, y0a),
        (x0 + dx, y0a),
        (x1 - dx, y0b),
        (x1, y0b),
        (x1, y1b),
        (x1 - dx, y1b),
        (x0 + dx, y1a),
        (x0, y1a),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def _allocate_flows(
    source_pos: dict[Hashable, tuple[float, float]],
    dest_pos: dict[Hashable, tuple[float, float]],
    flows: Counter[tuple[Hashable, Hashable]],
    *,
    source_order: list[Hashable],
    dest_order: list[Hashable],
) -> list[tuple[Hashable, Hashable, tuple[float, float], tuple[float, float], int]]:
    source_rank = {item: rank for rank, item in enumerate(source_order)}
    dest_rank = {item: rank for rank, item in enumerate(dest_order)}
    source_cursor = {key: source_pos[key][0] for key in source_pos}
    dest_cursor = {key: dest_pos[key][0] for key in dest_pos}
    source_total: dict[Hashable, int] = defaultdict(int)
    dest_total: dict[Hashable, int] = defaultdict(int)
    for (source, dest), count in flows.items():
        source_total[source] += int(count)
        dest_total[dest] += int(count)
    ordered = sorted(
        flows,
        key=lambda item: (
            source_rank.get(item[0], 10**9),
            dest_rank.get(item[1], 10**9),
        ),
    )
    out = []
    for source, dest in ordered:
        count = int(flows[(source, dest)])
        s0, s1 = source_pos[source]
        d0, d1 = dest_pos[dest]
        hs = (s1 - s0) * count / max(1, source_total[source])
        hd = (d1 - d0) * count / max(1, dest_total[dest])
        ya = (source_cursor[source], source_cursor[source] + hs)
        yb = (dest_cursor[dest], dest_cursor[dest] + hd)
        source_cursor[source] += hs
        dest_cursor[dest] += hd
        out.append((source, dest, ya, yb, count))
    return out


def _draw_block(
    ax: Any,
    x: float,
    y0: float,
    y1: float,
    *,
    width: float,
    color: str,
    edgecolor: str = "#202124",
    alpha: float = 0.92,
    linewidth: float = 0.28,
) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (x - width / 2, y0),
            width,
            y1 - y0,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=3,
        )
    )


def _draw_alluvial(data: AlluvialData, output: Path, metadata_output: Path, args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _figure_setup()
    output.parent.mkdir(parents=True, exist_ok=True)
    subset_display = {
        "all": "all-state slice",
        "deep": "deep-state slice",
        "boundary": "boundary-state slice",
    }.get(str(args.subset), f"{args.subset} slice")
    if args.visual_style == "classic":
        layout = {
            "figsize": (4.55, 2.05),
            "stack_y0": 0.09,
            "stack_y1": 0.89,
            "major_gap": 0.016,
            "rail_xs": (),
            "basin_x": 0.14,
            "support_x": 0.50,
            "family_x": 0.86,
            "left_start_x": 0.18,
            "left_end_x": 0.49,
            "right_start_x": 0.51,
            "right_end_x": 0.82,
            "block_width": 0.050,
            "support_width": 0.030,
            "left_alpha": 0.18,
            "right_alpha": 0.24,
            "support_alpha": 0.75,
            "block_alpha": 0.92,
            "block_linewidth": 0.28,
            "block_label_weight": "bold",
            "header_y": 0.965,
            "header_size": 7.0,
            "caption_y": 0.035,
            "caption_size": 5.7,
            "pad_inches": 0.02,
        }
    else:
        layout = {
            "figsize": (4.32, 1.58),
            "stack_y0": 0.115,
            "stack_y1": 0.865,
            "major_gap": 0.022,
            "rail_xs": (0.20, 0.50, 0.80),
            "basin_x": 0.145,
            "support_x": 0.50,
            "family_x": 0.855,
            "left_start_x": 0.185,
            "left_end_x": 0.492,
            "right_start_x": 0.508,
            "right_end_x": 0.815,
            "block_width": 0.048,
            "support_width": 0.026,
            "left_alpha": 0.15,
            "right_alpha": 0.19,
            "support_alpha": 0.62,
            "block_alpha": 0.98,
            "block_linewidth": 0.32,
            "block_label_weight": "normal",
            "header_y": 0.935,
            "header_size": 7.5,
            "caption_y": 0.044,
            "caption_size": 5.9,
            "pad_inches": 0.012,
        }

    basin_counts = Counter(int(item) for item in data.basin_labels.tolist())
    support_counts = Counter(data.support_keys)
    family_counts = Counter(int(item) for item in data.families.tolist())
    support_to_family: dict[tuple[int, ...], int] = {}
    support_to_dominant_basin: dict[tuple[int, ...], int] = {}
    support_to_family_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    support_to_basin_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for key, family, basin in zip(data.support_keys, data.families.tolist(), data.basin_labels.tolist()):
        support_to_family_counts[key][int(family)] += 1
        support_to_basin_counts[key][int(basin)] += 1
    for key in support_counts:
        support_to_family[key] = support_to_family_counts[key].most_common(1)[0][0]
        support_to_dominant_basin[key] = support_to_basin_counts[key].most_common(1)[0][0]

    flow_basin_support: Counter[tuple[Hashable, Hashable]] = Counter()
    flow_support_family: Counter[tuple[Hashable, Hashable]] = Counter()
    for basin, key, family in zip(data.basin_labels.tolist(), data.support_keys, data.families.tolist()):
        flow_basin_support[(int(basin), key)] += 1
        flow_support_family[(key, int(family))] += 1

    basin_order = sorted(basin_counts)
    family_order = sorted(
        family_counts,
        key=lambda family: (
            data.family_to_basin.get(int(family), 10**9),
            -family_counts[int(family)],
            int(family),
        ),
    )
    family_rank = {family: rank for rank, family in enumerate(family_order)}
    support_order = sorted(
        support_counts,
        key=lambda key: (
            data.family_to_basin.get(support_to_family[key], 10**9),
            family_rank.get(support_to_family[key], 10**9),
            support_to_dominant_basin[key],
            -support_counts[key],
            key,
        ),
    )

    stack_y0 = float(layout["stack_y0"])
    stack_y1 = float(layout["stack_y1"])
    major_gap = float(layout["major_gap"])
    basin_pos = _stack_positions(dict(basin_counts), basin_order, y0=stack_y0, y1=stack_y1, gap=major_gap)
    support_pos = _stack_positions(dict(support_counts), support_order, y0=stack_y0, y1=stack_y1, gap=0.0)
    family_pos = _stack_positions(dict(family_counts), family_order, y0=stack_y0, y1=stack_y1, gap=major_gap)

    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(max(data.candidate.basin_count, 1))]
    basin_color = {basin: colors[rank % len(colors)] for rank, basin in enumerate(basin_order)}
    family_color = {
        family: basin_color.get(data.family_to_basin.get(family, -1), "#8a8f98")
        for family in family_order
    }
    support_color = {
        key: family_color.get(support_to_family[key], "#8a8f98")
        for key in support_order
    }

    fig, ax = plt.subplots(figsize=layout["figsize"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Subtle column rails make the white gutters intentional at manuscript scale.
    for x in layout["rail_xs"]:
        ax.plot([x, x], [stack_y0 - 0.008, stack_y1 + 0.008], color="#eef1f4", lw=0.35, zorder=0)

    left_alloc = _allocate_flows(
        basin_pos,
        support_pos,
        flow_basin_support,
        source_order=basin_order,
        dest_order=support_order,
    )
    right_alloc = _allocate_flows(
        support_pos,
        family_pos,
        flow_support_family,
        source_order=support_order,
        dest_order=family_order,
    )
    for source, _dest, ya, yb, _count in left_alloc:
        _draw_ribbon(
            ax,
            layout["left_start_x"],
            ya,
            layout["left_end_x"],
            yb,
            color=basin_color[int(source)],
            alpha=float(layout["left_alpha"]),
        )
    for _source, dest, ya, yb, _count in right_alloc:
        _draw_ribbon(
            ax,
            layout["right_start_x"],
            ya,
            layout["right_end_x"],
            yb,
            color=family_color[int(dest)],
            alpha=float(layout["right_alpha"]),
        )

    for basin in basin_order:
        y0, y1 = basin_pos[basin]
        _draw_block(
            ax,
            layout["basin_x"],
            y0,
            y1,
            width=float(layout["block_width"]),
            color=basin_color[basin],
            edgecolor="#27303a",
            alpha=float(layout["block_alpha"]),
            linewidth=float(layout["block_linewidth"]),
        )
        if y1 - y0 >= 0.040:
            ax.text(
                layout["basin_x"],
                0.5 * (y0 + y1),
                f"B{basin}",
                ha="center",
                va="center",
                fontsize=6.6,
                fontweight=layout["block_label_weight"],
                color="white",
                zorder=4,
            )

    for key in support_order:
        y0, y1 = support_pos[key]
        _draw_block(
            ax,
            layout["support_x"],
            y0,
            y1,
            width=float(layout["support_width"]),
            color=support_color[key],
            edgecolor=support_color[key],
            alpha=float(layout["support_alpha"]),
            linewidth=0.08,
        )

    for family in family_order:
        y0, y1 = family_pos[family]
        _draw_block(
            ax,
            layout["family_x"],
            y0,
            y1,
            width=float(layout["block_width"]),
            color=family_color[family],
            edgecolor="#27303a",
            alpha=float(layout["block_alpha"]),
            linewidth=float(layout["block_linewidth"]),
        )
        label = f"F{family}"
        if y1 - y0 >= 0.040:
            ax.text(
                layout["family_x"],
                0.5 * (y0 + y1),
                label,
                ha="center",
                va="center",
                fontsize=6.6,
                fontweight=layout["block_label_weight"],
                color="white",
                zorder=4,
            )

    ax.text(
        layout["basin_x"],
        layout["header_y"],
        "Basin",
        ha="center",
        va="top",
        fontsize=float(layout["header_size"]),
        fontweight="normal",
        color="#111827",
    )
    ax.text(
        layout["support_x"],
        layout["header_y"],
        r"Exact $S_{\rm abs}$",
        ha="center",
        va="top",
        fontsize=float(layout["header_size"]),
        fontweight="normal",
        color="#111827",
    )
    ax.text(
        layout["family_x"],
        layout["header_y"],
        r"Family $F_{\rm abs}$",
        ha="center",
        va="top",
        fontsize=float(layout["header_size"]),
        fontweight="normal",
        color="#111827",
    )
    ax.text(
        0.50,
        layout["caption_y"],
        f"{data.exact_support_count} exact supports, merged to {data.family_count} families",
        ha="center",
        va="bottom",
        fontsize=float(layout["caption_size"]),
        color="#4b5563",
    )

    fig.savefig(output, bbox_inches="tight", pad_inches=float(layout["pad_inches"]))
    png = output.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", pad_inches=float(layout["pad_inches"]))
    plt.close(fig)

    family_summary = [
        {
            "family": int(family),
            "dominant_basin": int(data.family_to_basin.get(family, -1)),
            "count": int(family_counts[family]),
            "fraction": float(family_counts[family] / max(1, data.selected_state_count)),
            "purity": float(data.family_purity.get(family, 0.0)),
            "prototype_active_indices": [
                int(i)
                for i in np.flatnonzero(data.family_prototypes[int(family)]).tolist()
            ]
            if int(family) < len(data.family_prototypes)
            else [],
        }
        for family in family_order
    ]
    metadata = {
        "figure": output.stem,
        "output_pdf": str(output),
        "output_png": str(png),
        "selected_system": data.candidate.system,
        "root_label": data.candidate.root_label,
        "model_display": ROOT_DISPLAY.get(data.candidate.root_label, data.candidate.root_label),
        "seed": int(data.candidate.seed),
        "run_dir": str(data.candidate.run_dir),
        "source_csv": data.candidate.source_csv,
        "subset": data.candidate.subset,
        "depth_slice_mode": args.depth_slice_mode,
        "visual_style": args.visual_style,
        "support_rule": f"{args.support_scheme}:{args.support_value:g}",
        "support_rule_display": "S_abs",
        "family_rule_display": "F_abs",
        "family_jaccard": float(args.family_jaccard),
        "basin_count": int(data.candidate.basin_count),
        "represented_basin_count": int(data.represented_basin_count),
        "represented_basins": [int(item) for item in data.represented_basins],
        "support_family_count": int(data.family_count),
        "exact_support_count": int(data.exact_support_count),
        "latent_dim": int(data.latent_dim),
        "p256_proof": {
            "allowed_roots_only": sorted(ALLOWED_P256_ROOTS),
            "selected_root_is_allowed_p256_root": data.candidate.root_label in ALLOWED_P256_ROOTS,
            "selected_root_contains_p64": "p64" in data.candidate.root_label,
            "latent_dim_equals_256": int(data.latent_dim) == 256,
        },
        "selected_state_count": int(data.selected_state_count),
        "label_source": data.label_source,
        "num_trajectories": int(args.num_trajectories),
        "trajectory_length": int(args.trajectory_length),
        "eval_seed": int(args.eval_seed),
        "family_dominant_basin_agreement": float(data.family_agreement),
        "mean_family_purity": float(np.mean(list(data.family_purity.values()))) if data.family_purity else 0.0,
        "selection_reason": (
            "Selected because the manuscript-consistent absolute support rule "
            "produces one merged support family per represented evaluation basin "
            f"on the {subset_display}, while exact supports remain visible as "
            "unlabeled middle bars."
        ),
        "csv_screen_metrics": {
            "family_count": int(data.candidate.csv_family_count),
            "exact_support_count": int(data.candidate.csv_exact_count),
            "family_nmi": float(data.candidate.csv_family_nmi),
            "family_u": float(data.candidate.csv_family_u),
            "h_basin_given_family": float(data.candidate.csv_h_basin_given_family),
            "h_family_given_basin": float(data.candidate.csv_h_family_given_basin),
        },
        "families": family_summary,
    }
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _parse_csv_strings(raw: str) -> set[str] | None:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _parse_csv_ints(raw: str) -> set[int] | None:
    values = {int(item.strip()) for item in raw.split(",") if item.strip()}
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpretability-csv",
        type=Path,
        action="append",
        default=None,
        help="interpretability_rows.csv path; can be passed multiple times",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--roots", default="", help="optional comma-separated root_label filter")
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated seed filter")
    parser.add_argument("--excluded-systems", default=",".join(DEFAULT_EXCLUDED_SYSTEMS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--support-scheme", default=DEFAULT_SUPPORT_SCHEME, choices=["absolute"])
    parser.add_argument("--support-value", type=float, default=DEFAULT_SUPPORT_VALUE)
    parser.add_argument("--subset", default="deep", choices=["all", "deep", "boundary"])
    parser.add_argument("--depth-slice-mode", default="global", choices=["global", "per_basin"])
    parser.add_argument(
        "--visual-style",
        default="polished",
        choices=["polished", "classic"],
        help="classic keeps the original alluvial layout; polished uses the compact manuscript layout",
    )
    parser.add_argument("--family-jaccard", type=float, default=0.5)
    parser.add_argument("--num-trajectories", type=int, default=256)
    parser.add_argument("--trajectory-length", type=int, default=256)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--endpoint-rollout-steps", type=int, default=5000)
    parser.add_argument("--max-evaluated-candidates", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = args.interpretability_csv if args.interpretability_csv is not None else list(DEFAULT_INTERPRETABILITY_CSVS)
    rows = _read_rows(csv_paths)
    candidates = _screen_candidates(
        rows,
        roots=_parse_csv_strings(args.roots),
        systems=_parse_csv_strings(args.systems),
        seeds=_parse_csv_ints(args.seeds),
        excluded_systems=_parse_csv_strings(args.excluded_systems) or set(),
        support_scheme=args.support_scheme,
        support_value=args.support_value,
        subset=args.subset,
    )
    if not candidates:
        raise RuntimeError("No LISTA-family candidates found for the requested filters")

    evaluated: list[dict[str, Any]] = []
    best: AlluvialData | None = None
    for candidate in candidates[: args.max_evaluated_candidates]:
        print(
            "Evaluating "
            f"{candidate.system} / {ROOT_DISPLAY.get(candidate.root_label, candidate.root_label)} "
            f"seed {candidate.seed} "
            f"(CSV families {candidate.csv_family_count}, basins {candidate.basin_count})",
            flush=True,
        )
        data = _collect_alluvial_data(
            candidate,
            rows=rows,
            device=args.device,
            num_trajectories=args.num_trajectories,
            trajectory_length=args.trajectory_length,
            eval_seed=args.eval_seed,
            endpoint_rollout_steps=args.endpoint_rollout_steps,
            support_scheme=args.support_scheme,
            support_value=args.support_value,
            family_jaccard=args.family_jaccard,
            subset=args.subset,
            depth_slice_mode=args.depth_slice_mode,
        )
        evaluated.append(
            {
                "system": candidate.system,
                "root_label": candidate.root_label,
                "seed": candidate.seed,
                "basin_count": candidate.basin_count,
                "represented_basin_count": data.represented_basin_count,
                "support_family_count": data.family_count,
                "exact_support_count": data.exact_support_count,
                "family_dominant_basin_agreement": data.family_agreement,
            }
        )
        count_matches = data.family_count == candidate.basin_count
        if best is None:
            best = data
        elif count_matches and best.family_count != best.candidate.basin_count:
            best = data
        elif count_matches == (best.family_count == best.candidate.basin_count):
            score = (data.family_agreement, -data.exact_support_count)
            best_score = (best.family_agreement, -best.exact_support_count)
            if score > best_score:
                best = data
        if count_matches and data.family_agreement >= 0.98:
            break

    if best is None:
        raise RuntimeError("Candidate evaluation failed")
    metadata = _draw_alluvial(best, args.output, args.metadata_output, args)
    metadata["candidate_evaluations"] = evaluated
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
