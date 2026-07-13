#!/usr/bin/env python3
"""Prototype spatial variants of the basin -> support -> family alluvial.

These drafts intentionally reuse the same selected state-space grid points for
both the spatial panels and the alluvial counts. That makes the visual question
explicit: where are the audited points in state space, which exact supports do
they activate, and which support-family cluster do those supports merge into?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import torch

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    get_transition_rich_basin_count,
)
from tools.make_basin_support_family_alluvial import (
    ALLOWED_P256_ROOTS,
    DEFAULT_INTERPRETABILITY_CSVS,
    ROOT_DISPLAY,
    _allocate_flows,
    _draw_block,
    _draw_ribbon,
    _stack_positions,
)
from tools.make_benchmark_support_dysts_composite import (
    OKABE_ITO,
    SUPPORT_PANELS,
    _basin_labels_for_states,
    _dynamics_for_states,
    _encode_latents,
    _find_run_dir,
    _grid_states,
    _load_model_and_env,
    _read_rows,
)
from tools.make_support_family_index_codebook import _support_family_labels_and_prototypes
from tools.reduce_transition_rich_interpretability_metrics import _margin_subsets, _support_mask


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "figures" / "neurips_paper_2026" / "spatial_alluvial_prototypes"
DEFAULT_ROOT_LABEL = "lista_dense_signsplit_p256_hardinit_basin_partition"
DEFAULT_SYSTEM = "gated_local_linear"


@dataclass(frozen=True)
class SystemSpec:
    system: str
    title: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]


@dataclass(frozen=True)
class SpatialAlluvialData:
    system: str
    title: str
    root_label: str
    seed: int
    run_dir: Path
    label_source: str
    basin_label_source: str
    grid_points: int
    endpoint_rollout_steps: int
    support_scheme: str
    support_value: float
    family_jaccard: float
    subset: str
    depth_slice_mode: str
    xx: np.ndarray
    yy: np.ndarray
    states: torch.Tensor
    basin_labels_grid: np.ndarray
    selected_indices: np.ndarray
    selected_states: np.ndarray
    selected_basin_labels: np.ndarray
    selected_support_masks: np.ndarray
    support_keys: list[tuple[int, ...]]
    families: np.ndarray
    prototypes: list[np.ndarray]
    family_to_basin: dict[int, int]
    family_purity: dict[int, float]
    family_agreement: float
    latent_dim: int
    basin_count: int


def _figure_setup() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _support_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.flatnonzero(mask).tolist())


def _system_spec(system: str) -> SystemSpec:
    for spec in SUPPORT_PANELS:
        if spec.system == system:
            return SystemSpec(
                system=spec.system,
                title=spec.title,
                xlim=spec.xlim,
                ylim=spec.ylim,
            )
    known = ", ".join(sorted(spec.system for spec in SUPPORT_PANELS))
    raise ValueError(f"Unknown prototype system {system!r}; known support-panel systems: {known}")


def _read_many_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_rows(path))
    return rows


def _parse_csv_paths(paths: list[Path] | None) -> list[Path]:
    if paths is None:
        return [Path(path) for path in DEFAULT_INTERPRETABILITY_CSVS]
    return [Path(path) for path in paths]


def _centers_for_grid(env: Any, states: torch.Tensor, basin_labels: np.ndarray, basin_count: int) -> tuple[torch.Tensor, str]:
    for attr in ("points", "points_2d"):
        centers = getattr(env, attr, None)
        if isinstance(centers, torch.Tensor) and centers.ndim == 2 and centers.shape[0] >= basin_count:
            return centers.detach().cpu().to(dtype=states.dtype)[:basin_count, : states.shape[1]], f"env.{attr}"
        if centers is not None and not isinstance(centers, torch.Tensor):
            centers_t = torch.as_tensor(centers, dtype=states.dtype)
            if centers_t.ndim == 2 and centers_t.shape[0] >= basin_count:
                return centers_t.detach().cpu()[:basin_count, : states.shape[1]], f"env.{attr}"

    xy = states.detach().cpu().numpy()
    computed: list[np.ndarray] = []
    for basin in range(basin_count):
        keep = basin_labels == basin
        if bool(np.any(keep)):
            computed.append(xy[keep].mean(axis=0))
    if len(computed) >= 2:
        return torch.as_tensor(np.stack(computed, axis=0), dtype=states.dtype), "grid_label_centroids"
    raise RuntimeError("Could not infer at least two basin centers for deep/boundary subset selection")


def _dominant_basins(families: np.ndarray, basin_labels: np.ndarray) -> tuple[dict[int, int], dict[int, float]]:
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


def _load_data(args: argparse.Namespace) -> tuple[SpatialAlluvialData, Any]:
    if args.root_label not in ALLOWED_P256_ROOTS:
        raise ValueError(
            f"{args.root_label!r} is not in the p256-root allow-list used by the alluvial figure"
        )
    spec = _system_spec(args.system)
    rows = _read_many_rows(_parse_csv_paths(args.interpretability_csv))
    run_dir = _find_run_dir(rows, root_label=args.root_label, system=args.system, seed=args.seed)
    model, env = _load_model_and_env(run_dir, args.system, args.device)
    xx, yy, states = _grid_states(spec.xlim, spec.ylim, args.grid_points)
    basin_labels_t, basin_source = _basin_labels_for_states(
        env,
        args.system,
        states,
        endpoint_rollout_steps=args.endpoint_rollout_steps,
    )
    basin_labels = basin_labels_t.detach().cpu().numpy().astype(int, copy=False)
    basin_count = int(max(get_transition_rich_basin_count(args.system), int(basin_labels.max()) + 1))
    centers, center_source = _centers_for_grid(env, states, basin_labels, basin_count)

    latents = _encode_latents(model, states, args.device)
    support_masks = _support_mask(latents, scheme=args.support_scheme, value=args.support_value)

    subsets = _margin_subsets(
        states,
        centers,
        basin_labels=basin_labels,
        depth_slice_mode=args.depth_slice_mode,
    )
    if args.subset not in subsets:
        raise ValueError(f"Unknown subset {args.subset!r}; available subsets are {sorted(subsets)}")
    selected_mask = (basin_labels >= 0) & subsets[args.subset]
    selected_indices = np.flatnonzero(selected_mask)
    selected_support_masks = support_masks[selected_mask]
    selected_basin_labels = basin_labels[selected_mask]
    families, prototypes = _support_family_labels_and_prototypes(
        selected_support_masks,
        min_jaccard=args.family_jaccard,
    )
    support_keys = [_support_key(mask) for mask in selected_support_masks]
    family_to_basin, purity = _dominant_basins(families, selected_basin_labels)
    mapped = np.asarray([family_to_basin[int(item)] for item in families.tolist()], dtype=int)

    data = SpatialAlluvialData(
        system=args.system,
        title=spec.title,
        root_label=args.root_label,
        seed=int(args.seed),
        run_dir=run_dir,
        label_source=center_source,
        basin_label_source=basin_source,
        grid_points=int(args.grid_points),
        endpoint_rollout_steps=int(args.endpoint_rollout_steps),
        support_scheme=str(args.support_scheme),
        support_value=float(args.support_value),
        family_jaccard=float(args.family_jaccard),
        subset=str(args.subset),
        depth_slice_mode=str(args.depth_slice_mode),
        xx=xx,
        yy=yy,
        states=states,
        basin_labels_grid=basin_labels.reshape(args.grid_points, args.grid_points),
        selected_indices=selected_indices,
        selected_states=states[selected_mask].detach().cpu().numpy(),
        selected_basin_labels=selected_basin_labels,
        selected_support_masks=selected_support_masks,
        support_keys=support_keys,
        families=families.astype(int, copy=False),
        prototypes=prototypes,
        family_to_basin=family_to_basin,
        family_purity=purity,
        family_agreement=float(np.mean(mapped == selected_basin_labels)) if selected_basin_labels.size else 0.0,
        latent_dim=int(selected_support_masks.shape[-1]),
        basin_count=basin_count,
    )
    return data, env


def _colors(data: SpatialAlluvialData) -> list[str]:
    return [OKABE_ITO[idx % len(OKABE_ITO)] for idx in range(max(data.basin_count, 1))]


def _flow_components(data: SpatialAlluvialData) -> dict[str, Any]:
    basin_counts = Counter(int(item) for item in data.selected_basin_labels.tolist())
    support_counts = Counter(data.support_keys)
    family_counts = Counter(int(item) for item in data.families.tolist())

    support_to_family_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    support_to_basin_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    for support, family, basin in zip(data.support_keys, data.families.tolist(), data.selected_basin_labels.tolist()):
        support_to_family_counts[support][int(family)] += 1
        support_to_basin_counts[support][int(basin)] += 1
    support_to_family = {key: counts.most_common(1)[0][0] for key, counts in support_to_family_counts.items()}
    support_to_basin = {key: counts.most_common(1)[0][0] for key, counts in support_to_basin_counts.items()}

    flow_basin_support: Counter[tuple[Hashable, Hashable]] = Counter()
    flow_support_family: Counter[tuple[Hashable, Hashable]] = Counter()
    for basin, support, family in zip(data.selected_basin_labels.tolist(), data.support_keys, data.families.tolist()):
        flow_basin_support[(int(basin), support)] += 1
        flow_support_family[(support, int(family))] += 1

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
        key=lambda support: (
            data.family_to_basin.get(support_to_family[support], 10**9),
            family_rank.get(support_to_family[support], 10**9),
            support_to_basin[support],
            -support_counts[support],
            support,
        ),
    )

    colors = _colors(data)
    basin_color = {basin: colors[rank % len(colors)] for rank, basin in enumerate(basin_order)}
    family_color = {
        family: basin_color.get(data.family_to_basin.get(int(family), -1), "#8a8f98")
        for family in family_order
    }
    support_color = {
        support: family_color.get(support_to_family[support], "#8a8f98")
        for support in support_order
    }
    family_for_selected = np.asarray([int(item) for item in data.families.tolist()], dtype=int)
    selected_family_basin = np.asarray(
        [data.family_to_basin.get(int(family), -1) for family in family_for_selected],
        dtype=int,
    )
    return {
        "basin_counts": basin_counts,
        "support_counts": support_counts,
        "family_counts": family_counts,
        "support_to_family": support_to_family,
        "support_to_basin": support_to_basin,
        "flow_basin_support": flow_basin_support,
        "flow_support_family": flow_support_family,
        "basin_order": basin_order,
        "support_order": support_order,
        "family_order": family_order,
        "basin_color": basin_color,
        "support_color": support_color,
        "family_color": family_color,
        "selected_family_basin": selected_family_basin,
    }


def _draw_spatial_panel(
    ax: Any,
    data: SpatialAlluvialData,
    components: dict[str, Any],
    *,
    env: Any,
    title: str,
    point_size: float = 3.0,
    show_mismatch: bool = True,
) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = _colors(data)
    cmap = ListedColormap(colors[: data.basin_count])
    norm = BoundaryNorm(np.arange(-0.5, data.basin_count + 0.5, 1.0), data.basin_count)
    ax.pcolormesh(
        data.xx,
        data.yy,
        data.basin_labels_grid,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        alpha=0.11,
        rasterized=True,
    )
    mapped_basin = components["selected_family_basin"]
    ax.scatter(
        data.selected_states[:, 0],
        data.selected_states[:, 1],
        c=mapped_basin,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=0.68,
        linewidths=0,
        rasterized=True,
    )
    if show_mismatch:
        mismatch = mapped_basin != data.selected_basin_labels
        if bool(np.any(mismatch)):
            ax.scatter(
                data.selected_states[mismatch, 0],
                data.selected_states[mismatch, 1],
                c="#111111",
                s=max(point_size * 0.45, 0.8),
                alpha=0.48,
                linewidths=0,
                rasterized=True,
            )

    vx, vy, vstates = _grid_states(
        (float(data.xx.min()), float(data.xx.max())),
        (float(data.yy.min()), float(data.yy.max())),
        28,
    )
    velocity = _dynamics_for_states(env, vstates)
    u = velocity[:, 0].numpy().reshape(28, 28)
    v = velocity[:, 1].numpy().reshape(28, 28)
    speed = np.sqrt(u**2 + v**2)
    linewidth = 0.25 + 0.42 * speed / max(float(np.nanpercentile(speed, 95)), 1e-8)
    ax.streamplot(
        vx[0],
        vy[:, 0],
        u,
        v,
        color="#343941",
        linewidth=linewidth,
        density=1.0,
        arrowsize=0.50,
        zorder=4,
    )
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlim(float(data.xx.min()), float(data.xx.max()))
    ax.set_ylim(float(data.yy.min()), float(data.yy.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#29313a")
        spine.set_linewidth(0.62)


def _draw_alluvial(
    ax: Any,
    data: SpatialAlluvialData,
    components: dict[str, Any],
    *,
    y0: float = 0.11,
    y1: float = 0.86,
    basin_x: float = 0.12,
    support_x: float = 0.50,
    family_x: float = 0.88,
    left_start_x: float = 0.17,
    left_end_x: float = 0.49,
    right_start_x: float = 0.51,
    right_end_x: float = 0.83,
    show_basin_blocks: bool = True,
    show_caption: bool = True,
) -> dict[str, Any]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    basin_pos = _stack_positions(
        dict(components["basin_counts"]),
        components["basin_order"],
        y0=y0,
        y1=y1,
        gap=0.020,
    )
    support_pos = _stack_positions(
        dict(components["support_counts"]),
        components["support_order"],
        y0=y0,
        y1=y1,
        gap=0.0,
    )
    family_pos = _stack_positions(
        dict(components["family_counts"]),
        components["family_order"],
        y0=y0,
        y1=y1,
        gap=0.020,
    )

    for x in (basin_x + 0.06, support_x, family_x - 0.06):
        ax.plot([x, x], [y0 - 0.008, y1 + 0.008], color="#eef1f4", lw=0.34, zorder=0)

    left_alloc = _allocate_flows(
        basin_pos,
        support_pos,
        components["flow_basin_support"],
        source_order=components["basin_order"],
        dest_order=components["support_order"],
    )
    right_alloc = _allocate_flows(
        support_pos,
        family_pos,
        components["flow_support_family"],
        source_order=components["support_order"],
        dest_order=components["family_order"],
    )
    for source, _dest, ya, yb, _count in left_alloc:
        _draw_ribbon(
            ax,
            left_start_x,
            ya,
            left_end_x,
            yb,
            color=components["basin_color"][int(source)],
            alpha=0.15,
        )
    for _source, dest, ya, yb, _count in right_alloc:
        _draw_ribbon(
            ax,
            right_start_x,
            ya,
            right_end_x,
            yb,
            color=components["family_color"][int(dest)],
            alpha=0.21,
        )

    if show_basin_blocks:
        for basin in components["basin_order"]:
            block_y0, block_y1 = basin_pos[basin]
            _draw_block(
                ax,
                basin_x,
                block_y0,
                block_y1,
                width=0.062,
                color=components["basin_color"][basin],
                edgecolor="#27303a",
                alpha=0.96,
                linewidth=0.30,
            )
            if block_y1 - block_y0 >= 0.040:
                ax.text(
                    basin_x,
                    0.5 * (block_y0 + block_y1),
                    f"B{basin}",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color="white",
                    fontweight="normal",
                    zorder=4,
                )

    for support in components["support_order"]:
        block_y0, block_y1 = support_pos[support]
        _draw_block(
            ax,
            support_x,
            block_y0,
            block_y1,
            width=0.027,
            color=components["support_color"][support],
            edgecolor=components["support_color"][support],
            alpha=0.64,
            linewidth=0.06,
        )

    for family in components["family_order"]:
        block_y0, block_y1 = family_pos[family]
        _draw_block(
            ax,
            family_x,
            block_y0,
            block_y1,
            width=0.062,
            color=components["family_color"][family],
            edgecolor="#27303a",
            alpha=0.96,
            linewidth=0.30,
        )
        if block_y1 - block_y0 >= 0.040:
            ax.text(
                family_x,
                0.5 * (block_y0 + block_y1),
                f"F{family}",
                ha="center",
                va="center",
                fontsize=6.3,
                color="white",
                fontweight="normal",
                zorder=4,
            )

    ax.text(basin_x, 0.94, "Basin", ha="center", va="top", fontsize=7.1, color="#111827")
    ax.text(
        support_x,
        0.94,
        r"Exact $S_{\rm abs}$",
        ha="center",
        va="top",
        fontsize=7.1,
        color="#111827",
    )
    ax.text(
        family_x,
        0.94,
        r"Family $F_{\rm abs}$",
        ha="center",
        va="top",
        fontsize=7.1,
        color="#111827",
    )
    if show_caption:
        ax.text(
            0.5,
            0.035,
            f"{len(components['support_counts'])} exact supports, merged to {len(components['family_counts'])} families",
            ha="center",
            va="bottom",
            fontsize=5.9,
            color="#4b5563",
        )
    return {"basin_pos": basin_pos, "support_pos": support_pos, "family_pos": family_pos}


def _draw_barcode_rows(
    ax: Any,
    data: SpatialAlluvialData,
    components: dict[str, Any],
    *,
    title: str = "Prototype supports",
    max_families: int = 6,
) -> list[dict[str, Any]]:
    from matplotlib.patches import Rectangle

    family_counts = components["family_counts"]
    displayed = sorted(
        family_counts,
        key=lambda family: (
            data.family_to_basin.get(int(family), 10**9),
            -family_counts[int(family)],
            int(family),
        ),
    )[:max_families]
    ax.set_xlim(-0.5, data.latent_dim - 0.5)
    ax.set_ylim(-0.15, max(1, len(displayed)) + 0.55)
    ax.set_title(title, loc="left", pad=3)
    ax.set_yticks([])
    ax.set_xlabel("latent coordinate")
    tick_step = 32 if data.latent_dim <= 320 else 64
    ax.set_xticks(np.arange(0, data.latent_dim, tick_step))
    ax.tick_params(axis="x", length=2.4, pad=2)
    for x in np.arange(0, data.latent_dim, tick_step):
        ax.axvline(x - 0.5, color="#edf0f2", lw=0.40, zorder=0)

    metadata: list[dict[str, Any]] = []
    total = max(1, int(data.families.size))
    for row, family in enumerate(displayed[::-1]):
        y = row + 0.33
        color = components["family_color"][family]
        ax.add_patch(
            Rectangle(
                (-0.5, y - 0.11),
                data.latent_dim,
                0.22,
                facecolor="#f8f9fa",
                edgecolor="#aeb4bd",
                linewidth=0.45,
                zorder=1,
            )
        )
        active = np.flatnonzero(data.prototypes[int(family)].astype(bool)).astype(int)
        for coord in active.tolist():
            ax.add_patch(
                Rectangle(
                    (coord - 0.36, y - 0.105),
                    0.72,
                    0.21,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.92,
                    zorder=3,
                )
            )
        basin = int(data.family_to_basin.get(int(family), -1))
        mass = 100.0 * family_counts[int(family)] / total
        ax.text(
            -0.01,
            y,
            f"B{basin} F{family}  {mass:.0f}%",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.4,
            color=color,
        )
        metadata.append(
            {
                "family": int(family),
                "dominant_basin": basin,
                "count": int(family_counts[int(family)]),
                "fraction": float(family_counts[int(family)] / total),
                "prototype_active_indices": active.tolist(),
            }
        )

    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.55)
    return metadata[::-1]


def _draw_triptych(data: SpatialAlluvialData, components: dict[str, Any], env: Any, output: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _figure_setup()
    fig = plt.figure(figsize=(7.45, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.22, 0.98], wspace=0.20)
    ax_space = fig.add_subplot(gs[0, 0])
    ax_flow = fig.add_subplot(gs[0, 1])
    ax_code = fig.add_subplot(gs[0, 2])

    _draw_spatial_panel(
        ax_space,
        data,
        components,
        env=env,
        title="State-space points",
        point_size=3.2,
    )
    _draw_alluvial(ax_flow, data, components)
    barcode_metadata = _draw_barcode_rows(ax_code, data, components, title="Family prototypes")

    fig.subplots_adjust(left=0.022, right=0.995, top=0.88, bottom=0.17)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    png = output.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "prototype": "triptych",
        "output_pdf": str(output),
        "output_png": str(png),
        "description": "Side-by-side spatial map, basin-to-support-to-family alluvial, and family prototype barcodes.",
        "displayed_families": barcode_metadata,
    }


def _draw_basin_thumbnail(
    ax: Any,
    data: SpatialAlluvialData,
    components: dict[str, Any],
    *,
    basin: int,
    color: str,
) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = _colors(data)
    cmap = ListedColormap(colors[: data.basin_count])
    norm = BoundaryNorm(np.arange(-0.5, data.basin_count + 0.5, 1.0), data.basin_count)
    ax.pcolormesh(
        data.xx,
        data.yy,
        data.basin_labels_grid,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        alpha=0.10,
        rasterized=True,
    )
    keep = data.selected_basin_labels == basin
    ax.scatter(
        data.selected_states[keep, 0],
        data.selected_states[keep, 1],
        c=color,
        s=1.3,
        alpha=0.78,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlim(float(data.xx.min()), float(data.xx.max()))
    ax.set_ylim(float(data.yy.min()), float(data.yy.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#aeb4bd")
        spine.set_linewidth(0.45)


def _draw_source_inset_alluvial(
    data: SpatialAlluvialData,
    components: dict[str, Any],
    _env: Any,
    output: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    _figure_setup()
    fig, ax = plt.subplots(figsize=(6.95, 2.45))
    layout = _draw_alluvial(
        ax,
        data,
        components,
        y0=0.12,
        y1=0.86,
        basin_x=0.16,
        support_x=0.58,
        family_x=0.91,
        left_start_x=0.27,
        left_end_x=0.565,
        right_start_x=0.595,
        right_end_x=0.865,
        show_basin_blocks=False,
    )
    basin_pos = layout["basin_pos"]
    for basin in components["basin_order"]:
        y0, y1 = basin_pos[basin]
        inset = ax.inset_axes([0.020, y0, 0.205, max(0.020, y1 - y0)], transform=ax.transAxes)
        _draw_basin_thumbnail(
            inset,
            data,
            components,
            basin=int(basin),
            color=components["basin_color"][basin],
        )
        ax.text(
            0.238,
            0.5 * (y0 + y1),
            f"B{basin}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.4,
            color=components["basin_color"][basin],
        )
    ax.text(
        0.122,
        0.94,
        "State-space\nsource points",
        ha="center",
        va="top",
        fontsize=7.1,
        color="#111827",
    )
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    png = output.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "prototype": "source_insets",
        "output_pdf": str(output),
        "output_png": str(png),
        "description": "Alluvial whose source column is spatial thumbnails of the selected basin points.",
    }


def _representative_family_points(data: SpatialAlluvialData, components: dict[str, Any]) -> dict[int, tuple[float, float]]:
    representatives: dict[int, tuple[float, float]] = {}
    for family in components["family_order"]:
        keep = data.families == int(family)
        if not bool(np.any(keep)):
            continue
        points = data.selected_states[keep]
        center = points.mean(axis=0)
        idx = int(np.argmin(np.sum((points - center[None, :]) ** 2, axis=1)))
        representatives[int(family)] = (float(points[idx, 0]), float(points[idx, 1]))
    return representatives


def _draw_callouts(
    data: SpatialAlluvialData,
    components: dict[str, Any],
    env: Any,
    output: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    _figure_setup()
    fig = plt.figure(figsize=(6.65, 2.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.08], wspace=0.16)
    ax_space = fig.add_subplot(gs[0, 0])
    ax_code = fig.add_subplot(gs[0, 1])
    _draw_spatial_panel(
        ax_space,
        data,
        components,
        env=env,
        title="Spatial family regions",
        point_size=2.8,
        show_mismatch=True,
    )
    barcode_metadata = _draw_barcode_rows(
        ax_code,
        data,
        components,
        title="Support prototypes reached by those regions",
        max_families=8,
    )

    row_lookup: dict[int, float] = {}
    displayed = [entry["family"] for entry in barcode_metadata]
    for row, family in enumerate(displayed[::-1]):
        row_lookup[int(family)] = row + 0.33
    representatives = _representative_family_points(data, components)
    for family, point in representatives.items():
        if family not in row_lookup:
            continue
        color = components["family_color"].get(family, "#8a8f98")
        ax_space.scatter(
            [point[0]],
            [point[1]],
            s=34,
            marker="o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.75,
            zorder=8,
        )
        connector = ConnectionPatch(
            xyA=point,
            coordsA=ax_space.transData,
            xyB=(0.0, row_lookup[family]),
            coordsB=ax_code.get_yaxis_transform(),
            arrowstyle="-",
            color=color,
            linewidth=0.65,
            alpha=0.48,
            shrinkA=3.0,
            shrinkB=3.0,
            mutation_scale=1.0,
            zorder=2,
        )
        fig.add_artist(connector)

    fig.subplots_adjust(left=0.035, right=0.995, top=0.90, bottom=0.16)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    png = output.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "prototype": "family_callouts",
        "output_pdf": str(output),
        "output_png": str(png),
        "description": "Spatial support-family regions connected directly to their prototype active-coordinate barcodes.",
        "displayed_families": barcode_metadata,
    }


def _write_metadata(
    output_dir: Path,
    data: SpatialAlluvialData,
    components: dict[str, Any],
    prototypes: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = {
        "figure_set": "spatial_alluvial_prototypes",
        "system": data.system,
        "title": data.title,
        "root_label": data.root_label,
        "model_display": ROOT_DISPLAY.get(data.root_label, data.root_label),
        "seed": int(data.seed),
        "run_dir": str(data.run_dir),
        "grid_points": int(data.grid_points),
        "selected_state_count": int(data.selected_basin_labels.size),
        "selected_subset": data.subset,
        "depth_slice_mode": data.depth_slice_mode,
        "basin_count": int(data.basin_count),
        "represented_basins": sorted({int(item) for item in data.selected_basin_labels.tolist()}),
        "support_rule": f"{data.support_scheme}:{data.support_value:g}",
        "family_jaccard": float(data.family_jaccard),
        "latent_dim": int(data.latent_dim),
        "exact_support_count": int(len(components["support_counts"])),
        "support_family_count": int(len(components["family_counts"])),
        "family_dominant_basin_agreement": float(data.family_agreement),
        "mean_family_purity": float(np.mean(list(data.family_purity.values()))) if data.family_purity else 0.0,
        "basin_label_source": data.basin_label_source,
        "deep_subset_center_source": data.label_source,
        "prototypes": prototypes,
        "families": [
            {
                "family": int(family),
                "dominant_basin": int(data.family_to_basin.get(int(family), -1)),
                "count": int(components["family_counts"][int(family)]),
                "fraction": float(components["family_counts"][int(family)] / max(1, data.families.size)),
                "purity": float(data.family_purity.get(int(family), 0.0)),
            }
            for family in components["family_order"]
        ],
        "note": (
            "These are display prototypes from existing checkpoints. Basin labels and basin counts "
            "are evaluation-only; support-family construction uses only sparse support masks."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpretability-csv",
        type=Path,
        action="append",
        default=None,
        help="interpretability_rows.csv path; can be passed multiple times",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--root-label", default=DEFAULT_ROOT_LABEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-points", type=int, default=96)
    parser.add_argument("--endpoint-rollout-steps", type=int, default=5000)
    parser.add_argument("--support-scheme", default="absolute", choices=["absolute", "relative", "topk"])
    parser.add_argument("--support-value", type=float, default=1e-3)
    parser.add_argument("--family-jaccard", type=float, default=0.5)
    parser.add_argument("--subset", default="deep", choices=["all", "deep", "boundary"])
    parser.add_argument("--depth-slice-mode", default="per_basin", choices=["global", "per_basin"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, env = _load_data(args)
    components = _flow_components(data)

    stem_base = f"fig_spatial_alluvial_{args.system.replace(':', '_').replace('-', '_')}"
    prototypes = [
        _draw_triptych(
            data,
            components,
            env,
            args.output_dir / f"{stem_base}_prototype_a_triptych.pdf",
        ),
        _draw_source_inset_alluvial(
            data,
            components,
            env,
            args.output_dir / f"{stem_base}_prototype_b_source_insets.pdf",
        ),
        _draw_callouts(
            data,
            components,
            env,
            args.output_dir / f"{stem_base}_prototype_c_callouts.pdf",
        ),
    ]
    metadata = _write_metadata(args.output_dir, data, components, prototypes)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
