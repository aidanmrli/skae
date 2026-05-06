#!/usr/bin/env python3
"""Build draft paper visuals for sparse-support multibasin experiments.

The figures are deterministic data visualizations, not generated artwork:

1. support barcode map
2. basin -> exact support -> support-family alluvial diagram
3. wrong-support counterfactual triptych
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.make_benchmark_support_dysts_composite import (
    OKABE_ITO,
    ROOT_LISTA_SUPPORT,
    SUPPORT_PANELS,
    _basin_labels_for_states,
    _dynamics_for_states,
    _encode_latents,
    _find_run_dir,
    _grid_states,
    _load_model_and_env,
    _read_rows,
    _topk_support_mask,
)
from tools.make_support_family_index_codebook import _support_family_labels_and_prototypes
from tools.reduce_transition_rich_interpretability_metrics import (
    _encode_trajectories,
    _generate_observation_trajectories,
    _label_sequences_and_centers,
    _support_keys,
    _support_mask,
    canonical_support_masks_by_basin,
)


DEFAULT_INTERPRETABILITY_CSV = Path(
    "results/transition_rich_basin_partition_final_seed10_20260409/"
    "interpretability_final_pass1/interpretability_rows.csv"
)
DEFAULT_OUTPUT_DIR = Path("docs/figures/neurips_paper_2026")
DEFAULT_SYSTEM = "gated_local_linear"


@dataclass(frozen=True)
class SystemBundle:
    system: str
    title: str
    run_dir: Path
    model: Any
    env: Any
    xx: np.ndarray
    yy: np.ndarray
    states: torch.Tensor
    basin_labels: np.ndarray
    basin_label_source: str
    latents: np.ndarray
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def _figure_setup() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def _load_bundle(args: argparse.Namespace) -> SystemBundle:
    specs = {spec.system: spec for spec in SUPPORT_PANELS}
    if args.system not in specs:
        raise ValueError(f"Unknown default panel system: {args.system}")
    spec = specs[args.system]
    rows = _read_rows(args.interpretability_csv)
    run_dir = _find_run_dir(
        rows,
        root_label=args.root_label,
        system=spec.system,
        seed=args.seed,
    )
    model, env = _load_model_and_env(run_dir, spec.system, args.device)
    xx, yy, states = _grid_states(spec.xlim, spec.ylim, args.grid_points)
    basin_labels_t, basin_source = _basin_labels_for_states(
        env,
        spec.system,
        states,
        endpoint_rollout_steps=args.endpoint_rollout_steps,
    )
    latents = _encode_latents(model, states, args.device)
    return SystemBundle(
        system=spec.system,
        title=spec.title,
        run_dir=run_dir,
        model=model,
        env=env,
        xx=xx,
        yy=yy,
        states=states,
        basin_labels=basin_labels_t.numpy(),
        basin_label_source=basin_source,
        latents=latents,
        xlim=spec.xlim,
        ylim=spec.ylim,
    )


def _basin_colors(count: int) -> list[str]:
    return [OKABE_ITO[idx % len(OKABE_ITO)] for idx in range(count)]


def _family_dominant_basins(families: np.ndarray, basin_labels: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for family in sorted({int(item) for item in families.tolist()}):
        keep = families == family
        counts = Counter(int(item) for item in basin_labels[keep].tolist() if int(item) >= 0)
        out[family] = counts.most_common(1)[0][0] if counts else -1
    return out


def _draw_streamlines(ax, bundle: SystemBundle, *, vector_points: int = 30) -> None:
    vx, vy, vstates = _grid_states(bundle.xlim, bundle.ylim, vector_points)
    velocity = _dynamics_for_states(bundle.env, vstates)
    u = velocity[:, 0].numpy().reshape(vector_points, vector_points)
    v = velocity[:, 1].numpy().reshape(vector_points, vector_points)
    speed = np.sqrt(u**2 + v**2)
    linewidth = 0.25 + 0.45 * speed / max(float(np.nanpercentile(speed, 95)), 1e-8)
    ax.streamplot(
        vx[0],
        vy[:, 0],
        u,
        v,
        color="#30343a",
        linewidth=linewidth,
        density=1.0,
        arrowsize=0.5,
        zorder=4,
    )


def _draw_barcode_axis(ax, mask: np.ndarray, *, color: str, label: str = "") -> None:
    rgb_off = np.array([248, 249, 250], dtype=float) / 255.0
    import matplotlib.colors as mcolors

    rgb_on = np.asarray(mcolors.to_rgb(color), dtype=float)
    img = np.repeat(rgb_off[None, None, :], mask.shape[0], axis=1)
    img[0, mask.astype(bool)] = rgb_on
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#aeb4bd")
        spine.set_linewidth(0.5)
    if label:
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=18, fontsize=6.7)


def _support_barcode_map(bundle: SystemBundle, args: argparse.Namespace) -> dict[str, Any]:
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
    family_to_basin = _family_dominant_basins(families, bundle.basin_labels)
    support_basin = np.asarray([family_to_basin[int(item)] for item in families], dtype=int)

    family_counts = Counter(int(item) for item in families.tolist())
    displayed = []
    for basin in range(basin_count):
        candidates = [
            (family, count)
            for family, count in family_counts.items()
            if family_to_basin.get(int(family), -1) == basin
        ]
        if candidates:
            family = int(max(candidates, key=lambda item: item[1])[0])
            displayed.append(family)
    displayed = sorted(set(displayed), key=lambda fam: (family_to_basin[fam], -family_counts[fam]))

    _figure_setup()
    fig = plt.figure(figsize=(7.1, 3.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.48, 1.0], wspace=0.18)
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
        alpha=0.13,
        rasterized=True,
    )
    ax.scatter(
        bundle.states[:, 0].numpy(),
        bundle.states[:, 1].numpy(),
        c=support_basin,
        cmap=cmap,
        norm=norm,
        s=2.8,
        alpha=0.68,
        linewidths=0,
        rasterized=True,
    )
    mismatch = support_basin != bundle.basin_labels
    if bool(np.any(mismatch)):
        ax.scatter(
            bundle.states[mismatch, 0].numpy(),
            bundle.states[mismatch, 1].numpy(),
            c="#111111",
            s=1.0,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
        )
    _draw_streamlines(ax, bundle)

    for marker_index, family in enumerate(displayed):
        keep = np.flatnonzero(families == family)
        center = bundle.states[keep].numpy().mean(axis=0)
        point_idx = keep[np.argmin(np.sum((bundle.states[keep].numpy() - center) ** 2, axis=1))]
        basin = family_to_basin[family]
        ax.scatter(
            [float(bundle.states[point_idx, 0])],
            [float(bundle.states[point_idx, 1])],
            marker=["o", "s", "D", "^", "v"][marker_index % 5],
            s=58,
            facecolor=colors[basin % len(colors)],
            edgecolor="white",
            linewidth=1.0,
            zorder=8,
        )

    ax.set_title("Support-family atlas in state space", loc="left", fontweight="bold")
    ax.set_xlim(bundle.xlim)
    ax.set_ylim(bundle.ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    ax_code.axis("off")
    ax_code.text(
        0.0,
        1.0,
        "Prototype support barcodes",
        transform=ax_code.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=8.4,
    )
    ax_code.text(
        0.0,
        0.92,
        "active latent coordinates for the largest family in each basin",
        transform=ax_code.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color="#4b5563",
    )
    y0 = 0.74
    for row, family in enumerate(displayed):
        basin = family_to_basin[family]
        inset = ax_code.inset_axes([0.17, y0 - row * 0.18, 0.78, 0.062])
        _draw_barcode_axis(
            inset,
            prototypes[family].astype(bool),
            color=colors[basin % len(colors)],
        )
        mass = 100.0 * family_counts[family] / max(1, families.size)
        ax_code.text(
            0.0,
            y0 + 0.028 - row * 0.18,
            f"B{basin} F{family}",
            transform=ax_code.transAxes,
            ha="left",
            va="center",
            fontsize=7.0,
            color=colors[basin % len(colors)],
            fontweight="bold",
        )
        ax_code.text(
            0.17,
            y0 - 0.026 - row * 0.18,
            f"{mass:.1f}% of grid",
            transform=ax_code.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color="#4b5563",
        )
    ax_code.text(
        0.0,
        0.03,
        "Color uses the post-hoc dominant evaluation basin; training uses no basin labels.",
        transform=ax_code.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#4b5563",
    )

    output = args.output_dir / "fig_support_barcode_map_draft.pdf"
    png = output.with_suffix(".png")
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure": "support_barcode_map",
        "output_pdf": str(output),
        "output_png": str(png),
        "support_definition": f"topk:{args.topk}",
        "family_jaccard": args.family_jaccard,
        "support_family_count": int(len(family_counts)),
        "dominant_basin_agreement_on_grid": float(np.mean(support_basin == bundle.basin_labels)),
        "displayed_families": [
            {
                "family": int(family),
                "dominant_basin": int(family_to_basin[family]),
                "grid_fraction": float(family_counts[family] / max(1, families.size)),
                "prototype_indices": np.flatnonzero(prototypes[family]).astype(int).tolist(),
            }
            for family in displayed
        ],
    }


def _deep_like_mask(states: torch.Tensor, basin_labels: np.ndarray) -> np.ndarray:
    xy = states.numpy()
    basins = sorted({int(item) for item in basin_labels.tolist() if int(item) >= 0})
    centers = np.stack([xy[basin_labels == basin].mean(axis=0) for basin in basins], axis=0)
    distances = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=-1)
    sorted_distances = np.sort(distances, axis=1)
    margin = sorted_distances[:, 1] - sorted_distances[:, 0] if centers.shape[0] > 1 else sorted_distances[:, 0]
    keep = np.zeros_like(basin_labels, dtype=bool)
    for basin in basins:
        select = basin_labels == basin
        if bool(np.any(select)):
            threshold = np.quantile(margin[select], 0.75)
            keep[select] = margin[select] >= threshold
    return keep


def _stack_positions(
    counts: dict[str, int],
    order: list[str],
    *,
    y0: float = 0.06,
    y1: float = 0.94,
    gap: float = 0.012,
) -> dict[str, tuple[float, float]]:
    total = max(1, sum(counts[item] for item in order))
    available = max(0.01, (y1 - y0) - gap * max(0, len(order) - 1))
    cursor = y1
    out: dict[str, tuple[float, float]] = {}
    for item in order:
        height = available * counts[item] / total
        out[item] = (cursor - height, cursor)
        cursor -= height + gap
    return out


def _draw_ribbon(ax, x0: float, ya: tuple[float, float], x1: float, yb: tuple[float, float], *, color: str, alpha: float) -> None:
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    y0a, y1a = ya
    y0b, y1b = yb
    dx = 0.18 * (x1 - x0)
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


def _support_key(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8)).tobytes()


def _alluvial_figure(bundle: SystemBundle, args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    trajectories = _generate_observation_trajectories(
        bundle.env,
        num_trajectories=args.alluvial_num_trajectories,
        trajectory_length=args.alluvial_trajectory_length,
        eval_seed=args.eval_seed,
    )
    basin_labels_t, _centers, alluvial_label_source = _label_sequences_and_centers(
        bundle.env,
        trajectories,
        system_key=bundle.system,
        endpoint_rollout_steps=args.endpoint_rollout_steps,
    )
    trajectory_latents = _encode_trajectories(bundle.model, trajectories, args.device)
    support_mask = _support_mask(trajectory_latents, scheme="absolute", value=args.absolute_threshold)
    flat_support_mask = support_mask.reshape(-1, support_mask.shape[-1])
    flat_basin_labels = basin_labels_t.cpu().numpy().reshape(-1)
    families, prototypes = _support_family_labels_and_prototypes(
        flat_support_mask,
        min_jaccard=args.family_jaccard,
    )
    family_to_basin = _family_dominant_basins(families, flat_basin_labels)
    selected = np.arange(flat_basin_labels.size)

    keys = [_support_key(mask) for mask in flat_support_mask]
    support_counts = Counter(keys[idx] for idx in selected.tolist())
    top_keys = {key for key, _ in support_counts.most_common(args.alluvial_top_supports)}
    key_to_display = {key: f"S{rank + 1}" for rank, (key, _count) in enumerate(support_counts.most_common(args.alluvial_top_supports))}
    key_to_family: dict[bytes, int] = {}
    key_to_mask: dict[bytes, np.ndarray] = {}
    for key, family, mask in zip(keys, families.tolist(), flat_support_mask):
        key_to_family.setdefault(key, int(family))
        key_to_mask.setdefault(key, mask.astype(bool, copy=True))

    support_display: list[str] = []
    for idx in selected.tolist():
        key = keys[idx]
        if key in top_keys:
            support_display.append(key_to_display[key])
        else:
            support_display.append(f"other exact S -> F{int(families[idx])}")
    basin_display = [f"B{int(flat_basin_labels[idx])}" for idx in selected.tolist()]
    family_display = [f"F{int(families[idx])}" for idx in selected.tolist()]

    basin_counts = Counter(basin_display)
    support_display_counts = Counter(support_display)
    family_counts = Counter(family_display)
    flow_bs = Counter(zip(basin_display, support_display))
    flow_sf = Counter(zip(support_display, family_display))

    basin_order = [f"B{basin}" for basin in sorted({int(item) for item in flat_basin_labels[selected].tolist()})]
    support_order = sorted(
        support_display_counts,
        key=lambda label: (
            label.startswith("other"),
            -support_display_counts[label],
            label,
        ),
    )
    family_order = sorted(family_counts, key=lambda label: (family_to_basin.get(int(label[1:]), 999), label))
    basin_pos = _stack_positions(dict(basin_counts), basin_order, y0=0.08, y1=0.86, gap=0.018)
    support_pos = _stack_positions(dict(support_display_counts), support_order, y0=0.08, y1=0.86, gap=0.009)
    family_pos = _stack_positions(dict(family_counts), family_order, y0=0.08, y1=0.86, gap=0.018)

    basin_count = int(bundle.basin_labels.max()) + 1
    colors = _basin_colors(basin_count)
    category_color: dict[str, str] = {}
    for basin_label in basin_order:
        category_color[basin_label] = colors[int(basin_label[1:]) % len(colors)]
    for family_label in family_order:
        family = int(family_label[1:])
        category_color[family_label] = colors[family_to_basin.get(family, 0) % len(colors)]
    for support_label in support_order:
        related = [family for (support, family), count in flow_sf.items() if support == support_label and count > 0]
        if related:
            dominant_family = max(related, key=lambda family: flow_sf[(support_label, family)])
            category_color[support_label] = category_color[dominant_family]
        else:
            category_color[support_label] = "#8a8f98"

    def allocate(
        source_pos: dict[str, tuple[float, float]],
        dest_pos: dict[str, tuple[float, float]],
        flows: Counter[tuple[str, str]],
    ) -> list[tuple[str, str, tuple[float, float], tuple[float, float], int]]:
        source_cursor = {key: source_pos[key][0] for key in source_pos}
        dest_cursor = {key: dest_pos[key][0] for key in dest_pos}
        source_total = defaultdict(int)
        dest_total = defaultdict(int)
        for (source, dest), count in flows.items():
            source_total[source] += count
            dest_total[dest] += count
        out = []
        for source, dest in sorted(flows, key=lambda item: (item[0][0], item[0][1])):
            count = flows[(source, dest)]
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

    _figure_setup()
    fig, ax = plt.subplots(figsize=(7.05, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for source, dest, ya, yb, _count in allocate(basin_pos, support_pos, flow_bs):
        _draw_ribbon(ax, 0.16, ya, 0.50, yb, color=category_color[source], alpha=0.27)
    for source, dest, ya, yb, _count in allocate(support_pos, family_pos, flow_sf):
        _draw_ribbon(ax, 0.54, ya, 0.86, yb, color=category_color[dest], alpha=0.31)

    def draw_blocks(x: float, positions: dict[str, tuple[float, float]], order: list[str], *, width: float, label_kind: str) -> None:
        for label in order:
            y0, y1 = positions[label]
            color = category_color[label]
            ax.add_patch(
                Rectangle(
                    (x - width / 2, y0),
                    width,
                    y1 - y0,
                    facecolor=color,
                    edgecolor="#202124",
                    linewidth=0.35,
                    alpha=0.86,
                    zorder=3,
                )
            )
            shown = label
            if label.startswith("other"):
                shown = label.replace("other exact S -> ", "other→")
            min_height = 0.030 if label_kind == "support" else 0.020
            if (y1 - y0) >= min_height:
                ax.text(
                    x,
                    0.5 * (y0 + y1),
                    shown,
                    ha="center",
                    va="center",
                    fontsize=6.0 if label_kind == "support" else 6.4,
                    color="white" if label_kind != "support" else "#202124",
                    fontweight="bold" if label_kind != "support" else "normal",
                    zorder=4,
                )

    draw_blocks(0.12, basin_pos, basin_order, width=0.075, label_kind="basin")
    draw_blocks(0.52, support_pos, support_order, width=0.095, label_kind="support")
    draw_blocks(0.90, family_pos, family_order, width=0.075, label_kind="family")

    ax.text(0.12, 0.94, "evaluation\nbasin", ha="center", va="top", fontsize=7.4, fontweight="bold")
    ax.text(0.52, 0.94, "exact support\n$S_{\\rm abs}$", ha="center", va="top", fontsize=7.4, fontweight="bold")
    ax.text(0.90, 0.94, "support family\n$F_{\\rm abs}$", ha="center", va="top", fontsize=7.4, fontweight="bold")
    ax.text(
        0.01,
        0.015,
        "Evaluation trajectories; labels are used only after training to audit alignment. Thin exact supports are aggregated by destination family.",
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#4b5563",
    )
    ax.set_title(
        "Exact supports fragment; support families recover basin-scale structure",
        loc="left",
        fontsize=9.2,
        fontweight="bold",
        pad=5,
    )

    output = args.output_dir / "fig_basin_support_family_alluvial_draft.pdf"
    png = output.with_suffix(".png")
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure": "basin_support_family_alluvial",
        "output_pdf": str(output),
        "output_png": str(png),
        "support_definition": f"absolute:{args.absolute_threshold:g}",
        "family_jaccard": args.family_jaccard,
        "label_source": alluvial_label_source,
        "num_trajectories": int(args.alluvial_num_trajectories),
        "trajectory_length": int(args.alluvial_trajectory_length),
        "selected_state_count": int(selected.size),
        "exact_support_count_selected": int(len(support_counts)),
        "displayed_exact_support_count": int(len(top_keys)),
        "family_count_selected": int(len(family_counts)),
        "family_dominant_basins": {str(key): int(value) for key, value in family_to_basin.items()},
        "top_exact_supports": [
            {
                "label": key_to_display[key],
                "count": int(count),
                "family": int(key_to_family[key]),
                "active_indices": np.flatnonzero(key_to_mask[key]).astype(int).tolist(),
            }
            for key, count in support_counts.most_common(args.alluvial_top_supports)
        ],
    }


def _env_rollout(env, x0: torch.Tensor, horizon: int) -> torch.Tensor:
    xs = [x0.detach().cpu()]
    current = x0.unsqueeze(0)
    for _ in range(horizon):
        current = env.step(current).detach()
        xs.append(current.squeeze(0).cpu())
    return torch.stack(xs, dim=0)


def _latent_rollout_path(model, z0: torch.Tensor, horizon: int, *, freeze_mask: torch.Tensor | None = None) -> torch.Tensor:
    xs = []
    z = z0.unsqueeze(0)
    if freeze_mask is not None:
        z = z * freeze_mask
    xs.append(model.decode(z).squeeze(0).detach().cpu())
    for _ in range(horizon):
        z = model.step_latent(z)
        if freeze_mask is not None:
            z = z * freeze_mask
        xs.append(model.decode(z).squeeze(0).detach().cpu())
    return torch.stack(xs, dim=0)


def _pick_counterfactual(bundle: SystemBundle, args: argparse.Namespace) -> dict[str, Any]:
    support_mask = _support_mask(bundle.latents, scheme="absolute", value=args.absolute_threshold)
    deep_mask = _deep_like_mask(bundle.states, bundle.basin_labels)
    templates = canonical_support_masks_by_basin(
        support_mask[None, :, :],
        bundle.basin_labels[None, :],
        deep_mask,
    )
    if len(templates) < 2:
        templates = canonical_support_masks_by_basin(
            support_mask[None, :, :],
            bundle.basin_labels[None, :],
            np.ones_like(bundle.basin_labels, dtype=bool),
        )
    if len(templates) < 2:
        raise RuntimeError("Need at least two canonical basin supports for counterfactual")

    xy = bundle.states.numpy()
    basin_ids = sorted(templates)
    centers = {basin: xy[bundle.basin_labels == basin].mean(axis=0) for basin in basin_ids}
    candidates: list[int] = []
    for basin in basin_ids:
        select = np.flatnonzero(deep_mask & (bundle.basin_labels == basin))
        if select.size == 0:
            continue
        center = centers[basin]
        ranked = select[np.argsort(np.sum((xy[select] - center) ** 2, axis=1))]
        candidates.extend(ranked[: args.counterfactual_candidates].tolist())

    best: dict[str, Any] | None = None
    with torch.no_grad():
        for idx in candidates:
            basin = int(bundle.basin_labels[idx])
            if basin not in templates:
                continue
            z0 = torch.from_numpy(bundle.latents[idx]).to(device=args.device, dtype=torch.float32)
            x0 = bundle.states[idx].to(device=args.device, dtype=torch.float32)
            true_path = _env_rollout(bundle.env, x0, args.counterfactual_horizon)
            own_mask = torch.from_numpy(templates[basin].astype(np.float32)).to(device=args.device).unsqueeze(0)
            own_path = _latent_rollout_path(bundle.model, z0, args.counterfactual_horizon, freeze_mask=own_mask)
            own_err = float(((own_path[-1] - true_path[-1]) ** 2).mean().item())
            for wrong_basin, wrong_template in templates.items():
                if int(wrong_basin) == basin:
                    continue
                wrong_mask = torch.from_numpy(wrong_template.astype(np.float32)).to(device=args.device).unsqueeze(0)
                wrong_path = _latent_rollout_path(bundle.model, z0, args.counterfactual_horizon, freeze_mask=wrong_mask)
                wrong_err = float(((wrong_path[-1] - true_path[-1]) ** 2).mean().item())
                score = wrong_err / max(own_err, 1e-12)
                if best is None or score > best["score"]:
                    base_path = _latent_rollout_path(bundle.model, z0, args.counterfactual_horizon)
                    base_err = float(((base_path[-1] - true_path[-1]) ** 2).mean().item())
                    best = {
                        "state_index": int(idx),
                        "source_basin": int(basin),
                        "wrong_basin": int(wrong_basin),
                        "score": float(score),
                        "true_path": true_path.numpy(),
                        "base_path": base_path.numpy(),
                        "own_path": own_path.numpy(),
                        "wrong_path": wrong_path.numpy(),
                        "base_err": base_err,
                        "own_err": own_err,
                        "wrong_err": wrong_err,
                        "own_mask": templates[basin].astype(bool),
                        "wrong_mask": wrong_template.astype(bool),
                    }
    if best is None:
        raise RuntimeError("No counterfactual candidate found")
    return best


def _draw_basin_background(ax, bundle: SystemBundle) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    basin_count = int(bundle.basin_labels.max()) + 1
    cmap = ListedColormap(_basin_colors(basin_count))
    norm = BoundaryNorm(np.arange(-0.5, basin_count + 0.5, 1), basin_count)
    ax.pcolormesh(
        bundle.xx,
        bundle.yy,
        bundle.basin_labels.reshape(bundle.xx.shape),
        cmap=cmap,
        norm=norm,
        shading="nearest",
        alpha=0.08,
        rasterized=True,
    )


def _counterfactual_figure(bundle: SystemBundle, args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    chosen = _pick_counterfactual(bundle, args)
    _figure_setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharex=True, sharey=True)
    panels = [
        ("Global $K$", chosen["base_path"], "#0072B2", chosen["base_err"], None),
        ("Correct support", chosen["own_path"], "#009E73", chosen["own_err"], chosen["own_mask"]),
        ("Wrong-basin support", chosen["wrong_path"], "#D55E00", chosen["wrong_err"], chosen["wrong_mask"]),
    ]
    all_paths = np.concatenate(
        [chosen["true_path"], chosen["base_path"], chosen["own_path"], chosen["wrong_path"]],
        axis=0,
    )
    lo = np.nanmin(all_paths[:, :2], axis=0)
    hi = np.nanmax(all_paths[:, :2], axis=0)
    span = np.maximum(hi - lo, 1e-3)
    lo = lo - 0.2 * span
    hi = hi + 0.2 * span
    xlim = (max(bundle.xlim[0], float(lo[0])), min(bundle.xlim[1], float(hi[0])))
    ylim = (max(bundle.ylim[0], float(lo[1])), min(bundle.ylim[1], float(hi[1])))
    if xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
        xlim, ylim = bundle.xlim, bundle.ylim

    for ax, (title, path, color, err, mask) in zip(axes, panels):
        _draw_basin_background(ax, bundle)
        ax.plot(chosen["true_path"][:, 0], chosen["true_path"][:, 1], color="#5f6368", lw=1.7, label="true")
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.55, label="model")
        ax.scatter(chosen["true_path"][0, 0], chosen["true_path"][0, 1], s=20, color="#202124", zorder=5)
        ax.scatter(chosen["true_path"][-1, 0], chosen["true_path"][-1, 1], s=28, color="#5f6368", marker="x", zorder=5)
        ax.scatter(path[-1, 0], path[-1, 1], s=30, color=color, marker="x", zorder=6)
        ax.set_title(title, fontweight="bold", color=color if mask is not None else "#202124")
        ax.text(
            0.03,
            0.04,
            f"final MSE {err:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d0d5dd", "linewidth": 0.4},
        )
        if mask is not None:
            inset = ax.inset_axes([0.08, 0.87, 0.84, 0.075])
            _draw_barcode_axis(inset, mask, color=color)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)

    axes[0].legend(frameon=False, loc="upper left", fontsize=6.1, handlelength=1.5)
    fig.suptitle(
        "Wrong-support counterfactual: same state, different active coordinates",
        x=0.02,
        ha="left",
        fontsize=9.2,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.01,
        f"Source basin B{chosen['source_basin']} with canonical support from B{chosen['wrong_basin']} substituted in the right panel.",
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.025, right=0.995, top=0.82, bottom=0.13, wspace=0.045)
    output = args.output_dir / "fig_wrong_support_counterfactual_draft.pdf"
    png = output.with_suffix(".png")
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure": "wrong_support_counterfactual",
        "output_pdf": str(output),
        "output_png": str(png),
        "support_definition": f"absolute:{args.absolute_threshold:g}",
        "horizon": int(args.counterfactual_horizon),
        "state_index": int(chosen["state_index"]),
        "source_basin": int(chosen["source_basin"]),
        "wrong_basin": int(chosen["wrong_basin"]),
        "base_final_mse": float(chosen["base_err"]),
        "correct_support_final_mse": float(chosen["own_err"]),
        "wrong_support_final_mse": float(chosen["wrong_err"]),
        "wrong_over_correct_final_mse": float(chosen["score"]),
        "correct_support_active_indices": np.flatnonzero(chosen["own_mask"]).astype(int).tolist(),
        "wrong_support_active_indices": np.flatnonzero(chosen["wrong_mask"]).astype(int).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpretability-csv", type=Path, default=DEFAULT_INTERPRETABILITY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_OUTPUT_DIR / "fig_visual_drafts_metadata.json")
    parser.add_argument("--root-label", default=ROOT_LISTA_SUPPORT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-points", type=int, default=96)
    parser.add_argument("--endpoint-rollout-steps", type=int, default=5000)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--absolute-threshold", type=float, default=1e-3)
    parser.add_argument("--family-jaccard", type=float, default=0.5)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--alluvial-num-trajectories", type=int, default=192)
    parser.add_argument("--alluvial-trajectory-length", type=int, default=96)
    parser.add_argument("--alluvial-top-supports", type=int, default=10)
    parser.add_argument("--counterfactual-horizon", type=int, default=20)
    parser.add_argument("--counterfactual-candidates", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_bundle(args)
    metadata = {
        "system": bundle.system,
        "title": bundle.title,
        "root_label": args.root_label,
        "seed": int(args.seed),
        "run_dir": str(bundle.run_dir),
        "grid_points": int(args.grid_points),
        "basin_label_source": bundle.basin_label_source,
        "figures": [
            _support_barcode_map(bundle, args),
            _alluvial_figure(bundle, args),
            _counterfactual_figure(bundle, args),
        ],
    }
    args.metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
