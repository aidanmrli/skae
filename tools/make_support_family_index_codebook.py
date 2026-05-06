#!/usr/bin/env python3
"""Build a paper figure showing active latent indices by support family.

The figure is a companion to the multibasin support overlays: it shows which
latent coordinates are active in prototype exact top-k support masks for
learned support families across selected models and three-basin systems.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    TRANSITION_RICH_BASIN_PARTITION_SYSTEMS,
)
from skae.data import _infer_reset_bounds
from tools.make_benchmark_support_dysts_composite import (
    OKABE_ITO,
    _basin_labels_for_states,
    _encode_latents,
    _find_run_dir,
    _grid_states,
    _load_model_and_env,
    _topk_support_mask,
)
from tools.reduce_transition_rich_interpretability_metrics import (
    _generate_observation_trajectories,
    _label_sequences_and_centers,
    _margin_subsets,
)


DEFAULT_INTERPRETABILITY_CSVS = (
    Path(
        "results/transition_rich_table2_5model_seed15_backfill_20260428/"
        "interpretability_pass0/interpretability_rows.csv"
    ),
    Path(
        "results/transition_rich_lista_dense_p256_hardinit_table123_20260430/"
        "interpretability_pass0/interpretability_rows.csv"
    ),
    Path(
        "results/transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428/"
        "interpretability_pass0/interpretability_rows.csv"
    ),
)
DEFAULT_OUTPUT = Path(
    "docs/figures/neurips_paper_2026/fig_support_family_index_codebook.pdf"
)
DEFAULT_METADATA_OUTPUT = Path(
    "docs/figures/neurips_paper_2026/fig_support_family_index_codebook.json"
)
ROOT_LISTA_P256 = "lista_dense_signsplit_p256_hardinit_basin_partition"
ROOT_LISTA_BD = "lista_blockdiag_signsplit_hardinit_basin_partition"
ROOT_LISTA_SB_P256 = "lista_dense_softblock_signsplit_p256_hardinit_basin_partition"


@dataclass(frozen=True)
class SystemSpec:
    system: str
    title: str
    seed: int
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None


@dataclass(frozen=True)
class ModelSpec:
    root_label: str
    title: str


@dataclass(frozen=True)
class SupportDefinition:
    scheme: str
    value: float
    name: str
    support_display: str
    family_display: str


RETAINED_15_EXCLUDED_SYSTEMS = {
    "multiwell_strong_transition",
    "claude:checkerboard_potential",
}

SYSTEM_TITLES = {
    "multiwell_strong_transition": "Strong transition wells",
    "gated_local_linear": "Local-linear gates",
    "gated_transfer_linear": "Transfer gates",
    "claude:arrested_spiral": "Arrested spiral",
    "claude:cal_asymmetric_3": "Asymmetric wells",
    "claude:cal_high_cross_3": "High cross",
    "claude:cal_hexagon_6": "Hexagon wells",
    "claude:cal_octagon_8": "Octagon wells",
    "claude:cal_pentagon_5": "Pentagon wells",
    "claude:cal_square_4": "Square wells",
    "claude:checkerboard_potential": "Checkerboard potential",
    "claude:duffing_triple_well": "Duffing triple well",
    "claude:snic_multi": "SNIC multi",
    "claude:transition_routes_4": "Transition routes",
    "claude:var_depth_gradient_4": "Depth gradient",
    "claude:var_diamond_4": "Diamond wells",
    "claude:var_l_shape_5": "L-shaped wells",
}

SYSTEM_BOUNDS = {
    "multiwell_strong_transition": ((-2.5, 2.5), (-2.5, 2.5)),
    "gated_local_linear": ((-4.0, 4.0), (-4.0, 4.0)),
    "gated_transfer_linear": ((-2.8, 2.8), (-2.8, 2.8)),
    "claude:cal_asymmetric_3": ((-3.0, 3.0), (-2.5, 3.0)),
    "claude:cal_high_cross_3": ((-3.3, 3.3), (-3.3, 3.3)),
    "claude:cal_hexagon_6": ((-3.3, 3.3), (-3.3, 3.3)),
    "claude:cal_octagon_8": ((-4.0, 4.0), (-4.0, 4.0)),
    "claude:cal_pentagon_5": ((-3.3, 3.3), (-3.3, 3.3)),
    "claude:cal_square_4": ((-3.3, 3.3), (-3.3, 3.3)),
    "claude:transition_routes_4": ((-3.5, 3.5), (-3.5, 3.5)),
    "claude:var_depth_gradient_4": ((-3.0, 3.0), (-3.0, 3.0)),
    "claude:var_diamond_4": ((-3.5, 3.5), (-3.5, 3.5)),
    "claude:var_l_shape_5": ((-3.5, 3.5), (-3.5, 3.5)),
}

SYSTEM_SPECS: dict[str, SystemSpec] = {}
for manifest_system in TRANSITION_RICH_BASIN_PARTITION_SYSTEMS:
    bounds = SYSTEM_BOUNDS.get(manifest_system.system_key)
    SYSTEM_SPECS[manifest_system.system_key] = SystemSpec(
        system=manifest_system.system_key,
        title=SYSTEM_TITLES.get(manifest_system.system_key, manifest_system.system_key),
        seed=0,
        xlim=bounds[0] if bounds is not None else None,
        ylim=bounds[1] if bounds is not None else None,
    )

MODEL_SPECS: dict[str, ModelSpec] = {
    ROOT_LISTA_P256: ModelSpec(root_label=ROOT_LISTA_P256, title="LISTA"),
    ROOT_LISTA_BD: ModelSpec(root_label=ROOT_LISTA_BD, title="LISTA-BD"),
    ROOT_LISTA_SB_P256: ModelSpec(root_label=ROOT_LISTA_SB_P256, title="LISTA-SB"),
}
DEFAULT_SYSTEMS = "claude:cal_high_cross_3,claude:cal_asymmetric_3"
DEFAULT_ROOTS = f"{ROOT_LISTA_P256},{ROOT_LISTA_BD}"


@dataclass(frozen=True)
class FamilyRecord:
    family: int
    dominant_basin: int
    count: int
    fraction: float
    basin_fraction: float
    prototype_indices: tuple[int, ...]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_many_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_rows(path))
    return rows


def _parse_csv_paths(value: str | Path) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    return [Path(item.strip()) for item in str(value).split(",") if item.strip()]


def _retained_15_system_keys() -> list[str]:
    return [
        item.system_key
        for item in TRANSITION_RICH_BASIN_PARTITION_SYSTEMS
        if item.system_key not in RETAINED_15_EXCLUDED_SYSTEMS
    ]


def _fixed_17_system_keys() -> list[str]:
    return [item.system_key for item in TRANSITION_RICH_BASIN_PARTITION_SYSTEMS]


def _expand_system_token(token: str) -> list[str]:
    if token == "retained15":
        return _retained_15_system_keys()
    if token == "fixed17":
        return _fixed_17_system_keys()
    return [token]


def _parse_system_specs(value: str) -> list[SystemSpec]:
    specs = []
    seen: set[str] = set()
    keys = []
    for token in [item.strip() for item in value.split(",") if item.strip()]:
        keys.extend(_expand_system_token(token))
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        if key not in SYSTEM_SPECS:
            known = ", ".join(sorted(SYSTEM_SPECS))
            raise ValueError(f"Unknown system {key!r}; known systems: {known}")
        specs.append(SYSTEM_SPECS[key])
    if not specs:
        raise ValueError("At least one system is required")
    return specs


def _parse_model_specs(value: str) -> list[ModelSpec]:
    specs = []
    for root_label in [item.strip() for item in value.split(",") if item.strip()]:
        if root_label not in MODEL_SPECS:
            known = ", ".join(sorted(MODEL_SPECS))
            raise ValueError(f"Unknown root {root_label!r}; known roots: {known}")
        specs.append(MODEL_SPECS[root_label])
    if not specs:
        raise ValueError("At least one model root is required")
    return specs


def _parse_support_definition(value: str | None, *, fallback_topk: int) -> SupportDefinition:
    raw = f"topk:{int(fallback_topk)}" if value is None else value.strip()
    if ":" not in raw:
        raise ValueError(f"Support definition must be scheme:value, got {raw!r}")
    scheme, raw_value = [item.strip() for item in raw.split(":", 1)]
    if scheme in {"abs", "absolute"}:
        threshold = float(raw_value)
        if threshold < 0.0:
            raise ValueError(f"absolute support threshold must be non-negative, got {threshold}")
        return SupportDefinition(
            scheme="absolute",
            value=threshold,
            name=f"absolute:{threshold:.6g}",
            support_display="S_abs",
            family_display="F_abs",
        )
    if scheme == "topk":
        topk = int(raw_value)
        if topk <= 0:
            raise ValueError(f"topk support size must be positive, got {topk}")
        return SupportDefinition(
            scheme="topk",
            value=float(topk),
            name=f"topk:{topk}",
            support_display=f"S_top{topk}",
            family_display=f"F_top{topk}",
        )
    known = "absolute, topk"
    raise ValueError(f"Unknown support definition scheme {scheme!r}; known schemes: {known}")


def _support_mask_from_latents(latents: np.ndarray, definition: SupportDefinition) -> np.ndarray:
    abs_latents = np.abs(latents)
    if definition.scheme == "absolute":
        return abs_latents > float(definition.value)
    if definition.scheme == "topk":
        return _topk_support_mask(latents, int(definition.value))
    raise ValueError(f"Unsupported support scheme {definition.scheme!r}")


def _support_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(np.flatnonzero(mask).astype(int).tolist())


def _binary_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return 1.0 if union <= 0.0 else inter / union


def _support_family_labels_and_prototypes(
    support_mask: np.ndarray,
    *,
    min_jaccard: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Mirror the reducer's greedy Jaccard family construction.

    This version also returns the fixed representative vectors that created the
    families, which are what the manuscript defines as family prototypes.
    """

    if support_mask.ndim != 2:
        raise ValueError("support_mask must have shape [num_states, latent_dim]")

    keys = [_support_key(mask) for mask in support_mask]
    key_counts = Counter(keys)
    key_masks: dict[tuple[int, ...], np.ndarray] = {}
    for key, mask in zip(keys, support_mask):
        if key not in key_masks:
            key_masks[key] = mask.astype(bool, copy=True)

    prototypes: list[np.ndarray] = []
    key_to_family: dict[tuple[int, ...], int] = {}
    for key, _count in sorted(key_counts.items(), key=lambda item: (-item[1], item[0])):
        mask = key_masks[key]
        best_family = None
        best_similarity = -1.0
        for family_id, prototype in enumerate(prototypes):
            similarity = _binary_jaccard(mask, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_family = family_id
        if best_family is not None and best_similarity >= float(min_jaccard):
            key_to_family[key] = best_family
        else:
            key_to_family[key] = len(prototypes)
            prototypes.append(mask)

    labels = np.asarray([key_to_family[key] for key in keys], dtype=np.int64)
    return labels, prototypes


def _family_records(
    family_labels: np.ndarray,
    basin_labels: np.ndarray,
    prototypes: Iterable[np.ndarray],
) -> list[FamilyRecord]:
    total = max(int(family_labels.size), 1)
    basin_totals = Counter(int(item) for item in basin_labels.tolist() if int(item) >= 0)
    records = []
    for family, prototype in enumerate(prototypes):
        keep = family_labels == family
        if not bool(np.any(keep)):
            continue
        basin_counts = Counter(int(item) for item in basin_labels[keep].tolist() if int(item) >= 0)
        dominant_basin = basin_counts.most_common(1)[0][0] if basin_counts else -1
        dominant_basin_count = basin_counts.get(dominant_basin, 0)
        basin_total = basin_totals.get(dominant_basin, 0)
        basin_fraction = 0.0 if basin_total <= 0 else float(dominant_basin_count / basin_total)
        records.append(
            FamilyRecord(
                family=int(family),
                dominant_basin=int(dominant_basin),
                count=int(keep.sum()),
                fraction=float(keep.sum() / total),
                basin_fraction=basin_fraction,
                prototype_indices=tuple(int(i) for i in np.flatnonzero(prototype).tolist()),
            )
        )
    return records


def _select_display_families(
    records: list[FamilyRecord],
    *,
    max_families: int,
    min_per_basin: int,
) -> list[FamilyRecord]:
    by_basin: dict[int, list[FamilyRecord]] = defaultdict(list)
    for record in records:
        by_basin[record.dominant_basin].append(record)
    for basin_records in by_basin.values():
        basin_records.sort(key=lambda record: (-record.count, record.family))

    selected: list[FamilyRecord] = []
    seen: set[int] = set()
    for basin in sorted(by_basin):
        for record in by_basin[basin][:min_per_basin]:
            selected.append(record)
            seen.add(record.family)

    remaining = sorted(
        (record for record in records if record.family not in seen),
        key=lambda record: (-record.count, record.dominant_basin, record.family),
    )
    for record in remaining:
        if len(selected) >= max_families:
            break
        selected.append(record)

    return sorted(selected, key=lambda record: (record.dominant_basin, -record.count, record.family))


def _system_slug(system: str) -> str:
    return system.replace(":", "_").replace("/", "_")


def _bounds_for_system(
    env: Any,
    *,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float], str]:
    if xlim is not None and ylim is not None:
        return xlim, ylim, "configured"

    base_env = getattr(env, "unwrapped", env)
    try:
        low, high = _infer_reset_bounds(base_env)
        return (
            (float(low[0]), float(high[0])),
            (float(low[1]), float(high[1])),
            "inferred_reset_bounds",
        )
    except Exception:
        init_range = float(getattr(base_env, "init_range", 3.5))
        return (-init_range, init_range), (-init_range, init_range), "init_range_fallback"


def _states_and_basin_labels(
    env: Any,
    *,
    system: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    grid_points: int,
    endpoint_rollout_steps: int,
    subset: str,
    depth_slice_mode: str,
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
) -> tuple[torch.Tensor, np.ndarray, dict[str, Any]]:
    if subset == "all":
        xlim_resolved, ylim_resolved, bounds_source = _bounds_for_system(
            env,
            xlim=xlim,
            ylim=ylim,
        )
        _xx, _yy, states = _grid_states(xlim_resolved, ylim_resolved, grid_points)
        basin_labels_t, basin_source = _basin_labels_for_states(
            env,
            system,
            states,
            endpoint_rollout_steps=endpoint_rollout_steps,
        )
        basin_labels = basin_labels_t.numpy()
        return states, basin_labels, {
            "state_source": "grid",
            "subset": "all",
            "depth_slice_mode": None,
            "grid_points": int(grid_points),
            "num_trajectories": None,
            "trajectory_length": None,
            "eval_seed": None,
            "xlim": [float(xlim_resolved[0]), float(xlim_resolved[1])],
            "ylim": [float(ylim_resolved[0]), float(ylim_resolved[1])],
            "bounds_source": bounds_source,
            "basin_label_source": basin_source,
            "state_count_before_subset": int(states.shape[0]),
            "state_count": int(states.shape[0]),
        }

    trajectories = _generate_observation_trajectories(
        env,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        eval_seed=eval_seed,
    )
    basin_labels_t, centers, basin_source = _label_sequences_and_centers(
        env,
        trajectories,
        system_key=system,
        endpoint_rollout_steps=endpoint_rollout_steps,
    )
    subset_masks = _margin_subsets(
        trajectories,
        centers,
        basin_labels=basin_labels_t.cpu().numpy() if depth_slice_mode == "per_basin" else None,
        depth_slice_mode=depth_slice_mode,
    )
    if subset not in subset_masks:
        known = ", ".join(sorted(subset_masks))
        raise ValueError(f"Unknown subset {subset!r}; known subsets: {known}")
    flat_states = trajectories.reshape(-1, trajectories.shape[-1])
    flat_labels = basin_labels_t.cpu().numpy().reshape(-1)
    keep = np.asarray(subset_masks[subset], dtype=bool) & (flat_labels >= 0)
    if not bool(np.any(keep)):
        raise ValueError(f"Subset {subset!r} is empty for {system!r}")
    states = flat_states[torch.as_tensor(keep, dtype=torch.bool)]
    basin_labels = flat_labels[keep]
    return states, basin_labels, {
        "state_source": "observation_trajectories",
        "subset": subset,
        "depth_slice_mode": depth_slice_mode,
        "grid_points": None,
        "num_trajectories": int(num_trajectories),
        "trajectory_length": int(trajectory_length),
        "eval_seed": int(eval_seed),
        "xlim": None,
        "ylim": None,
        "bounds_source": None,
        "basin_label_source": basin_source,
        "state_count_before_subset": int(flat_states.shape[0]),
        "state_count": int(states.shape[0]),
    }


def _collect_system(
    *,
    rows: list[dict[str, str]],
    system: str,
    seed: int,
    title: str,
    model_title: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    root_label: str,
    device: str,
    grid_points: int,
    endpoint_rollout_steps: int,
    family_jaccard: float,
    support_definition: SupportDefinition,
    subset: str,
    depth_slice_mode: str,
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
    max_families: int,
    min_per_basin: int,
) -> dict[str, Any]:
    run_dir = _find_run_dir(rows, root_label=root_label, system=system, seed=seed)
    model, env = _load_model_and_env(run_dir, system, device)
    states, basin_labels, state_metadata = _states_and_basin_labels(
        env,
        system=system,
        xlim=xlim,
        ylim=ylim,
        grid_points=grid_points,
        endpoint_rollout_steps=endpoint_rollout_steps,
        subset=subset,
        depth_slice_mode=depth_slice_mode,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        eval_seed=eval_seed,
    )
    latents = _encode_latents(model, states, device)
    support_mask = _support_mask_from_latents(latents, support_definition)
    family_labels, prototypes = _support_family_labels_and_prototypes(
        support_mask,
        min_jaccard=family_jaccard,
    )
    records = _family_records(family_labels, basin_labels, prototypes)
    display_records = _select_display_families(
        records,
        max_families=max_families,
        min_per_basin=min_per_basin,
    )
    displayed_families = {record.family for record in display_records}
    displayed_coverage = float(
        np.mean(np.asarray([item in displayed_families for item in family_labels.tolist()], dtype=bool))
    )
    return {
        "system": system,
        "title": title,
        "model_title": model_title,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "root_label": root_label,
        "support_definition": support_definition.name,
        "support_rule_display": support_definition.support_display,
        "family_rule_display": support_definition.family_display,
        "family_jaccard": float(family_jaccard),
        **state_metadata,
        "latent_dim": int(support_mask.shape[1]),
        "family_count": int(len(records)),
        "displayed_family_count": int(len(display_records)),
        "displayed_grid_fraction": displayed_coverage,
        "families": [
            {
                "family": record.family,
                "dominant_basin": record.dominant_basin,
                "count": record.count,
                "fraction": record.fraction,
                "basin_fraction": record.basin_fraction,
                "prototype_indices": list(record.prototype_indices),
                "displayed": record.family in displayed_families,
            }
            for record in records
        ],
        "display_records": display_records,
    }


def _plot_codebook(
    systems: list[dict[str, Any]],
    *,
    output: Path,
    output_png: Path | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    basin_colors = {idx: OKABE_ITO[idx % len(OKABE_ITO)] for idx in range(20)}
    model_order = list(dict.fromkeys(str(system["model_title"]) for system in systems))
    system_order = list(dict.fromkeys(str(system["system"]) for system in systems))
    nrows = len(model_order)
    ncols = len(system_order)
    panel_lookup = {
        (str(system["model_title"]), str(system["system"])): system
        for system in systems
    }

    max_rows = max(len(system["display_records"]) for system in systems)
    fig_height = max(2.5, 0.9 + 0.29 * max_rows * nrows)
    fig_width = 7.25
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=False,
    )

    for row_index, model_title in enumerate(model_order):
        for col_index, system_key in enumerate(system_order):
            ax = axes[row_index, col_index]
            system = panel_lookup[(model_title, system_key)]
            records: list[FamilyRecord] = system["display_records"]
            latent_dim = int(system["latent_dim"])
            y_positions = np.arange(len(records), dtype=float)
            tick_step = 8 if latent_dim <= 80 else 32

            ax.set_facecolor("#fbfbfb")
            for start in range(0, latent_dim, 64):
                if (start // 64) % 2 == 1:
                    ax.axvspan(
                        start - 0.5,
                        min(start + 64, latent_dim) - 0.5,
                        color="#f1f3f4",
                        alpha=0.72,
                        linewidth=0,
                        zorder=0,
                    )

            current_basin = None
            group_start = 0
            sentinel = FamilyRecord(-1, -999, 0, 0.0, 0.0, tuple())
            for family_row, record in enumerate(records + [sentinel]):
                if current_basin is None:
                    current_basin = record.dominant_basin
                if record.dominant_basin != current_basin:
                    color = basin_colors.get(current_basin, "#5f6368")
                    ax.axhspan(
                        group_start - 0.5,
                        family_row - 0.5,
                        color=color,
                        alpha=0.055,
                        linewidth=0,
                        zorder=0,
                    )
                    ax.axhline(family_row - 0.5, color="#c8ccd2", linewidth=0.5, zorder=1)
                    current_basin = record.dominant_basin
                    group_start = family_row

            for tick in range(0, latent_dim + 1, tick_step):
                ax.axvline(tick - 0.5, color="#dfe3e8", linewidth=0.4, zorder=1)

            for family_row, record in enumerate(records):
                color = basin_colors.get(record.dominant_basin, "#5f6368")
                alpha = 0.28 + 0.72 * np.sqrt(np.clip(record.basin_fraction, 0.0, 1.0))
                ax.vlines(
                    list(record.prototype_indices),
                    family_row - 0.22,
                    family_row + 0.22,
                    colors=color,
                    alpha=alpha,
                    linewidth=1.15,
                    zorder=3,
                )
                ax.vlines(
                    list(record.prototype_indices),
                    family_row - 0.22,
                    family_row + 0.22,
                    colors="#202124",
                    alpha=min(0.36, 0.55 * alpha),
                    linewidth=0.22,
                    zorder=4,
                )

            y_labels = [
                f"B{record.dominant_basin} F{record.family} {100.0 * record.basin_fraction:.0f}%"
                for record in records
            ]
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels)
            ax.set_xlim(-0.8, latent_dim - 0.2)
            ax.set_ylim(len(records) - 0.5, -0.5)
            ax.set_xticks(np.arange(0, latent_dim, tick_step))
            ax.tick_params(axis="x", length=2.0, pad=1.3)
            ax.tick_params(axis="y", length=0.0, pad=2.0)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#c8ccd2")
            if row_index == nrows - 1:
                ax.set_xlabel("latent coordinate index")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            ax.set_title(
                f"{model_title}  |  {system['title']}  |  {system['family_rule_display']}",
                loc="left",
                fontsize=8.8,
                pad=6,
            )

    bottom = 0.19 if nrows == 1 else 0.13
    fig.subplots_adjust(left=0.125, right=0.99, top=0.95, bottom=bottom, hspace=0.30, wspace=0.24)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _collect_panels(
    *,
    rows: list[dict[str, str]],
    system_specs: list[SystemSpec],
    model_specs: list[ModelSpec],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    systems = []
    for model_spec in model_specs:
        for system_spec in system_specs:
            print(
                f"Collecting support-index codebook: {model_spec.title} / {system_spec.system}",
                flush=True,
            )
            systems.append(
                _collect_system(
                    rows=rows,
                    system=system_spec.system,
                    seed=system_spec.seed,
                    title=system_spec.title,
                    model_title=model_spec.title,
                    xlim=system_spec.xlim,
                    ylim=system_spec.ylim,
                    root_label=model_spec.root_label,
                    device=args.device,
                    grid_points=args.grid_points,
                    endpoint_rollout_steps=args.endpoint_rollout_steps,
                    family_jaccard=args.family_jaccard,
                    support_definition=args.resolved_support_definition,
                    subset=args.subset,
                    depth_slice_mode=args.depth_slice_mode,
                    num_trajectories=args.num_trajectories,
                    trajectory_length=args.trajectory_length,
                    eval_seed=args.eval_seed,
                    max_families=args.max_families_per_system,
                    min_per_basin=args.min_families_per_basin,
                )
            )
            gc.collect()
    return systems


def _metadata_for_figure(
    *,
    output: Path,
    output_png: Path | None,
    csv_paths: list[Path],
    model_specs: list[ModelSpec],
    systems: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "output_pdf": str(output),
        "output_png": str(output_png) if output_png is not None else None,
        "interpretability_csvs": [str(path) for path in csv_paths],
        "root_labels": [model_spec.root_label for model_spec in model_specs],
        "support_definition": args.resolved_support_definition.name,
        "support_rule_display": args.resolved_support_definition.support_display,
        "family_rule_display": args.resolved_support_definition.family_display,
        "subset": args.subset,
        "depth_slice_mode": args.depth_slice_mode if args.subset != "all" else None,
        "family_jaccard": float(args.family_jaccard),
        "description": (
            "Rows are learned support families for matched dz=256 models. "
            "Each row shows the fixed prototype active-coordinate indices "
            "created by the greedy Jaccard family construction. Dominant basin "
            "labels are used only for post-hoc coloring and annotation; mark "
            "opacity encodes within-basin family coverage."
        ),
        "systems": [
            {
                key: value
                for key, value in system.items()
                if key != "display_records"
            }
            for system in systems
        ],
    }


def _write_figure_and_metadata(
    *,
    systems: list[dict[str, Any]],
    output: Path,
    output_png: Path | None,
    metadata_output: Path,
    csv_paths: list[Path],
    model_specs: list[ModelSpec],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if output_png is None and output.suffix.lower() == ".pdf":
        output_png = output.with_suffix(".png")
    _plot_codebook(systems, output=output, output_png=output_png)
    metadata = _metadata_for_figure(
        output=output,
        output_png=output_png,
        csv_paths=csv_paths,
        model_specs=model_specs,
        systems=systems,
        args=args,
    )
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_figure(args: argparse.Namespace) -> dict[str, Any]:
    csv_paths = _parse_csv_paths(args.interpretability_csv)
    rows = _read_many_rows(csv_paths)
    system_specs = _parse_system_specs(args.systems)
    model_specs = _parse_model_specs(args.roots)
    args.resolved_support_definition = _parse_support_definition(
        args.support_definition,
        fallback_topk=args.topk,
    )

    if args.batch_output_dir is not None:
        output_dir = args.batch_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_output = (
            output_dir / "manifest.json"
            if args.metadata_output == DEFAULT_METADATA_OUTPUT
            else args.metadata_output
        )
        figures = []
        for system_spec in system_specs:
            slug = _system_slug(system_spec.system)
            output = output_dir / f"support_family_index_codebook_{slug}.pdf"
            metadata_output = output_dir / f"support_family_index_codebook_{slug}.json"
            systems = _collect_panels(
                rows=rows,
                system_specs=[system_spec],
                model_specs=model_specs,
                args=args,
            )
            metadata = _write_figure_and_metadata(
                systems=systems,
                output=output,
                output_png=output.with_suffix(".png"),
                metadata_output=metadata_output,
                csv_paths=csv_paths,
                model_specs=model_specs,
                args=args,
            )
            figures.append(
                {
                    "system": system_spec.system,
                    "title": system_spec.title,
                    "output_pdf": str(output),
                    "output_png": str(output.with_suffix(".png")),
                    "metadata": str(metadata_output),
                    "panels": [
                        {
                            "model_title": panel["model_title"],
                            "family_count": panel["family_count"],
                            "displayed_family_count": panel["displayed_family_count"],
                            "displayed_grid_fraction": panel["displayed_grid_fraction"],
                            "basin_label_source": panel["basin_label_source"],
                            "bounds_source": panel["bounds_source"],
                            "state_source": panel["state_source"],
                            "subset": panel["subset"],
                            "depth_slice_mode": panel["depth_slice_mode"],
                            "state_count": panel["state_count"],
                        }
                        for panel in metadata["systems"]
                    ],
                }
            )
        manifest = {
            "output_dir": str(output_dir),
            "manifest_output": str(manifest_output),
            "systems_requested": [spec.system for spec in system_specs],
            "system_count": len(system_specs),
            "root_labels": [spec.root_label for spec in model_specs],
            "support_definition": args.resolved_support_definition.name,
            "support_rule_display": args.resolved_support_definition.support_display,
            "family_rule_display": args.resolved_support_definition.family_display,
            "subset": args.subset,
            "depth_slice_mode": args.depth_slice_mode if args.subset != "all" else None,
            "family_jaccard": float(args.family_jaccard),
            "figures": figures,
        }
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {
            "output_pdf": None,
            "output_png": None,
            "output_dir": str(output_dir),
            "manifest_output": str(manifest_output),
            "system_count": len(system_specs),
        }

    systems = _collect_panels(
        rows=rows,
        system_specs=system_specs,
        model_specs=model_specs,
        args=args,
    )
    output_png = args.output_png
    return _write_figure_and_metadata(
        systems=systems,
        output=args.output,
        output_png=output_png,
        metadata_output=args.metadata_output,
        csv_paths=csv_paths,
        model_specs=model_specs,
        args=args,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpretability-csv",
        default=",".join(str(path) for path in DEFAULT_INTERPRETABILITY_CSVS),
        help="comma-separated interpretability CSV files to search for run directories",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-png", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument(
        "--batch-output-dir",
        type=Path,
        default=None,
        help="if set, write one two-model codebook figure per selected system into this directory",
    )
    parser.add_argument(
        "--roots",
        default=DEFAULT_ROOTS,
        help="comma-separated model root labels to render",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-points", type=int, default=84)
    parser.add_argument("--num-trajectories", type=int, default=128)
    parser.add_argument("--trajectory-length", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--endpoint-rollout-steps", type=int, default=360)
    parser.add_argument("--family-jaccard", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--support-definition",
        default=None,
        help="support rule as scheme:value, e.g. topk:8 or absolute:0.001; defaults to topk:<--topk>",
    )
    parser.add_argument("--max-families-per-system", type=int, default=6)
    parser.add_argument("--min-families-per-basin", type=int, default=1)
    parser.add_argument(
        "--subset",
        choices=["all", "deep", "boundary"],
        default="all",
        help="state subset used to build support families; all uses a plotting grid, deep/boundary use generated observation trajectories",
    )
    parser.add_argument(
        "--depth-slice-mode",
        choices=["global", "per_basin"],
        default="global",
        help="deep/boundary subset convention for trajectory-based subsets",
    )
    parser.add_argument(
        "--systems",
        default=DEFAULT_SYSTEMS,
        help="comma-separated system keys to render; aliases: retained15, fixed17",
    )
    return parser.parse_args()


def main() -> None:
    metadata = build_figure(parse_args())
    print(
        json.dumps(
            {
                key: metadata.get(key)
                for key in ("output_pdf", "output_png", "output_dir", "manifest_output", "system_count")
                if key in metadata
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
