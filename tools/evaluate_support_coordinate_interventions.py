#!/usr/bin/env python3
"""Evaluate initial-latent coordinate interventions on sparse supports.

This is a focused ablation for separating the selected sparse coordinates from
the numerical coefficient values at a fixed initial state.  It uses one trained
checkpoint, selects basin-interior held-out states, perturbs only the initial
latent code, and then runs the model's usual autonomous latent rollout.

Interventions:

1. Coordinate dropping: zero the largest-magnitude active coordinate in
   S_abs(x0), then the top two, and so on up to --max_drop.
2. Random support: keep the active coefficient values from S_abs(x0), but move
   them to randomly chosen inactive latent coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from skae.benchmarks.controlled_alignment import (
    _assign_nearest_centers,
    _generate_trajectories,
    _label_sequences_and_centers,
    _load_checkpoint_model,
    tie_inclusive_high_center_margin_mask,
)
from tools.reduce_transition_rich_interpretability_metrics import _load_latest_specs

PAPER_PALETTE = {
    "baseline": "#111111",
    "drop_top_1": "#0072B2",
    "drop_top_2": "#56B4E9",
    "drop_top_3": "#009E73",
    "drop_top_5": "#D55E00",
    "drop_top_10": "#CC79A7",
    "random": "#D55E00",
    "random_dark": "#882255",
}


def _configure_paper_style() -> None:
    """Match the matplotlib style used by the NeurIPS paper figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


DEFAULT_HORIZONS = "1,3,5,7,9,11,13,15,17,19,21"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True, help="forecasting_rows.csv used to discover the checkpoint")
    parser.add_argument("--output_dir", required=True, help="directory for CSV, JSON, and plot artifacts")
    parser.add_argument(
        "--root_label",
        default="lista_dense_signsplit_p256_hardinit_basin_partition",
        help="single model root label to evaluate",
    )
    parser.add_argument("--system", default="gated_local_linear", help="single system_key to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="single training seed to evaluate")
    parser.add_argument("--num_initial_points", type=int, default=15)
    parser.add_argument("--num_candidate_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=64)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--support_definition",
        default="absolute:0.001",
        help="support rule used to define the initial active set",
    )
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--max_drop", type=int, default=10)
    parser.add_argument("--random_support_repeats", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument(
        "--depth_slice_mode",
        choices=["global", "per_basin"],
        default="per_basin",
        help="candidate initial states are sampled from this deep-slice rule",
    )
    parser.add_argument(
        "--require_stable_true_basin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="only select starts whose true trajectory stays in the initial basin through max horizon",
    )
    parser.add_argument("--plot_format", default="pdf,png", help="comma-separated plot extensions")
    return parser.parse_args()


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_support_definition(raw: str) -> Tuple[str, float]:
    if ":" not in raw:
        raise ValueError(f"support definition must have form scheme:value, got {raw!r}")
    scheme, value_raw = raw.split(":", 1)
    scheme = scheme.strip()
    if scheme not in {"absolute", "relative", "topk"}:
        raise ValueError(f"unsupported support scheme {scheme!r}")
    return scheme, float(value_raw)


def _support_mask(latents: np.ndarray, *, scheme: str, value: float) -> np.ndarray:
    magnitudes = np.abs(latents)
    if scheme == "absolute":
        return magnitudes > float(value)
    if scheme == "relative":
        maxima = magnitudes.max(axis=-1, keepdims=True)
        return magnitudes > float(value) * np.maximum(maxima, 1e-12)
    if scheme == "topk":
        k = int(value)
        if k <= 0:
            raise ValueError("top-k support size must be positive")
        if k >= magnitudes.shape[-1]:
            return np.ones_like(magnitudes, dtype=bool)
        indices = np.argpartition(
            magnitudes, kth=magnitudes.shape[-1] - k, axis=-1
        )[..., -k:]
        mask = np.zeros_like(magnitudes, dtype=bool)
        np.put_along_axis(mask, indices, True, axis=-1)
        return mask
    raise ValueError(f"unsupported support scheme {scheme!r}")


def _high_center_margin_mask(
    states: torch.Tensor,
    centers: torch.Tensor,
    labels: np.ndarray,
    scope: str,
) -> np.ndarray:
    if scope == "per_basin":
        return tie_inclusive_high_center_margin_mask(states, centers, labels)
    if scope != "global":
        raise ValueError(f"unsupported center-margin scope {scope!r}")
    flat = states.reshape(-1, states.shape[-1])
    distances = torch.cdist(flat, centers.to(dtype=flat.dtype))
    if distances.shape[1] < 2:
        return np.ones(flat.shape[0], dtype=bool)
    nearest = torch.topk(distances, k=2, largest=False, dim=1).values
    margins = (nearest[:, 1] - nearest[:, 0]).cpu().numpy()
    return margins >= float(np.quantile(margins, 0.75))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _latest_spec(rows_csv: Path, root_label: str, system: str, seed: int):
    specs = _load_latest_specs(
        rows_csv,
        root_labels=[root_label],
        systems=[system],
        seeds=[seed],
    )
    if len(specs) != 1:
        raise ValueError(
            f"Expected exactly one spec for root={root_label}, system={system}, seed={seed}; found {len(specs)}"
        )
    return specs[0]


def _select_initial_indices(
    *,
    trajectories: torch.Tensor,
    basin_labels: np.ndarray,
    centers: torch.Tensor,
    subset_mask: np.ndarray,
    max_horizon: int,
    num_points: int,
    rng: np.random.Generator,
    require_stable_true_basin: bool,
) -> np.ndarray:
    if trajectories.shape[1] <= max_horizon:
        raise ValueError(
            f"trajectory_length={trajectories.shape[1]} must exceed max horizon {max_horizon}"
        )

    valid = subset_mask.reshape(basin_labels.shape).copy()
    valid[:, trajectories.shape[1] - max_horizon :] = False
    if require_stable_true_basin:
        start_basins = basin_labels[:, :-max_horizon]
        stable = np.ones_like(start_basins, dtype=bool)
        for step in range(1, max_horizon + 1):
            stable &= basin_labels[:, step : step + start_basins.shape[1]] == start_basins
        valid[:, :-max_horizon] &= stable

    candidate_pairs = np.argwhere(valid)
    if candidate_pairs.shape[0] < num_points:
        raise RuntimeError(
            f"Only {candidate_pairs.shape[0]} valid candidate starts for {num_points} requested "
            f"(stable={require_stable_true_basin}, max_horizon={max_horizon})."
        )

    by_basin: Dict[int, List[np.ndarray]] = defaultdict(list)
    for pair in candidate_pairs:
        by_basin[int(basin_labels[int(pair[0]), int(pair[1])])].append(pair)

    selected: List[np.ndarray] = []
    basin_ids = sorted(by_basin)
    if basin_ids:
        base = num_points // len(basin_ids)
        rem = num_points % len(basin_ids)
        used = set()
        for offset, basin in enumerate(basin_ids):
            target = base + (1 if offset < rem else 0)
            pool = np.asarray(by_basin[basin], dtype=np.int64)
            if pool.shape[0] == 0 or target <= 0:
                continue
            take = min(target, pool.shape[0])
            chosen = rng.choice(pool.shape[0], size=take, replace=False)
            for item in pool[np.sort(chosen)]:
                key = (int(item[0]), int(item[1]))
                used.add(key)
                selected.append(item)

        if len(selected) < num_points:
            remaining = [
                pair
                for pair in candidate_pairs
                if (int(pair[0]), int(pair[1])) not in used
            ]
            remaining_arr = np.asarray(remaining, dtype=np.int64)
            need = num_points - len(selected)
            chosen = rng.choice(remaining_arr.shape[0], size=need, replace=False)
            selected.extend(remaining_arr[np.sort(chosen)])

    if len(selected) < num_points:
        raise RuntimeError(
            f"Could only select {len(selected)} starts after basin balancing; requested {num_points}"
        )

    selected_arr = np.asarray(selected[:num_points], dtype=np.int64)
    order = np.lexsort((selected_arr[:, 1], selected_arr[:, 0]))
    return selected_arr[order]


def _rollout_decode(model, z0: torch.Tensor, *, max_horizon: int) -> torch.Tensor:
    z = z0
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for _step in range(max_horizon):
            z = model.step_latent(z)
            preds.append(model.decode(z).detach().cpu())
    return torch.stack(preds, dim=1)


def _label_states(env, states: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    if hasattr(env, "basin_label"):
        flat = states.reshape(-1, states.shape[-1])
        labels = env.basin_label(flat)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        return labels.reshape(states.shape[:-1]).to(dtype=torch.long)
    return _assign_nearest_centers(states, centers)


def _top_active_indices(z: np.ndarray, active_mask: np.ndarray) -> List[List[int]]:
    out: List[List[int]] = []
    for row, mask in zip(z, active_mask):
        active = np.flatnonzero(mask)
        order = active[np.argsort(-np.abs(row[active]))]
        out.append([int(item) for item in order.tolist()])
    return out


def _make_drop_z(z0: torch.Tensor, top_indices: Sequence[Sequence[int]], drop_count: int) -> torch.Tensor:
    z_mod = z0.clone()
    for row_idx, ordered in enumerate(top_indices):
        for coord in ordered[:drop_count]:
            z_mod[row_idx, int(coord)] = 0.0
    return z_mod


def _make_random_support_z(
    z0: torch.Tensor,
    active_mask: np.ndarray,
    *,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    z_np = z0.detach().cpu().numpy()
    z_mod = np.zeros_like(z_np)
    moves: List[Dict[str, object]] = []
    dim = z_np.shape[1]
    all_indices = np.arange(dim)
    for row_idx, mask in enumerate(active_mask):
        src = np.flatnonzero(mask)
        values = z_np[row_idx, src].copy()
        rng.shuffle(values)
        inactive = np.flatnonzero(~mask)
        if inactive.size >= src.size:
            dst = rng.choice(inactive, size=src.size, replace=False)
        else:
            # This fallback is unlikely for the sparse LISTA runs, but keeps the
            # intervention defined for dense supports by minimizing exact reuse.
            dst = rng.permutation(all_indices)[: src.size]
        z_mod[row_idx, dst] = values
        moves.append(
            {
                "point_id": int(row_idx),
                "source_indices": " ".join(str(int(item)) for item in src.tolist()),
                "destination_indices": " ".join(str(int(item)) for item in dst.tolist()),
            }
        )
    return torch.from_numpy(z_mod).to(device=z0.device, dtype=z0.dtype), moves


def _condition_metrics(
    *,
    condition: str,
    intervention_type: str,
    drop_count: Optional[int],
    random_repeat: Optional[int],
    preds: torch.Tensor,
    true_future: torch.Tensor,
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    initial_basins: np.ndarray,
    horizons: Sequence[int],
    metadata: Dict[str, object],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    err = ((preds - true_future.cpu()) ** 2).mean(dim=-1).numpy()
    pred_np = pred_labels.cpu().numpy()
    true_np = true_labels.cpu().numpy()

    horizon_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []
    for horizon in horizons:
        h = int(horizon)
        idx = h - 1
        at_h = err[:, idx]
        cum_sum = err[:, :h].sum(axis=1)
        cum_mean = err[:, :h].mean(axis=1)
        mismatch_true = pred_np[:, idx] != true_np[:, idx]
        mismatch_initial = pred_np[:, idx] != initial_basins
        any_mismatch_true = np.any(pred_np[:, :h] != true_np[:, :h], axis=1)
        row = {
            **metadata,
            "condition": condition,
            "intervention_type": intervention_type,
            "drop_count": "" if drop_count is None else int(drop_count),
            "random_repeat": "" if random_repeat is None else int(random_repeat),
            "horizon": h,
            "num_initial_points": int(err.shape[0]),
            "mse_at_h_mean": float(np.mean(at_h)),
            "mse_at_h_median": float(np.median(at_h)),
            "cumulative_mse_sum_mean": float(np.mean(cum_sum)),
            "cumulative_mse_sum_median": float(np.median(cum_sum)),
            "cumulative_mse_mean_mean": float(np.mean(cum_mean)),
            "basin_mismatch_vs_true_future_fraction": float(np.mean(mismatch_true)),
            "basin_mismatch_vs_initial_fraction": float(np.mean(mismatch_initial)),
            "any_basin_mismatch_vs_true_future_fraction": float(np.mean(any_mismatch_true)),
        }
        horizon_rows.append(row)

        for point_id in range(err.shape[0]):
            point_rows.append(
                {
                    **metadata,
                    "condition": condition,
                    "intervention_type": intervention_type,
                    "drop_count": "" if drop_count is None else int(drop_count),
                    "random_repeat": "" if random_repeat is None else int(random_repeat),
                    "point_id": int(point_id),
                    "horizon": h,
                    "mse_at_h": float(at_h[point_id]),
                    "cumulative_mse_sum": float(cum_sum[point_id]),
                    "cumulative_mse_mean": float(cum_mean[point_id]),
                    "initial_basin": int(initial_basins[point_id]),
                    "true_basin_at_h": int(true_np[point_id, idx]),
                    "pred_basin_at_h": int(pred_np[point_id, idx]),
                    "basin_mismatch_vs_true_future": bool(mismatch_true[point_id]),
                    "basin_mismatch_vs_initial": bool(mismatch_initial[point_id]),
                }
            )
    return horizon_rows, point_rows


def _plot_metric_curves(
    rows: Sequence[Dict[str, object]],
    *,
    output_dir: Path,
    plot_formats: Sequence[str],
    metric: str,
    ylabel: str,
    filename_stem: str,
    log_y: bool,
) -> None:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for condition in sorted(grouped, key=_condition_sort_key):
        sub = sorted(grouped[condition], key=lambda item: int(item["horizon"]))
        x = [int(item["horizon"]) for item in sub]
        y = [float(item[metric]) for item in sub]
        style = _condition_style(condition)
        ax.plot(x, y, marker=style["marker"], lw=style["lw"], alpha=style["alpha"], label=condition)
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({int(row["horizon"]) for row in rows}))
    if log_y:
        positives = [float(row[metric]) for row in rows if float(row[metric]) > 0.0]
        if positives:
            ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in plot_formats:
        fig.savefig(output_dir / f"{filename_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _baseline_curve(
    rows: Sequence[Dict[str, object]],
    *,
    metric: str,
) -> Tuple[List[int], List[float]]:
    pairs: List[Tuple[int, float]] = []
    for row in rows:
        if row["condition"] != "baseline":
            continue
        value = float(row[metric])
        if math.isfinite(value):
            pairs.append((int(row["horizon"]), value))
    pairs.sort(key=lambda item: item[0])
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _plot_drop_absolute_curves(
    rows: Sequence[Dict[str, object]],
    *,
    output_dir: Path,
    plot_formats: Sequence[str],
    metric: str,
    ylabel: str,
    filename_stem: str,
    title: str,
    drop_counts: Sequence[int] = (1, 2, 3, 5, 10),
    log_y: bool = False,
) -> None:
    base_x, base_y = _baseline_curve(rows, metric=metric)
    row_by_condition_horizon = {
        (str(row["condition"]), int(row["horizon"])): row
        for row in rows
    }
    horizons = base_x
    if not horizons:
        return

    fig, ax = plt.subplots(figsize=(4.65, 3.15), constrained_layout=True)
    ax.plot(
        base_x,
        base_y,
        marker="o",
        ms=4.4,
        lw=2.0,
        color=PAPER_PALETTE["baseline"],
        label="standard rollout",
        zorder=4,
    )
    for drop_count in drop_counts:
        condition = f"drop_top_{drop_count}"
        x: List[int] = []
        y: List[float] = []
        for horizon in horizons:
            row = row_by_condition_horizon.get((condition, horizon))
            if row is None:
                continue
            value = float(row[metric])
            if math.isfinite(value):
                x.append(horizon)
                y.append(value)
        if x:
            ax.plot(
                x,
                y,
                marker="o",
                ms=3.7,
                lw=1.55,
                color=PAPER_PALETTE.get(condition),
                label=f"drop top-{drop_count}",
            )
    ax.set_xlabel(r"Rollout horizon $H$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(horizons)
    positives = [float(v) for line in ax.lines for v in line.get_ydata() if float(v) > 0.0]
    if log_y and positives:
        ax.set_yscale("log")
    ax.grid(True, which="both", lw=0.45, alpha=0.38)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        borderaxespad=0.0,
        ncol=3,
        handlelength=1.6,
        columnspacing=0.95,
        handletextpad=0.45,
    )
    for ext in plot_formats:
        fig.savefig(output_dir / f"{filename_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    bootstrap_reps: int = 2000,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    center = float(np.mean(arr))
    if arr.size == 1 or bootstrap_reps <= 0:
        return center, center, center
    indices = rng.integers(0, arr.size, size=(bootstrap_reps, arr.size))
    boot = arr[indices].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return center, float(lo), float(hi)


def _plot_drop_absolute_curves_with_bands(
    point_rows: Sequence[Dict[str, object]],
    *,
    output_dir: Path,
    plot_formats: Sequence[str],
    point_metric: str,
    ylabel: str,
    filename_stem: str,
    title: str,
    drop_counts: Sequence[int] = (1, 2, 3, 5, 10),
    bootstrap_reps: int = 2000,
    ci_level: float = 0.95,
    log_y: bool = False,
) -> None:
    conditions = ["baseline", *[f"drop_top_{drop_count}" for drop_count in drop_counts]]
    values: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in point_rows:
        condition = str(row["condition"])
        if condition not in conditions:
            continue
        value = float(row[point_metric])
        if math.isfinite(value):
            values[(condition, int(row["horizon"]))].append(value)

    horizons = sorted({horizon for condition, horizon in values if condition == "baseline"})
    if not horizons:
        return

    fig, ax = plt.subplots(figsize=(4.65, 3.15), constrained_layout=True)
    rng = np.random.default_rng(20260506)
    for condition in conditions:
        x: List[int] = []
        y: List[float] = []
        lo: List[float] = []
        hi: List[float] = []
        for horizon in horizons:
            sample = values.get((condition, horizon), [])
            if not sample:
                continue
            center, lower, upper = _bootstrap_mean_ci(
                sample,
                rng=rng,
                bootstrap_reps=bootstrap_reps,
                ci_level=ci_level,
            )
            if math.isfinite(center):
                x.append(horizon)
                y.append(center)
                lo.append(lower)
                hi.append(upper)
        if not x:
            continue
        color = PAPER_PALETTE["baseline"] if condition == "baseline" else PAPER_PALETTE.get(condition)
        label = "standard rollout" if condition == "baseline" else condition.replace("_", " ").replace("top ", "top-")
        zorder = 4 if condition == "baseline" else 3
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0, zorder=zorder - 1)
        ax.plot(
            x,
            y,
            marker="o",
            ms=4.4 if condition == "baseline" else 3.7,
            lw=2.0 if condition == "baseline" else 1.55,
            color=color,
            label=label,
            zorder=zorder,
        )

    ax.set_xlabel(r"Rollout horizon $H$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(horizons)
    positives = [float(v) for line in ax.lines for v in line.get_ydata() if float(v) > 0.0]
    if log_y and positives:
        ax.set_yscale("log")
    ax.grid(True, which="both", lw=0.45, alpha=0.38)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        borderaxespad=0.0,
        ncol=3,
        handlelength=1.6,
        columnspacing=0.95,
        handletextpad=0.45,
    )
    for ext in plot_formats:
        fig.savefig(output_dir / f"{filename_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _plot_random_absolute_band(
    horizon_rows: Sequence[Dict[str, object]],
    point_rows: Sequence[Dict[str, object]],
    *,
    output_dir: Path,
    plot_formats: Sequence[str],
    horizon_metric: str,
    point_metric: str,
    ylabel: str,
    filename_stem: str,
    title: str,
    log_y: bool = False,
) -> None:
    base_x, base_y = _baseline_curve(horizon_rows, metric=horizon_metric)
    values_by_horizon: Dict[int, List[float]] = defaultdict(list)
    for row in point_rows:
        condition = str(row["condition"])
        if not condition.startswith("random_support"):
            continue
        horizon = int(row["horizon"])
        value = float(row[point_metric])
        if math.isfinite(value):
            values_by_horizon[horizon].append(value)

    horizons = base_x
    x = [h for h in horizons if values_by_horizon.get(h)]
    if not x:
        return
    arrs = [np.asarray(values_by_horizon[h], dtype=float) for h in x]
    med = np.asarray([float(np.median(arr)) for arr in arrs])
    mean = np.asarray([float(np.mean(arr)) for arr in arrs])
    q25 = np.asarray([float(np.percentile(arr, 25)) for arr in arrs])
    q75 = np.asarray([float(np.percentile(arr, 75)) for arr in arrs])

    fig, ax = plt.subplots(figsize=(4.65, 3.15), constrained_layout=True)
    ax.plot(
        base_x,
        base_y,
        marker="o",
        ms=4.4,
        lw=2.0,
        color=PAPER_PALETTE["baseline"],
        label="standard rollout",
        zorder=4,
    )
    ax.fill_between(
        x,
        q25,
        q75,
        color=PAPER_PALETTE["random"],
        alpha=0.18,
        lw=0,
        label="random shuffle IQR",
    )
    ax.plot(
        x,
        med,
        marker="o",
        ms=3.9,
        color=PAPER_PALETTE["random"],
        lw=1.8,
        label="random shuffle median",
    )
    ax.plot(
        x,
        mean,
        marker="s",
        ms=3.5,
        color=PAPER_PALETTE["random_dark"],
        lw=1.35,
        ls=":",
        label="random shuffle mean",
    )
    ax.set_xlabel(r"Rollout horizon $H$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(horizons)
    positives = [*q25.tolist(), *q75.tolist(), *med.tolist(), *mean.tolist(), *base_y]
    if log_y and any(value > 0.0 for value in positives):
        ax.set_yscale("log")
    ax.grid(True, which="both", lw=0.45, alpha=0.38)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        borderaxespad=0.0,
        ncol=2,
        handlelength=1.6,
        columnspacing=1.1,
        handletextpad=0.5,
    )
    for ext in plot_formats:
        fig.savefig(output_dir / f"{filename_stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _worst_intervention_condition(
    rows: Sequence[Dict[str, object]],
    *,
    metric: str,
    horizon: int,
) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for row in rows:
        condition = str(row["condition"])
        if int(row["horizon"]) != int(horizon):
            continue
        if condition != "drop_top_10" and not condition.startswith("random_support"):
            continue
        value = float(row[metric])
        if math.isfinite(value):
            candidates.append((value, condition))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _condition_sort_key(condition: str) -> Tuple[int, int, str]:
    if condition == "baseline":
        return (0, 0, condition)
    if condition.startswith("drop_top_"):
        try:
            return (1, int(condition.rsplit("_", 1)[1]), condition)
        except ValueError:
            return (1, 999, condition)
    if condition.startswith("random_support"):
        return (2, 0, condition)
    return (3, 0, condition)


def _condition_style(condition: str) -> Dict[str, object]:
    if condition == "baseline":
        return {"marker": "o", "lw": 2.2, "alpha": 1.0}
    if condition.startswith("random_support"):
        return {"marker": "s", "lw": 2.0, "alpha": 0.9}
    return {"marker": ".", "lw": 1.2, "alpha": 0.75}


def _trajectory_intervention_label(condition: str) -> str:
    if condition.startswith("random_support"):
        return "random shuffled\nsupport rollout"
    if condition.startswith("drop_top_"):
        try:
            count = int(condition.rsplit("_", 1)[1])
            return f"drop top-{count}\nrollout"
        except ValueError:
            pass
    return condition.replace("_", " ")


def _plot_trajectories(
    *,
    output_dir: Path,
    plot_formats: Sequence[str],
    env,
    x0: torch.Tensor,
    true_future: torch.Tensor,
    condition_predictions: Dict[str, torch.Tensor],
    worst_condition: Optional[str],
) -> None:
    if true_future.shape[-1] != 2:
        return

    if "baseline" not in condition_predictions or worst_condition not in condition_predictions:
        return

    plot_steps = min(20, true_future.shape[1])
    x0_np = x0.detach().cpu().numpy()
    true_np = true_future.numpy()
    base_np = condition_predictions["baseline"].numpy()
    worst_np = condition_predictions[worst_condition].numpy()

    true_seq = np.concatenate([x0_np[:, None, :], true_np[:, :plot_steps, :]], axis=1)
    base_seq = np.concatenate([x0_np[:, None, :], base_np[:, :plot_steps, :]], axis=1)
    worst_seq = np.concatenate([x0_np[:, None, :], worst_np[:, :plot_steps, :]], axis=1)

    axis_points = np.concatenate(
        [
            true_seq.reshape(-1, 2),
            base_seq.reshape(-1, 2),
            x0_np.reshape(-1, 2),
        ],
        axis=0,
    )
    finite = np.isfinite(axis_points).all(axis=1)
    axis_points = axis_points[finite]
    if axis_points.size == 0:
        return
    mins = axis_points.min(axis=0)
    maxs = axis_points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    mins = mins - 0.16 * span
    maxs = maxs + 0.16 * span

    xs = np.linspace(mins[0], maxs[0], 23)
    ys = np.linspace(mins[1], maxs[1], 23)
    xx, yy = np.meshgrid(xs, ys)
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        next_grid = env.step(grid).detach().cpu().numpy()
    delta = next_grid - grid.numpy()
    speed = np.linalg.norm(delta, axis=1)
    scale = np.maximum(speed, 1e-8)
    uu = (delta[:, 0] / scale).reshape(xx.shape)
    vv = (delta[:, 1] / scale).reshape(yy.shape)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), constrained_layout=True)
    ax.quiver(
        xx,
        yy,
        uu,
        vv,
        color="0.15",
        alpha=0.16,
        width=0.002,
        headwidth=3,
        headlength=4,
        scale=28,
        zorder=0,
    )
    for point_id in range(true_seq.shape[0]):
        ax.plot(
            true_seq[point_id, :, 0],
            true_seq[point_id, :, 1],
            color="0.55",
            alpha=0.18,
            lw=0.85,
            label="true\ndynamics" if point_id == 0 else None,
            zorder=1,
        )
        ax.plot(
            base_seq[point_id, :, 0],
            base_seq[point_id, :, 1],
            color="0.0",
            alpha=0.82,
            lw=1.15,
            label="standard\nrollout" if point_id == 0 else None,
            zorder=2,
        )
        ax.plot(
            worst_seq[point_id, :, 0],
            worst_seq[point_id, :, 1],
            color="#dc2626",
            alpha=0.52,
            lw=1.0,
            label=_trajectory_intervention_label(worst_condition) if point_id == 0 else None,
            zorder=3,
        )
    ax.scatter(
        x0_np[:, 0],
        x0_np[:, 1],
        s=8,
        color="black",
        alpha=0.5,
        zorder=4,
        label="initial\nstates",
    )
    centers = getattr(env.unwrapped if hasattr(env, "unwrapped") else env, "points_2d", None)
    if isinstance(centers, torch.Tensor) and centers.shape[-1] == 2:
        centers_np = centers.detach().cpu().numpy()
        in_view = (
            (centers_np[:, 0] >= mins[0])
            & (centers_np[:, 0] <= maxs[0])
            & (centers_np[:, 1] >= mins[1])
            & (centers_np[:, 1] <= maxs[1])
        )
        if np.any(in_view):
            ax.scatter(
                centers_np[in_view, 0],
                centers_np[in_view, 1],
                marker="x",
                s=35,
                color="0.0",
                alpha=0.55,
                zorder=5,
                label="fixed\npoints",
            )
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Support intervention trajectories")
    ax.grid(True, lw=0.45, alpha=0.28)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        borderaxespad=0.0,
        handlelength=1.6,
        ncol=3,
        columnspacing=0.9,
        labelspacing=0.55,
        handletextpad=0.55,
    )
    for ext in plot_formats:
        fig.savefig(output_dir / f"trajectory_vector_field_worst_intervention.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available in this allocation.")

    rows_csv = Path(args.rows_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_formats = [item.strip().lstrip(".") for item in args.plot_format.split(",") if item.strip()]
    _configure_paper_style()

    horizons = sorted(set(_parse_csv_ints(args.horizons)))
    if not horizons or min(horizons) <= 0:
        raise ValueError("--horizons must contain positive integers")
    max_horizon = max(horizons)
    if args.trajectory_length <= max_horizon:
        raise ValueError("--trajectory_length must be greater than the largest horizon")

    support_scheme, support_value = _parse_support_definition(args.support_definition)
    spec = _latest_spec(rows_csv, args.root_label, args.system, args.seed)
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    started = time.time()
    env, model = _load_checkpoint_model(checkpoint_path, spec.system_key, args.device)
    trajectories = _generate_trajectories(
        env,
        num_trajectories=args.num_candidate_trajectories,
        trajectory_length=args.trajectory_length,
        eval_seed=args.eval_seed,
    )
    basin_labels_t, centers, label_source = _label_sequences_and_centers(
        env,
        trajectories,
        system_key=spec.system_key,
        endpoint_rollout_steps=args.endpoint_rollout_steps,
    )
    basin_labels = basin_labels_t.cpu().numpy()
    candidate_mask = _high_center_margin_mask(
        trajectories,
        centers,
        basin_labels,
        args.depth_slice_mode,
    )
    rng = np.random.default_rng(args.eval_seed + args.seed)
    selected = _select_initial_indices(
        trajectories=trajectories,
        basin_labels=basin_labels,
        centers=centers,
        subset_mask=candidate_mask,
        max_horizon=max_horizon,
        num_points=args.num_initial_points,
        rng=rng,
        require_stable_true_basin=args.require_stable_true_basin,
    )

    traj_idx = selected[:, 0]
    time_idx = selected[:, 1]
    x0 = trajectories[traj_idx.tolist(), time_idx.tolist()]
    true_future = torch.stack(
        [
            trajectories[int(tid), int(t0) + 1 : int(t0) + max_horizon + 1]
            for tid, t0 in selected.tolist()
        ],
        dim=0,
    )
    initial_basins = basin_labels[traj_idx, time_idx].astype(np.int64)
    true_labels = basin_labels_t[traj_idx.tolist(), :].new_empty((selected.shape[0], max_horizon))
    for point_id, (tid, t0) in enumerate(selected.tolist()):
        true_labels[point_id] = basin_labels_t[int(tid), int(t0) + 1 : int(t0) + max_horizon + 1]

    with torch.no_grad():
        z0 = model.encode(x0.to(args.device, dtype=torch.float32)).detach()
    z0_np = z0.cpu().numpy()
    active_mask = _support_mask(z0_np, scheme=support_scheme, value=support_value)
    top_indices = _top_active_indices(z0_np, active_mask)

    support_sizes = active_mask.sum(axis=1)
    if int(support_sizes.min()) <= 0:
        raise RuntimeError("At least one selected initial state has empty support; cannot run interventions.")

    metadata = {
        "root_label": spec.root_label,
        "system_key": spec.system_key,
        "system_name": spec.system_name,
        "seed": int(spec.seed),
        "run_dir": spec.run_dir,
        "checkpoint_path": str(checkpoint_path),
        "support_definition": args.support_definition,
        "depth_slice_mode": args.depth_slice_mode,
        "label_source": label_source,
        "eval_seed": int(args.eval_seed),
        "num_candidate_trajectories": int(args.num_candidate_trajectories),
        "trajectory_length": int(args.trajectory_length),
        "max_horizon": int(max_horizon),
        "mean_initial_support_size": float(np.mean(support_sizes)),
        "min_initial_support_size": int(np.min(support_sizes)),
        "max_initial_support_size": int(np.max(support_sizes)),
    }

    initial_rows: List[Dict[str, object]] = []
    for point_id, (tid, t0) in enumerate(selected.tolist()):
        top10 = top_indices[point_id][: args.max_drop]
        initial_rows.append(
            {
                **metadata,
                "point_id": int(point_id),
                "trajectory_index": int(tid),
                "time_index": int(t0),
                "initial_basin": int(initial_basins[point_id]),
                "support_size": int(support_sizes[point_id]),
                "top_indices": " ".join(str(item) for item in top10),
                "top_abs_values": " ".join(f"{abs(float(z0_np[point_id, item])):.8g}" for item in top10),
                "x0": " ".join(f"{float(v):.8g}" for v in x0[point_id].tolist()),
            }
        )
    _write_csv(output_dir / "initial_points.csv", initial_rows)

    horizon_rows: List[Dict[str, object]] = []
    point_rows: List[Dict[str, object]] = []
    random_move_rows: List[Dict[str, object]] = []
    condition_predictions: Dict[str, torch.Tensor] = {}

    def evaluate_condition(
        condition: str,
        intervention_type: str,
        z_init: torch.Tensor,
        *,
        drop_count: Optional[int] = None,
        random_repeat: Optional[int] = None,
    ) -> None:
        preds = _rollout_decode(model, z_init, max_horizon=max_horizon)
        pred_labels = _label_states(env, preds, centers)
        h_rows, p_rows = _condition_metrics(
            condition=condition,
            intervention_type=intervention_type,
            drop_count=drop_count,
            random_repeat=random_repeat,
            preds=preds,
            true_future=true_future,
            pred_labels=pred_labels,
            true_labels=true_labels,
            initial_basins=initial_basins,
            horizons=horizons,
            metadata=metadata,
        )
        horizon_rows.extend(h_rows)
        point_rows.extend(p_rows)
        condition_predictions[condition] = preds

    evaluate_condition("baseline", "none", z0)

    max_drop = min(int(args.max_drop), int(np.min(support_sizes)))
    for drop_count in range(1, max_drop + 1):
        evaluate_condition(
            f"drop_top_{drop_count}",
            "coordinate_dropping",
            _make_drop_z(z0, top_indices, drop_count),
            drop_count=drop_count,
        )

    for repeat in range(int(args.random_support_repeats)):
        support_rng = np.random.default_rng(args.random_seed + repeat)
        z_random, moves = _make_random_support_z(z0, active_mask, rng=support_rng)
        condition = f"random_support_{repeat}"
        evaluate_condition(
            condition,
            "random_support",
            z_random,
            random_repeat=repeat,
        )
        for move in moves:
            random_move_rows.append({**metadata, "random_repeat": int(repeat), **move})

    _write_csv(output_dir / "intervention_horizon_metrics.csv", horizon_rows)
    _write_csv(output_dir / "intervention_point_metrics.csv", point_rows)
    _write_csv(output_dir / "random_support_moves.csv", random_move_rows)

    _plot_drop_absolute_curves_with_bands(
        point_rows,
        output_dir=output_dir,
        plot_formats=plot_formats,
        point_metric="cumulative_mse_sum",
        ylabel="Mean accumulated MSE",
        filename_stem="coordinate_dropping_accumulated_mse",
        title="Coordinate dropping: accumulated forecast error",
    )
    _plot_random_absolute_band(
        horizon_rows,
        point_rows,
        output_dir=output_dir,
        plot_formats=plot_formats,
        horizon_metric="cumulative_mse_sum_mean",
        point_metric="cumulative_mse_sum",
        ylabel="Accumulated MSE",
        filename_stem="random_support_accumulated_mse",
        title="Random support shuffle: accumulated forecast error",
    )
    _plot_drop_absolute_curves_with_bands(
        point_rows,
        output_dir=output_dir,
        plot_formats=plot_formats,
        point_metric="mse_at_h",
        ylabel="Mean MSE at horizon",
        filename_stem="coordinate_dropping_horizon_mse",
        title="Coordinate dropping: horizon forecast error",
    )
    _plot_random_absolute_band(
        horizon_rows,
        point_rows,
        output_dir=output_dir,
        plot_formats=plot_formats,
        horizon_metric="mse_at_h_mean",
        point_metric="mse_at_h",
        ylabel="MSE at horizon",
        filename_stem="random_support_horizon_mse",
        title="Random support shuffle: horizon forecast error",
    )

    _plot_metric_curves(
        horizon_rows,
        output_dir=output_dir,
        plot_formats=plot_formats,
        metric="basin_mismatch_vs_initial_fraction",
        ylabel="Predicted basin mismatch fraction",
        filename_stem="basin_mismatch_curves",
        log_y=False,
    )
    worst_condition = _worst_intervention_condition(
        horizon_rows,
        metric="cumulative_mse_sum_mean",
        horizon=max_horizon,
    )
    _plot_trajectories(
        output_dir=output_dir,
        plot_formats=plot_formats,
        env=env,
        x0=x0,
        true_future=true_future.cpu(),
        condition_predictions=condition_predictions,
        worst_condition=worst_condition,
    )

    summary = {
        **metadata,
        "rows_csv": str(rows_csv),
        "output_dir": str(output_dir),
        "horizons": horizons,
        "conditions": sorted({str(row["condition"]) for row in horizon_rows}, key=_condition_sort_key),
        "worst_intervention_condition": worst_condition,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote coordinate-intervention artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
