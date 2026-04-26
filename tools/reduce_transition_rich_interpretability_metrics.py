#!/usr/bin/env python3
"""Reduce study-plan interpretability metrics for the fixed transition-rich shortlist.

This reducer intentionally replaces the older trajectory-majority-support
protocol with state-level support metrics aligned to
`docs/planning/interpretability_study_plan.md`:

- `H(B|S)` / `H(S|B)` and NMI for exact supports
- `U_exact`: dominant exact-support mass per basin
- support-family metrics via greedy Jaccard clustering on exact supports
- within-trajectory support persistence and chatter
- support-switch alignment against basin switches when they occur
- support-conditioned versus basin-conditioned local-operator separation
- deep-basin counterfactual support projection tests

It evaluates multiple support definitions (absolute, relative, top-k) and,
when a model exposes block structure, also reports dominant-group metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.benchmarks.transition_rich_basin_partition_manifest import get_transition_rich_basin_count
from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, make_env
from skae.model import make_model

EPS = 1e-12


@dataclass(frozen=True)
class RunSpec:
    root_label: str
    system_key: str
    system_name: str
    seed: int
    run_dir: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True, help="forecasting_rows.csv used to discover runs")
    parser.add_argument("--output_dir", required=True, help="directory for reduced metric artifacts")
    parser.add_argument(
        "--root_labels",
        default="lista_dense_basin_partition,lista_blockdiag_basin_partition",
        help="comma-separated root labels to include",
    )
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated integer seed filter")
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--min_operator_transitions", type=int, default=128)
    parser.add_argument(
        "--absolute_thresholds",
        default="1e-3,3e-3,1e-2",
        help="comma-separated absolute support thresholds",
    )
    parser.add_argument(
        "--relative_thresholds",
        default="0.05,0.1,0.2",
        help="comma-separated relative support thresholds",
    )
    parser.add_argument(
        "--topk_values",
        default="4,8,16",
        help="comma-separated top-k support values",
    )
    parser.add_argument(
        "--family_jaccard_threshold",
        type=float,
        default=0.5,
        help="greedy Jaccard threshold used to merge exact supports into support families",
    )
    parser.add_argument(
        "--freeze_support_horizons",
        default="1,5,10,20",
        help="comma-separated rollout horizons for canonical support-freeze interventions",
    )
    parser.add_argument(
        "--max_freeze_states",
        type=int,
        default=2048,
        help="maximum number of state starts per freeze-support horizon",
    )
    parser.add_argument(
        "--max_jacobian_states",
        type=int,
        default=128,
        help="maximum number of sampled states per subset for effective-Jacobian diagnostics",
    )
    parser.add_argument(
        "--min_jacobian_states",
        type=int,
        default=16,
        help="minimum states required for a support/basin Jacobian family",
    )
    parser.add_argument(
        "--save_visuals",
        action="store_true",
        help="save per-run visual diagnostics for selected support definitions",
    )
    parser.add_argument(
        "--visual_supports",
        default="",
        help=(
            "comma-separated support definitions to visualize, formatted as "
            "'scheme:value' (for example 'absolute:0.001,topk:8'); "
            "if empty with --save_visuals, only the first support definition is rendered"
        ),
    )
    parser.add_argument(
        "--visual_max_points",
        type=int,
        default=5000,
        help="maximum number of phase-portrait points to render in each visual artifact",
    )
    parser.add_argument(
        "--visual_max_switch_trajectories",
        type=int,
        default=64,
        help="maximum number of switch trajectories to render in each switch raster",
    )
    parser.add_argument(
        "--progress_every_runs",
        type=int,
        default=1,
        help="print a reducer progress line after every N completed run specs",
    )
    parser.add_argument(
        "--flush_every_runs",
        type=int,
        default=0,
        help=(
            "rewrite partial reducer artifacts after every N completed run specs; "
            "set to 0 to only write final outputs"
        ),
    )
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_visual_supports(raw: str) -> Optional[set[str]]:
    selectors = {item.strip() for item in raw.split(",") if item.strip()}
    return selectors if selectors else None


def _run_timestamp_key(run_dir: str) -> Tuple[str, str]:
    path = Path(run_dir)
    stem = path.name
    match = re.fullmatch(r"\d{8}-\d{6}", stem)
    if match:
        return stem, run_dir
    return "", run_dir


def _load_latest_specs(
    rows_csv: Path,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
) -> List[RunSpec]:
    selected_roots = set(root_labels)
    selected_systems = set(systems)
    selected_seeds = set(seeds)
    best_rows: Dict[Tuple[str, str, int], Dict[str, str]] = {}

    with rows_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            root_label = str(row.get("root_label", "")).strip()
            if root_label not in selected_roots:
                continue
            system_key = str(row.get("system_key", "")).strip()
            if selected_systems and system_key not in selected_systems:
                continue
            seed = int(row.get("seed", 0))
            if selected_seeds and seed not in selected_seeds:
                continue
            key = (root_label, system_key, seed)
            incumbent = best_rows.get(key)
            if incumbent is None or _run_timestamp_key(row["run_dir"]) > _run_timestamp_key(incumbent["run_dir"]):
                best_rows[key] = row

    specs = [
        RunSpec(
            root_label=row["root_label"],
            system_key=row["system_key"],
            system_name=row.get("system_name", row["system_key"]),
            seed=int(row["seed"]),
            run_dir=row["run_dir"],
        )
        for row in best_rows.values()
    ]
    return sorted(specs, key=lambda item: (item.root_label, item.system_key, item.seed))


def _load_checkpoint_model(
    checkpoint_path: Path,
    system_key: str,
    device: str,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = Config.from_dict(checkpoint["config"])
    cfg.ENV.ENV_NAME = system_key
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    load_model_state_dict_compat(model, checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return cfg, env, model


def _generate_observation_trajectories(
    env,
    *,
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
) -> torch.Tensor:
    vec_env = VectorWrapper(env, num_trajectories)
    rng = torch.Generator().manual_seed(eval_seed)
    return vec_env.generate_sequence_batch(rng=rng, window_length=trajectory_length).float()


def _long_rollout(env, states: torch.Tensor, steps: int) -> torch.Tensor:
    current = states.clone()
    for _ in range(max(0, steps)):
        current = env.step(current)
    return current


def _label_from_native_method(method, sequences: torch.Tensor) -> torch.Tensor:
    flat = sequences.reshape(-1, sequences.shape[-1])
    labels = method(flat)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    return labels.reshape(sequences.shape[:-1]).to(dtype=torch.long)


def _kmeans_centers(points: torch.Tensor, num_centers: int, num_iters: int = 25) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError("points must have shape [N, dim]")
    if points.shape[0] < num_centers:
        raise ValueError("Need at least as many points as centers")

    centers = [points[0]]
    while len(centers) < num_centers:
        current = torch.stack(centers, dim=0)
        min_dists = torch.cdist(points, current).min(dim=1).values
        centers.append(points[min_dists.argmax()])
    centers = torch.stack(centers, dim=0).clone()

    for _ in range(num_iters):
        assignments = torch.cdist(points, centers).argmin(dim=1)
        new_centers = []
        for center_idx in range(num_centers):
            mask = assignments == center_idx
            if bool(mask.any()):
                new_centers.append(points[mask].mean(dim=0))
            else:
                new_centers.append(centers[center_idx])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


def _estimate_basin_centers(
    env,
    trajectories: torch.Tensor,
    basin_count: int,
    endpoint_rollout_steps: int,
) -> torch.Tensor:
    endpoints = trajectories[:, -1, :]
    converged = _long_rollout(env, endpoints, endpoint_rollout_steps)
    return _kmeans_centers(converged, basin_count)


def _assign_nearest_centers(sequences: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    flat = sequences.reshape(-1, sequences.shape[-1])
    labels = torch.cdist(flat, centers).argmin(dim=1)
    return labels.reshape(sequences.shape[:-1]).to(dtype=torch.long)


def _label_sequences_and_centers(
    env,
    trajectories: torch.Tensor,
    *,
    system_key: str,
    endpoint_rollout_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    basin_count = int(get_transition_rich_basin_count(system_key))
    if hasattr(env, "basin_label"):
        basin_labels = _label_from_native_method(env.basin_label, trajectories)
        centers = getattr(env, "points", None)
        if isinstance(centers, torch.Tensor) and centers.ndim == 2:
            return basin_labels, centers.to(dtype=trajectories.dtype), "native_basin_label+env_points"
        estimated = _estimate_basin_centers(env, trajectories, basin_count, endpoint_rollout_steps)
        return basin_labels, estimated, "native_basin_label+estimated_centers"
    centers = getattr(env, "points", None)
    if isinstance(centers, torch.Tensor) and centers.ndim == 2:
        return _assign_nearest_centers(trajectories, centers), centers.to(dtype=trajectories.dtype), "env_points"
    estimated = _estimate_basin_centers(env, trajectories, basin_count, endpoint_rollout_steps)
    return _assign_nearest_centers(trajectories, estimated), estimated, "estimated_centers"


def _encode_trajectories(model, trajectories: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        flat = trajectories.reshape(-1, trajectories.shape[-1]).to(device)
        latents = model.encode(flat).reshape(*trajectories.shape[:2], -1).detach().cpu().numpy()
    return latents


def _support_mask(latents: np.ndarray, *, scheme: str, value: float) -> np.ndarray:
    abs_latents = np.abs(latents)
    if scheme == "absolute":
        return abs_latents > float(value)
    if scheme == "relative":
        max_abs = abs_latents.max(axis=-1, keepdims=True)
        return abs_latents > (float(value) * np.maximum(max_abs, EPS))
    if scheme == "topk":
        k = int(value)
        if k <= 0:
            raise ValueError(f"top-k must be positive, got {k}")
        dim = abs_latents.shape[-1]
        if k >= dim:
            return np.ones_like(abs_latents, dtype=bool)
        indices = np.argpartition(abs_latents, kth=dim - k, axis=-1)[..., -k:]
        mask = np.zeros_like(abs_latents, dtype=bool)
        np.put_along_axis(mask, indices, True, axis=-1)
        return mask
    raise ValueError(f"Unknown support scheme '{scheme}'")


def _support_keys(mask: np.ndarray) -> np.ndarray:
    packed = np.packbits(mask.astype(np.uint8), axis=-1)
    return np.asarray([row.tobytes() for row in packed.reshape(-1, packed.shape[-1])], dtype=object).reshape(
        mask.shape[:-1]
    )


def _dominant_group_labels(
    latents: np.ndarray,
    block_sizes: Sequence[int],
    *,
    offset: int = 0,
) -> np.ndarray:
    labels = np.zeros(latents.shape[:-1], dtype=np.int64)
    group_energies = []
    cursor = int(offset)
    for block_size in block_sizes:
        group_latents = latents[..., cursor : cursor + block_size]
        group_energies.append(np.linalg.norm(group_latents, axis=-1))
        cursor += block_size
    stacked = np.stack(group_energies, axis=-1)
    labels = stacked.argmax(axis=-1)
    return labels


def _jaccard_for_consecutive(mask: np.ndarray) -> np.ndarray:
    inter = np.logical_and(mask[:, :-1], mask[:, 1:]).sum(axis=-1).astype(np.float64)
    union = np.logical_or(mask[:, :-1], mask[:, 1:]).sum(axis=-1).astype(np.float64)
    out = np.ones_like(inter, dtype=np.float64)
    valid = union > 0.0
    out[valid] = inter[valid] / union[valid]
    return out


def _entropy_from_counter(counter: Counter[object]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        prob = float(count) / total
        entropy -= prob * math.log(prob + EPS)
    return entropy


def conditional_entropy(x: Sequence[object], y: Sequence[object]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    joint = Counter(zip(x, y))
    marginal_y = Counter(y)
    return _entropy_from_counter(joint) - _entropy_from_counter(marginal_y)


def normalized_mutual_information(x: Sequence[object], y: Sequence[object]) -> float:
    hx = _entropy_from_counter(Counter(x))
    hy = _entropy_from_counter(Counter(y))
    if hx <= 0.0 or hy <= 0.0:
        return 0.0
    mi = hx + hy - _entropy_from_counter(Counter(zip(x, y)))
    return float(mi / max(math.sqrt(hx * hy), EPS))


def dominant_support_mass_per_basin(supports: Sequence[object], basins: Sequence[int]) -> Optional[float]:
    support_by_basin: Dict[int, Counter[object]] = defaultdict(Counter)
    for support_key, basin in zip(supports, basins):
        if basin < 0:
            continue
        support_by_basin[int(basin)][support_key] += 1
    if not support_by_basin:
        return None
    masses = []
    for counter in support_by_basin.values():
        total = float(sum(counter.values()))
        masses.append(max(counter.values()) / max(total, 1.0))
    return float(np.mean(masses))


def _binary_jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return 1.0
    return inter / union


def support_family_labels(support_mask: np.ndarray, *, min_jaccard: float) -> np.ndarray:
    if support_mask.ndim != 3:
        raise ValueError("support_mask must have shape [num_trajectories, trajectory_length, latent_dim]")
    flat_mask = support_mask.reshape(-1, support_mask.shape[-1])
    flat_keys = _support_keys(support_mask).reshape(-1)
    key_counts = Counter(flat_keys.tolist())
    key_masks: Dict[object, np.ndarray] = {}
    for key, mask in zip(flat_keys.tolist(), flat_mask):
        if key not in key_masks:
            key_masks[key] = mask.astype(bool, copy=True)

    prototypes: List[np.ndarray] = []
    key_to_family: Dict[object, int] = {}
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

    labels = np.asarray([key_to_family[key] for key in flat_keys.tolist()], dtype=np.int64)
    return labels.reshape(support_mask.shape[:-1])


def canonical_support_masks_by_basin(
    support_mask: np.ndarray,
    basin_labels: np.ndarray,
    candidate_mask: np.ndarray,
) -> Dict[int, np.ndarray]:
    flat_support = support_mask.reshape(-1, support_mask.shape[-1])
    flat_keys = _support_keys(support_mask).reshape(-1)
    flat_basins = basin_labels.reshape(-1)
    flat_candidate = candidate_mask.reshape(-1)

    support_by_basin: Dict[int, Counter[object]] = defaultdict(Counter)
    key_to_mask: Dict[object, np.ndarray] = {}
    for mask, key, basin, keep in zip(flat_support, flat_keys.tolist(), flat_basins.tolist(), flat_candidate.tolist()):
        if not keep or int(basin) < 0:
            continue
        basin_int = int(basin)
        support_by_basin[basin_int][key] += 1
        if key not in key_to_mask:
            key_to_mask[key] = mask.astype(bool, copy=True)

    canonical: Dict[int, np.ndarray] = {}
    for basin, counter in support_by_basin.items():
        top_key = counter.most_common(1)[0][0]
        canonical[basin] = key_to_mask[top_key]
    return canonical


def _margin_subsets(states: torch.Tensor, centers: torch.Tensor) -> Dict[str, np.ndarray]:
    flat = states.reshape(-1, states.shape[-1])
    dists = torch.cdist(flat, centers.to(dtype=flat.dtype))
    if dists.shape[1] < 2:
        valid = np.ones(flat.shape[0], dtype=bool)
        return {"all": valid, "deep": valid, "boundary": valid}
    smallest = torch.topk(dists, k=2, largest=False, dim=1).values
    margins = (smallest[:, 1] - smallest[:, 0]).cpu().numpy()
    deep_cut = float(np.quantile(margins, 0.75))
    boundary_cut = float(np.quantile(margins, 0.25))
    return {
        "all": np.ones_like(margins, dtype=bool),
        "deep": margins >= deep_cut,
        "boundary": margins <= boundary_cut,
    }


def _fit_linear_operator(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> Optional[np.ndarray]:
    if x.shape[0] == 0 or y.shape[0] == 0 or x.shape != y.shape:
        return None
    dim = x.shape[1]
    xtx = x.T @ x
    xty = x.T @ y
    reg = ridge_lambda * np.eye(dim, dtype=x.dtype)
    try:
        return np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        return None


def _fit_operator_families(
    latents: np.ndarray,
    basin_labels: np.ndarray,
    class_labels: np.ndarray,
    *,
    ridge_lambda: float,
    min_transitions: int,
) -> Tuple[Dict[int, np.ndarray], Dict[object, np.ndarray], Dict[object, int], Counter[object]]:
    x = latents[:, :-1, :].reshape(-1, latents.shape[-1])
    y = latents[:, 1:, :].reshape(-1, latents.shape[-1])
    basin_t = basin_labels[:, :-1].reshape(-1)
    class_t = class_labels[:, :-1].reshape(-1)

    basin_ops: Dict[int, np.ndarray] = {}
    for basin in sorted({int(item) for item in basin_t.tolist() if int(item) >= 0}):
        mask = basin_t == basin
        if int(mask.sum()) < min_transitions:
            continue
        operator = _fit_linear_operator(x[mask], y[mask], ridge_lambda)
        if operator is not None:
            basin_ops[basin] = operator

    class_counts = Counter(class_t.tolist())
    class_ops: Dict[object, np.ndarray] = {}
    class_major_basin: Dict[object, int] = {}
    for class_id, count in class_counts.items():
        if count < min_transitions:
            continue
        mask = class_t == class_id
        operator = _fit_linear_operator(x[mask], y[mask], ridge_lambda)
        if operator is None:
            continue
        basins_here = basin_t[mask]
        basin_counter = Counter(int(item) for item in basins_here.tolist() if int(item) >= 0)
        if not basin_counter:
            continue
        class_ops[class_id] = operator
        class_major_basin[class_id] = basin_counter.most_common(1)[0][0]
    return basin_ops, class_ops, class_major_basin, class_counts


def operator_distance_summary(
    latents: np.ndarray,
    basin_labels: np.ndarray,
    class_labels: np.ndarray,
    *,
    ridge_lambda: float,
    min_transitions: int,
) -> Dict[str, Optional[float]]:
    basin_ops, class_ops, class_major_basin, _class_counts = _fit_operator_families(
        latents,
        basin_labels,
        class_labels,
        ridge_lambda=ridge_lambda,
        min_transitions=min_transitions,
    )

    class_ids = list(class_ops.keys())
    if not class_ids:
        return {
            "operator_class_count": 0.0,
            "operator_support_vs_basin_fro_mean": None,
            "operator_within_basin_fro_mean": None,
            "operator_between_basin_fro_mean": None,
            "operator_between_over_within": None,
        }

    support_vs_basin = []
    for class_id in class_ids:
        basin = class_major_basin[class_id]
        basin_op = basin_ops.get(basin)
        if basin_op is None:
            continue
        support_vs_basin.append(float(np.linalg.norm(class_ops[class_id] - basin_op, ord="fro")))

    within = []
    between = []
    for i, class_i in enumerate(class_ids):
        basin_i = class_major_basin[class_i]
        op_i = class_ops[class_i]
        for class_j in class_ids[i + 1 :]:
            basin_j = class_major_basin[class_j]
            dist = float(np.linalg.norm(op_i - class_ops[class_j], ord="fro"))
            if basin_i == basin_j:
                within.append(dist)
            else:
                between.append(dist)

    within_mean = float(np.mean(within)) if within else None
    between_mean = float(np.mean(between)) if between else None
    ratio = None
    if within_mean is not None and between_mean is not None and within_mean > 0.0:
        ratio = between_mean / within_mean
    return {
        "operator_class_count": float(len(class_ids)),
        "operator_support_vs_basin_fro_mean": (
            float(np.mean(support_vs_basin)) if support_vs_basin else None
        ),
        "operator_within_basin_fro_mean": within_mean,
        "operator_between_basin_fro_mean": between_mean,
        "operator_between_over_within": ratio,
    }


def class_transition_metrics(
    class_labels: np.ndarray,
    basin_labels: np.ndarray,
) -> Dict[str, Optional[float]]:
    same_basin = basin_labels[:, :-1] == basin_labels[:, 1:]
    basin_switch = np.logical_not(same_basin)
    class_switch = class_labels[:, :-1] != class_labels[:, 1:]
    exact_persistence = float(np.mean(~class_switch[same_basin])) if np.any(same_basin) else None
    on_switch = float(np.mean(class_switch[basin_switch])) if np.any(basin_switch) else None
    off_switch = float(np.mean(class_switch[same_basin])) if np.any(same_basin) else None
    return {
        "support_persistence": exact_persistence,
        "support_switch_on_basin_switch": on_switch,
        "support_switch_off_basin_switch": off_switch,
        "basin_switch_fraction": float(np.mean(basin_switch)),
    }


def switch_timing_metrics(
    class_labels: np.ndarray,
    basin_labels: np.ndarray,
) -> Dict[str, Optional[float]]:
    if class_labels.shape != basin_labels.shape:
        raise ValueError("class_labels and basin_labels must have the same shape")
    if class_labels.ndim != 2:
        raise ValueError("switch_timing_metrics expects [num_trajectories, trajectory_length] arrays")

    transfer_count = 0
    detected_count = 0
    delays: List[float] = []
    abs_delays: List[float] = []
    false_switches: List[float] = []
    chatter_counts: List[float] = []
    pre_dwells: List[float] = []
    post_dwells: List[float] = []

    for traj_classes, traj_basins in zip(class_labels, basin_labels):
        basin_switches = np.flatnonzero(traj_basins[1:] != traj_basins[:-1]) + 1
        if basin_switches.size == 0:
            continue

        transfer_count += 1
        true_switch = int(basin_switches[0])
        class_switches = np.flatnonzero(traj_classes[1:] != traj_classes[:-1]) + 1
        false_switches.append(float(np.sum(class_switches < true_switch)))

        post_true_switches = class_switches[class_switches >= true_switch]
        if post_true_switches.size == 0:
            continue

        detected_count += 1
        detected_switch = int(post_true_switches[0])
        delay = float(detected_switch - true_switch)
        delays.append(delay)
        abs_delays.append(abs(delay))

        chatter_counts.append(float(max(int(post_true_switches.size) - 1, 0)))
        previous_switches = class_switches[class_switches < detected_switch]
        last_previous = int(previous_switches[-1]) if previous_switches.size > 0 else 0
        next_switches = class_switches[class_switches > detected_switch]
        next_switch = int(next_switches[0]) if next_switches.size > 0 else int(len(traj_classes))
        pre_dwells.append(float(detected_switch - last_previous))
        post_dwells.append(float(max(next_switch - detected_switch, 0)))

    detected_fraction = None
    miss_fraction = None
    if transfer_count > 0:
        detected_fraction = float(detected_count) / float(transfer_count)
        miss_fraction = 1.0 - detected_fraction

    return {
        "switch_trajectory_count": float(transfer_count),
        "switch_detected_fraction": detected_fraction,
        "switch_miss_fraction": miss_fraction,
        "switch_delay_mean": float(np.mean(delays)) if delays else None,
        "switch_delay_abs_mean": float(np.mean(abs_delays)) if abs_delays else None,
        "switch_false_switches_mean": float(np.mean(false_switches)) if false_switches else None,
        "switch_chatter_mean": float(np.mean(chatter_counts)) if chatter_counts else None,
        "switch_pre_dwell_mean": float(np.mean(pre_dwells)) if pre_dwells else None,
        "switch_post_dwell_mean": float(np.mean(post_dwells)) if post_dwells else None,
    }


def support_transition_metrics(
    support_mask: np.ndarray,
    support_keys: np.ndarray,
    basin_labels: np.ndarray,
) -> Dict[str, Optional[float]]:
    base_metrics = class_transition_metrics(support_keys, basin_labels)
    jaccard = _jaccard_for_consecutive(support_mask)
    same_basin = basin_labels[:, :-1] == basin_labels[:, 1:]
    exact_jaccard = float(np.mean(jaccard[same_basin])) if np.any(same_basin) else None
    return {**base_metrics, "support_jaccard_mean": exact_jaccard}


def _class_metrics(
    class_labels: np.ndarray,
    basin_labels: np.ndarray,
) -> Dict[str, Optional[float]]:
    class_flat = class_labels.reshape(-1).tolist()
    basin_flat = [int(item) for item in basin_labels.reshape(-1).tolist()]
    h_b_given_c = conditional_entropy(basin_flat, class_flat)
    h_c_given_b = conditional_entropy(class_flat, basin_flat)
    nmi = normalized_mutual_information(class_flat, basin_flat)
    u_exact = dominant_support_mass_per_basin(class_flat, basin_flat)
    unique_classes = float(len(set(class_flat)))
    return {
        "h_basin_given_class": h_b_given_c,
        "h_class_given_basin": h_c_given_b,
        "class_nmi": nmi,
        "u_exact": u_exact,
        "unique_class_count": unique_classes,
    }


def support_projection_metrics(
    model,
    latents: np.ndarray,
    trajectories: torch.Tensor,
    basin_labels: np.ndarray,
    support_templates: Dict[int, np.ndarray],
    subset_mask: np.ndarray,
    *,
    device: str,
) -> Dict[str, Optional[float]]:
    none_metrics = {
        "support_projection_state_count": 0.0,
        "support_projection_template_count": float(len(support_templates)),
        "support_projection_base_mse": None,
        "support_projection_self_mse": None,
        "support_projection_wrong_mse": None,
        "support_projection_self_over_base": None,
        "support_projection_wrong_over_base": None,
        "support_projection_wrong_minus_self": None,
    }
    if not support_templates:
        return none_metrics

    subset_state_mask = subset_mask.reshape(basin_labels.shape)
    transition_mask = np.logical_and(subset_state_mask[:, :-1], basin_labels[:, :-1] == basin_labels[:, 1:])
    if not np.any(transition_mask):
        return none_metrics

    flat_transition_mask = transition_mask.reshape(-1)
    transition_indices = np.flatnonzero(flat_transition_mask)
    flat_basin = basin_labels[:, :-1].reshape(-1)[transition_indices]
    template_available = np.asarray([int(basin) in support_templates for basin in flat_basin], dtype=bool)
    if not np.any(template_available):
        return none_metrics

    valid_indices = transition_indices[template_available]
    flat_latents = latents[:, :-1, :].reshape(-1, latents.shape[-1])[valid_indices]
    current_basins = flat_basin[template_available]
    true_next = trajectories[:, 1:, :].reshape(-1, trajectories.shape[-1])[valid_indices.tolist()]

    z_current = torch.from_numpy(flat_latents).to(device=device, dtype=torch.float32)
    true_next = true_next.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        base_pred = model.decode(model.step_latent(z_current))
        base_err = ((base_pred - true_next) ** 2).mean(dim=-1)

        own_err_chunks: List[torch.Tensor] = []
        wrong_err_chunks: List[torch.Tensor] = []
        unique_basins = sorted({int(basin) for basin in current_basins.tolist()})
        for basin in unique_basins:
            basin_mask = current_basins == basin
            if basin not in support_templates:
                continue
            z_b = z_current[basin_mask]
            x_b = true_next[basin_mask]
            own_mask = torch.from_numpy(support_templates[basin].astype(np.float32)).to(device=device).unsqueeze(0)
            own_pred = model.decode(model.step_latent(z_b * own_mask))
            own_err_chunks.append(((own_pred - x_b) ** 2).mean(dim=-1))

            wrong_masks = [
                torch.from_numpy(template.astype(np.float32)).to(device=device).unsqueeze(0)
                for basin_id, template in sorted(support_templates.items())
                if int(basin_id) != basin
            ]
            if wrong_masks:
                wrong_errs = []
                for wrong_mask in wrong_masks:
                    wrong_pred = model.decode(model.step_latent(z_b * wrong_mask))
                    wrong_errs.append(((wrong_pred - x_b) ** 2).mean(dim=-1))
                wrong_err_chunks.append(torch.stack(wrong_errs, dim=0).mean(dim=0))

    if not own_err_chunks:
        return none_metrics

    base_mean = float(base_err.mean().item())
    own_mean = float(torch.cat(own_err_chunks, dim=0).mean().item())
    wrong_mean = None
    if wrong_err_chunks:
        wrong_mean = float(torch.cat(wrong_err_chunks, dim=0).mean().item())

    self_ratio = None
    wrong_ratio = None
    if base_mean > EPS:
        self_ratio = own_mean / base_mean
        if wrong_mean is not None:
            wrong_ratio = wrong_mean / base_mean
    return {
        "support_projection_state_count": float(z_current.shape[0]),
        "support_projection_template_count": float(len(support_templates)),
        "support_projection_base_mse": base_mean,
        "support_projection_self_mse": own_mean,
        "support_projection_wrong_mse": wrong_mean,
        "support_projection_self_over_base": self_ratio,
        "support_projection_wrong_over_base": wrong_ratio,
        "support_projection_wrong_minus_self": (
            None if wrong_mean is None else wrong_mean - own_mean
        ),
    }


def _rollout_from_latent(
    model,
    latent: torch.Tensor,
    *,
    horizon: int,
    freeze_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    z = latent
    if freeze_mask is not None:
        z = z * freeze_mask
    for _ in range(horizon):
        z = model.step_latent(z)
        if freeze_mask is not None:
            z = z * freeze_mask
    return model.decode(z)


def freeze_support_rollout_metrics(
    model,
    latents: np.ndarray,
    trajectories: torch.Tensor,
    basin_labels: np.ndarray,
    support_templates: Dict[int, np.ndarray],
    subset_mask: np.ndarray,
    *,
    device: str,
    horizons: Sequence[int],
    max_states_per_horizon: int,
    sample_seed: int,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "support_freeze_template_count": float(len(support_templates)),
    }
    if not support_templates:
        out["support_freeze_longest_horizon"] = None
        out["support_freeze_longest_self_over_base"] = None
        out["support_freeze_longest_wrong_over_base"] = None
        return out

    subset_state_mask = subset_mask.reshape(basin_labels.shape)
    rng = np.random.default_rng(sample_seed)
    longest_horizon = None

    for horizon in sorted({int(item) for item in horizons if int(item) > 0}):
        longest_horizon = horizon
        horizon_key = f"h{horizon}"
        default_fields = {
            f"support_freeze_state_count_{horizon_key}": 0.0,
            f"support_freeze_base_mse_{horizon_key}": None,
            f"support_freeze_self_mse_{horizon_key}": None,
            f"support_freeze_wrong_mse_{horizon_key}": None,
            f"support_freeze_self_over_base_{horizon_key}": None,
            f"support_freeze_wrong_over_base_{horizon_key}": None,
        }
        out.update(default_fields)

        if horizon >= trajectories.shape[1]:
            continue

        start_mask = subset_state_mask[:, :-horizon].copy()
        start_basins = basin_labels[:, :-horizon]
        for step in range(1, horizon + 1):
            start_mask &= basin_labels[:, step : step + start_mask.shape[1]] == start_basins

        candidate_indices = np.argwhere(start_mask)
        if candidate_indices.size == 0:
            continue

        if max_states_per_horizon > 0 and candidate_indices.shape[0] > max_states_per_horizon:
            selected = rng.choice(candidate_indices.shape[0], size=max_states_per_horizon, replace=False)
            candidate_indices = candidate_indices[np.sort(selected)]

        traj_idx = candidate_indices[:, 0]
        time_idx = candidate_indices[:, 1]
        current_basins = basin_labels[traj_idx, time_idx]
        template_available = np.asarray([int(basin) in support_templates for basin in current_basins], dtype=bool)
        if not np.any(template_available):
            continue

        traj_idx = traj_idx[template_available]
        time_idx = time_idx[template_available]
        current_basins = current_basins[template_available]
        z_current = torch.from_numpy(latents[traj_idx, time_idx]).to(device=device, dtype=torch.float32)
        true_future = trajectories[traj_idx.tolist(), (time_idx + horizon).tolist()].to(
            device=device, dtype=torch.float32
        )

        with torch.no_grad():
            base_pred = _rollout_from_latent(model, z_current, horizon=horizon)
            base_err = ((base_pred - true_future) ** 2).mean(dim=-1)

            own_err_chunks: List[torch.Tensor] = []
            wrong_err_chunks: List[torch.Tensor] = []
            for basin in sorted({int(item) for item in current_basins.tolist()}):
                basin_mask = current_basins == basin
                if basin not in support_templates:
                    continue
                z_b = z_current[basin_mask]
                x_b = true_future[basin_mask]
                own_mask = torch.from_numpy(support_templates[basin].astype(np.float32)).to(
                    device=device
                ).unsqueeze(0)
                own_pred = _rollout_from_latent(model, z_b, horizon=horizon, freeze_mask=own_mask)
                own_err_chunks.append(((own_pred - x_b) ** 2).mean(dim=-1))

                wrong_masks = [
                    torch.from_numpy(template.astype(np.float32)).to(device=device).unsqueeze(0)
                    for basin_id, template in sorted(support_templates.items())
                    if int(basin_id) != basin
                ]
                if wrong_masks:
                    wrong_errs = []
                    for wrong_mask in wrong_masks:
                        wrong_pred = _rollout_from_latent(model, z_b, horizon=horizon, freeze_mask=wrong_mask)
                        wrong_errs.append(((wrong_pred - x_b) ** 2).mean(dim=-1))
                    wrong_err_chunks.append(torch.stack(wrong_errs, dim=0).mean(dim=0))

        if not own_err_chunks:
            continue

        base_mean = float(base_err.mean().item())
        own_mean = float(torch.cat(own_err_chunks, dim=0).mean().item())
        wrong_mean = (
            float(torch.cat(wrong_err_chunks, dim=0).mean().item())
            if wrong_err_chunks
            else None
        )
        own_ratio = None if base_mean <= EPS else own_mean / base_mean
        wrong_ratio = None if base_mean <= EPS or wrong_mean is None else wrong_mean / base_mean

        out.update(
            {
                f"support_freeze_state_count_{horizon_key}": float(z_current.shape[0]),
                f"support_freeze_base_mse_{horizon_key}": base_mean,
                f"support_freeze_self_mse_{horizon_key}": own_mean,
                f"support_freeze_wrong_mse_{horizon_key}": wrong_mean,
                f"support_freeze_self_over_base_{horizon_key}": own_ratio,
                f"support_freeze_wrong_over_base_{horizon_key}": wrong_ratio,
            }
        )

    if longest_horizon is not None:
        longest_key = f"h{longest_horizon}"
        out["support_freeze_longest_horizon"] = float(longest_horizon)
        out["support_freeze_longest_self_over_base"] = out.get(
            f"support_freeze_self_over_base_{longest_key}"
        )
        out["support_freeze_longest_wrong_over_base"] = out.get(
            f"support_freeze_wrong_over_base_{longest_key}"
        )
    else:
        out["support_freeze_longest_horizon"] = None
        out["support_freeze_longest_self_over_base"] = None
        out["support_freeze_longest_wrong_over_base"] = None
    return out


def _sample_state_indices(
    subset_masks: Dict[str, np.ndarray],
    *,
    max_states_per_subset: int,
    sample_seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(sample_seed)
    sampled_indices_by_subset: Dict[str, np.ndarray] = {}
    union: List[np.ndarray] = []
    for subset_name, mask in subset_masks.items():
        indices = np.flatnonzero(mask)
        if max_states_per_subset > 0 and indices.size > max_states_per_subset:
            indices = np.sort(rng.choice(indices, size=max_states_per_subset, replace=False))
        sampled_indices_by_subset[subset_name] = indices
        union.append(indices)

    if not union or sum(indices.size for indices in sampled_indices_by_subset.values()) == 0:
        return np.empty(0, dtype=np.int64), {name: np.empty(0, dtype=bool) for name in subset_masks}

    union_indices = np.unique(np.concatenate(union))
    subset_local_masks = {
        subset_name: np.isin(union_indices, indices, assume_unique=False)
        for subset_name, indices in sampled_indices_by_subset.items()
    }
    return union_indices, subset_local_masks


def _compute_state_jacobians(
    model,
    states: torch.Tensor,
    *,
    device: str,
) -> np.ndarray:
    if states.ndim != 2:
        raise ValueError("states must have shape [num_states, state_dim]")
    if states.numel() == 0:
        state_dim = int(states.shape[-1]) if states.ndim == 2 else 0
        return np.empty((0, state_dim, state_dim), dtype=np.float32)

    model.eval()
    outputs: List[np.ndarray] = []
    with torch.enable_grad():
        for state in states.to(device=device, dtype=torch.float32):
            x = state.detach().clone().requires_grad_(True)

            def step_fn(inp: torch.Tensor) -> torch.Tensor:
                encoded = model.encode(inp.unsqueeze(0))
                advanced = model.step_latent(encoded)
                return model.decode(advanced).squeeze(0)

            jacobian = torch.autograd.functional.jacobian(step_fn, x)
            outputs.append(jacobian.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.stack(outputs, axis=0)


def jacobian_distance_summary(
    jacobians: np.ndarray,
    basin_labels: np.ndarray,
    class_labels: np.ndarray,
    *,
    min_states: int,
    true_jacobians: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    if jacobians.ndim != 3:
        raise ValueError("jacobians must have shape [num_states, state_dim, state_dim]")
    if basin_labels.shape[0] != jacobians.shape[0] or class_labels.shape[0] != jacobians.shape[0]:
        raise ValueError("jacobian summaries require aligned jacobians, basin_labels, and class_labels")
    if true_jacobians is not None and true_jacobians.shape != jacobians.shape:
        raise ValueError("true_jacobians must match jacobians when provided")

    basin_means: Dict[int, np.ndarray] = {}
    for basin in sorted({int(item) for item in basin_labels.tolist() if int(item) >= 0}):
        mask = basin_labels == basin
        if int(mask.sum()) < min_states:
            continue
        basin_means[basin] = jacobians[mask].mean(axis=0)

    class_counts = Counter(class_labels.tolist())
    class_means: Dict[object, np.ndarray] = {}
    class_major_basin: Dict[object, int] = {}
    for class_id, count in class_counts.items():
        if count < min_states:
            continue
        mask = class_labels == class_id
        basin_counter = Counter(int(item) for item in basin_labels[mask].tolist() if int(item) >= 0)
        if not basin_counter:
            continue
        class_means[class_id] = jacobians[mask].mean(axis=0)
        class_major_basin[class_id] = basin_counter.most_common(1)[0][0]

    class_ids = list(class_means.keys())
    if not class_ids:
        return {
            "jacobian_state_count": float(jacobians.shape[0]),
            "jacobian_class_count": 0.0,
            "jacobian_support_vs_basin_fro_mean": None,
            "jacobian_within_basin_fro_mean": None,
            "jacobian_between_basin_fro_mean": None,
            "jacobian_between_over_within": None,
            "jacobian_support_vs_true_fro_mean": None,
            "jacobian_basin_vs_true_fro_mean": None,
        }

    support_vs_basin = []
    for class_id in class_ids:
        basin = class_major_basin[class_id]
        basin_mean = basin_means.get(basin)
        if basin_mean is None:
            continue
        support_vs_basin.append(float(np.linalg.norm(class_means[class_id] - basin_mean, ord="fro")))

    within = []
    between = []
    for i, class_i in enumerate(class_ids):
        basin_i = class_major_basin[class_i]
        jac_i = class_means[class_i]
        for class_j in class_ids[i + 1 :]:
            basin_j = class_major_basin[class_j]
            dist = float(np.linalg.norm(jac_i - class_means[class_j], ord="fro"))
            if basin_i == basin_j:
                within.append(dist)
            else:
                between.append(dist)

    within_mean = float(np.mean(within)) if within else None
    between_mean = float(np.mean(between)) if between else None
    ratio = None
    if within_mean is not None and between_mean is not None and within_mean > 0.0:
        ratio = between_mean / within_mean

    support_vs_true = None
    basin_vs_true = None
    if true_jacobians is not None:
        true_basin_means: Dict[int, np.ndarray] = {}
        for basin, basin_mean in basin_means.items():
            basin_mask = basin_labels == basin
            true_basin_means[basin] = true_jacobians[basin_mask].mean(axis=0)
        support_true_dists = []
        for class_id in class_ids:
            basin = class_major_basin[class_id]
            true_mean = true_basin_means.get(basin)
            if true_mean is None:
                continue
            support_true_dists.append(
                float(np.linalg.norm(class_means[class_id] - true_mean, ord="fro"))
            )
        basin_true_dists = [
            float(np.linalg.norm(basin_mean - true_basin_means[basin], ord="fro"))
            for basin, basin_mean in basin_means.items()
            if basin in true_basin_means
        ]
        support_vs_true = float(np.mean(support_true_dists)) if support_true_dists else None
        basin_vs_true = float(np.mean(basin_true_dists)) if basin_true_dists else None

    return {
        "jacobian_state_count": float(jacobians.shape[0]),
        "jacobian_class_count": float(len(class_ids)),
        "jacobian_support_vs_basin_fro_mean": (
            float(np.mean(support_vs_basin)) if support_vs_basin else None
        ),
        "jacobian_within_basin_fro_mean": within_mean,
        "jacobian_between_basin_fro_mean": between_mean,
        "jacobian_between_over_within": ratio,
        "jacobian_support_vs_true_fro_mean": support_vs_true,
        "jacobian_basin_vs_true_fro_mean": basin_vs_true,
    }


def _prefix_metrics(prefix: str, metrics: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _masked_class_metrics(
    class_labels: np.ndarray,
    basin_labels: np.ndarray,
    subset_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    class_flat = class_labels.reshape(-1)[subset_mask]
    basin_flat = basin_labels.reshape(-1)[subset_mask]
    if class_flat.size == 0:
        return {
            "h_basin_given_class": None,
            "h_class_given_basin": None,
            "class_nmi": None,
            "u_exact": None,
            "unique_class_count": 0.0,
        }
    return _class_metrics(class_flat.reshape(-1, 1), basin_flat.reshape(-1, 1))


def _block_layout_from_model(model) -> Tuple[int, Sequence[int]]:
    if hasattr(model, "d_global") and hasattr(model, "num_basins") and hasattr(model, "d_basin"):
        return int(getattr(model, "d_global")), [int(getattr(model, "d_basin"))] * int(getattr(model, "num_basins"))
    soft_block_sizes = getattr(model, "_soft_block_sizes", None)
    if isinstance(soft_block_sizes, list) and soft_block_sizes:
        return 0, [int(size) for size in soft_block_sizes]
    block_sizes = getattr(model, "_k_block_sizes", None)
    if isinstance(block_sizes, list) and block_sizes:
        return 0, [int(size) for size in block_sizes]
    return 0, []


def _stringify_value(scheme: str, value: float) -> str:
    if scheme == "topk":
        return str(int(value))
    return f"{value:.6g}"


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _ensure_matplotlib():
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  # pylint: disable=unused-import


def _support_slug(support_name: str) -> str:
    return (
        support_name.replace(":", "__")
        .replace("/", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def _label_codes(labels: np.ndarray) -> Tuple[np.ndarray, List[object]]:
    unique = list(dict.fromkeys(labels.tolist()))
    mapping = {label: idx for idx, label in enumerate(unique)}
    codes = np.asarray([mapping[label] for label in labels.tolist()], dtype=np.int64)
    return codes, unique


def _save_phase_portrait_labels(
    states: np.ndarray,
    labels: np.ndarray,
    path: Path,
    *,
    title: str,
    max_points: int,
) -> None:
    if states.shape[1] < 2 or states.shape[0] == 0:
        return
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    if max_points > 0 and states.shape[0] > max_points:
        keep = np.sort(rng.choice(states.shape[0], size=max_points, replace=False))
        states = states[keep]
        labels = labels[keep]

    codes, unique = _label_codes(labels)
    cmap = plt.get_cmap("tab20", max(len(unique), 1))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    scatter = ax.scatter(
        states[:, 0],
        states[:, 1],
        c=codes,
        cmap=cmap,
        s=8,
        alpha=0.7,
        linewidths=0.0,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label("Support family id")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_entropy_map(
    states: np.ndarray,
    labels: np.ndarray,
    path: Path,
    *,
    title: str,
    bins: int = 48,
) -> None:
    if states.shape[1] < 2 or states.shape[0] == 0:
        return
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    codes, _ = _label_codes(labels)
    x = states[:, 0]
    y = states[:, 1]
    x_edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    y_edges = np.linspace(float(y.min()), float(y.max()), bins + 1)
    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, bins - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, bins - 1)

    entropy = np.full((bins, bins), np.nan, dtype=np.float32)
    for ix in range(bins):
        for iy in range(bins):
            mask = (x_bin == ix) & (y_bin == iy)
            if not np.any(mask):
                continue
            counts = Counter(codes[mask].tolist())
            entropy[iy, ix] = float(_entropy_from_counter(counts))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(
        entropy,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Local support entropy")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_switch_raster(
    class_labels: np.ndarray,
    basin_labels: np.ndarray,
    path: Path,
    *,
    title: str,
    max_trajectories: int,
) -> None:
    if class_labels.ndim != 2 or basin_labels.ndim != 2:
        return
    transfer_indices = [
        idx
        for idx in range(class_labels.shape[0])
        if np.any(basin_labels[idx, 1:] != basin_labels[idx, :-1])
    ]
    if not transfer_indices:
        return
    transfer_indices = transfer_indices[:max_trajectories]
    selected = class_labels[transfer_indices]
    flat_codes, _ = _label_codes(selected.reshape(-1))
    raster = flat_codes.reshape(selected.shape)

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9, max(3, 0.18 * len(transfer_indices))))
    im = ax.imshow(raster, aspect="auto", interpolation="nearest", cmap="tab20")
    for row_index, traj_index in enumerate(transfer_indices):
        true_switches = np.flatnonzero(basin_labels[traj_index, 1:] != basin_labels[traj_index, :-1]) + 1
        if true_switches.size > 0:
            ax.scatter(
                [int(true_switches[0])],
                [row_index],
                marker="|",
                s=200,
                linewidths=2.0,
                color="white",
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Support family id")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Transfer trajectory")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_basin_confusion_heatmap(
    class_labels: np.ndarray,
    basin_labels: np.ndarray,
    path: Path,
    *,
    title: str,
    max_classes: int = 20,
) -> None:
    flat_classes = class_labels.reshape(-1)
    flat_basins = basin_labels.reshape(-1)
    class_counts = Counter(flat_classes.tolist())
    top_classes = [label for label, _count in class_counts.most_common(max_classes)]
    if not top_classes:
        return
    class_to_index = {label: idx for idx, label in enumerate(top_classes)}
    valid_mask = np.asarray([label in class_to_index for label in flat_classes.tolist()], dtype=bool)
    if not np.any(valid_mask):
        return

    basins = sorted({int(item) for item in flat_basins[valid_mask].tolist() if int(item) >= 0})
    if not basins:
        return
    basin_to_index = {basin: idx for idx, basin in enumerate(basins)}
    matrix = np.zeros((len(basins), len(top_classes)), dtype=np.float32)
    for basin, class_label in zip(flat_basins[valid_mask].tolist(), flat_classes[valid_mask].tolist()):
        basin_idx = basin_to_index.get(int(basin))
        class_idx = class_to_index.get(class_label)
        if basin_idx is None or class_idx is None:
            continue
        matrix[basin_idx, class_idx] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = matrix / np.maximum(row_sums, 1.0)

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(max(6, 0.4 * len(top_classes)), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Fraction of states")
    ax.set_xticks(np.arange(len(top_classes)))
    ax.set_xticklabels([str(index) for index in range(len(top_classes))], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(basins)))
    ax.set_yticklabels([f"B{basin}" for basin in basins])
    ax.set_xlabel("Top support families")
    ax.set_ylabel("True basin")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_operator_distance_heatmap(
    latents: np.ndarray,
    basin_labels: np.ndarray,
    class_labels: np.ndarray,
    path: Path,
    *,
    title: str,
    ridge_lambda: float,
    min_transitions: int,
    max_classes: int = 20,
) -> None:
    basin_ops, class_ops, class_major_basin, class_counts = _fit_operator_families(
        latents,
        basin_labels,
        class_labels,
        ridge_lambda=ridge_lambda,
        min_transitions=min_transitions,
    )
    del basin_ops
    if not class_ops:
        return

    ordered_classes = [
        class_id for class_id, _count in class_counts.most_common()
        if class_id in class_ops
    ][:max_classes]
    if not ordered_classes:
        return

    matrix = np.zeros((len(ordered_classes), len(ordered_classes)), dtype=np.float32)
    for i, class_i in enumerate(ordered_classes):
        for j, class_j in enumerate(ordered_classes):
            matrix[i, j] = float(
                np.linalg.norm(class_ops[class_i] - class_ops[class_j], ord="fro")
            )

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Operator Frobenius distance")
    tick_labels = [f"C{i}/B{class_major_basin[class_id]}" for i, class_id in enumerate(ordered_classes)]
    ax.set_xticks(np.arange(len(ordered_classes)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ordered_classes)))
    ax.set_yticklabels(tick_labels)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_visual_suite(
    *,
    visual_dir: Path,
    trajectories: torch.Tensor,
    latents: np.ndarray,
    basin_labels: np.ndarray,
    family_labels: np.ndarray,
    support_name: str,
    ridge_lambda: float,
    min_operator_transitions: int,
    max_points: int,
    max_switch_trajectories: int,
) -> None:
    states = trajectories.reshape(-1, trajectories.shape[-1]).cpu().numpy()
    flat_family = family_labels.reshape(-1).astype(object)
    visual_dir.mkdir(parents=True, exist_ok=True)

    _save_phase_portrait_labels(
        states,
        flat_family,
        visual_dir / f"{_support_slug(support_name)}__support_phase_portrait.png",
        title=f"Support families: {support_name}",
        max_points=max_points,
    )
    _save_entropy_map(
        states,
        flat_family,
        visual_dir / f"{_support_slug(support_name)}__support_entropy_map.png",
        title=f"Support entropy map: {support_name}",
    )
    _save_switch_raster(
        family_labels.astype(object),
        basin_labels,
        visual_dir / f"{_support_slug(support_name)}__support_switch_raster.png",
        title=f"Support-switch raster: {support_name}",
        max_trajectories=max_switch_trajectories,
    )
    _save_basin_confusion_heatmap(
        family_labels.astype(object),
        basin_labels,
        visual_dir / f"{_support_slug(support_name)}__support_basin_confusion.png",
        title=f"Basin/support-family confusion: {support_name}",
    )
    _save_operator_distance_heatmap(
        latents,
        basin_labels,
        family_labels.astype(object),
        visual_dir / f"{_support_slug(support_name)}__operator_distance_heatmap.png",
        title=f"Operator distances: {support_name}",
        ridge_lambda=ridge_lambda,
        min_transitions=min_operator_transitions,
    )


def _write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["root_label"]), str(row["support_scheme"]), str(row["subset"]))].append(row)

    lines = [
        "# Interpretability Summary",
        "",
        "This summary uses state-level support metrics from the interpretability study plan.",
        "",
        (
            "| root | support | subset | mean H(B|S) | mean H(S|B) | mean NMI | "
            "mean U_exact | mean H(F|B) | mean own/base | mean freeze/base | "
            "mean persistence | mean op between/within | mean jac between/within | mean switch delay |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (root_label, support_scheme, subset), group_rows in sorted(grouped.items()):
        lines.append(
            f"| `{root_label}` | `{support_scheme}` | `{subset}` | "
            f"{_format_float(_safe_mean(row.get('h_basin_given_support') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('h_support_given_basin') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('support_nmi') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('u_exact') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('family_h_family_given_basin') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('support_projection_self_over_base') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('support_freeze_longest_self_over_base') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('support_persistence') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('operator_between_over_within') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('jacobian_between_over_within') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('support_switch_delay_mean') for row in group_rows))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    freeze_support_horizons: Sequence[int],
    support_definitions: Sequence[Tuple[str, float]],
    visual_supports: Optional[set[str]],
    num_specs: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "rows_csv": args.rows_csv,
                "root_labels": list(root_labels),
                "systems": list(systems),
                "seeds": list(seeds),
                "num_trajectories": args.num_trajectories,
                "trajectory_length": args.trajectory_length,
                "family_jaccard_threshold": args.family_jaccard_threshold,
                "freeze_support_horizons": list(freeze_support_horizons),
                "max_freeze_states": args.max_freeze_states,
                "max_jacobian_states": args.max_jacobian_states,
                "min_jacobian_states": args.min_jacobian_states,
                "save_visuals": bool(args.save_visuals),
                "visual_supports": sorted(visual_supports) if visual_supports is not None else [],
                "visual_max_points": args.visual_max_points,
                "visual_max_switch_trajectories": args.visual_max_switch_trajectories,
                "support_definitions": [
                    {"scheme": scheme, "value": value} for scheme, value in support_definitions
                ],
                "num_runs": num_specs,
                "completed_runs": completed_specs,
                "remaining_runs": max(0, num_specs - completed_specs),
                "num_rows": len(rows),
                "num_failures": len(failures),
                "status": status,
            },
            indent=2,
        )
    )


def _write_progress(
    path: Path,
    *,
    completed_specs: int,
    num_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    elapsed_seconds: float,
    last_spec: Optional[RunSpec],
    last_status: Optional[str],
    last_error: Optional[str],
) -> None:
    payload: Dict[str, object] = {
        "completed_runs": completed_specs,
        "num_runs": num_specs,
        "remaining_runs": max(0, num_specs - completed_specs),
        "num_rows": len(rows),
        "num_failures": len(failures),
        "elapsed_seconds": elapsed_seconds,
    }
    if last_spec is not None:
        payload["last_completed_spec"] = {
            "root_label": last_spec.root_label,
            "system_key": last_spec.system_key,
            "seed": last_spec.seed,
            "run_dir": last_spec.run_dir,
            "status": last_status,
        }
    if last_error:
        payload["last_error"] = last_error
    path.write_text(json.dumps(payload, indent=2))


def _flush_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    freeze_support_horizons: Sequence[int],
    support_definitions: Sequence[Tuple[str, float]],
    visual_supports: Optional[set[str]],
    num_specs: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
    last_spec: Optional[RunSpec],
    last_status: Optional[str],
    last_error: Optional[str],
) -> None:
    _write_csv(output_dir / "interpretability_rows.csv", rows)
    _write_summary(output_dir / "interpretability_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(list(failures), indent=2))
    _write_manifest(
        output_dir / "manifest.json",
        args=args,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        freeze_support_horizons=freeze_support_horizons,
        support_definitions=support_definitions,
        visual_supports=visual_supports,
        num_specs=num_specs,
        completed_specs=completed_specs,
        rows=rows,
        failures=failures,
        status=status,
    )
    _write_progress(
        output_dir / "progress.json",
        completed_specs=completed_specs,
        num_specs=num_specs,
        rows=rows,
        failures=failures,
        elapsed_seconds=elapsed_seconds,
        last_spec=last_spec,
        last_status=last_status,
        last_error=last_error,
    )


def reduce_run(
    spec: RunSpec,
    *,
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
    endpoint_rollout_steps: int,
    device: str,
    ridge_lambda: float,
    min_operator_transitions: int,
    support_definitions: Sequence[Tuple[str, float]],
    family_jaccard_threshold: float,
    freeze_support_horizons: Sequence[int],
    max_freeze_states: int,
    max_jacobian_states: int,
    min_jacobian_states: int,
    save_visuals: bool,
    visual_supports: Optional[set[str]],
    visual_output_dir: Optional[Path],
    visual_max_points: int,
    visual_max_switch_trajectories: int,
) -> List[Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    cfg, env, model = _load_checkpoint_model(checkpoint_path, spec.system_key, device)
    trajectories = _generate_observation_trajectories(
        env,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        eval_seed=eval_seed,
    )
    basin_labels, centers, label_source = _label_sequences_and_centers(
        env,
        trajectories,
        system_key=spec.system_key,
        endpoint_rollout_steps=endpoint_rollout_steps,
    )
    latents = _encode_trajectories(model, trajectories, device)
    block_offset, block_sizes = _block_layout_from_model(model)
    subset_masks = _margin_subsets(trajectories, centers)
    sampled_jacobian_indices, jacobian_subset_masks = _sample_state_indices(
        subset_masks,
        max_states_per_subset=max_jacobian_states,
        sample_seed=eval_seed + spec.seed,
    )
    flat_states = trajectories.reshape(-1, trajectories.shape[-1])
    sampled_jacobians: Optional[np.ndarray] = None
    sampled_true_jacobians: Optional[np.ndarray] = None
    sampled_basin_labels = basin_labels.reshape(-1).cpu().numpy()[sampled_jacobian_indices]
    if sampled_jacobian_indices.size > 0:
        sampled_states = flat_states[sampled_jacobian_indices.tolist()]
        sampled_jacobians = _compute_state_jacobians(model, sampled_states, device=device)
        if hasattr(env, "get_local_jacobian"):
            try:
                with torch.no_grad():
                    true_jacobians = env.get_local_jacobian(sampled_states.to(dtype=torch.float32))
                if isinstance(true_jacobians, torch.Tensor):
                    sampled_true_jacobians = true_jacobians.detach().cpu().numpy().astype(np.float32, copy=False)
            except Exception:
                sampled_true_jacobians = None

    rows: List[Dict[str, object]] = []
    basin_labels_np = basin_labels.cpu().numpy()
    default_visual_emitted = False

    for scheme, value in support_definitions:
        support_name = f"{scheme}:{_stringify_value(scheme, value)}"
        support_mask = _support_mask(latents, scheme=scheme, value=value)
        support_codes = _support_keys(support_mask)
        transition_metrics = support_transition_metrics(support_mask, support_codes, basin_labels_np)
        support_switch_metrics = _prefix_metrics(
            "support_",
            switch_timing_metrics(support_codes.astype(object), basin_labels_np),
        )
        support_operator_metrics = operator_distance_summary(
            latents,
            basin_labels_np,
            support_codes,
            ridge_lambda=ridge_lambda,
            min_transitions=min_operator_transitions,
        )
        family_labels = support_family_labels(support_mask, min_jaccard=family_jaccard_threshold)
        family_transition_metrics = class_transition_metrics(family_labels.astype(object), basin_labels_np)
        family_operator_metrics = operator_distance_summary(
            latents,
            basin_labels_np,
            family_labels.astype(object),
            ridge_lambda=ridge_lambda,
            min_transitions=min_operator_transitions,
        )
        family_switch_metrics = _prefix_metrics(
            "family_",
            switch_timing_metrics(family_labels.astype(object), basin_labels_np),
        )
        deep_support_templates = canonical_support_masks_by_basin(
            support_mask,
            basin_labels_np,
            subset_masks["deep"],
        )
        deep_projection_metrics = support_projection_metrics(
            model,
            latents,
            trajectories,
            basin_labels_np,
            deep_support_templates,
            subset_masks["deep"],
            device=device,
        )
        deep_freeze_metrics = freeze_support_rollout_metrics(
            model,
            latents,
            trajectories,
            basin_labels_np,
            deep_support_templates,
            subset_masks["deep"],
            device=device,
            horizons=freeze_support_horizons,
            max_states_per_horizon=max_freeze_states,
            sample_seed=eval_seed + spec.seed,
        )

        group_labels = None
        group_transition_metrics: Dict[str, Optional[float]] = {}
        group_operator_metrics: Dict[str, Optional[float]] = {}
        group_switch_metrics: Dict[str, Optional[float]] = {}
        if block_sizes:
            group_labels = _dominant_group_labels(latents, block_sizes, offset=block_offset)
            group_transition_metrics = class_transition_metrics(group_labels.astype(object), basin_labels_np)
            group_operator_metrics = operator_distance_summary(
                latents,
                basin_labels_np,
                group_labels.astype(object),
                ridge_lambda=ridge_lambda,
                min_transitions=min_operator_transitions,
            )
            group_switch_metrics = _prefix_metrics(
                "group_",
                switch_timing_metrics(group_labels.astype(object), basin_labels_np),
            )

        if save_visuals and visual_output_dir is not None:
            should_save_visuals = False
            if visual_supports is None and not default_visual_emitted:
                should_save_visuals = True
                default_visual_emitted = True
            elif visual_supports is not None and support_name in visual_supports:
                should_save_visuals = True
            if should_save_visuals:
                save_visual_suite(
                    visual_dir=visual_output_dir / spec.root_label / f"{spec.system_key}_seed{spec.seed}",
                    trajectories=trajectories,
                    latents=latents,
                    basin_labels=basin_labels_np,
                    family_labels=family_labels,
                    support_name=support_name,
                    ridge_lambda=ridge_lambda,
                    min_operator_transitions=min_operator_transitions,
                    max_points=visual_max_points,
                    max_switch_trajectories=visual_max_switch_trajectories,
                )

        for subset_name, subset_mask in subset_masks.items():
            support_metrics = _masked_class_metrics(support_codes, basin_labels_np, subset_mask)
            jacobian_metrics: Dict[str, Optional[float]] = {
                "jacobian_state_count": 0.0,
                "jacobian_class_count": 0.0,
                "jacobian_support_vs_basin_fro_mean": None,
                "jacobian_within_basin_fro_mean": None,
                "jacobian_between_basin_fro_mean": None,
                "jacobian_between_over_within": None,
                "jacobian_support_vs_true_fro_mean": None,
                "jacobian_basin_vs_true_fro_mean": None,
            }
            if sampled_jacobians is not None and subset_name in jacobian_subset_masks:
                subset_local_mask = jacobian_subset_masks[subset_name]
                if bool(np.any(subset_local_mask)):
                    subset_true_jacobians = (
                        sampled_true_jacobians[subset_local_mask]
                        if sampled_true_jacobians is not None
                        else None
                    )
                    jacobian_metrics = jacobian_distance_summary(
                        sampled_jacobians[subset_local_mask],
                        sampled_basin_labels[subset_local_mask],
                        support_codes.reshape(-1)[sampled_jacobian_indices][subset_local_mask].astype(object),
                        min_states=min_jacobian_states,
                        true_jacobians=subset_true_jacobians,
                    )
            row = {
                "root_label": spec.root_label,
                "system_key": spec.system_key,
                "system_name": spec.system_name,
                "seed": spec.seed,
                "run_dir": spec.run_dir,
                "support_scheme": support_name,
                "subset": subset_name,
                "label_source": label_source,
                "num_states": int(subset_mask.sum()),
                "h_basin_given_support": support_metrics["h_basin_given_class"],
                "h_support_given_basin": support_metrics["h_class_given_basin"],
                "support_nmi": support_metrics["class_nmi"],
                "u_exact": support_metrics["u_exact"],
                "unique_support_count": support_metrics["unique_class_count"],
                "family_jaccard_threshold": float(family_jaccard_threshold),
                "family_h_basin_given_family": None,
                "family_h_family_given_basin": None,
                "family_nmi": None,
                "family_u": None,
                "family_unique_count": None,
                "family_persistence": None,
                "family_switch_off_basin_switch": None,
                "family_operator_between_over_within": None,
                "mean_support_size": float(support_mask.reshape(-1, support_mask.shape[-1])[subset_mask].sum(axis=1).mean())
                if int(subset_mask.sum()) > 0
                else None,
                **transition_metrics,
                **support_switch_metrics,
                **support_operator_metrics,
                **jacobian_metrics,
                "support_projection_state_count": None,
                "support_projection_template_count": None,
                "support_projection_base_mse": None,
                "support_projection_self_mse": None,
                "support_projection_wrong_mse": None,
                "support_projection_self_over_base": None,
                "support_projection_wrong_over_base": None,
                "support_projection_wrong_minus_self": None,
                **{
                    key: (value if subset_name == "deep" else None)
                    for key, value in deep_freeze_metrics.items()
                },
                "group_h_basin_given_group": None,
                "group_h_group_given_basin": None,
                "group_nmi": None,
                "group_unique_count": None,
                "group_persistence": None,
                "group_switch_off_basin_switch": None,
                "group_operator_between_over_within": None,
                "family_switch_trajectory_count": None,
                "family_switch_detected_fraction": None,
                "family_switch_miss_fraction": None,
                "family_switch_delay_mean": None,
                "family_switch_delay_abs_mean": None,
                "family_switch_false_switches_mean": None,
                "family_switch_chatter_mean": None,
                "family_switch_pre_dwell_mean": None,
                "family_switch_post_dwell_mean": None,
                "group_switch_trajectory_count": None,
                "group_switch_detected_fraction": None,
                "group_switch_miss_fraction": None,
                "group_switch_delay_mean": None,
                "group_switch_delay_abs_mean": None,
                "group_switch_false_switches_mean": None,
                "group_switch_chatter_mean": None,
                "group_switch_pre_dwell_mean": None,
                "group_switch_post_dwell_mean": None,
            }
            family_metrics = _masked_class_metrics(family_labels.astype(object), basin_labels_np, subset_mask)
            row.update(
                {
                    "family_h_basin_given_family": family_metrics["h_basin_given_class"],
                    "family_h_family_given_basin": family_metrics["h_class_given_basin"],
                    "family_nmi": family_metrics["class_nmi"],
                    "family_u": family_metrics["u_exact"],
                    "family_unique_count": family_metrics["unique_class_count"],
                    "family_persistence": family_transition_metrics.get("support_persistence"),
                    "family_switch_off_basin_switch": family_transition_metrics.get(
                        "support_switch_off_basin_switch"
                    ),
                    "family_operator_between_over_within": family_operator_metrics.get(
                        "operator_between_over_within"
                    ),
                    **family_switch_metrics,
                }
            )
            if subset_name == "deep":
                row.update(deep_projection_metrics)
            if group_labels is not None:
                group_metrics = _masked_class_metrics(group_labels.astype(object), basin_labels_np, subset_mask)
                row.update(
                    {
                        "group_h_basin_given_group": group_metrics["h_basin_given_class"],
                        "group_h_group_given_basin": group_metrics["h_class_given_basin"],
                        "group_nmi": group_metrics["class_nmi"],
                        "group_unique_count": group_metrics["unique_class_count"],
                        "group_persistence": group_transition_metrics.get("support_persistence"),
                        "group_switch_off_basin_switch": group_transition_metrics.get(
                            "support_switch_off_basin_switch"
                        ),
                        "group_operator_between_over_within": group_operator_metrics.get(
                            "operator_between_over_within"
                        ),
                        **group_switch_metrics,
                    }
                )
            rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    root_labels = _parse_csv_strings(args.root_labels)
    systems = _parse_csv_strings(args.systems)
    seeds = _parse_csv_ints(args.seeds)
    support_definitions: List[Tuple[str, float]] = []
    support_definitions.extend(("absolute", value) for value in _parse_csv_floats(args.absolute_thresholds))
    support_definitions.extend(("relative", value) for value in _parse_csv_floats(args.relative_thresholds))
    support_definitions.extend(("topk", float(value)) for value in _parse_csv_ints(args.topk_values))
    freeze_support_horizons = _parse_csv_ints(args.freeze_support_horizons)
    visual_supports = _parse_visual_supports(args.visual_supports)

    specs = _load_latest_specs(
        Path(args.rows_csv),
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    num_specs = len(specs)
    start_time = time.time()
    progress_every_runs = max(1, int(args.progress_every_runs))
    flush_every_runs = max(0, int(args.flush_every_runs))
    last_completed_spec: Optional[RunSpec] = None
    last_completed_status: Optional[str] = None
    last_completed_error: Optional[str] = None
    _flush_outputs(
        output_dir,
        args=args,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        freeze_support_horizons=freeze_support_horizons,
        support_definitions=support_definitions,
        visual_supports=visual_supports,
        num_specs=num_specs,
        completed_specs=0,
        rows=rows,
        failures=failures,
        status="running",
        elapsed_seconds=0.0,
        last_spec=None,
        last_status=None,
        last_error=None,
    )
    for spec_index, spec in enumerate(specs, start=1):
        last_completed_spec = spec
        last_completed_status = "ok"
        last_completed_error = None
        try:
            spec_rows = reduce_run(
                spec,
                num_trajectories=args.num_trajectories,
                trajectory_length=args.trajectory_length,
                eval_seed=args.eval_seed,
                endpoint_rollout_steps=args.endpoint_rollout_steps,
                device=args.device,
                ridge_lambda=args.ridge_lambda,
                min_operator_transitions=args.min_operator_transitions,
                support_definitions=support_definitions,
                family_jaccard_threshold=args.family_jaccard_threshold,
                freeze_support_horizons=freeze_support_horizons,
                max_freeze_states=args.max_freeze_states,
                max_jacobian_states=args.max_jacobian_states,
                min_jacobian_states=args.min_jacobian_states,
                save_visuals=bool(args.save_visuals),
                visual_supports=visual_supports,
                visual_output_dir=(output_dir / "visuals") if args.save_visuals else None,
                visual_max_points=args.visual_max_points,
                visual_max_switch_trajectories=args.visual_max_switch_trajectories,
            )
            rows.extend(spec_rows)
        except Exception as exc:  # pragma: no cover - reducer should keep going across bad runs
            last_completed_status = "failed"
            last_completed_error = repr(exc)
            failures.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
                    "error": last_completed_error,
                }
            )
        elapsed_seconds = time.time() - start_time
        if spec_index % progress_every_runs == 0 or spec_index == num_specs:
            print(
                (
                    f"[{spec_index}/{num_specs}] {last_completed_status} "
                    f"root={spec.root_label} system={spec.system_key} seed={spec.seed} "
                    f"rows={len(rows)} failures={len(failures)} elapsed_s={elapsed_seconds:.1f}"
                ),
                flush=True,
            )
        if flush_every_runs > 0 and (spec_index % flush_every_runs == 0 or spec_index == num_specs):
            _flush_outputs(
                output_dir,
                args=args,
                root_labels=root_labels,
                systems=systems,
                seeds=seeds,
                freeze_support_horizons=freeze_support_horizons,
                support_definitions=support_definitions,
                visual_supports=visual_supports,
                num_specs=num_specs,
                completed_specs=spec_index,
                rows=rows,
                failures=failures,
                status="running" if spec_index < num_specs else "complete",
                elapsed_seconds=elapsed_seconds,
                last_spec=last_completed_spec,
                last_status=last_completed_status,
                last_error=last_completed_error,
            )

    _flush_outputs(
        output_dir,
        args=args,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        freeze_support_horizons=freeze_support_horizons,
        support_definitions=support_definitions,
        visual_supports=visual_supports,
        num_specs=num_specs,
        completed_specs=num_specs,
        rows=rows,
        failures=failures,
        status="complete",
        elapsed_seconds=time.time() - start_time,
        last_spec=last_completed_spec,
        last_status=last_completed_status,
        last_error=last_completed_error,
    )
    print(
        json.dumps(
            {
                "num_runs": len(specs),
                "num_rows": len(rows),
                "num_failures": len(failures),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
