#!/usr/bin/env python3
"""Reduce basin-support metrics for the fixed transition-rich shortlist.

This script computes the paper-facing basin-support metrics requested for the
fixed 17-system shortlist:

- support-group purity
- recurring-support / retained-trajectory coverage
- local-vs-global-vs-shuffled H-step NRMSE

It reads a collected forecasting CSV, selects the latest run for each
``(root_label, system_key, seed)`` triple, regenerates a shared held-out
trajectory corpus from the saved checkpoint, and writes per-run plus summary
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    get_transition_rich_basin_count,
)
from skae.config import Config
from skae.data import VectorWrapper, make_env
from skae.model import make_model
from tools.evaluate_lqr_readiness import _make_projection_basis, _nrmse, ridge_fit_row_linear

EPS = 1e-9


@dataclass(frozen=True)
class RunSpec:
    root_label: str
    system_key: str
    system_name: str
    seed: int
    run_dir: str
    root_path: str
    train_env_name: str


@dataclass
class RunMetrics:
    root_label: str
    system_key: str
    system_name: str
    seed: int
    run_dir: str
    checkpoint_path: str
    num_trajectories: int
    trajectory_length: int
    support_group_count: int
    retained_support_group_count: int
    retained_trajectory_count: int
    retained_trajectory_coverage: float
    support_group_purity: float
    weighted_support_group_purity: float
    local_1step_nrmse: Optional[float]
    global_1step_nrmse: Optional[float]
    shuffled_1step_nrmse: Optional[float]
    local_h_nrmse: Optional[float]
    global_h_nrmse: Optional[float]
    shuffled_h_nrmse: Optional[float]
    local_beats_global_h: Optional[bool]
    local_beats_shuffled_h: Optional[bool]
    status: str
    note: str = ""


@dataclass
class EvalCorpus:
    system_key: str
    obs_trajectories: torch.Tensor
    endpoint_basin_labels: torch.Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True, help="Forecasting rows CSV used to discover runs")
    parser.add_argument("--output_dir", required=True, help="Directory for reduced metric artifacts")
    parser.add_argument(
        "--root_labels",
        default="lista_dense_basin_partition,lista_blockdiag_basin_partition",
        help="Comma-separated root labels to include",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated integer seed filter; blank keeps every seed in the CSV",
    )
    parser.add_argument(
        "--systems",
        default="",
        help="Optional comma-separated system_key filter; blank keeps every system in the CSV",
    )
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--support_threshold", type=float, default=1e-3)
    parser.add_argument("--retained_min_total", type=int, default=5)
    parser.add_argument("--retained_min_train", type=int, default=3)
    parser.add_argument("--retained_min_test", type=int, default=1)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--max_state_dim", type=int, default=32)
    parser.add_argument("--horizon_h", type=int, default=20)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num_workers", type=int, default=1)
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


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
    seeds: Sequence[int],
    systems: Sequence[str],
) -> List[RunSpec]:
    selected_roots = set(root_labels)
    selected_seeds = set(seeds)
    selected_systems = set(systems)
    best_rows: Dict[Tuple[str, str, int], Dict[str, str]] = {}

    with rows_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            root_label = row["root_label"]
            if root_label not in selected_roots:
                continue
            seed = int(row["seed"])
            if selected_seeds and seed not in selected_seeds:
                continue
            if selected_systems and row["system_key"] not in selected_systems:
                continue
            key = (root_label, row["system_key"], seed)
            incumbent = best_rows.get(key)
            if incumbent is None or _run_timestamp_key(row["run_dir"]) > _run_timestamp_key(incumbent["run_dir"]):
                best_rows[key] = row

    specs = [
        RunSpec(
            root_label=row["root_label"],
            system_key=row["system_key"],
            system_name=row["system_name"],
            seed=int(row["seed"]),
            run_dir=row["run_dir"],
            root_path=row["root_path"],
            train_env_name=row["train_env_name"],
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return cfg, env, model


def _load_checkpoint_env(
    checkpoint_path: Optional[Path],
    system_key: str,
):
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = Config.from_dict(checkpoint["config"])
    else:
        cfg = Config()
    cfg.ENV.ENV_NAME = system_key
    env = make_env(cfg)
    return cfg, env


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


def _support_tuple(z_traj: torch.Tensor, threshold: float) -> Tuple[int, ...]:
    votes = (z_traj.abs() > threshold).float().mean(dim=0)
    return tuple(int(value) for value in (votes > 0.5).cpu().tolist())


def _long_rollout(env, states: torch.Tensor, steps: int) -> torch.Tensor:
    current = states.clone()
    for _ in range(max(0, steps)):
        current = env.step(current)
    return current


def _label_from_native_method(method, states: torch.Tensor) -> torch.Tensor:
    labels = method(states)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    return labels.to(dtype=torch.long)


def _assign_nearest_centers(states: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return torch.cdist(states, centers).argmin(dim=1).to(dtype=torch.long)


def _coerce_center_rows(
    raw_centers,
    *,
    state_dim: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if isinstance(raw_centers, torch.Tensor):
        centers = raw_centers
        if centers.ndim == 1:
            centers = centers.unsqueeze(0)
        if centers.ndim != 2 or centers.shape[1] < state_dim:
            return None
        return centers[:, :state_dim].to(dtype=dtype)

    if not isinstance(raw_centers, (list, tuple)) or not raw_centers:
        return None

    rows: List[List[float]] = []
    for item in raw_centers:
        if isinstance(item, torch.Tensor):
            values = item.detach().cpu().reshape(-1).tolist()
        elif isinstance(item, (list, tuple)):
            values = list(item)
        else:
            return None
        if len(values) < state_dim:
            return None
        rows.append([float(values[idx]) for idx in range(state_dim)])
    return torch.tensor(rows, dtype=dtype)


def _extract_catalog_centers(
    env,
    *,
    basin_count: int,
    state_dim: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    system = getattr(env, "system", None)
    system_name = str(getattr(env, "system_name", ""))
    if system is None:
        return None

    if system_name == "arrested_spiral" and hasattr(system, "well_centers"):
        centers = _coerce_center_rows(system.well_centers, state_dim=state_dim, dtype=dtype)
        if centers is not None and centers.shape[0] + 1 == basin_count:
            origin = torch.zeros((1, state_dim), dtype=dtype)
            return torch.cat([centers, origin], dim=0)

    for attr_name in (
        "points",
        "centers",
        "well_centers",
        "room_centers",
        "dipoles",
        "patterns",
        "_wells",
        "wells",
    ):
        if not hasattr(system, attr_name):
            continue
        centers = _coerce_center_rows(getattr(system, attr_name), state_dim=state_dim, dtype=dtype)
        if centers is not None and centers.shape[0] == basin_count:
            return centers
    return None


def _kmeans_centers(points: torch.Tensor, num_centers: int, num_iters: int = 25) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError("points must have shape [N, dim]")
    if points.shape[0] < num_centers:
        raise ValueError("Need at least as many points as requested centers")

    centers = [points[0]]
    while len(centers) < num_centers:
        current = torch.stack(centers, dim=0)
        dists = torch.cdist(points, current)
        min_dists = dists.min(dim=1).values
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


def _endpoint_basin_labels(
    env,
    obs_trajectories: torch.Tensor,
    *,
    basin_count: int,
    endpoint_rollout_steps: int,
) -> torch.Tensor:
    final_states = obs_trajectories[:, -1, :]

    if hasattr(env, "basin_label"):
        return _label_from_native_method(env.basin_label, final_states)

    if hasattr(env, "points"):
        centers = env.points
        if isinstance(centers, torch.Tensor) and centers.ndim == 2 and centers.shape[1] == final_states.shape[1]:
            return _assign_nearest_centers(final_states, centers.to(dtype=final_states.dtype))

    catalog_centers = _extract_catalog_centers(
        env,
        basin_count=basin_count,
        state_dim=final_states.shape[1],
        dtype=final_states.dtype,
    )
    if catalog_centers is not None:
        return _assign_nearest_centers(final_states, catalog_centers)

    converged = _long_rollout(env, final_states, endpoint_rollout_steps)

    estimated_centers = _kmeans_centers(converged, basin_count)
    return _assign_nearest_centers(converged, estimated_centers)


def _prepare_eval_corpus(
    spec: RunSpec,
    *,
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
    endpoint_rollout_steps: int,
) -> EvalCorpus:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    _, env = _load_checkpoint_env(checkpoint_path if checkpoint_path.exists() else None, spec.system_key)

    obs_trajectories = _generate_observation_trajectories(
        env,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        eval_seed=eval_seed,
    )
    basin_count = int(get_transition_rich_basin_count(spec.system_key))
    endpoint_basin_labels = _endpoint_basin_labels(
        env,
        obs_trajectories,
        basin_count=basin_count,
        endpoint_rollout_steps=endpoint_rollout_steps,
    ).cpu()
    return EvalCorpus(
        system_key=spec.system_key,
        obs_trajectories=obs_trajectories,
        endpoint_basin_labels=endpoint_basin_labels,
    )


def _global_train_test_split(num_items: int, test_fraction: float, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_items, dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(indices)

    n_test = max(1, int(round(test_fraction * num_items)))
    n_test = min(max(1, n_test), num_items - 1)
    test_ids = np.sort(indices[:n_test])
    train_ids = np.sort(indices[n_test:])
    return train_ids, test_ids


def _fit_linear_map(
    latent_trajectories: np.ndarray,
    trajectory_ids: Sequence[int],
    *,
    ridge_lambda: float,
    max_state_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.concatenate([latent_trajectories[tid][:-1] for tid in trajectory_ids], axis=0)
    y_train = np.concatenate([latent_trajectories[tid][1:] for tid in trajectory_ids], axis=0)

    centroid = x_train.mean(axis=0)
    xc_train = x_train - centroid
    yc_train = y_train - centroid

    basis = _make_projection_basis(xc_train, max_state_dim=max_state_dim)
    x_train_p = xc_train @ basis
    y_train_p = yc_train @ basis
    a_row = ridge_fit_row_linear(x_train_p, y_train_p, l2_reg=ridge_lambda)
    return centroid, basis, a_row


def _collect_eval_arrays(
    latent_trajectories: np.ndarray,
    trajectory_ids: Sequence[int],
    *,
    centroid: np.ndarray,
    basis: np.ndarray,
    a_row: np.ndarray,
    horizon_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one_true: List[np.ndarray] = []
    one_pred: List[np.ndarray] = []
    h_true: List[np.ndarray] = []
    h_pred: List[np.ndarray] = []
    a_h = np.linalg.matrix_power(a_row, horizon_h)

    for tid in trajectory_ids:
        traj = latent_trajectories[tid]
        x_proj = (traj[:-1] - centroid) @ basis
        y_proj = (traj[1:] - centroid) @ basis
        one_true.append(y_proj)
        one_pred.append(x_proj @ a_row)

        max_start = traj.shape[0] - horizon_h
        for start in range(max_start):
            x0_proj = (traj[start] - centroid) @ basis
            y_h_proj = (traj[start + horizon_h] - centroid) @ basis
            h_true.append(y_h_proj[None, :])
            h_pred.append((x0_proj @ a_h)[None, :])

    def _stack(chunks: List[np.ndarray], dim: int) -> np.ndarray:
        if not chunks:
            return np.empty((0, dim), dtype=np.float64)
        return np.concatenate(chunks, axis=0).astype(np.float64)

    dim = basis.shape[1]
    return (
        _stack(one_true, dim),
        _stack(one_pred, dim),
        _stack(h_true, dim),
        _stack(h_pred, dim),
    )


def _fit_and_eval_partition(
    latent_trajectories: np.ndarray,
    partition: Dict[str, Dict[str, List[int]]],
    *,
    ridge_lambda: float,
    max_state_dim: int,
    horizon_h: int,
) -> Tuple[Optional[float], Optional[float]]:
    all_one_true: List[np.ndarray] = []
    all_one_pred: List[np.ndarray] = []
    all_h_true: List[np.ndarray] = []
    all_h_pred: List[np.ndarray] = []

    for group in partition.values():
        train_ids = group["train"]
        test_ids = group["test"]
        if not train_ids or not test_ids:
            continue

        centroid, basis, a_row = _fit_linear_map(
            latent_trajectories,
            train_ids,
            ridge_lambda=ridge_lambda,
            max_state_dim=max_state_dim,
        )
        one_true, one_pred, h_true, h_pred = _collect_eval_arrays(
            latent_trajectories,
            test_ids,
            centroid=centroid,
            basis=basis,
            a_row=a_row,
            horizon_h=horizon_h,
        )
        if one_true.size:
            all_one_true.append(one_true)
            all_one_pred.append(one_pred)
        if h_true.size:
            all_h_true.append(h_true)
            all_h_pred.append(h_pred)

    if not all_one_true:
        return None, None

    one_true = np.concatenate(all_one_true, axis=0)
    one_pred = np.concatenate(all_one_pred, axis=0)
    h_true = np.concatenate(all_h_true, axis=0) if all_h_true else np.empty((0, 0), dtype=np.float64)
    h_pred = np.concatenate(all_h_pred, axis=0) if all_h_pred else np.empty((0, 0), dtype=np.float64)

    one_nrmse = _nrmse(one_true, one_pred)
    h_nrmse = _nrmse(h_true, h_pred) if h_true.size and h_pred.size else None
    return float(one_nrmse), (float(h_nrmse) if h_nrmse is not None else None)


def _build_shuffled_partition(
    retained_partition: Dict[str, Dict[str, List[int]]],
    *,
    shuffle_seed: int,
) -> Dict[str, Dict[str, List[int]]]:
    train_pool = sorted(tid for group in retained_partition.values() for tid in group["train"])
    test_pool = sorted(tid for group in retained_partition.values() for tid in group["test"])

    rng = np.random.default_rng(shuffle_seed)
    shuffled_train = rng.permutation(train_pool)
    shuffled_test = rng.permutation(test_pool)

    out: Dict[str, Dict[str, List[int]]] = {}
    train_offset = 0
    test_offset = 0
    for group_name, group in retained_partition.items():
        train_count = len(group["train"])
        test_count = len(group["test"])
        out[group_name] = {
            "train": sorted(int(item) for item in shuffled_train[train_offset : train_offset + train_count].tolist()),
            "test": sorted(int(item) for item in shuffled_test[test_offset : test_offset + test_count].tolist()),
        }
        train_offset += train_count
        test_offset += test_count
    return out


def _evaluate_single_run(
    spec: RunSpec,
    *,
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
    split_seed: int,
    shuffle_seed: int,
    support_threshold: float,
    retained_min_total: int,
    retained_min_train: int,
    retained_min_test: int,
    test_fraction: float,
    ridge_lambda: float,
    max_state_dim: int,
    horizon_h: int,
    endpoint_rollout_steps: int,
    device: str,
    obs_trajectories: Optional[torch.Tensor] = None,
    endpoint_basin_labels: Optional[torch.Tensor] = None,
) -> RunMetrics:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    if not checkpoint_path.exists():
        return RunMetrics(
            root_label=spec.root_label,
            system_key=spec.system_key,
            system_name=spec.system_name,
            seed=spec.seed,
            run_dir=spec.run_dir,
            checkpoint_path=str(checkpoint_path),
            num_trajectories=num_trajectories,
            trajectory_length=trajectory_length,
            support_group_count=0,
            retained_support_group_count=0,
            retained_trajectory_count=0,
            retained_trajectory_coverage=0.0,
            support_group_purity=0.0,
            weighted_support_group_purity=0.0,
            local_1step_nrmse=None,
            global_1step_nrmse=None,
            shuffled_1step_nrmse=None,
            local_h_nrmse=None,
            global_h_nrmse=None,
            shuffled_h_nrmse=None,
            local_beats_global_h=None,
            local_beats_shuffled_h=None,
            status="missing_checkpoint",
            note="checkpoint.pt not found",
        )

    cfg, env, model = _load_checkpoint_model(checkpoint_path, spec.system_key, device)

    if obs_trajectories is None:
        obs_trajectories = _generate_observation_trajectories(
            env,
            num_trajectories=num_trajectories,
            trajectory_length=trajectory_length,
            eval_seed=eval_seed,
        )
    with torch.no_grad():
        latent_trajectories = model.encode(obs_trajectories.to(device)).cpu().numpy().astype(np.float64)

    if endpoint_basin_labels is None:
        basin_count = int(get_transition_rich_basin_count(spec.system_key))
        endpoint_basin_labels = _endpoint_basin_labels(
            env,
            obs_trajectories,
            basin_count=basin_count,
            endpoint_rollout_steps=endpoint_rollout_steps,
        ).cpu()
    endpoint_basin_labels_np = endpoint_basin_labels.numpy().astype(np.int64)

    support_groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for tid in range(num_trajectories):
        support_groups[_support_tuple(torch.from_numpy(latent_trajectories[tid]), support_threshold)].append(tid)

    train_ids, test_ids = _global_train_test_split(num_trajectories, test_fraction, split_seed)
    train_set = set(int(item) for item in train_ids.tolist())
    test_set = set(int(item) for item in test_ids.tolist())

    retained_partition: Dict[str, Dict[str, List[int]]] = {}
    purity_values: List[float] = []
    weighted_purity_total = 0.0
    retained_trajectory_total = 0

    sorted_groups = sorted(support_groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for group_index, (support, tids) in enumerate(sorted_groups):
        if len(tids) < retained_min_total:
            continue
        train_group = sorted(tid for tid in tids if tid in train_set)
        test_group = sorted(tid for tid in tids if tid in test_set)
        if len(train_group) < retained_min_train or len(test_group) < retained_min_test:
            continue

        group_key = f"group_{group_index:03d}"
        retained_partition[group_key] = {"train": train_group, "test": test_group, "all": sorted(tids)}

        labels = endpoint_basin_labels_np[np.array(tids, dtype=np.int64)]
        dominant_count = Counter(labels.tolist()).most_common(1)[0][1]
        purity = float(dominant_count) / float(len(tids))
        purity_values.append(purity)
        weighted_purity_total += purity * float(len(tids))
        retained_trajectory_total += len(tids)

    retained_group_count = len(retained_partition)
    coverage = float(retained_trajectory_total) / float(max(num_trajectories, 1))
    support_group_purity = float(np.mean(purity_values)) if purity_values else 0.0
    weighted_support_group_purity = (
        weighted_purity_total / float(retained_trajectory_total) if retained_trajectory_total > 0 else 0.0
    )

    if not retained_partition:
        return RunMetrics(
            root_label=spec.root_label,
            system_key=spec.system_key,
            system_name=spec.system_name,
            seed=spec.seed,
            run_dir=spec.run_dir,
            checkpoint_path=str(checkpoint_path),
            num_trajectories=num_trajectories,
            trajectory_length=trajectory_length,
            support_group_count=len(support_groups),
            retained_support_group_count=0,
            retained_trajectory_count=0,
            retained_trajectory_coverage=0.0,
            support_group_purity=0.0,
            weighted_support_group_purity=0.0,
            local_1step_nrmse=None,
            global_1step_nrmse=None,
            shuffled_1step_nrmse=None,
            local_h_nrmse=None,
            global_h_nrmse=None,
            shuffled_h_nrmse=None,
            local_beats_global_h=None,
            local_beats_shuffled_h=None,
            status="no_retained_groups",
            note="No support group survived the retained-group criteria",
        )

    local_partition = {key: {"train": value["train"], "test": value["test"]} for key, value in retained_partition.items()}
    global_partition = {
        "global": {
            "train": sorted(tid for value in retained_partition.values() for tid in value["train"]),
            "test": sorted(tid for value in retained_partition.values() for tid in value["test"]),
        }
    }
    shuffled_partition = _build_shuffled_partition(local_partition, shuffle_seed=shuffle_seed)

    local_1, local_h = _fit_and_eval_partition(
        latent_trajectories,
        local_partition,
        ridge_lambda=ridge_lambda,
        max_state_dim=max_state_dim,
        horizon_h=horizon_h,
    )
    global_1, global_h = _fit_and_eval_partition(
        latent_trajectories,
        global_partition,
        ridge_lambda=ridge_lambda,
        max_state_dim=max_state_dim,
        horizon_h=horizon_h,
    )
    shuffled_1, shuffled_h = _fit_and_eval_partition(
        latent_trajectories,
        shuffled_partition,
        ridge_lambda=ridge_lambda,
        max_state_dim=max_state_dim,
        horizon_h=horizon_h,
    )

    return RunMetrics(
        root_label=spec.root_label,
        system_key=spec.system_key,
        system_name=spec.system_name,
        seed=spec.seed,
        run_dir=spec.run_dir,
        checkpoint_path=str(checkpoint_path),
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        support_group_count=len(support_groups),
        retained_support_group_count=retained_group_count,
        retained_trajectory_count=retained_trajectory_total,
        retained_trajectory_coverage=coverage,
        support_group_purity=support_group_purity,
        weighted_support_group_purity=weighted_support_group_purity,
        local_1step_nrmse=local_1,
        global_1step_nrmse=global_1,
        shuffled_1step_nrmse=shuffled_1,
        local_h_nrmse=local_h,
        global_h_nrmse=global_h,
        shuffled_h_nrmse=shuffled_h,
        local_beats_global_h=(local_h < global_h) if local_h is not None and global_h is not None else None,
        local_beats_shuffled_h=(local_h < shuffled_h) if local_h is not None and shuffled_h is not None else None,
        status="ok",
    )


def _evaluate_single_run_star(kwargs: Dict[str, object]) -> RunMetrics:
    spec = kwargs.pop("spec")
    return _evaluate_single_run(spec, **kwargs)


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(finite))


def _write_per_run_csv(path: Path, rows: Sequence[RunMetrics]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(RunMetrics(
        root_label="",
        system_key="",
        system_name="",
        seed=0,
        run_dir="",
        checkpoint_path="",
        num_trajectories=0,
        trajectory_length=0,
        support_group_count=0,
        retained_support_group_count=0,
        retained_trajectory_count=0,
        retained_trajectory_coverage=0.0,
        support_group_purity=0.0,
        weighted_support_group_purity=0.0,
        local_1step_nrmse=None,
        global_1step_nrmse=None,
        shuffled_1step_nrmse=None,
        local_h_nrmse=None,
        global_h_nrmse=None,
        shuffled_h_nrmse=None,
        local_beats_global_h=None,
        local_beats_shuffled_h=None,
        status="",
        note="",
    )).keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _write_summary_markdown(path: Path, rows: Sequence[RunMetrics]) -> None:
    grouped: Dict[str, List[RunMetrics]] = defaultdict(list)
    for row in rows:
        grouped[row.root_label].append(row)

    lines: List[str] = []
    lines.append("# Transition-Rich Basin-Support Metric Summary")
    lines.append("")
    lines.append("This summary uses the paper-facing recurring-support protocol:")
    lines.append("")
    lines.append("- `256` trajectories of length `256` generated from reset seeds `42+i`")
    lines.append("- majority support threshold `1e-3`")
    lines.append("- retained groups require at least `5` total trajectories, then `>=3` train and `>=1` test trajectory after the global `80/20` split")
    lines.append("- local/global/shuffled fits use centered latent states, top-`32` PCA projection, ridge `1e-4`, and `H=20`")
    lines.append("")

    lines.append("## Root-Level Summary")
    lines.append("")
    lines.append("| Root | Systems | Mean Purity | Mean Coverage | Local < Global (`H20`) | Local < Shuffled (`H20`) | Coverage >= 0.60 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for root_label in sorted(grouped):
        root_rows = grouped[root_label]
        ok_rows = [row for row in root_rows if row.status == "ok"]
        systems = len(root_rows)
        purity = _safe_mean(row.support_group_purity for row in ok_rows)
        coverage = _safe_mean(row.retained_trajectory_coverage for row in ok_rows)
        local_lt_global = sum(1 for row in ok_rows if row.local_beats_global_h)
        local_lt_shuffled = sum(1 for row in ok_rows if row.local_beats_shuffled_h)
        coverage_gate = sum(1 for row in ok_rows if row.retained_trajectory_coverage >= 0.60)
        lines.append(
            f"| `{root_label}` | {systems} | {_format_float(purity)} | {_format_float(coverage)} | "
            f"{local_lt_global}/{len(ok_rows)} | {local_lt_shuffled}/{len(ok_rows)} | {coverage_gate}/{len(ok_rows)} |"
        )

    for root_label in sorted(grouped):
        lines.append("")
        lines.append(f"## {root_label}")
        lines.append("")
        lines.append("| System | Purity | Coverage | L20 | G20 | S20 | Status |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in sorted(grouped[root_label], key=lambda item: item.system_name):
            lines.append(
                f"| `{row.system_key}` | {_format_float(row.support_group_purity)} | "
                f"{_format_float(row.retained_trajectory_coverage)} | {_format_float(row.local_h_nrmse)} | "
                f"{_format_float(row.global_h_nrmse)} | {_format_float(row.shuffled_h_nrmse)} | `{row.status}` |"
            )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available in this allocation.")
    if args.device != "cpu" and args.num_workers > 1:
        print(
            "[reduce] forcing --num_workers=1 for accelerator mode to avoid multi-process device contention",
            flush=True,
        )
        args.num_workers = 1

    rows_csv = Path(args.rows_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_labels = _parse_csv_strings(args.root_labels)
    seeds = _parse_csv_ints(args.seeds)
    systems = _parse_csv_strings(args.systems)
    specs = _load_latest_specs(rows_csv, root_labels=root_labels, seeds=seeds, systems=systems)
    if not specs:
        raise ValueError(
            f"No runs matched rows_csv={rows_csv}, root_labels={root_labels}, systems={systems}, seeds={seeds}"
        )

    manifest = {
        "rows_csv": str(rows_csv),
        "root_labels": root_labels,
        "systems": systems,
        "seeds": seeds,
        "num_specs": len(specs),
        "specs": [asdict(spec) for spec in specs],
        "protocol": {
            "num_trajectories": args.num_trajectories,
            "trajectory_length": args.trajectory_length,
            "eval_seed": args.eval_seed,
            "split_seed": args.split_seed,
            "shuffle_seed": args.shuffle_seed,
            "support_threshold": args.support_threshold,
            "retained_min_total": args.retained_min_total,
            "retained_min_train": args.retained_min_train,
            "retained_min_test": args.retained_min_test,
            "test_fraction": args.test_fraction,
            "ridge_lambda": args.ridge_lambda,
            "max_state_dim": args.max_state_dim,
            "horizon_h": args.horizon_h,
            "endpoint_rollout_steps": args.endpoint_rollout_steps,
            "device": args.device,
            "shared_eval_corpus_by_system": args.num_workers <= 1,
            "num_workers": args.num_workers,
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    worker_kwargs = {
        "num_trajectories": args.num_trajectories,
        "trajectory_length": args.trajectory_length,
        "eval_seed": args.eval_seed,
        "split_seed": args.split_seed,
        "shuffle_seed": args.shuffle_seed,
        "support_threshold": args.support_threshold,
        "retained_min_total": args.retained_min_total,
        "retained_min_train": args.retained_min_train,
        "retained_min_test": args.retained_min_test,
        "test_fraction": args.test_fraction,
        "ridge_lambda": args.ridge_lambda,
        "max_state_dim": args.max_state_dim,
        "horizon_h": args.horizon_h,
        "endpoint_rollout_steps": args.endpoint_rollout_steps,
        "device": args.device,
    }

    rows: List[RunMetrics] = []
    if args.num_workers <= 1:
        corpus_cache: Dict[str, EvalCorpus] = {}
        total_systems = len({spec.system_key for spec in specs})
        total_runs = len(specs)
        for index, spec in enumerate(specs, start=1):
            if spec.system_key not in corpus_cache:
                print(
                    f"[reduce] preparing shared eval corpus {len(corpus_cache) + 1}/{total_systems}: "
                    f"{spec.system_key}",
                    flush=True,
                )
                corpus_cache[spec.system_key] = _prepare_eval_corpus(
                    spec,
                    num_trajectories=args.num_trajectories,
                    trajectory_length=args.trajectory_length,
                    eval_seed=args.eval_seed,
                    endpoint_rollout_steps=args.endpoint_rollout_steps,
                )
            print(
                f"[reduce] evaluating run {index}/{total_runs}: "
                f"{spec.root_label} | {spec.system_key} | seed={spec.seed}",
                flush=True,
            )
            corpus = corpus_cache[spec.system_key]
            rows.append(
                _evaluate_single_run(
                    spec,
                    obs_trajectories=corpus.obs_trajectories,
                    endpoint_basin_labels=corpus.endpoint_basin_labels,
                    **worker_kwargs,
                )
            )
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for spec in specs:
                payload = dict(worker_kwargs)
                payload["spec"] = spec
                futures.append(executor.submit(_evaluate_single_run_star, payload))
            for future in as_completed(futures):
                rows.append(future.result())

    rows = sorted(rows, key=lambda item: (item.root_label, item.system_key, item.seed))
    (output_dir / "per_run_metrics.json").write_text(json.dumps([asdict(row) for row in rows], indent=2))
    _write_per_run_csv(output_dir / "per_run_metrics.csv", rows)
    _write_summary_markdown(output_dir / "summary.md", rows)

    print(f"Wrote basin-support metrics for {len(rows)} run(s) to {output_dir}")


if __name__ == "__main__":
    main()
