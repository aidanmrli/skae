#!/usr/bin/env python3
"""Evaluate self-routed local forecasting without oracle basin routing.

This packet answers the deployment-facing follow-up to the centered-chart
mechanism study:

1. Can the model's own current support route forecasting into a local law
   without oracle basin labels?
2. Does that self-routed local forecast beat one global Koopman matrix?
3. Does LISTA provide a better routing signal than the dense no-sparsity MLP?

The main modes are intentionally narrow:

- ``global_k``: autonomous latent rollout with one learned Koopman matrix
- ``support_local_centered``: exact-support routed centered local operators
- ``family_local_centered``: support-family routed centered local operators
- ``support_gated_k``: exact-support-gated global K in centered coordinates
- ``support_block_gated_k``: block-union-gated global K when block structure exists

All local operators are fit on separate trajectories from the held-out forecast
rollouts so the forecast read is not contaminated by operator-fit reuse.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

EPS = 1e-12
INVALID_ROUTE = "__invalid__"
FALLBACK_ROUTE = "__global_fallback__"


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REDUCER = _load_module(
    "reduce_transition_rich_interpretability_metrics.py",
    "reduce_transition_rich_interpretability_metrics_self_routed_forecasting",
)
OPSEL = _load_module(
    "evaluate_transition_rich_operator_selection.py",
    "evaluate_transition_rich_operator_selection_self_routed_forecasting",
)
CENTERED = _load_module(
    "evaluate_transition_rich_centered_chart_mechanism.py",
    "evaluate_transition_rich_centered_chart_mechanism_self_routed_forecasting",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csvs", required=True, help="comma-separated forecasting_rows.csv files")
    parser.add_argument("--output_dir", required=True, help="directory for self-routed forecasting artifacts")
    parser.add_argument("--root_labels", required=True, help="comma-separated root labels to include")
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated integer seed filter")
    parser.add_argument(
        "--support_definitions",
        default="relative:0.1,topk:8",
        help="comma-separated support definitions formatted as scheme:value",
    )
    parser.add_argument(
        "--depth_strata",
        default="all,q1,q2,q3,q4",
        help="comma-separated initial-state depth strata from {all,q1,q2,q3,q4,boundary,deep}",
    )
    parser.add_argument(
        "--rollout_modes",
        default="global_k,support_gated_k,support_block_gated_k,support_local_centered,family_local_centered",
        help=(
            "comma-separated rollout modes from "
            "{global_k,support_gated_k,support_block_gated_k,support_local_centered,"
            "family_local_centered,oracle_basin_local_centered,latent_kmeans_local_centered,"
            "random_count_matched_local_centered}"
        ),
    )
    parser.add_argument(
        "--horizons",
        default="100,500,1000",
        help="comma-separated rollout horizons to summarize",
    )
    parser.add_argument("--fit_num_trajectories", type=int, default=256)
    parser.add_argument("--fit_trajectory_length", type=int, default=256)
    parser.add_argument("--fit_eval_seed", type=int, default=42)
    parser.add_argument("--forecast_num_trajectories", type=int, default=128)
    parser.add_argument("--forecast_eval_seed", type=int, default=314)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--label_mode",
        default="auto",
        choices=["auto", "native", "env_points", "estimated_centers"],
        help="how to construct basin labels and centers for benchmark evaluation",
    )
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--min_operator_transitions", type=int, default=128)
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.5)
    parser.add_argument("--max_partition_classes", type=int, default=256)
    parser.add_argument("--progress_every_runs", type=int, default=1)
    parser.add_argument("--flush_every_runs", type=int, default=1)
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="disable automatic resume from existing shard outputs in output_dir",
    )
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_support_definitions(raw: str) -> List[Tuple[str, float]]:
    definitions: List[Tuple[str, float]] = []
    for item in _parse_csv_strings(raw):
        if ":" not in item:
            raise ValueError(f"Support definition must be scheme:value, got '{item}'")
        scheme, raw_value = item.split(":", 1)
        scheme = scheme.strip()
        raw_value = raw_value.strip()
        if scheme == "topk":
            definitions.append((scheme, float(int(raw_value))))
        else:
            definitions.append((scheme, float(raw_value)))
    return definitions


def _stringify_support_definition(scheme: str, value: float) -> str:
    if scheme == "topk":
        return f"{scheme}:{int(value)}"
    return f"{scheme}:{value:.6g}"


def _parse_horizons(raw: str) -> List[int]:
    horizons = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not horizons or min(horizons) <= 0:
        raise ValueError("horizons must be a comma-separated list of positive integers")
    return horizons


def _initial_depth_masks(states: torch.Tensor, centers: torch.Tensor) -> Dict[str, np.ndarray]:
    flat = states.reshape(-1, states.shape[-1])
    dists = torch.cdist(flat, centers.to(dtype=flat.dtype))
    if dists.shape[1] < 2:
        valid = np.ones(flat.shape[0], dtype=bool)
        return {
            "all": valid,
            "q1": valid,
            "q2": np.zeros_like(valid, dtype=bool),
            "q3": np.zeros_like(valid, dtype=bool),
            "q4": valid,
            "boundary": valid,
            "deep": valid,
        }
    smallest = torch.topk(dists, k=2, largest=False, dim=1).values
    margins = (smallest[:, 1] - smallest[:, 0]).cpu().numpy()
    q1, q2, q3 = np.quantile(margins, [0.25, 0.5, 0.75])
    masks = {
        "all": np.ones_like(margins, dtype=bool),
        "q1": margins <= q1,
        "q2": np.logical_and(margins > q1, margins <= q2),
        "q3": np.logical_and(margins > q2, margins <= q3),
        "q4": margins > q3,
    }
    masks["boundary"] = masks["q1"]
    masks["deep"] = masks["q4"]
    return masks


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if float(denominator) <= EPS:
        return None
    return float(numerator) / float(denominator)


def _coerce_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lower() in {"none", "nan", "n/a"}:
            return None
        value = stripped
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _support_keys_2d(mask: np.ndarray) -> np.ndarray:
    return REDUCER._support_keys(mask)


def _support_key_to_str(key: object) -> str:
    if isinstance(key, bytes):
        return key.hex()
    return str(key)


def _fit_count_matched_random_labels(reference_labels: np.ndarray, *, seed: int) -> np.ndarray:
    counts = Counter(reference_labels.tolist())
    labels: List[object] = []
    for index, count in enumerate(counts.values()):
        labels.extend([f"random::{index}"] * int(count))
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return np.asarray(labels, dtype=object)


def _dict_centers_from_labels(x: np.ndarray, labels: np.ndarray) -> Dict[object, np.ndarray]:
    centers: Dict[object, np.ndarray] = {}
    for label in Counter(labels.tolist()).keys():
        mask = labels == label
        if bool(np.any(mask)):
            centers[label] = x[mask].mean(axis=0).astype(np.float32, copy=False)
    return centers


def _assign_nearest_dict_centers(x: np.ndarray, centers: Dict[object, np.ndarray]) -> np.ndarray:
    labels = list(centers.keys())
    out = np.empty(x.shape[0], dtype=object)
    out[:] = None
    if not labels or x.size == 0:
        return out
    center_matrix = np.stack([centers[label] for label in labels], axis=0).astype(np.float32, copy=False)
    distances = ((x[:, None, :] - center_matrix[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    for row_index, center_index in enumerate(nearest.tolist()):
        out[row_index] = labels[int(center_index)]
    return out


def _assign_nearest_array_centers(x: np.ndarray, centers: Optional[np.ndarray]) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=object)
    out[:] = None
    if centers is None or centers.size == 0 or x.size == 0:
        return out
    center_matrix = centers.astype(np.float32, copy=False)
    distances = ((x[:, None, :] - center_matrix[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1).astype(object)


def _fit_class_bundle(
    fit_latents: np.ndarray,
    fit_basin_labels: np.ndarray,
    *,
    scheme: str,
    value: float,
    ridge_lambda: float,
    min_operator_transitions: int,
    family_jaccard_threshold: float,
    block_masks: Dict[int, np.ndarray],
    max_partition_classes: int,
    random_control_seed: int,
) -> Dict[str, object]:
    support_mask = REDUCER._support_mask(fit_latents, scheme=scheme, value=value)
    support_keys = REDUCER._support_keys(support_mask)
    family_labels = REDUCER.support_family_labels(
        support_mask,
        min_jaccard=family_jaccard_threshold,
    )

    x_fit = fit_latents[:, :-1, :].reshape(-1, fit_latents.shape[-1]).astype(np.float32, copy=False)
    y_fit = fit_latents[:, 1:, :].reshape(-1, fit_latents.shape[-1]).astype(np.float32, copy=False)
    support_cur = support_keys[:, :-1].reshape(-1).astype(object)
    family_cur = family_labels[:, :-1].reshape(-1).astype(object)

    support_ops, support_centers, support_counts = CENTERED._fit_partition_centered(
        x_fit,
        y_fit,
        support_cur,
        ridge_lambda,
        min_transitions=min_operator_transitions,
    )
    family_ops, family_centers, family_counts = CENTERED._fit_partition_centered(
        x_fit,
        y_fit,
        family_cur,
        ridge_lambda,
        min_transitions=min_operator_transitions,
    )
    basin_cur = fit_basin_labels[:, :-1].reshape(-1).astype(object)
    basin_ops, basin_centers, basin_counts = CENTERED._fit_partition_centered(
        x_fit,
        y_fit,
        basin_cur,
        ridge_lambda,
        min_transitions=min_operator_transitions,
    )

    target_control_classes = int(len(family_counts))
    latent_kmeans_ops: Dict[object, np.ndarray] = {}
    latent_kmeans_centers: Dict[object, np.ndarray] = {}
    latent_kmeans_counts: Counter[object] = Counter()
    latent_kmeans_assignment_centers: Dict[object, np.ndarray] = {}
    random_ops: Dict[object, np.ndarray] = {}
    random_centers: Dict[object, np.ndarray] = {}
    random_counts: Counter[object] = Counter()
    random_assignment_centers: Dict[object, np.ndarray] = {}
    if 2 <= target_control_classes <= int(max_partition_classes) and x_fit.shape[0] >= target_control_classes:
        kmeans_center_tensor = REDUCER._kmeans_centers(
            torch.from_numpy(x_fit).to(dtype=torch.float32),
            target_control_classes,
        )
        kmeans_labels = REDUCER._assign_nearest_centers(
            torch.from_numpy(x_fit).unsqueeze(0).to(dtype=torch.float32),
            kmeans_center_tensor,
        ).reshape(-1).cpu().numpy().astype(object)
        latent_kmeans_ops, latent_kmeans_centers, latent_kmeans_counts = CENTERED._fit_partition_centered(
            x_fit,
            y_fit,
            kmeans_labels,
            ridge_lambda,
            min_transitions=min_operator_transitions,
        )
        latent_kmeans_assignment_centers = {
            int(index): center.cpu().numpy().astype(np.float32, copy=False)
            for index, center in enumerate(kmeans_center_tensor)
        }

        random_labels = _fit_count_matched_random_labels(
            family_cur,
            seed=random_control_seed,
        )
        random_ops, random_centers, random_counts = CENTERED._fit_partition_centered(
            x_fit,
            y_fit,
            random_labels,
            ridge_lambda,
            min_transitions=min_operator_transitions,
        )
        random_assignment_centers = _dict_centers_from_labels(x_fit, random_labels)

    flat_support_mask = support_mask[:, :-1, :].reshape(-1, support_mask.shape[-1])
    support_prototypes = OPSEL._prototype_masks_from_exact_support(
        support_cur,
        support_cur,
        flat_support_mask,
        class_kind="support",
    )
    family_prototypes = OPSEL._prototype_masks_from_exact_support(
        family_cur,
        support_cur,
        flat_support_mask,
        class_kind="family",
    )
    support_block_masks = {
        key: mask
        for key, prototype in support_prototypes.items()
        if (mask := CENTERED._block_mask_from_support_mask(prototype, block_masks)) is not None
    }

    key_to_family: Dict[object, object] = {}
    for support_key, family_id in zip(support_cur.tolist(), family_cur.tolist()):
        if support_key not in key_to_family:
            key_to_family[support_key] = family_id

    return {
        "support_ops": support_ops,
        "support_centers": support_centers,
        "support_counts": support_counts,
        "support_prototypes": support_prototypes,
        "support_block_masks": support_block_masks,
        "family_ops": family_ops,
        "family_centers": family_centers,
        "family_counts": family_counts,
        "family_prototypes": family_prototypes,
        "support_key_to_family": key_to_family,
        "basin_ops": basin_ops,
        "basin_centers": basin_centers,
        "basin_counts": basin_counts,
        "latent_kmeans_ops": latent_kmeans_ops,
        "latent_kmeans_centers": latent_kmeans_centers,
        "latent_kmeans_counts": latent_kmeans_counts,
        "latent_kmeans_assignment_centers": latent_kmeans_assignment_centers,
        "random_ops": random_ops,
        "random_centers": random_centers,
        "random_counts": random_counts,
        "random_assignment_centers": random_assignment_centers,
    }


def _assign_family_ids(
    support_masks: np.ndarray,
    support_keys: np.ndarray,
    *,
    support_key_to_family: Dict[object, object],
    family_prototypes: Dict[object, np.ndarray],
    min_jaccard: float,
    cache: Dict[object, object],
) -> np.ndarray:
    out = np.empty(support_masks.shape[0], dtype=object)
    for idx, (support_key, support_mask) in enumerate(zip(support_keys.tolist(), support_masks)):
        family_id = support_key_to_family.get(support_key)
        if family_id is not None:
            out[idx] = family_id
            continue
        if support_key in cache:
            out[idx] = cache[support_key]
            continue
        best_family = None
        best_similarity = -1.0
        for candidate_family, prototype in family_prototypes.items():
            similarity = REDUCER._binary_jaccard(support_mask, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_family = candidate_family
        assigned = best_family if best_family is not None and best_similarity >= float(min_jaccard) else None
        cache[support_key] = assigned
        out[idx] = assigned
    return out


def _predict_global(latent: np.ndarray, global_k: np.ndarray) -> np.ndarray:
    return latent @ global_k


def _predict_centered(latent: np.ndarray, center: np.ndarray, operator: np.ndarray) -> np.ndarray:
    return center + (latent - center) @ operator


def _predict_gated_k(
    latent: np.ndarray,
    center: np.ndarray,
    global_k: np.ndarray,
    source_mask: np.ndarray,
) -> np.ndarray:
    centered = latent - center
    return center + (centered * source_mask.astype(centered.dtype, copy=False)) @ global_k


def _apply_centered_partition(
    current_latent: np.ndarray,
    labels: np.ndarray,
    *,
    operators: Dict[object, np.ndarray],
    centers: Dict[object, np.ndarray],
    fallback: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    next_valid = fallback.copy()
    used = np.zeros(current_latent.shape[0], dtype=bool)
    route_labels = np.full(current_latent.shape[0], FALLBACK_ROUTE, dtype=object)
    for label in {item for item in labels.tolist() if item is not None}:
        select = labels == label
        operator = operators.get(label)
        center = centers.get(label)
        if operator is None or center is None:
            continue
        next_valid[select] = _predict_centered(current_latent[select], center, operator)
        used[select] = True
        route_labels[select] = label
    return next_valid, used, route_labels


def _summarize_route_metrics(
    predictions: np.ndarray,
    used_local: np.ndarray,
    route_labels: np.ndarray,
) -> Dict[str, np.ndarray]:
    valid_steps = np.all(np.isfinite(predictions), axis=-1)
    batch = predictions.shape[1]
    coverage = np.full(batch, np.nan, dtype=np.float32)
    fallback = np.full(batch, np.nan, dtype=np.float32)
    switch = np.full(batch, np.nan, dtype=np.float32)
    valid_fraction = np.full(batch, np.nan, dtype=np.float32)

    for batch_idx in range(batch):
        valid_mask = valid_steps[:, batch_idx]
        if not bool(np.any(valid_mask)):
            continue
        valid_fraction[batch_idx] = float(valid_mask.mean())
        coverage[batch_idx] = float(used_local[valid_mask, batch_idx].mean())
        fallback[batch_idx] = 1.0 - coverage[batch_idx]
        labels = route_labels[valid_mask, batch_idx]
        if labels.shape[0] <= 1:
            switch[batch_idx] = 0.0
        else:
            switch[batch_idx] = float(np.mean(labels[1:] != labels[:-1]))

    return {
        "coverage_per_ic": coverage,
        "fallback_per_ic": fallback,
        "switch_per_ic": switch,
        "valid_fraction_per_ic": valid_fraction,
    }


def _compute_horizon_stats(
    predictions: np.ndarray,
    true_future: np.ndarray,
    horizons: Sequence[int],
    subset_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    if not bool(np.any(subset_mask)):
        for horizon in horizons:
            out[f"h{int(horizon)}_mean"] = None
            out[f"h{int(horizon)}_std"] = None
            out[f"h{int(horizon)}_num_valid"] = 0.0
        return out

    squared_diff = np.mean((predictions - true_future) ** 2, axis=-1)[:, subset_mask]
    for horizon in horizons:
        cut = min(int(horizon), squared_diff.shape[0])
        horizon_errors = squared_diff[:cut]
        per_ic = np.nanmean(horizon_errors, axis=0)
        valid = np.isfinite(per_ic)
        if not bool(np.any(valid)):
            out[f"h{int(horizon)}_mean"] = None
            out[f"h{int(horizon)}_std"] = None
            out[f"h{int(horizon)}_num_valid"] = 0.0
            continue
        valid_errors = per_ic[valid]
        out[f"h{int(horizon)}_mean"] = float(valid_errors.mean())
        out[f"h{int(horizon)}_std"] = float(valid_errors.std(ddof=0)) if valid_errors.size > 1 else 0.0
        out[f"h{int(horizon)}_num_valid"] = float(valid_errors.size)
    return out


def _compute_subset_route_summary(
    route_metrics: Dict[str, np.ndarray],
    subset_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    if not bool(np.any(subset_mask)):
        return {
            "route_coverage_fraction": None,
            "fallback_fraction": None,
            "route_switch_rate": None,
            "valid_step_fraction": None,
        }
    out: Dict[str, Optional[float]] = {}
    for key, values in (
        ("route_coverage_fraction", route_metrics["coverage_per_ic"]),
        ("fallback_fraction", route_metrics["fallback_per_ic"]),
        ("route_switch_rate", route_metrics["switch_per_ic"]),
        ("valid_step_fraction", route_metrics["valid_fraction_per_ic"]),
    ):
        clean = values[subset_mask]
        clean = clean[np.isfinite(clean)]
        out[key] = float(clean.mean()) if clean.size else None
    return out


def _rollout_self_routed(
    model,
    x0: torch.Tensor,
    max_horizon: int,
    *,
    device: str,
    global_k: np.ndarray,
    support_scheme: str,
    support_value: float,
    family_jaccard_threshold: float,
    rollout_mode: str,
    support_ops: Dict[object, np.ndarray],
    support_centers: Dict[object, np.ndarray],
    support_block_masks: Dict[object, np.ndarray],
    family_ops: Dict[object, np.ndarray],
    family_centers: Dict[object, np.ndarray],
    family_prototypes: Dict[object, np.ndarray],
    support_key_to_family: Dict[object, object],
    basin_ops: Dict[object, np.ndarray],
    basin_centers: Dict[object, np.ndarray],
    state_partition_centers: Optional[np.ndarray],
    latent_kmeans_ops: Dict[object, np.ndarray],
    latent_kmeans_centers: Dict[object, np.ndarray],
    latent_kmeans_assignment_centers: Dict[object, np.ndarray],
    random_ops: Dict[object, np.ndarray],
    random_centers: Dict[object, np.ndarray],
    random_assignment_centers: Dict[object, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    model.eval()
    model_device = next(model.parameters()).device
    x0 = x0.to(model_device)
    with torch.no_grad():
        latent = model.encode(x0).detach().cpu().numpy().astype(np.float32, copy=False)

    batch, latent_dim = latent.shape
    obs_dim = int(x0.shape[-1])
    predictions = np.full((max_horizon, batch, obs_dim), np.nan, dtype=np.float32)
    used_local = np.zeros((max_horizon, batch), dtype=bool)
    route_labels = np.empty((max_horizon, batch), dtype=object)
    route_labels[:] = INVALID_ROUTE
    valid = np.ones(batch, dtype=bool)
    family_cache: Dict[object, object] = {}

    for step in range(max_horizon):
        if not bool(np.any(valid)):
            break
        next_latent = np.full((batch, latent_dim), np.nan, dtype=np.float32)
        valid_indices = np.flatnonzero(valid)
        current_latent = latent[valid_indices]

        if rollout_mode == "global_k":
            next_valid = _predict_global(current_latent, global_k)
            next_latent[valid_indices] = next_valid
            route_labels[step, valid_indices] = "global_k"
        else:
            support_masks = REDUCER._support_mask(current_latent, scheme=support_scheme, value=support_value)
            support_keys = _support_keys_2d(support_masks)
            next_valid = _predict_global(current_latent, global_k)
            labels = np.full(valid_indices.shape[0], FALLBACK_ROUTE, dtype=object)

            if rollout_mode == "support_local_centered":
                for support_key in np.unique(support_keys):
                    select = support_keys == support_key
                    operator = support_ops.get(support_key)
                    center = support_centers.get(support_key)
                    if operator is None or center is None:
                        continue
                    next_valid[select] = _predict_centered(current_latent[select], center, operator)
                    used_local[step, valid_indices[select]] = True
                    labels[select] = support_key

            elif rollout_mode == "family_local_centered":
                family_ids = _assign_family_ids(
                    support_masks,
                    support_keys,
                    support_key_to_family=support_key_to_family,
                    family_prototypes=family_prototypes,
                    min_jaccard=family_jaccard_threshold,
                    cache=family_cache,
                )
                for family_id in {item for item in family_ids.tolist() if item is not None}:
                    select = family_ids == family_id
                    operator = family_ops.get(family_id)
                    center = family_centers.get(family_id)
                    if operator is None or center is None:
                        continue
                    next_valid[select] = _predict_centered(current_latent[select], center, operator)
                    used_local[step, valid_indices[select]] = True
                    labels[select] = family_id

            elif rollout_mode == "support_gated_k":
                for support_key in np.unique(support_keys):
                    select = support_keys == support_key
                    center = support_centers.get(support_key)
                    if center is None:
                        continue
                    next_valid[select] = _predict_gated_k(
                        current_latent[select],
                        center,
                        global_k,
                        support_masks[select],
                    )
                    used_local[step, valid_indices[select]] = True
                    labels[select] = support_key

            elif rollout_mode == "support_block_gated_k":
                for support_key in np.unique(support_keys):
                    select = support_keys == support_key
                    center = support_centers.get(support_key)
                    block_mask = support_block_masks.get(support_key)
                    if center is None or block_mask is None:
                        continue
                    next_valid[select] = _predict_gated_k(
                        current_latent[select],
                        center,
                        global_k,
                        block_mask,
                    )
                    used_local[step, valid_indices[select]] = True
                    labels[select] = f"block::{_support_key_to_str(support_key)}"

            elif rollout_mode == "oracle_basin_local_centered":
                with torch.no_grad():
                    current_state = model.decode(
                        torch.from_numpy(current_latent).to(device=model_device, dtype=x0.dtype)
                    ).detach().cpu().numpy().astype(np.float32, copy=False)
                basin_ids = _assign_nearest_array_centers(current_state, state_partition_centers)
                next_valid, used, labels = _apply_centered_partition(
                    current_latent,
                    basin_ids,
                    operators=basin_ops,
                    centers=basin_centers,
                    fallback=next_valid,
                )
                used_local[step, valid_indices[used]] = True

            elif rollout_mode == "latent_kmeans_local_centered":
                kmeans_ids = _assign_nearest_dict_centers(current_latent, latent_kmeans_assignment_centers)
                next_valid, used, labels = _apply_centered_partition(
                    current_latent,
                    kmeans_ids,
                    operators=latent_kmeans_ops,
                    centers=latent_kmeans_centers,
                    fallback=next_valid,
                )
                used_local[step, valid_indices[used]] = True

            elif rollout_mode == "random_count_matched_local_centered":
                random_ids = _assign_nearest_dict_centers(current_latent, random_assignment_centers)
                next_valid, used, labels = _apply_centered_partition(
                    current_latent,
                    random_ids,
                    operators=random_ops,
                    centers=random_centers,
                    fallback=next_valid,
                )
                used_local[step, valid_indices[used]] = True

            else:
                raise ValueError(f"Unknown rollout mode '{rollout_mode}'")

            next_latent[valid_indices] = next_valid
            route_labels[step, valid_indices] = labels

        with torch.no_grad():
            pred_state = model.decode(
                torch.from_numpy(next_latent).to(device=model_device, dtype=x0.dtype)
            ).detach().cpu().numpy()
        predictions[step] = pred_state.astype(np.float32, copy=False)

        finite_mask = np.logical_and(
            np.all(np.isfinite(next_latent), axis=1),
            np.all(np.isfinite(pred_state), axis=1),
        )
        valid = np.logical_and(valid, finite_mask)
        latent = next_latent

    route_metrics = _summarize_route_metrics(predictions, used_local, route_labels)
    return predictions, route_metrics


def _attach_global_ratios(rows: Sequence[Dict[str, object]], horizons: Sequence[int]) -> None:
    grouped: Dict[Tuple[str, str, int, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["system_key"]),
            str(row["root_label"]),
            int(row["seed"]),
            str(row["support_definition"]),
            str(row["depth_stratum"]),
        )
        grouped[key].append(row)

    for group_rows in grouped.values():
        global_row = next((row for row in group_rows if row.get("rollout_mode") == "global_k"), None)
        if global_row is None:
            continue
        for row in group_rows:
            for horizon in horizons:
                global_mean = global_row.get(f"h{int(horizon)}_mean")
                row[f"h{int(horizon)}_over_global"] = _safe_ratio(
                    row.get(f"h{int(horizon)}_mean"),
                    global_mean,
                )


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_write_text(path, "")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _atomic_write_text(path, buffer.getvalue())


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [numeric for value in values if (numeric := _coerce_optional_float(value)) is not None]
    return float(np.mean(clean)) if clean else None


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _write_summary(path: Path, rows: Sequence[Dict[str, object]], horizons: Sequence[int]) -> None:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["root_label"]),
            str(row["support_definition"]),
            str(row["depth_stratum"]),
            str(row["rollout_mode"]),
        )
        grouped[key].append(row)

    lines = [
        "# Self-Routed Local Forecasting Summary",
        "",
        "Non-oracle rollout comparison: one global Koopman matrix versus self-routed support-conditioned local laws.",
        "",
        "| root | support | initial depth | mode | mean coverage | mean fallback | mean switch | "
        + " | ".join([f"mean H{int(h)}" for h in horizons])
        + " | "
        + " | ".join([f"mean H{int(h)}/global" for h in horizons])
        + " |",
        "|---|---|---|---|---:|---:|---:|"
        + "".join(["---:|" for _ in horizons])
        + "".join(["---:|" for _ in horizons]),
    ]
    for key, group_rows in sorted(grouped.items()):
        root_label, support_definition, depth_stratum, rollout_mode = key
        pieces = [
            f"| `{root_label}` | `{support_definition}` | `{depth_stratum}` | `{rollout_mode}` | "
            f"{_format_float(_safe_mean(row.get('route_coverage_fraction') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('fallback_fraction') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('route_switch_rate') for row in group_rows))} |"
        ]
        for horizon in horizons:
            pieces.append(f" {_format_float(_safe_mean(row.get(f'h{int(horizon)}_mean') for row in group_rows))} |")
        for horizon in horizons:
            pieces.append(f" {_format_float(_safe_mean(row.get(f'h{int(horizon)}_over_global') for row in group_rows))} |")
        lines.append("".join(pieces))
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    rows_csvs: Sequence[str],
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    support_definitions: Sequence[Tuple[str, float]],
    depth_strata: Sequence[str],
    rollout_modes: Sequence[str],
    horizons: Sequence[int],
    num_specs: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            {
                "rows_csvs": list(rows_csvs),
                "root_labels": list(root_labels),
                "systems": list(systems),
                "seeds": list(seeds),
                "support_definitions": [
                    {"scheme": scheme, "value": value} for scheme, value in support_definitions
                ],
                "depth_strata": list(depth_strata),
                "rollout_modes": list(rollout_modes),
                "horizons": list(horizons),
                "fit_num_trajectories": int(args.fit_num_trajectories),
                "fit_trajectory_length": int(args.fit_trajectory_length),
                "fit_eval_seed": int(args.fit_eval_seed),
                "forecast_num_trajectories": int(args.forecast_num_trajectories),
                "forecast_eval_seed": int(args.forecast_eval_seed),
                "label_mode": args.label_mode,
                "ridge_lambda": float(args.ridge_lambda),
                "min_operator_transitions": int(args.min_operator_transitions),
                "family_jaccard_threshold": float(args.family_jaccard_threshold),
                "max_partition_classes": int(args.max_partition_classes),
                "no_resume": bool(args.no_resume),
                "num_runs": num_specs,
                "completed_runs": completed_specs,
                "remaining_runs": max(0, num_specs - completed_specs),
                "num_rows": len(rows),
                "num_failures": len(failures),
                "status": status,
            },
            indent=2,
        ),
    )


def _write_progress(
    path: Path,
    *,
    completed_specs: int,
    num_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    elapsed_seconds: float,
    last_spec: Optional[object],
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
    _atomic_write_text(path, json.dumps(payload, indent=2))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(content)
    tmp_path.replace(path)


def _spec_key_from_parts(root_label: object, system_key: object, seed: object, run_dir: object) -> Tuple[str, str, int, str]:
    return (
        str(root_label),
        str(system_key),
        int(seed),
        str(run_dir),
    )


def _spec_key(spec: object) -> Tuple[str, str, int, str]:
    return _spec_key_from_parts(spec.root_label, spec.system_key, spec.seed, spec.run_dir)


def _load_existing_rows(path: Path) -> Tuple[List[Dict[str, object]], set[Tuple[str, str, int, str]]]:
    if not path.exists():
        return [], set()
    text = path.read_text().strip()
    if not text:
        return [], set()
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    completed_keys = {
        _spec_key_from_parts(
            row.get("root_label", ""),
            row.get("system_key", ""),
            row.get("seed", 0),
            row.get("run_dir", ""),
        )
        for row in rows
        if row.get("run_dir")
    }
    return rows, completed_keys


def _load_existing_failure_map(path: Path) -> Dict[Tuple[str, str, int, str], Dict[str, object]]:
    if not path.exists():
        return {}
    text = path.read_text().strip()
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, list):
        return {}
    failure_map: Dict[Tuple[str, str, int, str], Dict[str, object]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            key = _spec_key_from_parts(
                item.get("root_label", ""),
                item.get("system_key", ""),
                item.get("seed", 0),
                item.get("run_dir", ""),
            )
        except Exception:
            continue
        failure_map[key] = dict(item)
    return failure_map


def _sorted_failures(failure_map: Dict[Tuple[str, str, int, str], Dict[str, object]]) -> List[Dict[str, object]]:
    return [failure_map[key] for key in sorted(failure_map.keys())]


def _flush_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    rows_csvs: Sequence[str],
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    support_definitions: Sequence[Tuple[str, float]],
    depth_strata: Sequence[str],
    rollout_modes: Sequence[str],
    horizons: Sequence[int],
    num_specs: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
    last_spec: Optional[object],
    last_status: Optional[str],
    last_error: Optional[str],
) -> None:
    _write_csv(output_dir / "self_routed_forecasting_rows.csv", rows)
    _write_summary(output_dir / "self_routed_forecasting_summary.md", rows, horizons)
    _atomic_write_text(output_dir / "failures.json", json.dumps(list(failures), indent=2))
    _write_manifest(
        output_dir / "manifest.json",
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        depth_strata=depth_strata,
        rollout_modes=rollout_modes,
        horizons=horizons,
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


def evaluate_run(
    spec,
    *,
    support_definitions: Sequence[Tuple[str, float]],
    depth_strata: Sequence[str],
    rollout_modes: Sequence[str],
    horizons: Sequence[int],
    fit_num_trajectories: int,
    fit_trajectory_length: int,
    fit_eval_seed: int,
    forecast_num_trajectories: int,
    forecast_eval_seed: int,
    endpoint_rollout_steps: int,
    device: str,
    label_mode: str,
    ridge_lambda: float,
    min_operator_transitions: int,
    family_jaccard_threshold: float,
    max_partition_classes: int,
) -> List[Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    _cfg, env, model = REDUCER._load_checkpoint_model(checkpoint_path, spec.system_key, device)
    model_device = next(model.parameters()).device
    global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
    block_offset, block_sizes = REDUCER._block_layout_from_model(model)
    block_masks = OPSEL._block_masks_from_layout(block_offset, block_sizes, global_k.shape[0]) if block_sizes else {}

    fit_trajectories = REDUCER._generate_observation_trajectories(
        env,
        num_trajectories=fit_num_trajectories,
        trajectory_length=fit_trajectory_length,
        eval_seed=fit_eval_seed,
    )
    fit_basin_labels, centers, label_source = OPSEL._label_sequences_for_mode(
        env,
        fit_trajectories,
        system_key=spec.system_key,
        endpoint_rollout_steps=endpoint_rollout_steps,
        label_mode=label_mode,
    )
    fit_basin_labels_np = (
        fit_basin_labels.cpu().numpy()
        if isinstance(fit_basin_labels, torch.Tensor)
        else np.asarray(fit_basin_labels)
    )
    state_partition_centers = (
        centers.detach().cpu().numpy().astype(np.float32, copy=False)
        if isinstance(centers, torch.Tensor)
        else np.asarray(centers, dtype=np.float32)
    )
    fit_latents = REDUCER._encode_trajectories(model, fit_trajectories, device)

    max_horizon = int(max(horizons))
    forecast_trajectories = REDUCER._generate_observation_trajectories(
        env,
        num_trajectories=forecast_num_trajectories,
        trajectory_length=max_horizon + 1,
        eval_seed=forecast_eval_seed,
    )
    initial_states = forecast_trajectories[:, 0, :]
    true_future = forecast_trajectories[:, 1 : max_horizon + 1, :].cpu().numpy().astype(np.float32, copy=False)
    depth_masks = _initial_depth_masks(initial_states, centers)

    rows: List[Dict[str, object]] = []
    for scheme, value in support_definitions:
        support_definition = _stringify_support_definition(scheme, value)
        class_bundle = _fit_class_bundle(
            fit_latents,
            fit_basin_labels_np,
            scheme=scheme,
            value=value,
            ridge_lambda=ridge_lambda,
            min_operator_transitions=min_operator_transitions,
            family_jaccard_threshold=family_jaccard_threshold,
            block_masks=block_masks,
            max_partition_classes=max_partition_classes,
            random_control_seed=20260430 + int(spec.seed),
        )

        support_class_count_total = int(len(class_bundle["support_counts"]))
        family_class_count_total = int(len(class_bundle["family_counts"]))
        basin_class_count_total = int(len(class_bundle["basin_counts"]))
        latent_kmeans_class_count_total = int(len(class_bundle["latent_kmeans_counts"]))
        random_class_count_total = int(len(class_bundle["random_counts"]))
        support_modes_allowed = support_class_count_total <= int(max_partition_classes)

        predictions_by_mode: Dict[str, np.ndarray] = {}
        route_metrics_by_mode: Dict[str, Dict[str, np.ndarray]] = {}
        for rollout_mode in rollout_modes:
            if rollout_mode in {"support_local_centered", "support_gated_k", "support_block_gated_k"} and not support_modes_allowed:
                continue
            if rollout_mode == "support_block_gated_k" and not block_masks:
                continue
            predictions, route_metrics = _rollout_self_routed(
                model,
                initial_states,
                max_horizon,
                device=device,
                global_k=global_k,
                support_scheme=scheme,
                support_value=value,
                family_jaccard_threshold=family_jaccard_threshold,
                rollout_mode=rollout_mode,
                support_ops=class_bundle["support_ops"],
                support_centers=class_bundle["support_centers"],
                support_block_masks=class_bundle["support_block_masks"],
                family_ops=class_bundle["family_ops"],
                family_centers=class_bundle["family_centers"],
                family_prototypes=class_bundle["family_prototypes"],
                support_key_to_family=class_bundle["support_key_to_family"],
                basin_ops=class_bundle["basin_ops"],
                basin_centers=class_bundle["basin_centers"],
                state_partition_centers=state_partition_centers,
                latent_kmeans_ops=class_bundle["latent_kmeans_ops"],
                latent_kmeans_centers=class_bundle["latent_kmeans_centers"],
                latent_kmeans_assignment_centers=class_bundle["latent_kmeans_assignment_centers"],
                random_ops=class_bundle["random_ops"],
                random_centers=class_bundle["random_centers"],
                random_assignment_centers=class_bundle["random_assignment_centers"],
            )
            predictions_by_mode[rollout_mode] = predictions
            route_metrics_by_mode[rollout_mode] = route_metrics

        for depth_stratum in depth_strata:
            subset_mask = depth_masks.get(depth_stratum)
            if subset_mask is None or not bool(np.any(subset_mask)):
                continue
            for rollout_mode in rollout_modes:
                if rollout_mode not in predictions_by_mode:
                    rows.append(
                        {
                            "root_label": spec.root_label,
                            "system_key": spec.system_key,
                            "system_name": spec.system_name,
                            "seed": spec.seed,
                            "run_dir": spec.run_dir,
                            "support_definition": support_definition,
                            "depth_stratum": depth_stratum,
                            "rollout_mode": rollout_mode,
                            "label_mode": label_mode,
                            "label_source": label_source,
                            "fit_num_trajectories": float(fit_num_trajectories),
                            "fit_trajectory_length": float(fit_trajectory_length),
                            "forecast_num_trajectories": float(forecast_num_trajectories),
                            "fit_support_class_count_total": float(support_class_count_total),
                            "fit_support_class_count_fit": float(len(class_bundle["support_ops"])),
                            "fit_family_class_count_total": float(family_class_count_total),
                            "fit_family_class_count_fit": float(len(class_bundle["family_ops"])),
                            "fit_basin_class_count_total": float(basin_class_count_total),
                            "fit_basin_class_count_fit": float(len(class_bundle["basin_ops"])),
                            "fit_latent_kmeans_class_count_total": float(latent_kmeans_class_count_total),
                            "fit_latent_kmeans_class_count_fit": float(len(class_bundle["latent_kmeans_ops"])),
                            "fit_random_class_count_total": float(random_class_count_total),
                            "fit_random_class_count_fit": float(len(class_bundle["random_ops"])),
                            "route_coverage_fraction": None,
                            "fallback_fraction": None,
                            "route_switch_rate": None,
                            "valid_step_fraction": None,
                            "skip_reason": (
                                "support_class_count>max_partition_classes"
                                if rollout_mode in {"support_local_centered", "support_gated_k", "support_block_gated_k"}
                                and not support_modes_allowed
                                else "block_structure_unavailable"
                            ),
                        }
                    )
                    continue

                route_summary = _compute_subset_route_summary(route_metrics_by_mode[rollout_mode], subset_mask)
                horizon_stats = _compute_horizon_stats(
                    predictions_by_mode[rollout_mode],
                    np.transpose(true_future, (1, 0, 2)),
                    horizons,
                    subset_mask,
                )
                rows.append(
                    {
                        "root_label": spec.root_label,
                        "system_key": spec.system_key,
                        "system_name": spec.system_name,
                        "seed": spec.seed,
                        "run_dir": spec.run_dir,
                        "support_definition": support_definition,
                        "depth_stratum": depth_stratum,
                        "rollout_mode": rollout_mode,
                        "label_mode": label_mode,
                        "label_source": label_source,
                        "fit_num_trajectories": float(fit_num_trajectories),
                        "fit_trajectory_length": float(fit_trajectory_length),
                        "fit_eval_seed": float(fit_eval_seed),
                        "forecast_num_trajectories": float(forecast_num_trajectories),
                        "forecast_eval_seed": float(forecast_eval_seed),
                        "fit_support_class_count_total": float(support_class_count_total),
                        "fit_support_class_count_fit": float(len(class_bundle["support_ops"])),
                        "fit_family_class_count_total": float(family_class_count_total),
                        "fit_family_class_count_fit": float(len(class_bundle["family_ops"])),
                        "fit_basin_class_count_total": float(basin_class_count_total),
                        "fit_basin_class_count_fit": float(len(class_bundle["basin_ops"])),
                        "fit_latent_kmeans_class_count_total": float(latent_kmeans_class_count_total),
                        "fit_latent_kmeans_class_count_fit": float(len(class_bundle["latent_kmeans_ops"])),
                        "fit_random_class_count_total": float(random_class_count_total),
                        "fit_random_class_count_fit": float(len(class_bundle["random_ops"])),
                        **route_summary,
                        **horizon_stats,
                        "skip_reason": "",
                    }
                )

    _attach_global_ratios(rows, horizons)
    return rows


def main() -> None:
    args = _parse_args()
    rows_csvs = _parse_csv_strings(args.rows_csvs)
    root_labels = _parse_csv_strings(args.root_labels)
    systems = _parse_csv_strings(args.systems)
    seeds = _parse_csv_ints(args.seeds)
    support_definitions = _parse_support_definitions(args.support_definitions)
    depth_strata = _parse_csv_strings(args.depth_strata)
    rollout_modes = _parse_csv_strings(args.rollout_modes)
    horizons = _parse_horizons(args.horizons)
    specs = OPSEL._load_latest_specs(
        [Path(item) for item in rows_csvs],
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    completed_keys: set[Tuple[str, str, int, str]] = set()
    failure_map: Dict[Tuple[str, str, int, str], Dict[str, object]] = {}
    if not args.no_resume:
        rows, completed_keys = _load_existing_rows(output_dir / "self_routed_forecasting_rows.csv")
        failure_map = _load_existing_failure_map(output_dir / "failures.json")
        for key in completed_keys:
            failure_map.pop(key, None)
    failures: List[Dict[str, object]] = _sorted_failures(failure_map)
    num_specs = len(specs)
    specs_to_run = [spec for spec in specs if _spec_key(spec) not in completed_keys]
    initial_completed_count = len(completed_keys)
    start_time = time.time()
    progress_every_runs = max(1, int(args.progress_every_runs))
    flush_every_runs = max(0, int(args.flush_every_runs))
    last_completed_spec = None
    last_completed_status: Optional[str] = None
    last_completed_error: Optional[str] = None

    if completed_keys:
        print(
            f"Resuming from existing outputs in {output_dir}: "
            f"{len(completed_keys)}/{num_specs} specs already completed; "
            f"{len(specs_to_run)} remaining."
        )

    _flush_outputs(
        output_dir,
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        depth_strata=depth_strata,
        rollout_modes=rollout_modes,
        horizons=horizons,
        num_specs=num_specs,
        completed_specs=len(completed_keys),
        rows=rows,
        failures=failures,
        status="running",
        elapsed_seconds=0.0,
        last_spec=None,
        last_status=None,
        last_error=None,
    )

    for remaining_index, spec in enumerate(specs_to_run, start=1):
        index = initial_completed_count + remaining_index
        spec_key = _spec_key(spec)
        try:
            run_rows = evaluate_run(
                spec,
                support_definitions=support_definitions,
                depth_strata=depth_strata,
                rollout_modes=rollout_modes,
                horizons=horizons,
                fit_num_trajectories=args.fit_num_trajectories,
                fit_trajectory_length=args.fit_trajectory_length,
                fit_eval_seed=args.fit_eval_seed,
                forecast_num_trajectories=args.forecast_num_trajectories,
                forecast_eval_seed=args.forecast_eval_seed,
                endpoint_rollout_steps=args.endpoint_rollout_steps,
                device=args.device,
                label_mode=args.label_mode,
                ridge_lambda=args.ridge_lambda,
                min_operator_transitions=args.min_operator_transitions,
                family_jaccard_threshold=args.family_jaccard_threshold,
                max_partition_classes=args.max_partition_classes,
            )
            rows.extend(run_rows)
            completed_keys.add(spec_key)
            failure_map.pop(spec_key, None)
            last_completed_spec = spec
            last_completed_status = "ok"
            last_completed_error = None
            if index % progress_every_runs == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{index}/{num_specs}] ok root={spec.root_label} system={spec.system_key} seed={spec.seed} "
                    f"rows={len(rows)} failures={len(failures)} elapsed_s={elapsed:.1f}"
                )
        except Exception as exc:  # pragma: no cover - surfaced in artifact logs
            failure_map[spec_key] = {
                "root_label": spec.root_label,
                "system_key": spec.system_key,
                "seed": spec.seed,
                "run_dir": spec.run_dir,
                "error": repr(exc),
            }
            last_completed_spec = spec
            last_completed_status = "error"
            last_completed_error = repr(exc)
            print(
                f"[{index}/{num_specs}] error root={spec.root_label} system={spec.system_key} seed={spec.seed}: {exc}",
                file=sys.stderr,
            )
        if flush_every_runs and index % flush_every_runs == 0:
            failures = _sorted_failures(failure_map)
            _flush_outputs(
                output_dir,
                args=args,
                rows_csvs=rows_csvs,
                root_labels=root_labels,
                systems=systems,
                seeds=seeds,
                support_definitions=support_definitions,
                depth_strata=depth_strata,
                rollout_modes=rollout_modes,
                horizons=horizons,
                num_specs=num_specs,
                completed_specs=len(completed_keys),
                rows=rows,
                failures=failures,
                status="running",
                elapsed_seconds=time.time() - start_time,
                last_spec=last_completed_spec,
                last_status=last_completed_status,
                last_error=last_completed_error,
            )

    failures = _sorted_failures(failure_map)
    _flush_outputs(
        output_dir,
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        depth_strata=depth_strata,
        rollout_modes=rollout_modes,
        horizons=horizons,
        num_specs=num_specs,
        completed_specs=len(completed_keys),
        rows=rows,
        failures=failures,
        status=(
            "complete"
            if len(completed_keys) == num_specs and not failures
            else "complete_with_failures"
            if len(completed_keys) == num_specs
            else "partial_with_failures"
        ),
        elapsed_seconds=time.time() - start_time,
        last_spec=last_completed_spec,
        last_status=last_completed_status,
        last_error=last_completed_error,
    )


if __name__ == "__main__":
    main()
