#!/usr/bin/env python3
"""Evaluate centered local-law and support-gated Koopman mechanisms.

This is the decision-grade replacement for the earlier raw zero-intercept
local-fit read. It answers a narrower and more defensible question:

1. In centered chart coordinates, do basin/support/family-conditioned local
   slopes improve held-out one-step latent prediction over the learned global
   Koopman matrix and over one global centered slope?
2. Does the learned global Koopman matrix already behave like a support- or
   block-selected local law when only the active support coordinates are allowed
   to drive prediction?
3. Do those effects strengthen deep inside basins and weaken near boundaries,
   and do they survive when compared against the dense no-sparsity tanh MLP?
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

EPS = 1e-12


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
    "reduce_transition_rich_interpretability_metrics_centered_chart_mech",
)
OPSEL = _load_module(
    "evaluate_transition_rich_operator_selection.py",
    "evaluate_transition_rich_operator_selection_centered_chart_mech",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csvs", required=True, help="comma-separated forecasting_rows.csv files")
    parser.add_argument("--output_dir", required=True, help="directory for centered mechanism artifacts")
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
        help="comma-separated depth strata from {all,q1,q2,q3,q4,boundary,deep}",
    )
    parser.add_argument(
        "--transition_regimes",
        default="all_current,persistent_current",
        help="comma-separated regimes from {all_current,persistent_current}",
    )
    parser.add_argument(
        "--partition_kinds",
        default="basin,family,support",
        help="comma-separated partition kinds to evaluate",
    )
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument(
        "--label_mode",
        default="auto",
        choices=["auto", "native", "env_points", "estimated_centers"],
        help="how to construct basin labels for evaluation",
    )
    parser.add_argument("--min_operator_transitions", type=int, default=128)
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.5)
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--num_random_partitions", type=int, default=8)
    parser.add_argument("--latent_kmeans_max_classes", type=int, default=16)
    parser.add_argument("--max_partition_classes", type=int, default=256)
    parser.add_argument("--progress_every_runs", type=int, default=1)
    parser.add_argument("--flush_every_runs", type=int, default=0)
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


def _flatten_transition_arrays(
    latents: np.ndarray,
    basin_labels: np.ndarray,
    trajectories: torch.Tensor,
    depth_state_masks: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    x = latents[:, :-1, :].reshape(-1, latents.shape[-1]).astype(np.float32, copy=False)
    y = latents[:, 1:, :].reshape(-1, latents.shape[-1]).astype(np.float32, copy=False)
    basin_labels_np = np.asarray(basin_labels)
    basin_cur = basin_labels[:, :-1].reshape(-1)
    basin_next = basin_labels[:, 1:].reshape(-1)
    states_cur = trajectories[:, :-1, :].reshape(-1, trajectories.shape[-1]).cpu().numpy()
    transition_depth_masks = {
        name: mask.reshape(trajectories.shape[0], trajectories.shape[1])[:, :-1].reshape(-1)
        for name, mask in depth_state_masks.items()
    }
    return {
        "x": x,
        "y": y,
        "basin_cur": basin_labels_np[:, :-1].reshape(-1),
        "basin_next": basin_labels_np[:, 1:].reshape(-1),
        "states_cur": states_cur,
        "depth_masks": transition_depth_masks,
    }


def _depth_strata_masks(states: torch.Tensor, centers: torch.Tensor) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    flat = states.reshape(-1, states.shape[-1])
    dists = torch.cdist(flat, centers.to(dtype=flat.dtype))
    if dists.shape[1] < 2:
        valid = np.ones(flat.shape[0], dtype=bool)
        return {
            "all": valid,
            "q1": valid,
            "q2": np.zeros_like(valid, dtype=bool),
            "q3": np.zeros_like(valid, dtype=bool),
            "q4": np.zeros_like(valid, dtype=bool),
            "boundary": valid,
            "deep": valid,
        }, np.ones(flat.shape[0], dtype=np.float32)
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
    return masks, margins.astype(np.float32, copy=False)


def _fit_centered_slope(
    x: np.ndarray,
    y: np.ndarray,
    center: np.ndarray,
    ridge_lambda: float,
) -> Optional[np.ndarray]:
    return REDUCER._fit_linear_operator(x - center, y - center, ridge_lambda)


def _predict_centered(x: np.ndarray, center: np.ndarray, operator: np.ndarray) -> np.ndarray:
    return center + (x - center) @ operator


def _prediction_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((pred - target) ** 2, axis=1)


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if float(denominator) <= EPS:
        return None
    return float(numerator) / float(denominator)


def _fit_global_centered(
    x_train: np.ndarray,
    y_train: np.ndarray,
    ridge_lambda: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if x_train.shape[0] == 0:
        return None, None
    center = x_train.mean(axis=0)
    operator = _fit_centered_slope(x_train, y_train, center, ridge_lambda)
    if operator is None:
        return None, None
    return operator, center


def _fit_partition_centered(
    x_train: np.ndarray,
    y_train: np.ndarray,
    labels_train: np.ndarray,
    ridge_lambda: float,
    *,
    min_transitions: int,
) -> Tuple[Dict[object, np.ndarray], Dict[object, np.ndarray], Counter[object]]:
    class_counts = Counter(labels_train.tolist())
    operators: Dict[object, np.ndarray] = {}
    centers: Dict[object, np.ndarray] = {}
    for class_id, count in class_counts.items():
        if int(count) < min_transitions:
            continue
        mask = labels_train == class_id
        center = x_train[mask].mean(axis=0)
        operator = _fit_centered_slope(x_train[mask], y_train[mask], center, ridge_lambda)
        if operator is None:
            continue
        operators[class_id] = operator
        centers[class_id] = center.astype(np.float32, copy=False)
    return operators, centers, class_counts


def _evaluate_partition_centered(
    x_train: np.ndarray,
    y_train: np.ndarray,
    labels_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels_test: np.ndarray,
    *,
    ridge_lambda: float,
    min_transitions: int,
    global_k: np.ndarray,
) -> Dict[str, Optional[float]]:
    global_k_mse = float(OPSEL._predict_mse(x_test, y_test, global_k).mean()) if x_test.size else None

    global_centered_operator, global_centered_center = _fit_global_centered(x_train, y_train, ridge_lambda)
    global_centered_mse = None
    if global_centered_operator is not None and global_centered_center is not None:
        pred = _predict_centered(x_test, global_centered_center, global_centered_operator)
        global_centered_mse = float(_prediction_mse(pred, y_test).mean())

    operators, centers, class_counts = _fit_partition_centered(
        x_train,
        y_train,
        labels_train,
        ridge_lambda,
        min_transitions=min_transitions,
    )
    covered_mask = np.asarray([label in operators for label in labels_test.tolist()], dtype=bool)
    partition_centered_mse = None
    if bool(np.any(covered_mask)):
        errors: List[np.ndarray] = []
        for class_id, operator in operators.items():
            mask = np.logical_and(covered_mask, labels_test == class_id)
            if not bool(np.any(mask)):
                continue
            pred = _predict_centered(x_test[mask], centers[class_id], operator)
            errors.append(_prediction_mse(pred, y_test[mask]))
        if errors:
            partition_centered_mse = float(np.concatenate(errors, axis=0).mean())

    return {
        "class_count_total": float(len(class_counts)),
        "class_count_fit": float(len(operators)),
        "test_transition_count": float(x_test.shape[0]),
        "test_covered_count": float(int(covered_mask.sum())),
        "test_coverage_fraction": float(float(covered_mask.mean()) if covered_mask.size else 0.0),
        "global_k_test_mse": global_k_mse,
        "global_centered_test_mse": global_centered_mse,
        "partition_centered_test_mse": partition_centered_mse,
        "partition_over_global_k": _safe_ratio(partition_centered_mse, global_k_mse),
        "partition_over_global_centered": _safe_ratio(partition_centered_mse, global_centered_mse),
        "global_centered_over_global_k": _safe_ratio(global_centered_mse, global_k_mse),
        "operators": operators,
        "centers": centers,
    }


def _aggregate_metric_dicts(metric_dicts: Sequence[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    if not metric_dicts:
        return {}
    keys = list(metric_dicts[0].keys())
    aggregated: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [metrics.get(key) for metrics in metric_dicts]
        clean = [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
        aggregated[key] = float(np.mean(clean)) if clean else None
        aggregated[f"{key}_std"] = float(np.std(clean)) if clean else None
    aggregated["replicate_count"] = float(len(metric_dicts))
    return aggregated


def _evaluate_random_partition_centered(
    x_subset: np.ndarray,
    y_subset: np.ndarray,
    labels_subset: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    ridge_lambda: float,
    min_transitions: int,
    global_k: np.ndarray,
    num_random_partitions: int,
    rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    metric_dicts: List[Dict[str, Optional[float]]] = []
    for _ in range(num_random_partitions):
        permuted = labels_subset[rng.permutation(labels_subset.shape[0])]
        metrics = _evaluate_partition_centered(
            x_subset[train_mask],
            y_subset[train_mask],
            permuted[train_mask],
            x_subset[test_mask],
            y_subset[test_mask],
            permuted[test_mask],
            ridge_lambda=ridge_lambda,
            min_transitions=min_transitions,
            global_k=global_k,
        )
        metric_dicts.append({key: value for key, value in metrics.items() if key not in {"operators", "centers"}})
    return _aggregate_metric_dicts(metric_dicts)


def _evaluate_latent_kmeans_centered(
    x_subset: np.ndarray,
    y_subset: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    target_class_count: int,
    ridge_lambda: float,
    min_transitions: int,
    global_k: np.ndarray,
) -> Optional[Dict[str, Optional[float]]]:
    if target_class_count < 2:
        return None
    if int(train_mask.sum()) < target_class_count:
        return None
    centers = REDUCER._kmeans_centers(
        torch.from_numpy(x_subset[train_mask]).to(dtype=torch.float32),
        target_class_count,
    ).cpu()
    train_labels = REDUCER._assign_nearest_centers(
        torch.from_numpy(x_subset[train_mask]).unsqueeze(0).to(dtype=torch.float32),
        centers,
    ).reshape(-1).cpu().numpy()
    test_labels = REDUCER._assign_nearest_centers(
        torch.from_numpy(x_subset[test_mask]).unsqueeze(0).to(dtype=torch.float32),
        centers,
    ).reshape(-1).cpu().numpy()
    metrics = _evaluate_partition_centered(
        x_subset[train_mask],
        y_subset[train_mask],
        train_labels.astype(object),
        x_subset[test_mask],
        y_subset[test_mask],
        test_labels.astype(object),
        ridge_lambda=ridge_lambda,
        min_transitions=min_transitions,
        global_k=global_k,
    )
    return {key: value for key, value in metrics.items() if key not in {"operators", "centers"}}


def _block_mask_from_support_mask(prototype_mask: np.ndarray, block_masks: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    if not block_masks:
        return None
    block_union = np.zeros_like(prototype_mask, dtype=bool)
    for block_mask in block_masks.values():
        if bool(np.any(np.logical_and(prototype_mask, block_mask))):
            block_union = np.logical_or(block_union, block_mask)
    return block_union if bool(np.any(block_union)) else None


def _prototype_maps(
    support_mask: np.ndarray,
    support_keys: np.ndarray,
    family_labels: np.ndarray,
    basin_labels: np.ndarray,
    deep_candidate_mask: np.ndarray,
    *,
    block_masks: Dict[int, np.ndarray],
) -> Dict[str, Dict[object, np.ndarray]]:
    flat_support = support_mask.reshape(-1, support_mask.shape[-1])
    flat_support_keys = support_keys.reshape(-1)
    flat_family = family_labels.reshape(-1).astype(object)

    support_prototypes = OPSEL._prototype_masks_from_exact_support(
        flat_support_keys.astype(object),
        flat_support_keys,
        flat_support,
        class_kind="support",
    )
    family_prototypes = OPSEL._prototype_masks_from_exact_support(
        flat_family,
        flat_support_keys,
        flat_support,
        class_kind="family",
    )
    basin_prototypes = REDUCER.canonical_support_masks_by_basin(
        support_mask,
        basin_labels,
        deep_candidate_mask,
    )

    out: Dict[str, Dict[object, np.ndarray]] = {
        "support": support_prototypes,
        "family": family_prototypes,
        "basin": basin_prototypes,
    }
    out["support_block"] = {
        key: mask
        for key, prototype in support_prototypes.items()
        if (mask := _block_mask_from_support_mask(prototype, block_masks)) is not None
    }
    out["family_block"] = {
        key: mask
        for key, prototype in family_prototypes.items()
        if (mask := _block_mask_from_support_mask(prototype, block_masks)) is not None
    }
    out["basin_block"] = {
        key: mask
        for key, prototype in basin_prototypes.items()
        if (mask := _block_mask_from_support_mask(prototype, block_masks)) is not None
    }
    return out


def _predict_gated_k(
    x: np.ndarray,
    center: np.ndarray,
    global_k: np.ndarray,
    source_mask: np.ndarray,
    *,
    output_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    centered = x - center
    pred = (centered * source_mask.astype(centered.dtype, copy=False)) @ global_k
    if output_mask is not None:
        pred = pred * output_mask.astype(pred.dtype, copy=False)
    return center + pred


def _evaluate_gated_k_metrics(
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels_test: np.ndarray,
    partition_metrics: Dict[str, Optional[float]],
    prototype_masks: Dict[object, np.ndarray],
    block_masks_by_class: Dict[object, np.ndarray],
    global_k: np.ndarray,
) -> Dict[str, Optional[float]]:
    operators = partition_metrics.get("operators")
    centers = partition_metrics.get("centers")
    if not isinstance(operators, dict) or not isinstance(centers, dict):
        return {
            "input_gated_k_test_mse": None,
            "input_gated_over_global_k": None,
            "input_gated_over_partition_centered": None,
            "submatrix_gated_k_test_mse": None,
            "submatrix_gated_over_global_k": None,
            "submatrix_gated_over_partition_centered": None,
            "block_submatrix_k_test_mse": None,
            "block_submatrix_over_global_k": None,
            "block_submatrix_over_partition_centered": None,
        }

    covered_mask = np.asarray(
        [label in operators and label in prototype_masks for label in labels_test.tolist()],
        dtype=bool,
    )
    if not bool(np.any(covered_mask)):
        return {
            "input_gated_k_test_mse": None,
            "input_gated_over_global_k": None,
            "input_gated_over_partition_centered": None,
            "submatrix_gated_k_test_mse": None,
            "submatrix_gated_over_global_k": None,
            "submatrix_gated_over_partition_centered": None,
            "block_submatrix_k_test_mse": None,
            "block_submatrix_over_global_k": None,
            "block_submatrix_over_partition_centered": None,
        }

    input_errors: List[np.ndarray] = []
    submatrix_errors: List[np.ndarray] = []
    block_errors: List[np.ndarray] = []
    for class_id in operators.keys():
        if class_id not in prototype_masks or class_id not in centers:
            continue
        mask = np.logical_and(covered_mask, labels_test == class_id)
        if not bool(np.any(mask)):
            continue
        center = centers[class_id]
        prototype = prototype_masks[class_id]
        pred_input = _predict_gated_k(x_test[mask], center, global_k, prototype, output_mask=None)
        input_errors.append(_prediction_mse(pred_input, y_test[mask]))
        pred_sub = _predict_gated_k(x_test[mask], center, global_k, prototype, output_mask=prototype)
        submatrix_errors.append(_prediction_mse(pred_sub, y_test[mask]))
        block_mask = block_masks_by_class.get(class_id)
        if block_mask is not None:
            pred_block = _predict_gated_k(x_test[mask], center, global_k, block_mask, output_mask=block_mask)
            block_errors.append(_prediction_mse(pred_block, y_test[mask]))

    input_mse = float(np.concatenate(input_errors, axis=0).mean()) if input_errors else None
    submatrix_mse = float(np.concatenate(submatrix_errors, axis=0).mean()) if submatrix_errors else None
    block_mse = float(np.concatenate(block_errors, axis=0).mean()) if block_errors else None
    global_k_mse = partition_metrics.get("global_k_test_mse")
    partition_mse = partition_metrics.get("partition_centered_test_mse")
    return {
        "input_gated_k_test_mse": input_mse,
        "input_gated_over_global_k": _safe_ratio(input_mse, global_k_mse),
        "input_gated_over_partition_centered": _safe_ratio(input_mse, partition_mse),
        "submatrix_gated_k_test_mse": submatrix_mse,
        "submatrix_gated_over_global_k": _safe_ratio(submatrix_mse, global_k_mse),
        "submatrix_gated_over_partition_centered": _safe_ratio(submatrix_mse, partition_mse),
        "block_submatrix_k_test_mse": block_mse,
        "block_submatrix_over_global_k": _safe_ratio(block_mse, global_k_mse),
        "block_submatrix_over_partition_centered": _safe_ratio(block_mse, partition_mse),
    }


def _mean_mask_size(prototype_masks: Dict[object, np.ndarray]) -> Optional[float]:
    if not prototype_masks:
        return None
    return float(np.mean([mask.sum() for mask in prototype_masks.values()]))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def _write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["root_label"]),
            str(row["support_definition"]),
            str(row["depth_stratum"]),
            str(row["transition_regime"]),
            str(row["partition_kind"]),
            str(row["control_kind"]),
        )
        grouped[key].append(row)

    lines = [
        "# Centered Chart Mechanism Summary",
        "",
        "Centered local-law and support-gated Koopman diagnostics on the fixed transition-rich shortlist.",
        "",
        (
            "| root | support | depth | regime | partition | control | mean count | mean persist | "
            "mean part/globalK | mean part/globalCentered | "
            "mean input-gated/globalK | mean submatrix/globalK | mean block-submatrix/globalK |"
        ),
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, group_rows in sorted(grouped.items()):
        root_label, support_definition, depth_stratum, transition_regime, partition_kind, control_kind = key
        lines.append(
            f"| `{root_label}` | `{support_definition}` | `{depth_stratum}` | `{transition_regime}` | "
            f"`{partition_kind}` | `{control_kind}` | "
            f"{_format_float(_safe_mean(row.get('transition_count') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('persistent_fraction_subset') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('partition_over_global_k') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('partition_over_global_centered') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('input_gated_over_global_k') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('submatrix_gated_over_global_k') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('block_submatrix_over_global_k') for row in group_rows))} |"
        )
    path.write_text("\n".join(lines) + "\n")


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
    transition_regimes: Sequence[str],
    partition_kinds: Sequence[str],
    num_specs: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
) -> None:
    path.write_text(
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
                "transition_regimes": list(transition_regimes),
                "partition_kinds": list(partition_kinds),
                "label_mode": args.label_mode,
                "num_trajectories": args.num_trajectories,
                "trajectory_length": args.trajectory_length,
                "train_fraction": args.train_fraction,
                "num_random_partitions": args.num_random_partitions,
                "latent_kmeans_max_classes": args.latent_kmeans_max_classes,
                "max_partition_classes": args.max_partition_classes,
                "min_operator_transitions": args.min_operator_transitions,
                "family_jaccard_threshold": args.family_jaccard_threshold,
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
    path.write_text(json.dumps(payload, indent=2))


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
    transition_regimes: Sequence[str],
    partition_kinds: Sequence[str],
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
    _write_csv(output_dir / "centered_chart_mechanism_rows.csv", rows)
    _write_summary(output_dir / "centered_chart_mechanism_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(list(failures), indent=2))
    _write_manifest(
        output_dir / "manifest.json",
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        depth_strata=depth_strata,
        transition_regimes=transition_regimes,
        partition_kinds=partition_kinds,
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
    transition_regimes: Sequence[str],
    partition_kinds: Sequence[str],
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
    endpoint_rollout_steps: int,
    device: str,
    ridge_lambda: float,
    min_operator_transitions: int,
    family_jaccard_threshold: float,
    train_fraction: float,
    num_random_partitions: int,
    latent_kmeans_max_classes: int,
    max_partition_classes: int,
    label_mode: str,
) -> List[Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    _cfg, env, model = REDUCER._load_checkpoint_model(checkpoint_path, spec.system_key, device)
    trajectories = REDUCER._generate_observation_trajectories(
        env,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        eval_seed=eval_seed,
    )
    basin_labels, centers, label_source = OPSEL._label_sequences_for_mode(
        env,
        trajectories,
        system_key=spec.system_key,
        endpoint_rollout_steps=endpoint_rollout_steps,
        label_mode=label_mode,
    )
    basin_labels_np = basin_labels.cpu().numpy() if isinstance(basin_labels, torch.Tensor) else np.asarray(basin_labels)
    latents = REDUCER._encode_trajectories(model, trajectories, device)
    depth_state_masks, margins = _depth_strata_masks(trajectories, centers)
    block_offset, block_sizes = REDUCER._block_layout_from_model(model)
    block_masks = OPSEL._block_masks_from_layout(block_offset, block_sizes, latents.shape[-1]) if block_sizes else {}
    with torch.no_grad():
        global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)

    arrays = _flatten_transition_arrays(latents, basin_labels, trajectories, depth_state_masks)
    x_all = arrays["x"]
    y_all = arrays["y"]
    basin_cur = arrays["basin_cur"]
    basin_next = arrays["basin_next"]
    depth_transition_masks = arrays["depth_masks"]

    rows: List[Dict[str, object]] = []
    for scheme, value in support_definitions:
        support_definition = _stringify_support_definition(scheme, value)
        support_mask = REDUCER._support_mask(latents, scheme=scheme, value=value)
        support_keys = REDUCER._support_keys(support_mask)
        family_labels = REDUCER.support_family_labels(
            support_mask,
            min_jaccard=family_jaccard_threshold,
        )
        prototype_maps = _prototype_maps(
            support_mask,
            support_keys,
            family_labels,
            basin_labels_np,
            depth_state_masks["q4"].reshape(trajectories.shape[0], trajectories.shape[1]),
            block_masks=block_masks,
        )

        support_cur = support_keys[:, :-1].reshape(-1).astype(object)
        support_next = support_keys[:, 1:].reshape(-1).astype(object)
        family_cur = family_labels[:, :-1].reshape(-1).astype(object)
        family_next = family_labels[:, 1:].reshape(-1).astype(object)
        basin_cur_obj = basin_cur.astype(object)
        basin_next_obj = basin_next.astype(object)

        labels_by_partition = {
            "basin": (basin_cur_obj, basin_next_obj, prototype_maps["basin"], prototype_maps["basin_block"]),
            "family": (family_cur, family_next, prototype_maps["family"], prototype_maps["family_block"]),
            "support": (support_cur, support_next, prototype_maps["support"], prototype_maps["support_block"]),
        }

        for depth_stratum in depth_strata:
            if depth_stratum not in depth_transition_masks:
                continue
            subset_mask = depth_transition_masks[depth_stratum]
            if not bool(np.any(subset_mask)):
                continue
            for partition_kind in partition_kinds:
                if partition_kind not in labels_by_partition:
                    continue
                labels_cur, labels_next, prototype_masks, block_proto_masks = labels_by_partition[partition_kind]
                persistent_mask = np.logical_and(subset_mask, labels_cur == labels_next)
                persistent_fraction = float(np.mean((labels_cur == labels_next)[subset_mask])) if bool(np.any(subset_mask)) else None
                for transition_regime in transition_regimes:
                    if transition_regime == "all_current":
                        regime_mask = subset_mask
                    elif transition_regime == "persistent_current":
                        regime_mask = persistent_mask
                    else:
                        raise ValueError(f"Unknown transition regime '{transition_regime}'")
                    if not bool(np.any(regime_mask)):
                        continue

                    x_subset = x_all[regime_mask]
                    y_subset = y_all[regime_mask]
                    labels_subset = labels_cur[regime_mask]
                    target_unique_classes = len(set(labels_subset.tolist()))
                    rng = np.random.default_rng(
                        eval_seed + spec.seed * 1000 + int(np.count_nonzero(regime_mask)) + target_unique_classes
                    )
                    train_mask, test_mask = OPSEL._train_test_split(
                        x_subset.shape[0],
                        train_fraction=train_fraction,
                        rng=rng,
                    )

                    common_fields: Dict[str, object] = {
                        "root_label": spec.root_label,
                        "system_key": spec.system_key,
                        "system_name": spec.system_name,
                        "seed": spec.seed,
                        "run_dir": spec.run_dir,
                        "support_definition": support_definition,
                        "depth_stratum": depth_stratum,
                        "transition_regime": transition_regime,
                        "partition_kind": partition_kind,
                        "label_source": label_source,
                        "label_mode": label_mode,
                        "ridge_lambda": ridge_lambda,
                        "train_fraction": train_fraction,
                        "subset_transition_count": float(int(subset_mask.sum())),
                        "transition_count": float(x_subset.shape[0]),
                        "persistent_fraction_subset": persistent_fraction,
                        "prototype_mean_support_size": _mean_mask_size(prototype_masks),
                        "prototype_mean_block_size": _mean_mask_size(block_proto_masks),
                    }

                    if target_unique_classes > max_partition_classes:
                        rows.append(
                            {
                                **common_fields,
                                "control_kind": "skipped",
                                "class_count_total": float(target_unique_classes),
                                "class_count_fit": 0.0,
                                "test_transition_count": float(x_subset[test_mask].shape[0]),
                                "test_covered_count": 0.0,
                                "test_coverage_fraction": 0.0,
                                "global_k_test_mse": None,
                                "global_centered_test_mse": None,
                                "partition_centered_test_mse": None,
                                "partition_over_global_k": None,
                                "partition_over_global_centered": None,
                                "global_centered_over_global_k": None,
                                "input_gated_k_test_mse": None,
                                "input_gated_over_global_k": None,
                                "input_gated_over_partition_centered": None,
                                "submatrix_gated_k_test_mse": None,
                                "submatrix_gated_over_global_k": None,
                                "submatrix_gated_over_partition_centered": None,
                                "block_submatrix_k_test_mse": None,
                                "block_submatrix_over_global_k": None,
                                "block_submatrix_over_partition_centered": None,
                                "skip_reason": f"class_count>{max_partition_classes}",
                            }
                        )
                        continue

                    partition_metrics = _evaluate_partition_centered(
                        x_subset[train_mask],
                        y_subset[train_mask],
                        labels_subset[train_mask],
                        x_subset[test_mask],
                        y_subset[test_mask],
                        labels_subset[test_mask],
                        ridge_lambda=ridge_lambda,
                        min_transitions=min_operator_transitions,
                        global_k=global_k,
                    )
                    gated_metrics = _evaluate_gated_k_metrics(
                        x_subset[test_mask],
                        y_subset[test_mask],
                        labels_subset[test_mask],
                        partition_metrics,
                        prototype_masks,
                        block_proto_masks,
                        global_k,
                    )
                    rows.append(
                        {
                            **common_fields,
                            "control_kind": "none",
                            **{key: value for key, value in partition_metrics.items() if key not in {"operators", "centers"}},
                            **gated_metrics,
                        }
                    )

                    random_metrics = _evaluate_random_partition_centered(
                        x_subset,
                        y_subset,
                        labels_subset,
                        train_mask,
                        test_mask,
                        ridge_lambda=ridge_lambda,
                        min_transitions=min_operator_transitions,
                        global_k=global_k,
                        num_random_partitions=num_random_partitions,
                        rng=rng,
                    )
                    rows.append(
                        {
                            **common_fields,
                            "control_kind": "random_count_matched",
                            **random_metrics,
                            "input_gated_k_test_mse": None,
                            "input_gated_over_global_k": None,
                            "input_gated_over_partition_centered": None,
                            "submatrix_gated_k_test_mse": None,
                            "submatrix_gated_over_global_k": None,
                            "submatrix_gated_over_partition_centered": None,
                            "block_submatrix_k_test_mse": None,
                            "block_submatrix_over_global_k": None,
                            "block_submatrix_over_partition_centered": None,
                        }
                    )

                    if 2 <= target_unique_classes <= latent_kmeans_max_classes:
                        kmeans_metrics = _evaluate_latent_kmeans_centered(
                            x_subset,
                            y_subset,
                            train_mask,
                            test_mask,
                            target_class_count=target_unique_classes,
                            ridge_lambda=ridge_lambda,
                            min_transitions=min_operator_transitions,
                            global_k=global_k,
                        )
                        if kmeans_metrics is not None:
                            rows.append(
                                {
                                    **common_fields,
                                    "control_kind": "latent_kmeans",
                                    **kmeans_metrics,
                                    "input_gated_k_test_mse": None,
                                    "input_gated_over_global_k": None,
                                    "input_gated_over_partition_centered": None,
                                    "submatrix_gated_k_test_mse": None,
                                    "submatrix_gated_over_global_k": None,
                                    "submatrix_gated_over_partition_centered": None,
                                    "block_submatrix_k_test_mse": None,
                                    "block_submatrix_over_global_k": None,
                                    "block_submatrix_over_partition_centered": None,
                                }
                            )
    return rows


def main() -> None:
    args = _parse_args()
    rows_csvs = _parse_csv_strings(args.rows_csvs)
    root_labels = _parse_csv_strings(args.root_labels)
    systems = _parse_csv_strings(args.systems)
    seeds = _parse_csv_ints(args.seeds)
    support_definitions = _parse_support_definitions(args.support_definitions)
    depth_strata = _parse_csv_strings(args.depth_strata)
    transition_regimes = _parse_csv_strings(args.transition_regimes)
    partition_kinds = _parse_csv_strings(args.partition_kinds)
    specs = OPSEL._load_latest_specs(
        [Path(item) for item in rows_csvs],
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
    last_completed_spec = None
    last_completed_status: Optional[str] = None
    last_completed_error: Optional[str] = None

    _flush_outputs(
        output_dir,
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        depth_strata=depth_strata,
        transition_regimes=transition_regimes,
        partition_kinds=partition_kinds,
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

    for index, spec in enumerate(specs, start=1):
        try:
            run_rows = evaluate_run(
                spec,
                support_definitions=support_definitions,
                depth_strata=depth_strata,
                transition_regimes=transition_regimes,
                partition_kinds=partition_kinds,
                num_trajectories=args.num_trajectories,
                trajectory_length=args.trajectory_length,
                eval_seed=args.eval_seed,
                endpoint_rollout_steps=args.endpoint_rollout_steps,
                device=args.device,
                ridge_lambda=args.ridge_lambda,
                min_operator_transitions=args.min_operator_transitions,
                family_jaccard_threshold=args.family_jaccard_threshold,
                train_fraction=args.train_fraction,
                num_random_partitions=args.num_random_partitions,
                latent_kmeans_max_classes=args.latent_kmeans_max_classes,
                max_partition_classes=args.max_partition_classes,
                label_mode=args.label_mode,
            )
            rows.extend(run_rows)
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
            failures.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
                    "error": repr(exc),
                }
            )
            last_completed_spec = spec
            last_completed_status = "error"
            last_completed_error = repr(exc)
            print(
                f"[{index}/{num_specs}] error root={spec.root_label} system={spec.system_key} seed={spec.seed}: {exc}",
                file=sys.stderr,
            )
        if flush_every_runs and index % flush_every_runs == 0:
            _flush_outputs(
                output_dir,
                args=args,
                rows_csvs=rows_csvs,
                root_labels=root_labels,
                systems=systems,
                seeds=seeds,
                support_definitions=support_definitions,
                depth_strata=depth_strata,
                transition_regimes=transition_regimes,
                partition_kinds=partition_kinds,
                num_specs=num_specs,
                completed_specs=index,
                rows=rows,
                failures=failures,
                status="running",
                elapsed_seconds=time.time() - start_time,
                last_spec=last_completed_spec,
                last_status=last_completed_status,
                last_error=last_completed_error,
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
        transition_regimes=transition_regimes,
        partition_kinds=partition_kinds,
        num_specs=num_specs,
        completed_specs=num_specs,
        rows=rows,
        failures=failures,
        status="complete" if not failures else "complete_with_failures",
        elapsed_seconds=time.time() - start_time,
        last_spec=last_completed_spec,
        last_status=last_completed_status,
        last_error=last_completed_error,
    )


if __name__ == "__main__":
    main()
