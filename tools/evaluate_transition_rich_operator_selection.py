#!/usr/bin/env python3
"""Evaluate whether supports select useful local linear operators.

This tool reuses the transition-rich interpretability reducer's checkpoint and
labeling utilities, but focuses on the paper-critical mechanism question:

1. Do support-, family-, or group-conditioned local linear fits reduce held-out
   latent prediction error relative to one global operator?
2. Do they beat count-matched random partitions and simple geometry-matched
   latent k-means partitions?
3. Does the model's learned global Koopman matrix already behave like a
   support-selected local operator family when projected onto the active chart?
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

EPS = 1e-12


@dataclass(frozen=True)
class RunSpec:
    root_label: str
    system_key: str
    system_name: str
    seed: int
    run_dir: str


def _load_reducer_helpers():
    reducer_path = Path(__file__).with_name("reduce_transition_rich_interpretability_metrics.py")
    spec = importlib.util.spec_from_file_location("reduce_transition_rich_interpretability_metrics", reducer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reducer helpers from {reducer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REDUCER = _load_reducer_helpers()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows_csvs",
        required=True,
        help="comma-separated forecasting_rows.csv files used to discover runs",
    )
    parser.add_argument("--output_dir", required=True, help="directory for operator-selection artifacts")
    parser.add_argument("--root_labels", required=True, help="comma-separated root labels to include")
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated integer seed filter")
    parser.add_argument(
        "--support_definitions",
        default="absolute:0.001",
        help="comma-separated support definitions formatted as scheme:value",
    )
    parser.add_argument(
        "--subsets",
        default="all,deep,boundary",
        help="comma-separated subset names to evaluate",
    )
    parser.add_argument(
        "--partition_kinds",
        default="support,family,group",
        help="comma-separated local partition kinds to evaluate",
    )
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument(
        "--operator_fit_kind",
        default="linear",
        choices=["linear", "affine"],
        help="fit zero-intercept linear operators or affine operators with a learned bias term",
    )
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
    parser.add_argument(
        "--latent_kmeans_max_classes",
        type=int,
        default=16,
        help="skip latent-kmeans controls when the base partition has more classes than this",
    )
    parser.add_argument(
        "--max_partition_classes",
        type=int,
        default=256,
        help="skip exact-support families with more classes than this to avoid pathological partitions",
    )
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


def _run_timestamp_key(run_dir: str) -> Tuple[str, str]:
    return REDUCER._run_timestamp_key(run_dir)


def _load_latest_specs(
    rows_csvs: Sequence[Path],
    *,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
) -> List[RunSpec]:
    selected_roots = set(root_labels)
    selected_systems = set(systems)
    selected_seeds = set(seeds)
    best_rows: Dict[Tuple[str, str, int], Dict[str, str]] = {}

    for rows_csv in rows_csvs:
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


def _predict_operator(x: np.ndarray, operator: np.ndarray) -> np.ndarray:
    if operator.shape[0] == x.shape[1]:
        return x @ operator
    if operator.shape[0] == x.shape[1] + 1:
        return x @ operator[:-1, :] + operator[-1, :]
    raise ValueError(
        f"Operator shape {operator.shape} is incompatible with input dim {x.shape[1]}"
    )


def _predict_mse(x: np.ndarray, y: np.ndarray, operator: np.ndarray) -> np.ndarray:
    pred = _predict_operator(x, operator)
    return np.mean((pred - y) ** 2, axis=1)


def _fit_operator(
    x: np.ndarray,
    y: np.ndarray,
    ridge_lambda: float,
    *,
    fit_kind: str,
) -> Optional[np.ndarray]:
    if fit_kind == "linear":
        return REDUCER._fit_linear_operator(x, y, ridge_lambda)
    if fit_kind != "affine":
        raise ValueError(f"Unknown operator fit kind '{fit_kind}'")
    if x.shape[0] == 0 or y.shape[0] == 0 or x.shape != y.shape:
        return None
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    x_aug = np.concatenate([x, ones], axis=1)
    xtx = x_aug.T @ x_aug
    xty = x_aug.T @ y
    reg = ridge_lambda * np.eye(x_aug.shape[1], dtype=x.dtype)
    reg[-1, -1] = 0.0
    try:
        return np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        return None


def _mask_global_operator(
    global_k: np.ndarray,
    support_mask: np.ndarray,
    *,
    fit_kind: str,
) -> np.ndarray:
    masked_linear = global_k * np.outer(support_mask, support_mask)
    if fit_kind == "linear":
        return masked_linear
    if fit_kind == "affine":
        augmented = np.zeros((masked_linear.shape[0] + 1, masked_linear.shape[1]), dtype=masked_linear.dtype)
        augmented[:-1, :] = masked_linear
        return augmented
    raise ValueError(f"Unknown operator fit kind '{fit_kind}'")


def _label_sequences_for_mode(
    env,
    trajectories: torch.Tensor,
    *,
    system_key: str,
    endpoint_rollout_steps: int,
    label_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if label_mode == "auto":
        return REDUCER._label_sequences_and_centers(
            env,
            trajectories,
            system_key=system_key,
            endpoint_rollout_steps=endpoint_rollout_steps,
        )

    basin_count = int(REDUCER.get_transition_rich_basin_count(system_key))
    centers = getattr(env, "points", None)
    centers_tensor = (
        centers.to(dtype=trajectories.dtype)
        if isinstance(centers, torch.Tensor) and centers.ndim == 2
        else None
    )

    if label_mode == "native":
        if not hasattr(env, "basin_label"):
            raise ValueError(f"System '{system_key}' does not expose native basin labels")
        basin_labels = REDUCER._label_from_native_method(env.basin_label, trajectories)
        if centers_tensor is not None:
            return basin_labels, centers_tensor, "native_basin_label+env_points"
        estimated = REDUCER._estimate_basin_centers(env, trajectories, basin_count, endpoint_rollout_steps)
        return basin_labels, estimated, "native_basin_label+estimated_centers"

    if label_mode == "env_points":
        if centers_tensor is None:
            raise ValueError(f"System '{system_key}' does not expose env.points for label_mode=env_points")
        return REDUCER._assign_nearest_centers(trajectories, centers_tensor), centers_tensor, "env_points"

    if label_mode == "estimated_centers":
        estimated = REDUCER._estimate_basin_centers(env, trajectories, basin_count, endpoint_rollout_steps)
        return REDUCER._assign_nearest_centers(trajectories, estimated), estimated, "estimated_centers"

    raise ValueError(f"Unknown label mode '{label_mode}'")


def _train_test_split(
    num_items: int,
    *,
    train_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_items <= 1:
        train_mask = np.ones(num_items, dtype=bool)
        test_mask = np.zeros(num_items, dtype=bool)
        return train_mask, test_mask
    order = rng.permutation(num_items)
    train_size = int(round(train_fraction * num_items))
    train_size = max(1, min(num_items - 1, train_size))
    train_mask = np.zeros(num_items, dtype=bool)
    train_mask[order[:train_size]] = True
    return train_mask, ~train_mask


def _class_distance_summary(
    operators: Dict[object, np.ndarray],
    class_major_basin: Dict[object, int],
    basin_operators: Dict[int, np.ndarray],
) -> Dict[str, Optional[float]]:
    if not operators:
        return {
            "operator_class_count": 0.0,
            "operator_vs_basin_fro_mean": None,
            "operator_within_basin_fro_mean": None,
            "operator_between_basin_fro_mean": None,
            "operator_between_over_within": None,
        }

    class_ids = list(operators.keys())
    support_vs_basin: List[float] = []
    within: List[float] = []
    between: List[float] = []
    for class_id in class_ids:
        basin = class_major_basin.get(class_id)
        basin_op = basin_operators.get(basin) if basin is not None else None
        if basin_op is not None:
            support_vs_basin.append(float(np.linalg.norm(operators[class_id] - basin_op, ord="fro")))
    for index, class_i in enumerate(class_ids):
        basin_i = class_major_basin.get(class_i)
        op_i = operators[class_i]
        for class_j in class_ids[index + 1 :]:
            basin_j = class_major_basin.get(class_j)
            dist = float(np.linalg.norm(op_i - operators[class_j], ord="fro"))
            if basin_i is not None and basin_i == basin_j:
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
        "operator_vs_basin_fro_mean": float(np.mean(support_vs_basin)) if support_vs_basin else None,
        "operator_within_basin_fro_mean": within_mean,
        "operator_between_basin_fro_mean": between_mean,
        "operator_between_over_within": ratio,
    }


def _fit_partition_operators(
    x_train: np.ndarray,
    y_train: np.ndarray,
    labels_train: np.ndarray,
    basin_train: np.ndarray,
    *,
    ridge_lambda: float,
    fit_kind: str,
    min_transitions: int,
) -> Tuple[Dict[object, np.ndarray], Dict[object, int], Counter[object]]:
    class_counts = Counter(labels_train.tolist())
    operators: Dict[object, np.ndarray] = {}
    class_major_basin: Dict[object, int] = {}
    for class_id, count in class_counts.items():
        if int(count) < min_transitions:
            continue
        mask = labels_train == class_id
        operator = _fit_operator(
            x_train[mask],
            y_train[mask],
            ridge_lambda,
            fit_kind=fit_kind,
        )
        if operator is None:
            continue
        basin_counter = Counter(int(item) for item in basin_train[mask].tolist() if int(item) >= 0)
        if not basin_counter:
            continue
        operators[class_id] = operator
        class_major_basin[class_id] = basin_counter.most_common(1)[0][0]
    return operators, class_major_basin, class_counts


def _evaluate_partition(
    x_train: np.ndarray,
    y_train: np.ndarray,
    labels_train: np.ndarray,
    basin_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels_test: np.ndarray,
    basin_test: np.ndarray,
    *,
    ridge_lambda: float,
    fit_kind: str,
    min_transitions: int,
    global_operator: np.ndarray,
    basin_operators: Dict[int, np.ndarray],
) -> Dict[str, Optional[float]]:
    operators, class_major_basin, class_counts = _fit_partition_operators(
        x_train,
        y_train,
        labels_train,
        basin_train,
        ridge_lambda=ridge_lambda,
        fit_kind=fit_kind,
        min_transitions=min_transitions,
    )
    covered_mask = np.asarray([label in operators for label in labels_test.tolist()], dtype=bool)
    global_mse_on_covered = None
    partition_mse = None
    if bool(np.any(covered_mask)):
        partition_errors: List[np.ndarray] = []
        for class_id, operator in operators.items():
            mask = np.logical_and(covered_mask, labels_test == class_id)
            if not bool(np.any(mask)):
                continue
            partition_errors.append(_predict_mse(x_test[mask], y_test[mask], operator))
        if partition_errors:
            partition_mse = float(np.concatenate(partition_errors, axis=0).mean())
            global_mse_on_covered = float(_predict_mse(x_test[covered_mask], y_test[covered_mask], global_operator).mean())
    ratio = None
    gain = None
    if partition_mse is not None and global_mse_on_covered is not None and global_mse_on_covered > EPS:
        ratio = partition_mse / global_mse_on_covered
        gain = global_mse_on_covered - partition_mse
    return {
        "class_count_total": float(len(class_counts)),
        "class_count_fit": float(len(operators)),
        "test_transition_count": float(x_test.shape[0]),
        "test_covered_count": float(int(covered_mask.sum())),
        "test_coverage_fraction": float(float(covered_mask.mean()) if covered_mask.size else 0.0),
        "global_test_mse_on_covered": global_mse_on_covered,
        "partition_test_mse": partition_mse,
        "partition_over_global_on_covered": ratio,
        "partition_gain_vs_global": gain,
        **_class_distance_summary(operators, class_major_basin, basin_operators),
    }


def _prototype_masks_from_exact_support(
    class_labels: np.ndarray,
    support_keys: np.ndarray,
    support_masks: np.ndarray,
    *,
    class_kind: str,
    block_masks: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[object, np.ndarray]:
    if class_kind == "support":
        out: Dict[object, np.ndarray] = {}
        for key, mask in zip(support_keys.tolist(), support_masks):
            if key not in out:
                out[key] = mask.astype(bool, copy=True)
        return out
    if class_kind == "family":
        out = {}
        key_to_mask: Dict[object, np.ndarray] = {}
        for key, mask in zip(support_keys.tolist(), support_masks):
            if key not in key_to_mask:
                key_to_mask[key] = mask.astype(bool, copy=True)
        class_to_support_counter: Dict[object, Counter[object]] = defaultdict(Counter)
        for class_id, support_key in zip(class_labels.tolist(), support_keys.tolist()):
            class_to_support_counter[class_id][support_key] += 1
        for class_id, counter in class_to_support_counter.items():
            top_support_key = counter.most_common(1)[0][0]
            out[class_id] = key_to_mask[top_support_key]
        return out
    if class_kind == "group":
        return dict(block_masks or {})
    return {}


def _masked_operator_metrics(
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels_test: np.ndarray,
    partition_metrics: Dict[str, Optional[float]],
    prototype_masks: Dict[object, np.ndarray],
    global_operator: np.ndarray,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    labels_train: np.ndarray,
    basin_train: np.ndarray,
    ridge_lambda: float,
    fit_kind: str,
    min_transitions: int,
) -> Dict[str, Optional[float]]:
    operators, _class_major_basin, _class_counts = _fit_partition_operators(
        x_train,
        y_train,
        labels_train,
        basin_train,
        ridge_lambda=ridge_lambda,
        fit_kind=fit_kind,
        min_transitions=min_transitions,
    )
    covered_mask = np.asarray(
        [label in operators and label in prototype_masks for label in labels_test.tolist()],
        dtype=bool,
    )
    if not bool(np.any(covered_mask)):
        return {
            "masked_k_test_mse": None,
            "masked_k_over_partition_test": None,
            "masked_k_over_global_on_covered": None,
            "masked_k_fro_vs_partition_mean": None,
            "masked_k_coverage_fraction": 0.0,
        }

    masked_errors: List[np.ndarray] = []
    frobenius_diffs: List[float] = []
    for class_id, operator in operators.items():
        if class_id not in prototype_masks:
            continue
        mask = np.logical_and(covered_mask, labels_test == class_id)
        if not bool(np.any(mask)):
            continue
        support_mask = prototype_masks[class_id].astype(np.float32)
        masked_operator = _mask_global_operator(global_operator, support_mask, fit_kind=fit_kind)
        masked_errors.append(_predict_mse(x_test[mask], y_test[mask], masked_operator))
        frobenius_diffs.append(float(np.linalg.norm(masked_operator - operator, ord="fro")))
    if not masked_errors:
        return {
            "masked_k_test_mse": None,
            "masked_k_over_partition_test": None,
            "masked_k_over_global_on_covered": None,
            "masked_k_fro_vs_partition_mean": None,
            "masked_k_coverage_fraction": float(float(covered_mask.mean()) if covered_mask.size else 0.0),
        }
    masked_mse = float(np.concatenate(masked_errors, axis=0).mean())
    partition_mse = partition_metrics.get("partition_test_mse")
    global_mse = partition_metrics.get("global_test_mse_on_covered")
    return {
        "masked_k_test_mse": masked_mse,
        "masked_k_over_partition_test": (
            None
            if partition_mse is None or float(partition_mse) <= EPS
            else masked_mse / float(partition_mse)
        ),
        "masked_k_over_global_on_covered": (
            None
            if global_mse is None or float(global_mse) <= EPS
            else masked_mse / float(global_mse)
        ),
        "masked_k_fro_vs_partition_mean": float(np.mean(frobenius_diffs)) if frobenius_diffs else None,
        "masked_k_coverage_fraction": float(float(covered_mask.mean()) if covered_mask.size else 0.0),
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


def _evaluate_random_controls(
    x_subset: np.ndarray,
    y_subset: np.ndarray,
    labels_subset: np.ndarray,
    basin_subset: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    ridge_lambda: float,
    fit_kind: str,
    min_transitions: int,
    global_operator: np.ndarray,
    basin_operators: Dict[int, np.ndarray],
    num_random_partitions: int,
    rng: np.random.Generator,
) -> Dict[str, Optional[float]]:
    metric_dicts: List[Dict[str, Optional[float]]] = []
    for _ in range(num_random_partitions):
        permuted_labels = labels_subset[rng.permutation(labels_subset.shape[0])]
        metric_dicts.append(
            _evaluate_partition(
                x_subset[train_mask],
                y_subset[train_mask],
                permuted_labels[train_mask],
                basin_subset[train_mask],
                x_subset[test_mask],
                y_subset[test_mask],
                permuted_labels[test_mask],
                basin_subset[test_mask],
                ridge_lambda=ridge_lambda,
                fit_kind=fit_kind,
                min_transitions=min_transitions,
                global_operator=global_operator,
                basin_operators=basin_operators,
            )
        )
    return _aggregate_metric_dicts(metric_dicts)


def _evaluate_latent_kmeans_control(
    x_subset: np.ndarray,
    y_subset: np.ndarray,
    basin_subset: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    target_class_count: int,
    ridge_lambda: float,
    fit_kind: str,
    min_transitions: int,
    global_operator: np.ndarray,
    basin_operators: Dict[int, np.ndarray],
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
    return _evaluate_partition(
        x_subset[train_mask],
        y_subset[train_mask],
        train_labels.astype(object),
        basin_subset[train_mask],
        x_subset[test_mask],
        y_subset[test_mask],
        test_labels.astype(object),
        basin_subset[test_mask],
        ridge_lambda=ridge_lambda,
        fit_kind=fit_kind,
        min_transitions=min_transitions,
        global_operator=global_operator,
        basin_operators=basin_operators,
    )


def _block_masks_from_layout(block_offset: int, block_sizes: Sequence[int], latent_dim: int) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    cursor = int(block_offset)
    for group_id, block_size in enumerate(block_sizes):
        mask = np.zeros(latent_dim, dtype=bool)
        mask[cursor : cursor + int(block_size)] = True
        out[group_id] = mask
        cursor += int(block_size)
    return out


def _fit_basin_operators(
    x_train: np.ndarray,
    y_train: np.ndarray,
    basin_train: np.ndarray,
    *,
    ridge_lambda: float,
    fit_kind: str,
    min_transitions: int,
) -> Dict[int, np.ndarray]:
    basin_operators: Dict[int, np.ndarray] = {}
    for basin in sorted({int(item) for item in basin_train.tolist() if int(item) >= 0}):
        mask = basin_train == basin
        if int(mask.sum()) < min_transitions:
            continue
        operator = _fit_operator(
            x_train[mask],
            y_train[mask],
            ridge_lambda,
            fit_kind=fit_kind,
        )
        if operator is not None:
            basin_operators[basin] = operator
    return basin_operators


def _stringify_support_definition(scheme: str, value: float) -> str:
    if scheme == "topk":
        return f"{scheme}:{int(value)}"
    return f"{scheme}:{value:.6g}"


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
    grouped: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["root_label"]),
            str(row.get("operator_fit_kind", "linear")),
            str(row.get("label_mode", "auto")),
            str(row["support_definition"]),
            str(row["subset"]),
            str(row["partition_kind"]),
            str(row["control_kind"]),
        )
        grouped[key].append(row)

    lines = [
        "# Operator Selection Summary",
        "",
        "Held-out latent one-step operator fits on the fixed transition-rich shortlist.",
        "",
        (
            "| root | fit | label mode | support | subset | partition | control | "
            "mean coverage | mean part/global | mean part gain | "
            "mean op vs basin | mean masked-K/global | mean masked-K/part |"
        ),
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, group_rows in sorted(grouped.items()):
        root_label, operator_fit_kind, label_mode, support_definition, subset, partition_kind, control_kind = key
        lines.append(
            f"| `{root_label}` | `{operator_fit_kind}` | `{label_mode}` | `{support_definition}` | `{subset}` | `{partition_kind}` | `{control_kind}` | "
            f"{_format_float(_safe_mean(row.get('test_coverage_fraction') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('partition_over_global_on_covered') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('partition_gain_vs_global') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('operator_vs_basin_fro_mean') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('masked_k_over_global_on_covered') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('masked_k_over_partition_test') for row in group_rows))} |"
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
    subsets: Sequence[str],
    partition_kinds: Sequence[str],
    operator_fit_kind: str,
    label_mode: str,
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
                "subsets": list(subsets),
                "partition_kinds": list(partition_kinds),
                "operator_fit_kind": operator_fit_kind,
                "label_mode": label_mode,
                "num_trajectories": args.num_trajectories,
                "trajectory_length": args.trajectory_length,
                "min_operator_transitions": args.min_operator_transitions,
                "family_jaccard_threshold": args.family_jaccard_threshold,
                "train_fraction": args.train_fraction,
                "num_random_partitions": args.num_random_partitions,
                "latent_kmeans_max_classes": args.latent_kmeans_max_classes,
                "max_partition_classes": args.max_partition_classes,
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
    rows_csvs: Sequence[str],
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    support_definitions: Sequence[Tuple[str, float]],
    subsets: Sequence[str],
    partition_kinds: Sequence[str],
    operator_fit_kind: str,
    label_mode: str,
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
    _write_csv(output_dir / "operator_selection_rows.csv", rows)
    _write_summary(output_dir / "operator_selection_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(list(failures), indent=2))
    _write_manifest(
        output_dir / "manifest.json",
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        subsets=subsets,
        partition_kinds=partition_kinds,
        operator_fit_kind=operator_fit_kind,
        label_mode=label_mode,
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
    spec: RunSpec,
    *,
    support_definitions: Sequence[Tuple[str, float]],
    subsets: Sequence[str],
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
    operator_fit_kind: str,
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
    basin_labels, centers, label_source = _label_sequences_for_mode(
        env,
        trajectories,
        system_key=spec.system_key,
        endpoint_rollout_steps=endpoint_rollout_steps,
        label_mode=label_mode,
    )
    latents = REDUCER._encode_trajectories(model, trajectories, device)
    subset_masks = REDUCER._margin_subsets(trajectories, centers)
    block_offset, block_sizes = REDUCER._block_layout_from_model(model)
    block_masks = _block_masks_from_layout(block_offset, block_sizes, latents.shape[-1]) if block_sizes else {}
    with torch.no_grad():
        global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)

    x_all = latents[:, :-1, :].reshape(-1, latents.shape[-1]).astype(np.float32, copy=False)
    y_all = latents[:, 1:, :].reshape(-1, latents.shape[-1]).astype(np.float32, copy=False)
    basin_all = basin_labels[:, :-1].reshape(-1).cpu().numpy()
    subset_transition_masks = {
        subset_name: subset_masks[subset_name].reshape(basin_labels.shape)[:, :-1].reshape(-1)
        for subset_name in subsets
        if subset_name in subset_masks
    }
    rows: List[Dict[str, object]] = []

    for scheme, value in support_definitions:
        support_definition = _stringify_support_definition(scheme, value)
        support_mask_full = REDUCER._support_mask(latents, scheme=scheme, value=value)
        support_keys_full = REDUCER._support_keys(support_mask_full)
        family_labels_full = REDUCER.support_family_labels(
            support_mask_full,
            min_jaccard=family_jaccard_threshold,
        )
        group_labels_full = None
        if block_sizes and "group" in partition_kinds:
            group_labels_full = REDUCER._dominant_group_labels(latents, block_sizes, offset=block_offset)

        support_source_masks = support_mask_full[:, :-1, :].reshape(-1, latents.shape[-1])
        support_keys_source = support_keys_full[:, :-1].reshape(-1)
        family_labels_source = family_labels_full[:, :-1].reshape(-1).astype(object)
        group_labels_source = (
            group_labels_full[:, :-1].reshape(-1).astype(object)
            if group_labels_full is not None
            else None
        )

        for subset_name, subset_mask in subset_transition_masks.items():
            if not bool(np.any(subset_mask)):
                continue
            x_subset = x_all[subset_mask]
            y_subset = y_all[subset_mask]
            basin_subset = basin_all[subset_mask]
            support_masks_subset = support_source_masks[subset_mask]
            support_keys_subset = support_keys_source[subset_mask].astype(object)
            family_labels_subset = family_labels_source[subset_mask]
            group_labels_subset = (
                group_labels_source[subset_mask] if group_labels_source is not None else None
            )

            split_rng = np.random.default_rng(eval_seed + spec.seed + int(np.sum(subset_mask)))
            train_mask, test_mask = _train_test_split(
                x_subset.shape[0],
                train_fraction=train_fraction,
                rng=split_rng,
            )
            if not bool(np.any(test_mask)):
                continue

            global_operator = _fit_operator(
                x_subset[train_mask],
                y_subset[train_mask],
                ridge_lambda,
                fit_kind=operator_fit_kind,
            )
            if global_operator is None:
                continue
            basin_operators = _fit_basin_operators(
                x_subset[train_mask],
                y_subset[train_mask],
                basin_subset[train_mask],
                ridge_lambda=ridge_lambda,
                fit_kind=operator_fit_kind,
                min_transitions=min_operator_transitions,
            )
            global_test_mse = float(_predict_mse(x_subset[test_mask], y_subset[test_mask], global_operator).mean())
            basin_metrics = _evaluate_partition(
                x_subset[train_mask],
                y_subset[train_mask],
                basin_subset[train_mask].astype(object),
                basin_subset[train_mask],
                x_subset[test_mask],
                y_subset[test_mask],
                basin_subset[test_mask].astype(object),
                basin_subset[test_mask],
                ridge_lambda=ridge_lambda,
                fit_kind=operator_fit_kind,
                min_transitions=min_operator_transitions,
                global_operator=global_operator,
                basin_operators=basin_operators,
            )
            common_fields = {
                "root_label": spec.root_label,
                "system_key": spec.system_key,
                "system_name": spec.system_name,
                "seed": spec.seed,
                "run_dir": spec.run_dir,
                "support_definition": support_definition,
                "subset": subset_name,
                "label_source": label_source,
                "label_mode": label_mode,
                "operator_fit_kind": operator_fit_kind,
                "train_fraction": float(train_fraction),
                "subset_transition_count": float(x_subset.shape[0]),
                "global_test_mse_full": global_test_mse,
                "basin_test_mse": basin_metrics.get("partition_test_mse"),
                "basin_over_global_on_covered": basin_metrics.get("partition_over_global_on_covered"),
            }
            rows.append(
                {
                    **common_fields,
                    "partition_kind": "global",
                    "control_kind": "none",
                    "class_count_total": 1.0,
                    "class_count_fit": 1.0,
                    "test_transition_count": float(x_subset[test_mask].shape[0]),
                    "test_covered_count": float(x_subset[test_mask].shape[0]),
                    "test_coverage_fraction": 1.0,
                    "global_test_mse_on_covered": global_test_mse,
                    "partition_test_mse": global_test_mse,
                    "partition_over_global_on_covered": 1.0,
                    "partition_gain_vs_global": 0.0,
                    "operator_class_count": 1.0,
                    "operator_vs_basin_fro_mean": None,
                    "operator_within_basin_fro_mean": None,
                    "operator_between_basin_fro_mean": None,
                    "operator_between_over_within": None,
                    "masked_k_test_mse": None,
                    "masked_k_over_partition_test": None,
                    "masked_k_over_global_on_covered": None,
                    "masked_k_fro_vs_partition_mean": None,
                    "masked_k_coverage_fraction": None,
                    "replicate_count": 1.0,
                }
            )
            rows.append(
                {
                    **common_fields,
                    "partition_kind": "basin",
                    "control_kind": "none",
                    **basin_metrics,
                    "masked_k_test_mse": None,
                    "masked_k_over_partition_test": None,
                    "masked_k_over_global_on_covered": None,
                    "masked_k_fro_vs_partition_mean": None,
                    "masked_k_coverage_fraction": None,
                    "replicate_count": 1.0,
                }
            )

            partition_payloads: List[Tuple[str, np.ndarray, Dict[object, np.ndarray]]] = [
                ("support", support_keys_subset, _prototype_masks_from_exact_support(
                    support_keys_subset,
                    support_keys_subset,
                    support_masks_subset,
                    class_kind="support",
                )),
                ("family", family_labels_subset, _prototype_masks_from_exact_support(
                    family_labels_subset,
                    support_keys_subset,
                    support_masks_subset,
                    class_kind="family",
                )),
            ]
            if group_labels_subset is not None:
                partition_payloads.append(
                    (
                        "group",
                        group_labels_subset,
                        _prototype_masks_from_exact_support(
                            group_labels_subset,
                            support_keys_subset,
                            support_masks_subset,
                            class_kind="group",
                            block_masks=block_masks,
                        ),
                    )
                )

            for partition_kind, labels_subset, prototype_masks in partition_payloads:
                if partition_kind not in partition_kinds:
                    continue
                unique_classes = len(set(labels_subset.tolist()))
                if unique_classes > max_partition_classes:
                    rows.append(
                        {
                            **common_fields,
                            "partition_kind": partition_kind,
                            "control_kind": "skipped",
                            "class_count_total": float(unique_classes),
                            "class_count_fit": 0.0,
                            "test_transition_count": float(x_subset[test_mask].shape[0]),
                            "test_covered_count": 0.0,
                            "test_coverage_fraction": 0.0,
                            "global_test_mse_on_covered": None,
                            "partition_test_mse": None,
                            "partition_over_global_on_covered": None,
                            "partition_gain_vs_global": None,
                            "operator_class_count": 0.0,
                            "operator_vs_basin_fro_mean": None,
                            "operator_within_basin_fro_mean": None,
                            "operator_between_basin_fro_mean": None,
                            "operator_between_over_within": None,
                            "masked_k_test_mse": None,
                            "masked_k_over_partition_test": None,
                            "masked_k_over_global_on_covered": None,
                            "masked_k_fro_vs_partition_mean": None,
                            "masked_k_coverage_fraction": None,
                            "replicate_count": 1.0,
                            "skip_reason": f"class_count>{max_partition_classes}",
                        }
                    )
                    continue

                base_metrics = _evaluate_partition(
                    x_subset[train_mask],
                    y_subset[train_mask],
                    labels_subset[train_mask],
                    basin_subset[train_mask],
                    x_subset[test_mask],
                    y_subset[test_mask],
                    labels_subset[test_mask],
                    basin_subset[test_mask],
                    ridge_lambda=ridge_lambda,
                    fit_kind=operator_fit_kind,
                    min_transitions=min_operator_transitions,
                    global_operator=global_operator,
                    basin_operators=basin_operators,
                )
                masked_metrics = _masked_operator_metrics(
                    x_subset[test_mask],
                    y_subset[test_mask],
                    labels_subset[test_mask],
                    base_metrics,
                    prototype_masks,
                    global_k,
                    x_train=x_subset[train_mask],
                    y_train=y_subset[train_mask],
                    labels_train=labels_subset[train_mask],
                    basin_train=basin_subset[train_mask],
                    ridge_lambda=ridge_lambda,
                    fit_kind=operator_fit_kind,
                    min_transitions=min_operator_transitions,
                )
                rows.append(
                    {
                        **common_fields,
                        "partition_kind": partition_kind,
                        "control_kind": "none",
                        **base_metrics,
                        **masked_metrics,
                        "replicate_count": 1.0,
                    }
                )

                random_rng = np.random.default_rng(
                    eval_seed + spec.seed * 1000 + x_subset.shape[0] + unique_classes
                )
                random_metrics = _evaluate_random_controls(
                    x_subset,
                    y_subset,
                    labels_subset,
                    basin_subset,
                    train_mask,
                    test_mask,
                    ridge_lambda=ridge_lambda,
                    fit_kind=operator_fit_kind,
                    min_transitions=min_operator_transitions,
                    global_operator=global_operator,
                    basin_operators=basin_operators,
                    num_random_partitions=num_random_partitions,
                    rng=random_rng,
                )
                rows.append(
                    {
                        **common_fields,
                        "partition_kind": partition_kind,
                        "control_kind": "random_count_matched",
                        **random_metrics,
                        "masked_k_test_mse": None,
                        "masked_k_over_partition_test": None,
                        "masked_k_over_global_on_covered": None,
                        "masked_k_fro_vs_partition_mean": None,
                        "masked_k_coverage_fraction": None,
                    }
                )

                if 2 <= unique_classes <= latent_kmeans_max_classes:
                    kmeans_metrics = _evaluate_latent_kmeans_control(
                        x_subset,
                        y_subset,
                        basin_subset,
                        train_mask,
                        test_mask,
                        target_class_count=unique_classes,
                        ridge_lambda=ridge_lambda,
                        fit_kind=operator_fit_kind,
                        min_transitions=min_operator_transitions,
                        global_operator=global_operator,
                        basin_operators=basin_operators,
                    )
                    if kmeans_metrics is not None:
                        rows.append(
                            {
                                **common_fields,
                                "partition_kind": partition_kind,
                                "control_kind": "latent_kmeans",
                                **kmeans_metrics,
                                "masked_k_test_mse": None,
                                "masked_k_over_partition_test": None,
                                "masked_k_over_global_on_covered": None,
                                "masked_k_fro_vs_partition_mean": None,
                                "masked_k_coverage_fraction": None,
                                "replicate_count": 1.0,
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
    subsets = _parse_csv_strings(args.subsets)
    partition_kinds = _parse_csv_strings(args.partition_kinds)
    specs = _load_latest_specs(
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
    last_completed_spec: Optional[RunSpec] = None
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
        subsets=subsets,
        partition_kinds=partition_kinds,
        operator_fit_kind=args.operator_fit_kind,
        label_mode=args.label_mode,
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
            rows.extend(
                evaluate_run(
                    spec,
                    support_definitions=support_definitions,
                    subsets=subsets,
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
                    operator_fit_kind=args.operator_fit_kind,
                    label_mode=args.label_mode,
                )
            )
        except Exception as exc:  # pragma: no cover - keep reducer moving across bad runs
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
                rows_csvs=rows_csvs,
                root_labels=root_labels,
                systems=systems,
                seeds=seeds,
                support_definitions=support_definitions,
                subsets=subsets,
                partition_kinds=partition_kinds,
                operator_fit_kind=args.operator_fit_kind,
                label_mode=args.label_mode,
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
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        subsets=subsets,
        partition_kinds=partition_kinds,
        operator_fit_kind=args.operator_fit_kind,
        label_mode=args.label_mode,
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
