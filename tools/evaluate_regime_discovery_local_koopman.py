#!/usr/bin/env python3
"""Compare learned supports with explicit regime-discovery local Koopman baselines.

The evaluator keeps a trained Koopman autoencoder fixed and fits centered local
linear latent transition maps under several route variables:

- one learned global K from the checkpoint;
- learned support-family routes from the model's sparse support;
- oracle basin labels, used only as a benchmark upper-bound route;
- k-means, diagonal-GMM, and spectral-clustering partitions of raw state,
  dense latent values, sparse latent values, and binary supports.

Cluster counts can be matched to the known benchmark basin count or to the
learned support-family count. The latter is label-free; the former is an
evaluation-only stress test of simple unsupervised partitions.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import signal
import sys
import time
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

EPS = 1e-12
STOP_REQUESTED = False


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
    "reduce_transition_rich_interpretability_metrics_regime_discovery",
)
OPSEL = _load_module(
    "evaluate_transition_rich_operator_selection.py",
    "evaluate_transition_rich_operator_selection_regime_discovery",
)
CENTERED = _load_module(
    "evaluate_transition_rich_centered_chart_mechanism.py",
    "evaluate_transition_rich_centered_chart_mechanism_regime_discovery",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csvs", required=True, help="comma-separated forecasting_rows.csv files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--root_labels", required=True, help="comma-separated root labels")
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated integer seed filter")
    parser.add_argument("--support_definition", default="topk:8")
    parser.add_argument(
        "--feature_views",
        default="raw_state,dense_latent,sparse_latent_values,support_binary",
        help="comma-separated views from {raw_state,dense_latent,sparse_latent_values,support_binary}",
    )
    parser.add_argument(
        "--cluster_methods",
        default="kmeans,gmm_diag,spectral",
        help="comma-separated methods from {kmeans,gmm_diag,spectral}",
    )
    parser.add_argument(
        "--cluster_count_modes",
        default="basin_count,support_family_count",
        help="comma-separated modes from {basin_count,support_family_count}",
    )
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--label_mode", default="auto", choices=["auto", "native", "env_points", "estimated_centers"])
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--min_operator_transitions", type=int, default=128)
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.5)
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--cluster_fit_max_samples", type=int, default=4096)
    parser.add_argument("--spectral_neighbors", type=int, default=20)
    parser.add_argument("--decode_batch_size", type=int, default=4096)
    parser.add_argument("--progress_every_runs", type=int, default=1)
    parser.add_argument("--flush_every_runs", type=int, default=1)
    parser.add_argument("--max_runtime_seconds", type=int, default=0)
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def _request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; will flush and stop after the current spec.", flush=True)


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_support_definition(raw: str) -> Tuple[str, float]:
    if ":" not in raw:
        raise ValueError(f"Support definition must be scheme:value, got '{raw}'")
    scheme, value = raw.split(":", 1)
    scheme = scheme.strip()
    value = value.strip()
    if scheme == "topk":
        return scheme, float(int(value))
    return scheme, float(value)


def _stringify_support_definition(scheme: str, value: float) -> str:
    if scheme == "topk":
        return f"{scheme}:{int(value)}"
    return f"{scheme}:{value:.6g}"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(content)
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        _atomic_write_text(path, "")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _atomic_write_text(path, buffer.getvalue())


def _safe_mean(values: Iterable[object]) -> Optional[float]:
    clean = []
    for value in values:
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return float(np.mean(clean)) if clean else None


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or float(denominator) <= EPS:
        return None
    return float(numerator) / float(denominator)


def _format_float(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{float(value):.4g}"


def _prediction_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((pred - target) ** 2, axis=1)


def _purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    if labels_true.size == 0:
        return float("nan")
    hits = 0
    for pred_label in set(labels_pred.tolist()):
        mask = labels_pred == pred_label
        if not bool(np.any(mask)):
            continue
        counts = Counter(int(item) for item in labels_true[mask].tolist())
        if counts:
            hits += counts.most_common(1)[0][1]
    return float(hits / labels_true.size)


def _encode_categorical(labels: np.ndarray) -> np.ndarray:
    mapping: Dict[str, int] = {}
    codes = np.empty(labels.shape[0], dtype=np.int64)
    for index, item in enumerate(labels.tolist()):
        key = repr(item)
        if key not in mapping:
            mapping[key] = len(mapping)
        codes[index] = mapping[key]
    return codes


def _cluster_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, Optional[float]]:
    valid = labels_true >= 0
    if not bool(np.any(valid)):
        return {"basin_ari": None, "basin_nmi": None, "basin_purity": None}
    y = labels_true[valid].astype(int, copy=False)
    p = _encode_categorical(labels_pred[valid])
    if len(set(y.tolist())) < 2 or len(set(p.tolist())) < 2:
        same = float(len(set(y.tolist())) == len(set(p.tolist())) == 1)
        return {"basin_ari": same, "basin_nmi": same, "basin_purity": _purity(y, p)}
    return {
        "basin_ari": float(adjusted_rand_score(y, p)),
        "basin_nmi": float(normalized_mutual_info_score(y, p)),
        "basin_purity": _purity(y, p),
    }


def _train_test_split(num_items: int, *, train_fraction: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if num_items <= 1:
        return np.ones(num_items, dtype=bool), np.zeros(num_items, dtype=bool)
    order = rng.permutation(num_items)
    train_size = int(round(train_fraction * num_items))
    train_size = max(1, min(num_items - 1, train_size))
    train = np.zeros(num_items, dtype=bool)
    train[order[:train_size]] = True
    return train, ~train


def _subsample_indices(num_items: int, max_items: int, rng: np.random.Generator) -> np.ndarray:
    if max_items <= 0 or num_items <= max_items:
        return np.arange(num_items)
    return np.sort(rng.choice(num_items, size=max_items, replace=False))


def _fit_predict_clusters(
    train_features: np.ndarray,
    test_features: np.ndarray,
    *,
    method: str,
    n_clusters: int,
    seed: int,
    max_samples: int,
    spectral_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")
    if train_features.shape[0] < n_clusters:
        raise ValueError("not enough training samples for requested clusters")

    rng = np.random.default_rng(seed)
    fit_idx = _subsample_indices(train_features.shape[0], max_samples, rng)
    fit_features = train_features[fit_idx]
    if fit_features.shape[0] < n_clusters:
        raise ValueError("subsample too small for requested clusters")

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        model.fit(fit_features)
        return model.predict(train_features).astype(object), model.predict(test_features).astype(object), "direct_predict"

    if method == "gmm_diag":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            reg_covar=1e-5,
            random_state=seed,
            max_iter=200,
            n_init=3,
        )
        model.fit(fit_features)
        return model.predict(train_features).astype(object), model.predict(test_features).astype(object), "direct_predict"

    if method == "spectral":
        neighbors = max(1, min(int(spectral_neighbors), fit_features.shape[0] - 1))
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=neighbors,
            assign_labels="kmeans",
            random_state=seed,
        )
        fit_labels = model.fit_predict(fit_features)
        centers = []
        for cluster_id in range(n_clusters):
            mask = fit_labels == cluster_id
            if bool(np.any(mask)):
                centers.append(fit_features[mask].mean(axis=0))
            else:
                centers.append(fit_features[rng.integers(0, fit_features.shape[0])])
        center_array = np.stack(centers, axis=0).astype(np.float32, copy=False)

        def assign(features: np.ndarray) -> np.ndarray:
            diff = features[:, None, :] - center_array[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            return np.argmin(d2, axis=1).astype(object)

        return assign(train_features), assign(test_features), "nearest_spectral_centroid"

    raise ValueError(f"Unknown cluster method '{method}'")


def _fit_partition_metrics(
    z_train: np.ndarray,
    z_next_train: np.ndarray,
    labels_train: np.ndarray,
    basin_train: np.ndarray,
    z_test: np.ndarray,
    z_next_test: np.ndarray,
    labels_test: np.ndarray,
    basin_test: np.ndarray,
    *,
    global_k: np.ndarray,
    ridge_lambda: float,
    min_transitions: int,
) -> Dict[str, object]:
    operators, centers, counts = CENTERED._fit_partition_centered(
        z_train,
        z_next_train,
        labels_train,
        ridge_lambda,
        min_transitions=min_transitions,
    )
    covered = np.asarray([label in operators for label in labels_test.tolist()], dtype=bool)
    partition_mse = None
    if bool(np.any(covered)):
        errors: List[np.ndarray] = []
        for label, operator in operators.items():
            mask = np.logical_and(covered, labels_test == label)
            if not bool(np.any(mask)):
                continue
            pred = CENTERED._predict_centered(z_test[mask], centers[label], operator)
            errors.append(_prediction_mse(pred, z_next_test[mask]))
        if errors:
            partition_mse = float(np.concatenate(errors, axis=0).mean())
    global_mse_on_covered = (
        float(OPSEL._predict_mse(z_test[covered], z_next_test[covered], global_k).mean())
        if bool(np.any(covered))
        else None
    )
    full_global_mse = float(OPSEL._predict_mse(z_test, z_next_test, global_k).mean()) if z_test.size else None
    return {
        "class_count_total": float(len(counts)),
        "class_count_fit": float(len(operators)),
        "test_transition_count": float(z_test.shape[0]),
        "test_covered_count": float(int(covered.sum())),
        "test_coverage_fraction": float(float(covered.mean()) if covered.size else 0.0),
        "global_k_test_mse_full": full_global_mse,
        "global_k_test_mse_on_covered": global_mse_on_covered,
        "partition_latent_mse": partition_mse,
        "partition_over_global_on_covered": _safe_ratio(partition_mse, global_mse_on_covered),
        "partition_over_global_full": _safe_ratio(partition_mse, full_global_mse),
        **_cluster_metrics(basin_test, labels_test),
    }


def _decode_array(model, latents: np.ndarray, *, device: str, dtype: torch.dtype, batch_size: int) -> np.ndarray:
    model_device = next(model.parameters()).device
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, latents.shape[0], batch_size):
            chunk = torch.from_numpy(latents[start : start + batch_size]).to(device=model_device, dtype=dtype)
            outs.append(model.decode(chunk).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outs, axis=0) if outs else np.empty((0, 0), dtype=np.float32)


def _feature_matrix(
    name: str,
    *,
    states: np.ndarray,
    latents: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    if name == "raw_state":
        return states.astype(np.float32, copy=False)
    if name == "dense_latent":
        return latents.astype(np.float32, copy=False)
    if name == "sparse_latent_values":
        return (latents * support_mask.astype(latents.dtype, copy=False)).astype(np.float32, copy=False)
    if name == "support_binary":
        return support_mask.astype(np.float32, copy=False)
    raise ValueError(f"Unknown feature view '{name}'")


def _support_family_train_test_labels(
    train_support_mask: np.ndarray,
    test_support_mask: np.ndarray,
    *,
    family_jaccard_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    train_keys = REDUCER._support_keys(train_support_mask).reshape(-1).astype(object)
    train_families = REDUCER.support_family_labels(
        train_support_mask.reshape(1, train_support_mask.shape[0], train_support_mask.shape[1]),
        min_jaccard=family_jaccard_threshold,
    ).reshape(-1).astype(object)
    prototypes = OPSEL._prototype_masks_from_exact_support(
        train_families,
        train_keys,
        train_support_mask,
        class_kind="family",
    )
    key_to_family: Dict[object, object] = {}
    for key, family in zip(train_keys.tolist(), train_families.tolist()):
        if key not in key_to_family:
            key_to_family[key] = family
    test_keys = REDUCER._support_keys(test_support_mask).reshape(-1).astype(object)
    test_families = np.empty(test_support_mask.shape[0], dtype=object)
    fallback_count = 0
    for index, (key, mask) in enumerate(zip(test_keys.tolist(), test_support_mask)):
        family = key_to_family.get(key)
        if family is not None:
            test_families[index] = family
            continue
        best_family = None
        best_similarity = -1.0
        for candidate_family, prototype in prototypes.items():
            similarity = REDUCER._binary_jaccard(mask, prototype)
            if similarity > best_similarity:
                best_similarity = similarity
                best_family = candidate_family
        if best_family is not None and best_similarity >= family_jaccard_threshold:
            test_families[index] = best_family
        else:
            test_families[index] = "__unassigned_support_family__"
            fallback_count += 1
    return train_families, test_families, len(set(train_families.tolist())), f"jaccard_unassigned={fallback_count}"


def evaluate_run(
    spec,
    *,
    support_scheme: str,
    support_value: float,
    feature_views: Sequence[str],
    cluster_methods: Sequence[str],
    cluster_count_modes: Sequence[str],
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
    endpoint_rollout_steps: int,
    device: str,
    label_mode: str,
    ridge_lambda: float,
    min_operator_transitions: int,
    family_jaccard_threshold: float,
    train_fraction: float,
    cluster_fit_max_samples: int,
    spectral_neighbors: int,
    decode_batch_size: int,
) -> List[Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    _cfg, env, model = REDUCER._load_checkpoint_model(checkpoint_path, spec.system_key, device)
    model_device = next(model.parameters()).device
    global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
    trajectories = REDUCER._generate_observation_trajectories(
        env,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        eval_seed=eval_seed,
    )
    basin_labels, _centers, label_source = OPSEL._label_sequences_for_mode(
        env,
        trajectories,
        system_key=spec.system_key,
        endpoint_rollout_steps=endpoint_rollout_steps,
        label_mode=label_mode,
    )
    latents = REDUCER._encode_trajectories(model, trajectories, device).astype(np.float32, copy=False)
    support_mask = REDUCER._support_mask(latents, scheme=support_scheme, value=support_value)

    z_all = latents[:, :-1, :].reshape(-1, latents.shape[-1]).astype(np.float32, copy=False)
    z_next_all = latents[:, 1:, :].reshape(-1, latents.shape[-1]).astype(np.float32, copy=False)
    state_all = trajectories[:, :-1, :].reshape(-1, trajectories.shape[-1]).cpu().numpy().astype(np.float32, copy=False)
    state_next_all = trajectories[:, 1:, :].reshape(-1, trajectories.shape[-1]).cpu().numpy().astype(np.float32, copy=False)
    support_all = support_mask[:, :-1, :].reshape(-1, support_mask.shape[-1]).astype(bool, copy=False)
    basin_all = basin_labels[:, :-1].reshape(-1).cpu().numpy().astype(np.int64, copy=False)

    split_rng = np.random.default_rng(eval_seed + spec.seed * 1009 + z_all.shape[0])
    train_mask, test_mask = _train_test_split(z_all.shape[0], train_fraction=train_fraction, rng=split_rng)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    if test_idx.size == 0:
        raise RuntimeError("No held-out transitions after train/test split")

    z_train, z_test = z_all[train_idx], z_all[test_idx]
    z_next_train, z_next_test = z_next_all[train_idx], z_next_all[test_idx]
    basin_train, basin_test = basin_all[train_idx], basin_all[test_idx]
    support_train, support_test = support_all[train_idx], support_all[test_idx]
    state_train, state_test = state_all[train_idx], state_all[test_idx]
    state_next_test = state_next_all[test_idx]

    support_definition = _stringify_support_definition(support_scheme, support_value)
    common = {
        "root_label": spec.root_label,
        "system_key": spec.system_key,
        "system_name": spec.system_name,
        "seed": spec.seed,
        "run_dir": spec.run_dir,
        "support_definition": support_definition,
        "label_mode": label_mode,
        "label_source": label_source,
        "num_trajectories": float(num_trajectories),
        "trajectory_length": float(trajectory_length),
        "eval_seed": float(eval_seed),
        "train_fraction": float(train_fraction),
        "ridge_lambda": float(ridge_lambda),
        "min_operator_transitions": float(min_operator_transitions),
        "family_jaccard_threshold": float(family_jaccard_threshold),
        "cluster_fit_max_samples": float(cluster_fit_max_samples),
    }

    rows: List[Dict[str, object]] = []
    global_pred_z = z_test @ global_k
    global_latent_mse = float(_prediction_mse(global_pred_z, z_next_test).mean())
    decoded_global = _decode_array(
        model,
        global_pred_z,
        device=device,
        dtype=trajectories.dtype,
        batch_size=decode_batch_size,
    )
    global_state_mse = float(_prediction_mse(decoded_global, state_next_test).mean())
    rows.append(
        {
            **common,
            "route_kind": "global_k",
            "feature_view": "none",
            "cluster_method": "none",
            "cluster_count_mode": "none",
            "cluster_count": 1.0,
            "assignment_mode": "checkpoint_global_k",
            "class_count_total": 1.0,
            "class_count_fit": 1.0,
            "test_transition_count": float(z_test.shape[0]),
            "test_covered_count": float(z_test.shape[0]),
            "test_coverage_fraction": 1.0,
            "global_k_test_mse_full": global_latent_mse,
            "global_k_test_mse_on_covered": global_latent_mse,
            "partition_latent_mse": global_latent_mse,
            "partition_over_global_on_covered": 1.0,
            "partition_over_global_full": 1.0,
            "partition_state_mse": global_state_mse,
            "partition_state_over_global": 1.0,
            "basin_ari": None,
            "basin_nmi": None,
            "basin_purity": None,
        }
    )

    family_train, family_test, support_family_count, support_assign_note = _support_family_train_test_labels(
        support_train,
        support_test,
        family_jaccard_threshold=family_jaccard_threshold,
    )
    route_specs: List[Tuple[str, str, str, str, np.ndarray, np.ndarray, int, str]] = [
        (
            "learned_support_family",
            "support_binary",
            "support_family",
            "support_family_count",
            family_train,
            family_test,
            support_family_count,
            support_assign_note,
        ),
        (
            "oracle_basin",
            "raw_state",
            "oracle_basin",
            "basin_count",
            basin_train.astype(object),
            basin_test.astype(object),
            len(set(int(item) for item in basin_train.tolist() if int(item) >= 0)),
            "evaluation_only_true_basin_labels",
        ),
    ]

    for route_kind, feature_view, method, count_mode, labels_train, labels_test, cluster_count, assignment_mode in route_specs:
        metrics = _fit_partition_metrics(
            z_train,
            z_next_train,
            labels_train,
            basin_train,
            z_test,
            z_next_test,
            labels_test,
            basin_test,
            global_k=global_k,
            ridge_lambda=ridge_lambda,
            min_transitions=min_operator_transitions,
        )
        state_ratio = None
        state_mse = None
        rows.append(
            {
                **common,
                "route_kind": route_kind,
                "feature_view": feature_view,
                "cluster_method": method,
                "cluster_count_mode": count_mode,
                "cluster_count": float(cluster_count),
                "assignment_mode": assignment_mode,
                **metrics,
                "partition_state_mse": state_mse,
                "partition_state_over_global": state_ratio,
            }
        )

    count_values: Dict[str, int] = {
        "basin_count": max(2, len(set(int(item) for item in basin_train.tolist() if int(item) >= 0))),
        "support_family_count": max(2, int(support_family_count)),
    }
    feature_cache = {
        feature: (
            _feature_matrix(feature, states=state_train, latents=z_train, support_mask=support_train),
            _feature_matrix(feature, states=state_test, latents=z_test, support_mask=support_test),
        )
        for feature in feature_views
    }
    for feature_view in feature_views:
        train_features, test_features = feature_cache[feature_view]
        for count_mode in cluster_count_modes:
            n_clusters = int(count_values[count_mode])
            for method in cluster_methods:
                try:
                    labels_train, labels_test, assignment_mode = _fit_predict_clusters(
                        train_features,
                        test_features,
                        method=method,
                        n_clusters=n_clusters,
                        seed=eval_seed + spec.seed * 1000 + n_clusters + len(rows),
                        max_samples=cluster_fit_max_samples,
                        spectral_neighbors=spectral_neighbors,
                    )
                    metrics = _fit_partition_metrics(
                        z_train,
                        z_next_train,
                        labels_train,
                        basin_train,
                        z_test,
                        z_next_test,
                        labels_test,
                        basin_test,
                        global_k=global_k,
                        ridge_lambda=ridge_lambda,
                        min_transitions=min_operator_transitions,
                    )
                    rows.append(
                        {
                            **common,
                            "route_kind": "explicit_regime_discovery",
                            "feature_view": feature_view,
                            "cluster_method": method,
                            "cluster_count_mode": count_mode,
                            "cluster_count": float(n_clusters),
                            "assignment_mode": assignment_mode,
                            **metrics,
                            "partition_state_mse": None,
                            "partition_state_over_global": None,
                        }
                    )
                except Exception as exc:  # keep per-baseline failures local to the row
                    rows.append(
                        {
                            **common,
                            "route_kind": "explicit_regime_discovery",
                            "feature_view": feature_view,
                            "cluster_method": method,
                            "cluster_count_mode": count_mode,
                            "cluster_count": float(n_clusters),
                            "assignment_mode": "failed",
                            "skip_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

    return rows


def _load_completed_keys(path: Path) -> set[Tuple[str, str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    completed: set[Tuple[str, str, int]] = set()
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                completed.add((str(row["root_label"]), str(row["system_key"]), int(row["seed"])))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def _write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("root_label", "")),
            str(row.get("route_kind", "")),
            str(row.get("feature_view", "")),
            str(row.get("cluster_method", "")),
            str(row.get("cluster_count_mode", "")),
        )
        grouped[key].append(row)
    lines = [
        "# Regime-Discovery Local Koopman Summary",
        "",
        "Held-out one-step latent local Koopman comparison against learned supports and explicit unsupervised partitions.",
        "",
        "| root | route | feature | method | count | coverage | latent/global | basin ARI | basin NMI | purity |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, group in sorted(grouped.items()):
        root, route, feature, method, count = key
        lines.append(
            f"| `{root}` | `{route}` | `{feature}` | `{method}` | `{count}` | "
            f"{_format_float(_safe_mean(row.get('test_coverage_fraction') for row in group))} | "
            f"{_format_float(_safe_mean(row.get('partition_over_global_on_covered') for row in group))} | "
            f"{_format_float(_safe_mean(row.get('basin_ari') for row in group))} | "
            f"{_format_float(_safe_mean(row.get('basin_nmi') for row in group))} | "
            f"{_format_float(_safe_mean(row.get('basin_purity') for row in group))} |"
        )
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _flush(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    specs_total: int,
    specs_completed: int,
    status: str,
    elapsed: float,
) -> None:
    _write_csv(output_dir / "regime_discovery_local_koopman_rows.csv", rows)
    _write_summary(output_dir / "regime_discovery_local_koopman_summary.md", rows)
    _atomic_write_text(output_dir / "failures.json", json.dumps(list(failures), indent=2))
    _atomic_write_text(
        output_dir / "manifest.json",
        json.dumps(
            {
                "rows_csvs": _parse_csv_strings(args.rows_csvs),
                "root_labels": _parse_csv_strings(args.root_labels),
                "systems": _parse_csv_strings(args.systems),
                "seeds": _parse_csv_ints(args.seeds),
                "support_definition": args.support_definition,
                "feature_views": _parse_csv_strings(args.feature_views),
                "cluster_methods": _parse_csv_strings(args.cluster_methods),
                "cluster_count_modes": _parse_csv_strings(args.cluster_count_modes),
                "num_trajectories": args.num_trajectories,
                "trajectory_length": args.trajectory_length,
                "train_fraction": args.train_fraction,
                "cluster_fit_max_samples": args.cluster_fit_max_samples,
                "num_runs": specs_total,
                "completed_runs": specs_completed,
                "remaining_runs": max(0, specs_total - specs_completed),
                "num_rows": len(rows),
                "num_failures": len(failures),
                "elapsed_seconds": elapsed,
                "status": status,
            },
            indent=2,
        ),
    )


def main() -> None:
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    args = _parse_args()
    rows_csvs = [Path(item) for item in _parse_csv_strings(args.rows_csvs)]
    root_labels = _parse_csv_strings(args.root_labels)
    systems = _parse_csv_strings(args.systems)
    seeds = _parse_csv_ints(args.seeds)
    feature_views = _parse_csv_strings(args.feature_views)
    cluster_methods = _parse_csv_strings(args.cluster_methods)
    cluster_count_modes = _parse_csv_strings(args.cluster_count_modes)
    support_scheme, support_value = _parse_support_definition(args.support_definition)

    valid_features = {"raw_state", "dense_latent", "sparse_latent_values", "support_binary"}
    valid_methods = {"kmeans", "gmm_diag", "spectral"}
    valid_count_modes = {"basin_count", "support_family_count"}
    unknown_features = set(feature_views) - valid_features
    unknown_methods = set(cluster_methods) - valid_methods
    unknown_count_modes = set(cluster_count_modes) - valid_count_modes
    if unknown_features or unknown_methods or unknown_count_modes:
        raise ValueError(
            f"Unknown options: features={sorted(unknown_features)} "
            f"methods={sorted(unknown_methods)} count_modes={sorted(unknown_count_modes)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = OPSEL._load_latest_specs(rows_csvs, root_labels=root_labels, systems=systems, seeds=seeds)
    completed_keys = set() if args.no_resume else _load_completed_keys(output_dir / "regime_discovery_local_koopman_rows.csv")
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    if not args.no_resume and (output_dir / "regime_discovery_local_koopman_rows.csv").exists():
        with (output_dir / "regime_discovery_local_koopman_rows.csv").open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)

    start = time.time()
    completed_count = len(completed_keys)
    for index, spec in enumerate(specs, start=1):
        key = (spec.root_label, spec.system_key, int(spec.seed))
        if key in completed_keys:
            continue
        try:
            run_rows = evaluate_run(
                spec,
                support_scheme=support_scheme,
                support_value=support_value,
                feature_views=feature_views,
                cluster_methods=cluster_methods,
                cluster_count_modes=cluster_count_modes,
                num_trajectories=args.num_trajectories,
                trajectory_length=args.trajectory_length,
                eval_seed=args.eval_seed,
                endpoint_rollout_steps=args.endpoint_rollout_steps,
                device=args.device,
                label_mode=args.label_mode,
                ridge_lambda=args.ridge_lambda,
                min_operator_transitions=args.min_operator_transitions,
                family_jaccard_threshold=args.family_jaccard_threshold,
                train_fraction=args.train_fraction,
                cluster_fit_max_samples=args.cluster_fit_max_samples,
                spectral_neighbors=args.spectral_neighbors,
                decode_batch_size=args.decode_batch_size,
            )
            rows.extend(run_rows)
            completed_keys.add(key)
            completed_count += 1
            status = "ok"
            error = ""
        except Exception as exc:
            failures.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            completed_count += 1
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        if args.progress_every_runs and (index % int(args.progress_every_runs) == 0):
            print(
                f"[{index}/{len(specs)}] root={spec.root_label} system={spec.system_key} "
                f"seed={spec.seed} status={status} rows={len(rows)} failures={len(failures)} {error}",
                flush=True,
            )
        if args.flush_every_runs and (index % int(args.flush_every_runs) == 0):
            _flush(
                output_dir,
                args=args,
                rows=rows,
                failures=failures,
                specs_total=len(specs),
                specs_completed=completed_count,
                status="running",
                elapsed=time.time() - start,
            )
        if STOP_REQUESTED:
            break
        if args.max_runtime_seconds and (time.time() - start) >= float(args.max_runtime_seconds):
            break

    final_status = "complete" if completed_count >= len(specs) and not STOP_REQUESTED else "partial"
    _flush(
        output_dir,
        args=args,
        rows=rows,
        failures=failures,
        specs_total=len(specs),
        specs_completed=completed_count,
        status=final_status,
        elapsed=time.time() - start,
    )


if __name__ == "__main__":
    main()
