#!/usr/bin/env python3
"""Evaluate unsupervised mixture/gated local-linear dynamics baselines.

The baselines fit local affine one-step state-space maps from observed
trajectory pairs. Routing is unsupervised:

- k-means hard local linear;
- GMM hard local linear;
- GMM soft-gated local linear.

Known benchmark basin counts and labels are used only as optional diagnostics.
They are never used to assign samples during fitting.

Example:
    uv run python tools/evaluate_mixture_local_linear_baselines.py \
        --systems duffing,blended,multiwell \
        --seeds 0,1,2 \
        --component_mode fixed \
        --num_components 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from skae.config import apply_env_dt_override, get_config, get_env_dt
from skae.data import generate_trajectory, make_env


DEFAULT_SYSTEMS = (
    "gated_local_linear",
    "gated_transfer_linear",
    "claude:arrested_spiral",
    "claude:cal_asymmetric_3",
    "claude:cal_high_cross_3",
    "claude:cal_hexagon_6",
    "claude:cal_octagon_8",
    "claude:cal_pentagon_5",
    "claude:cal_square_4",
    "claude:duffing_triple_well",
    "claude:snic_multi",
    "claude:transition_routes_4",
    "claude:var_depth_gradient_4",
    "claude:var_diamond_4",
    "claude:var_l_shape_5",
)
METHOD_ALIASES = {
    "kmeans": "kmeans_hard",
    "kmeans_hard": "kmeans_hard",
    "gmm": "gmm_hard",
    "gmm_hard": "gmm_hard",
    "hard_gmm": "gmm_hard",
    "gmm_soft": "gmm_soft",
    "soft_gmm": "gmm_soft",
}
EPS = 1e-12
MAX_ABS_STATE_FOR_FIT = 1e6


def _finite_bounded_rows(
    states: np.ndarray,
    *,
    max_abs: float = MAX_ABS_STATE_FOR_FIT,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    finite = np.isfinite(states).all(axis=1)
    if states.size == 0:
        return finite
    return finite & (np.max(np.abs(states), axis=1) <= float(max_abs))


def _is_dysts_system(system: str) -> bool:
    return system.lower().startswith("dysts:")


def _resolve_dysts_dt(
    system: str,
    *,
    config_name: str,
    multiplier: float,
    standardize: bool,
) -> Optional[float]:
    if not _is_dysts_system(system) or multiplier <= 0.0:
        return None
    base_cfg = get_config(config_name)
    base_cfg.ENV.ENV_NAME = system
    base_cfg.ENV.DYSTS.STANDARDIZE = bool(standardize)
    base_env = make_env(base_cfg)
    base_dt = getattr(base_env.unwrapped, "dt", None)
    if base_dt is None:
        return None
    return float(base_dt) * float(multiplier)


@dataclass
class StateScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, states: np.ndarray) -> "StateScaler":
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2:
            raise ValueError("StateScaler.fit expects a 2D array.")
        # Dysts trajectories at large dt can numerically diverge; do not let
        # those states set the standardization scale for local-linear fits.
        valid = _finite_bounded_rows(states)
        fit_states = states[valid]
        if fit_states.size == 0:
            valid = np.isfinite(states).all(axis=1)
            fit_states = states[valid]
        if fit_states.size == 0:
            mean = np.zeros(states.shape[1], dtype=np.float64)
            scale = np.ones(states.shape[1], dtype=np.float64)
        else:
            mean = np.mean(fit_states, axis=0)
            scale = np.std(fit_states, axis=0)
            scale = np.where(scale > EPS, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, states: np.ndarray) -> np.ndarray:
        return (np.asarray(states, dtype=np.float64) - self.mean) / self.scale

    def inverse_transform(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(states, dtype=np.float64) * self.scale + self.mean


@dataclass
class SplitData:
    train_trajectories: np.ndarray
    test_trajectories: np.ndarray
    train_trajectories_scaled: np.ndarray
    test_trajectories_scaled: np.ndarray
    scaler: StateScaler
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_train_scaled: np.ndarray
    y_train_scaled: np.ndarray
    x_test_scaled: np.ndarray
    y_test_scaled: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class FitResult:
    model: "BaseLocalLinearModel"
    fit_seconds: float
    component_counts: List[float]
    train_one_step_mse: Optional[float]
    train_one_step_mse_per_dim: Optional[float]
    test_one_step_mse: Optional[float]
    test_one_step_mse_per_dim: Optional[float]
    diagnostic_labels: np.ndarray
    diagnostic_assignments: np.ndarray


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in _parse_csv_strings(raw)]


def _normalize_methods(raw: str) -> List[str]:
    methods = []
    for item in _parse_csv_strings(raw):
        key = item.lower()
        if key not in METHOD_ALIASES:
            raise ValueError(
                f"Unknown method '{item}'. Expected one of {sorted(METHOD_ALIASES)}."
            )
        canonical = METHOD_ALIASES[key]
        if canonical not in methods:
            methods.append(canonical)
    if not methods:
        raise ValueError("At least one method is required.")
    return methods


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(content)
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        _atomic_write_text(path, "")
        return
    seen: set[str] = set()
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue())


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_float(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{float(value):.6g}"


def _finite_state_mse(pred: np.ndarray, target: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
    finite = np.isfinite(pred).all(axis=-1) & np.isfinite(target).all(axis=-1)
    if not bool(np.any(finite)):
        return None, None, 0.0
    diff = pred[finite] - target[finite]
    raw = np.sum(diff * diff, axis=1)
    # Raw MSE sums over state dimensions; means are over finite samples.
    return float(np.mean(raw)), float(np.mean(raw / pred.shape[-1])), float(np.mean(finite))


def _finite_sample_errors(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    finite = np.isfinite(pred).all(axis=-1) & np.isfinite(target).all(axis=-1)
    raw = np.full(pred.shape[0], np.nan, dtype=np.float64)
    if bool(np.any(finite)):
        diff = pred[finite] - target[finite]
        raw[finite] = np.sum(diff * diff, axis=1)
    return raw, float(np.mean(finite)) if finite.size else 0.0


def _fit_ridge_affine(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge_lambda: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Fit y = [x, 1] @ W with ridge on linear terms only."""
    if x.size == 0:
        return None
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        valid &= np.isfinite(weights) & (weights > EPS)
    if not bool(np.any(valid)):
        return None

    x_valid = np.asarray(x[valid], dtype=np.float64)
    y_valid = np.asarray(y[valid], dtype=np.float64)
    x_aug = np.concatenate(
        [x_valid, np.ones((x_valid.shape[0], 1), dtype=np.float64)],
        axis=1,
    )

    if sample_weight is not None:
        w = np.sqrt(np.asarray(sample_weight[valid], dtype=np.float64)).reshape(-1, 1)
        x_aug = x_aug * w
        y_valid = y_valid * w

    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(ridge_lambda)
    reg[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y_valid
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _predict_affine(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return x_aug @ weights


def _predict_all_components(x: np.ndarray, maps: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return np.einsum("nd,kdp->nkp", x_aug, maps, optimize=True)


def _fit_component_maps(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_components: int,
    ridge_lambda: float,
    hard_labels: Optional[np.ndarray] = None,
    soft_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[float]]:
    global_map = _fit_ridge_affine(x, y, ridge_lambda=ridge_lambda)
    if global_map is None:
        raise ValueError("Cannot fit even the fallback global affine map.")

    maps = []
    counts: List[float] = []
    for component in range(num_components):
        if hard_labels is not None:
            mask = hard_labels == component
            counts.append(float(np.sum(mask)))
            local_map = _fit_ridge_affine(
                x[mask],
                y[mask],
                ridge_lambda=ridge_lambda,
            )
        elif soft_weights is not None:
            weights = soft_weights[:, component]
            counts.append(float(np.sum(weights)))
            local_map = _fit_ridge_affine(
                x,
                y,
                ridge_lambda=ridge_lambda,
                sample_weight=weights,
            )
        else:
            raise ValueError("Expected hard_labels or soft_weights.")
        maps.append(global_map if local_map is None else local_map)
    return np.stack(maps, axis=0), counts


class BaseLocalLinearModel:
    method: str
    maps: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def assignments(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class KMeansHardLocalLinear(BaseLocalLinearModel):
    def __init__(self, clusterer: KMeans, maps: np.ndarray):
        self.method = "kmeans_hard"
        self.clusterer = clusterer
        self.maps = maps

    def assignments(self, x: np.ndarray) -> np.ndarray:
        return self.clusterer.predict(np.asarray(x, dtype=np.float64))

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        labels = self.assignments(x)
        all_pred = _predict_all_components(x, self.maps)
        return all_pred[np.arange(x.shape[0]), labels]


class GMMHardLocalLinear(BaseLocalLinearModel):
    def __init__(self, gmm: GaussianMixture, maps: np.ndarray):
        self.method = "gmm_hard"
        self.gmm = gmm
        self.maps = maps

    def assignments(self, x: np.ndarray) -> np.ndarray:
        return self.gmm.predict(np.asarray(x, dtype=np.float64))

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        labels = self.assignments(x)
        all_pred = _predict_all_components(x, self.maps)
        return all_pred[np.arange(x.shape[0]), labels]


class GMMSoftLocalLinear(BaseLocalLinearModel):
    def __init__(self, gmm: GaussianMixture, maps: np.ndarray):
        self.method = "gmm_soft"
        self.gmm = gmm
        self.maps = maps

    def assignments(self, x: np.ndarray) -> np.ndarray:
        responsibilities = self.gmm.predict_proba(np.asarray(x, dtype=np.float64))
        return responsibilities.argmax(axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        responsibilities = self.gmm.predict_proba(x)
        all_pred = _predict_all_components(x, self.maps)
        return np.einsum("nk,nkd->nd", responsibilities, all_pred, optimize=True)


def _fit_model(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    num_components: int,
    ridge_lambda: float,
    seed: int,
) -> Tuple[BaseLocalLinearModel, List[float]]:
    if method == "kmeans_hard":
        clusterer = KMeans(
            n_clusters=num_components,
            n_init=10,
            random_state=int(seed),
        )
        labels = clusterer.fit_predict(x_train)
        maps, counts = _fit_component_maps(
            x_train,
            y_train,
            num_components=num_components,
            ridge_lambda=ridge_lambda,
            hard_labels=labels,
        )
        return KMeansHardLocalLinear(clusterer, maps), counts

    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type="full",
        reg_covar=1e-6,
        n_init=3,
        max_iter=200,
        random_state=int(seed),
    )
    gmm.fit(x_train)
    if method == "gmm_hard":
        labels = gmm.predict(x_train)
        maps, counts = _fit_component_maps(
            x_train,
            y_train,
            num_components=num_components,
            ridge_lambda=ridge_lambda,
            hard_labels=labels,
        )
        return GMMHardLocalLinear(gmm, maps), counts
    if method == "gmm_soft":
        responsibilities = gmm.predict_proba(x_train)
        maps, counts = _fit_component_maps(
            x_train,
            y_train,
            num_components=num_components,
            ridge_lambda=ridge_lambda,
            soft_weights=responsibilities,
        )
        return GMMSoftLocalLinear(gmm, maps), counts
    raise ValueError(f"Unknown method '{method}'.")


def _generate_trajectories(
    env,
    *,
    num_trajectories: int,
    trajectory_length: int,
    seed: int,
) -> np.ndarray:
    trajectories = []
    with torch.no_grad():
        for idx in range(num_trajectories):
            rng = torch.Generator().manual_seed(int(seed) + 1009 * idx)
            init_state = env.reset(rng)
            tail = generate_trajectory(env.step, init_state, length=trajectory_length)
            trajectory = torch.cat([init_state.unsqueeze(0), tail], dim=0)
            trajectories.append(trajectory.detach().cpu().to(dtype=torch.float64).numpy())
    return np.stack(trajectories, axis=0)


def _split_trajectories(
    trajectories: np.ndarray,
    *,
    train_fraction: float,
    seed: int,
) -> SplitData:
    num_trajectories = trajectories.shape[0]
    if num_trajectories < 2:
        raise ValueError("At least two trajectories are required for a train/test split.")
    train_count = int(round(float(train_fraction) * num_trajectories))
    train_count = max(1, min(num_trajectories - 1, train_count))
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_trajectories)
    train_indices = np.sort(order[:train_count])
    test_indices = np.sort(order[train_count:])
    train = trajectories[train_indices]
    test = trajectories[test_indices]
    state_dim = trajectories.shape[-1]
    scaler = StateScaler.fit(train.reshape(-1, state_dim))
    train_scaled = scaler.transform(train.reshape(-1, state_dim)).reshape(train.shape)
    test_scaled = scaler.transform(test.reshape(-1, state_dim)).reshape(test.shape)
    x_train = train[:, :-1, :].reshape(-1, state_dim)
    y_train = train[:, 1:, :].reshape(-1, state_dim)
    x_test = test[:, :-1, :].reshape(-1, state_dim)
    y_test = test[:, 1:, :].reshape(-1, state_dim)
    x_train_scaled = train_scaled[:, :-1, :].reshape(-1, state_dim)
    y_train_scaled = train_scaled[:, 1:, :].reshape(-1, state_dim)
    x_test_scaled = test_scaled[:, :-1, :].reshape(-1, state_dim)
    y_test_scaled = test_scaled[:, 1:, :].reshape(-1, state_dim)
    train_pair_mask = (
        _finite_bounded_rows(x_train)
        & _finite_bounded_rows(y_train)
        & _finite_bounded_rows(x_train_scaled)
        & _finite_bounded_rows(y_train_scaled)
    )
    test_pair_mask = (
        _finite_bounded_rows(x_test)
        & _finite_bounded_rows(y_test)
        & _finite_bounded_rows(x_test_scaled)
        & _finite_bounded_rows(y_test_scaled)
    )
    if not np.any(train_pair_mask):
        raise ValueError("No finite train one-step pairs remain after filtering.")
    return SplitData(
        train_trajectories=train,
        test_trajectories=test,
        train_trajectories_scaled=train_scaled,
        test_trajectories_scaled=test_scaled,
        scaler=scaler,
        x_train=x_train[train_pair_mask],
        y_train=y_train[train_pair_mask],
        x_test=x_test[test_pair_mask],
        y_test=y_test[test_pair_mask],
        x_train_scaled=x_train_scaled[train_pair_mask],
        y_train_scaled=y_train_scaled[train_pair_mask],
        x_test_scaled=x_test_scaled[test_pair_mask],
        y_test_scaled=y_test_scaled[test_pair_mask],
        train_indices=train_indices,
        test_indices=test_indices,
    )


def _known_basin_count(env, system: str) -> Optional[int]:
    base = env.unwrapped
    if str(system).lower() == "duffing":
        return 2
    for attr in ("num_basins", "num_patterns"):
        value = getattr(base, attr, None)
        if value is not None:
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            if count > 0:
                return count
    for attr in ("points", "centers", "points_2d"):
        value = getattr(base, attr, None)
        if value is not None and hasattr(value, "shape") and len(value.shape) >= 1:
            count = int(value.shape[0])
            if count > 0:
                return count
    return None


def _resolve_num_components(
    *,
    requested: int,
    component_mode: str,
    known_count: Optional[int],
    max_samples: int,
) -> Tuple[int, str]:
    if component_mode == "known_basin_count":
        if known_count is not None and known_count > 0:
            raw = int(known_count)
            resolved_mode = "known_basin_count"
        else:
            raw = int(requested)
            resolved_mode = "fixed_fallback_no_known_basin_count"
    elif component_mode == "fixed":
        raw = int(requested)
        resolved_mode = "fixed"
    else:
        raise ValueError(
            f"Unknown component_mode '{component_mode}'. "
            "Expected 'fixed' or 'known_basin_count'."
        )
    if raw < 1:
        raise ValueError("--num_components must be positive.")
    return max(1, min(raw, int(max_samples))), resolved_mode


def _labels_from_tensor_call(fn, states: np.ndarray) -> Optional[np.ndarray]:
    tensor = torch.as_tensor(states, dtype=torch.float32)
    try:
        labels = fn(tensor)
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        labels = labels.detach().cpu().numpy().reshape(-1)
        if labels.shape[0] == states.shape[0]:
            return labels.astype(np.int64, copy=False)
    except Exception:
        pass
    labels_list = []
    try:
        for row in tensor:
            label = fn(row)
            if isinstance(label, torch.Tensor):
                label = int(label.detach().cpu().reshape(-1)[0].item())
            labels_list.append(int(label))
        return np.asarray(labels_list, dtype=np.int64)
    except Exception:
        return None


def _diagnostic_state_labels(env, system: str, states: np.ndarray) -> Tuple[np.ndarray, str]:
    base = env.unwrapped
    if hasattr(base, "basin_label"):
        labels = _labels_from_tensor_call(base.basin_label, states)
        if labels is not None:
            return labels, "basin_label"
    if hasattr(base, "get_basin_label"):
        labels = _labels_from_tensor_call(base.get_basin_label, states)
        if labels is not None:
            return labels, "get_basin_label"
    if str(system).lower() == "duffing":
        return (states[:, 0] >= 0.0).astype(np.int64), "duffing_x_sign"
    points = getattr(base, "points", None)
    if points is not None:
        points_np = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else np.asarray(points)
        if points_np.ndim == 2 and points_np.shape[1] == states.shape[1]:
            distances = np.linalg.norm(states[:, None, :] - points_np[None, :, :], axis=-1)
            return distances.argmin(axis=1).astype(np.int64), "nearest_attractor_point"
    return np.full(states.shape[0], -1, dtype=np.int64), "unavailable"


def _purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> Optional[float]:
    if labels_true.size == 0:
        return None
    hits = 0
    for pred_label in set(labels_pred.tolist()):
        mask = labels_pred == pred_label
        if not bool(np.any(mask)):
            continue
        counts = Counter(int(item) for item in labels_true[mask].tolist())
        if counts:
            hits += counts.most_common(1)[0][1]
    return float(hits / labels_true.size)


def _cluster_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    valid = labels_true >= 0
    if not bool(np.any(valid)):
        return {
            "diagnostic_label_ari": None,
            "diagnostic_label_nmi": None,
            "diagnostic_label_purity": None,
        }
    y = labels_true[valid].astype(np.int64, copy=False)
    p = labels_pred[valid].astype(np.int64, copy=False)
    purity = _purity(y, p)
    if len(set(y.tolist())) < 2 or len(set(p.tolist())) < 2:
        same = float(len(set(y.tolist())) == len(set(p.tolist())) == 1)
        return {
            "diagnostic_label_ari": same,
            "diagnostic_label_nmi": same,
            "diagnostic_label_purity": purity,
        }
    return {
        "diagnostic_label_ari": float(adjusted_rand_score(y, p)),
        "diagnostic_label_nmi": float(normalized_mutual_info_score(y, p)),
        "diagnostic_label_purity": purity,
    }


def _rollout_mse_by_horizon(
    model: BaseLocalLinearModel,
    trajectories: np.ndarray,
    scaler: StateScaler,
    horizons: Sequence[int],
) -> Dict[int, Dict[str, object]]:
    horizon_list = sorted({int(h) for h in horizons if int(h) > 0})
    if not horizon_list:
        raise ValueError("At least one positive horizon is required.")
    max_available = trajectories.shape[1] - 1
    horizon_list = [h for h in horizon_list if h <= max_available]
    if not horizon_list:
        raise ValueError(
            f"No requested horizons fit trajectory_length={max_available}."
        )
    max_horizon = max(horizon_list)
    num_start_times = trajectories.shape[1] - max_horizon
    starts = trajectories[:, :num_start_times, :].reshape(-1, trajectories.shape[-1])
    pred = scaler.transform(starts)
    state_dim = trajectories.shape[-1]
    step_raw_errors: List[np.ndarray] = []
    out: Dict[int, Dict[str, object]] = {}
    wanted = set(horizon_list)
    for step in range(1, max_horizon + 1):
        next_pred = np.full_like(pred, np.nan)
        valid_pred = _finite_bounded_rows(pred)
        if bool(np.any(valid_pred)):
            next_pred[valid_pred] = model.predict(pred[valid_pred])
        pred = next_pred
        pred_raw = scaler.inverse_transform(pred)
        target = trajectories[:, step : step + num_start_times, :].reshape(
            -1, trajectories.shape[-1]
        )
        raw_errors, endpoint_finite_fraction = _finite_sample_errors(pred_raw, target)
        step_raw_errors.append(raw_errors)
        if step not in wanted:
            continue

        endpoint = step_raw_errors[step - 1]
        raw_stack = np.stack(step_raw_errors[:step], axis=1)
        valid_counts = np.isfinite(raw_stack).sum(axis=1)
        cumulative = np.full(raw_stack.shape[0], np.nan, dtype=np.float64)
        valid_starts = valid_counts > 0
        # Cumulative MSE means over rollout steps per start state.
        cumulative[valid_starts] = np.nansum(raw_stack[valid_starts], axis=1) / valid_counts[valid_starts]
        finite = np.isfinite(cumulative)
        out[step] = {
            "rollout_starts": int(starts.shape[0]),
            "rollout_mse": _safe_float(np.nanmean(cumulative)) if bool(np.any(finite)) else None,
            "rollout_finite_fraction": float(np.mean(finite)) if finite.size else 0.0,
            "endpoint_mse_mean": _safe_float(np.nanmean(endpoint)) if np.isfinite(endpoint).any() else None,
            "endpoint_mse_per_dim_mean": (
                _safe_float(np.nanmean(endpoint / state_dim)) if np.isfinite(endpoint).any() else None
            ),
            "endpoint_finite_fraction": endpoint_finite_fraction,
            "cumulative_mse_mean": _safe_float(np.nanmean(cumulative)) if bool(np.any(finite)) else None,
            "cumulative_mse_per_dim_mean": (
                _safe_float(np.nanmean(cumulative / state_dim)) if bool(np.any(finite)) else None
            ),
        }
    return out


def _fit_and_evaluate(
    method: str,
    split: SplitData,
    *,
    num_components: int,
    ridge_lambda: float,
    seed: int,
    diagnostic_labels: np.ndarray,
) -> FitResult:
    start = time.perf_counter()
    model, counts = _fit_model(
        method,
        split.x_train_scaled,
        split.y_train_scaled,
        num_components=num_components,
        ridge_lambda=ridge_lambda,
        seed=seed,
    )
    fit_seconds = time.perf_counter() - start

    train_pred = split.scaler.inverse_transform(model.predict(split.x_train_scaled))
    test_pred = split.scaler.inverse_transform(model.predict(split.x_test_scaled))
    train_mse, train_mse_per_dim, _ = _finite_state_mse(train_pred, split.y_train)
    test_mse, test_mse_per_dim, _ = _finite_state_mse(test_pred, split.y_test)
    assignments = model.assignments(split.x_train_scaled)
    return FitResult(
        model=model,
        fit_seconds=fit_seconds,
        component_counts=counts,
        train_one_step_mse=train_mse,
        train_one_step_mse_per_dim=train_mse_per_dim,
        test_one_step_mse=test_mse,
        test_one_step_mse_per_dim=test_mse_per_dim,
        diagnostic_labels=diagnostic_labels,
        diagnostic_assignments=assignments,
    )


def _error_rows(
    *,
    system: str,
    seed: int,
    method: str,
    horizons: Sequence[int],
    skip_reason: str,
    args: argparse.Namespace,
    env_dt: float,
    state_dim: int,
    known_count: Optional[int],
    num_components: int,
    resolved_component_mode: str,
    train_trajectories: int,
    test_trajectories: int,
    train_pairs: int,
    test_pairs: int,
    diagnostic_label_source: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for horizon in horizons:
        rows.append(
            {
                "system": system,
                "seed": int(seed),
                "method": method,
                "status": "error",
                "skip_reason": skip_reason,
                "component_mode": args.component_mode,
                "resolved_component_mode": resolved_component_mode,
                "requested_num_components": int(args.num_components),
                "num_components": int(num_components),
                "known_basin_count": known_count,
                "ridge_lambda": float(args.ridge_lambda),
                "state_scaling": "train_standardized",
                "env_dt": env_dt,
                "state_dim": int(state_dim),
                "num_trajectories": int(args.num_trajectories),
                "trajectory_length": int(args.trajectory_length),
                "train_fraction": float(args.train_fraction),
                "train_trajectories": int(train_trajectories),
                "test_trajectories": int(test_trajectories),
                "train_pairs": int(train_pairs),
                "test_pairs": int(test_pairs),
                "rollout_starts": 0,
                "horizon": int(horizon),
                "rollout_mse": None,
                "rollout_finite_fraction": 0.0,
                "endpoint_mse_mean": None,
                "endpoint_mse_per_dim_mean": None,
                "endpoint_finite_fraction": 0.0,
                "cumulative_mse_mean": None,
                "cumulative_mse_per_dim_mean": None,
                "train_one_step_mse": None,
                "train_one_step_mse_per_dim": None,
                "test_one_step_mse": None,
                "test_one_step_mse_per_dim": None,
                "fit_seconds": None,
                "eval_seconds": None,
                "diagnostic_label_source": diagnostic_label_source,
                "component_counts_json": "[]",
            }
        )
    return rows


def _summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    groups: Dict[Tuple[str, str, int], List[float]] = defaultdict(list)
    final_groups: Dict[Tuple[str, str], Tuple[int, List[float]]] = {}
    for row in rows:
        mse = _safe_float(row.get("rollout_mse"))
        if mse is None:
            continue
        key = (str(row["system"]), str(row["method"]), int(row["horizon"]))
        groups[key].append(mse)
        final_key = (str(row["system"]), str(row["method"]))
        horizon = int(row["horizon"])
        if final_key not in final_groups or horizon > final_groups[final_key][0]:
            final_groups[final_key] = (horizon, [mse])
        elif horizon == final_groups[final_key][0]:
            final_groups[final_key][1].append(mse)

    by_system_method_horizon = []
    for (system, method, horizon), values in sorted(groups.items()):
        by_system_method_horizon.append(
            {
                "system": system,
                "method": method,
                "horizon": horizon,
                # Mean over run rows, usually seeds, for this system/method/horizon.
                "mean_rollout_mse": float(np.mean(values)),
                "std_rollout_mse": float(np.std(values)),
                "num_rows": len(values),
            }
        )

    by_method_horizon: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for item in by_system_method_horizon:
        by_method_horizon[(item["method"], item["horizon"])].append(
            item["mean_rollout_mse"]
        )
    aggregate = []
    for (method, horizon), values in sorted(by_method_horizon.items()):
        aggregate.append(
            {
                "method": method,
                "horizon": horizon,
                # Mean over system-level means for this method/horizon.
                "mean_system_mean_rollout_mse": float(np.mean(values)),
                "std_system_mean_rollout_mse": float(np.std(values)),
                "num_systems": len(values),
            }
        )

    final_horizon = []
    for (system, method), (horizon, values) in sorted(final_groups.items()):
        final_horizon.append(
            {
                "system": system,
                "method": method,
                "horizon": horizon,
                # Mean over run rows at the largest available horizon for this pair.
                "mean_rollout_mse": float(np.mean(values)),
                "num_rows": len(values),
            }
        )
    return {
        "num_rows": len(rows),
        "by_system_method_horizon": by_system_method_horizon,
        "aggregate_by_method_horizon": aggregate,
        "final_horizon_by_system_method": final_horizon,
    }


def _write_markdown_summary(
    path: Path,
    *,
    rows: Sequence[Dict[str, object]],
    summary: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Mixture Local-Linear Baseline Summary",
        "",
        f"- Rows: {len(rows)}",
        f"- Systems: `{','.join(_parse_csv_strings(args.systems))}`",
        f"- Methods: `{','.join(_normalize_methods(args.methods))}`",
        f"- Component mode: `{args.component_mode}`; fixed components: `{args.num_components}`",
        f"- Trajectories per run: `{args.num_trajectories}`; length: `{args.trajectory_length}`",
        "",
        "## Aggregate Mean MSE",
        "",
        "| method | horizon | mean system-mean rollout MSE | systems |",
        "|---|---:|---:|---:|",
    ]
    for item in summary["aggregate_by_method_horizon"]:
        lines.append(
            "| {method} | {horizon} | {mse} | {systems} |".format(
                method=item["method"],
                horizon=item["horizon"],
                mse=_format_float(item["mean_system_mean_rollout_mse"]),
                systems=item["num_systems"],
            )
        )
    lines.extend(
        [
            "",
            "## Final Horizon Per System",
            "",
            "| system | method | horizon | mean rollout MSE |",
            "|---|---|---:|---:|",
        ]
    )
    for item in summary["final_horizon_by_system_method"]:
        lines.append(
            "| {system} | {method} | {horizon} | {mse} |".format(
                system=item["system"],
                method=item["method"],
                horizon=item["horizon"],
                mse=_format_float(item["mean_rollout_mse"]),
            )
        )
    lines.extend(
        [
            "",
            "Known basin counts and label agreement columns are diagnostics only; fits use only observed states and one-step successors.",
            "",
        ]
    )
    _atomic_write_text(path, "\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        default=",".join(DEFAULT_SYSTEMS),
        help="Comma-separated environment names.",
    )
    parser.add_argument("--seeds", default="0", help="Comma-separated integer seeds.")
    parser.add_argument(
        "--methods",
        default="kmeans_hard,gmm_hard,gmm_soft",
        help="Comma-separated methods: kmeans_hard,gmm_hard,gmm_soft.",
    )
    parser.add_argument("--num_components", type=int, default=4)
    parser.add_argument(
        "--component_mode",
        default="fixed",
        choices=["fixed", "known_basin_count"],
        help="Use fixed --num_components or the known benchmark basin count as an evaluation diagnostic baseline.",
    )
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=1000)
    parser.add_argument("--train_fraction", type=float, default=0.6)
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument(
        "--horizons",
        default="100,500,1000",
        help="Comma-separated rollout horizons.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/mixture_local_linear_baselines",
        help="Directory for rows.csv, summary.json, and summary.md.",
    )
    parser.add_argument(
        "--env_dt",
        type=float,
        default=0.0,
        help="Optional positive dt override applied through skae.config.",
    )
    parser.add_argument(
        "--dysts_dt_multiplier",
        type=float,
        default=0.0,
        help="If >0 for dysts:* systems, use multiplier times the intrinsic Dysts dt.",
    )
    parser.add_argument(
        "--dysts_standardize",
        type=int,
        default=0,
        help="Set to 1 to evaluate dysts:* systems in standardized coordinates.",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="Config preset passed to skae.config.get_config.",
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=1,
        help="Torch CPU threads for deterministic lightweight trajectory generation.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_trajectories < 2:
        raise ValueError("--num_trajectories must be at least 2.")
    if args.trajectory_length < 1:
        raise ValueError("--trajectory_length must be positive.")
    if not (0.0 < float(args.train_fraction) < 1.0):
        raise ValueError("--train_fraction must be in (0, 1).")
    if args.ridge_lambda < 0.0:
        raise ValueError("--ridge_lambda must be nonnegative.")
    if args.dysts_dt_multiplier < 0.0:
        raise ValueError("--dysts_dt_multiplier must be nonnegative.")
    _normalize_methods(args.methods)
    _parse_csv_ints(args.seeds)
    _parse_csv_ints(args.horizons)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))

    systems = _parse_csv_strings(args.systems)
    seeds = _parse_csv_ints(args.seeds)
    methods = _normalize_methods(args.methods)
    horizons = sorted({int(h) for h in _parse_csv_ints(args.horizons) if int(h) > 0})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.csv"

    rows: List[Dict[str, object]] = []
    for system in systems:
        for seed in seeds:
            print(f"[system={system} seed={seed}] generating trajectories", flush=True)
            try:
                cfg = get_config(args.config)
                cfg.SEED = int(seed)
                cfg.ENV.ENV_NAME = system
                if _is_dysts_system(system):
                    cfg.ENV.DYSTS.STANDARDIZE = bool(int(args.dysts_standardize))
                if args.env_dt > 0.0:
                    apply_env_dt_override(cfg, float(args.env_dt), env_name=system)
                else:
                    dysts_dt = _resolve_dysts_dt(
                        system,
                        config_name=args.config,
                        multiplier=float(args.dysts_dt_multiplier),
                        standardize=bool(int(args.dysts_standardize)),
                    )
                    if dysts_dt is not None and dysts_dt > 0.0:
                        apply_env_dt_override(cfg, dysts_dt, env_name=system)
                env = make_env(cfg)
                env_dt = getattr(env.unwrapped, "dt", get_env_dt(cfg, system))
                trajectory_seed = _stable_seed("trajectories", system, seed)
                trajectories = _generate_trajectories(
                    env,
                    num_trajectories=int(args.num_trajectories),
                    trajectory_length=int(args.trajectory_length),
                    seed=trajectory_seed,
                )
                split_seed = _stable_seed("split", system, seed)
                split = _split_trajectories(
                    trajectories,
                    train_fraction=float(args.train_fraction),
                    seed=split_seed,
                )
                known_count = _known_basin_count(env, system)
                num_components, resolved_component_mode = _resolve_num_components(
                    requested=int(args.num_components),
                    component_mode=str(args.component_mode),
                    known_count=known_count,
                    max_samples=split.x_train.shape[0],
                )
                diagnostic_labels, label_source = _diagnostic_state_labels(
                    env,
                    system,
                    split.x_train,
                )
            except Exception as exc:
                print(f"[system={system} seed={seed}] ERROR generating data: {exc}", flush=True)
                for method in methods:
                    rows.extend(
                        _error_rows(
                            system=system,
                            seed=seed,
                            method=method,
                            horizons=horizons,
                            skip_reason=str(exc),
                            args=args,
                            env_dt=float("nan"),
                            state_dim=-1,
                            known_count=None,
                            num_components=int(args.num_components),
                            resolved_component_mode=str(args.component_mode),
                            train_trajectories=0,
                            test_trajectories=0,
                            train_pairs=0,
                            test_pairs=0,
                            diagnostic_label_source="",
                        )
                    )
                _write_csv(rows_path, rows)
                continue

            for method in methods:
                method_seed = _stable_seed("fit", system, seed, method)
                print(
                    f"[system={system} seed={seed}] fitting {method} "
                    f"(components={num_components})",
                    flush=True,
                )
                try:
                    fit = _fit_and_evaluate(
                        method,
                        split,
                        num_components=num_components,
                        ridge_lambda=float(args.ridge_lambda),
                        seed=method_seed,
                        diagnostic_labels=diagnostic_labels,
                    )
                    metrics = _cluster_metrics(
                        fit.diagnostic_labels,
                        fit.diagnostic_assignments,
                    )
                    eval_start = time.perf_counter()
                    horizon_mse = _rollout_mse_by_horizon(
                        fit.model,
                        split.test_trajectories,
                        split.scaler,
                        horizons,
                    )
                    eval_seconds = time.perf_counter() - eval_start
                    component_counts_json = json.dumps(
                        fit.component_counts,
                        separators=(",", ":"),
                    )
                    for horizon, rollout_metrics in horizon_mse.items():
                        row = {
                            "system": system,
                            "seed": int(seed),
                            "method": method,
                            "status": "ok",
                            "skip_reason": "",
                            "component_mode": args.component_mode,
                            "resolved_component_mode": resolved_component_mode,
                            "requested_num_components": int(args.num_components),
                            "num_components": int(num_components),
                            "known_basin_count": known_count,
                            "ridge_lambda": float(args.ridge_lambda),
                            "state_scaling": "train_standardized",
                            "env_dt": float(env_dt),
                            "state_dim": int(trajectories.shape[-1]),
                            "num_trajectories": int(args.num_trajectories),
                            "trajectory_length": int(args.trajectory_length),
                            "train_fraction": float(args.train_fraction),
                            "train_trajectories": int(split.train_trajectories.shape[0]),
                            "test_trajectories": int(split.test_trajectories.shape[0]),
                            "train_pairs": int(split.x_train.shape[0]),
                            "test_pairs": int(split.x_test.shape[0]),
                            "rollout_starts": int(rollout_metrics["rollout_starts"]),
                            "horizon": int(horizon),
                            "rollout_mse": rollout_metrics["rollout_mse"],
                            "rollout_finite_fraction": rollout_metrics["rollout_finite_fraction"],
                            "endpoint_mse_mean": rollout_metrics["endpoint_mse_mean"],
                            "endpoint_mse_per_dim_mean": rollout_metrics["endpoint_mse_per_dim_mean"],
                            "endpoint_finite_fraction": rollout_metrics["endpoint_finite_fraction"],
                            "cumulative_mse_mean": rollout_metrics["cumulative_mse_mean"],
                            "cumulative_mse_per_dim_mean": rollout_metrics["cumulative_mse_per_dim_mean"],
                            "train_one_step_mse": fit.train_one_step_mse,
                            "train_one_step_mse_per_dim": fit.train_one_step_mse_per_dim,
                            "test_one_step_mse": fit.test_one_step_mse,
                            "test_one_step_mse_per_dim": fit.test_one_step_mse_per_dim,
                            "fit_seconds": fit.fit_seconds,
                            "eval_seconds": eval_seconds,
                            "diagnostic_label_source": label_source,
                            "component_counts_json": component_counts_json,
                        }
                        row.update(metrics)
                        rows.append(row)
                except Exception as exc:
                    print(
                        f"[system={system} seed={seed}] ERROR method={method}: {exc}",
                        flush=True,
                    )
                    rows.extend(
                        _error_rows(
                            system=system,
                            seed=seed,
                            method=method,
                            horizons=horizons,
                            skip_reason=str(exc),
                            args=args,
                            env_dt=float(env_dt),
                            state_dim=int(trajectories.shape[-1]),
                            known_count=known_count,
                            num_components=int(num_components),
                            resolved_component_mode=resolved_component_mode,
                            train_trajectories=int(split.train_trajectories.shape[0]),
                            test_trajectories=int(split.test_trajectories.shape[0]),
                            train_pairs=int(split.x_train.shape[0]),
                            test_pairs=int(split.x_test.shape[0]),
                            diagnostic_label_source=label_source,
                        )
                    )
                _write_csv(rows_path, rows)

    _write_csv(rows_path, rows)
    summary = _summarize_rows(rows)
    payload = {
        "args": vars(args),
        "summary": summary,
    }
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    _write_markdown_summary(
        output_dir / "summary.md",
        rows=rows,
        summary=summary,
        args=args,
    )
    ok_rows = sum(1 for row in rows if row.get("status") == "ok")
    error_rows = sum(1 for row in rows if row.get("status") != "ok")
    print(f"Wrote {rows_path}", flush=True)
    print(f"Wrote {output_dir / 'summary.json'}", flush=True)
    print(f"Wrote {output_dir / 'summary.md'}", flush=True)
    if error_rows:
        print(f"Failing because {error_rows} error row(s) were emitted", flush=True)
        raise SystemExit(1)
    if ok_rows == 0:
        print("Failing because no successful rows were emitted", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
