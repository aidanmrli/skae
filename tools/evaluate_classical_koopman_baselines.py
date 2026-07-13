#!/usr/bin/env python3
"""Evaluate standalone classical Koopman baselines on 2D benchmark systems.

The script fits one-step linear maps in observable space and evaluates fully
autonomous rollouts from held-out initial conditions. It does not use basin
labels for fitting; labels are reported only as optional evaluation metadata
when an environment exposes them cheaply.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.config import apply_env_dt_override, get_config, get_env_dt
from skae.data import VectorWrapper, make_env


DEFAULT_SYSTEMS: Tuple[str, ...] = (
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

SUPPORTED_METHODS: Tuple[str, ...] = ("dmd", "edmd_poly", "rbf_dictionary_edmd")
EPS = 1e-12


def _parse_str_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(item) for item in _parse_str_list(raw)]


def _stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(content)
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        _atomic_write_text(path, "")
        return

    preferred = [
        "system",
        "seed",
        "method",
        "status",
        "skip_reason",
        "horizon",
        "endpoint_mse_mean",
        "endpoint_mse_median",
        "endpoint_mse_per_dim_mean",
        "cumulative_mse_mean",
        "cumulative_mse_median",
        "cumulative_mse_per_dim_mean",
        "finite_fraction",
        "env_dt",
        "state_dim",
        "feature_dim",
        "train_trajectories",
        "test_trajectories",
        "train_transitions",
        "num_trajectories",
        "trajectory_length",
        "train_fraction",
        "ridge_lambda",
        "edmd_degree",
        "kernel_centers_requested",
        "kernel_centers_used",
        "kernel_gamma",
        "test_initial_basin_hist",
        "test_final_basin_hist",
    ]
    fieldnames: List[str] = []
    seen = set()
    for key in preferred:
        if any(key in row for row in rows):
            fieldnames.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _atomic_write_text(path, buffer.getvalue())


def _finite_mean(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    # Mean over finite entries supplied by the caller.
    return float(finite.mean())


def _finite_median(values: np.ndarray) -> Optional[float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _format_optional_float(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.6g}"


@dataclass
class StateScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, states: np.ndarray) -> "StateScaler":
        mean = states.mean(axis=0, keepdims=True)
        scale = states.std(axis=0, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean.astype(np.float64), scale=scale.astype(np.float64))

    def transform(self, states: np.ndarray) -> np.ndarray:
        return (np.asarray(states, dtype=np.float64) - self.mean) / self.scale

    def inverse_transform(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(states, dtype=np.float64) * self.scale + self.mean


class IdentityFeatureMap:
    name = "identity"

    def __init__(self) -> None:
        self.feature_dim: Optional[int] = None

    def fit(self, states: np.ndarray, rng: np.random.Generator) -> "IdentityFeatureMap":
        del rng
        self.feature_dim = int(states.shape[1])
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(states, dtype=np.float64)


class PolynomialFeatureMap:
    name = "polynomial"

    def __init__(self, degree: int, include_bias: bool = True) -> None:
        if degree < 1:
            raise ValueError("edmd_degree must be >= 1")
        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        self.powers: List[Tuple[int, ...]] = []
        self.feature_dim: Optional[int] = None

    def fit(self, states: np.ndarray, rng: np.random.Generator) -> "PolynomialFeatureMap":
        del rng
        from itertools import combinations_with_replacement

        state_dim = int(states.shape[1])
        powers: List[Tuple[int, ...]] = []
        if self.include_bias:
            powers.append(tuple([0] * state_dim))
        for total_degree in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(state_dim), total_degree):
                power = [0] * state_dim
                for axis in combo:
                    power[axis] += 1
                powers.append(tuple(power))
        self.powers = powers
        self.feature_dim = len(powers)
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        if not self.powers:
            raise RuntimeError("PolynomialFeatureMap must be fitted before transform().")
        columns = []
        for power in self.powers:
            value = np.ones(states.shape[0], dtype=np.float64)
            for axis, exponent in enumerate(power):
                if exponent:
                    value *= states[:, axis] ** exponent
            columns.append(value)
        return np.stack(columns, axis=1)


class RBFDictionaryFeatureMap:
    name = "rbf_dictionary"

    def __init__(
        self,
        num_centers: int,
        gamma: float = 0.0,
        include_bias: bool = True,
        include_linear: bool = True,
    ) -> None:
        if num_centers < 1:
            raise ValueError("kernel_centers must be >= 1")
        self.num_centers = int(num_centers)
        self.gamma = float(gamma)
        self.include_bias = bool(include_bias)
        self.include_linear = bool(include_linear)
        self.centers: Optional[np.ndarray] = None
        self.gamma_used: Optional[float] = None
        self.feature_dim: Optional[int] = None

    @staticmethod
    def _pairwise_sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_norm = np.sum(x * x, axis=1, keepdims=True)
        y_norm = np.sum(y * y, axis=1, keepdims=True).T
        return np.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0)

    def fit(self, states: np.ndarray, rng: np.random.Generator) -> "RBFDictionaryFeatureMap":
        states = np.asarray(states, dtype=np.float64)
        center_count = min(self.num_centers, states.shape[0])
        if center_count == states.shape[0]:
            center_idx = np.arange(states.shape[0])
        else:
            center_idx = np.sort(rng.choice(states.shape[0], size=center_count, replace=False))
        centers = states[center_idx].copy()
        if self.gamma > 0.0:
            gamma_used = self.gamma
        elif centers.shape[0] <= 1:
            gamma_used = 1.0
        else:
            sqdist = self._pairwise_sqdist(centers, centers)
            upper = sqdist[np.triu_indices_from(sqdist, k=1)]
            positive = upper[upper > EPS]
            median_sqdist = float(np.median(positive)) if positive.size else 1.0
            gamma_used = 1.0 / max(median_sqdist, EPS)

        self.centers = centers
        self.gamma_used = float(gamma_used)
        dim = center_count
        if self.include_bias:
            dim += 1
        if self.include_linear:
            dim += int(states.shape[1])
        self.feature_dim = int(dim)
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        if self.centers is None or self.gamma_used is None:
            raise RuntimeError("RBFDictionaryFeatureMap must be fitted before transform().")
        parts = []
        if self.include_bias:
            parts.append(np.ones((states.shape[0], 1), dtype=np.float64))
        if self.include_linear:
            parts.append(states)
        sqdist = self._pairwise_sqdist(states, self.centers)
        parts.append(np.exp(-self.gamma_used * sqdist))
        return np.concatenate(parts, axis=1)


@dataclass
class FittedKoopmanBaseline:
    method: str
    scaler: StateScaler
    feature_map: object
    koopman_matrix: np.ndarray
    decoder_matrix: np.ndarray
    train_transitions: int

    @property
    def feature_dim(self) -> int:
        return int(self.koopman_matrix.shape[0])

    def rollout(self, initial_states: np.ndarray, horizon: int) -> np.ndarray:
        z = self.feature_map.transform(self.scaler.transform(initial_states))
        predictions = np.full(
            (initial_states.shape[0], int(horizon), initial_states.shape[1]),
            np.nan,
            dtype=np.float64,
        )
        for step in range(int(horizon)):
            with np.errstate(over="ignore", invalid="ignore"):
                z = z @ self.koopman_matrix
                next_state_scaled = z @ self.decoder_matrix
            if not np.isfinite(z).any() and not np.isfinite(next_state_scaled).any():
                break
            predictions[:, step, :] = self.scaler.inverse_transform(next_state_scaled)
        return predictions


def _ridge_solve(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ridge_lambda = float(ridge_lambda)
    if ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be nonnegative")
    if ridge_lambda == 0.0:
        return np.linalg.lstsq(x, y, rcond=None)[0]

    gram = x.T @ x
    rhs = x.T @ y
    gram = gram + ridge_lambda * np.eye(gram.shape[0], dtype=np.float64)
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ rhs


def _make_feature_map(
    method: str,
    *,
    edmd_degree: int,
    kernel_centers: int,
    kernel_gamma: float,
) -> object:
    if method == "dmd":
        return IdentityFeatureMap()
    if method == "edmd_poly":
        return PolynomialFeatureMap(edmd_degree, include_bias=True)
    if method == "rbf_dictionary_edmd":
        return RBFDictionaryFeatureMap(
            kernel_centers,
            gamma=kernel_gamma,
            include_bias=True,
            include_linear=True,
        )
    raise ValueError(f"Unknown method '{method}'. Expected one of {SUPPORTED_METHODS}.")


def _subsample_pairs(
    x: np.ndarray,
    y: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_pairs <= 0 or x.shape[0] <= max_pairs:
        return x, y
    idx = np.sort(rng.choice(x.shape[0], size=int(max_pairs), replace=False))
    return x[idx], y[idx]


def fit_baseline(
    method: str,
    train_trajectories: np.ndarray,
    *,
    edmd_degree: int,
    kernel_centers: int,
    kernel_gamma: float,
    ridge_lambda: float,
    max_train_pairs: int,
    rng: np.random.Generator,
) -> FittedKoopmanBaseline:
    all_train_states = train_trajectories.reshape(-1, train_trajectories.shape[-1])
    scaler = StateScaler.fit(all_train_states)

    x = train_trajectories[:, :-1, :].reshape(-1, train_trajectories.shape[-1])
    y = train_trajectories[:, 1:, :].reshape(-1, train_trajectories.shape[-1])
    x, y = _subsample_pairs(x, y, max_pairs=max_train_pairs, rng=rng)

    x_scaled = scaler.transform(x)
    y_scaled = scaler.transform(y)
    feature_map = _make_feature_map(
        method,
        edmd_degree=edmd_degree,
        kernel_centers=kernel_centers,
        kernel_gamma=kernel_gamma,
    ).fit(x_scaled, rng)

    phi_x = feature_map.transform(x_scaled)
    phi_y = feature_map.transform(y_scaled)
    koopman_matrix = _ridge_solve(phi_x, phi_y, ridge_lambda)
    if method == "dmd":
        decoder_matrix = np.eye(phi_x.shape[1], x_scaled.shape[1], dtype=np.float64)
    else:
        decoder_matrix = _ridge_solve(phi_x, x_scaled, ridge_lambda)

    return FittedKoopmanBaseline(
        method=method,
        scaler=scaler,
        feature_map=feature_map,
        koopman_matrix=koopman_matrix,
        decoder_matrix=decoder_matrix,
        train_transitions=int(x.shape[0]),
    )


def _evaluate_rollout(
    model: FittedKoopmanBaseline,
    test_trajectories: np.ndarray,
    horizons: Sequence[int],
) -> Dict[int, Dict[str, Optional[float]]]:
    max_horizon = max(horizons)
    predictions = model.rollout(test_trajectories[:, 0, :], horizon=max_horizon)
    targets = test_trajectories[:, 1 : max_horizon + 1, :]
    squared_error = (predictions - targets) ** 2
    # Raw MSE sums squared error over state dimensions.
    step_squared_error = squared_error.sum(axis=2)
    state_dim = test_trajectories.shape[-1]

    metrics: Dict[int, Dict[str, Optional[float]]] = {}
    for horizon in horizons:
        endpoint = step_squared_error[:, horizon - 1]
        # Cumulative MSE means over rollout steps per test trajectory.
        cumulative = step_squared_error[:, :horizon].mean(axis=1)
        finite = np.isfinite(cumulative)
        metrics[horizon] = {
            # These means are over held-out test trajectories.
            "endpoint_mse_mean": _finite_mean(endpoint),
            "endpoint_mse_median": _finite_median(endpoint),
            "endpoint_mse_per_dim_mean": _finite_mean(endpoint / state_dim),
            "cumulative_mse_mean": _finite_mean(cumulative),
            "cumulative_mse_median": _finite_median(cumulative),
            "cumulative_mse_per_dim_mean": _finite_mean(
                cumulative / state_dim
            ),
            "finite_fraction": float(finite.mean()) if finite.size else None,
        }
    return metrics


def _split_trajectories(
    num_trajectories: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if num_trajectories < 2:
        raise ValueError("num_trajectories must be at least 2")
    order = rng.permutation(num_trajectories)
    train_count = int(round(num_trajectories * train_fraction))
    train_count = max(1, min(num_trajectories - 1, train_count))
    train_idx = np.sort(order[:train_count])
    test_idx = np.sort(order[train_count:])
    return train_idx, test_idx


def _resolve_default_dt(system: str, explicit_env_dt: Optional[float]) -> Optional[float]:
    if explicit_env_dt is not None and explicit_env_dt > 0.0:
        return float(explicit_env_dt)
    try:
        from skae.benchmarks.paper_benchmark_manifest import resolve_system_default_dt

        return float(resolve_system_default_dt(system))
    except Exception:
        return None


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


def _make_env_for_system(
    system: str,
    *,
    seed: int,
    config_name: str,
    explicit_env_dt: Optional[float],
    dysts_dt_multiplier: float,
    dysts_standardize: bool,
) -> Tuple[object, float]:
    cfg = get_config(config_name)
    cfg.SEED = int(seed)
    cfg.ENV.ENV_NAME = system
    if _is_dysts_system(system):
        cfg.ENV.DYSTS.STANDARDIZE = bool(dysts_standardize)
    if explicit_env_dt is not None and explicit_env_dt > 0.0:
        requested_dt = float(explicit_env_dt)
    elif _is_dysts_system(system) and dysts_dt_multiplier > 0.0:
        requested_dt = _resolve_dysts_dt(
            system,
            config_name=config_name,
            multiplier=float(dysts_dt_multiplier),
            standardize=bool(dysts_standardize),
        )
    else:
        requested_dt = _resolve_default_dt(system, None)
    if requested_dt is not None and requested_dt > 0.0:
        apply_env_dt_override(cfg, requested_dt, env_name=system)
    env = make_env(cfg)
    env_dt = getattr(env.unwrapped, "dt", None)
    if env_dt is None:
        try:
            env_dt = get_env_dt(cfg, system)
        except Exception:
            env_dt = requested_dt
    return env, float(env_dt) if env_dt is not None else float("nan")


def _generate_trajectories(
    env: object,
    *,
    system: str,
    seed: int,
    num_trajectories: int,
    trajectory_length: int,
) -> np.ndarray:
    torch_seed = _stable_seed("trajectory", system, seed)
    rng = torch.Generator().manual_seed(torch_seed)
    vector_env = VectorWrapper(env, batch_size=int(num_trajectories))
    with torch.no_grad():
        trajectories = vector_env.generate_sequence_batch(
            rng=rng,
            window_length=int(trajectory_length),
        )
    return trajectories.detach().cpu().numpy().astype(np.float64)


def _hist(labels: Optional[np.ndarray]) -> str:
    if labels is None or labels.size == 0:
        return ""
    values, counts = np.unique(labels.astype(np.int64), return_counts=True)
    payload = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return json.dumps(payload, sort_keys=True)


def _maybe_basin_labels(env: object, states: np.ndarray) -> Optional[np.ndarray]:
    base_env = env.unwrapped
    tensor = torch.as_tensor(states, dtype=torch.float32)
    with torch.no_grad():
        if hasattr(base_env, "basin_label"):
            labels = base_env.basin_label(tensor)
        elif hasattr(base_env, "get_basin_label"):
            labels = base_env.get_basin_label(tensor)
        elif base_env.__class__.__name__ == "Duffing":
            labels = (tensor[..., 0] >= 0.0).to(dtype=torch.int64)
        elif hasattr(base_env, "points"):
            points = torch.as_tensor(getattr(base_env, "points"), dtype=tensor.dtype)
            if points.ndim != 2 or points.shape[1] != tensor.shape[-1]:
                return None
            diff = tensor.unsqueeze(-2) - points.unsqueeze(0)
            labels = torch.linalg.vector_norm(diff, dim=-1).argmin(dim=-1)
        else:
            return None

    if isinstance(labels, int):
        return np.asarray([labels], dtype=np.int64)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    return labels.detach().cpu().numpy().astype(np.int64).reshape(-1)


def _base_row(
    *,
    system: str,
    seed: int,
    method: str,
    env_dt: float,
    state_dim: int,
    num_trajectories: int,
    trajectory_length: int,
    train_fraction: float,
    train_count: int,
    test_count: int,
    ridge_lambda: float,
    edmd_degree: int,
    kernel_centers: int,
    labels_initial: Optional[np.ndarray],
    labels_final: Optional[np.ndarray],
) -> Dict[str, object]:
    return {
        "system": system,
        "seed": int(seed),
        "method": method,
        "env_dt": env_dt,
        "state_dim": int(state_dim),
        "num_trajectories": int(num_trajectories),
        "trajectory_length": int(trajectory_length),
        "train_fraction": float(train_fraction),
        "train_trajectories": int(train_count),
        "test_trajectories": int(test_count),
        "ridge_lambda": float(ridge_lambda),
        "edmd_degree": int(edmd_degree),
        "kernel_centers_requested": int(kernel_centers),
        "test_initial_basin_hist": _hist(labels_initial),
        "test_final_basin_hist": _hist(labels_final),
    }


def _error_rows(
    *,
    horizons: Sequence[int],
    skip_reason: str,
    **base: object,
) -> List[Dict[str, object]]:
    rows = []
    for horizon in horizons:
        row = dict(base)
        row.update({"status": "error", "skip_reason": skip_reason, "horizon": int(horizon)})
        rows.append(row)
    return rows


def _aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            key = (str(row["system"]), str(row["method"]), int(row["horizon"]))
        except KeyError:
            continue
        group = grouped.setdefault(
            key,
            {
                "system": key[0],
                "method": key[1],
                "horizon": key[2],
                "runs": 0,
                "endpoint_values": [],
                "cumulative_values": [],
            },
        )
        group["runs"] += 1
        for column, target in [
            ("endpoint_mse_mean", "endpoint_values"),
            ("cumulative_mse_mean", "cumulative_values"),
        ]:
            value = row.get(column)
            if value not in (None, "") and math.isfinite(float(value)):
                group[target].append(float(value))

    aggregates = []
    for group in sorted(grouped.values(), key=lambda item: (item["system"], item["horizon"], item["method"])):
        endpoint = np.asarray(group.pop("endpoint_values"), dtype=np.float64)
        cumulative = np.asarray(group.pop("cumulative_values"), dtype=np.float64)
        # Summary means are over run rows in this system/method/horizon bucket.
        group["endpoint_mse_mean_across_runs"] = _finite_mean(endpoint)
        group["endpoint_mse_median_across_runs"] = _finite_median(endpoint)
        group["cumulative_mse_mean_across_runs"] = _finite_mean(cumulative)
        group["cumulative_mse_median_across_runs"] = _finite_median(cumulative)
        aggregates.append(group)
    return aggregates


def _write_summary_json(path: Path, args: argparse.Namespace, rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    aggregates = _aggregate_rows(rows)
    payload = {
        "config": {
            "systems": _parse_str_list(args.systems),
            "seeds": _parse_int_list(args.seeds),
            "methods": _parse_str_list(args.methods),
            "horizons": _parse_int_list(args.horizons),
            "num_trajectories": args.num_trajectories,
            "trajectory_length": args.trajectory_length,
            "train_fraction": args.train_fraction,
            "edmd_degree": args.edmd_degree,
            "kernel_centers": args.kernel_centers,
            "kernel_gamma": args.kernel_gamma,
            "ridge_lambda": args.ridge_lambda,
            "max_train_pairs": args.max_train_pairs,
            "config_name": args.config_name,
            "env_dt": args.env_dt,
            "dysts_dt_multiplier": args.dysts_dt_multiplier,
            "dysts_standardize": int(args.dysts_standardize),
        },
        "num_rows": len(rows),
        "num_ok_rows": sum(1 for row in rows if row.get("status") == "ok"),
        "aggregates": aggregates,
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, default=_json_default) + "\n")
    return aggregates


def _write_markdown_summary(
    path: Path,
    args: argparse.Namespace,
    aggregates: Sequence[Dict[str, object]],
) -> None:
    horizons = _parse_int_list(args.horizons)
    largest_horizon = max(horizons)
    lines = [
        "# Classical Koopman Baseline Summary",
        "",
        f"- Systems: `{', '.join(_parse_str_list(args.systems))}`",
        f"- Methods: `{', '.join(_parse_str_list(args.methods))}`",
        f"- Seeds: `{', '.join(str(seed) for seed in _parse_int_list(args.seeds))}`",
        f"- Trajectories: {args.num_trajectories} x length {args.trajectory_length}",
        f"- Train fraction: {args.train_fraction:.3g}",
        "",
        f"Endpoint MSE table at horizon {largest_horizon}:",
        "",
        "| system | method | runs | endpoint MSE mean | cumulative MSE mean |",
        "|---|---|---:|---:|---:|",
    ]
    rows = [row for row in aggregates if int(row["horizon"]) == largest_horizon]
    for row in rows:
        lines.append(
            "| `{system}` | `{method}` | {runs} | {endpoint} | {cumulative} |".format(
                system=row["system"],
                method=row["method"],
                runs=row["runs"],
                endpoint=_format_optional_float(row.get("endpoint_mse_mean_across_runs")),
                cumulative=_format_optional_float(row.get("cumulative_mse_mean_across_runs")),
            )
        )
    if not rows:
        lines.append("| N/A | N/A | 0 | N/A | N/A |")
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `rows.csv`: per-system, per-seed, per-method, per-horizon metrics.",
            "- `summary.json`: aggregate metrics and run configuration.",
            "- `summary.md`: this short paper-facing summary.",
        ]
    )
    _atomic_write_text(path, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--methods", default=",".join(SUPPORTED_METHODS))
    parser.add_argument("--horizons", default="100,500,1000")
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=1000)
    parser.add_argument("--train_fraction", type=float, default=0.6)
    parser.add_argument("--edmd_degree", type=int, default=3)
    parser.add_argument("--kernel_centers", type=int, default=128)
    parser.add_argument(
        "--kernel_gamma",
        type=float,
        default=0.0,
        help="RBF gamma; <=0 uses a deterministic median-distance heuristic.",
    )
    parser.add_argument("--ridge_lambda", type=float, default=1e-6)
    parser.add_argument(
        "--max_train_pairs",
        type=int,
        default=0,
        help="Optional deterministic cap on transition pairs used for fitting; 0 uses all.",
    )
    parser.add_argument("--output_dir", default="runs/classical_koopman_baselines")
    parser.add_argument(
        "--env_dt",
        type=float,
        default=0.0,
        help="Optional global timestep override; 0 uses paper-manifest or config defaults.",
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
    parser.add_argument("--config_name", default="default")
    parser.add_argument(
        "--allow_non_2d",
        action="store_true",
        help="Permit non-2D systems. Defaults are restricted to 2D paper-benchmark systems.",
    )
    parser.add_argument("--torch_threads", type=int, default=1)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> Tuple[List[str], List[int], List[str], List[int]]:
    systems = _parse_str_list(args.systems)
    seeds = _parse_int_list(args.seeds)
    methods = [method.lower() for method in _parse_str_list(args.methods)]
    horizons = sorted(set(_parse_int_list(args.horizons)))
    if not systems:
        raise ValueError("At least one system is required.")
    if not seeds:
        raise ValueError("At least one seed is required.")
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported: {SUPPORTED_METHODS}")
    if args.num_trajectories < 2:
        raise ValueError("num_trajectories must be at least 2.")
    if args.trajectory_length < 1:
        raise ValueError("trajectory_length must be at least 1.")
    horizons = [horizon for horizon in horizons if 1 <= horizon <= args.trajectory_length]
    if not horizons:
        raise ValueError("No valid horizons remain after clipping to trajectory_length.")
    if args.ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be nonnegative.")
    if args.dysts_dt_multiplier < 0.0:
        raise ValueError("dysts_dt_multiplier must be nonnegative.")
    return systems, seeds, methods, horizons


def run(args: argparse.Namespace) -> Tuple[Path, Path, Path, int, int]:
    systems, seeds, methods, horizons = _validate_args(args)
    if args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.csv"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    rows: List[Dict[str, object]] = []

    explicit_env_dt = float(args.env_dt) if args.env_dt and args.env_dt > 0.0 else None
    for system in systems:
        for seed in seeds:
            print(f"[classical] generating system={system} seed={seed}", flush=True)
            try:
                env, env_dt = _make_env_for_system(
                    system,
                    seed=seed,
                    config_name=args.config_name,
                    explicit_env_dt=explicit_env_dt,
                    dysts_dt_multiplier=float(args.dysts_dt_multiplier),
                    dysts_standardize=bool(int(args.dysts_standardize)),
                )
                state_dim = int(env.observation_size)
                if state_dim != 2 and not args.allow_non_2d:
                    raise ValueError(
                        f"System '{system}' has state_dim={state_dim}; "
                        "this evaluator defaults to 2D systems only. "
                        "Pass --allow_non_2d to override."
                    )
                trajectories = _generate_trajectories(
                    env,
                    system=system,
                    seed=seed,
                    num_trajectories=args.num_trajectories,
                    trajectory_length=args.trajectory_length,
                )
                split_rng = np.random.default_rng(_stable_seed("split", system, seed))
                train_idx, test_idx = _split_trajectories(
                    trajectories.shape[0],
                    train_fraction=float(args.train_fraction),
                    rng=split_rng,
                )
                train_trajectories = trajectories[train_idx]
                test_trajectories = trajectories[test_idx]
                labels_initial = _maybe_basin_labels(env, test_trajectories[:, 0, :])
                labels_final = _maybe_basin_labels(env, test_trajectories[:, -1, :])
            except Exception as exc:
                print(f"[classical] ERROR system={system} seed={seed}: {exc}", flush=True)
                for method in methods:
                    base = _base_row(
                        system=system,
                        seed=seed,
                        method=method,
                        env_dt=float("nan"),
                        state_dim=-1,
                        num_trajectories=args.num_trajectories,
                        trajectory_length=args.trajectory_length,
                        train_fraction=args.train_fraction,
                        train_count=0,
                        test_count=0,
                        ridge_lambda=args.ridge_lambda,
                        edmd_degree=args.edmd_degree,
                        kernel_centers=args.kernel_centers,
                        labels_initial=None,
                        labels_final=None,
                    )
                    rows.extend(_error_rows(horizons=horizons, skip_reason=str(exc), **base))
                _write_csv(rows_path, rows)
                continue

            for method in methods:
                print(f"[classical] fitting system={system} seed={seed} method={method}", flush=True)
                base = _base_row(
                    system=system,
                    seed=seed,
                    method=method,
                    env_dt=env_dt,
                    state_dim=state_dim,
                    num_trajectories=args.num_trajectories,
                    trajectory_length=args.trajectory_length,
                    train_fraction=args.train_fraction,
                    train_count=train_trajectories.shape[0],
                    test_count=test_trajectories.shape[0],
                    ridge_lambda=args.ridge_lambda,
                    edmd_degree=args.edmd_degree,
                    kernel_centers=args.kernel_centers,
                    labels_initial=labels_initial,
                    labels_final=labels_final,
                )
                try:
                    fit_rng = np.random.default_rng(_stable_seed("fit", system, seed, method))
                    model = fit_baseline(
                        method,
                        train_trajectories,
                        edmd_degree=args.edmd_degree,
                        kernel_centers=args.kernel_centers,
                        kernel_gamma=args.kernel_gamma,
                        ridge_lambda=args.ridge_lambda,
                        max_train_pairs=args.max_train_pairs,
                        rng=fit_rng,
                    )
                    metrics_by_horizon = _evaluate_rollout(model, test_trajectories, horizons)
                    kernel_centers_used = ""
                    kernel_gamma_used = ""
                    if isinstance(model.feature_map, RBFDictionaryFeatureMap):
                        kernel_centers_used = (
                            int(model.feature_map.centers.shape[0])
                            if model.feature_map.centers is not None
                            else ""
                        )
                        kernel_gamma_used = model.feature_map.gamma_used

                    for horizon, metrics in metrics_by_horizon.items():
                        row = dict(base)
                        row.update(metrics)
                        row.update(
                            {
                                "status": "ok",
                                "skip_reason": "",
                                "horizon": int(horizon),
                                "feature_dim": model.feature_dim,
                                "train_transitions": model.train_transitions,
                                "kernel_centers_used": kernel_centers_used,
                                "kernel_gamma": kernel_gamma_used,
                            }
                        )
                        rows.append(row)
                except Exception as exc:
                    print(
                        f"[classical] ERROR system={system} seed={seed} method={method}: {exc}",
                        flush=True,
                    )
                    rows.extend(_error_rows(horizons=horizons, skip_reason=str(exc), **base))
                _write_csv(rows_path, rows)

    aggregates = _write_summary_json(summary_json_path, args, rows)
    _write_markdown_summary(summary_md_path, args, aggregates)
    ok_rows = sum(1 for row in rows if row.get("status") == "ok")
    error_rows = sum(1 for row in rows if row.get("status") != "ok")
    return rows_path, summary_json_path, summary_md_path, ok_rows, error_rows


def main() -> None:
    args = parse_args()
    rows_path, summary_json_path, summary_md_path, ok_rows, error_rows = run(args)
    print(f"[classical] wrote {rows_path}", flush=True)
    print(f"[classical] wrote {summary_json_path}", flush=True)
    print(f"[classical] wrote {summary_md_path}", flush=True)
    if error_rows:
        print(f"[classical] failing because {error_rows} error row(s) were emitted", flush=True)
        raise SystemExit(1)
    if ok_rows == 0:
        print("[classical] failing because no successful rows were emitted", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
