#!/usr/bin/env python3
"""Evaluate label-free local Koopman/EDMD baselines.

This baseline fits a bank of local EDMD operators. Local regions are selected
only from unlabeled training trajectories by k-means clustering; basin labels,
basin counts, and attractor identities are never used for fitting or route
count selection. The number of local operators is selected from a fixed grid by
validation rollout error, then the chosen model is refit on the full training
split and evaluated on held-out rollouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans

from tools.evaluate_classical_koopman_baselines import (
    DEFAULT_SYSTEMS,
    EPS,
    IdentityFeatureMap,
    PolynomialFeatureMap,
    RBFDictionaryFeatureMap,
    StateScaler,
    _atomic_write_text,
    _finite_mean,
    _finite_median,
    _format_optional_float,
    _generate_trajectories,
    _hist,
    _json_default,
    _make_env_for_system,
    _maybe_basin_labels,
    _parse_int_list,
    _parse_str_list,
    _ridge_solve,
    _split_trajectories,
    _stable_seed,
)


SUPPORTED_METHODS: Tuple[str, ...] = (
    "local_dmd_kmeans",
    "local_edmd_poly_kmeans",
    "local_rbf_edmd_kmeans",
)
DEFAULT_METHODS: Tuple[str, ...] = (
    "local_edmd_poly_kmeans",
    "local_rbf_edmd_kmeans",
)
MAX_ABS_STATE_FOR_FIT = 1e6


@dataclass(frozen=True)
class MethodSpec:
    feature_method: str
    route_space: str = "state"


METHOD_SPECS: Dict[str, MethodSpec] = {
    "local_dmd_kmeans": MethodSpec("dmd", "state"),
    "local_edmd_poly_kmeans": MethodSpec("edmd_poly", "state"),
    "local_rbf_edmd_kmeans": MethodSpec("rbf_dictionary_edmd", "state"),
}


@dataclass
class LocalEDMDModel:
    method: str
    feature_method: str
    route_space: str
    scaler: StateScaler
    feature_map: object
    router: KMeans
    koopman_matrices: np.ndarray
    decoder_matrix: np.ndarray
    train_transitions: int
    component_counts: List[int]
    fitted_component_count: int
    selected_num_components: int

    @property
    def feature_dim(self) -> int:
        return int(self.koopman_matrices.shape[1])

    def _route_features(self, scaled_states: np.ndarray, phi_states: np.ndarray) -> np.ndarray:
        if self.route_space == "state":
            return scaled_states
        if self.route_space == "feature":
            return phi_states
        raise ValueError(f"Unknown route_space '{self.route_space}'.")

    def predict_next(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        out = np.full_like(states, np.nan, dtype=np.float64)
        if states.size == 0:
            return out
        valid = np.isfinite(states).all(axis=1)
        if not bool(np.any(valid)):
            return out

        scaled = self.scaler.transform(states[valid])
        phi = self.feature_map.transform(scaled)
        route_features = self._route_features(scaled, phi)
        route_valid = np.isfinite(route_features).all(axis=1) & np.isfinite(phi).all(axis=1)
        if not bool(np.any(route_valid)):
            return out

        valid_indices = np.flatnonzero(valid)[route_valid]
        phi_valid = phi[route_valid]
        route_valid_features = route_features[route_valid]
        labels = self.router.predict(route_valid_features)

        phi_next = np.full_like(phi_valid, np.nan, dtype=np.float64)
        for label in np.unique(labels):
            mask = labels == label
            with np.errstate(over="ignore", invalid="ignore"):
                phi_next[mask] = phi_valid[mask] @ self.koopman_matrices[int(label)]
        with np.errstate(over="ignore", invalid="ignore"):
            next_scaled = phi_next @ self.decoder_matrix
        out[valid_indices] = self.scaler.inverse_transform(next_scaled)
        return out

    def rollout(self, initial_states: np.ndarray, horizon: int) -> np.ndarray:
        current = np.asarray(initial_states, dtype=np.float64).copy()
        predictions = np.full(
            (current.shape[0], int(horizon), current.shape[1]),
            np.nan,
            dtype=np.float64,
        )
        for step in range(int(horizon)):
            current = self.predict_next(current)
            predictions[:, step, :] = current
        return predictions


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
        "selected_num_components",
        "validation_score",
        "num_components_grid",
        "selection_horizons",
        "validation_fraction",
        "fitted_component_count",
        "component_counts",
        "feature_method",
        "route_space",
        "feature_dim",
        "train_transitions",
        "env_dt",
        "state_dim",
        "train_trajectories",
        "validation_trajectories",
        "test_trajectories",
        "num_trajectories",
        "trajectory_length",
        "train_fraction",
        "ridge_lambda",
        "edmd_degree",
        "kernel_centers_requested",
        "kernel_centers_used",
        "kernel_gamma",
        "min_component_transitions",
        "max_abs_state_for_fit",
        "test_initial_basin_hist",
        "test_final_basin_hist",
    ]
    seen: set[str] = set()
    fieldnames: List[str] = []
    for key in preferred:
        if any(key in row for row in rows):
            seen.add(key)
            fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, buffer.getvalue())


def _finite_bounded_rows(states: np.ndarray, *, max_abs: float) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("_finite_bounded_rows expects a 2D array.")
    finite = np.isfinite(states).all(axis=1)
    if states.size == 0:
        return finite
    return finite & (np.max(np.abs(states), axis=1) <= float(max_abs))


def _fit_scaler(train_trajectories: np.ndarray, *, max_abs_state_for_fit: float) -> StateScaler:
    states = train_trajectories.reshape(-1, train_trajectories.shape[-1])
    valid = _finite_bounded_rows(states, max_abs=max_abs_state_for_fit)
    fit_states = states[valid]
    if fit_states.size == 0:
        finite = np.isfinite(states).all(axis=1)
        fit_states = states[finite]
    if fit_states.size == 0:
        fit_states = np.zeros((1, train_trajectories.shape[-1]), dtype=np.float64)
    return StateScaler.fit(fit_states)


def _make_feature_map(
    feature_method: str,
    *,
    edmd_degree: int,
    kernel_centers: int,
    kernel_gamma: float,
) -> object:
    if feature_method == "dmd":
        return IdentityFeatureMap()
    if feature_method == "edmd_poly":
        return PolynomialFeatureMap(edmd_degree, include_bias=True)
    if feature_method == "rbf_dictionary_edmd":
        return RBFDictionaryFeatureMap(
            kernel_centers,
            gamma=kernel_gamma,
            include_bias=True,
            include_linear=True,
        )
    raise ValueError(f"Unknown feature_method '{feature_method}'.")


def _transition_pairs(
    trajectories: np.ndarray,
    *,
    scaler: StateScaler,
    max_train_pairs: int,
    max_abs_state_for_fit: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    x = trajectories[:, :-1, :].reshape(-1, trajectories.shape[-1])
    y = trajectories[:, 1:, :].reshape(-1, trajectories.shape[-1])
    valid = _finite_bounded_rows(x, max_abs=max_abs_state_for_fit)
    valid &= _finite_bounded_rows(y, max_abs=max_abs_state_for_fit)
    x = x[valid]
    y = y[valid]
    if max_train_pairs > 0 and x.shape[0] > int(max_train_pairs):
        idx = np.sort(rng.choice(x.shape[0], size=int(max_train_pairs), replace=False))
        x = x[idx]
        y = y[idx]
    return scaler.transform(x), scaler.transform(y)


def _fit_component_maps(
    phi_x: np.ndarray,
    phi_y: np.ndarray,
    labels: np.ndarray,
    *,
    num_components: int,
    ridge_lambda: float,
    min_component_transitions: int,
) -> Tuple[np.ndarray, List[int], int]:
    if phi_x.shape[0] == 0:
        raise ValueError("No finite transition pairs are available for local EDMD fitting.")
    finite = np.isfinite(phi_x).all(axis=1) & np.isfinite(phi_y).all(axis=1)
    if not bool(np.any(finite)):
        raise ValueError("No finite feature transition pairs are available for local EDMD fitting.")
    phi_x = phi_x[finite]
    phi_y = phi_y[finite]
    labels = labels[finite]

    global_map = _ridge_solve(phi_x, phi_y, ridge_lambda)
    maps = []
    counts: List[int] = []
    fitted = 0
    for component in range(int(num_components)):
        mask = labels == component
        count = int(np.sum(mask))
        counts.append(count)
        if count >= int(min_component_transitions):
            maps.append(_ridge_solve(phi_x[mask], phi_y[mask], ridge_lambda))
            fitted += 1
        else:
            maps.append(global_map)
    return np.stack(maps, axis=0), counts, fitted


def _fit_local_edmd_model(
    method: str,
    train_trajectories: np.ndarray,
    *,
    num_components: int,
    edmd_degree: int,
    kernel_centers: int,
    kernel_gamma: float,
    ridge_lambda: float,
    max_train_pairs: int,
    min_component_transitions: int,
    max_abs_state_for_fit: float,
    seed: int,
) -> LocalEDMDModel:
    if method not in METHOD_SPECS:
        raise ValueError(f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}.")
    if int(num_components) < 1:
        raise ValueError("num_components must be positive.")
    spec = METHOD_SPECS[method]
    rng = np.random.default_rng(int(seed))
    scaler = _fit_scaler(train_trajectories, max_abs_state_for_fit=max_abs_state_for_fit)
    x_scaled, y_scaled = _transition_pairs(
        train_trajectories,
        scaler=scaler,
        max_train_pairs=int(max_train_pairs),
        max_abs_state_for_fit=float(max_abs_state_for_fit),
        rng=rng,
    )
    if x_scaled.shape[0] == 0:
        raise ValueError("No finite transition pairs are available after filtering.")

    feature_map = _make_feature_map(
        spec.feature_method,
        edmd_degree=edmd_degree,
        kernel_centers=kernel_centers,
        kernel_gamma=kernel_gamma,
    ).fit(x_scaled, rng)
    phi_x = feature_map.transform(x_scaled)
    phi_y = feature_map.transform(y_scaled)
    route_features = x_scaled if spec.route_space == "state" else phi_x
    route_valid = np.isfinite(route_features).all(axis=1) & np.isfinite(phi_x).all(axis=1)
    if not bool(np.any(route_valid)):
        raise ValueError("No finite routing features are available.")
    route_features = route_features[route_valid]
    phi_x = phi_x[route_valid]
    phi_y = phi_y[route_valid]
    num_components = max(1, min(int(num_components), int(route_features.shape[0])))

    router = KMeans(n_clusters=num_components, n_init=10, random_state=int(seed))
    labels = router.fit_predict(route_features)
    koopman_matrices, counts, fitted_count = _fit_component_maps(
        phi_x,
        phi_y,
        labels,
        num_components=num_components,
        ridge_lambda=ridge_lambda,
        min_component_transitions=min_component_transitions,
    )
    decoder_matrix = _ridge_solve(phi_x, x_scaled[route_valid], ridge_lambda)

    return LocalEDMDModel(
        method=method,
        feature_method=spec.feature_method,
        route_space=spec.route_space,
        scaler=scaler,
        feature_map=feature_map,
        router=router,
        koopman_matrices=koopman_matrices,
        decoder_matrix=decoder_matrix,
        train_transitions=int(phi_x.shape[0]),
        component_counts=counts,
        fitted_component_count=int(fitted_count),
        selected_num_components=int(num_components),
    )


def _evaluate_rollout(
    model: LocalEDMDModel,
    trajectories: np.ndarray,
    horizons: Sequence[int],
) -> Dict[int, Dict[str, Optional[float]]]:
    horizon_list = sorted({int(h) for h in horizons if int(h) > 0})
    if not horizon_list:
        raise ValueError("At least one positive horizon is required.")
    max_available = trajectories.shape[1] - 1
    horizon_list = [h for h in horizon_list if h <= max_available]
    if not horizon_list:
        raise ValueError(f"No requested horizons fit trajectory_length={max_available}.")
    max_horizon = max(horizon_list)
    predictions = model.rollout(trajectories[:, 0, :], horizon=max_horizon)
    targets = trajectories[:, 1 : max_horizon + 1, :]
    with np.errstate(over="ignore", invalid="ignore"):
        squared_error = (predictions - targets) ** 2
        step_squared_error = squared_error.sum(axis=2)
    state_dim = trajectories.shape[-1]

    metrics: Dict[int, Dict[str, Optional[float]]] = {}
    for horizon in horizon_list:
        endpoint = step_squared_error[:, horizon - 1]
        cumulative = step_squared_error[:, :horizon].mean(axis=1)
        finite = np.isfinite(cumulative)
        metrics[horizon] = {
            "endpoint_mse_mean": _finite_mean(endpoint),
            "endpoint_mse_median": _finite_median(endpoint),
            "endpoint_mse_per_dim_mean": _finite_mean(endpoint / state_dim),
            "cumulative_mse_mean": _finite_mean(cumulative),
            "cumulative_mse_median": _finite_median(cumulative),
            "cumulative_mse_per_dim_mean": _finite_mean(cumulative / state_dim),
            "finite_fraction": float(finite.mean()) if finite.size else None,
        }
    return metrics


def _split_fit_validation(
    train_trajectories: np.ndarray,
    *,
    validation_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    count = train_trajectories.shape[0]
    if count < 2:
        raise ValueError("At least two training trajectories are required for validation selection.")
    order = rng.permutation(count)
    validation_count = int(round(count * float(validation_fraction)))
    validation_count = max(1, min(count - 1, validation_count))
    validation_idx = np.sort(order[:validation_count])
    fit_idx = np.sort(order[validation_count:])
    return train_trajectories[fit_idx], train_trajectories[validation_idx]


def _score_validation(metrics: Dict[int, Dict[str, Optional[float]]], horizons: Sequence[int]) -> float:
    values: List[float] = []
    for horizon in horizons:
        row = metrics.get(int(horizon))
        if not row:
            continue
        value = _safe_float(row.get("cumulative_mse_mean"))
        if value is not None and value >= 0.0:
            values.append(max(value, EPS))
    if not values:
        return float("inf")
    return float(np.mean(np.log(values)))


def _parse_grid(raw: str) -> List[int]:
    grid = sorted({int(item) for item in _parse_int_list(raw) if int(item) > 0})
    if not grid:
        raise ValueError("num_components_grid must contain at least one positive integer.")
    return grid


def _select_and_fit(
    method: str,
    train_trajectories: np.ndarray,
    *,
    num_components_grid: Sequence[int],
    validation_fraction: float,
    selection_horizons: Sequence[int],
    edmd_degree: int,
    kernel_centers: int,
    kernel_gamma: float,
    ridge_lambda: float,
    max_train_pairs: int,
    min_component_transitions: int,
    max_abs_state_for_fit: float,
    seed: int,
) -> Tuple[LocalEDMDModel, Dict[str, object]]:
    split_rng = np.random.default_rng(_stable_seed("local-edmd-validation", method, seed))
    fit_trajectories, validation_trajectories = _split_fit_validation(
        train_trajectories,
        validation_fraction=validation_fraction,
        rng=split_rng,
    )
    max_components = max(1, fit_trajectories[:, :-1, :].reshape(-1, fit_trajectories.shape[-1]).shape[0])
    candidates: List[Dict[str, object]] = []
    for requested_k in num_components_grid:
        k = max(1, min(int(requested_k), int(max_components)))
        if any(int(row["num_components"]) == k for row in candidates):
            continue
        try:
            candidate = _fit_local_edmd_model(
                method,
                fit_trajectories,
                num_components=k,
                edmd_degree=edmd_degree,
                kernel_centers=kernel_centers,
                kernel_gamma=kernel_gamma,
                ridge_lambda=ridge_lambda,
                max_train_pairs=max_train_pairs,
                min_component_transitions=min_component_transitions,
                max_abs_state_for_fit=max_abs_state_for_fit,
                seed=_stable_seed("local-edmd-candidate", method, seed, k),
            )
            metrics = _evaluate_rollout(candidate, validation_trajectories, selection_horizons)
            score = _score_validation(metrics, selection_horizons)
            candidates.append(
                {
                    "num_components": int(candidate.selected_num_components),
                    "score": float(score),
                    "status": "ok" if math.isfinite(score) else "nonfinite_score",
                    "fitted_component_count": int(candidate.fitted_component_count),
                    "component_counts": candidate.component_counts,
                }
            )
        except Exception as exc:
            candidates.append(
                {
                    "num_components": int(k),
                    "score": float("inf"),
                    "status": "error",
                    "skip_reason": str(exc),
                }
            )

    valid_candidates = [
        row
        for row in candidates
        if row.get("status") == "ok" and math.isfinite(float(row.get("score", float("inf"))))
    ]
    if not valid_candidates:
        raise ValueError(f"No valid local EDMD route-count candidates for method={method}.")
    selected = min(valid_candidates, key=lambda row: (float(row["score"]), int(row["num_components"])))
    selected_k = int(selected["num_components"])
    model = _fit_local_edmd_model(
        method,
        train_trajectories,
        num_components=selected_k,
        edmd_degree=edmd_degree,
        kernel_centers=kernel_centers,
        kernel_gamma=kernel_gamma,
        ridge_lambda=ridge_lambda,
        max_train_pairs=max_train_pairs,
        min_component_transitions=min_component_transitions,
        max_abs_state_for_fit=max_abs_state_for_fit,
        seed=_stable_seed("local-edmd-final", method, seed, selected_k),
    )
    selection = {
        "selected_num_components": int(model.selected_num_components),
        "validation_score": float(selected["score"]),
        "candidate_scores": candidates,
        "fit_trajectories": int(fit_trajectories.shape[0]),
        "validation_trajectories": int(validation_trajectories.shape[0]),
    }
    return model, selection


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
    validation_count: int,
    test_count: int,
    ridge_lambda: float,
    edmd_degree: int,
    kernel_centers: int,
    num_components_grid: Sequence[int],
    selection_horizons: Sequence[int],
    validation_fraction: float,
    min_component_transitions: int,
    max_abs_state_for_fit: float,
    labels_initial: Optional[np.ndarray],
    labels_final: Optional[np.ndarray],
) -> Dict[str, object]:
    spec = METHOD_SPECS.get(method)
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
        "validation_trajectories": int(validation_count),
        "test_trajectories": int(test_count),
        "ridge_lambda": float(ridge_lambda),
        "edmd_degree": int(edmd_degree),
        "kernel_centers_requested": int(kernel_centers),
        "num_components_grid": ",".join(str(item) for item in num_components_grid),
        "selection_horizons": ",".join(str(item) for item in selection_horizons),
        "validation_fraction": float(validation_fraction),
        "min_component_transitions": int(min_component_transitions),
        "max_abs_state_for_fit": float(max_abs_state_for_fit),
        "feature_method": spec.feature_method if spec else "",
        "route_space": spec.route_space if spec else "",
        "test_initial_basin_hist": _hist(labels_initial),
        "test_final_basin_hist": _hist(labels_final),
    }


def _error_rows(
    *,
    horizons: Sequence[int],
    skip_reason: str,
    **base: object,
) -> List[Dict[str, object]]:
    return [
        {**base, "status": "error", "skip_reason": skip_reason, "horizon": int(horizon)}
        for horizon in horizons
    ]


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
            value = _safe_float(row.get(column))
            if value is not None:
                group[target].append(value)

    aggregates = []
    for group in sorted(grouped.values(), key=lambda item: (item["system"], item["horizon"], item["method"])):
        endpoint = np.asarray(group.pop("endpoint_values"), dtype=np.float64)
        cumulative = np.asarray(group.pop("cumulative_values"), dtype=np.float64)
        group["endpoint_mse_mean_across_runs"] = _finite_mean(endpoint)
        group["endpoint_mse_median_across_runs"] = _finite_median(endpoint)
        group["cumulative_mse_mean_across_runs"] = _finite_mean(cumulative)
        group["cumulative_mse_median_across_runs"] = _finite_median(cumulative)
        aggregates.append(group)
    return aggregates


def _write_summary_json(
    path: Path,
    args: argparse.Namespace,
    rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    aggregates = _aggregate_rows(rows)
    payload = {
        "config": {
            "systems": _parse_str_list(args.systems),
            "seeds": _parse_int_list(args.seeds),
            "methods": _parse_str_list(args.methods),
            "horizons": _parse_int_list(args.horizons),
            "selection_horizons": _parse_int_list(args.selection_horizons)
            if args.selection_horizons
            else _parse_int_list(args.horizons),
            "num_components_grid": _parse_grid(args.num_components_grid),
            "validation_fraction": args.validation_fraction,
            "num_trajectories": args.num_trajectories,
            "trajectory_length": args.trajectory_length,
            "train_fraction": args.train_fraction,
            "edmd_degree": args.edmd_degree,
            "kernel_centers": args.kernel_centers,
            "kernel_gamma": args.kernel_gamma,
            "ridge_lambda": args.ridge_lambda,
            "max_train_pairs": args.max_train_pairs,
            "min_component_transitions": args.min_component_transitions,
            "max_abs_state_for_fit": args.max_abs_state_for_fit,
            "config_name": args.config_name,
            "env_dt": args.env_dt,
            "dysts_dt_multiplier": args.dysts_dt_multiplier,
            "dysts_standardize": int(args.dysts_standardize),
            "rollout_update": "decode_and_reencode_predicted_state_each_step",
            "label_policy": "label_free_primary; basin labels/counts diagnostics only",
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
        "# Label-Free Local Koopman/EDMD Baseline Summary",
        "",
        f"- Systems: `{', '.join(_parse_str_list(args.systems))}`",
        f"- Methods: `{', '.join(_parse_str_list(args.methods))}`",
        f"- Seeds: `{', '.join(str(seed) for seed in _parse_int_list(args.seeds))}`",
        f"- Route-count grid: `{args.num_components_grid}`",
        f"- Selection horizons: `{args.selection_horizons or args.horizons}`",
        f"- Validation fraction: {args.validation_fraction:.3g}",
        "- Label policy: labels and basin counts are not used for fitting or selection.",
        "- Rollout update: decode prediction, then recompute deterministic EDMD features for the next routed step.",
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
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--horizons", default="100,500,1000")
    parser.add_argument(
        "--selection_horizons",
        default="",
        help="Comma-separated validation horizons; empty uses --horizons.",
    )
    parser.add_argument("--num_components_grid", default="1,2,4,8,16")
    parser.add_argument("--validation_fraction", type=float, default=0.25)
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
    parser.add_argument("--max_train_pairs", type=int, default=0)
    parser.add_argument("--min_component_transitions", type=int, default=64)
    parser.add_argument("--max_abs_state_for_fit", type=float, default=MAX_ABS_STATE_FOR_FIT)
    parser.add_argument("--output_dir", default="runs/local_edmd_koopman_baselines")
    parser.add_argument("--env_dt", type=float, default=0.0)
    parser.add_argument("--dysts_dt_multiplier", type=float, default=0.0)
    parser.add_argument("--dysts_standardize", type=int, default=0)
    parser.add_argument("--config_name", default="default")
    parser.add_argument("--allow_non_2d", action="store_true")
    parser.add_argument("--torch_threads", type=int, default=1)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> Tuple[List[str], List[int], List[str], List[int], List[int], List[int]]:
    systems = _parse_str_list(args.systems)
    seeds = _parse_int_list(args.seeds)
    methods = [method.lower() for method in _parse_str_list(args.methods)]
    horizons = sorted(set(_parse_int_list(args.horizons)))
    grid = _parse_grid(args.num_components_grid)
    selection_horizons = (
        sorted(set(_parse_int_list(args.selection_horizons)))
        if args.selection_horizons
        else list(horizons)
    )
    if not systems:
        raise ValueError("At least one system is required.")
    if not seeds:
        raise ValueError("At least one seed is required.")
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Supported: {SUPPORTED_METHODS}")
    if args.num_trajectories < 3:
        raise ValueError("num_trajectories must be at least 3 for train/validation/test splitting.")
    if args.trajectory_length < 1:
        raise ValueError("trajectory_length must be at least 1.")
    horizons = [horizon for horizon in horizons if 1 <= horizon <= args.trajectory_length]
    selection_horizons = [
        horizon for horizon in selection_horizons if 1 <= horizon <= args.trajectory_length
    ]
    if not horizons:
        raise ValueError("No valid horizons remain after clipping to trajectory_length.")
    if not selection_horizons:
        raise ValueError("No valid selection horizons remain after clipping to trajectory_length.")
    if args.ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be nonnegative.")
    if args.min_component_transitions < 1:
        raise ValueError("min_component_transitions must be positive.")
    if args.max_abs_state_for_fit <= 0.0:
        raise ValueError("max_abs_state_for_fit must be positive.")
    if not 0.0 < float(args.train_fraction) < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    if not 0.0 < float(args.validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    return systems, seeds, methods, horizons, selection_horizons, grid


def run(args: argparse.Namespace) -> Tuple[Path, Path, Path, int, int]:
    systems, seeds, methods, horizons, selection_horizons, grid = _validate_args(args)
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
            print(f"[local-edmd] generating system={system} seed={seed}", flush=True)
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
                        f"System '{system}' has state_dim={state_dim}; pass --allow_non_2d to override."
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
                split_rng = np.random.default_rng(_stable_seed("local-edmd-counts", system, seed))
                _, validation_trajectories = _split_fit_validation(
                    train_trajectories,
                    validation_fraction=float(args.validation_fraction),
                    rng=split_rng,
                )
                labels_initial = _maybe_basin_labels(env, test_trajectories[:, 0, :])
                labels_final = _maybe_basin_labels(env, test_trajectories[:, -1, :])
            except Exception as exc:
                print(f"[local-edmd] ERROR system={system} seed={seed}: {exc}", flush=True)
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
                        validation_count=0,
                        test_count=0,
                        ridge_lambda=args.ridge_lambda,
                        edmd_degree=args.edmd_degree,
                        kernel_centers=args.kernel_centers,
                        num_components_grid=grid,
                        selection_horizons=selection_horizons,
                        validation_fraction=args.validation_fraction,
                        min_component_transitions=args.min_component_transitions,
                        max_abs_state_for_fit=args.max_abs_state_for_fit,
                        labels_initial=None,
                        labels_final=None,
                    )
                    rows.extend(_error_rows(horizons=horizons, skip_reason=str(exc), **base))
                _write_csv(rows_path, rows)
                continue

            for method in methods:
                print(f"[local-edmd] fitting system={system} seed={seed} method={method}", flush=True)
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
                    validation_count=validation_trajectories.shape[0],
                    test_count=test_trajectories.shape[0],
                    ridge_lambda=args.ridge_lambda,
                    edmd_degree=args.edmd_degree,
                    kernel_centers=args.kernel_centers,
                    num_components_grid=grid,
                    selection_horizons=selection_horizons,
                    validation_fraction=args.validation_fraction,
                    min_component_transitions=args.min_component_transitions,
                    max_abs_state_for_fit=args.max_abs_state_for_fit,
                    labels_initial=labels_initial,
                    labels_final=labels_final,
                )
                try:
                    model, selection = _select_and_fit(
                        method,
                        train_trajectories,
                        num_components_grid=grid,
                        validation_fraction=float(args.validation_fraction),
                        selection_horizons=selection_horizons,
                        edmd_degree=args.edmd_degree,
                        kernel_centers=args.kernel_centers,
                        kernel_gamma=args.kernel_gamma,
                        ridge_lambda=args.ridge_lambda,
                        max_train_pairs=args.max_train_pairs,
                        min_component_transitions=args.min_component_transitions,
                        max_abs_state_for_fit=args.max_abs_state_for_fit,
                        seed=_stable_seed("select", system, seed, method),
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
                                "selected_num_components": model.selected_num_components,
                                "validation_score": selection["validation_score"],
                                "candidate_scores_json": json.dumps(
                                    selection["candidate_scores"], default=_json_default
                                ),
                                "fitted_component_count": model.fitted_component_count,
                                "component_counts": json.dumps(model.component_counts),
                                "kernel_centers_used": kernel_centers_used,
                                "kernel_gamma": kernel_gamma_used,
                            }
                        )
                        rows.append(row)
                except Exception as exc:
                    print(
                        f"[local-edmd] ERROR system={system} seed={seed} method={method}: {exc}",
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
    print(f"[local-edmd] wrote {rows_path}", flush=True)
    print(f"[local-edmd] wrote {summary_json_path}", flush=True)
    print(f"[local-edmd] wrote {summary_md_path}", flush=True)
    if error_rows:
        print(f"[local-edmd] failing because {error_rows} error row(s) were emitted", flush=True)
        raise SystemExit(1)
    if ok_rows == 0:
        print("[local-edmd] failing because no successful rows were emitted", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
