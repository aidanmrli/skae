"""Behavior-preserving local polynomial EDMD model port from commit c27490d."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans

from experiments.neurips_2026.baselines.classical import (
    EPS,
    PolynomialFeatureMap,
    StateScaler,
    _ridge_solve,
    _stable_seed,
)
from experiments.neurips_2026.local_edmd_reproduction.contract import (
    FEATURE_METHOD,
    METHOD_ID,
    ROUTE_SPACE,
)


@dataclass
class LocalEDMDModel:
    """A bank of local EDMD maps with state-space k-means routing."""

    scaler: StateScaler
    feature_map: PolynomialFeatureMap
    router: KMeans
    koopman_matrices: np.ndarray
    decoder_matrix: np.ndarray
    train_transitions: int
    component_counts: List[int]
    fitted_component_count: int
    selected_num_components: int
    method: str = METHOD_ID
    feature_method: str = FEATURE_METHOD
    route_space: str = ROUTE_SPACE

    @property
    def feature_dim(self) -> int:
        return int(self.koopman_matrices.shape[1])

    def predict_next(self, states: np.ndarray) -> np.ndarray:
        """Decode one step after routing each current predicted state."""

        states = np.asarray(states, dtype=np.float64)
        output = np.full_like(states, np.nan, dtype=np.float64)
        if states.size == 0:
            return output
        valid = np.isfinite(states).all(axis=1)
        if not bool(np.any(valid)):
            return output
        scaled = self.scaler.transform(states[valid])
        phi = self.feature_map.transform(scaled)
        route_valid = np.isfinite(scaled).all(axis=1) & np.isfinite(phi).all(axis=1)
        if not bool(np.any(route_valid)):
            return output
        valid_indices = np.flatnonzero(valid)[route_valid]
        phi_valid = phi[route_valid]
        labels = self.router.predict(scaled[route_valid])
        phi_next = np.full_like(phi_valid, np.nan, dtype=np.float64)
        for label in np.unique(labels):
            mask = labels == label
            with np.errstate(over="ignore", invalid="ignore"):
                phi_next[mask] = phi_valid[mask] @ self.koopman_matrices[int(label)]
        with np.errstate(over="ignore", invalid="ignore"):
            next_scaled = phi_next @ self.decoder_matrix
        output[valid_indices] = self.scaler.inverse_transform(next_scaled)
        return output

    def rollout(self, initial_states: np.ndarray, horizon: int) -> np.ndarray:
        """Reroute decoded predictions on every autonomous rollout step."""

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


def _finite_bounded_rows(states: np.ndarray, *, max_abs: float) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("Expected a two-dimensional state array")
    finite = np.isfinite(states).all(axis=1)
    return finite & (np.max(np.abs(states), axis=1) <= float(max_abs))


def _fit_scaler(
    trajectories: np.ndarray, *, max_abs_state_for_fit: float
) -> StateScaler:
    states = trajectories.reshape(-1, trajectories.shape[-1])
    fit_states = states[_finite_bounded_rows(states, max_abs=max_abs_state_for_fit)]
    if fit_states.size == 0:
        fit_states = states[np.isfinite(states).all(axis=1)]
    if fit_states.size == 0:
        fit_states = np.zeros((1, trajectories.shape[-1]), dtype=np.float64)
    return StateScaler.fit(fit_states)


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
    x, y = x[valid], y[valid]
    if max_train_pairs > 0 and x.shape[0] > int(max_train_pairs):
        indices = np.sort(
            rng.choice(x.shape[0], size=int(max_train_pairs), replace=False)
        )
        x, y = x[indices], y[indices]
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
    finite = np.isfinite(phi_x).all(axis=1) & np.isfinite(phi_y).all(axis=1)
    if not bool(np.any(finite)):
        raise ValueError("No finite feature transition pairs are available")
    phi_x, phi_y, labels = phi_x[finite], phi_y[finite], labels[finite]
    global_map = _ridge_solve(phi_x, phi_y, ridge_lambda)
    maps: List[np.ndarray] = []
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


def fit_local_edmd_model(
    trajectories: np.ndarray,
    *,
    num_components: int,
    edmd_degree: int,
    ridge_lambda: float,
    max_train_pairs: int,
    min_component_transitions: int,
    max_abs_state_for_fit: float,
    seed: int,
) -> LocalEDMDModel:
    """Fit one candidate with exactly the historical estimator semantics."""

    rng = np.random.default_rng(int(seed))
    scaler = _fit_scaler(
        trajectories, max_abs_state_for_fit=max_abs_state_for_fit
    )
    x_scaled, y_scaled = _transition_pairs(
        trajectories,
        scaler=scaler,
        max_train_pairs=max_train_pairs,
        max_abs_state_for_fit=max_abs_state_for_fit,
        rng=rng,
    )
    if x_scaled.shape[0] == 0:
        raise ValueError("No finite transition pairs are available after filtering")
    feature_map = PolynomialFeatureMap(edmd_degree, include_bias=True).fit(
        x_scaled, rng
    )
    phi_x = feature_map.transform(x_scaled)
    phi_y = feature_map.transform(y_scaled)
    route_valid = np.isfinite(x_scaled).all(axis=1) & np.isfinite(phi_x).all(axis=1)
    if not bool(np.any(route_valid)):
        raise ValueError("No finite routing features are available")
    route_features = x_scaled[route_valid]
    phi_x, phi_y = phi_x[route_valid], phi_y[route_valid]
    num_components = max(1, min(int(num_components), int(route_features.shape[0])))
    router = KMeans(n_clusters=num_components, n_init=10, random_state=int(seed))
    labels = router.fit_predict(route_features)
    matrices, counts, fitted = _fit_component_maps(
        phi_x,
        phi_y,
        labels,
        num_components=num_components,
        ridge_lambda=ridge_lambda,
        min_component_transitions=min_component_transitions,
    )
    decoder = _ridge_solve(phi_x, x_scaled[route_valid], ridge_lambda)
    return LocalEDMDModel(
        scaler=scaler,
        feature_map=feature_map,
        router=router,
        koopman_matrices=matrices,
        decoder_matrix=decoder,
        train_transitions=int(phi_x.shape[0]),
        component_counts=counts,
        fitted_component_count=int(fitted),
        selected_num_components=int(num_components),
    )


def split_fit_validation(
    trajectories: np.ndarray,
    *,
    validation_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split only the outer training trajectories for route-count selection."""

    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between zero and one")
    count = trajectories.shape[0]
    if count < 2:
        raise ValueError("At least two training trajectories are required")
    order = rng.permutation(count)
    validation_count = max(1, min(count - 1, int(round(count * validation_fraction))))
    validation_indices = np.sort(order[:validation_count])
    fit_indices = np.sort(order[validation_count:])
    return trajectories[fit_indices], trajectories[validation_indices]


def score_validation(
    metrics: Dict[int, Dict[str, Optional[float]]], horizons: Sequence[int]
) -> float:
    values: List[float] = []
    for horizon in horizons:
        value = metrics.get(int(horizon), {}).get("cumulative_mse_mean")
        if value is not None and math.isfinite(float(value)) and float(value) >= 0.0:
            values.append(max(float(value), EPS))
    return float(np.mean(np.log(values))) if values else float("inf")


def select_and_fit(
    trajectories: np.ndarray,
    *,
    num_components_grid: Sequence[int],
    validation_fraction: float,
    selection_horizons: Sequence[int],
    edmd_degree: int,
    ridge_lambda: float,
    max_train_pairs: int,
    min_component_transitions: int,
    max_abs_state_for_fit: float,
    seed: int,
    evaluator,
) -> Tuple[LocalEDMDModel, Dict[str, object]]:
    """Select k on held-out training trajectories, then refit all training data."""

    split_rng = np.random.default_rng(
        _stable_seed("local-edmd-validation", METHOD_ID, seed)
    )
    fit_trajectories, validation_trajectories = split_fit_validation(
        trajectories,
        validation_fraction=validation_fraction,
        rng=split_rng,
    )
    max_components = max(
        1,
        fit_trajectories[:, :-1, :].reshape(-1, fit_trajectories.shape[-1]).shape[0],
    )
    candidates: List[Dict[str, object]] = []
    for requested_k in num_components_grid:
        k = max(1, min(int(requested_k), int(max_components)))
        if any(int(row["num_components"]) == k for row in candidates):
            continue
        try:
            candidate = fit_local_edmd_model(
                fit_trajectories,
                num_components=k,
                edmd_degree=edmd_degree,
                ridge_lambda=ridge_lambda,
                max_train_pairs=max_train_pairs,
                min_component_transitions=min_component_transitions,
                max_abs_state_for_fit=max_abs_state_for_fit,
                seed=_stable_seed("local-edmd-candidate", METHOD_ID, seed, k),
            )
            metrics = evaluator(candidate, validation_trajectories, selection_horizons)
            score = score_validation(metrics, selection_horizons)
            candidates.append(
                {
                    "num_components": int(candidate.selected_num_components),
                    "score": float(score),
                    "status": "ok" if math.isfinite(score) else "nonfinite_score",
                    "fitted_component_count": int(candidate.fitted_component_count),
                    "component_counts": candidate.component_counts,
                }
            )
        except Exception as error:  # historical evaluator recorded candidate failures
            candidates.append(
                {
                    "num_components": int(k),
                    "score": float("inf"),
                    "status": "error",
                    "skip_reason": str(error),
                }
            )
    valid = [
        row
        for row in candidates
        if row.get("status") == "ok"
        and math.isfinite(float(row.get("score", float("inf"))))
    ]
    if not valid:
        raise ValueError("No valid local EDMD route-count candidates")
    selected = min(
        valid, key=lambda row: (float(row["score"]), int(row["num_components"]))
    )
    selected_k = int(selected["num_components"])
    model = fit_local_edmd_model(
        trajectories,
        num_components=selected_k,
        edmd_degree=edmd_degree,
        ridge_lambda=ridge_lambda,
        max_train_pairs=max_train_pairs,
        min_component_transitions=min_component_transitions,
        max_abs_state_for_fit=max_abs_state_for_fit,
        seed=_stable_seed("local-edmd-final", METHOD_ID, seed, selected_k),
    )
    return model, {
        "selected_num_components": model.selected_num_components,
        "validation_score": float(selected["score"]),
        "candidate_scores": candidates,
        "fit_trajectories": int(fit_trajectories.shape[0]),
        "validation_trajectories": int(validation_trajectories.shape[0]),
    }
