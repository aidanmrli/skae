"""Frozen numerical estimands for the distinct-law V2 evaluator."""

from __future__ import annotations

import math
import itertools
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


EPS = 1e-12


def sample_centered_disk(count: int, radius: float, seed: int) -> np.ndarray:
    """Draw deterministic area-uniform offsets inside a two-dimensional disk."""
    if count < 3 or radius <= 0:
        raise ValueError((count, radius))
    rng = np.random.default_rng(seed)
    radial = radius * np.sqrt(np.maximum(rng.random(count), np.finfo(float).eps))
    angle = 2.0 * np.pi * rng.random(count)
    return np.column_stack((radial * np.cos(angle), radial * np.sin(angle))).astype(
        np.float32
    )


def rk4_step_matrix(matrix: np.ndarray, dt: float) -> np.ndarray:
    """Classical fourth-order Runge--Kutta polynomial for a linear system."""
    matrix = np.asarray(matrix, dtype=np.float64)
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    dt_matrix = float(dt) * matrix
    return (
        identity + dt_matrix + dt_matrix @ dt_matrix / 2.0
        + dt_matrix @ dt_matrix @ dt_matrix / 6.0
        + dt_matrix @ dt_matrix @ dt_matrix @ dt_matrix / 24.0
    )


def authenticate_local_geometry(
    env, points_by_basin: list[np.ndarray] | np.ndarray, *, category: str,
    dt: float, max_abs_error: float,
) -> dict[str, Any]:
    """Authenticate basin membership and the analytic local RK4 map."""
    centers = env.unwrapped.points_2d.detach().cpu().numpy().astype(np.float64)
    matrices = env.unwrapped.basin_matrices.detach().cpu().numpy()
    steps = np.stack([rk4_step_matrix(matrix, dt) for matrix in matrices])
    if len(points_by_basin) != centers.shape[0]:
        raise ValueError(f"{category}: expected {centers.shape[0]} basin rows")
    rows, total_count, total_region_matches = [], 0, 0
    maxima, all_finite = [], True
    for basin, raw_points in enumerate(points_by_basin):
        points = np.asarray(raw_points, dtype=np.float32).reshape(-1, centers.shape[1])
        if points.shape[0] == 0:
            raise ValueError(f"{category}: empty basin {basin} point set")
        tensor = torch.from_numpy(points)
        with torch.no_grad():
            labels = env.unwrapped.region_label(tensor).detach().cpu().numpy()
            observed = env.step(tensor).detach().cpu().numpy().astype(np.float64)
        expected = centers[basin] + (
            points.astype(np.float64) - centers[basin]
        ) @ steps[basin].T
        finite = bool(
            np.isfinite(points).all() and np.isfinite(observed).all()
            and np.isfinite(expected).all()
        )
        error = (
            float(np.max(np.abs(observed - expected))) if finite else None
        )
        region_matches = int(np.sum(labels == basin))
        count = int(points.shape[0])
        rows.append({
            "basin": basin, "point_count": count,
            "region_match_count": region_matches,
            "all_points_in_intended_region": region_matches == count,
            "analytic_rk4_max_abs_error": error,
            "finite": finite,
        })
        total_count += count
        total_region_matches += region_matches
        all_finite = all_finite and finite
        if error is not None:
            maxima.append(error)
    maximum = max(maxima) if maxima else None
    passed = bool(
        all_finite and total_region_matches == total_count
        and maximum is not None and maximum <= float(max_abs_error)
    )
    return {
        "category": category,
        "point_count_by_basin": [row["point_count"] for row in rows],
        "total_point_count": total_count,
        "region_match_count": total_region_matches,
        "all_points_in_intended_region": total_region_matches == total_count,
        "analytic_rk4_max_abs_error": maximum,
        "maximum_allowed_abs_error": float(max_abs_error),
        "finite": all_finite,
        "passed": passed,
        "rows": rows,
    }


def match_families_to_basins(
    assignments: np.ndarray, retained: np.ndarray, basin_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluation-only one-to-one matching of label-free sparse families."""
    assignments = np.asarray(assignments, dtype=np.int64)
    retained_ids = np.flatnonzero(np.asarray(retained, dtype=bool))
    if assignments.ndim != 2 or assignments.shape[0] != basin_count:
        raise ValueError(assignments.shape)
    if retained_ids.size < basin_count:
        return (
            np.full(basin_count, -1, dtype=np.int64),
            np.zeros(basin_count, dtype=np.float64),
            np.zeros((basin_count, retained_ids.size), dtype=np.int64),
        )
    counts = np.asarray(
        [[np.sum(assignments[basin] == family) for family in retained_ids]
         for basin in range(basin_count)],
        dtype=np.int64,
    )
    rows, columns = linear_sum_assignment(-counts)
    mapping = np.full(basin_count, -1, dtype=np.int64)
    rates = np.zeros(basin_count, dtype=np.float64)
    for basin, column in zip(rows, columns):
        mapping[basin] = int(retained_ids[column])
        rates[basin] = float(counts[basin, column] / assignments.shape[1])
    return mapping, rates, counts


def law_cost_summary(
    predicted: np.ndarray, true_matrices: np.ndarray,
) -> dict[str, Any]:
    """Score all three-law assignments without changing a predicted chart."""
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(true_matrices, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.shape[0] != 3:
        raise ValueError((predicted.shape, truth.shape))
    costs = np.asarray(
        [[np.linalg.norm(predicted[basin] - truth[law], ord="fro")
          for law in range(3)] for basin in range(3)],
        dtype=np.float64,
    )
    rows, columns = linear_sum_assignment(costs)
    assignment = np.full(3, -1, dtype=np.int64)
    assignment[rows] = columns
    assignment_costs = {
        "".join(str(value) for value in permutation): float(
            sum(costs[basin, permutation[basin]] for basin in range(3))
        )
        for permutation in itertools.permutations(range(3))
    }
    identity_cost = assignment_costs["012"]
    best_nonidentity = min(
        value for key, value in assignment_costs.items() if key != "012"
    )
    row_ratios, own_relative = [], []
    for basin in range(3):
        nearest_wrong = min(costs[basin, law] for law in range(3) if law != basin)
        row_ratios.append(float(costs[basin, basin] / max(nearest_wrong, EPS)))
        own_relative.append(float(
            costs[basin, basin]
            / max(np.linalg.norm(truth[basin], ord="fro"), EPS)
        ))
    return {
        "cost_matrix": costs.tolist(),
        "optimal_assignment": assignment.tolist(),
        "identity_is_unique_optimum": bool(
            np.array_equal(assignment, np.arange(3))
            and identity_cost + 1e-12 < best_nonidentity
        ),
        "assignment_costs": assignment_costs,
        "identity_over_best_nonidentity": float(
            identity_cost / max(best_nonidentity, EPS)
        ),
        "own_over_nearest_wrong_by_basin": row_ratios,
        "own_relative_error_by_basin": own_relative,
        "max_own_over_nearest_wrong": float(max(row_ratios)),
        "max_own_relative_error": float(max(own_relative)),
    }


def model_estimand(
    model,
    x: torch.Tensor,
    mask: np.ndarray | torch.Tensor | None,
    estimand: str,
    permutation: np.ndarray | None = None,
) -> torch.Tensor:
    """Apply one unchanged K and return a frozen H/G physical estimand."""
    z = model.encode(x)
    if permutation is not None:
        index = torch.as_tensor(permutation, dtype=torch.long, device=z.device)
        z = z.index_select(-1, index)
    if estimand == "g_global":
        return model.decode(z @ model.kmatrix()) - model.decode(z)
    if estimand == "h_global":
        return model.decode(z @ model.kmatrix()) - x
    if mask is None:
        raise ValueError("A support mask is required for block/source modes")
    p = torch.as_tensor(mask, dtype=z.dtype, device=z.device)
    if permutation is not None:
        index = torch.as_tensor(permutation, dtype=torch.long, device=z.device)
        p = p.index_select(-1, index)
    source = z * p
    stepped = source @ model.kmatrix()
    if estimand in {"g_block", "h_block"}:
        stepped = stepped * p
    elif estimand != "g_source":
        raise ValueError(estimand)
    if estimand == "h_block":
        return model.decode(stepped) - x
    return model.decode(stepped) - model.decode(source)


def autograd_jacobian(
    model,
    center: np.ndarray,
    mask: np.ndarray | None,
    estimand: str,
    permutation: np.ndarray | None = None,
) -> np.ndarray:
    parameter = next(model.parameters())
    x = torch.as_tensor(center, dtype=parameter.dtype, device=parameter.device)
    x = x.detach().clone().requires_grad_(True)

    def function(value: torch.Tensor) -> torch.Tensor:
        return model_estimand(model, value, mask, estimand, permutation)

    jacobian = torch.autograd.functional.jacobian(
        function, x, create_graph=False, strict=True, vectorize=False
    )
    return jacobian.detach().cpu().numpy().astype(np.float64)


def central_difference_jacobian(
    model,
    center: np.ndarray,
    mask: np.ndarray,
    estimand: str,
    epsilon: float,
) -> np.ndarray:
    parameter = next(model.parameters())
    center_tensor = torch.as_tensor(
        center, dtype=parameter.dtype, device=parameter.device
    )
    columns = []
    with torch.no_grad():
        for coordinate in range(center_tensor.numel()):
            offset = torch.zeros_like(center_tensor)
            offset[coordinate] = float(epsilon)
            plus = model_estimand(model, center_tensor + offset, mask, estimand)
            minus = model_estimand(model, center_tensor - offset, mask, estimand)
            columns.append((plus - minus) / (2.0 * float(epsilon)))
    return torch.stack(columns, dim=1).cpu().numpy().astype(np.float64)


def evaluate_update_points(
    model,
    points: np.ndarray,
    mask: np.ndarray | None,
    estimand: str,
    batch_size: int = 4096,
) -> np.ndarray:
    parameter = next(model.parameters())
    x = torch.as_tensor(points, dtype=parameter.dtype)
    chunks = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            chunks.append(
                model_estimand(
                    model, x[start : start + batch_size].to(parameter.device), mask, estimand
                ).cpu()
            )
    return torch.cat(chunks).numpy().astype(np.float64)


def decoded_state(model, latent: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return model.decode(latent).detach().cpu().numpy().astype(np.float64)


def decoder_linearity_diagnostics(
    model, latent_dim: int,
) -> dict[str, float | bool]:
    """Probe zero preservation, additivity, and homogeneity off-axis."""
    if latent_dim < 4:
        raise ValueError("Decoder linearity probe requires at least four coordinates")
    parameter = next(model.parameters())
    left = torch.zeros(latent_dim, dtype=parameter.dtype, device=parameter.device)
    right = torch.zeros_like(left)
    left[:4] = torch.tensor(
        [0.25, -0.5, 0.75, 0.125], dtype=left.dtype, device=left.device
    )
    right[:4] = torch.tensor(
        [-0.6, 0.4, 0.2, -0.3], dtype=right.dtype, device=right.device
    )
    scale = 1.75
    with torch.no_grad():
        zero_error = torch.max(torch.abs(model.decode(torch.zeros_like(left)))).item()
        additive_error = torch.max(torch.abs(
            model.decode(left + right) - model.decode(left) - model.decode(right)
        )).item()
        homogeneous_error = torch.max(torch.abs(
            model.decode(scale * left) - scale * model.decode(left)
        )).item()
    threshold = 1e-6
    return {
        "zero_error": float(zero_error),
        "additive_error": float(additive_error),
        "homogeneous_error": float(homogeneous_error),
        "zero_preserving": zero_error <= threshold,
        "additive": additive_error <= threshold,
        "homogeneous": homogeneous_error <= threshold,
        "linear": max(zero_error, additive_error, homogeneous_error) <= threshold,
    }


def center_forecast_metrics(
    model,
    center: np.ndarray,
    mask: np.ndarray,
    true_update_rms: float,
) -> dict[str, float]:
    parameter = next(model.parameters())
    x = torch.as_tensor(center, dtype=parameter.dtype, device=parameter.device)
    p = torch.as_tensor(mask, dtype=parameter.dtype, device=parameter.device)
    with torch.no_grad():
        z = model.encode(x)
        source = z * p
        support_reconstruction = model.decode(source)
        restricted_forecast = model.decode((source @ model.kmatrix()) * p)
        full_reconstruction = model.decode(z)
        full_forecast = model.decode(z @ model.kmatrix())
        k_update = restricted_forecast - support_reconstruction
    scale = max(float(true_update_rms), EPS)
    return {
        "support_reconstruction": float(
            torch.linalg.vector_norm(support_reconstruction - x).item() / scale
        ),
        "k_induced_update": float(torch.linalg.vector_norm(k_update).item() / scale),
        "restricted_forecast": float(
            torch.linalg.vector_norm(restricted_forecast - x).item() / scale
        ),
        "full_reconstruction": float(
            torch.linalg.vector_norm(full_reconstruction - x).item() / scale
        ),
        "full_forecast": float(
            torch.linalg.vector_norm(full_forecast - x).item() / scale
        ),
    }


def antithetic_directions(count: int, seed: int) -> np.ndarray:
    if count <= 0 or count % 2:
        raise ValueError("Antithetic direction count must be positive and even")
    rng = np.random.default_rng(seed)
    half = rng.normal(size=(count // 2, 2))
    half /= np.linalg.norm(half, axis=1, keepdims=True)
    return np.concatenate([half, -half], axis=0).astype(np.float64)


def fit_origin_slope(offsets: np.ndarray, changes: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float64)
    changes = np.asarray(changes, dtype=np.float64)
    if offsets.ndim != 2 or offsets.shape != changes.shape:
        raise ValueError((offsets.shape, changes.shape))
    coefficients, *_ = np.linalg.lstsq(offsets, changes, rcond=None)
    return coefficients.T


def origin_linear_fit_diagnostics(
    offsets: np.ndarray, changes: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Fit an origin slope and return energy-normalized RMS nonlinearity."""
    matrix = fit_origin_slope(offsets, changes)
    fitted = np.asarray(offsets, dtype=np.float64) @ matrix.T
    changes = np.asarray(changes, dtype=np.float64)
    residual_rms = math.sqrt(float(np.mean(np.sum((changes - fitted) ** 2, axis=1))))
    change_rms = math.sqrt(float(np.mean(np.sum(changes**2, axis=1))))
    relative = residual_rms / max(change_rms, EPS)
    return matrix, residual_rms, change_rms, relative


def finite_radius_sweep(
    model,
    center: np.ndarray,
    mask: np.ndarray,
    autograd_matrix: np.ndarray,
    true_matrices: np.ndarray,
    radii: list[float],
    directions: np.ndarray,
    true_index: int,
    estimand: str,
) -> list[dict[str, Any]]:
    center_update = evaluate_update_points(
        model, np.asarray(center)[None, :], mask, estimand
    )[0]
    records = []
    for radius in radii:
        offsets = float(radius) * directions
        updates = evaluate_update_points(
            model, np.asarray(center)[None, :] + offsets, mask, estimand
        )
        matrix, residual_rms, change_rms, relative_residual = (
            origin_linear_fit_diagnostics(offsets, updates - center_update)
        )
        costs = [
            float(np.linalg.norm(matrix - truth, ord="fro"))
            for truth in true_matrices
        ]
        records.append(
            {
                "radius": float(radius),
                "matrix": matrix.tolist(),
                "autograd_agreement": float(
                    np.linalg.norm(matrix - autograd_matrix, ord="fro")
                    / max(np.linalg.norm(autograd_matrix, ord="fro"), EPS)
                ),
                "linear_fit_residual_rms": residual_rms,
                "centered_change_rms": change_rms,
                "normalized_linear_fit_residual": relative_residual,
                "law_costs": costs,
                "own_law_is_nearest": bool(int(np.argmin(costs)) == true_index),
            }
        )
    return records


def true_update_rms(offsets: np.ndarray, matrix: np.ndarray) -> float:
    updates = np.asarray(offsets, dtype=np.float64) @ np.asarray(matrix).T
    return math.sqrt(float(np.mean(np.sum(updates**2, axis=1))))


def _point_vector_rms(value: torch.Tensor) -> float:
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError(f"Expected nonempty [points, coordinates], got {value.shape}")
    return float(torch.sqrt(value.double().square().sum() / value.shape[0]).item())


def direct_latent_closure(
    model,
    points: np.ndarray,
    mask: np.ndarray,
    permutation: np.ndarray | None = None,
) -> dict[str, float]:
    """Score family-projected-code and descriptive whole-matrix leakage."""
    parameter = next(model.parameters())
    x = torch.as_tensor(points, dtype=parameter.dtype, device=parameter.device)
    p = torch.as_tensor(mask, dtype=parameter.dtype, device=parameter.device)
    with torch.no_grad():
        z = model.encode(x)
        if permutation is not None:
            index = torch.as_tensor(permutation, dtype=torch.long, device=z.device)
            z = z.index_select(-1, index)
            p = p.index_select(-1, index)
        source = z * p
        stepped = source @ model.kmatrix()
        outside = stepped * (1.0 - p)
        change = stepped - source
        numerator_rms = _point_vector_rms(outside)
        change_rms = _point_vector_rms(change)
        stepped_rms = _point_vector_rms(stepped)
        k_matrix = model.kmatrix()
        source_rows = k_matrix * p[:, None]
        matrix_outside = source_rows * (1.0 - p)[None, :]
        matrix_change = source_rows - torch.diag(p)
        matrix_source_norm = float(torch.linalg.matrix_norm(source_rows.double()).item())
        matrix_change_norm = float(torch.linalg.matrix_norm(matrix_change.double()).item())
        matrix_outside_norm = float(torch.linalg.matrix_norm(matrix_outside.double()).item())
    return {
        "outside_rms": numerator_rms,
        "change_denominator_rms": change_rms,
        "change_normalized_leakage": numerator_rms / max(change_rms, EPS),
        "raw_activity_leakage": numerator_rms / max(stepped_rms, EPS),
        "matrix_raw_leakage": matrix_outside_norm / max(matrix_source_norm, EPS),
        "matrix_change_leakage": matrix_outside_norm / max(matrix_change_norm, EPS),
    }
