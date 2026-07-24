"""Evaluation-only fate and support-family metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def allen_cahn_centers(radius: float = 1.5) -> torch.Tensor:
    angles = torch.arange(4, dtype=torch.float64) * (0.5 * torch.pi)
    return float(radius) * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)


def modal_well_fates(flat_fields: torch.Tensor, *, grid_size: int = 16) -> torch.Tensor:
    if flat_fields.ndim != 2 or flat_fields.shape[1] != 2 * grid_size * grid_size:
        raise ValueError("Expected flattened two-channel fields")
    fields = flat_fields.reshape(-1, grid_size, grid_size, 2).double()
    centers = allen_cahn_centers().to(fields)
    distances = (fields.unsqueeze(-2) - centers.view(1, 1, 1, 4, 2)).square().sum(-1)
    pixel = distances.argmin(-1)
    counts = torch.stack([(pixel == index).sum((1, 2)) for index in range(4)], dim=1)
    return counts.argmax(1).long().cpu()


def _entropy(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / max(1, counts.sum())
    return float(-(probabilities * np.log(probabilities)).sum())


def _conditional_entropy(left: np.ndarray, given: np.ndarray) -> float:
    result = 0.0
    for value in np.unique(given):
        keep = given == value
        result += float(keep.mean()) * _entropy(left[keep])
    return result


def _purity(fates: np.ndarray, families: np.ndarray) -> float:
    correct = 0
    for family in np.unique(families):
        values, counts = np.unique(fates[families == family], return_counts=True)
        del values
        correct += int(counts.max())
    return float(correct / max(1, families.size))


def alignment_metrics(assignments: np.ndarray, fates: np.ndarray) -> dict[str, Any]:
    assignments = np.asarray(assignments, dtype=np.int64)
    fates = np.asarray(fates, dtype=np.int64)
    if assignments.shape != fates.shape or assignments.ndim != 1:
        raise ValueError("assignments and fates must be paired vectors")
    covered = assignments >= 0
    if not np.any(covered):
        return {
            "defined_on_covered_rows": False,
            "coverage": 0.0,
            "covered_count": 0,
            "ari": 0.0,
            "nmi": 0.0,
            "purity": 0.0,
            "h_fate_given_family_over_h_fate": 1.0,
            "h_family_given_fate_over_h_family": 1.0,
            "family_count": 0,
        }
    family = assignments[covered]
    fate = fates[covered]
    h_fate, h_family = _entropy(fate), _entropy(family)
    return {
        "defined_on_covered_rows": True,
        "coverage": float(covered.mean()),
        "covered_count": int(covered.sum()),
        "ari": float(adjusted_rand_score(fate, family)),
        "nmi": float(normalized_mutual_info_score(fate, family)),
        "purity": _purity(fate, family),
        "h_fate_given_family_over_h_fate": (
            0.0 if h_fate <= 1e-15 else _conditional_entropy(fate, family) / h_fate
        ),
        "h_family_given_fate_over_h_family": (
            0.0 if h_family <= 1e-15 else _conditional_entropy(family, fate) / h_family
        ),
        "family_count": int(np.unique(family).size),
    }


def jaccard_rows(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Masks must be paired [trajectory, coordinate] arrays")
    intersection = np.logical_and(reference, candidate).sum(1)
    union = np.logical_or(reference, candidate).sum(1)
    return np.divide(
        intersection,
        union,
        out=np.ones(reference.shape[0], dtype=np.float64),
        where=union > 0,
    )


def truth_difficulty(
    x0: torch.Tensor, x200: torch.Tensor, x400: torch.Tensor
) -> dict[str, float]:
    for value in (x0, x200, x400):
        if value.ndim != 2:
            raise ValueError("Truth fields must be [trajectory, state]")
    numerator = float((x400.double() - x200.double()).square().mean())
    denominator = float((x200.double() - x0.double()).square().mean())
    fate200 = modal_well_fates(x200)
    fate400 = modal_well_fates(x400)
    ratio = numerator / max(denominator, 1e-20)
    changes = float((fate200 != fate400).double().mean())
    return {
        "continued_change_mse": numerator,
        "h200_change_from_initial_mse": denominator,
        "continued_change_ratio": ratio,
        "modal_fate_change_fraction": changes,
        "dynamic_temporal_extrapolation": bool(ratio >= 0.05 or changes >= 0.05),
    }


def modal_accuracy(predicted: torch.Tensor, truth: torch.Tensor) -> float:
    predicted_fate = modal_well_fates(predicted)
    truth_fate = modal_well_fates(truth)
    return float((predicted_fate == truth_fate).double().mean())


def normalized_entropy_score(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else 1.0 - value
