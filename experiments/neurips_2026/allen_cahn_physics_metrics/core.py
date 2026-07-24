"""Prospectively frozen Allen--Cahn physics metrics.

All kernels operate on direct-rollout float32 fields, retain every trajectory
and every physical time through T=20, and return per-trajectory curves before
the frozen cell-level reduction.  Known wells are used only for scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


HORIZON = 200
GRID_SIZE = 16
CHANNELS = 2
NUM_WELLS = 4


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: str
    family: str


METRIC_SPECS = (
    MetricSpec("nearest_well_pixel_disagreement", "lower", "phase_assignment"),
    MetricSpec("modal_well_accuracy", "higher", "phase_assignment"),
    MetricSpec("well_area_fraction_tv_error", "lower", "phase_assignment"),
    MetricSpec("interface_edge_disagreement", "lower", "interface_geometry"),
    MetricSpec("free_energy_absolute_error", "lower", "thermodynamics"),
    MetricSpec("potential_energy_absolute_error", "lower", "thermodynamics"),
    MetricSpec("gradient_energy_absolute_error", "lower", "thermodynamics"),
)
METRIC_NAMES = tuple(spec.name for spec in METRIC_SPECS)


def _require_fields(fields: torch.Tensor, *, times: int) -> None:
    expected = (GRID_SIZE, GRID_SIZE, CHANNELS)
    if fields.ndim != 5 or fields.shape[1] != times or tuple(fields.shape[-3:]) != expected:
        raise ValueError(
            f"Expected [trajectory,{times},{GRID_SIZE},{GRID_SIZE},{CHANNELS}], "
            f"got {tuple(fields.shape)}"
        )
    if fields.dtype != torch.float32 or not bool(torch.isfinite(fields).all()):
        raise FloatingPointError("Every scored field must be finite float32")


def _require_centers(centers: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
    if tuple(centers.shape) != (NUM_WELLS, CHANNELS):
        raise ValueError(f"Expected four two-dimensional wells, got {tuple(centers.shape)}")
    if not bool(torch.isfinite(centers).all()):
        raise FloatingPointError("Well centers are nonfinite")
    return centers.to(device=fields.device, dtype=fields.dtype)


def nearest_well_labels(fields: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Assign every pixel to its nearest frozen well; ties use center order."""

    device_centers = _require_centers(centers, fields)
    distances = (fields.unsqueeze(-2) - device_centers).square().sum(dim=-1)
    labels = distances.argmin(dim=-1)
    if labels.dtype != torch.int64:
        raise AssertionError("Nearest-well labels are not int64")
    return labels


def _well_fractions_and_modal(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.stack(
        [(labels == well).sum(dim=(-2, -1)) for well in range(NUM_WELLS)], dim=-1
    )
    fractions = counts.to(torch.float64) / float(GRID_SIZE * GRID_SIZE)
    maximum = counts.max(dim=-1, keepdim=True).values
    modal = counts.argmax(dim=-1)
    tied = (counts == maximum).sum(dim=-1) > 1
    return fractions, modal, tied


def _interface_edges(labels: torch.Tensor) -> torch.Tensor:
    horizontal = labels != torch.roll(labels, shifts=-1, dims=-1)
    vertical = labels != torch.roll(labels, shifts=-1, dims=-2)
    return torch.stack((horizontal, vertical), dim=-1)


def energy_components(
    fields: torch.Tensor,
    centers: torch.Tensor,
    *,
    beta: float,
    reaction_strength: float,
    diffusion: float,
) -> dict[str, torch.Tensor]:
    """Return the exact discrete gradient-flow energy and its components.

    On the unit periodic square with spacing ``1 / GRID_SIZE``, the pinned RHS
    is the negative L2 gradient of

      mean_x[-r/(2 beta) logsumexp_i(-beta ||u-c_i||^2)]
      + diffusion/2 * sum_{x,forward edges} ||u(x+e)-u(x)||^2.

    The additive potential constant cancels in every reported prediction-truth
    error.  Energy arithmetic is float64; forecast fields remain float32.
    """

    if beta <= 0 or reaction_strength <= 0 or diffusion < 0:
        raise ValueError("Invalid frozen Allen--Cahn energy coefficient")
    values = fields.to(torch.float64)
    device_centers = centers.to(device=fields.device, dtype=torch.float64)
    squared_distances = (values.unsqueeze(-2) - device_centers).square().sum(dim=-1)
    potential_per_pixel = -torch.logsumexp(-float(beta) * squared_distances, dim=-1)
    potential_per_pixel = potential_per_pixel * (float(reaction_strength) / (2.0 * float(beta)))
    potential = potential_per_pixel.mean(dim=(-2, -1))
    forward_x = torch.roll(values, shifts=-1, dims=-2) - values
    forward_y = torch.roll(values, shifts=-1, dims=-3) - values
    gradient = 0.5 * float(diffusion) * (
        forward_x.square().sum(dim=-1).sum(dim=(-2, -1))
        + forward_y.square().sum(dim=-1).sum(dim=(-2, -1))
    )
    free = potential + gradient
    result = {"potential": potential, "gradient": gradient, "free": free}
    if any(not bool(value.isfinite().all()) for value in result.values()):
        raise FloatingPointError("Energy functional produced a nonfinite value")
    return result


def field_features(
    fields: torch.Tensor,
    centers: torch.Tensor,
    *,
    beta: float,
    reaction_strength: float,
    diffusion: float,
) -> dict[str, torch.Tensor]:
    labels = nearest_well_labels(fields, centers)
    fractions, modal, modal_tie = _well_fractions_and_modal(labels)
    energy = energy_components(
        fields,
        centers,
        beta=beta,
        reaction_strength=reaction_strength,
        diffusion=diffusion,
    )
    return {
        "labels": labels,
        "fractions": fractions,
        "modal": modal,
        "modal_tie": modal_tie,
        "interfaces": _interface_edges(labels),
        **energy,
    }


def compare_feature_curves(
    candidate: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return mandatory per-trajectory-by-time physics curves."""

    curves = {
        "nearest_well_pixel_disagreement": (candidate["labels"] != truth["labels"])
        .to(torch.float64)
        .mean(dim=(-2, -1)),
        "modal_well_accuracy": (candidate["modal"] == truth["modal"]).to(torch.float64),
        "well_area_fraction_tv_error": 0.5
        * (candidate["fractions"] - truth["fractions"]).abs().sum(dim=-1),
        "interface_edge_disagreement": (candidate["interfaces"] != truth["interfaces"])
        .to(torch.float64)
        .mean(dim=(-3, -2, -1)),
        "free_energy_absolute_error": (candidate["free"] - truth["free"]).abs(),
        "potential_energy_absolute_error": (
            candidate["potential"] - truth["potential"]
        ).abs(),
        "gradient_energy_absolute_error": (
            candidate["gradient"] - truth["gradient"]
        ).abs(),
    }
    if tuple(curves) != METRIC_NAMES:
        raise AssertionError("Physics metric roster drifted")
    expected = truth["modal"].shape
    if any(tuple(value.shape) != expected for value in curves.values()):
        raise AssertionError("A physics metric has the wrong trajectory/time shape")
    if any(not bool(value.isfinite().all()) for value in curves.values()):
        raise FloatingPointError("A per-trajectory physics metric is nonfinite")
    return curves


def score_candidate(
    predictions: torch.Tensor,
    truth: torch.Tensor,
    centers: torch.Tensor,
    *,
    beta: float,
    reaction_strength: float,
    diffusion: float,
    truth_features: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Reduce every mandatory curve only after strict per-value finiteness."""

    _require_fields(predictions, times=HORIZON)
    _require_fields(truth, times=HORIZON)
    if predictions.shape != truth.shape:
        raise ValueError("Prediction and truth panels differ")
    frozen_truth = dict(truth_features) if truth_features is not None else field_features(
        truth,
        centers,
        beta=beta,
        reaction_strength=reaction_strength,
        diffusion=diffusion,
    )
    candidate = field_features(
        predictions,
        centers,
        beta=beta,
        reaction_strength=reaction_strength,
        diffusion=diffusion,
    )
    per_trajectory = compare_feature_curves(candidate, frozen_truth)
    denominator = torch.arange(1, HORIZON + 1, dtype=torch.float64, device=truth.device)
    curves: dict[str, dict[str, list[float]]] = {}
    for name in METRIC_NAMES:
        instantaneous = per_trajectory[name].mean(dim=0)
        cumulative = instantaneous.cumsum(dim=0) / denominator
        curves[name] = {
            "instantaneous": instantaneous.detach().cpu().tolist(),
            "cumulative": cumulative.detach().cpu().tolist(),
        }
    diagnostics = {
        "truth_modal_tie_rate": frozen_truth["modal_tie"]
        .to(torch.float64)
        .mean(dim=0)
        .detach()
        .cpu()
        .tolist(),
        "candidate_modal_tie_rate": candidate["modal_tie"]
        .to(torch.float64)
        .mean(dim=0)
        .detach()
        .cpu()
        .tolist(),
    }
    validate_score_record({"curves": curves, "diagnostics": diagnostics})
    return {"curves": curves, "diagnostics": diagnostics}


def validate_score_record(record: Mapping[str, Any]) -> None:
    curves = record.get("curves")
    if not isinstance(curves, Mapping) or set(curves) != set(METRIC_NAMES):
        raise AssertionError("Stored physics curve roster is incomplete or reordered")
    horizon = torch.arange(1, HORIZON + 1, dtype=torch.float64)
    for name in METRIC_NAMES:
        instantaneous = torch.as_tensor(curves[name]["instantaneous"], dtype=torch.float64)
        cumulative = torch.as_tensor(curves[name]["cumulative"], dtype=torch.float64)
        if tuple(instantaneous.shape) != (HORIZON,) or tuple(cumulative.shape) != (HORIZON,):
            raise AssertionError(f"Stored {name} curve is not H200 complete")
        if not bool(instantaneous.isfinite().all()) or not bool(cumulative.isfinite().all()):
            raise FloatingPointError(f"Stored {name} curve is nonfinite")
        torch.testing.assert_close(
            cumulative,
            instantaneous.cumsum(dim=0) / horizon,
            rtol=1e-12,
            atol=1e-14,
        )
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "truth_modal_tie_rate",
        "candidate_modal_tie_rate",
    }:
        raise AssertionError("Tie diagnostics are incomplete")
    for values in diagnostics.values():
        tensor = torch.as_tensor(values, dtype=torch.float64)
        if tuple(tensor.shape) != (HORIZON,) or not bool(tensor.isfinite().all()):
            raise FloatingPointError("Tie diagnostic is incomplete or nonfinite")
        if bool(((tensor < 0) | (tensor > 1)).any()):
            raise AssertionError("Tie diagnostic lies outside [0,1]")
