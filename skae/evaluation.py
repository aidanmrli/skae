"""Comprehensive evaluation utilities for Koopman Autoencoder models.

This module implements the evaluation protocol described in the research
specification. It supports multiple rollout strategies, computes horizon-wise
mean-squared error metrics, and produces qualitative plots such as phase
portraits and MSE-vs-horizon curves.

Key features
------------
- Rollout generators for:
  * No reencoding (latent-only evolution)
  * Every-step reencoding (state-space evolution)
  * Periodic reencoding with configurable period k
- Evaluation over multiple dynamical systems, horizons, and reencoding periods
- Aggregation of metrics across unseen initial conditions (mean ± std)
- Automatic selection of the best periodic reencoding period per horizon
- Qualitative plots with ground truth trajectories in transparent gray

The public entry point is :func:`evaluate_model`, which returns a nested metrics
dictionary and optionally saves metrics/plots to disk.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.config import Config
from skae.data import (
    DystsTrajectoryCache,
    LyapunovMultiAttractor,
    VectorWrapper,
    generate_trajectory,
    make_env,
)
from skae.model import KoopmanMachine


# ---------------------------------------------------------------------------
# Rollout generators
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_no_reencode(model: KoopmanMachine, x0: torch.Tensor, horizon: int) -> torch.Tensor:
    """Roll out the Koopman dynamics without reencoding.

    Args:
        model: Trained Koopman machine.
        x0: Initial states with shape ``[batch, state_dim]``.
        horizon: Number of prediction steps.

    Returns:
        Predicted trajectory with shape ``[horizon, batch, state_dim]``.
    """
    model.eval()
    device = next(model.parameters()).device
    x0 = x0.to(device)

    latent = model.encode(x0)
    predictions: List[torch.Tensor] = []

    for _ in range(horizon):
        latent = model.step_latent(latent)
        x_pred = model.decode(latent)
        predictions.append(x_pred)

        if not torch.isfinite(x_pred).all():
            # Mark remaining steps as NaN to signal explosion
            nan_frame = torch.full_like(x_pred, torch.nan)
            predictions.extend([nan_frame] * (horizon - len(predictions)))
            break

    return torch.stack(predictions, dim=0)


def _normalized_projection_gap(
    reencoded_latent: torch.Tensor,
    predicted_latent: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return per-sample normalized projection gaps."""
    numerator = torch.norm(reencoded_latent - predicted_latent, dim=-1)
    denominator = torch.norm(predicted_latent, dim=-1).clamp(min=eps)
    return numerator / denominator


def _reencode_latent(
    model: KoopmanMachine,
    predicted_state: torch.Tensor,
    predicted_latent: torch.Tensor,
    *,
    use_dynamics_prior: bool,
) -> torch.Tensor:
    """Reencode a predicted state, optionally warm-started by the predicted latent."""
    latent_prior = predicted_latent if use_dynamics_prior else None
    return model.encode_with_prior(predicted_state, latent_prior=latent_prior)


def _empty_rollout_diagnostics(
    *,
    horizon: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Allocate empty diagnostics for reset-aware rollout modes."""
    return {
        "projection_gap": torch.full((horizon, batch_size), torch.nan, device=device),
        "ambiguity_score": torch.full((horizon, batch_size), torch.nan, device=device),
        "spillover_score": torch.full((horizon, batch_size), torch.nan, device=device),
        "support_margin_ratio": torch.full((horizon, batch_size), torch.nan, device=device),
        "reset_mask": torch.zeros((horizon, batch_size), dtype=torch.bool, device=device),
        "threshold_trigger_mask": torch.zeros((horizon, batch_size), dtype=torch.bool, device=device),
        "interval_trigger_mask": torch.zeros((horizon, batch_size), dtype=torch.bool, device=device),
        "proj_trigger_mask": torch.zeros((horizon, batch_size), dtype=torch.bool, device=device),
        "ambiguity_trigger_mask": torch.zeros((horizon, batch_size), dtype=torch.bool, device=device),
        "spillover_trigger_mask": torch.zeros((horizon, batch_size), dtype=torch.bool, device=device),
        "support_margin_trigger_mask": torch.zeros((horizon, batch_size), dtype=torch.bool, device=device),
    }


def _event_trigger_group_slices(model: KoopmanMachine) -> Optional[Tuple[slice, ...]]:
    """Infer latent group slices for ambiguity/spillover triggers."""

    if all(hasattr(model, attr) for attr in ("d_global", "num_basins", "d_basin")):
        d_global = int(getattr(model, "d_global"))
        num_basins = int(getattr(model, "num_basins"))
        d_basin = int(getattr(model, "d_basin"))
        return tuple(
            slice(d_global + basin_index * d_basin, d_global + (basin_index + 1) * d_basin)
            for basin_index in range(num_basins)
        )

    k_structure = getattr(model, "_k_structure", None)
    if k_structure == "block_diagonal":
        sizes = tuple(int(block_size) for block_size in getattr(model, "_k_block_sizes", ()))
    else:
        sizes = tuple(int(block_size) for block_size in getattr(model, "_soft_block_sizes", ()))
    if not sizes:
        return None

    offset = 0
    group_slices: List[slice] = []
    for block_size in sizes:
        group_slices.append(slice(offset, offset + block_size))
        offset += block_size
    return tuple(group_slices)


def _group_energies_from_slices(
    z: torch.Tensor,
    group_slices: Sequence[slice],
) -> Optional[torch.Tensor]:
    """Compute L2 group energies for the provided latent slices."""

    if not group_slices:
        return None
    energies = [
        torch.norm(z[..., group_slice], p=2, dim=-1, keepdim=True)
        for group_slice in group_slices
    ]
    return torch.cat(energies, dim=-1)


def _support_margin_ratio(
    z: torch.Tensor,
    *,
    support_threshold: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return min-active-to-threshold ratios used for support-fragility triggers."""

    threshold = max(float(support_threshold), eps)
    z_abs = z.abs()
    active_mask = z_abs > threshold
    min_active = torch.where(
        active_mask,
        z_abs,
        torch.full_like(z_abs, float("inf")),
    ).min(dim=-1).values
    has_active = active_mask.any(dim=-1)
    min_active = torch.where(has_active, min_active, torch.zeros_like(min_active))
    return min_active / threshold


def _compute_event_trigger_scores(
    model: KoopmanMachine,
    latent: torch.Tensor,
    predicted_latent: torch.Tensor,
    reencoded_latent: torch.Tensor,
    *,
    support_threshold: float,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compute label-free event-trigger scores from the current rollout state."""

    batch_shape = predicted_latent.shape[:-1]
    empty_score = torch.full(batch_shape, torch.nan, device=predicted_latent.device, dtype=predicted_latent.dtype)
    scores: Dict[str, torch.Tensor] = {
        "projection_gap": _normalized_projection_gap(reencoded_latent, predicted_latent, eps=eps),
        "ambiguity_score": empty_score.clone(),
        "spillover_score": empty_score.clone(),
        "support_margin_ratio": _support_margin_ratio(
            predicted_latent,
            support_threshold=support_threshold,
            eps=eps,
        ),
    }

    group_slices = _event_trigger_group_slices(model)
    if group_slices:
        current_energies = _group_energies_from_slices(latent, group_slices)
        next_energies = _group_energies_from_slices(predicted_latent, group_slices)
        if current_energies is not None and next_energies is not None:
            next_total = next_energies.sum(dim=-1, keepdim=True).clamp(min=eps)
            next_probs = next_energies / next_total
            scores["ambiguity_score"] = 1.0 - next_probs.max(dim=-1).values

            dominant_group = current_energies.argmax(dim=-1, keepdim=True)
            dominant_next_energy = next_energies.gather(dim=-1, index=dominant_group).squeeze(-1)
            scores["spillover_score"] = 1.0 - dominant_next_energy / next_total.squeeze(-1)

    return scores


def _format_event_threshold_tag(value: float | int) -> str:
    """Format thresholds for compact metric/mode names."""

    return str(value).replace("-", "m").replace(".", "p")


def _event_trigger_mode_name(settings: "EvaluationSettings") -> str:
    """Build a stable mode name for the configured event-trigger policy."""

    if (
        settings.event_trigger_ambiguity_threshold is None
        and settings.event_trigger_spillover_threshold is None
        and settings.event_trigger_support_margin_min_ratio is None
    ):
        if settings.event_trigger_proj_threshold is not None:
            return f"event_proj_{_format_event_threshold_tag(settings.event_trigger_proj_threshold)}"
        if settings.event_trigger_max_interval > 0:
            return f"event_interval_{int(settings.event_trigger_max_interval)}"
        return "event_trigger"

    parts: List[str] = []
    if settings.event_trigger_proj_threshold is not None:
        parts.append(f"proj_{_format_event_threshold_tag(settings.event_trigger_proj_threshold)}")
    if settings.event_trigger_ambiguity_threshold is not None:
        parts.append(f"amb_{_format_event_threshold_tag(settings.event_trigger_ambiguity_threshold)}")
    if settings.event_trigger_spillover_threshold is not None:
        parts.append(f"spill_{_format_event_threshold_tag(settings.event_trigger_spillover_threshold)}")
    if settings.event_trigger_support_margin_min_ratio is not None:
        parts.append(f"margin_{_format_event_threshold_tag(settings.event_trigger_support_margin_min_ratio)}")
    if settings.event_trigger_max_interval > 0:
        parts.append(f"maxint_{int(settings.event_trigger_max_interval)}")
    return "event_hybrid_" + "_".join(parts)


@torch.no_grad()
def _rollout_periodic_reencode_with_diagnostics(
    model: KoopmanMachine,
    x0: torch.Tensor,
    horizon: int,
    period: int,
    *,
    use_dynamics_prior: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Periodic-reset rollout plus projection-gap diagnostics."""

    if period <= 0:
        raise ValueError("period must be a positive integer")

    model.eval()
    device = next(model.parameters()).device
    x0 = x0.to(device)

    latent = model.encode(x0)
    batch_size = int(x0.shape[0])
    diagnostics = _empty_rollout_diagnostics(horizon=horizon, batch_size=batch_size, device=device)
    predictions: List[torch.Tensor] = []

    for step in range(horizon):
        predicted_latent = model.step_latent(latent)
        predicted_state = model.decode(predicted_latent)
        predictions.append(predicted_state)

        if not torch.isfinite(predicted_state).all():
            nan_frame = torch.full_like(predicted_state, torch.nan)
            predictions.extend([nan_frame] * (horizon - len(predictions)))
            break

        reencoded_latent = _reencode_latent(
            model,
            predicted_state,
            predicted_latent,
            use_dynamics_prior=use_dynamics_prior,
        )
        diagnostics["projection_gap"][step] = _normalized_projection_gap(
            reencoded_latent, predicted_latent
        )

        should_reset = (step + 1) % period == 0
        if should_reset:
            diagnostics["reset_mask"][step] = True
            latent = reencoded_latent
        else:
            latent = predicted_latent

    return torch.stack(predictions, dim=0), diagnostics


@torch.no_grad()
def rollout_every_step_reencode(
    model: KoopmanMachine,
    x0: torch.Tensor,
    horizon: int,
    *,
    use_dynamics_prior: bool = False,
) -> torch.Tensor:
    """Roll out the Koopman dynamics with reencoding at every step."""
    predictions, _ = _rollout_periodic_reencode_with_diagnostics(
        model,
        x0,
        horizon,
        period=1,
        use_dynamics_prior=use_dynamics_prior,
    )
    return predictions


@torch.no_grad()
def rollout_periodic_reencode(
    model: KoopmanMachine,
    x0: torch.Tensor,
    horizon: int,
    period: int,
    *,
    use_dynamics_prior: bool = False,
) -> torch.Tensor:
    """Roll out the Koopman dynamics with periodic reencoding every *period* steps."""
    predictions, _ = _rollout_periodic_reencode_with_diagnostics(
        model,
        x0,
        horizon,
        period=period,
        use_dynamics_prior=use_dynamics_prior,
    )
    return predictions


@torch.no_grad()
def rollout_event_trigger_reencode(
    model: KoopmanMachine,
    x0: torch.Tensor,
    horizon: int,
    proj_threshold: Optional[float],
    *,
    ambiguity_threshold: Optional[float] = None,
    spillover_threshold: Optional[float] = None,
    support_margin_min_ratio: Optional[float] = None,
    support_threshold: float = 1e-3,
    min_dwell: int = 0,
    max_interval: int = 0,
    use_dynamics_prior: bool = False,
) -> torch.Tensor:
    """Roll out with hybrid event-triggered reencoding."""
    predictions, _ = _rollout_event_trigger_reencode_with_diagnostics(
        model,
        x0,
        horizon,
        proj_threshold=proj_threshold,
        ambiguity_threshold=ambiguity_threshold,
        spillover_threshold=spillover_threshold,
        support_margin_min_ratio=support_margin_min_ratio,
        support_threshold=support_threshold,
        min_dwell=min_dwell,
        max_interval=max_interval,
        use_dynamics_prior=use_dynamics_prior,
    )
    return predictions


@torch.no_grad()
def _rollout_event_trigger_reencode_with_diagnostics(
    model: KoopmanMachine,
    x0: torch.Tensor,
    horizon: int,
    proj_threshold: Optional[float],
    *,
    ambiguity_threshold: Optional[float] = None,
    spillover_threshold: Optional[float] = None,
    support_margin_min_ratio: Optional[float] = None,
    support_threshold: float = 1e-3,
    min_dwell: int = 0,
    max_interval: int = 0,
    use_dynamics_prior: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Hybrid event-triggered rollout plus reset diagnostics."""

    if proj_threshold is not None and proj_threshold < 0.0:
        raise ValueError("proj_threshold must be non-negative")
    if ambiguity_threshold is not None and ambiguity_threshold < 0.0:
        raise ValueError("ambiguity_threshold must be non-negative")
    if spillover_threshold is not None and spillover_threshold < 0.0:
        raise ValueError("spillover_threshold must be non-negative")
    if support_margin_min_ratio is not None and support_margin_min_ratio < 0.0:
        raise ValueError("support_margin_min_ratio must be non-negative")
    if support_threshold < 0.0:
        raise ValueError("support_threshold must be non-negative")
    if min_dwell < 0:
        raise ValueError("min_dwell must be non-negative")
    if max_interval < 0:
        raise ValueError("max_interval must be non-negative")
    if (
        proj_threshold is None
        and ambiguity_threshold is None
        and spillover_threshold is None
        and support_margin_min_ratio is None
        and max_interval <= 0
    ):
        raise ValueError("At least one event-trigger threshold or max_interval must be set")
    if (
        ambiguity_threshold is not None or spillover_threshold is not None
    ) and _event_trigger_group_slices(model) is None:
        raise ValueError(
            "ambiguity/spillover triggers require block-diagonal, soft-block, or structured basin groups"
        )

    model.eval()
    device = next(model.parameters()).device
    x0 = x0.to(device)

    latent = model.encode(x0)
    batch_size = int(x0.shape[0])
    steps_since_reset = torch.zeros(batch_size, dtype=torch.long, device=device)
    diagnostics = _empty_rollout_diagnostics(horizon=horizon, batch_size=batch_size, device=device)
    predictions: List[torch.Tensor] = []

    for step in range(horizon):
        predicted_latent = model.step_latent(latent)
        predicted_state = model.decode(predicted_latent)
        predictions.append(predicted_state)

        if not torch.isfinite(predicted_state).all():
            nan_frame = torch.full_like(predicted_state, torch.nan)
            predictions.extend([nan_frame] * (horizon - len(predictions)))
            break

        reencoded_latent = _reencode_latent(
            model,
            predicted_state,
            predicted_latent,
            use_dynamics_prior=use_dynamics_prior,
        )
        scores = _compute_event_trigger_scores(
            model,
            latent,
            predicted_latent,
            reencoded_latent,
            support_threshold=support_threshold,
        )
        projection_gap = scores["projection_gap"]
        diagnostics["projection_gap"][step] = projection_gap
        diagnostics["ambiguity_score"][step] = scores["ambiguity_score"]
        diagnostics["spillover_score"][step] = scores["spillover_score"]
        diagnostics["support_margin_ratio"][step] = scores["support_margin_ratio"]

        next_steps_since_reset = steps_since_reset + 1
        dwell_mask = next_steps_since_reset > min_dwell
        proj_trigger = (
            projection_gap > proj_threshold
            if proj_threshold is not None
            else torch.zeros_like(projection_gap, dtype=torch.bool)
        )
        ambiguity_score = scores["ambiguity_score"]
        ambiguity_trigger = (
            ambiguity_score > ambiguity_threshold
            if ambiguity_threshold is not None
            else torch.zeros_like(projection_gap, dtype=torch.bool)
        )
        spillover_score = scores["spillover_score"]
        spillover_trigger = (
            spillover_score > spillover_threshold
            if spillover_threshold is not None
            else torch.zeros_like(projection_gap, dtype=torch.bool)
        )
        support_margin_ratio = scores["support_margin_ratio"]
        support_margin_trigger = (
            support_margin_ratio < support_margin_min_ratio
            if support_margin_min_ratio is not None
            else torch.zeros_like(projection_gap, dtype=torch.bool)
        )
        proj_trigger = proj_trigger & dwell_mask
        ambiguity_trigger = ambiguity_trigger & dwell_mask
        spillover_trigger = spillover_trigger & dwell_mask
        support_margin_trigger = support_margin_trigger & dwell_mask
        threshold_trigger = (
            proj_trigger
            | ambiguity_trigger
            | spillover_trigger
            | support_margin_trigger
        )
        interval_trigger = (
            next_steps_since_reset >= max_interval
            if max_interval > 0
            else torch.zeros_like(threshold_trigger)
        )
        should_reset = threshold_trigger | interval_trigger

        diagnostics["proj_trigger_mask"][step] = proj_trigger
        diagnostics["ambiguity_trigger_mask"][step] = ambiguity_trigger
        diagnostics["spillover_trigger_mask"][step] = spillover_trigger
        diagnostics["support_margin_trigger_mask"][step] = support_margin_trigger
        diagnostics["threshold_trigger_mask"][step] = threshold_trigger
        diagnostics["interval_trigger_mask"][step] = interval_trigger
        diagnostics["reset_mask"][step] = should_reset

        latent = torch.where(
            should_reset.unsqueeze(-1),
            reencoded_latent,
            predicted_latent,
        )
        steps_since_reset = torch.where(
            should_reset,
            torch.zeros_like(next_steps_since_reset),
            next_steps_since_reset,
        )

    return torch.stack(predictions, dim=0), diagnostics


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def _compute_horizon_metric_stats(
    per_step_values: torch.Tensor,
    horizon: int,
) -> Tuple[float, float, List[float], int]:
    """Compute mean ± std for a horizon-wise scalar metric.

    Args:
        per_step_values: Tensor ``[time, batch]`` with one scalar metric per
            step and initial condition.
        horizon: Horizon length (<= time dimension of per_step_values).

    Returns:
        Tuple ``(mean, std, per_ic, num_valid)`` where *per_ic* is a list of the
        per-initial-condition metric values used for aggregation.
    """

    horizon = min(horizon, per_step_values.size(0))
    horizon_errors = per_step_values[:horizon]

    # Mean over rollout time first, ignoring NaNs from exploding rollouts.
    per_ic = torch.nanmean(horizon_errors, dim=0)
    valid_mask = torch.isfinite(per_ic)

    if valid_mask.sum() == 0:
        return float("nan"), float("nan"), [], 0

    valid_errors = per_ic[valid_mask]
    # Then mean over initial conditions that retained a finite time average.
    mean = valid_errors.mean().item()
    std = valid_errors.std(unbiased=False).item() if valid_errors.numel() > 1 else 0.0
    return mean, std, valid_errors.tolist(), int(valid_mask.sum().item())


def _finite_coverage_stats(predictions: torch.Tensor, horizon: int) -> Dict[str, float | int]:
    """Summarize rollout finiteness through a requested horizon.

    The MSE reducer intentionally uses finite-prefix ``nanmean`` for historical
    continuity. These fields make the masking explicit: a rollout can have a
    finite prefix MSE while failing to remain finite through the whole horizon.
    """

    horizon = min(horizon, predictions.size(0))
    if horizon <= 0:
        return {
            "num_initial_conditions": int(predictions.size(1)) if predictions.dim() >= 2 else 0,
            "num_full_horizon_finite": 0,
            "full_horizon_finite_fraction": float("nan"),
            "finite_step_fraction": float("nan"),
            "mean_finite_prefix_length": float("nan"),
            "median_finite_prefix_length": float("nan"),
            "min_finite_prefix_length": 0,
        }

    finite_steps = torch.isfinite(predictions[:horizon]).all(dim=-1)
    if finite_steps.dim() != 2:
        raise ValueError(
            "Expected predictions with shape [time, batch, dim] for finite coverage stats."
        )

    batch_size = int(finite_steps.size(1))
    full_horizon_finite = finite_steps.all(dim=0)
    # Prefix length is stricter than total finite count if a mode ever produces
    # intermittent non-finite values.
    finite_prefix = finite_steps.cumprod(dim=0).sum(dim=0).to(torch.float32)
    return {
        "num_initial_conditions": batch_size,
        "num_full_horizon_finite": int(full_horizon_finite.sum().item()),
        "full_horizon_finite_fraction": float(full_horizon_finite.to(torch.float32).mean().item())
        if batch_size > 0
        else float("nan"),
        "finite_step_fraction": float(finite_steps.to(torch.float32).mean().item()),
        "mean_finite_prefix_length": float(finite_prefix.mean().item()) if batch_size > 0 else float("nan"),
        "median_finite_prefix_length": float(finite_prefix.median().item()) if batch_size > 0 else float("nan"),
        "min_finite_prefix_length": int(finite_prefix.min().item()) if batch_size > 0 else 0,
    }


def _cumulative_mse_curve(per_step_values: torch.Tensor) -> List[float]:
    """Compute cumulative MSE curve averaged across initial conditions."""

    time_steps = per_step_values.size(0)
    steps = torch.arange(1, time_steps + 1, dtype=torch.float32, device=per_step_values.device)
    cumulative = torch.cumsum(per_step_values, dim=0)
    with torch.no_grad():
        # Divide by elapsed time, then mean over initial conditions.
        curve = torch.nanmean(cumulative / steps.view(-1, 1), dim=1)
    return curve.cpu().tolist()


def _safe_float_for_best(value: object) -> Optional[float]:
    """Return a finite float or ``None`` for best-mode summaries."""
    if value is None:
        return None
    out = float(value)
    return out if math.isfinite(out) else None


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def _ensure_matplotlib():
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  # pylint: disable=unused-import


def _save_phase_portrait_overlay(
    true_sequences: torch.Tensor,
    predicted_sequences: Dict[str, torch.Tensor],
    path: Path,
    max_samples: int = 20,
) -> None:
    """Save a phase portrait overlay plot.

    Args:
        true_sequences: Tensor with shape ``[batch, time + 1, state_dim]``.
        predicted_sequences: Mapping from mode name to tensor with shape
            ``[batch, time, state_dim]``.
        path: Output path for the PNG file.
        max_samples: Maximum number of trajectories to render.
    """

    if true_sequences.size(-1) < 2:
        return  # Phase portrait not meaningful

    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    path.parent.mkdir(parents=True, exist_ok=True)

    batch = true_sequences.size(0)
    indices = torch.arange(batch)

    # Filter trajectories with finite predictions for all modes
    finite_mask = torch.ones(batch, dtype=torch.bool)
    true_xy = true_sequences[:, :, :2]

    for preds in predicted_sequences.values():
        flat = preds.reshape(preds.size(0), -1)
        finite_mask &= torch.isfinite(flat).all(dim=1)

    indices = indices[finite_mask]
    if indices.numel() == 0:
        return

    indices = indices[:max_samples]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Plot ground truth trajectories in light gray
    for idx in indices.tolist():
        gt = true_xy[idx].cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], color=(0.5, 0.5, 0.5), alpha=0.25, linewidth=1.5)

    rng = np.random.default_rng(42)
    colors = {
        mode: mcolors.hsv_to_rgb([float(rng.random()), 0.65 + 0.3 * float(rng.random()), 0.9])
        for mode in predicted_sequences.keys()
    }

    for mode, preds in predicted_sequences.items():
        color = colors[mode]
        for idx in indices.tolist():
            pred_xy = torch.cat([
                true_xy[idx, :1],
                preds[idx, :, :2],
            ], dim=0).cpu().numpy()
            ax.plot(
                pred_xy[:, 0],
                pred_xy[:, 1],
                color=color,
                alpha=0.9,
                linewidth=1.2,
                label=mode if idx == indices[0].item() else None,
            )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Phase portrait (1000-step)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal", adjustable="box")
    # ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_phase_portrait_single_mode(
    true_sequences: torch.Tensor,
    predicted: torch.Tensor,
    path: Path,
    max_samples: int = 20,
    title: Optional[str] = None,
    axis_lim: float = 2.5,
) -> None:
    """Save a phase portrait for a single rollout mode, coloring each trajectory.

    Args:
        true_sequences: Tensor with shape ``[batch, time + 1, state_dim]``.
        predicted: Tensor with shape ``[batch, time, state_dim]`` for the mode.
        path: Output PNG path.
        max_samples: Maximum number of trajectories to render.
        title: Optional plot title.
        axis_lim: Axis limits for the plot (default 2.5).
    """

    if true_sequences.size(-1) < 2:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    path.parent.mkdir(parents=True, exist_ok=True)

    batch = true_sequences.size(0)
    true_xy = true_sequences[:, :, :2]

    # Keep only trajectories with finite predictions
    flat = predicted.reshape(predicted.size(0), -1)
    finite_mask = torch.isfinite(flat).all(dim=1)
    indices = torch.arange(batch)[finite_mask]
    if indices.numel() == 0:
        return
    indices = indices[:max_samples]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Colormap per-trajectory
    cmap = cm.get_cmap("tab20", indices.numel())

    for j, idx in enumerate(indices.tolist()):
        # predicted (plot first, underneath)
        pred_xy = torch.cat([true_xy[idx, :1], predicted[idx, :, :2]], dim=0).cpu().numpy()
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], color=cmap(j), linewidth=1.5, zorder=2)

        # ground truth in light gray (plot last, on top)
        gt = true_xy[idx].cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], color=(0.6, 0.6, 0.6), alpha=0.5, linewidth=1.5, zorder=3)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or "Phase portrait (single mode)")
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_aspect("equal", adjustable="box")
    # ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _save_mse_curve_plot(curves: Dict[str, List[float]], path: Path, highlight_horizons: Sequence[int]) -> None:
    """Save MSE vs horizon curves for each rollout mode."""

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for mode, curve in curves.items():
        xs = np.arange(1, len(curve) + 1)
        ax.plot(xs, curve, linewidth=2, label=mode)

    for horizon in highlight_horizons:
        ax.axvline(horizon, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Mean MSE")
    ax.set_title("MSE vs horizon")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_horizon_mse_plot(
    mode_metrics: Dict[str, Dict],
    horizons: Sequence[int],
    modes: Sequence[str],
    path: Path,
) -> None:
    """Save mean ± std horizon MSE for selected modes."""

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    any_plotted = False

    for mode in modes:
        mode_data = mode_metrics.get(mode)
        if not mode_data:
            continue
        xs: List[int] = []
        means: List[float] = []
        stds: List[float] = []
        for horizon in horizons:
            horizon_data = mode_data.get("horizons", {}).get(str(horizon))
            if not horizon_data:
                continue
            mean = horizon_data.get("mean")
            std = horizon_data.get("std", 0.0)
            if mean is None or not np.isfinite(mean):
                continue
            xs.append(int(horizon))
            means.append(float(mean))
            stds.append(float(std) if std is not None else 0.0)
        if not xs:
            continue
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=3,
            label=mode,
        )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Mean MSE ± std")
    ax.set_title("Horizon MSE (selected modes)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_error_curve_single_mode(
    errors: torch.Tensor,
    path: Path,
    title: Optional[str] = None,
) -> None:
    """Save per-timestep mean L2 error for a single rollout mode."""

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    error_np = errors.cpu().numpy()
    steps = np.arange(1, error_np.shape[0] + 1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(steps, error_np, linewidth=2)
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("Mean L2 error")
    ax.set_title(title or "Per-step prediction error")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_error_curve_combined(
    errors_by_mode: Dict[str, torch.Tensor],
    path: Path,
    highlight_steps: Optional[Sequence[int]] = None,
) -> None:
    """Save combined per-step mean error curves for all rollout modes."""

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for mode, errors in errors_by_mode.items():
        error_np = errors.cpu().numpy()
        steps = np.arange(1, error_np.shape[0] + 1)
        ax.plot(steps, error_np, linewidth=2, label=mode)

    if highlight_steps is not None:
        for step in highlight_steps:
            if step <= 0:
                continue
            ax.axvline(step, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Prediction step")
    ax.set_ylabel("Mean L2 error")
    ax.set_title("Per-step prediction error (all modes)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_vector_magnitude_histogram(
    magnitudes: np.ndarray,
    path: Path,
    title: str,
    bins: int = 30,
) -> None:
    """Save a histogram of vector magnitudes used in a phase portrait."""

    flat = np.asarray(magnitudes, dtype=np.float32).ravel()
    if flat.size == 0:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(flat, bins=bins, color="#4682B4", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Vector magnitude")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _estimate_learned_attractors(
    model: KoopmanMachine,
    grid_lim: float,
    num_samples: int,
    num_steps: int,
    tolerance: float,
    device: torch.device,
    seed: int = 7,
) -> np.ndarray:
    """Estimate attractor locations of the learned system via rollouts."""

    rng = np.random.default_rng(seed)
    samples = rng.uniform(-grid_lim, grid_lim, size=(num_samples, 2)).astype(np.float32)
    attractors: List[np.ndarray] = []

    print(
        f"[lyapunov] Estimating learned attractors (samples={num_samples}, "
        f"steps={num_steps}, tolerance={tolerance})",
        flush=True,
    )

    report_interval = max(1, num_samples // 5)
    for idx, sample in enumerate(samples):
        state = torch.from_numpy(sample).to(device)
        with torch.no_grad():
            for _ in range(num_steps):
                state = model.step_env(state.unsqueeze(0)).squeeze(0)
            # Refinement pass: run additional steps so convergent systems land
            # close to their fixed-point/limit attractor before clustering.
            for _ in range(num_steps):
                state = model.step_env(state.unsqueeze(0)).squeeze(0)
                if not torch.isfinite(state).all():
                    break
        final_state = state.cpu().numpy()

        # Skip if final state is not finite or has extreme values
        if not np.isfinite(final_state).all() or np.abs(final_state).max() > 1e6:
            print(f"[forecasting explosion] Skipping non-finite final state: {final_state}", flush=True)
            continue

        # NOTE: check this? we are just making the first final 
        # state an attractor
        if not attractors:
            attractors.append(final_state)
            continue

        existing = np.asarray(attractors)
        dists = np.linalg.norm(existing - final_state, axis=1)
        if float(dists.min()) > tolerance:
            attractors.append(final_state)

        if (idx + 1) % report_interval == 0 or (idx + 1) == num_samples:
            print(
                f"[lyapunov]   processed {idx + 1}/{num_samples} samples "
                f"(unique attractors={len(attractors)})",
                flush=True,
            )

    if not attractors:
        print("[lyapunov] No attractors detected; returning empty array.", flush=True)
        return np.empty((0, samples.shape[1]), dtype=np.float32)

    stacked = np.stack(attractors, axis=0)
    print(
        f"[lyapunov] Attractor estimation complete: {len(attractors)} unique points.",
        flush=True,
    )
    return stacked


def _save_lyapunov_phase_portrait_comparison(
    model: KoopmanMachine,
    env: LyapunovMultiAttractor,
    path: Path,
    num_trajectories: int = 12,
    grid_lim: float = 3.0,
    grid_n: int = 15,
) -> Dict[str, str]:
    """Notebook-style comparison plot for the Lyapunov system with extras."""

    print("[lyapunov] Preparing phase portrait comparison...", flush=True)

    # Lazy imports for plotting and optional Voronoi
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from scipy.spatial import Voronoi  # type: ignore
        HAS_SCIPY = True
    except Exception:  # pragma: no cover - optional
        HAS_SCIPY = False

    device = next(model.parameters()).device
    dt = float(env.dt)

    # Colors per-attractor (tab20 like in the notebook)
    import matplotlib.cm as cm
    true_points = env.points.cpu().numpy()

    # Estimate learned attractors numerically to build Voronoi regions.
    learned_points = _estimate_learned_attractors(
        model=model,
        grid_lim=grid_lim,
        num_samples=min(max(grid_n**2, 64), 100),
        num_steps=max(int(8.0 / dt), 75),
        tolerance=0.2,
        device=device,
    )
    print(
        f"[lyapunov] Learned attractor candidates: {learned_points.shape if learned_points.size else (0, 2)}",
        flush=True,
    )

    produced_files: Dict[str, str] = {}
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, title, use_learned in (
        (axes[0], "True System", False),
        (axes[1], "Learned System", True),
    ):
        print(f"[lyapunov] Rendering '{title}' panel (use_learned={use_learned})", flush=True)
        display_points = (
            learned_points if use_learned and learned_points.size > 0 else true_points
        )
        num_points = max(len(display_points), 1)
        colors = cm.tab20(np.linspace(0, 1, num_points))

        # Optional Voronoi regions for both systems (true + learned estimate)
        if HAS_SCIPY and len(display_points) >= 3:
            # Filter display_points to only include finite, reasonable values
            valid_mask = np.isfinite(display_points).all(axis=1) & (np.abs(display_points).max(axis=1) < 1e6)
            valid_points = display_points[valid_mask]
            
            if len(valid_points) >= 3:
                try:
                    # Use 'QJ' option to joggle input for numerical stability
                    vor = Voronoi(valid_points, qhull_options='QJ')
                    for i, point_idx in enumerate(vor.point_region):
                        region = vor.regions[point_idx]
                        if not region or -1 in region:
                            continue
                        verts = np.array([vor.vertices[j] for j in region])
                        if len(verts) > 0:
                            ax.fill(
                                verts[:, 0],
                                verts[:, 1],
                                color=colors[i % len(colors)],
                                alpha=0.2 if use_learned else 0.25,
                                zorder=1,
                            )
                    for simplex in vor.ridge_vertices:
                        simplex = np.asarray(simplex)
                        if np.all(simplex >= 0):
                            ax.plot(
                                vor.vertices[simplex, 0],
                                vor.vertices[simplex, 1],
                                'k-',
                                linewidth=1.0,
                                alpha=0.7 if use_learned else 0.8,
                                zorder=2,
                            )
                except Exception as e:
                    print(f"[lyapunov] Voronoi failed for {title}, skipping regions: {e}", flush=True)

        # Grid and vector field (approximate using one-step delta / dt)
        xs = np.linspace(-grid_lim, grid_lim, grid_n)
        ys = np.linspace(-grid_lim, grid_lim, grid_n)
        X, Y = np.meshgrid(xs, ys)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(grid_n):
            for j in range(grid_n):
                state_np = np.array([X[i, j], Y[i, j]], dtype=np.float32)
                state_t = torch.from_numpy(state_np)
                if use_learned:
                    with torch.no_grad():
                        nx = model.step_env(state_t.to(device).unsqueeze(0)).squeeze(0).cpu()
                else:
                    nx = env.step(state_t)
                vel = (nx - state_t) / dt
                U[i, j], V[i, j] = float(vel[0].item()), float(vel[1].item())

        magnitudes = np.sqrt(U**2 + V**2)
        scale_den = np.where(magnitudes == 0, 1.0, magnitudes)
        U_n, V_n = U / scale_den, V / scale_den
        max_mag = float(magnitudes.max()) if magnitudes.size else 0.0
        linewidths = (
            0.75 + 2.25 * (magnitudes / (max_mag + 1e-6))
            if max_mag > 0
            else np.full_like(magnitudes, 0.75)
        )
        ax.quiver(
            X,
            Y,
            U_n,
            V_n,
            color='gray',
            alpha=0.65,
            scale=25,
            linewidths=linewidths.ravel(),
            zorder=3,
        )

        hist_suffix = "learned" if use_learned else "true"
        hist_path = path.parent / f"phase_portrait_vector_hist_{hist_suffix}.png"
        print(
            f"[lyapunov] Saving vector magnitude histogram ({hist_suffix}) to {hist_path}",
            flush=True,
        )
        _save_vector_magnitude_histogram(
            magnitudes,
            hist_path,
            title=f"{title} vector magnitudes",
        )
        produced_files[f"phase_portrait_vector_hist_{hist_suffix}"] = str(hist_path)

        marker_style = 's' if use_learned else 'o'
        for k, p in enumerate(display_points):
            ax.plot(
                p[0],
                p[1],
                marker_style,
                color=colors[k % len(colors)],
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=2,
                zorder=6,
            )

        # Simulate trajectories from random initial conditions
        rng = np.random.default_rng(42)
        comparison_points = display_points if len(display_points) > 0 else true_points
        for _ in range(num_trajectories):
            x0 = rng.uniform(-2.5, 2.5, size=2).astype(np.float32)
            state = torch.from_numpy(x0)
            traj = [x0.copy()]
            steps = int(8.0 / dt)
            for _step in range(steps):
                if use_learned:
                    with torch.no_grad():
                        state = model.step_env(state.to(device).unsqueeze(0)).squeeze(0).cpu()
                else:
                    state = env.step(state)
                traj.append(state.numpy().copy())

            traj_arr = np.asarray(traj)
            final = traj_arr[-1]
            dists = np.linalg.norm(comparison_points - final, axis=1)
            idx = int(np.argmin(dists))
            color = colors[idx % len(colors)]
            ax.plot(traj_arr[:, 0], traj_arr[:, 1], color=color, lw=2.0, alpha=0.9, zorder=4)
            ax.plot(
                x0[0],
                x0[1],
                marker_style,
                color=color,
                markersize=6,
                alpha=0.9,
                markeredgecolor='white',
                markeredgewidth=1,
                zorder=5,
            )

        ax.set_xlim(-grid_lim, grid_lim)
        ax.set_ylim(-grid_lim, grid_lim)
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_title(
            title if not use_learned else f"{title} (Voronoi est.)",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[lyapunov] Phase portrait comparison saved to {path}", flush=True)

    produced_files["phase_portrait_comparison"] = str(path)
    return produced_files

# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


def _make_km_env_n_step(
    model: KoopmanMachine,
    x: torch.Tensor,
    length: int,
    reencode_at_every: int,
    *,
    use_dynamics_prior: bool = False,
) -> torch.Tensor:
    """Torch analogue of notebooks/koopman_copy.py::make_km_env_n_step."""
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        if reencode_at_every == 1:
            traj, _ = _rollout_periodic_reencode_with_diagnostics(
                model,
                x,
                length,
                period=1,
                use_dynamics_prior=use_dynamics_prior,
            )
            return traj.detach().cpu()
        elif reencode_at_every == 0:
            latents = []
            latent = model.encode(x)
            for _ in range(length):
                latent = model.step_latent(latent)
                latents.append(latent)

            latents_stack = torch.stack(latents, dim=0)
            return model.decode(latents_stack).detach().cpu()
        else:
            traj, _ = _rollout_periodic_reencode_with_diagnostics(
                model,
                x,
                length,
                period=reencode_at_every,
                use_dynamics_prior=use_dynamics_prior,
            )
            return traj.detach().cpu()

    raise RuntimeError("Failed to generate Koopman rollout")


def _save_jax_style_phase_portraits(
    model: KoopmanMachine,
    base_env,
    cfg: Config,
    settings: "EvaluationSettings",
    path: Path,
    plot_dim: int = 2,
) -> None:
    """Replicate notebooks/koopman_copy.py phase-portrait generation exactly."""
    if base_env.observation_size < 2:
        return
    if plot_dim == 3 and base_env.observation_size < 3:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    batch_size = settings.phase_portrait_batch_size
    length = settings.phase_portrait_length
    reencode_periods = settings.phase_portrait_reencode_periods

    vec_env = VectorWrapper(base_env, batch_size)
    rng = torch.Generator().manual_seed(cfg.SEED + settings.seed_offset + 999)
    init_states = vec_env.reset(rng)  # CPU tensor

    trajectories = {}
    for period in reencode_periods:
        traj = _make_km_env_n_step(
            model,
            init_states,
            length,
            period,
            use_dynamics_prior=settings.use_dynamics_prior,
        )
        trajectories[period] = traj  # [length, batch, obs_dim] on CPU

    num_modes = len(reencode_periods)
    if plot_dim == 3:
        fig = plt.figure(figsize=(6 * num_modes, 5))
        axes = [[fig.add_subplot(1, num_modes, idx + 1, projection="3d") for idx in range(num_modes)]]
    else:
        fig, axes = plt.subplots(
            1, num_modes, figsize=(6 * num_modes, 5), squeeze=False
        )

    for ax, period in zip(axes[0], reencode_periods):
        traj = trajectories[period]
        if plot_dim == 3 and traj.shape[-1] >= 3:
            for idx in range(traj.shape[1]):
                ax.plot(traj[:, idx, 0], traj[:, idx, 1], traj[:, idx, 2])
        else:
            ax.plot(traj[:, :, 0], traj[:, :, 1])
        if period == 0:
            title = "reencode [x]"
        elif period == 1:
            title = "reencode @ 1"
        else:
            title = f"reencode @ {period}"
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        if plot_dim == 3 and traj.shape[-1] >= 3:
            ax.set_zlabel("x3")
        else:
            ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


@dataclass
class EvaluationSettings:
    """Container for evaluation hyper-parameters."""

    systems: Sequence[str] = (
        # "pendulum",
        "duffing",
        # "lotka_volterra",
        # "lorenz63",
        # "parabolic",
        "lyapunov",
    )
    horizons: Sequence[int] = tuple(range(100, 1001, 100))
    periodic_reencode_periods: Sequence[int] = (10, 25, 50, 100)
    batch_size: int = 100
    phase_portrait_samples: int = 20
    phase_portrait_length: int = 200
    phase_portrait_reencode_periods: Sequence[int] = (0, 1, 10, 25, 50)
    phase_portrait_batch_size: int = 256
    phase_portrait_dims: Sequence[int] = (2,)
    seed_offset: int = 12345
    # Dysts-specific extended reencode periods
    dysts_periodic_reencode_periods: Sequence[int] = (
        50,
        75,
        100,
        200,
        400,
        600,
        1000,
    )
    dysts_phase_portrait_reencode_periods: Sequence[int] = (0, 1, 100, 200, 300, 400, 500, 1000)
    use_dynamics_prior: bool = False
    event_trigger_proj_threshold: Optional[float] = None
    event_trigger_ambiguity_threshold: Optional[float] = None
    event_trigger_spillover_threshold: Optional[float] = None
    event_trigger_support_margin_min_ratio: Optional[float] = None
    event_trigger_support_threshold: float = 1e-3
    event_trigger_min_dwell: int = 0
    event_trigger_max_interval: int = 0
    save_rollout_artifacts: bool = False
    save_plots: bool = False
    include_per_ic_values: bool = False
    include_error_curves: bool = False


def _save_rollout_artifacts(
    *,
    path: Path,
    system: str,
    cfg: Config,
    settings: EvaluationSettings,
    periodic_periods: Sequence[int],
    init_states: torch.Tensor,
    true_future: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    mode_diagnostics: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    """Persist rollout tensors for downstream diagnosis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": system,
        "seed": int(cfg.SEED),
        "config": cfg.to_dict(),
        "evaluation_settings": {
            "systems": list(settings.systems),
            "horizons": list(settings.horizons),
            "periodic_reencode_periods": list(settings.periodic_reencode_periods),
            "dysts_periodic_reencode_periods": list(settings.dysts_periodic_reencode_periods),
            "use_dynamics_prior": bool(settings.use_dynamics_prior),
            "event_trigger_proj_threshold": (
                None
                if settings.event_trigger_proj_threshold is None
                else float(settings.event_trigger_proj_threshold)
            ),
            "event_trigger_ambiguity_threshold": (
                None
                if settings.event_trigger_ambiguity_threshold is None
                else float(settings.event_trigger_ambiguity_threshold)
            ),
            "event_trigger_spillover_threshold": (
                None
                if settings.event_trigger_spillover_threshold is None
                else float(settings.event_trigger_spillover_threshold)
            ),
            "event_trigger_support_margin_min_ratio": (
                None
                if settings.event_trigger_support_margin_min_ratio is None
                else float(settings.event_trigger_support_margin_min_ratio)
            ),
            "event_trigger_support_threshold": float(settings.event_trigger_support_threshold),
            "event_trigger_min_dwell": int(settings.event_trigger_min_dwell),
            "event_trigger_max_interval": int(settings.event_trigger_max_interval),
            "batch_size": int(settings.batch_size),
            "seed_offset": int(settings.seed_offset),
            "save_rollout_artifacts": bool(settings.save_rollout_artifacts),
            "save_plots": bool(settings.save_plots),
            "include_per_ic_values": bool(settings.include_per_ic_values),
            "include_error_curves": bool(settings.include_error_curves),
        },
        "periodic_periods_used": list(periodic_periods),
        "init_states": init_states.detach().cpu().contiguous(),
        "true_future": true_future.detach().cpu().contiguous(),
        "true_sequences": torch.cat(
            [init_states.detach().cpu().unsqueeze(0), true_future.detach().cpu()],
            dim=0,
        ).transpose(0, 1).contiguous(),
        "predictions": {
            mode_name: pred.detach().cpu().contiguous()
            for mode_name, pred in predictions.items()
        },
        "mode_diagnostics": {
            mode_name: {
                key: value.detach().cpu().contiguous()
                for key, value in diagnostics.items()
            }
            for mode_name, diagnostics in mode_diagnostics.items()
        },
    }
    torch.save(payload, path)


def evaluate_model(
    model: KoopmanMachine,
    cfg: Config,
    device: torch.device | str = "cuda",
    settings: Optional[EvaluationSettings] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Evaluate a trained Koopman model using the standardized protocol.

    Args:
        model: Trained Koopman machine.
        cfg: Configuration used during training (provides baseline hyper-params).
        device: Device for model inference.
        settings: Optional evaluation settings. Defaults to the research spec.
        output_dir: Optional path to save ``metrics.json`` and any explicitly
            requested artifacts or plots.

    Returns:
        Nested dictionary with metrics for each system and rollout mode.
    """

    if settings is None:
        settings = EvaluationSettings()

    print(
        f"[evaluate_model] Starting evaluation for systems={tuple(settings.systems)} "
        f"with horizons={tuple(settings.horizons)}",
        flush=True,
    )

    model = model.to(device)
    model.eval()

    max_horizon = max(settings.horizons)
    results: Dict[str, Dict] = {}

    for system in settings.systems:
        print(f"[evaluate_model] -> System '{system}': preparing environment...", flush=True)
        eval_cfg = Config.from_dict(cfg.to_dict())
        eval_cfg.ENV.ENV_NAME = system

        base_env = make_env(eval_cfg)
        if base_env.observation_size != model.observation_size:
            # Skip incompatible systems to avoid runtime errors
            print(
                f"[evaluate_model] -> System '{system}': skipped because "
                f"observation size {base_env.observation_size} != model {model.observation_size}",
                flush=True,
            )
            continue

        rng = torch.Generator().manual_seed(cfg.SEED + settings.seed_offset)
        use_cached_eval = (
            system.lower().startswith("dysts:")
            and bool(getattr(eval_cfg.ENV.DYSTS, "USE_NATIVE_CACHE", False))
        )
        if use_cached_eval:
            requested_split = str(getattr(eval_cfg.ENV.DYSTS, "CACHE_SPLIT", "")).strip().lower()
            if requested_split in ("", "train"):
                eval_cfg.ENV.DYSTS.CACHE_SPLIT = "test"
            print(
                f"[evaluate_model] -> System '{system}': loading cached ground-truth trajectories "
                f"(split={eval_cfg.ENV.DYSTS.CACHE_SPLIT}, batch={settings.batch_size}, horizon={max_horizon})",
                flush=True,
            )
            eval_cache = DystsTrajectoryCache(base_env.unwrapped, eval_cfg)
            seq = eval_cache.sample_sequence_batch(
                rng,
                batch_size=settings.batch_size,
                window_length=max_horizon,
                device=None,
            )
            init_states = seq[:, 0, :].contiguous()
            true_future = seq[:, 1:, :].transpose(0, 1).contiguous()
        else:
            vec_env = VectorWrapper(base_env, settings.batch_size)
            init_states = vec_env.reset(rng)  # CPU tensor

            # Generate ground truth trajectories (time-major)
            print(
                f"[evaluate_model] -> System '{system}': generating ground-truth trajectory "
                f"(batch={settings.batch_size}, horizon={max_horizon})",
                flush=True,
            )
            true_future = generate_trajectory(vec_env.step, init_states, length=max_horizon)

        # Prepare initial states on device for model rollout
        init_states_device = init_states.to(device)

        predictions: Dict[str, torch.Tensor] = {}
        mode_diagnostics: Dict[str, Dict[str, torch.Tensor]] = {}
        print(
            f"[evaluate_model] -> System '{system}': running rollout modes...",
            flush=True,
        )
        predictions["no_reencode"] = rollout_no_reencode(model, init_states_device, max_horizon)
        every_step_predictions, every_step_diagnostics = _rollout_periodic_reencode_with_diagnostics(
            model,
            init_states_device,
            max_horizon,
            period=1,
            use_dynamics_prior=settings.use_dynamics_prior,
        )
        predictions["every_step"] = every_step_predictions
        mode_diagnostics["every_step"] = every_step_diagnostics

        # Use extended reencode periods for dysts systems
        is_dysts = system.lower().startswith("dysts:")
        periodic_periods = (
            settings.dysts_periodic_reencode_periods if is_dysts
            else settings.periodic_reencode_periods
        )
        for period in periodic_periods:
            mode_name = f"periodic_{period}"
            periodic_predictions, periodic_diagnostics = _rollout_periodic_reencode_with_diagnostics(
                model,
                init_states_device,
                max_horizon,
                period=period,
                use_dynamics_prior=settings.use_dynamics_prior,
            )
            predictions[mode_name] = periodic_predictions
            mode_diagnostics[mode_name] = periodic_diagnostics

        if (
            settings.event_trigger_proj_threshold is not None
            or settings.event_trigger_ambiguity_threshold is not None
            or settings.event_trigger_spillover_threshold is not None
            or settings.event_trigger_support_margin_min_ratio is not None
            or settings.event_trigger_max_interval > 0
        ):
            mode_name = _event_trigger_mode_name(settings)
            event_predictions, event_diagnostics = _rollout_event_trigger_reencode_with_diagnostics(
                model,
                init_states_device,
                max_horizon,
                proj_threshold=(
                    None
                    if settings.event_trigger_proj_threshold is None
                    else float(settings.event_trigger_proj_threshold)
                ),
                ambiguity_threshold=(
                    None
                    if settings.event_trigger_ambiguity_threshold is None
                    else float(settings.event_trigger_ambiguity_threshold)
                ),
                spillover_threshold=(
                    None
                    if settings.event_trigger_spillover_threshold is None
                    else float(settings.event_trigger_spillover_threshold)
                ),
                support_margin_min_ratio=(
                    None
                    if settings.event_trigger_support_margin_min_ratio is None
                    else float(settings.event_trigger_support_margin_min_ratio)
                ),
                support_threshold=float(settings.event_trigger_support_threshold),
                min_dwell=int(settings.event_trigger_min_dwell),
                max_interval=int(settings.event_trigger_max_interval),
                use_dynamics_prior=settings.use_dynamics_prior,
            )
            predictions[mode_name] = event_predictions
            mode_diagnostics[mode_name] = event_diagnostics

        mode_metrics: Dict[str, Dict] = {}
        periodic_summary: Dict[str, Dict[str, float]] = {str(h): {} for h in settings.horizons}
        best_reset_summary: Dict[str, Dict[str, float]] = {str(h): {} for h in settings.horizons}
        per_step_errors: Dict[str, torch.Tensor] = {}

        # Convert ground truth to match predictions for metric computation
        true_future_cpu = true_future.float()

        print(
            f"[evaluate_model] -> System '{system}': computing metrics for {len(predictions)} modes...",
            flush=True,
        )
        for mode_name, pred in predictions.items():
            pred_cpu = pred.detach().cpu().float()

            obs_dim = max(1, int(true_future_cpu.shape[-1]))
            obs_dim_sqrt = math.sqrt(float(obs_dim))

            l2_error = torch.norm(pred_cpu - true_future_cpu, dim=-1)
            per_step_error = l2_error.mean(dim=1)
            per_step_error_per_dim = (l2_error / obs_dim_sqrt).mean(dim=1)

            # Raw MSE sums squared error over state dimensions.
            squared_diff = torch.sum((pred_cpu - true_future_cpu) ** 2, dim=-1)
            squared_diff = torch.where(torch.isfinite(squared_diff), squared_diff, torch.nan)
            # Per-dimension MSE divides the raw state-summed value once.
            squared_diff_per_dim = squared_diff / float(obs_dim)
            squared_diff_per_dim = torch.where(
                torch.isfinite(squared_diff_per_dim), squared_diff_per_dim, torch.nan
            )
            l2_error_per_dim = torch.where(torch.isfinite(l2_error), l2_error / obs_dim_sqrt, torch.nan)

            horizons_metrics = {}
            for horizon in settings.horizons:
                if system == "parabolic" and horizon > 100:
                    # Skip 1000-step metric for parabolic attractor
                    continue

                mean, std, per_ic, num_valid = _compute_horizon_metric_stats(squared_diff, horizon)
                per_dim_mean, per_dim_std, per_dim_ic, per_dim_num_valid = _compute_horizon_metric_stats(
                    squared_diff_per_dim, horizon
                )
                rmse_per_dim_mean, rmse_per_dim_std, rmse_per_dim_ic, rmse_per_dim_num_valid = (
                    _compute_horizon_metric_stats(l2_error_per_dim, horizon)
                )
                finite_coverage = _finite_coverage_stats(pred_cpu, horizon)
                num_initial_conditions = int(finite_coverage["num_initial_conditions"])
                horizon_metrics = {
                    "mean": mean,
                    "std": std,
                    "num_valid": num_valid,
                    "num_valid_fraction": (
                        float(num_valid) / float(num_initial_conditions)
                        if num_initial_conditions > 0
                        else float("nan")
                    ),
                    "per_dim_mean": per_dim_mean,
                    "per_dim_std": per_dim_std,
                    "per_dim_num_valid": per_dim_num_valid,
                    "per_dim_num_valid_fraction": (
                        float(per_dim_num_valid) / float(num_initial_conditions)
                        if num_initial_conditions > 0
                        else float("nan")
                    ),
                    "rmse_per_dim_mean": rmse_per_dim_mean,
                    "rmse_per_dim_std": rmse_per_dim_std,
                    "rmse_per_dim_num_valid": rmse_per_dim_num_valid,
                    "rmse_per_dim_num_valid_fraction": (
                        float(rmse_per_dim_num_valid) / float(num_initial_conditions)
                        if num_initial_conditions > 0
                        else float("nan")
                    ),
                    **finite_coverage,
                }
                if settings.include_per_ic_values:
                    horizon_metrics.update(
                        {
                            "values": per_ic,
                            "per_dim_values": per_dim_ic,
                            "rmse_per_dim_values": rmse_per_dim_ic,
                        }
                    )
                horizons_metrics[str(horizon)] = horizon_metrics

                if mode_name.startswith("periodic_") and num_valid > 0:
                    periodic_summary[str(horizon)][mode_name] = mean

            mode_entry = {"horizons": horizons_metrics}
            if settings.include_error_curves or settings.save_plots:
                mode_entry.update(
                    {
                        "mse_curve": _cumulative_mse_curve(squared_diff),
                        "mse_curve_per_dim": _cumulative_mse_curve(squared_diff_per_dim),
                        "l2_error_curve": per_step_error.cpu().tolist(),
                        "l2_error_curve_per_dim": per_step_error_per_dim.cpu().tolist(),
                    }
                )
            mode_metrics[mode_name] = mode_entry
            if settings.save_plots:
                per_step_errors[mode_name] = per_step_error
            diagnostics = mode_diagnostics.get(mode_name)
            if diagnostics:
                reset_mask = diagnostics["reset_mask"].float()
                mode_metrics[mode_name]["reset_rate_mean"] = float(reset_mask.mean().item())
                mode_metrics[mode_name]["reset_count_mean"] = float(reset_mask.sum(dim=0).mean().item())
                diagnostic_summary = {
                    "projection_gap": "projection_gap_mean",
                    "ambiguity_score": "ambiguity_score_mean",
                    "spillover_score": "spillover_score_mean",
                    "support_margin_ratio": "support_margin_ratio_mean",
                }
                for diagnostic_key, metric_key in diagnostic_summary.items():
                    values = diagnostics[diagnostic_key]
                    finite_values = values[torch.isfinite(values)]
                    mode_metrics[mode_name][metric_key] = (
                        float(finite_values.mean().item()) if finite_values.numel() > 0 else None
                    )

                trigger_summary = {
                    "threshold_trigger_mask": "threshold_trigger_rate_mean",
                    "interval_trigger_mask": "interval_trigger_rate_mean",
                    "proj_trigger_mask": "proj_trigger_rate_mean",
                    "ambiguity_trigger_mask": "ambiguity_trigger_rate_mean",
                    "spillover_trigger_mask": "spillover_trigger_rate_mean",
                    "support_margin_trigger_mask": "support_margin_trigger_rate_mean",
                }
                for trigger_key, metric_key in trigger_summary.items():
                    trigger_mask = diagnostics[trigger_key].float()
                    mode_metrics[mode_name][metric_key] = float(trigger_mask.mean().item())

        # Determine best periodic reencoding period per horizon
        best_periodic: Dict[str, Dict[str, float]] = {}
        for horizon in settings.horizons:
            horizon_key = str(horizon)
            if system == "parabolic" and horizon > 100:
                continue

            candidates = periodic_summary[horizon_key]
            if not candidates:
                continue

            best_mode = min(candidates.items(), key=lambda item: item[1])
            best_mode_name = best_mode[0]
            best_horizon_metrics = mode_metrics.get(best_mode_name, {}).get("horizons", {}).get(horizon_key, {})
            best_periodic[horizon_key] = {
                "mode": best_mode_name,
                "mean": best_mode[1],
                "per_dim_mean": best_horizon_metrics.get("per_dim_mean"),
                "rmse_per_dim_mean": best_horizon_metrics.get("rmse_per_dim_mean"),
                "num_valid_fraction": best_horizon_metrics.get("num_valid_fraction"),
                "full_horizon_finite_fraction": best_horizon_metrics.get("full_horizon_finite_fraction"),
                "finite_step_fraction": best_horizon_metrics.get("finite_step_fraction"),
                "num_full_horizon_finite": best_horizon_metrics.get("num_full_horizon_finite"),
                "median_finite_prefix_length": best_horizon_metrics.get("median_finite_prefix_length"),
                "min_finite_prefix_length": best_horizon_metrics.get("min_finite_prefix_length"),
            }

            if best_mode_name in mode_metrics:
                best_reset_summary[horizon_key][best_mode_name] = best_mode[1]

        for horizon in settings.horizons:
            horizon_key = str(horizon)
            if system == "parabolic" and horizon > 100:
                continue

            for mode_name, mode_data in mode_metrics.items():
                if mode_name == "no_reencode":
                    continue
                horizon_metrics = mode_data.get("horizons", {}).get(horizon_key)
                if horizon_metrics is None:
                    continue
                mean_value = _safe_float_for_best(horizon_metrics.get("mean"))
                if mean_value is None:
                    continue
                best_reset_summary[horizon_key][mode_name] = mean_value

        best_reset: Dict[str, Dict[str, float]] = {}
        for horizon in settings.horizons:
            horizon_key = str(horizon)
            if system == "parabolic" and horizon > 100:
                continue
            candidates = best_reset_summary[horizon_key]
            if not candidates:
                continue
            best_mode_name, best_mean = min(candidates.items(), key=lambda item: item[1])
            best_horizon_metrics = mode_metrics.get(best_mode_name, {}).get("horizons", {}).get(horizon_key, {})
            best_reset[horizon_key] = {
                "mode": best_mode_name,
                "mean": best_mean,
                "per_dim_mean": best_horizon_metrics.get("per_dim_mean"),
                "rmse_per_dim_mean": best_horizon_metrics.get("rmse_per_dim_mean"),
                "num_valid_fraction": best_horizon_metrics.get("num_valid_fraction"),
                "full_horizon_finite_fraction": best_horizon_metrics.get("full_horizon_finite_fraction"),
                "finite_step_fraction": best_horizon_metrics.get("finite_step_fraction"),
                "num_full_horizon_finite": best_horizon_metrics.get("num_full_horizon_finite"),
                "median_finite_prefix_length": best_horizon_metrics.get("median_finite_prefix_length"),
                "min_finite_prefix_length": best_horizon_metrics.get("min_finite_prefix_length"),
            }

        # Save qualitative plots when requested
        files: Dict[str, str] = {}
        if output_dir is not None:
            system_dir = output_dir / system
            if settings.save_rollout_artifacts or settings.save_plots:
                system_dir.mkdir(parents=True, exist_ok=True)

            if settings.save_rollout_artifacts:
                artifact_path = system_dir / "rollout_artifacts.pt"
                _save_rollout_artifacts(
                    path=artifact_path,
                    system=system,
                    cfg=eval_cfg,
                    settings=settings,
                    periodic_periods=periodic_periods,
                    init_states=init_states,
                    true_future=true_future_cpu,
                    predictions=predictions,
                    mode_diagnostics=mode_diagnostics,
                )
                files["rollout_artifacts"] = str(artifact_path)

            if settings.save_plots:
                print(
                    f"[evaluate_model] -> System '{system}': saving plots to {system_dir}",
                    flush=True,
                )

                # DYSTS: phase portraits should be long and include 3D by default.
                # The qualitative plots are the main debugging signal for chaotic flows.
                # Note: 30000 steps can be heavy; we also reduce the default portrait batch size
                # if it's set very large to avoid excessive memory usage.
                is_dysts = system.lower().startswith("dysts:")
                old_portrait_length = settings.phase_portrait_length
                old_portrait_dims = settings.phase_portrait_dims
                old_portrait_bs = settings.phase_portrait_batch_size
                old_portrait_reencode = settings.phase_portrait_reencode_periods
                if is_dysts:
                    settings.phase_portrait_length = 30000
                    settings.phase_portrait_dims = (2, 3)
                    if settings.phase_portrait_batch_size > 64:
                        settings.phase_portrait_batch_size = 32
                    # Use extended reencode periods for dysts phase portraits
                    settings.phase_portrait_reencode_periods = settings.dysts_phase_portrait_reencode_periods

                # JAX-style phase portrait grid (matches notebooks/koopman_copy.py)
                for plot_dim in settings.phase_portrait_dims:
                    suffix = "2D" if plot_dim == 2 else "3D"
                    portrait_path = system_dir / f"phase_portrait_plot_eval_{suffix}.png"
                    _save_jax_style_phase_portraits(
                        model=model,
                        base_env=base_env,
                        cfg=cfg,
                        settings=settings,
                        path=portrait_path,
                        plot_dim=plot_dim,
                    )
                    files[f"phase_portrait_plot_eval_{suffix}"] = str(portrait_path)

                # Restore settings (avoid cross-system contamination)
                if is_dysts:
                    settings.phase_portrait_length = old_portrait_length
                    settings.phase_portrait_dims = old_portrait_dims
                    settings.phase_portrait_batch_size = old_portrait_bs
                    settings.phase_portrait_reencode_periods = old_portrait_reencode

                curves = {
                    mode: data["mse_curve"]
                    for mode, data in mode_metrics.items()
                }
                curve_path = system_dir / "mse_vs_horizon.png"
                _save_mse_curve_plot(curves, curve_path, settings.horizons)
                files["mse_curve"] = str(curve_path)

                selected_modes = [
                    mode
                    for mode in ("every_step", "periodic_10", "periodic_25")
                    if mode in mode_metrics
                ]
                if selected_modes:
                    horizon_mse_path = system_dir / "horizon_mse_selected.png"
                    _save_horizon_mse_plot(
                        mode_metrics,
                        settings.horizons,
                        selected_modes,
                        horizon_mse_path,
                    )
                    files["horizon_mse_selected"] = str(horizon_mse_path)

                # Per-mode error curves (analogous to notebook plot_eval)
                for mode_name, errors in per_step_errors.items():
                    error_path = system_dir / f"error_curve_{mode_name}.png"
                    _save_error_curve_single_mode(
                        errors,
                        error_path,
                        title=f"Per-step error ({mode_name})",
                    )
                    files[f"error_curve_{mode_name}"] = str(error_path)

                combined_error_path = system_dir / "error_curve_combined.png"
                _save_error_curve_combined(
                    per_step_errors,
                    combined_error_path,
                    highlight_steps=settings.horizons,
                )
                files["error_curve_combined"] = str(combined_error_path)

                # Additional notebook-style comparison for Lyapunov system
                if system == "lyapunov":
                    try:
                        lyap_env = make_env(eval_cfg)
                        comp_path = system_dir / "phase_portrait_comparison.png"
                        print(
                            "[evaluate_model] -> System 'lyapunov': generating comparison + hist plots...",
                            flush=True,
                        )
                        lyap_files = _save_lyapunov_phase_portrait_comparison(
                            model,
                            lyap_env,
                            comp_path,
                        )
                        files.update(lyap_files)
                    except Exception as e:  # pragma: no cover - visualization best-effort
                        # Don't fail evaluation if visualization fails
                        print(f"[warn] Lyapunov comparison plot failed: {e}")

        results[system] = {
            "modes": mode_metrics,
            "best_periodic": best_periodic,
            "best_reset": best_reset,
            "files": files,
        }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(results, f, indent=2)
        results["metrics_file"] = str(metrics_path)

    print("[evaluate_model] Finished evaluation for all requested systems.", flush=True)
    return results


__all__ = [
    "EvaluationSettings",
    "evaluate_model",
    "rollout_every_step_reencode",
    "rollout_event_trigger_reencode",
    "rollout_no_reencode",
    "rollout_periodic_reencode",
]
