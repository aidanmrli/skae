"""Forecast rollouts and the paper's forecasting metrics.

Rollout modes (all use only the model's own predictions):

* ``direct``      ``x_hat_{t+h} = D(E(x_t) K^h)`` -- no re-encoding;
* ``periodic_m``  advance ``K`` for ``m`` steps, decode, re-encode the decoded
  prediction and continue (``m = 1`` is ``every_step``).

These are :meth:`skae.KoopmanAutoencoder.forecast` plus per-start divergence
handling: from its first non-finite prediction, or its first prediction farther
out than ``DIVERGENCE_RADIUS`` x the largest training-state norm, a start is
forecast by the training-mean state and counted in a divergence rate. For
horizon ``H`` a start's error is the mean over ``h <= H`` of the squared error
``||x_hat - x||^2`` (summed over state dimensions); the per-run score is the
mean over starts. One re-encoding period is chosen per run on validation starts
(:func:`select_period`), and the test error at that period is reported as
``selected_periodic``.

:func:`load_run_model` loads a trained run for evaluation and every analysis;
:func:`fallback_and_radius` computes the divergence settings from training states.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

from paper import config
from paper.config import EvalConfig, RunConfig
from paper.systems import make_system
from skae.models import KoopmanAutoencoder

# The reported checkpoint: the weights after the last update.
PRIMARY_CHECKPOINT = "final.pt"


class TrainedRun(NamedTuple):
    """A trained run, as :func:`load_run_model` returns it."""

    model: KoopmanAutoencoder  # in eval mode
    cfg: RunConfig
    system: object             # paper.systems.make_system(cfg.benchmark, cfg.system)
    step: int                  # the last update
    fallback: torch.Tensor     # [d] the training-mean state, which forecasts a start after it diverges
    radius: float              # a prediction farther out (DIVERGENCE_RADIUS x the largest training-state norm) diverges


def load_run_model(run_dir: Path, device: str = "cpu") -> TrainedRun:
    """A run's reported model (``final.pt``) with its configuration, system and divergence settings."""
    payload = torch.load(Path(run_dir) / PRIMARY_CHECKPOINT, map_location=device, weights_only=True)
    cfg = RunConfig.from_dict(payload["config"])
    system = make_system(cfg.benchmark, cfg.system)
    model = KoopmanAutoencoder(cfg.model, system.dim)
    model.load_state_dict(payload["model"])
    return TrainedRun(model.to(device).eval(), cfg, system, int(payload["step"]), payload["state_mean"],
                      float(config.DIVERGENCE_RADIUS * payload["state_max_norm"]))


def fallback_and_radius(train_trajectories: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """The divergence settings of training states ``[..., d]``: the training-mean state ``[d]``, which forecasts a
    start after it diverges, and the radius ``DIVERGENCE_RADIUS`` x the largest training-state norm."""
    states = train_trajectories.reshape(-1, train_trajectories.shape[-1])
    return states.mean(dim=0), config.DIVERGENCE_RADIUS * float(torch.linalg.vector_norm(states, dim=-1).max().item())


@torch.no_grad()
def rollout_with_fallback(model, x0: torch.Tensor, horizon: int, period: Optional[int], fallback: torch.Tensor,
                          radius: float, frozen_support: bool = False):
    """Per-start divergence handling: ``(predictions [horizon, B, d], diverged_at [B])``.

    A start whose prediction becomes non-finite or has norm above ``radius``
    is forecast by ``fallback`` from that step on; ``diverged_at`` is that step
    (``horizon`` if never). With ``frozen_support`` (the re-encoding ablation),
    every re-encoded code keeps only the coordinates active in the start's code
    (``|z_i| > 1e-3``).
    """
    model.eval()
    fallback = fallback.to(device=x0.device, dtype=x0.dtype)
    advance = model.latent_stepper()
    latent = model.encode(x0)
    support = latent.abs() > config.SUPPORT_THRESHOLD
    dead = torch.zeros(x0.shape[0], dtype=torch.bool, device=x0.device)
    diverged_at = torch.full((x0.shape[0],), horizon, dtype=torch.long, device=x0.device)
    predictions: List[torch.Tensor] = []
    for step in range(horizon):
        latent = advance(latent)
        x_hat = model.decode(latent)
        bad = ~torch.isfinite(x_hat).all(dim=-1) | (torch.linalg.vector_norm(x_hat, dim=-1) > radius)
        newly = bad & ~dead
        diverged_at = torch.where(newly, torch.full_like(diverged_at, step), diverged_at)
        dead = dead | newly
        x_out = torch.where(dead[:, None], fallback, x_hat)
        predictions.append(x_out)
        if period is not None and (step + 1) % period == 0:
            latent = model.encode(x_out)
            if frozen_support:
                latent = latent * support.to(latent.dtype)
        latent = torch.where(dead[:, None], torch.zeros_like(latent), latent)
    return torch.stack(predictions, dim=0), diverged_at


def horizon_statistics(values: torch.Tensor, horizon: int) -> Dict[str, object]:
    """Statistics over starts of the per-step errors ``values`` ``[steps, B]`` up to ``horizon``.

    A start's score is the mean of its non-NaN errors over the first ``horizon``
    steps; starts without a finite score are left out. Returns ``mean`` and
    ``std`` (population) over the kept starts, their number ``num_valid`` and
    their scores ``values`` ``[num_valid]``.
    """
    per_start = torch.nanmean(values[:horizon], dim=0)
    valid = torch.isfinite(per_start)
    if valid.sum() == 0:
        return {"mean": float("nan"), "std": float("nan"), "num_valid": 0, "values": []}
    kept = per_start[valid]
    return {
        "mean": kept.mean().item(),
        "std": kept.std(unbiased=False).item() if kept.numel() > 1 else 0.0,
        "num_valid": int(valid.sum().item()),
        "values": kept.tolist(),
    }


def horizon_scores(predictions: torch.Tensor, diverged_at: torch.Tensor, truth: torch.Tensor,
                   horizons: Sequence[int]) -> Dict[str, Dict[str, object]]:
    """Per horizon ``H`` (key ``str(H)``): :func:`horizon_statistics` of the squared error and the divergence rate."""
    predictions, diverged_at = predictions.cpu().float(), diverged_at.cpu()
    squared = torch.sum((predictions - truth[:predictions.shape[0]].float().cpu()) ** 2, dim=-1)
    squared = torch.where(torch.isfinite(squared), squared, torch.nan)
    scores = {}
    for h in horizons:
        stats = horizon_statistics(squared, h)
        stats["divergence_rate"] = float((diverged_at < h).float().mean().item())
        scores[str(h)] = stats
    return scores


def forecast_metrics(model, starts: torch.Tensor, truth: torch.Tensor, cfg: EvalConfig, device: torch.device,
                     fallback: torch.Tensor, radius: float, horizons: Optional[Sequence[int]] = None,
                     periods: Optional[Sequence[int]] = None, selected_period: Optional[int] = None,
                     periodic_only: bool = False) -> Dict[str, object]:
    """Evaluate the rollout modes on held-out starts ``[B, d]`` with truth ``[H, B, d]``.

    ``fallback`` is the training-mean state that replaces a diverged start's
    predictions; ``radius`` is the norm beyond which a prediction also counts
    as diverged. Horizons and periods default to ``cfg``'s.

    Returns ``{"modes": {mode: {str(H): stats}}}`` for the modes ``direct``,
    ``every_step`` and ``periodic_<m>`` (only the last with ``periodic_only``);
    ``stats`` holds the keys of :func:`horizon_statistics` and
    ``divergence_rate``. With ``selected_period``, also ``selected_period`` and
    ``selected_periodic``: the ``stats`` of that period without ``values``.
    """
    horizons = tuple(horizons or cfg.horizons)
    periods = tuple(periods or cfg.periods)
    x0 = starts.to(device)
    modes = {} if periodic_only else {"direct": None, "every_step": 1}
    modes.update({f"periodic_{m}": m for m in periods})
    results: Dict[str, object] = {"modes": {}}
    for name, period in modes.items():
        predictions, diverged_at = rollout_with_fallback(model, x0, max(horizons), period, fallback, radius)
        results["modes"][name] = horizon_scores(predictions, diverged_at, truth, horizons)
    if selected_period is not None:
        chosen = results["modes"][f"periodic_{selected_period}"]
        results["selected_period"] = int(selected_period)
        results["selected_periodic"] = {h: {k: v for k, v in stats.items() if k != "values"}
                                        for h, stats in chosen.items()}
    return results


def select_period(model, starts: torch.Tensor, truth: torch.Tensor, cfg: EvalConfig, fallback: torch.Tensor,
                  device: torch.device, radius: float):
    """Re-encoding period with the lowest validation MSE at ``SELECTION_HORIZON``: ``(period, mse)``."""
    h = config.SELECTION_HORIZON
    results = forecast_metrics(model, starts, truth, cfg, device=device, fallback=fallback, radius=radius,
                               horizons=(h,), periodic_only=True)
    scores = {int(name.split("_")[1]): stats[str(h)]["mean"] for name, stats in results["modes"].items()}
    finite = {m: v for m, v in scores.items() if math.isfinite(v)}
    if not finite:
        return int(min(cfg.periods)), float("nan")
    period = min(finite, key=finite.get)
    return period, float(finite[period])
