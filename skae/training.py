"""Training windows from trajectories, and a small training loop."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional

import torch

from skae.losses import SupportLossConfig, WindowSimilarity
from skae.models import KoopmanAutoencoder, training_loss

LR_SCHEDULES = ("constant", "cosine_tail")


def trajectory_windows(trajectories: torch.Tensor, window_length: int, stride: int = 1) -> torch.Tensor:
    """Cut trajectories ``[N, T, D]`` into overlapping windows ``[W, window_length + 1, D]``.

    A window is a state and the ``window_length`` states after it. Windows
    start every ``stride`` steps and never run from one trajectory into the
    next; they are ordered by trajectory, then by start time.
    """
    if trajectories.ndim != 3:
        raise ValueError(f"trajectories must have shape [N, T, D], got {tuple(trajectories.shape)}")
    if window_length < 1 or trajectories.shape[1] < window_length + 1:
        raise ValueError(f"windows of {window_length} steps need window_length >= 1 and trajectories of at "
                         f"least {window_length + 1} states, got {trajectories.shape[1]}")
    if not bool(torch.isfinite(trajectories).all()):
        raise ValueError("trajectories contain NaN or infinite values")
    windows = trajectories.unfold(1, window_length + 1, stride)  # [N, windows per trajectory, D, L + 1]
    return windows.transpose(-1, -2).reshape(-1, window_length + 1, trajectories.shape[-1])


def cosine_tail(step: int, steps: int, tail: float = 0.25, final: float = 0.01) -> float:
    """Learning-rate factor of update ``step`` of ``steps``: 1, then along a cosine over the last ``tail`` of the
    updates down to ``final``, reached at the last update when there are at least ``2 / tail`` updates."""
    phase = max(0.0, (step - (1.0 - tail) * steps) / max(1.0, tail * steps - 1.0))
    if phase <= 0.0:
        return 1.0
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, phase)))


def _calibrate(model: KoopmanAutoencoder, windows: torch.Tensor) -> WindowSimilarity:
    """:meth:`WindowSimilarity.calibrate` on up to 512 training windows ``[W, L + 1, obs_dim]``, normalized as
    the model normalizes them."""
    windows = windows.detach().to(device="cpu", dtype=torch.float32)
    if model.cfg.input_normalization:
        mean, scale = model._input_mean.detach().cpu().float(), model._input_scale.detach().cpu().float()
        return WindowSimilarity.calibrate(windows, mean, scale, window_states=windows.shape[1])
    return WindowSimilarity.calibrate(windows, window_states=windows.shape[1])


def fit(
    model: KoopmanAutoencoder,
    windows: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    koopman_lr: Optional[float] = None,
    lr_schedule: str = "constant",
    support: Optional[SupportLossConfig] = None,
) -> List[float]:
    """Train ``model`` on windows ``[W, L + 1, obs_dim]`` and return the mean loss of each epoch.

    Each epoch visits the windows in a random order drawn from a generator
    seeded with ``seed``, in minibatches of ``batch_size``, and takes one
    optimizer step on :func:`skae.training_loss` per minibatch. The optimizer
    is Adam with learning rate ``lr`` unless ``optimizer`` is given. Windows
    are moved to the model's device and dtype. Seed the model's
    initialization separately (``torch.manual_seed`` before building it).

    Options, off by default:

    * ``koopman_lr``: Adam's learning rate for ``K`` (``lr`` for the other
      parameters). The paper's runs trained ``K`` 10 times (multibasin) or 5
      times (Dysts) more slowly than the rest.
    * ``lr_schedule="cosine_tail"``: the learning rates stay constant for the
      first 75% of the updates, then decay along a cosine to 1%
      (:func:`cosine_tail`), as in the paper's Dysts runs; ``fit`` restores
      them when it returns.
    * ``support``: adds the support losses of :mod:`skae.losses` (LISTA needs
      ``lista_output="none"``). With a pair loss, the pair weights are
      calibrated once on the windows (:meth:`WindowSimilarity.calibrate`);
      a minibatch of one window gets no pair term.

    The paper's runs use the trainer in ``paper/training`` instead, which
    draws batches per step rather than per epoch and uses AdamW, with weight
    decay on the multibasin benchmark.
    """
    parameter = next(model.parameters())
    windows = windows.to(device=parameter.device, dtype=parameter.dtype)
    if windows.ndim != 3 or windows.shape[1] < 2 or windows.shape[2] != model.obs_dim:
        raise ValueError(f"windows must have shape [W, L + 1, {model.obs_dim}] with L >= 1, "
                         f"got {tuple(windows.shape)}")
    if len(windows) == 0:
        raise ValueError("windows is empty: fit needs at least one window")
    if not bool(torch.isfinite(windows).all()):
        raise ValueError("windows contain NaN or infinite values")
    if lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {lr_schedule!r}")
    if optimizer is None:
        if koopman_lr is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            koopman = {id(p) for p in model.koopman_parameters()}
            optimizer = torch.optim.Adam([
                {"params": [p for p in model.parameters() if id(p) not in koopman], "lr": lr},
                {"params": model.koopman_parameters(), "lr": koopman_lr},
            ])
    elif koopman_lr is not None:
        raise ValueError("pass koopman_lr or optimizer, not both")
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    pairs = support is not None and support.active and support.pair_weight > 0.0
    if pairs:
        unpaired = replace(support, pair_weight=0.0)  # for a minibatch of one window
        similarity = _calibrate(model, windows) if len(windows) > 1 else None
    generator = torch.Generator().manual_seed(seed)
    steps, step = epochs * math.ceil(len(windows) / batch_size), 0

    model.train()
    history = []
    for _ in range(epochs):
        total = 0.0
        for indices in torch.randperm(len(windows), generator=generator).split(batch_size):
            if lr_schedule == "cosine_tail":
                for group, base_lr in zip(optimizer.param_groups, base_lrs):
                    group["lr"] = base_lr * cosine_tail(step, steps)
            batch = windows[indices.to(windows.device)]
            if support is None:
                loss, _ = training_loss(model, batch)
            elif not pairs:
                loss, _ = training_loss(model, batch, support=support)
            elif len(batch) < 2:
                loss, _ = training_loss(model, batch, support=unpaired)
            else:
                loss, _ = training_loss(model, batch, support=support,
                                        pair_weights=similarity(model.normalize_observations(batch)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
            step += 1
        history.append(total / len(windows))
    if lr_schedule == "cosine_tail":
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
    return history
