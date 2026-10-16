"""Training windows from trajectories, and a small training loop."""

from __future__ import annotations

from typing import List, Optional

import torch

from skae.models import KoopmanAutoencoder, training_loss


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


def fit(
    model: KoopmanAutoencoder,
    windows: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> List[float]:
    """Train ``model`` on windows ``[W, L + 1, obs_dim]`` and return the mean loss of each epoch.

    Each epoch visits the windows in a random order drawn from a generator
    seeded with ``seed``, in minibatches of ``batch_size``, and takes one
    optimizer step on :func:`skae.training_loss` per minibatch. The optimizer
    is Adam with learning rate ``lr`` unless ``optimizer`` is given. Windows
    are moved to the model's device and dtype. Seed the model's
    initialization separately (``torch.manual_seed`` before building it).

    The paper's runs use the trainer in ``paper/training`` instead, which
    draws batches per step rather than per epoch and additionally uses AdamW
    parameter groups (a smaller learning rate and no weight decay for ``K``)
    and, on Dysts, a learning-rate schedule.
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
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    model.train()
    history = []
    for _ in range(epochs):
        total = 0.0
        for indices in torch.randperm(len(windows), generator=generator).split(batch_size):
            loss, _ = training_loss(model, windows[indices.to(windows.device)])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        history.append(total / len(windows))
    return history
