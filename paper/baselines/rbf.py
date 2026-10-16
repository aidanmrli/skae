"""Gaussian radial basis function (RBF) dictionary of the RBF EDMD baselines on both benchmarks."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch

BANDWIDTH_SAMPLE = 1024  # pooled states whose pair distances set the bandwidth
BANDWIDTH_PAIRS = 8192


def rbf_dictionary(rows: Callable[[np.ndarray], np.ndarray], total: int, count: int,
                   seed: int) -> Tuple[np.ndarray, float]:
    """Centers ``[count, d]`` and bandwidth drawn by ``default_rng(seed)`` from ``total`` pooled states.

    ``rows(indices)`` returns the standardized pooled states at ``indices``. One
    generator draws the centers, then a 1,024-state sample and 8,192 pairs of
    it; the bandwidth is the median pair distance (at least 1e-8).
    """
    rng = np.random.default_rng(seed)
    centers = rows(rng.choice(total, size=count, replace=total < count))
    sample = rows(rng.choice(total, size=min(BANDWIDTH_SAMPLE, total), replace=False))
    pairs = rng.integers(0, len(sample), size=(min(BANDWIDTH_PAIRS, len(sample) ** 2), 2))
    bandwidth = float(np.median(np.linalg.norm(sample[pairs[:, 0]] - sample[pairs[:, 1]], axis=1)))
    return centers, max(bandwidth, 1e-8)


def rbf_features(states, centers: np.ndarray, bandwidth: float):
    """Features ``[1, x, exp(-|x - c|^2 / (2 bandwidth^2)) for each center c]`` of standardized states.

    ``states`` is a NumPy array or a torch tensor ``[B, d]``; the result has the same type.
    """
    if isinstance(states, torch.Tensor):
        centers = torch.as_tensor(centers, dtype=states.dtype, device=states.device)
        squared = torch.square(states[:, None, :] - centers[None, :, :]).sum(dim=2)
        radial = torch.exp(-squared / (2.0 * bandwidth ** 2))
        return torch.cat((torch.ones_like(states[:, :1]), states, radial), dim=1)
    squared = np.square(states[:, None, :] - centers[None, :, :]).sum(axis=2)
    radial = np.exp(-squared / (2.0 * bandwidth ** 2))
    return np.concatenate((np.ones((len(states), 1)), states, radial), axis=1)
