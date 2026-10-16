"""Basin labels, support families and the high-margin state subset.

* Basin labels (evaluation only): each trajectory is labelled by the attractor
  it reaches (:func:`fate_labels_and_centers`).
* Support ``S(x) = {i : |z_i| > a}``. Distinct supports are visited in order of
  decreasing frequency and greedily merged into *support families* whenever
  their Jaccard similarity with an existing family prototype is at least ``j``
  (:func:`fit_support_families`); new supports join the most similar prototype
  (:func:`transfer_support_families`).
* High-margin states (:func:`high_margin_mask`): nearer their own basin center
  than any other, with an own-basin margin at or above the per-basin 75th
  percentile (ties kept).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from paper.data import reset_batch, rollout

ENDPOINT_ROLLOUT_STEPS = 5000
FATE_AVERAGE_STEPS = 500
MERGE_FRACTION = 0.05
_EPS = 1e-12


def reset_trajectories(system, base_seed: int, count: int, transitions: int) -> torch.Tensor:
    """``[count, transitions + 1, d]`` float32 trajectories from resets with seeds ``base_seed + i``."""
    starts = reset_batch(system.reset, int(base_seed), count)
    with torch.no_grad():
        path = torch.cat([starts.unsqueeze(0), rollout(system.step, starts, transitions)], dim=0)
    return path.transpose(0, 1).float()


def kmeans_centers(points: torch.Tensor, num_centers: int, num_iters: int = 25) -> torch.Tensor:
    """Centers ``[num_centers, d]`` of points ``[N, d]``: k-means seeded farthest-first from the first point.

    Deterministic; stops after ``num_iters`` updates or when the centers stop moving.
    """
    centers = [points[0]]
    while len(centers) < num_centers:
        distance = torch.cdist(points, torch.stack(centers, dim=0)).min(dim=1).values
        centers.append(points[distance.argmax()])
    current = torch.stack(centers, dim=0).clone()
    for _ in range(num_iters):
        assignment = torch.cdist(points, current).argmin(dim=1)
        updated = torch.stack(
            [points[assignment == k].mean(dim=0) if bool((assignment == k).any()) else current[k]
             for k in range(num_centers)],
            dim=0,
        )
        if torch.allclose(updated, current):
            break
        current = updated
    return current


def limit_points(system, trajectories: torch.Tensor) -> torch.Tensor:
    """``[N, d]``: mean of the final 500 of 5000 further steps from each trajectory's last state."""
    state = trajectories[:, -1, :].clone()
    tail = torch.zeros_like(state)
    for step in range(ENDPOINT_ROLLOUT_STEPS):
        state = system.step(state)
        if step >= ENDPOINT_ROLLOUT_STEPS - FATE_AVERAGE_STEPS:
            tail = tail + state
    return tail / FATE_AVERAGE_STEPS


def fate_centers(system, points: torch.Tensor) -> torch.Tensor:
    """Basin centers ``[K, d]``, ``K <= num_basins``: k-means of the limit points ``[N, d]`` with the system's basin
    count, then centers closer than 5% of the reset-box diagonal merge (the first one is kept).

    A system with fewer attractors than advertised therefore yields fewer labels.
    """
    centers = kmeans_centers(points, system.num_basins)
    low, high = system.reset_bounds()
    tolerance = MERGE_FRACTION * float(torch.linalg.vector_norm(high - low))
    merged: List[torch.Tensor] = []
    for center in centers:
        if all(float(torch.linalg.vector_norm(center - other)) > tolerance for other in merged):
            merged.append(center)
    return torch.stack(merged, dim=0)


def fate_labels(trajectories: torch.Tensor, points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """``[N, T]``: every state of a trajectory gets the center nearest its limit point."""
    fate = torch.cdist(points, centers).argmin(dim=1)
    return fate[:, None].expand(trajectories.shape[0], trajectories.shape[1]).long()


def fate_labels_and_centers(system, trajectories: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trajectory-fate labels ``[N, T]`` and centers ``[K, d]`` fitted on the same trajectories."""
    points = limit_points(system, trajectories)
    centers = fate_centers(system, points)
    return fate_labels(trajectories, points, centers), centers.to(dtype=trajectories.dtype)


def _support_keys(mask: np.ndarray) -> List[bytes]:
    packed = np.packbits(mask.astype(np.uint8), axis=-1)
    return [row.tobytes() for row in packed.reshape(-1, packed.shape[-1])]


def fit_support_families(masks: np.ndarray, min_jaccard: float) -> Tuple[np.ndarray, np.ndarray]:
    """Family prototypes ``[P, D]`` and labels ``masks.shape[:-1]`` of boolean supports ``[..., D]``.

    Distinct supports are visited by decreasing frequency (ties by packed bytes)
    and join the most similar prototype if the Jaccard similarity is at least
    ``min_jaccard`` (two empty supports: 1); otherwise they found a new family
    whose prototype they stay.
    """
    flat_masks = masks.reshape(-1, masks.shape[-1]).astype(bool)
    keys = _support_keys(flat_masks)
    counts = Counter(keys)
    first_mask: Dict[bytes, np.ndarray] = {}
    for key, mask in zip(keys, flat_masks):
        first_mask.setdefault(key, mask)
    prototypes = np.zeros((len(counts), flat_masks.shape[1]), dtype=np.float64)
    sizes = np.zeros(len(counts), dtype=np.float64)
    num_prototypes = 0
    family_of: Dict[bytes, int] = {}
    for key, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        mask = first_mask[key].astype(np.float64)
        size = float(mask.sum())
        if num_prototypes:
            intersection = prototypes[:num_prototypes] @ mask
            union = sizes[:num_prototypes] + size - intersection
            similarity = np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)
            best = int(np.argmax(similarity))
            if similarity[best] >= min_jaccard:
                family_of[key] = best
                continue
        family_of[key] = num_prototypes
        prototypes[num_prototypes], sizes[num_prototypes] = mask, size
        num_prototypes += 1
    labels = np.asarray([family_of[key] for key in keys], dtype=np.int64).reshape(masks.shape[:-1])
    return prototypes[:num_prototypes] > 0.5, labels


def transfer_support_families(masks: np.ndarray, prototypes: np.ndarray, min_jaccard: float) -> np.ndarray:
    """Most similar prototype of each support, or ``-1`` if its Jaccard similarity is below ``min_jaccard``.

    Counts are exact integers in float64, so the similarities equal those of
    :func:`fit_support_families` bit for bit.
    """
    flat_masks = masks.reshape(-1, masks.shape[-1]).astype(bool)
    _, first, inverse = np.unique(np.packbits(flat_masks, axis=-1), axis=0, return_index=True, return_inverse=True)
    unique, prototypes = flat_masks[first].astype(np.float64), prototypes.astype(np.float64)
    intersection = unique @ prototypes.T
    union = unique.sum(axis=1)[:, None] + prototypes.sum(axis=1)[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)
    best = similarity.argmax(axis=1)
    assigned = np.where(similarity[np.arange(best.size), best] >= min_jaccard, best, -1).astype(np.int64)
    return assigned[inverse.reshape(-1)].reshape(masks.shape[:-1])


def entropy(values: Sequence) -> float:
    """``H(values)`` in nats, with a ``1e-12`` offset inside the logarithm."""
    counter = Counter(values)
    total = float(sum(counter.values()))
    return -sum((c / total) * math.log(c / total + _EPS) for c in counter.values()) if total > 0 else 0.0


def conditional_entropy(x: Sequence, y: Sequence) -> float:
    """``H(x | y) = H(x, y) - H(y)`` in nats."""
    return entropy(list(zip(x, y))) - entropy(y)


def center_margins(states: torch.Tensor, centers: torch.Tensor, labels: np.ndarray) -> np.ndarray:
    """Per-state own-basin margin ``[N]`` (``+inf`` with fewer than two centers).

    The distance to the nearest center of another basin minus the distance to
    the state's own labelled center, computed elementwise in float64 (no
    matrix-product shortcut, so equal states get equal margins and the result
    does not depend on the CPU's vector width).
    """
    flat = states.reshape(-1, states.shape[-1])
    if centers.shape[0] < 2:
        return np.full(flat.shape[0], np.inf)
    diff = flat.to(torch.float64)[:, None, :] - centers.to(torch.float64)[None, :, :]
    distances = torch.sqrt((diff ** 2).sum(dim=-1)).cpu().numpy()
    flat_labels = np.asarray(labels).reshape(-1)
    if flat_labels.size and (flat_labels.min() < 0 or flat_labels.max() >= centers.shape[0]):
        raise ValueError("own-basin margins need every label to index a center")
    rows = np.arange(flat.shape[0])
    own = distances[rows, flat_labels]
    distances[rows, flat_labels] = np.inf
    return distances.min(axis=1) - own


def fit_margin_cutoffs(margins: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
    """``{label: cutoff}`` from margins ``[N]`` and labels (``N`` values): the per-label 75th percentile.

    The cutoff is ``-inf`` (keep every state) for a label with fewer than four
    states or a non-finite margin (one center).
    """
    flat_labels = np.asarray(labels).reshape(-1)
    cutoffs: Dict[int, float] = {}
    for label in np.unique(flat_labels):
        values = margins[flat_labels == label]
        finite = values.size >= 4 and bool(np.isfinite(values).all())
        cutoffs[int(label)] = float(np.quantile(values, 0.75)) if finite else -math.inf
    return cutoffs


def high_margin_mask(margins: np.ndarray, labels: np.ndarray, cutoffs: Dict[int, float]) -> np.ndarray:
    """Mask ``[N]`` of the states at or above their label's cutoff and nearer their own center than any other.

    Ties are kept and labels without a cutoff are dropped. Converged states are
    often exactly equal, so their margins tie and the subset can exceed a quarter.
    """
    flat_labels = np.asarray(labels).reshape(-1)
    threshold = np.asarray([cutoffs.get(int(label), math.inf) for label in flat_labels], dtype=np.float64)
    return (margins >= threshold) & (margins > 0)


def high_margin_slice(states: torch.Tensor, centers: torch.Tensor, labels: np.ndarray) -> np.ndarray:
    """Mask ``[N]`` of the high-margin states of ``states`` ``[..., d]`` (``N`` states, labels of the same
    shape), with the cutoffs fitted on the same states."""
    margins = center_margins(states, centers, labels)
    return high_margin_mask(margins, labels, fit_margin_cutoffs(margins, labels))
