"""Label-free window support losses.

For a window ``[B, T, d]`` of LISTA activities ``a`` (the margin ``u - t`` of
the last LISTA step, :class:`skae.models.LISTAEncoder`), the soft support is
``q = sigmoid((a - scale) / temperature)``, close to 1 on active coordinates
and 0 elsewhere. The terms are

* temporal:  ``mean_{b,k} (max_t q_btk - min_t q_btk)``, how much the support
  of a window changes over time;
* occupancy: ``mean q``;
* pair:      with ``u_b = mean_t q_b / ||mean_t q_b||``,
  ``c_bc = clamp(u_b . u_c, 0, 1)`` and detached weights ``w_bc`` in
  ``[0, 1]``, over the pairs ``b != c`` of the batch,
  ``sum w c / sum w + attraction * sum (1 - w)(1 - c) / sum (1 - w)``:
  dissimilar windows (``w`` near 1) are pushed to different supports and
  similar ones (``w`` near 0) pulled to the same support.

:class:`WindowSimilarity` computes ``w`` from the observed windows only.
:func:`skae.training_loss` adds ``temporal_weight * temporal +
occupancy_weight * occupancy + pair_weight * pair`` after its other terms.

The terms are computed in a fixed order and fixed dtypes, so that a run
reproduces bitwise on the same hardware. ``calibrate`` takes the raw training states with the
normalizer's ``mean`` and ``scale``, and :func:`support_terms` gathers the
three terms for ``training_loss``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class SupportLossConfig:
    """Weights and soft-support settings of the window support losses.

    Our experiments used ``temporal_weight=0.25`` and ``pair_weight=0.19``
    with ``reconstruction_weight=1`` and ``loss_time_reduction="mean"`` (the
    weights are relative to that loss); larger weights raised forecast errors.
    """

    temporal_weight: float = 0.0   # weight of the within-window support change
    pair_weight: float = 0.0       # weight of the geometry-weighted pair contrast
    occupancy_weight: float = 0.0  # weight of the mean soft support
    attraction: float = 1.0        # weight of the similar-pair (1 - c) part of the pair term
    scale: float = 1e-3            # activity at which the soft support is 1/2
    temperature: float = 0.01      # soft-support temperature

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"SupportLossConfig.{field.name} must be finite and nonnegative, got {value}")
        if self.scale <= 0.0 or self.temperature <= 0.0:
            raise ValueError("SupportLossConfig.scale and temperature must be positive")

    @property
    def active(self) -> bool:
        """Whether any term has a positive weight (otherwise the loss is the base loss)."""
        return self.temporal_weight > 0.0 or self.pair_weight > 0.0 or self.occupancy_weight > 0.0


def soft_support(activity: torch.Tensor, scale: float = 1e-3, temperature: float = 0.01) -> torch.Tensor:
    """``sigmoid((activity - scale) / temperature)``, elementwise."""
    return torch.sigmoid((activity - float(scale)) / float(temperature))


def temporal_range(q: torch.Tensor) -> torch.Tensor:
    """``mean(max_t q - min_t q)`` for soft supports ``q`` of shape ``[..., T, d]``, ``T >= 2``."""
    if q.ndim < 3 or q.shape[-2] < 2:
        raise ValueError(f"q must have shape [..., T, d] with T >= 2, got {tuple(q.shape)}")
    return (q.amax(dim=-2) - q.amin(dim=-2)).mean()


def pair_contrast(q: torch.Tensor, weights: torch.Tensor, attraction: float = 1.0) -> torch.Tensor:
    """The pair term of the module docstring for soft supports ``q`` ``[B, T, d]`` and weights ``[B, B]``.

    The diagonal of ``weights`` is ignored. Sums run in float64 and the result
    has the dtype of ``q``. A sum of zero weights drops its part (a zero with
    a gradient). With ``attraction = 0`` the weighted cosine is unclamped.
    """
    u = q.mean(dim=-2)
    u = u.reshape(-1, u.shape[-1])
    count = u.shape[0]
    if count < 2:
        raise ValueError("pair_contrast needs at least two windows")
    if weights.shape != (count, count):
        raise ValueError(f"weights must have shape ({count}, {count}), got {tuple(weights.shape)}")
    attraction = float(attraction)
    u64 = u.to(dtype=torch.float64)
    u64 = u64 / torch.linalg.vector_norm(u64, dim=1, keepdim=True).clamp_min(1e-8)
    weights64 = weights.detach().to(dtype=torch.float64)
    off_diagonal = 1.0 - torch.eye(count, dtype=weights64.dtype, device=weights64.device)
    weights64 = weights64 * off_diagonal
    if bool((weights64 < 0.0).any()) or (attraction > 0.0 and bool((weights64 > 1.0).any())):
        raise ValueError("pair weights must lie in [0, 1] (in [0, inf) with attraction = 0)")
    weight_sum = weights64.sum()
    cosine = u64 @ u64.transpose(0, 1)
    if attraction != 0.0:
        cosine = cosine.clamp(min=0.0, max=1.0)
    if float(weight_sum.item()) == 0.0:
        repulsion = u64.sum() * 0.0
    else:
        repulsion = (cosine * weights64).sum() / weight_sum
    if attraction == 0.0:
        return repulsion.to(dtype=q.dtype)
    complement = off_diagonal - weights64
    complement_sum = complement.sum()
    if float(complement_sum.item()) == 0.0:
        pull = u64.sum() * 0.0
    else:
        pull = ((1.0 - cosine) * complement).sum() / complement_sum
    return (repulsion + attraction * pull).to(dtype=q.dtype)


def support_terms(activity: torch.Tensor, support: SupportLossConfig,
                  pair_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """The unweighted ``temporal``, ``occupancy`` and ``pair`` terms for activities ``[B, T, d]``.

    ``pair`` is computed only with ``support.pair_weight > 0`` (then
    ``pair_weights`` is required) and is otherwise a constant zero. The pair
    term recomputes the soft support, which keeps the operation order (and
    so the bits) of earlier fits.
    """
    q = soft_support(activity, support.scale, support.temperature)
    terms = {"temporal": temporal_range(q), "occupancy": q.mean()}
    if support.pair_weight > 0.0:
        if pair_weights is None:
            raise ValueError("pair_weight > 0 needs pair_weights (see WindowSimilarity)")
        q = soft_support(activity, support.scale, support.temperature)
        terms["pair"] = pair_contrast(q, pair_weights, support.attraction)
    else:
        terms["pair"] = torch.zeros((), device=activity.device, dtype=activity.dtype)
    return terms


def _check_finite(values: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")


def _mean_features(windows: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Time means ``[B, 2m]`` of the features ``[cos(W x), sin(W x)] / sqrt(m)`` of windows ``[B, T, D]``."""
    _check_finite(windows, "windows")
    phase = torch.einsum("btd,md->btm", windows, frequencies)
    features = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1) * (1.0 / math.sqrt(int(frequencies.shape[0])))
    features = features.mean(dim=1)
    _check_finite(features, "window features")
    return features


def _squared_distances(features: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances ``[B, B]`` between rows, computed in float64 and returned in their dtype."""
    rows = features.to(dtype=torch.float64)
    centered = rows - rows[:1]
    norms = centered.square().sum(dim=1, keepdim=True)
    distances = norms + norms.transpose(0, 1)
    distances = distances - 2.0 * (centered @ centered.transpose(0, 1))
    distances = (0.5 * (distances + distances.transpose(0, 1))).clamp_min(0.0)
    distances.fill_diagonal_(0.0)
    return distances.to(dtype=features.dtype)


def _positive_median(values: torch.Tensor, name: str) -> float:
    positive = values[torch.isfinite(values) & (values > 0)]
    if positive.numel() == 0:
        raise ValueError(f"no finite positive {name}")
    return float(torch.median(positive).item())


class WindowSimilarity:
    """Detached pair weights ``w_bc = d2_bc / (d2_bc + s2)`` between observed windows.

    ``d2_bc`` is the squared distance between the time means of the random
    Fourier features ``[cos(W x), sin(W x)] / sqrt(m)`` of windows ``b`` and
    ``c`` (normalized observations), with ``m`` frequency rows
    ``W ~ N(0, I / b^2)`` drawn from a CPU generator seeded with ``seed``.
    The weights depend on the observations only, never on labels or codes.
    """

    def __init__(self, obs_dim: int, bandwidth: float, distance_scale_squared: float,
                 num_frequencies: int = 64, seed: int = 930001):
        if not all(math.isfinite(value) and value > 0.0 for value in (bandwidth, distance_scale_squared)):
            raise ValueError("bandwidth and distance_scale_squared must be finite and positive")
        self.bandwidth = float(bandwidth)
        self.distance_scale_squared = float(distance_scale_squared)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.frequencies = torch.randn((num_frequencies, obs_dim), generator=generator,
                                       dtype=torch.float64) / self.bandwidth
        _check_finite(self.frequencies, "frequencies (bandwidth too small)")

    @classmethod
    def calibrate(cls, states: torch.Tensor, mean: Optional[torch.Tensor] = None,
                  scale: Optional[torch.Tensor] = None, *, window_states: int = 64, num_windows: int = 512,
                  seed: int = 930001, num_frequencies: int = 64, frequency_seed: int = 930001) -> "WindowSimilarity":
        """Calibrate on training trajectories ``[N, T, D]`` (float32, CPU) only.

        Draws ``min(num_windows, N)`` windows of ``window_states`` states
        (trajectories, then starts, from a generator seeded with ``seed``),
        normalizes them with ``(x - mean) / scale``, and sets the bandwidth
        ``b`` to the median distance between as many evenly spaced states of
        these windows and ``s2`` to the median squared feature distance
        between distinct windows, both in float64. The weights are computed on
        normalized windows, so pass the normalizer's ``mean`` and ``scale``
        together, or neither when ``states`` are already normalized.
        """
        if states.ndim != 3 or states.shape[1] < window_states:
            raise ValueError(f"states must have shape [N, T >= {window_states}, D], got {tuple(states.shape)}")
        if (mean is None) != (scale is None):
            raise ValueError("pass mean and scale together (or neither for already normalized states)")
        _check_finite(states, "states")
        if mean is not None:
            _check_finite(mean, "mean")
            if not bool((scale > 0).all()):  # also rejects NaN
                raise ValueError("scale must be finite and positive")
            _check_finite(scale, "scale")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        count = min(num_windows, states.shape[0])
        trajectory = torch.randint(states.shape[0], (count,), generator=generator)
        starts = torch.randint(states.shape[1] - window_states + 1, (count,), generator=generator)
        windows = states[trajectory[:, None], starts[:, None] + torch.arange(window_states).unsqueeze(0)]
        if mean is not None:
            windows = (windows - mean) / scale
        windows = windows.to(dtype=torch.float64)
        rows = windows.reshape(-1, windows.shape[-1])
        selected = rows[torch.linspace(0, rows.shape[0] - 1, windows.shape[0]).round().long()]
        similarity = cls(states.shape[-1], _positive_median(torch.pdist(selected), "state distances"), 1.0,
                         num_frequencies, frequency_seed)
        distances = _squared_distances(_mean_features(windows, similarity.frequencies))
        upper = distances[torch.triu_indices(count, count, offset=1).unbind()]
        similarity.distance_scale_squared = _positive_median(upper, "feature distances")
        return similarity

    def __call__(self, windows: torch.Tensor) -> torch.Tensor:
        """Weights ``[B, B]`` (zero diagonal, dtype of ``windows``) for normalized windows ``[B, T, D]``, ``B >= 2``."""
        if windows.ndim != 3 or windows.shape[0] < 2 or windows.shape[-1] != self.frequencies.shape[1]:
            raise ValueError(f"windows must have shape [B >= 2, T, {self.frequencies.shape[1]}], "
                             f"got {tuple(windows.shape)}")
        frequencies = self.frequencies.to(device=windows.device, dtype=windows.dtype)
        distances = _squared_distances(_mean_features(windows.detach(), frequencies)).to(dtype=torch.float64)
        distances = 0.5 * (distances + distances.T)
        weights = distances / (distances + self.distance_scale_squared)
        eye = torch.eye(distances.shape[0], dtype=distances.dtype, device=distances.device)
        return (weights * (1.0 - eye)).detach().to(dtype=windows.dtype)
