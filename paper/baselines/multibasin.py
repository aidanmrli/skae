"""DMD, polynomial EDMD and RBF EDMD on the multibasin benchmark.

Each method lifts the standardized state with a fixed dictionary and fits one
linear map ``phi(x_{t+1}) ~ phi(x_t) K`` by ridge regression on the
mean-normalized normal equations ``(Phi^T Phi / N + lambda I) K = Phi^T Phi' / N``
(least squares at ``lambda = 0``):

* ``dmd``        the state itself;
* ``edmd_poly``  a constant and every monomial of degree 1-3;
* ``edmd_rbf``   a constant, the state and 256 or 1,024 Gaussian RBFs whose
                 centers are drawn with the run seed (:mod:`paper.baselines.rbf`).

The state is read back from its own coordinates in the dictionary. A fit for
(system, seed) uses the training windows of the neural run with that seed
(2,048 windows of eight transitions); no basin labels are used. Two rollouts
are reported: ``direct`` lifts once, ``selected_periodic`` decodes and re-lifts
the prediction every ``m`` steps. The ridge, the RBF count and ``m`` minimize
the MSE at H = 500 on the validation starts of the neural run; ties go to fewer
features, a larger ridge, then a shorter period. Test errors come from
:func:`paper.evaluate.forecast_metrics` on the neural test starts, with the
same divergence fallback as the neural runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from itertools import combinations_with_replacement
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from paper import config
from paper.baselines.rbf import rbf_dictionary, rbf_features
from paper.config import MODELS, paper_config
from paper.data import multibasin_training_windows, test_set, validation_forecast_set
from paper.evaluate import fallback_and_radius, forecast_metrics
from paper.systems import make_system

METHODS = ("dmd", "edmd_poly", "edmd_rbf")
RIDGES = (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
POLY_DEGREE = 3
RBF_CENTERS = (256, 1024)
FIELDS = ("benchmark", "system", "seed", "method", "rollout", "horizon", "mse", "divergence_rate")
CPU = torch.device("cpu")


def monomials(dim: int, degree: int) -> Tuple[Tuple[int, ...], ...]:
    """Exponents of the constant and of every monomial of degree 1 to ``degree``."""
    powers = [(0,) * dim]
    for order in range(1, degree + 1):
        for combo in combinations_with_replacement(range(dim), order):
            powers.append(tuple(combo.count(axis) for axis in range(dim)))
    return tuple(powers)


def power_features(states, powers: Sequence[Tuple[int, ...]]):
    """Monomial features ``[B, len(powers)]`` of states ``[B, d]`` (NumPy array or torch tensor)."""
    is_tensor = isinstance(states, torch.Tensor)
    columns = []
    for power in powers:
        value = torch.ones_like(states[:, 0]) if is_tensor else np.ones(states.shape[0])
        for axis, exponent in enumerate(power):
            if exponent:
                value = value * states[:, axis] ** exponent
        columns.append(value)
    return torch.stack(columns, dim=1) if is_tensor else np.stack(columns, axis=1)


def dictionaries(method: str, standardized: np.ndarray, seed: int) -> List[Tuple[Callable, int]]:
    """Candidate dictionaries of a method: ``(features, column of the first state coordinate)``."""
    dim = standardized.shape[1]
    if method == "dmd":
        return [(partial(power_features, powers=monomials(dim, 1)[1:]), 0)]
    if method == "edmd_poly":
        return [(partial(power_features, powers=monomials(dim, POLY_DEGREE)), 1)]
    candidates = []
    for count in RBF_CENTERS:
        centers, bandwidth = rbf_dictionary(lambda index: standardized[index], len(standardized), count, seed)
        candidates.append((partial(rbf_features, centers=centers, bandwidth=bandwidth), 1))
    return candidates


@dataclass
class Fit:
    """One ridge fit: the dictionary, the standardization and the lifted operator ``K``."""

    features: Callable  # standardized states -> dictionary features (NumPy or torch)
    first: int          # column of the first state coordinate in the features
    mean: np.ndarray    # training-state mean and standard deviation, [d]
    scale: np.ndarray
    operator: np.ndarray  # K, [p, p]
    ridge: float


def standardization(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Population mean and standard deviation of every window state (a deviation below 1e-8 becomes 1)."""
    states = windows.reshape(-1, windows.shape[-1])
    mean = states.mean(axis=0)
    scale = np.sqrt(np.mean((states - mean) ** 2, axis=0))
    return mean, np.where(scale < 1e-8, 1.0, scale)


def fit_ridges(windows: np.ndarray, mean: np.ndarray, scale: np.ndarray, features: Callable,
               first: int, ridges: Sequence[float] = RIDGES) -> List[Fit]:
    """One fit per ridge on every transition of the windows ``[N, L + 1, d]`` (float64)."""
    dim = windows.shape[-1]
    x = features((windows[:, :-1].reshape(-1, dim) - mean) / scale)
    y = features((windows[:, 1:].reshape(-1, dim) - mean) / scale)
    gram, cross = x.T @ x / x.shape[0], x.T @ y / x.shape[0]
    fits = []
    for ridge in ridges:
        regularized = gram + float(ridge) * np.eye(len(gram))
        if ridge == 0.0:
            operator = np.linalg.lstsq(regularized, cross, rcond=None)[0]
        else:
            try:
                operator = np.linalg.solve(regularized, cross)
            except np.linalg.LinAlgError:
                operator = np.linalg.pinv(regularized) @ cross
        fits.append(Fit(features, first, mean, scale, operator, float(ridge)))
    return fits


class LiftedModel(torch.nn.Module):
    """A fit with the ``encode``/``decode``/``latent_stepper`` interface of :mod:`paper.evaluate` (float64).

    ``encode`` lifts states ``[B, d]`` to features ``[B, p]``, ``decode`` reads
    the state back from its own feature columns, and one latent step is ``z K``.
    """

    def __init__(self, fit: Fit):
        super().__init__()
        self.features = fit.features
        dim = len(fit.mean)
        decoder = np.zeros((fit.operator.shape[0], dim))
        decoder[fit.first:fit.first + dim] = np.eye(dim)
        for name, value in (("mean", fit.mean), ("scale", fit.scale), ("operator", fit.operator), ("decoder", decoder)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.features((x.to(torch.float64) - self.mean) / self.scale)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.decoder) * self.scale + self.mean

    def latent_stepper(self) -> Callable[[torch.Tensor], torch.Tensor]:
        operator = self.operator
        return lambda z: z @ operator


def rank(score: float, fit: Fit, period: Optional[int] = None) -> tuple:
    """Selection key: lowest finite validation MSE, then fewer features, larger ridge, shorter period."""
    key = (score if math.isfinite(score) else math.inf, fit.operator.shape[0], -fit.ridge)
    return key if period is None else key + (period,)


def evaluate(system: str, seed: int, methods: Sequence[str] = METHODS) -> Dict[str, object]:
    """Validation choices and test rows (:data:`FIELDS`) of every method for one (system, seed)."""
    cfg = paper_config("multibasin", system, MODELS[0], seed)  # every model sees the same data
    dynamics = make_system("multibasin", system)
    windows = multibasin_training_windows(dynamics, cfg)
    fallback, radius = fallback_and_radius(windows)
    windows = np.asarray(windows.reshape(-1, *windows.shape[2:]), dtype=np.float64)
    validation, test = validation_forecast_set(dynamics, cfg), test_set(dynamics, cfg)

    def errors(fit, starts, truth, horizons, period=None):
        periods = None if period is None else (period,)
        return forecast_metrics(LiftedModel(fit), starts.to(torch.float64), truth, cfg.eval, device=CPU,
                                fallback=fallback, horizons=horizons, periods=periods,
                                periodic_only=period is not None, radius=radius)["modes"]

    mean, scale = standardization(windows)
    standardized = (windows.reshape(-1, windows.shape[-1]) - mean) / scale
    selection = config.SELECTION_HORIZON
    rows, choices = [], {}
    for method in methods:
        candidates = {"direct": [], "selected_periodic": []}
        for features, first in dictionaries(method, standardized, seed):
            for fit in fit_ridges(windows, mean, scale, features, first):
                modes = errors(fit, *validation, (selection,))
                score = modes["direct"][str(selection)]["mean"]
                candidates["direct"].append((rank(score, fit), fit, None, score))
                for m in cfg.eval.periods:
                    score = modes[f"periodic_{m}"][str(selection)]["mean"]
                    candidates["selected_periodic"].append((rank(score, fit, m), fit, m, score))
        for rollout, ranked in candidates.items():
            _, fit, period, score = min(ranked, key=lambda candidate: candidate[0])
            choices[f"{method}/{rollout}"] = {"ridge": fit.ridge, "features": int(fit.operator.shape[0]),
                                              "period": period, "validation_mse": score}
            modes = errors(fit, *test, cfg.eval.horizons, period)
            for horizon, stats in modes["direct" if period is None else f"periodic_{period}"].items():
                rows.append({"benchmark": "multibasin", "system": system, "seed": seed, "method": method,
                             "rollout": rollout, "horizon": int(horizon), "mse": stats["mean"],
                             "divergence_rate": stats["divergence_rate"]})
    return {"system": system, "seed": seed, "choices": choices, "rows": rows}


def sort_key(row: Dict[str, object]) -> tuple:
    """Row order of ``baselines_multibasin.csv``: system, seed, method, rollout, horizon."""
    return row["system"], int(row["seed"]), row["method"], row["rollout"], int(row["horizon"])
