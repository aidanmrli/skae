"""DMD, polynomial EDMD and RBF EDMD on the Dysts benchmark.

Each method is fit on every transition of the 200 training trajectories,
standardized by the training mean and standard deviation. The fit accumulates
the normal equations ``G = Phi^T Phi``, ``A = Phi^T Phi'`` over 64-state tiles
(stride 63; the last tile ends at the trajectory end and overlaps the one before
it) and solves
``(G + lambda tr(G) / p I) K = A`` in the eigenbasis of ``G`` (negative
eigenvalues clipped to zero). Five rows are reported, the ``model`` column of
``baselines_dysts.csv``:

* ``dmd``                                 constant and state (affine DMD), lifted once;
* ``edmd_poly_direct`` / ``edmd_poly``    constant and every monomial of degree 1-3,
                                          lifted once / re-lifted every ``m`` steps;
* ``edmd_rbf_direct`` / ``edmd_rbf``      constant, state and 256 or 1,024 Gaussian
                                          RBFs whose centers are drawn with the run seed.

Re-lifting maps the predicted state (never an observation) back to the
dictionary. The relative ridge ``lambda``, the RBF count and ``m`` of each
(system, row) were chosen on validation starts (:func:`select`) and are frozen
in ``dysts_settings.json``. Each seed forecasts the 100 test starts of the
neural run with that seed for 5,000 steps, scored like the neural runs: a
start diverges at its first non-finite prediction or prediction farther out
than 10 x the largest training-state norm and is then forecast by the
training-mean state; ``mse`` at horizon H is the mean over starts of the
squared error summed over state coordinates and averaged over the first H steps.
"""

from __future__ import annotations

import json
import math
from functools import partial
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from paper import config
from paper.baselines.rbf import rbf_dictionary, rbf_features
from paper.config import MODELS, paper_config, systems_for
from paper.data import sample_windows
from paper.evaluate import fallback_and_radius, horizon_statistics
from paper.systems.dysts import load_split

# Evidence rows: model label -> (method, re-lifted).
ROWS = {
    "dmd": ("dmd", False),
    "edmd_poly_direct": ("edmd_poly", False),
    "edmd_poly": ("edmd_poly", True),
    "edmd_rbf_direct": ("edmd_rbf", False),
    "edmd_rbf": ("edmd_rbf", True),
}
RELATIVE_RIDGES = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
RBF_CENTERS = (256, 1024)
SELECTION_SEED = 9000  # fit seed of the selection; validation starts use seed 9000 + VALIDATION_SEED_OFFSET
WINDOW, STRIDE, BATCH = 64, 63, 128  # tiles of the training pool, and tiles per accumulation step
SETTINGS = Path(__file__).with_name("dysts_settings.json")
FIELDS = ("benchmark", "system", "model", "seed", "horizon", "mse", "status", "divergence_rate")


class TrainingPool:
    """The training trajectories ``[N, T, d]`` (float32) as 64-state tiles, with their standardization."""

    def __init__(self, states: np.ndarray):
        self.states = np.asarray(states, dtype=np.float32)
        length = self.states.shape[1]
        starts = np.arange(0, length - WINDOW + 1, STRIDE)
        # The last tile ends at the trajectory end and so overlaps the one before it: with 30,001 states,
        # 51 of the 30,000 transitions of a trajectory enter the normal equations twice. The published
        # baselines were fit this way, so it is kept.
        self.starts = starts if starts[-1] == length - WINDOW else np.append(starts, length - WINDOW)
        self.size = len(self.states) * len(self.starts) * WINDOW  # states in all tiles
        # float64 population moments, rounded to float32 like the neural runs' input normalization
        states64 = self.states.astype(np.float64)
        mean = states64.mean(axis=(0, 1))
        scale = np.sqrt(np.mean((states64 - mean) ** 2, axis=(0, 1)))
        self.mean = mean.astype(np.float32).astype(np.float64)
        self.scale = np.maximum(scale, 1e-8).astype(np.float32).astype(np.float64)

    def rows(self, indices: np.ndarray) -> np.ndarray:
        """Standardized states at flat indices of the tiled pool (tile-major, 64 states per tile)."""
        tile, offset = np.divmod(indices, WINDOW)
        trajectory, position = np.divmod(tile, len(self.starts))
        return (self.states[trajectory, self.starts[position] + offset].astype(np.float64) - self.mean) / self.scale

    def batches(self) -> Iterator[np.ndarray]:
        """Standardized float64 tiles ``[<= 128, 64, d]``, trajectory by trajectory."""
        tiles = [(trajectory, start) for trajectory in range(len(self.states)) for start in self.starts]
        for first in range(0, len(tiles), BATCH):
            raw = np.stack([self.states[n, s:s + WINDOW] for n, s in tiles[first:first + BATCH]])
            yield (raw.astype(np.float64) - self.mean) / self.scale


def affine_features(states: np.ndarray) -> np.ndarray:
    """A constant and the state (affine DMD)."""
    return np.concatenate((np.ones((len(states), 1)), states), axis=1)


def poly3_features(states: np.ndarray) -> np.ndarray:
    """A constant and every monomial of degree 1-3, as products of state coordinates."""
    columns = [np.ones(len(states), dtype=np.float64)]
    for degree in (1, 2, 3):
        for axes in combinations_with_replacement(range(states.shape[1]), degree):
            column = np.ones(len(states), dtype=np.float64)
            for axis in axes:
                column *= states[:, axis]
            columns.append(column)
    return np.stack(columns, axis=1)


def dictionary(method: str, pool: TrainingPool, seed: int, centers: Optional[int]) -> Callable:
    """Feature map of a method; RBF centers are drawn from the tiled pool with ``seed``."""
    if method == "dmd":
        return affine_features
    if method == "edmd_poly":
        return poly3_features
    centers, bandwidth = rbf_dictionary(pool.rows, pool.size, centers, seed)
    return partial(rbf_features, centers=centers, bandwidth=bandwidth)


def fit(pool: TrainingPool, features: Callable) -> Callable[[float], np.ndarray]:
    """Accumulate the normal equations over the pool; returns ``relative ridge -> K``."""
    width = features(np.zeros((1, pool.states.shape[-1]))).shape[1]
    gram = np.zeros((width, width))
    cross = np.zeros((width, width))
    for standardized in pool.batches():
        count, length, dim = standardized.shape
        lifted = features(standardized.reshape(count * length, dim)).reshape(count, length, width)
        current, following = lifted[:, :-1].reshape(-1, width), lifted[:, 1:].reshape(-1, width)
        gram += current.T @ current
        cross += current.T @ following
    trace = float(np.trace(gram))
    eigenvalues, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    projected = vectors.T @ cross

    def operator(relative_ridge: float) -> np.ndarray:
        shifted = np.clip(eigenvalues, 0.0, None) + float(relative_ridge) * trace / width
        return vectors @ (projected / shifted[:, None])

    return operator


def predict(features: Callable, affine: bool, pool: TrainingPool, operator: np.ndarray, starts: np.ndarray,
            horizon: int, period: Optional[int] = None) -> np.ndarray:
    """Forecasts ``[horizon, B, d]`` (float64), re-lifting the predicted state every ``period`` steps."""
    state = (np.asarray(starts, dtype=np.float64) - pool.mean) / pool.scale
    lifted = features(state)
    dim = len(pool.mean)
    forecast = np.empty((len(state), horizon, dim))
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(horizon):
            lifted = lifted @ operator
            state = lifted[:, 1:1 + dim]
            forecast[:, step] = state * pool.scale + pool.mean
            if affine:
                lifted[:, 0] = 1.0  # the DMD constant stays exactly one
            if period is not None and (step + 1) % period == 0:
                lifted = features(state)
    return forecast.transpose(1, 0, 2)


def score(predictions: np.ndarray, truth: torch.Tensor, fallback: torch.Tensor, radius: float,
          horizons: Tuple[int, ...]) -> Dict[int, Dict[str, float]]:
    """Per-horizon MSE and divergence rate of forecasts ``[H, B, d]`` (in float32)."""
    predicted = torch.as_tensor(predictions, dtype=torch.float32)
    steps = predicted.shape[0]
    bad = ~torch.isfinite(predicted).all(dim=-1) | (torch.linalg.vector_norm(predicted, dim=-1) > radius)
    diverged_at = torch.where(bad.any(dim=0), bad.to(torch.long).argmax(dim=0), steps)
    dead = torch.arange(steps)[:, None] >= diverged_at[None, :]
    squared = torch.sum((torch.where(dead[..., None], fallback, predicted) - truth) ** 2, dim=-1)
    squared = torch.where(torch.isfinite(squared), squared, torch.nan)
    return {h: {"mse": horizon_statistics(squared, h)["mean"],
                "divergence_rate": float((diverged_at < h).float().mean().item())} for h in horizons}


def forecast_starts(trajectories: torch.Tensor, seed: int, count: int, horizon: int):
    """Starts ``[count, d]`` (NumPy) and truth ``[horizon, count, d]``, drawn as by :func:`paper.data.test_set`."""
    windows = sample_windows(trajectories, torch.Generator().manual_seed(seed), count, horizon)
    return windows[:, 0].contiguous().numpy(), windows[:, 1:].transpose(0, 1).contiguous()


def load_settings(path: Path = SETTINGS) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Per-system choices ``{system: {row: {relative_ridge, [rbf_centers], [period], validation_mse}}}``."""
    return json.loads(Path(path).read_text())["systems"]


def evaluate(system: str, seed: int, data_dir: Path, settings: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Rows (:data:`FIELDS`) of the five baselines for one (system, seed), with ``settings`` per row."""
    eval_cfg = paper_config("dysts", system, MODELS[0], seed).eval
    train, test = load_split(data_dir, system, "train"), load_split(data_dir, system, "test")
    pool = TrainingPool(train.numpy())
    fallback, radius = fallback_and_radius(train)
    horizon = max(eval_cfg.horizons)
    starts, truth = forecast_starts(test, seed + config.TEST_SEED_OFFSET, config.TEST_STARTS, horizon)
    fits, rows = {}, []
    for label, (method, _) in ROWS.items():
        choice = settings[label]
        key = (method, choice.get("rbf_centers"))
        if key not in fits:  # one pass over the pool per dictionary, shared by the direct and re-lifted rows
            features = dictionary(method, pool, seed, choice.get("rbf_centers"))
            fits[key] = features, fit(pool, features)
        features, operator = fits[key]
        forecast = predict(features, method == "dmd", pool, operator(choice["relative_ridge"]), starts, horizon,
                           choice.get("period"))
        for h, stats in score(forecast, truth, fallback, radius, eval_cfg.horizons).items():
            rows.append({"benchmark": "dysts", "system": system, "model": label, "seed": seed, "horizon": h,
                         "mse": stats["mse"], "status": "ok", "divergence_rate": stats["divergence_rate"]})
    return {"system": system, "seed": seed, "rows": rows}


def select_rank(choice: Dict[str, object]) -> tuple:
    """Selection key: lowest finite validation MSE, then fewer RBF centers, larger ridge, shorter period."""
    mse = choice["validation_mse"]
    return (mse if math.isfinite(mse) else math.inf, choice.get("rbf_centers", 0), -choice["relative_ridge"],
            choice.get("period", 0))


def select(system: str, data_dir: Path) -> Dict[str, object]:
    """Choose every row's ridge, RBF count and re-lift period on validation starts only.

    Each candidate is fit with seed 9000 and forecasts 100 starts of the
    validation split (seed 9000 + VALIDATION_SEED_OFFSET) for 500 steps; the
    lowest MSE wins (:func:`select_rank`). Returns the choices in the format of
    ``dysts_settings.json`` and the scores of all candidates.
    """
    eval_cfg = paper_config("dysts", system, MODELS[0], 0).eval
    train, validation = load_split(data_dir, system, "train"), load_split(data_dir, system, "val")
    pool = TrainingPool(train.numpy())
    fallback, radius = fallback_and_radius(train)
    horizon = config.SELECTION_HORIZON
    seed = SELECTION_SEED + config.VALIDATION_SEED_OFFSET
    starts, truth = forecast_starts(validation, seed, config.VALIDATION_STARTS, horizon)
    candidates: Dict[str, List[Dict[str, object]]] = {label: [] for label in ROWS}
    for method, counts in (("dmd", (None,)), ("edmd_poly", (None,)), ("edmd_rbf", RBF_CENTERS)):
        for count in counts:
            features = dictionary(method, pool, SELECTION_SEED, count)
            operator = fit(pool, features)
            for ridge in RELATIVE_RIDGES:
                matrix = operator(ridge)
                for label, (row_method, relifted) in ROWS.items():
                    if row_method != method:
                        continue
                    for period in eval_cfg.periods if relifted else (None,):
                        forecast = predict(features, method == "dmd", pool, matrix, starts, horizon, period)
                        metrics = score(forecast, truth, fallback, radius, (horizon,))
                        choice = {"relative_ridge": ridge, "rbf_centers": count, "period": period,
                                  "validation_mse": metrics[horizon]["mse"]}
                        candidates[label].append({key: value for key, value in choice.items() if value is not None})
    chosen = {label: min(options, key=select_rank) for label, options in candidates.items()}
    return {"system": system, "choices": chosen, "candidates": candidates}


def sort_key(row: Dict[str, object]) -> tuple:
    """Row order of ``baselines_dysts.csv``: system, row (as in :data:`ROWS`), seed, horizon."""
    return systems_for("dysts").index(row["system"]), list(ROWS).index(row["model"]), int(row["seed"]), int(row["horizon"])
