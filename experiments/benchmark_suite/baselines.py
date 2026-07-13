"""Classical baselines for the SKAE benchmark suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def persistence_rollout(x0: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(x0[None], int(horizon), axis=0).astype(np.float32)


@dataclass
class LinearDMD:
    matrix: np.ndarray

    def rollout(self, x0: np.ndarray, horizon: int) -> np.ndarray:
        x = x0.astype(np.float64)
        preds: List[np.ndarray] = []
        for _ in range(int(horizon)):
            x = x @ self.matrix
            preds.append(x.astype(np.float32))
        return np.stack(preds, axis=0)


def fit_dmd(trajectories: np.ndarray, *, ridge: float = 1e-8) -> LinearDMD:
    x = trajectories[:, :-1].reshape(-1, trajectories.shape[-1]).astype(np.float64)
    y = trajectories[:, 1:].reshape(-1, trajectories.shape[-1]).astype(np.float64)
    lhs = x.T @ x + float(ridge) * np.eye(x.shape[1])
    rhs = x.T @ y
    matrix = np.linalg.solve(lhs, rhs)
    return LinearDMD(matrix=matrix.astype(np.float32))


@dataclass
class TruncatedDMD:
    mean: np.ndarray
    basis: np.ndarray
    matrix: np.ndarray

    def rollout(self, x0: np.ndarray, horizon: int) -> np.ndarray:
        z = (x0 - self.mean) @ self.basis
        preds: List[np.ndarray] = []
        for _ in range(int(horizon)):
            z = z @ self.matrix
            preds.append((z @ self.basis.T + self.mean).astype(np.float32))
        return np.stack(preds, axis=0)


def fit_truncated_dmd(trajectories: np.ndarray, *, rank: int, ridge: float = 1e-8) -> TruncatedDMD:
    x_full = trajectories[:, :-1].reshape(-1, trajectories.shape[-1]).astype(np.float64)
    mean = x_full.mean(axis=0)
    centered = x_full - mean
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[: min(int(rank), vt.shape[0])].T
    reduced = (trajectories.astype(np.float64) - mean) @ basis
    model = fit_dmd(reduced, ridge=ridge)
    return TruncatedDMD(mean=mean.astype(np.float32), basis=basis.astype(np.float32), matrix=model.matrix)


@dataclass
class ARXModel:
    order_y: int
    order_u: int
    coef: np.ndarray
    intercept: float

    def _features(self, y_hist: np.ndarray, u_hist: np.ndarray, u_now: float) -> np.ndarray:
        y_part = y_hist[-self.order_y :][::-1]
        u_part = np.concatenate([np.asarray([u_now], dtype=np.float32), u_hist[-self.order_u :][::-1]])
        return np.concatenate([y_part, u_part]).astype(np.float32)

    def one_step_series(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        max_order = max(self.order_y, self.order_u)
        pred = np.full_like(y, np.nan, dtype=np.float32)
        for t in range(max_order, len(y)):
            feat = self._features(y[:t], u[:t], float(u[t]))
            pred[t] = float(feat @ self.coef + self.intercept)
        return pred

    def freerun(self, u: np.ndarray, y_warmup: np.ndarray, *, horizon: int) -> np.ndarray:
        y_hist = list(float(v) for v in y_warmup)
        preds: List[float] = []
        for step in range(int(horizon)):
            t = len(y_warmup) + step
            feat = self._features(np.asarray(y_hist, dtype=np.float32), u[:t], float(u[t]))
            value = float(feat @ self.coef + self.intercept)
            preds.append(value)
            y_hist.append(value)
        return np.asarray(preds, dtype=np.float32)


def _arx_design(u: np.ndarray, y: np.ndarray, order_y: int, order_u: int) -> Tuple[np.ndarray, np.ndarray]:
    max_order = max(int(order_y), int(order_u))
    rows: List[np.ndarray] = []
    targets: List[float] = []
    for t in range(max_order, len(y)):
        y_part = y[t - int(order_y) : t][::-1]
        u_part = np.concatenate([[u[t]], u[t - int(order_u) : t][::-1]])
        rows.append(np.concatenate([y_part, u_part]).astype(np.float32))
        targets.append(float(y[t]))
    return np.stack(rows, axis=0), np.asarray(targets, dtype=np.float32)


def fit_arx(u: np.ndarray, y: np.ndarray, *, order_y: int, order_u: int, ridge: float = 1e-6) -> ARXModel:
    x, target = _arx_design(u, y, order_y, order_u)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1).astype(np.float64)
    lhs = x_aug.T @ x_aug + float(ridge) * np.eye(x_aug.shape[1])
    rhs = x_aug.T @ target.astype(np.float64)
    params = np.linalg.solve(lhs, rhs)
    return ARXModel(order_y=int(order_y), order_u=int(order_u), coef=params[:-1].astype(np.float32), intercept=float(params[-1]))


def select_arx(
    train_u: np.ndarray,
    train_y: np.ndarray,
    val_u: np.ndarray,
    val_y: np.ndarray,
    *,
    orders: Sequence[int],
) -> Tuple[ARXModel, float]:
    best: Tuple[ARXModel, float] | None = None
    for order in orders:
        model = fit_arx(train_u, train_y, order_y=int(order), order_u=int(order))
        pred = model.one_step_series(val_u, val_y)
        mask = np.isfinite(pred)
        mse = float(np.mean((pred[mask] - val_y[mask]) ** 2)) if mask.any() else float("inf")
        if best is None or mse < best[1]:
            best = (model, mse)
    if best is None:
        raise RuntimeError("ARX selection failed.")
    return best
