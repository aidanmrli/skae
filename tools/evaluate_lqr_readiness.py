#!/usr/bin/env python
"""Evaluate label-free LQR readiness from latent trajectories.

This script is designed for the block_diagonal vs arrowhead decision workflow.
It does not use ground-truth basin labels for primary metrics.

Pipeline:
1. Load checkpoint and model.
2. Generate latent trajectories from the selected environment.
3. Discover regimes with unsupervised clustering (KMeans, B_proxy clusters).
4. Fit local linear models per discovered regime.
5. Compute DARE feasibility, closed-loop stability, and finite-horizon cost gains.
6. Save per-regime metrics and aggregate decision metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from skae.config import Config
from skae.data import generate_trajectory, make_env
from skae.model import make_model

EPS = 1e-9


@dataclass
class LatentTrajectoryDataset:
    """Container for latent trajectories and transition indexing."""

    trajectories: List[np.ndarray]  # each [T+1, zdim]
    x_all: np.ndarray  # [N, zdim]
    y_all: np.ndarray  # [N, zdim]
    traj_ids: np.ndarray  # [N]
    time_ids: np.ndarray  # [N]


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: str,
    system: Optional[str],
) -> Tuple[Dict[str, Any], Config, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = Config.from_dict(checkpoint["config"])
    if system is not None:
        cfg.ENV.ENV_NAME = system

    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return checkpoint, cfg, model


def collect_latent_trajectory_dataset(
    model,
    cfg: Config,
    system: str,
    num_trajectories: int,
    trajectory_length: int,
    seed: int,
    device: str,
) -> LatentTrajectoryDataset:
    """Generate latent trajectories and flattened transition pairs."""
    eval_cfg = Config.from_dict(cfg.to_dict())
    eval_cfg.ENV.ENV_NAME = system
    env = make_env(eval_cfg)

    trajectories: List[np.ndarray] = []
    x_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    traj_chunks: List[np.ndarray] = []
    time_chunks: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(num_trajectories):
            rng = torch.Generator().manual_seed(seed + i)
            x0 = env.reset(rng)
            traj = generate_trajectory(env.step, x0, length=trajectory_length)
            traj = torch.cat([x0.unsqueeze(0), traj], dim=0)  # [T+1, obs_dim]

            z = model.encode(traj.to(device)).cpu().numpy().astype(np.float64)
            trajectories.append(z)

            x_tr = z[:-1]
            y_tr = z[1:]
            t = np.arange(trajectory_length, dtype=np.int64)
            tid = np.full((trajectory_length,), i, dtype=np.int64)

            x_chunks.append(x_tr)
            y_chunks.append(y_tr)
            traj_chunks.append(tid)
            time_chunks.append(t)

    x_all = np.concatenate(x_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    traj_ids = np.concatenate(traj_chunks, axis=0)
    time_ids = np.concatenate(time_chunks, axis=0)

    return LatentTrajectoryDataset(
        trajectories=trajectories,
        x_all=x_all,
        y_all=y_all,
        traj_ids=traj_ids,
        time_ids=time_ids,
    )


def discover_regimes_kmeans(
    x_all: np.ndarray,
    b_proxy: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Discover regimes with KMeans on latent states."""
    kmeans = KMeans(
        n_clusters=b_proxy,
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    labels = kmeans.fit_predict(x_all)

    sil = float("nan")
    # Silhouette requires at least 2 clusters and non-trivial assignments.
    if b_proxy > 1 and x_all.shape[0] > b_proxy:
        unique = np.unique(labels)
        if unique.size > 1:
            sil = float(silhouette_score(x_all, labels))

    info = {
        "inertia": float(kmeans.inertia_),
        "silhouette": sil,
    }
    return labels.astype(np.int64), info


def ridge_fit_row_linear(
    x: np.ndarray,
    y: np.ndarray,
    l2_reg: float,
) -> np.ndarray:
    """Fit y ~= x @ A using ridge regression (row-vector convention)."""
    n = x.shape[1]
    xtx = x.T @ x
    reg = l2_reg * np.eye(n, dtype=np.float64)
    a = np.linalg.solve(xtx + reg, x.T @ y)
    return a


def _make_projection_basis(x_centered: np.ndarray, max_state_dim: int) -> np.ndarray:
    """Compute orthonormal projection basis from centered samples."""
    n = x_centered.shape[1]
    if max_state_dim >= n:
        return np.eye(n, dtype=np.float64)

    # Right singular vectors span feature space.
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    d = min(max_state_dim, vt.shape[0])
    return vt[:d].T.astype(np.float64)


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = float(np.mean((y_true - y_pred) ** 2))
    denom = float(np.sqrt(np.mean(y_true ** 2)) + EPS)
    return float(np.sqrt(mse) / denom)


def _build_control_matrix(x: np.ndarray, control_dim: int) -> np.ndarray:
    """Choose control directions from top principal axes of local state variation."""
    n = x.shape[1]
    m = max(1, min(control_dim, n))

    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    try:
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        b = evecs[:, order[:m]]
        if not np.all(np.isfinite(b)):
            raise np.linalg.LinAlgError("non-finite eigenvectors")
        return b.astype(np.float64)
    except np.linalg.LinAlgError:
        return np.eye(n, m, dtype=np.float64)


def solve_lqr(
    a_row: np.ndarray,
    b_col: np.ndarray,
    q_weight: float,
    r_weight: float,
) -> Tuple[bool, Optional[np.ndarray], Optional[float], Optional[str]]:
    """Solve DARE and return stabilizing gain.

    Args:
        a_row: row-vector dynamics matrix (x_next = x @ A_row)
        b_col: column-convention control matrix

    Returns:
        (success, gain, spectral_radius_closed_loop, failure_reason)
    """
    a_col = a_row.T
    n = a_col.shape[0]
    m = b_col.shape[1]
    q = q_weight * np.eye(n, dtype=np.float64)
    r = r_weight * np.eye(m, dtype=np.float64)

    try:
        p = solve_discrete_are(a_col, b_col, q, r)
        bt_p = b_col.T @ p
        k = np.linalg.solve(bt_p @ b_col + r, bt_p @ a_col)
        a_cl = a_col - b_col @ k
        rho = float(np.max(np.abs(np.linalg.eigvals(a_cl))))
        if not np.isfinite(rho):
            return False, None, None, "nonfinite_spectral_radius"
        return True, k, rho, None
    except Exception as exc:  # noqa: BLE001 - diagnostics are useful in sweeps.
        return False, None, None, str(exc)


def _finite_horizon_cost(
    a_col: np.ndarray,
    b_col: np.ndarray,
    k: Optional[np.ndarray],
    x0: np.ndarray,
    horizon: int,
    q_weight: float,
    r_weight: float,
) -> Tuple[float, float]:
    """Return (cost, final_norm)."""
    n = a_col.shape[0]
    q = q_weight * np.eye(n, dtype=np.float64)
    r = r_weight * np.eye(b_col.shape[1], dtype=np.float64)

    x = x0.reshape(-1, 1).astype(np.float64)
    total_cost = 0.0
    for _ in range(horizon):
        if k is None:
            u = np.zeros((b_col.shape[1], 1), dtype=np.float64)
        else:
            u = -k @ x
        total_cost += float((x.T @ q @ x + u.T @ r @ u).item())
        x = a_col @ x + b_col @ u

    return total_cost, float(np.linalg.norm(x))


def _bootstrap_regime_rng(seed: int, regime_id: int) -> np.random.Generator:
    return np.random.default_rng(seed * 1009 + regime_id * 9176 + 17)


def evaluate_single_regime(
    dataset: LatentTrajectoryDataset,
    regime_id: int,
    regime_indices: np.ndarray,
    horizon_h: int,
    lqr_horizon: int,
    min_regime_samples: int,
    test_fraction: float,
    ridge_lambda: float,
    max_state_dim: int,
    control_dim: int,
    q_weight: float,
    r_weight: float,
    rollout_samples: int,
    perturbation_scale: float,
    seed: int,
) -> Dict[str, Any]:
    """Fit and evaluate local linear/LQR metrics for one discovered regime."""
    out: Dict[str, Any] = {
        "regime_id": int(regime_id),
        "num_samples": int(regime_indices.size),
        "status": "ok",
    }

    if regime_indices.size < min_regime_samples:
        out.update(
            {
                "status": "insufficient_samples",
                "failure_reason": "insufficient_samples",
                "dare_success": False,
                "closed_loop_stable": False,
            }
        )
        return out

    rng = _bootstrap_regime_rng(seed, regime_id)
    shuffled = regime_indices.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(round(test_fraction * shuffled.size)))
    if shuffled.size - n_test < max(2, min_regime_samples // 2):
        n_test = max(1, shuffled.size - max(2, min_regime_samples // 2))

    test_idx = shuffled[:n_test]
    train_idx = shuffled[n_test:]
    if train_idx.size < 2:
        out.update(
            {
                "status": "insufficient_train",
                "failure_reason": "insufficient_train",
                "dare_success": False,
                "closed_loop_stable": False,
            }
        )
        return out

    x_train = dataset.x_all[train_idx]
    y_train = dataset.y_all[train_idx]
    x_test = dataset.x_all[test_idx]
    y_test = dataset.y_all[test_idx]

    centroid = x_train.mean(axis=0)
    xc_train = x_train - centroid
    yc_train = y_train - centroid
    xc_test = x_test - centroid
    yc_test = y_test - centroid

    basis = _make_projection_basis(xc_train, max_state_dim=max_state_dim)
    x_train_p = xc_train @ basis
    y_train_p = yc_train @ basis
    x_test_p = xc_test @ basis
    y_test_p = yc_test @ basis

    a_row = ridge_fit_row_linear(x_train_p, y_train_p, l2_reg=ridge_lambda)

    y_pred_1 = x_test_p @ a_row
    nrmse_1 = _nrmse(y_test_p, y_pred_1)

    # H-step latent prediction quality
    valid_h_true: List[np.ndarray] = []
    valid_h_pred: List[np.ndarray] = []
    a_h = np.linalg.matrix_power(a_row, horizon_h)
    for idx in test_idx:
        tid = int(dataset.traj_ids[idx])
        t = int(dataset.time_ids[idx])
        traj = dataset.trajectories[tid]
        target_t = t + horizon_h
        if target_t >= traj.shape[0]:
            continue

        x0_full = dataset.x_all[idx]
        x0_proj = (x0_full - centroid) @ basis
        pred_h = x0_proj @ a_h

        y_true_full = traj[target_t]
        y_true_proj = (y_true_full - centroid) @ basis

        valid_h_true.append(y_true_proj)
        valid_h_pred.append(pred_h)

    if valid_h_true:
        y_h_true = np.stack(valid_h_true, axis=0)
        y_h_pred = np.stack(valid_h_pred, axis=0)
        nrmse_h = _nrmse(y_h_true, y_h_pred)
    else:
        nrmse_h = float("nan")

    b_col = _build_control_matrix(x_train_p, control_dim=control_dim)
    dare_success, k_gain, rho_cl, dare_failure = solve_lqr(
        a_row=a_row,
        b_col=b_col,
        q_weight=q_weight,
        r_weight=r_weight,
    )

    out.update(
        {
            "num_train": int(train_idx.size),
            "num_test": int(test_idx.size),
            "state_dim_used": int(a_row.shape[0]),
            "control_dim_used": int(b_col.shape[1]),
            "local_fit_nrmse_1_step": float(nrmse_1),
            "local_fit_nrmse_h_step": float(nrmse_h),
            "dare_success": bool(dare_success),
            "dare_failure_reason": dare_failure,
            "closed_loop_spectral_radius": float(rho_cl) if rho_cl is not None else None,
            "closed_loop_stable": bool(rho_cl < 1.0) if rho_cl is not None else False,
        }
    )

    if not dare_success or k_gain is None:
        out.update(
            {
                "open_loop_cost_mean": None,
                "closed_loop_cost_mean": None,
                "closed_loop_cost_reduction": None,
                "recovery_success_rate": None,
            }
        )
        return out

    a_col = a_row.T

    # Sample initial points from test set for cost comparisons.
    n_samples = min(rollout_samples, x_test_p.shape[0])
    sampled = x_test_p[:n_samples]

    open_costs: List[float] = []
    closed_costs: List[float] = []
    open_norms: List[float] = []
    closed_norms: List[float] = []

    for x0 in sampled:
        c_open, n_open = _finite_horizon_cost(
            a_col=a_col,
            b_col=b_col,
            k=None,
            x0=x0,
            horizon=lqr_horizon,
            q_weight=q_weight,
            r_weight=r_weight,
        )
        c_closed, n_closed = _finite_horizon_cost(
            a_col=a_col,
            b_col=b_col,
            k=k_gain,
            x0=x0,
            horizon=lqr_horizon,
            q_weight=q_weight,
            r_weight=r_weight,
        )

        open_costs.append(c_open)
        closed_costs.append(c_closed)
        open_norms.append(n_open)
        closed_norms.append(n_closed)

    open_mean = float(np.mean(open_costs)) if open_costs else float("nan")
    closed_mean = float(np.mean(closed_costs)) if closed_costs else float("nan")

    if np.isfinite(open_mean) and open_mean > EPS and np.isfinite(closed_mean):
        cost_reduction = float((open_mean - closed_mean) / open_mean)
    else:
        cost_reduction = float("nan")

    # Recovery metric from local perturbations.
    pert_scale = float(np.std(x_train_p) + EPS) * perturbation_scale
    n_pert = max(8, min(64, rollout_samples))
    success = 0
    for _ in range(n_pert):
        x0 = rng.normal(loc=0.0, scale=pert_scale, size=(a_row.shape[0],))
        _, n_open = _finite_horizon_cost(
            a_col=a_col,
            b_col=b_col,
            k=None,
            x0=x0,
            horizon=lqr_horizon,
            q_weight=q_weight,
            r_weight=r_weight,
        )
        _, n_closed = _finite_horizon_cost(
            a_col=a_col,
            b_col=b_col,
            k=k_gain,
            x0=x0,
            horizon=lqr_horizon,
            q_weight=q_weight,
            r_weight=r_weight,
        )
        # Success if closed-loop contracts substantially more than open-loop.
        if n_closed < 0.5 * max(n_open, EPS):
            success += 1

    out.update(
        {
            "open_loop_cost_mean": open_mean,
            "closed_loop_cost_mean": closed_mean,
            "closed_loop_cost_reduction": cost_reduction,
            "recovery_success_rate": float(success / n_pert),
            "open_loop_final_norm_mean": float(np.mean(open_norms)) if open_norms else None,
            "closed_loop_final_norm_mean": float(np.mean(closed_norms)) if closed_norms else None,
        }
    )

    return out


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    return float(np.mean(finite))


def aggregate_metrics(regime_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate M1-M4 metrics from per-regime outputs."""
    evaluable = [r for r in regime_metrics if r.get("status") == "ok"]
    if not evaluable:
        return {
            "num_regimes_total": len(regime_metrics),
            "num_regimes_evaluable": 0,
            "m1_local_fit_nrmse_1_step": None,
            "m1_local_fit_nrmse_h_step": None,
            "m2_lqr_feasibility_rate": 0.0,
            "m3_closed_loop_stability_rate": 0.0,
            "m4_closed_loop_cost_reduction": None,
            "m4_recovery_success_rate": None,
        }

    dare_success = [bool(r.get("dare_success", False)) for r in evaluable]
    stable = [bool(r.get("closed_loop_stable", False)) for r in evaluable]

    return {
        "num_regimes_total": len(regime_metrics),
        "num_regimes_evaluable": len(evaluable),
        "m1_local_fit_nrmse_1_step": _safe_mean([r.get("local_fit_nrmse_1_step") for r in evaluable]),
        "m1_local_fit_nrmse_h_step": _safe_mean([r.get("local_fit_nrmse_h_step") for r in evaluable]),
        "m2_lqr_feasibility_rate": float(np.mean(dare_success)),
        "m3_closed_loop_stability_rate": float(np.mean(stable)),
        "m4_closed_loop_cost_reduction": _safe_mean([r.get("closed_loop_cost_reduction") for r in evaluable]),
        "m4_recovery_success_rate": _safe_mean([r.get("recovery_success_rate") for r in evaluable]),
    }


def _default_b_proxy(cfg: Config) -> int:
    if cfg.MODEL.STRUCTURED.ENABLED:
        return int(cfg.MODEL.STRUCTURED.NUM_BASINS)
    if cfg.ENV.ENV_NAME == "duffing":
        return 2
    if cfg.ENV.ENV_NAME == "lyapunov":
        return int(cfg.ENV.LYAPUNOV.NUM_BASINS)
    return 8


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate label-free LQR readiness from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt or last.pt")
    parser.add_argument("--system", type=str, default=None, help="Evaluation system override")
    parser.add_argument("--b_proxy", type=int, default=None, help="Assumed regime count for clustering")
    parser.add_argument("--num_trajectories", type=int, default=128)
    parser.add_argument("--trajectory_length", type=int, default=300)
    parser.add_argument("--horizon_h", type=int, default=20, help="H-step horizon for local fit metric")
    parser.add_argument("--lqr_horizon", type=int, default=40, help="Finite horizon for cost comparisons")
    parser.add_argument("--min_regime_samples", type=int, default=120)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--max_state_dim", type=int, default=32,
                        help="Max projected state dimension for local (A,B) fitting")
    parser.add_argument("--control_dim", type=int, default=8)
    parser.add_argument("--q_weight", type=float, default=1.0)
    parser.add_argument("--r_weight", type=float, default=0.1)
    parser.add_argument("--rollout_samples", type=int, default=64)
    parser.add_argument("--perturbation_scale", type=float, default=0.5)
    parser.add_argument("--kmeans_n_init", type=int, default=10)
    parser.add_argument("--kmeans_max_iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output_dir", type=str, default=None)

    # Optional metadata for downstream aggregation/decision.
    parser.add_argument("--stage", type=int, default=-1)
    parser.add_argument("--arm", type=str, default="unknown")
    parser.add_argument("--run_seed", type=int, default=None)
    parser.add_argument("--target_size", type=int, default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint, cfg, model = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        device=args.device,
        system=args.system,
    )

    system = args.system if args.system is not None else cfg.ENV.ENV_NAME
    b_proxy = int(args.b_proxy) if args.b_proxy is not None else _default_b_proxy(cfg)

    print(f"[LQR readiness] checkpoint={checkpoint_path}")
    print(f"[LQR readiness] system={system} b_proxy={b_proxy} device={args.device}")

    dataset = collect_latent_trajectory_dataset(
        model=model,
        cfg=cfg,
        system=system,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        seed=args.seed,
        device=args.device,
    )

    regime_labels, clustering_info = discover_regimes_kmeans(
        x_all=dataset.x_all,
        b_proxy=b_proxy,
        seed=args.seed,
        n_init=args.kmeans_n_init,
        max_iter=args.kmeans_max_iter,
    )

    regime_metrics: List[Dict[str, Any]] = []
    for regime_id in range(b_proxy):
        reg_idx = np.where(regime_labels == regime_id)[0]
        reg_metrics = evaluate_single_regime(
            dataset=dataset,
            regime_id=regime_id,
            regime_indices=reg_idx,
            horizon_h=args.horizon_h,
            lqr_horizon=args.lqr_horizon,
            min_regime_samples=args.min_regime_samples,
            test_fraction=args.test_fraction,
            ridge_lambda=args.ridge_lambda,
            max_state_dim=args.max_state_dim,
            control_dim=args.control_dim,
            q_weight=args.q_weight,
            r_weight=args.r_weight,
            rollout_samples=args.rollout_samples,
            perturbation_scale=args.perturbation_scale,
            seed=args.seed,
        )
        regime_metrics.append(reg_metrics)

    aggregate = aggregate_metrics(regime_metrics)

    meta = {
        "checkpoint": str(checkpoint_path),
        "system": system,
        "b_proxy": b_proxy,
        "stage": int(args.stage),
        "arm": str(args.arm),
        "run_seed": int(args.run_seed) if args.run_seed is not None else None,
        "target_size": int(args.target_size) if args.target_size is not None else int(cfg.MODEL.TARGET_SIZE),
        "model_name": cfg.MODEL.MODEL_NAME,
        "structured_enabled": bool(cfg.MODEL.STRUCTURED.ENABLED),
        "model_num_basins": int(cfg.MODEL.STRUCTURED.NUM_BASINS) if cfg.MODEL.STRUCTURED.ENABLED else None,
        "env_num_basins": int(cfg.ENV.LYAPUNOV.NUM_BASINS) if system == "lyapunov" else None,
        "checkpoint_step": checkpoint.get("step"),
    }

    regime_sizes = [int(np.sum(regime_labels == i)) for i in range(b_proxy)]
    payload = {
        "metadata": meta,
        "config": {
            "num_trajectories": int(args.num_trajectories),
            "trajectory_length": int(args.trajectory_length),
            "horizon_h": int(args.horizon_h),
            "lqr_horizon": int(args.lqr_horizon),
            "min_regime_samples": int(args.min_regime_samples),
            "test_fraction": float(args.test_fraction),
            "ridge_lambda": float(args.ridge_lambda),
            "max_state_dim": int(args.max_state_dim),
            "control_dim": int(args.control_dim),
            "q_weight": float(args.q_weight),
            "r_weight": float(args.r_weight),
            "rollout_samples": int(args.rollout_samples),
            "perturbation_scale": float(args.perturbation_scale),
            "seed": int(args.seed),
            "kmeans_n_init": int(args.kmeans_n_init),
            "kmeans_max_iter": int(args.kmeans_max_iter),
        },
        "clustering": {
            "regime_sizes": regime_sizes,
            **clustering_info,
        },
        "aggregate_metrics": aggregate,
        "per_regime_metrics": regime_metrics,
    }

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else checkpoint_path.parent / "lqr_readiness"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "lqr_readiness_summary.json"
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[LQR readiness] saved: {summary_path}")
    print(
        "[LQR readiness] aggregate "
        f"M2={aggregate.get('m2_lqr_feasibility_rate')} "
        f"M3={aggregate.get('m3_closed_loop_stability_rate')} "
        f"M4_cost={aggregate.get('m4_closed_loop_cost_reduction')}"
    )


if __name__ == "__main__":
    main()
