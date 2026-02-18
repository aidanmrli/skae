"""
Training-time support monitor for diagnosing basin-support correspondence.

This module provides real-time diagnostics during training to detect:
- Support collapse (all basins → same dimensions)
- Support fragmentation (same basin → inconsistent supports)
- Support overlap (different basins sharing too many dimensions)

Intended for use with LISTA-based models on multi-basin dynamical systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from skae.config import Config
from skae.data import make_env, generate_trajectory


@dataclass
class SupportDiagnostics:
    """Training-time diagnostics for basin-support correspondence."""

    # Core metrics (log these every eval)
    inter_basin_jaccard: float  # Mean Jaccard between basin supports (lower = better)
    intra_basin_consistency: float  # Mean consistency within basins (higher = better)
    unique_support_count: int  # Number of distinct support patterns
    mean_support_size: float  # Average number of active dimensions
    support_size_std: float  # Std of support sizes (high = imbalanced)

    # Derived health indicators
    separation_score: float  # 1 - jaccard (higher = better separation)
    support_ratio: float  # unique_supports / num_basins (1.0 = perfect)

    # Per-basin breakdown (for debugging)
    per_basin_support_sizes: Dict[int, float]
    per_basin_consistency: Dict[int, float]

    def is_healthy(self) -> bool:
        """Quick check if training is on track."""
        return (
            self.separation_score > 0.5 and  # Basins reasonably separated
            self.intra_basin_consistency > 0.5 and  # Consistent within basins
            self.support_ratio > 0.5  # At least half the basins have unique supports
        )

    def summary_str(self) -> str:
        """One-line summary for logging."""
        health = "✓" if self.is_healthy() else "✗"
        return (
            f"Support[{health}]: sep={self.separation_score:.2f} "
            f"cons={self.intra_basin_consistency:.2f} "
            f"uniq={self.unique_support_count} "
            f"size={self.mean_support_size:.1f}"
        )


class SupportMonitor:
    """
    Monitors basin-support correspondence during training.

    This is a lightweight diagnostic tool that samples a small number of
    trajectories per basin and computes support statistics. Designed to
    be called every ~500-1000 training steps without significant overhead.

    Usage:
        monitor = SupportMonitor(cfg, device='cuda')

        # During training loop:
        if step % 500 == 0:
            diagnostics = monitor.compute(model)
            logger.log_dict(diagnostics.to_dict(), step, prefix='support')
            print(diagnostics.summary_str())
    """

    def __init__(
        self,
        cfg: Config,
        device: str = 'cpu',
        num_trajectories_per_basin: int = 5,
        trajectory_length: int = 100,
        convergence_steps: int = 2000,
        support_threshold: float = 1e-3,
        seed: int = 42,
    ):
        """
        Args:
            cfg: Model/env configuration
            device: Device for model inference
            num_trajectories_per_basin: Trajectories to sample per basin
            trajectory_length: Length of each trajectory
            convergence_steps: Steps to roll forward for basin identification
            support_threshold: Threshold for determining active dimensions
            seed: Random seed for reproducibility
        """
        self.cfg = cfg
        self.device = device
        self.num_trajectories_per_basin = num_trajectories_per_basin
        self.trajectory_length = trajectory_length
        self.convergence_steps = convergence_steps
        self.support_threshold = support_threshold
        self.seed = seed

        # Create environment and identify basins
        self.env = make_env(cfg)
        self.system = cfg.ENV.ENV_NAME.lower()

        # Determine number of basins
        if self.system == 'duffing':
            self.num_basins = 2
        elif self.system == 'lyapunov' or self.system.startswith('multiwell'):
            self.num_basins = int(self.env.points.shape[0])
        elif self.system == 'blended':
            self.num_basins = 3
        else:
            # Default: try to infer or assume 1
            self.num_basins = getattr(self.env, 'num_basins', 1)

        # Pre-generate trajectories for each basin (done once)
        self._basin_trajectories: Optional[Dict[int, List[torch.Tensor]]] = None

    def _identify_basin(self, trajectory: torch.Tensor) -> int:
        """Identify which basin a trajectory converges to."""
        state = trajectory[-1].clone()

        # Roll forward to convergence
        for _ in range(self.convergence_steps):
            state = self.env.step(state)

        if self.system == 'duffing':
            return 0 if state[0].item() < 0 else 1
        elif self.system == 'lyapunov' or self.system.startswith('multiwell'):
            distances = torch.norm(state - self.env.points, dim=-1)
            return distances.argmin().item()
        elif self.system == 'blended':
            # Blended has 3 basins at known positions
            centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
            distances = torch.norm(state[:2] - centers, dim=-1)
            return distances.argmin().item()
        else:
            return 0

    def _generate_basin_trajectories(self):
        """Generate trajectories for each basin (cached)."""
        if self._basin_trajectories is not None:
            return

        self._basin_trajectories = {b: [] for b in range(self.num_basins)}
        basin_counts = {b: 0 for b in range(self.num_basins)}

        # Generate trajectories until we have enough per basin
        max_attempts = self.num_basins * self.num_trajectories_per_basin * 10
        attempt = 0

        while any(c < self.num_trajectories_per_basin for c in basin_counts.values()):
            if attempt >= max_attempts:
                break

            rng = torch.Generator().manual_seed(self.seed + attempt)
            init_state = self.env.reset(rng)

            traj = generate_trajectory(
                self.env.step,
                init_state,
                length=self.trajectory_length,
            )
            traj = torch.cat([init_state.unsqueeze(0), traj], dim=0)

            basin = self._identify_basin(traj)

            if basin_counts[basin] < self.num_trajectories_per_basin:
                self._basin_trajectories[basin].append(traj)
                basin_counts[basin] += 1

            attempt += 1

    def _compute_support(self, latents: torch.Tensor) -> np.ndarray:
        """Compute support from latent trajectory (mean aggregation)."""
        z_mean = latents.mean(dim=0)
        support = (z_mean.abs() > self.support_threshold).cpu().numpy().astype(np.int8)
        return support

    def _jaccard(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Jaccard similarity between two supports."""
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 1.0
        return float(inter) / float(union)

    @torch.no_grad()
    def compute(self, model) -> SupportDiagnostics:
        """
        Compute support diagnostics for the current model state.

        Args:
            model: The Koopman model (must have .encode() method)

        Returns:
            SupportDiagnostics with all computed metrics
        """
        self._generate_basin_trajectories()
        model.eval()

        # Collect supports per basin
        basin_supports: Dict[int, List[Tuple[int, ...]]] = {b: [] for b in range(self.num_basins)}

        for basin, trajectories in self._basin_trajectories.items():
            for traj in trajectories:
                states = traj.to(self.device)
                z = model.encode(states)
                support = self._compute_support(z)
                basin_supports[basin].append(tuple(support.tolist()))

        # Compute per-basin mode supports and consistency
        basin_mode_supports: Dict[int, Tuple[int, ...]] = {}
        per_basin_consistency: Dict[int, float] = {}
        per_basin_support_sizes: Dict[int, float] = {}

        for basin, supports in basin_supports.items():
            if not supports:
                per_basin_consistency[basin] = 0.0
                per_basin_support_sizes[basin] = 0.0
                basin_mode_supports[basin] = tuple([0] * len(supports[0]) if supports else [])
                continue

            # Find mode support
            counts: Dict[Tuple[int, ...], int] = {}
            for s in supports:
                counts[s] = counts.get(s, 0) + 1
            mode_support, mode_count = max(counts.items(), key=lambda kv: kv[1])

            basin_mode_supports[basin] = mode_support
            per_basin_consistency[basin] = mode_count / len(supports)
            per_basin_support_sizes[basin] = float(np.sum(mode_support))

        # Compute inter-basin Jaccard similarities
        jaccards = []
        basins = sorted(basin_mode_supports.keys())
        for i, bi in enumerate(basins):
            si = np.array(basin_mode_supports[bi], dtype=np.int8)
            for bj in basins[i + 1:]:
                sj = np.array(basin_mode_supports[bj], dtype=np.int8)
                jaccards.append(self._jaccard(si, sj))

        inter_basin_jaccard = float(np.mean(jaccards)) if jaccards else 0.0

        # Count unique supports
        unique_supports = len(set(basin_mode_supports.values()))

        # Aggregate statistics
        support_sizes = list(per_basin_support_sizes.values())
        mean_support_size = float(np.mean(support_sizes)) if support_sizes else 0.0
        support_size_std = float(np.std(support_sizes)) if support_sizes else 0.0
        mean_consistency = float(np.mean(list(per_basin_consistency.values()))) if per_basin_consistency else 0.0

        return SupportDiagnostics(
            inter_basin_jaccard=inter_basin_jaccard,
            intra_basin_consistency=mean_consistency,
            unique_support_count=unique_supports,
            mean_support_size=mean_support_size,
            support_size_std=support_size_std,
            separation_score=1.0 - inter_basin_jaccard,
            support_ratio=unique_supports / max(1, self.num_basins),
            per_basin_support_sizes=per_basin_support_sizes,
            per_basin_consistency=per_basin_consistency,
        )

    def to_log_dict(self, diagnostics: SupportDiagnostics) -> Dict[str, float]:
        """Convert diagnostics to a flat dict for logging."""
        return {
            'inter_basin_jaccard': diagnostics.inter_basin_jaccard,
            'intra_basin_consistency': diagnostics.intra_basin_consistency,
            'unique_support_count': float(diagnostics.unique_support_count),
            'mean_support_size': diagnostics.mean_support_size,
            'support_size_std': diagnostics.support_size_std,
            'separation_score': diagnostics.separation_score,
            'support_ratio': diagnostics.support_ratio,
        }
