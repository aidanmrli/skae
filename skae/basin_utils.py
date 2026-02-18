"""Utilities for basin-of-attraction labeling of dynamical-system trajectories.

Provides functions to identify which basin of attraction a trajectory converges
to, and a dataset class that pairs trajectories with their ground-truth basin
labels.  These are used by multiple evaluation tools (support uniqueness,
eigenvalue analysis, latent clustering, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from skae.config import Config
from skae.data import make_env, generate_trajectory, Duffing


# ---------------------------------------------------------------------------
# Basin Identification
# ---------------------------------------------------------------------------


def identify_duffing_basin(
    env: Duffing,
    trajectory: torch.Tensor,
    long_rollout_steps: int = 5000,
) -> int:
    """Identify Duffing basin: 0=left well, 1=right well."""
    state = trajectory[-1].clone()
    for _ in range(long_rollout_steps):
        state = env.step(state)
    return 0 if state[0].item() < 0 else 1


def identify_point_attractor_basin(
    env,
    trajectory: torch.Tensor,
    long_rollout_steps: int = 5000,
) -> int:
    """Identify basin by nearest attractor center after convergence."""
    state = trajectory[-1].clone()
    for _ in range(long_rollout_steps):
        state = env.step(state)
    distances = torch.norm(state - env.points, dim=-1)
    return distances.argmin().item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class BasinLabeledTrajectory:
    states: torch.Tensor
    final_basin: int


class BasinLabeledDataset:
    """Dataset of trajectories with ground-truth basin labels."""

    def __init__(
        self,
        system: str,
        cfg: Config,
        num_trajectories: int = 100,
        trajectory_length: int = 500,
        long_rollout_steps: int = 5000,
        seed: int = 42,
    ):
        self.system = system
        self.cfg = cfg
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.long_rollout_steps = long_rollout_steps
        self.seed = seed

        self.env = make_env(cfg)

        system_lower = system.lower()
        if system_lower == 'duffing':
            self.num_basins = 2
            self.basin_names = ['Left Well (x<0)', 'Right Well (x>0)']
        elif system_lower == 'lyapunov' or system_lower.startswith('multiwell'):
            self.num_basins = int(self.env.points.shape[0])
            self.basin_names = [f'Attractor {i}' for i in range(self.num_basins)]
        else:
            raise ValueError(
                f"Unknown system: {system}. Supported: duffing, lyapunov, multiwell*"
            )

        self.trajectories: List[BasinLabeledTrajectory] = []
        self._generate_trajectories()

    def _generate_trajectories(self):
        print(f"Generating {self.num_trajectories} trajectories for {self.system}...")
        for i in range(self.num_trajectories):
            env_rng = torch.Generator().manual_seed(self.seed + i)
            init_state = self.env.reset(env_rng)

            traj = generate_trajectory(
                self.env.step,
                init_state,
                length=self.trajectory_length,
            )
            traj = torch.cat([init_state.unsqueeze(0), traj], dim=0)

            if self.system.lower() == 'duffing':
                final_basin = identify_duffing_basin(self.env, traj, self.long_rollout_steps)
            else:
                final_basin = identify_point_attractor_basin(self.env, traj, self.long_rollout_steps)

            self.trajectories.append(
                BasinLabeledTrajectory(states=traj, final_basin=final_basin)
            )
