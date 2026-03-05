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


def identify_env_basin(
    env,
    trajectory: torch.Tensor,
    long_rollout_steps: int = 5000,
) -> int:
    """Identify basin label using the environment's native labeling when available."""
    state = trajectory[-1].clone()
    for _ in range(long_rollout_steps):
        state = env.step(state)

    if hasattr(env, "basin_label"):
        label = env.basin_label(state)
        if isinstance(label, torch.Tensor):
            if label.numel() != 1:
                raise ValueError("Expected scalar basin label tensor.")
            return int(label.item())
        return int(label)

    if isinstance(env, Duffing):
        return 0 if state[0].item() < 0 else 1

    if hasattr(env, "points"):
        distances = torch.norm(state - env.points, dim=-1)
        return int(distances.argmin().item())

    raise ValueError(
        "Environment does not expose basin labeling. "
        "Expected `basin_label(state)` or `points` attribute."
    )


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
        elif hasattr(self.env, "basin_label"):
            configured_num_basins = int(getattr(self.env, "num_basins", 0))
            self.num_basins = configured_num_basins if configured_num_basins > 0 else 0
            if system_lower == "hopfield" and self.num_basins > 0:
                self.basin_names = [f"Pattern {i}" for i in range(self.num_basins)]
            elif self.num_basins > 0:
                self.basin_names = [f'Basin {i}' for i in range(self.num_basins)]
            else:
                self.basin_names = []
        else:
            raise ValueError(
                f"Unknown system: {system}. Supported: duffing, lyapunov, multiwell*, "
                "or any environment that implements basin_label(state)."
            )

        self.trajectories: List[BasinLabeledTrajectory] = []
        self._generate_trajectories()

    def _generate_trajectories(self):
        print(f"Generating {self.num_trajectories} trajectories for {self.system}...")
        raw_trajectories: List[torch.Tensor] = []
        raw_labels: List[int] = []

        for i in range(self.num_trajectories):
            env_rng = torch.Generator().manual_seed(self.seed + i)
            init_state = self.env.reset(env_rng)

            traj = generate_trajectory(
                self.env.step,
                init_state,
                length=self.trajectory_length,
            )
            traj = torch.cat([init_state.unsqueeze(0), traj], dim=0)

            final_basin = identify_env_basin(self.env, traj, self.long_rollout_steps)
            raw_trajectories.append(traj)
            raw_labels.append(final_basin)
        has_fixed_index_space = (
            self.num_basins > 0
            and all(0 <= label < self.num_basins for label in raw_labels)
        )

        if has_fixed_index_space:
            mapped_labels = raw_labels
            if not self.basin_names:
                self.basin_names = [f"Basin {i}" for i in range(self.num_basins)]
        else:
            unique_labels = sorted(set(raw_labels))
            label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            mapped_labels = [label_to_index[label] for label in raw_labels]
            self.num_basins = len(unique_labels)

            system_lower = self.system.lower()
            if system_lower == "kuramoto":
                self.basin_names = [f"Winding q={label}" for label in unique_labels]
            elif system_lower == "competitive_lv":
                self.basin_names = [f"Survivor mask {label}" for label in unique_labels]
            else:
                self.basin_names = [f"Basin {label}" for label in unique_labels]

        for traj, mapped_basin in zip(raw_trajectories, mapped_labels):
            self.trajectories.append(
                BasinLabeledTrajectory(states=traj, final_basin=int(mapped_basin))
            )
