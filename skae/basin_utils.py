"""Utilities for basin-of-attraction labeling of dynamical-system trajectories.

Provides functions to identify which basin of attraction a trajectory converges
to, and a dataset class that pairs trajectories with their ground-truth basin
labels.  These are used by multiple evaluation tools (support uniqueness,
eigenvalue analysis, latent clustering, etc.).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

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
    raw_final_basin: int | None = None


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
        sampling_strategy: str = "random",
        target_raw_labels: Sequence[int] | None = None,
        trajectories_per_basin: int | None = None,
        max_attempts: int | None = None,
    ):
        self.system = system
        self.cfg = cfg
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length
        self.long_rollout_steps = long_rollout_steps
        self.seed = seed
        self.sampling_strategy = str(sampling_strategy).lower()
        self.target_raw_labels = (
            [int(label) for label in target_raw_labels]
            if target_raw_labels
            else None
        )
        self.trajectories_per_basin = trajectories_per_basin
        self.max_attempts = max_attempts

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
        self.raw_basin_labels: List[int] = []
        self.raw_to_mapped_label: Dict[int, int] = {}
        self.raw_basin_distribution: Dict[int, int] = {}
        self.mapped_basin_distribution: Dict[int, int] = {}
        self._generate_trajectories()

    def _draw_trajectory(self, sample_index: int) -> tuple[torch.Tensor, int]:
        env_rng = torch.Generator().manual_seed(self.seed + sample_index)
        init_state = self.env.reset(env_rng)

        traj = generate_trajectory(
            self.env.step,
            init_state,
            length=self.trajectory_length,
        )
        traj = torch.cat([init_state.unsqueeze(0), traj], dim=0)

        raw_label = identify_env_basin(self.env, traj, self.long_rollout_steps)
        return traj, raw_label

    def _build_basin_names(self, raw_labels: Sequence[int]) -> List[str]:
        system_lower = self.system.lower()
        if system_lower == "kuramoto":
            return [f"Winding q={label}" for label in raw_labels]
        if system_lower == "competitive_lv":
            return [f"Survivor mask {label}" for label in raw_labels]
        return [f"Basin {label}" for label in raw_labels]

    def _generate_random_trajectories(self) -> tuple[List[torch.Tensor], List[int]]:
        raw_trajectories: List[torch.Tensor] = []
        raw_labels: List[int] = []
        for i in range(self.num_trajectories):
            traj, raw_label = self._draw_trajectory(i)
            raw_trajectories.append(traj)
            raw_labels.append(raw_label)
        return raw_trajectories, raw_labels

    def _generate_balanced_trajectories(self) -> tuple[List[torch.Tensor], List[int]]:
        if self.trajectories_per_basin is None or self.trajectories_per_basin < 1:
            raise ValueError(
                "Balanced sampling requires `trajectories_per_basin >= 1`."
            )
        if not self.target_raw_labels:
            raise ValueError(
                "Balanced sampling requires explicit `target_raw_labels`."
            )

        target_raw_labels = list(dict.fromkeys(int(label) for label in self.target_raw_labels))
        counts = {label: 0 for label in target_raw_labels}
        raw_trajectories: List[torch.Tensor] = []
        raw_labels: List[int] = []

        max_attempts = self.max_attempts
        if max_attempts is None:
            max_attempts = len(target_raw_labels) * self.trajectories_per_basin * 50

        attempt = 0
        while any(count < self.trajectories_per_basin for count in counts.values()):
            if attempt >= max_attempts:
                missing = {
                    label: self.trajectories_per_basin - count
                    for label, count in counts.items()
                    if count < self.trajectories_per_basin
                }
                raise RuntimeError(
                    "Failed to sample requested basin quotas. "
                    f"targets={target_raw_labels}, counts={counts}, missing={missing}, "
                    f"max_attempts={max_attempts}"
                )

            traj, raw_label = self._draw_trajectory(attempt)
            if raw_label in counts and counts[raw_label] < self.trajectories_per_basin:
                raw_trajectories.append(traj)
                raw_labels.append(raw_label)
                counts[raw_label] += 1
            attempt += 1

        self.num_trajectories = len(raw_trajectories)
        return raw_trajectories, raw_labels

    def _generate_trajectories(self):
        print(
            f"Generating trajectories for {self.system} "
            f"(sampling={self.sampling_strategy})..."
        )
        if self.sampling_strategy == "balanced":
            raw_trajectories, raw_labels = self._generate_balanced_trajectories()
        elif self.sampling_strategy == "random":
            raw_trajectories, raw_labels = self._generate_random_trajectories()
        else:
            raise ValueError(
                f"Unknown sampling_strategy '{self.sampling_strategy}'. "
                "Expected 'random' or 'balanced'."
            )

        has_fixed_index_space = (
            self.num_basins > 0
            and all(0 <= label < self.num_basins for label in raw_labels)
            and self.target_raw_labels is None
        )

        if has_fixed_index_space:
            mapped_labels = raw_labels
            self.raw_basin_labels = list(range(self.num_basins))
            self.raw_to_mapped_label = {label: label for label in self.raw_basin_labels}
            if not self.basin_names:
                self.basin_names = [f"Basin {i}" for i in range(self.num_basins)]
        else:
            if self.target_raw_labels is not None:
                unique_labels = list(dict.fromkeys(int(label) for label in self.target_raw_labels))
            else:
                unique_labels = sorted(set(raw_labels))
            label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            mapped_labels = [label_to_index[label] for label in raw_labels]
            self.raw_basin_labels = unique_labels
            self.raw_to_mapped_label = dict(label_to_index)
            self.num_basins = len(unique_labels)
            self.basin_names = self._build_basin_names(unique_labels)

        if not self.raw_basin_labels:
            self.raw_basin_labels = list(range(self.num_basins))
        if not self.raw_to_mapped_label:
            self.raw_to_mapped_label = {
                label: idx for idx, label in enumerate(self.raw_basin_labels)
            }

        self.raw_basin_distribution = dict(sorted(Counter(raw_labels).items()))
        self.mapped_basin_distribution = dict(sorted(Counter(mapped_labels).items()))

        for traj, raw_label, mapped_basin in zip(raw_trajectories, raw_labels, mapped_labels):
            self.trajectories.append(
                BasinLabeledTrajectory(
                    states=traj,
                    final_basin=int(mapped_basin),
                    raw_final_basin=int(raw_label),
                )
            )
