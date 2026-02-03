"""
Evaluate whether each basin of attraction maps to a unique sparse support pattern.

This is a mechanistic-style diagnostic: do different basins consistently
activate different sets of latent coordinates?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from skae.config import Config
from skae.data import make_env, generate_trajectory, Duffing, LyapunovMultiAttractor
from skae.model import make_model


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


def identify_lyapunov_basin(
    env: LyapunovMultiAttractor,
    trajectory: torch.Tensor,
    long_rollout_steps: int = 5000,
) -> int:
    """Identify Lyapunov basin by nearest attractor after convergence."""
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
        elif system_lower == 'lyapunov':
            self.num_basins = int(self.env.points.shape[0])
            self.basin_names = [f'Attractor {i}' for i in range(self.num_basins)]
        else:
            raise ValueError(
                f"Unknown system: {system}. Supported: duffing, lyapunov"
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
                final_basin = identify_lyapunov_basin(self.env, traj, self.long_rollout_steps)

            self.trajectories.append(
                BasinLabeledTrajectory(states=traj, final_basin=final_basin)
            )


# ---------------------------------------------------------------------------
# Support Uniqueness Metrics
# ---------------------------------------------------------------------------


@dataclass
class SupportUniquenessResults:
    system_name: str
    model_name: str
    num_trajectories: int
    num_basins: int
    latent_dim: int
    support_threshold: float
    support_mode: str
    unique_mode_supports: int
    mode_collision_pairs: int
    mode_uniqueness_rate: float
    mean_basin_consistency: float
    mean_mode_support_size: float
    mean_pairwise_jaccard: float
    per_basin_consistency: Dict[int, float] = field(default_factory=dict)
    per_basin_support_size: Dict[int, float] = field(default_factory=dict)
    per_basin_active_indices: Dict[int, List[int]] = field(default_factory=dict)


def _support_from_latents(
    latents: torch.Tensor,
    threshold: float,
    mode: str,
) -> np.ndarray:
    if mode == "mean":
        z = latents.mean(dim=0)
        support = (z.abs() > threshold).cpu().numpy()
    elif mode == "last":
        z = latents[-1]
        support = (z.abs() > threshold).cpu().numpy()
    elif mode == "median":
        z = latents.median(dim=0).values
        support = (z.abs() > threshold).cpu().numpy()
    elif mode == "majority":
        votes = (latents.abs() > threshold).float().mean(dim=0)
        support = (votes > 0.5).cpu().numpy()
    else:
        raise ValueError(f"Unknown support mode '{mode}'")
    return support.astype(np.int8)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def compute_support_uniqueness(
    model,
    dataset: BasinLabeledDataset,
    device: str,
    support_threshold: float,
    support_mode: str,
) -> SupportUniquenessResults:
    model.eval()
    basin_supports: Dict[int, List[Tuple[int, ...]]] = {b: [] for b in range(dataset.num_basins)}

    with torch.no_grad():
        for traj in dataset.trajectories:
            states = traj.states.to(device)
            z = model.encode(states)
            support = _support_from_latents(z, support_threshold, support_mode)
            basin_supports[traj.final_basin].append(tuple(support.tolist()))

    # Mode support per basin + consistency
    per_basin_consistency: Dict[int, float] = {}
    per_basin_support_size: Dict[int, float] = {}
    per_basin_active_indices: Dict[int, List[int]] = {}
    basin_mode_supports: Dict[int, Tuple[int, ...]] = {}
    for basin, supports in basin_supports.items():
        if not supports:
            per_basin_consistency[basin] = 0.0
            per_basin_support_size[basin] = 0.0
            continue
        counts: Dict[Tuple[int, ...], int] = {}
        for s in supports:
            counts[s] = counts.get(s, 0) + 1
        mode_support, mode_count = max(counts.items(), key=lambda kv: kv[1])
        basin_mode_supports[basin] = mode_support
        per_basin_consistency[basin] = mode_count / max(1, len(supports))
        per_basin_support_size[basin] = float(np.sum(mode_support))
        per_basin_active_indices[basin] = [i for i, v in enumerate(mode_support) if v == 1]

    # Uniqueness across basins
    mode_support_list = list(basin_mode_supports.values())
    unique_mode_supports = len(set(mode_support_list))
    total_pairs = dataset.num_basins * (dataset.num_basins - 1) // 2
    collision_pairs = 0
    jaccards = []
    basins = sorted(basin_mode_supports.keys())
    for i, bi in enumerate(basins):
        si = np.array(basin_mode_supports[bi], dtype=np.int8)
        for bj in basins[i + 1:]:
            sj = np.array(basin_mode_supports[bj], dtype=np.int8)
            if np.array_equal(si, sj):
                collision_pairs += 1
            jaccards.append(_jaccard(si, sj))

    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
    uniqueness_rate = 1.0 - (collision_pairs / max(1, total_pairs))

    # Aggregate stats
    mean_consistency = float(np.mean(list(per_basin_consistency.values()))) if per_basin_consistency else 0.0
    mean_support_size = float(np.mean(list(per_basin_support_size.values()))) if per_basin_support_size else 0.0

    return SupportUniquenessResults(
        system_name=dataset.system,
        model_name=type(model).__name__,
        num_trajectories=len(dataset.trajectories),
        num_basins=dataset.num_basins,
        latent_dim=model.target_size,
        support_threshold=support_threshold,
        support_mode=support_mode,
        unique_mode_supports=unique_mode_supports,
        mode_collision_pairs=collision_pairs,
        mode_uniqueness_rate=uniqueness_rate,
        mean_basin_consistency=mean_consistency,
        mean_mode_support_size=mean_support_size,
        mean_pairwise_jaccard=mean_jaccard,
        per_basin_consistency=per_basin_consistency,
        per_basin_support_size=per_basin_support_size,
        per_basin_active_indices=per_basin_active_indices,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate support uniqueness across basins"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (last.pt or checkpoint.pt)')
    parser.add_argument('--system', type=str, default=None,
                        help='System to evaluate (defaults to checkpoint env). Supported: duffing, lyapunov')
    parser.add_argument('--num_trajectories', type=int, default=100,
                        help='Number of test trajectories')
    parser.add_argument('--trajectory_length', type=int, default=500,
                        help='Length of each trajectory')
    parser.add_argument('--long_rollout_steps', type=int, default=5000,
                        help='Steps for basin identification after trajectory end')
    parser.add_argument('--support_threshold', type=float, default=1e-3,
                        help='Threshold for nonzero support')
    parser.add_argument('--support_mode', type=str, default='mean',
                        choices=['mean', 'last', 'median', 'majority'],
                        help='How to aggregate support over a trajectory')
    parser.add_argument('--output_dir', type=str, default='results/support_uniqueness',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for trajectory generation')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run on')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = Config.from_dict(checkpoint['config'])

    if args.system is not None:
        cfg.ENV.ENV_NAME = args.system
    system = cfg.ENV.ENV_NAME
    print(f"Evaluating on system: {system}")

    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    dataset = BasinLabeledDataset(
        system=system,
        cfg=cfg,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        long_rollout_steps=args.long_rollout_steps,
        seed=args.seed,
    )

    results = compute_support_uniqueness(
        model,
        dataset,
        device=args.device,
        support_threshold=args.support_threshold,
        support_mode=args.support_mode,
    )

    results_path = output_dir / "support_uniqueness.json"
    with open(results_path, "w") as f:
        json.dump(asdict(results), f, indent=2)

    print("\nSupport uniqueness results:")
    print(f"  Unique mode supports: {results.unique_mode_supports}/{results.num_basins}")
    print(f"  Mode collisions (pairs): {results.mode_collision_pairs}")
    print(f"  Mode uniqueness rate: {results.mode_uniqueness_rate:.3f}")
    print(f"  Mean basin consistency: {results.mean_basin_consistency:.3f}")
    print(f"  Mean mode support size: {results.mean_mode_support_size:.1f}")
    print(f"  Mean pairwise Jaccard: {results.mean_pairwise_jaccard:.3f}")
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
