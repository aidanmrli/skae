"""Diagnose cosine separability sensitivity for latent representations.

Runs cosine separation across multiple aggregation modes and transformations
(raw, demeaned, PC1-removed) and saves a JSON summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from skae.config import Config
from skae.data import make_env
from skae.model import make_model

from evaluate_support_uniqueness import compute_cosine_diagnostics


def _parse_aggregations(value: str) -> List[str]:
    items = [v.strip() for v in value.split(',') if v.strip()]
    if not items:
        raise ValueError("No aggregations provided")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cosine separation diagnostics across aggregations"
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
    parser.add_argument('--aggregations', type=str, default='mean,median,mean_abs',
                        help='Comma-separated cosine aggregations to test')
    parser.add_argument('--output_dir', type=str, default='results/cosine_diagnostics',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for trajectory generation')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run on')

    args = parser.parse_args()

    aggregations = _parse_aggregations(args.aggregations)

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

    from skae.basin_utils import BasinLabeledDataset

    dataset = BasinLabeledDataset(
        system=system,
        cfg=cfg,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        long_rollout_steps=args.long_rollout_steps,
        seed=args.seed,
    )

    diagnostics = compute_cosine_diagnostics(
        model,
        dataset,
        args.device,
        aggregations=aggregations,
    )

    out_path = output_dir / 'cosine_diagnostics.json'
    with open(out_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nSaved cosine diagnostics to {out_path}")
    print("\nSummary (cosine separation):")
    for agg, variants in diagnostics.items():
        for variant, metrics in variants.items():
            print(
                f"  {agg:>8} | {variant:>11} | sep={metrics.get('cosine_separation_score', 0.0):.4f} "
                f"intra={metrics.get('mean_intra_basin_cosine', 0.0):.4f} "
                f"inter={metrics.get('mean_inter_basin_cosine', 0.0):.4f}"
            )


if __name__ == '__main__':
    main()
