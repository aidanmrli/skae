"""Generate a one-seed spatialized multibasin reaction-diffusion dataset.

Run through SLURM/salloc in this repository, e.g. via
``scripts/run_spatialized_reaction_diffusion_one_seed.sh``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skae.benchmarks.spatialized_reaction_diffusion import (
    SpatialReactionDiffusionConfig,
    generate_dataset,
    save_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the spatialized multibasin reaction-diffusion smoke dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=str, required=True, help="Output .pt path; .h5 requires h5py.")
    parser.add_argument("--source_system", type=str, default="cal_square_4")
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--diffusion", type=float, default=0.01)
    parser.add_argument("--rk4_dt", type=float, default=0.01)
    parser.add_argument("--substeps_per_observation", type=int, default=10)
    parser.add_argument("--trajectory_length", type=int, default=24)
    parser.add_argument("--label_extra_observations", type=int, default=24)
    parser.add_argument("--train_trajectories", type=int, default=48)
    parser.add_argument("--val_trajectories", type=int, default=12)
    parser.add_argument("--test_trajectories", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial_extent", type=float, default=1.0)
    parser.add_argument("--laplacian_scaling", type=str, default="continuum", choices=["continuum", "graph"])
    parser.add_argument("--min_regions", type=int, default=2)
    parser.add_argument("--max_regions", type=int, default=3)
    parser.add_argument("--mask_temperature", type=float, default=0.65)
    parser.add_argument("--low_frequency_cutoff", type=int, default=3)
    parser.add_argument("--noise_scale", type=float, default=0.03)
    parser.add_argument("--require_min_area_fraction", type=float, default=0.08)
    parser.add_argument("--max_initial_condition_attempts", type=int, default=32)
    parser.add_argument("--clip_value", type=float, default=8.0)
    parser.add_argument("--allen_cahn_beta", type=float, default=8.0)
    parser.add_argument("--allen_cahn_reaction_strength", type=float, default=1.0)
    parser.add_argument("--allen_cahn_center_radius", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SpatialReactionDiffusionConfig(
        source_system=args.source_system,
        grid_size=args.grid_size,
        diffusion=args.diffusion,
        rk4_dt=args.rk4_dt,
        substeps_per_observation=args.substeps_per_observation,
        trajectory_length=args.trajectory_length,
        label_extra_observations=args.label_extra_observations,
        train_trajectories=args.train_trajectories,
        val_trajectories=args.val_trajectories,
        test_trajectories=args.test_trajectories,
        seed=args.seed,
        spatial_extent=args.spatial_extent,
        laplacian_scaling=args.laplacian_scaling,
        min_regions=args.min_regions,
        max_regions=args.max_regions,
        mask_temperature=args.mask_temperature,
        low_frequency_cutoff=args.low_frequency_cutoff,
        noise_scale=args.noise_scale,
        require_min_area_fraction=args.require_min_area_fraction,
        max_initial_condition_attempts=args.max_initial_condition_attempts,
        clip_value=args.clip_value,
        allen_cahn_beta=args.allen_cahn_beta,
        allen_cahn_reaction_strength=args.allen_cahn_reaction_strength,
        allen_cahn_center_radius=args.allen_cahn_center_radius,
    )

    bundle = generate_dataset(cfg)
    output = Path(args.output)
    save_dataset(bundle, output)

    summary = {
        "output": str(output),
        "fields_shape": list(bundle["fields"].shape),
        "source_system": bundle["metadata"]["source_system_name"],
        "stored_dt": bundle["metadata"]["stored_dt"],
        "label_horizon_time": bundle["metadata"]["label_horizon_time"],
        "laplacian_scaling": bundle["metadata"]["laplacian_scaling"],
        "laplacian_scale": bundle["metadata"]["laplacian_scale"],
        "split_counts": {
            key: int(value.numel()) for key, value in bundle["split_indices"].items()
        },
        "global_basin_label_counts": {
            str(int(label)): int((bundle["global_basin_labels"] == label).sum().item())
            for label in sorted(bundle["global_basin_labels"].unique().tolist())
        },
        "majority_fraction_mean": float(bundle["majority_fractions"].mean().item()),
        "majority_fraction_min": float(bundle["majority_fractions"].min().item()),
        "observed_majority_fraction_mean": float(bundle["observed_majority_fractions"].mean().item()),
        "observed_majority_fraction_min": float(bundle["observed_majority_fractions"].min().item()),
        "invalid_value_count_total": int(bundle["invalid_value_counts"].sum().item()),
        "clipped_value_count_total": int(bundle["clipped_value_counts"].sum().item()),
        "label_policy": bundle["metadata"]["training_label_policy"],
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
