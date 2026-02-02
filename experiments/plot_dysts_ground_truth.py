"""Plot ground-truth trajectories for dysts systems."""

import argparse
from pathlib import Path
from typing import List, Optional

import torch

from skae.config import get_config
from skae.data import make_env, VectorWrapper, generate_trajectory


def get_system_list(systems_arg: str, custom_systems: Optional[List[str]] = None) -> List[str]:
    """Get list of systems based on argument."""
    if systems_arg == 'custom':
        if not custom_systems:
            raise ValueError("Must provide --custom_systems when using --systems custom")
        return custom_systems
    
    try:
        from skae.benchmarks.system_catalog import (
            QUICK_TEST,
            STANDARD_BENCHMARK,
            EXTENDED_BENCHMARK,
            get_multi_attractor_systems,
            get_multi_basin_systems,
            get_multiscroll_systems,
            filter_available_systems,
            get_all_systems,
        )
        
        if systems_arg == 'quick':
            return filter_available_systems(QUICK_TEST)
        if systems_arg == 'standard':
            return filter_available_systems(STANDARD_BENCHMARK)
        if systems_arg == 'extended':
            return filter_available_systems(EXTENDED_BENCHMARK)
        if systems_arg == 'multi_basin':
            return get_multi_basin_systems()
        if systems_arg == 'multi_attractor':
            return get_multi_attractor_systems()
        if systems_arg == 'multi_scroll':
            return get_multiscroll_systems()
        if systems_arg == 'full':
            return get_all_systems()
        raise ValueError(f"Unknown systems argument: {systems_arg}")
    except ImportError as e:
        print(f"Warning: Could not import system catalog: {e}")
        print("Using default quick test systems.")
        return ["Lorenz", "Rossler", "Chen", "Chua"]


def plot_phase_portrait(
    trajectories: torch.Tensor,
    output_path: Path,
    title: str,
    plot_dim: int = 2,
) -> None:
    """Plot phase portrait from trajectories [time, batch, dim]."""
    if trajectories.shape[-1] < 2:
        print(f"Skipping {title}: state dimension < 2")
        return
    
    import matplotlib
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    traj_np = trajectories.cpu().numpy()
    if plot_dim == 3 and trajectories.shape[-1] >= 3:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        for idx in range(traj_np.shape[1]):
            ax.plot(
                traj_np[:, idx, 0],
                traj_np[:, idx, 1],
                traj_np[:, idx, 2],
                alpha=0.8,
                linewidth=1.0,
            )
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        for idx in range(traj_np.shape[1]):
            ax.plot(traj_np[:, idx, 0], traj_np[:, idx, 1], alpha=0.8, linewidth=1.0)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal", adjustable="box")
    
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_trajectories_rk4(
    system_name: str,
    num_steps: int,
    num_trajectories: int,
    standardize: bool,
    ic_noise_scale: float,
    seed: int,
) -> torch.Tensor:
    """Generate trajectories using the same RK4 step used in training."""
    cfg = get_config("default")
    cfg.ENV.ENV_NAME = f"dysts:{system_name}"
    cfg.ENV.DYSTS.STANDARDIZE = standardize
    cfg.ENV.DYSTS.IC_NOISE_SCALE = ic_noise_scale
    
    base_env = make_env(cfg)
    vec_env = VectorWrapper(base_env, num_trajectories)
    rng = torch.Generator().manual_seed(seed)
    init_states = vec_env.reset(rng)
    traj = generate_trajectory(vec_env.step, init_states, length=num_steps)
    return torch.cat([init_states.unsqueeze(0), traj], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ground-truth dysts trajectories")
    parser.add_argument('--systems', type=str, default='multi_basin',
                        choices=[
                            'quick',
                            'standard',
                            'extended',
                            'multi_basin',
                            'multi_attractor',
                            'multi_scroll',
                            'full',
                            'custom',
                        ],
                        help='System set to use')
    parser.add_argument('--custom_systems', type=str, nargs='+', default=None,
                        help='Custom list of system names (requires --systems custom)')
    parser.add_argument('--num_steps', type=int, default=2000,
                        help='Number of steps per trajectory')
    parser.add_argument('--num_trajectories', type=int, default=32,
                        help='Number of trajectories to plot')
    parser.add_argument('--standardize', action='store_true',
                        help='Standardize dysts data (zero mean, unit variance).')
    parser.add_argument('--ic_noise_scale', type=float, default=0.2,
                        help='Scale for initial condition perturbation')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for initial conditions')
    parser.add_argument('--output_dir', type=Path, default=Path('runs/dysts_ground_truth'),
                        help='Base output directory for plots')
    
    args = parser.parse_args()
    systems = get_system_list(args.systems, args.custom_systems)
    
    if not systems:
        print("No systems to plot.")
        return
    
    print(f"Plotting ground-truth trajectories for {len(systems)} systems.")
    print(f"Output directory: {args.output_dir}")
    
    for system_name in systems:
        print(f"Generating trajectories for {system_name}...")
        trajectories = generate_trajectories_rk4(
            system_name=system_name,
            num_steps=args.num_steps,
            num_trajectories=args.num_trajectories,
            standardize=args.standardize,
            ic_noise_scale=args.ic_noise_scale,
            seed=args.seed,
        )
        
        system_dir = args.output_dir / system_name
        title = f"{system_name} ground-truth trajectories"
        
        output_2d = system_dir / "phase_portrait_truth_2D.png"
        plot_phase_portrait(
            trajectories=trajectories,
            output_path=output_2d,
            title=title,
            plot_dim=2,
        )
        print(f"  Saved: {output_2d}")
        
        output_3d = system_dir / "phase_portrait_truth_3D.png"
        plot_phase_portrait(
            trajectories=trajectories,
            output_path=output_3d,
            title=title,
            plot_dim=3,
        )
        print(f"  Saved: {output_3d}")


if __name__ == '__main__':
    main()
