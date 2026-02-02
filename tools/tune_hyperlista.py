"""Grid search for optimal HyperLISTA hyperparameters.

This script performs grid search over the 3 HyperLISTA hyperparameters:
- c_theta: Threshold scaling
- c_beta: Momentum coefficient  
- c_ss: Support selection ratio

Unlike traditional LISTA training which uses backpropagation over O(n² × L)
parameters, HyperLISTA has only 3 scalars, making grid search tractable.

Usage:
    python tune_hyperlista.py --env duffing --target_size 1024
    python tune_hyperlista.py --env dysts:Lorenz --target_size 2048 --fine_grid

Reference: "Hyperparameter Tuning is All You Need for LISTA" (Chen et al., NeurIPS 2021)
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch

from skae.config import Config, get_config
from skae.data import make_env, VectorWrapper
from skae.model import make_model


def evaluate_hyperparams(
    model: torch.nn.Module,
    x: torch.Tensor,
    nx: torch.Tensor,
    c_theta: float,
    c_beta: float,
    c_ss: float,
) -> Dict[str, float]:
    """Evaluate a specific hyperparameter configuration.
    
    Args:
        model: HyperLISTAKM model
        x: Current states [batch_size, obs_size]
        nx: Next states [batch_size, obs_size]
        c_theta: Threshold scaling
        c_beta: Momentum coefficient
        c_ss: Support selection ratio
        
    Returns:
        Dictionary of metrics including loss and sparsity
    """
    # Set hyperparameters (in-place modification)
    model.hyperlista.c_theta.data.fill_(c_theta)
    model.hyperlista.c_beta.data.fill_(c_beta)
    model.hyperlista.c_ss.data.fill_(c_ss)
    
    # Invalidate pseudo-inverse cache
    model.hyperlista._cached_D_pinv = None
    model.hyperlista._cached_D_hash = None
    
    # Single forward pass (no gradients)
    with torch.no_grad():
        loss, metrics = model.loss(x, nx)
    
    return {
        'loss': metrics['loss'],
        'residual_loss': metrics.get('residual_loss', 0.0),
        'reconst_loss': metrics.get('reconst_loss', 0.0),
        'sparsity_ratio': metrics.get('sparsity_ratio', 0.0),
    }


def grid_search(
    cfg: Config,
    device: str = 'cuda',
    batch_size: int = 512,
    c_theta_range: Optional[np.ndarray] = None,
    c_beta_range: Optional[np.ndarray] = None,
    c_ss_range: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Find optimal (c_theta, c_beta, c_ss) via grid search.
    
    Args:
        cfg: Configuration object
        device: Device to run on
        batch_size: Batch size for evaluation
        c_theta_range: Range of c_theta values to search
        c_beta_range: Range of c_beta values to search
        c_ss_range: Range of c_ss values to search
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_params dict, best_metrics dict)
    """
    # Default search grids (coarse)
    if c_theta_range is None:
        c_theta_range = np.linspace(1e-3, 1e-2, 5)
    if c_beta_range is None:
        c_beta_range = np.linspace(1e-3, 1e-2, 5)
    if c_ss_range is None:
        c_ss_range = np.linspace(0.1, 1.0, 5)
    
    # Create environment
    env = make_env(cfg)
    vec_env = VectorWrapper(env, batch_size=batch_size)
    
    # Generate validation data
    rng = torch.Generator().manual_seed(42)
    x = vec_env.reset(rng).to(device)
    nx = vec_env.step(x.cpu()).to(device)
    
    # Create model (with learnable hyperparams disabled for grid search)
    cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = False
    model = make_model(cfg, env.observation_size).to(device)
    model.eval()
    
    best_loss = float('inf')
    best_params = None
    best_metrics = None
    
    total_evals = len(c_theta_range) * len(c_beta_range) * len(c_ss_range)
    eval_count = 0
    
    if verbose:
        print(f"\nGrid search over {total_evals} configurations...")
        print(f"  c_theta: {c_theta_range}")
        print(f"  c_beta: {c_beta_range}")
        print(f"  c_ss: {c_ss_range}")
        print("-" * 60)
    
    for c_theta in c_theta_range:
        for c_beta in c_beta_range:
            for c_ss in c_ss_range:
                eval_count += 1
                
                metrics = evaluate_hyperparams(model, x, nx, c_theta, c_beta, c_ss)
                
                if metrics['loss'] < best_loss:
                    best_loss = metrics['loss']
                    best_params = {
                        'c_theta': float(c_theta),
                        'c_beta': float(c_beta),
                        'c_ss': float(c_ss),
                    }
                    best_metrics = metrics
                    
                    if verbose:
                        print(f"[{eval_count}/{total_evals}] New best: "
                              f"c_theta={c_theta:.4f}, c_beta={c_beta:.4f}, "
                              f"c_ss={c_ss:.4f}, loss={metrics['loss']:.6f}, "
                              f"sparsity={metrics['sparsity_ratio']:.3f}")
    
    if verbose:
        print("-" * 60)
        print(f"Best configuration found:")
        print(f"  c_theta = {best_params['c_theta']:.6f}")
        print(f"  c_beta  = {best_params['c_beta']:.6f}")
        print(f"  c_ss    = {best_params['c_ss']:.6f}")
        print(f"  loss    = {best_metrics['loss']:.6f}")
        print(f"  sparsity_ratio = {best_metrics['sparsity_ratio']:.3f}")
    
    return best_params, best_metrics


def fine_grid_search(
    cfg: Config,
    coarse_params: Dict[str, float],
    device: str = 'cuda',
    batch_size: int = 512,
    resolution: int = 11,
    range_factor: float = 0.5,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Refine search around a coarse solution.
    
    Args:
        cfg: Configuration object
        coarse_params: Best parameters from coarse search
        device: Device to run on
        batch_size: Batch size for evaluation
        resolution: Number of points per dimension
        range_factor: How far to search around coarse solution (as fraction)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (best_params dict, best_metrics dict)
    """
    # Create fine grid around coarse solution
    c_theta_center = coarse_params['c_theta']
    c_beta_center = coarse_params['c_beta']
    c_ss_center = coarse_params['c_ss']
    
    c_theta_range = np.linspace(
        c_theta_center * (1 - range_factor),
        c_theta_center * (1 + range_factor),
        resolution
    ).clip(min=1e-5)
    
    c_beta_range = np.linspace(
        c_beta_center * (1 - range_factor),
        c_beta_center * (1 + range_factor),
        resolution
    ).clip(min=1e-5)
    
    c_ss_range = np.linspace(
        max(0.01, c_ss_center * (1 - range_factor)),
        min(1.0, c_ss_center * (1 + range_factor)),
        resolution
    )
    
    if verbose:
        print(f"\nFine grid search around coarse solution...")
    
    return grid_search(
        cfg, device, batch_size,
        c_theta_range, c_beta_range, c_ss_range,
        verbose=verbose
    )


def main():
    parser = argparse.ArgumentParser(
        description='Grid search for HyperLISTA hyperparameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search on duffing oscillator
  python tune_hyperlista.py --env duffing --target_size 1024

  # Search with fine refinement
  python tune_hyperlista.py --env dysts:Lorenz --target_size 2048 --fine_grid

  # Custom coarse grid
  python tune_hyperlista.py --env pendulum --coarse_resolution 11
        """
    )
    
    parser.add_argument('--env', type=str, default='duffing',
                        help='Environment name')
    parser.add_argument('--target_size', type=int, default=1024,
                        help='Latent dimension')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for evaluation')
    parser.add_argument('--coarse_resolution', type=int, default=5,
                        help='Grid resolution for coarse search')
    parser.add_argument('--fine_grid', action='store_true',
                        help='Perform fine grid search after coarse search')
    parser.add_argument('--fine_resolution', type=int, default=11,
                        help='Grid resolution for fine search')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'mps', 'auto'],
                        help='Device to run on')
    parser.add_argument('--output_dir', type=str, default='./runs/hyperlista_tune',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Create config
    cfg = get_config('hyperlista')
    cfg.ENV.ENV_NAME = args.env
    cfg.MODEL.TARGET_SIZE = args.target_size
    cfg.SEED = args.seed
    
    # Coarse grid
    n = args.coarse_resolution
    c_theta_range = np.linspace(1e-3, 1e-2, n)
    c_beta_range = np.linspace(1e-3, 1e-2, n)
    c_ss_range = np.linspace(0.1, 1.0, n)
    
    # Run coarse search
    print(f"\n{'='*60}")
    print(f"HyperLISTA Hyperparameter Tuning")
    print(f"{'='*60}")
    print(f"Environment: {args.env}")
    print(f"Target size: {args.target_size}")
    print(f"Batch size: {args.batch_size}")
    
    coarse_params, coarse_metrics = grid_search(
        cfg, device, args.batch_size,
        c_theta_range, c_beta_range, c_ss_range,
        verbose=True
    )
    
    # Run fine search if requested
    if args.fine_grid:
        fine_params, fine_metrics = fine_grid_search(
            cfg, coarse_params, device, args.batch_size,
            resolution=args.fine_resolution,
            verbose=True
        )
        best_params = fine_params
        best_metrics = fine_metrics
    else:
        best_params = coarse_params
        best_metrics = coarse_metrics
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results = {
        'env': args.env,
        'target_size': args.target_size,
        'best_params': best_params,
        'best_metrics': best_metrics,
        'coarse_params': coarse_params,
        'coarse_metrics': coarse_metrics,
        'timestamp': timestamp,
        'device': device,
    }
    
    results_file = output_dir / f'tune_results_{args.env}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")
    
    # Print final recommendation
    print(f"\nRecommended config settings:")
    print(f"  cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = {best_params['c_theta']:.6f}")
    print(f"  cfg.MODEL.ENCODER.HYPERLISTA.C_BETA = {best_params['c_beta']:.6f}")
    print(f"  cfg.MODEL.ENCODER.HYPERLISTA.C_SS = {best_params['c_ss']:.6f}")
    print()


if __name__ == '__main__':
    main()
