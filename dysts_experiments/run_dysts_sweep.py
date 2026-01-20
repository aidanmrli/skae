"""Sweep training across multiple dysts systems.

This script enables large-scale benchmarking by training models on
multiple chaotic systems from the dysts library.

Usage:
    # Quick test on 4 systems
    uv run python dysts_experiments/run_dysts_sweep.py --systems quick --config lista --num_steps 5000
    
    # Standard benchmark on 12 systems
    uv run python dysts_experiments/run_dysts_sweep.py --systems standard --config lista --num_steps 10000
    
    # Multi-basin candidates (manual + metadata keywords)
    uv run python dysts_experiments/run_dysts_sweep.py --systems multi_basin --config lista --num_steps 10000
    
    # Custom system list
    uv run python dysts_experiments/run_dysts_sweep.py --systems custom --custom_systems Lorenz Rossler Chua --num_steps 10000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config, Config
from train import train, get_device


def get_system_list(systems_arg: str, custom_systems: Optional[List[str]] = None) -> List[str]:
    """Get list of systems based on argument.
    
    Args:
        systems_arg: One of 'quick', 'standard', 'extended', 'full', 'custom'
        custom_systems: List of custom system names if systems_arg == 'custom'
        
    Returns:
        List of system names to train on.
    """
    if systems_arg == 'custom':
        if not custom_systems:
            raise ValueError("Must provide --custom_systems when using --systems custom")
        return custom_systems
    
    try:
        from benchmarks.system_catalog import (
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
        elif systems_arg == 'standard':
            return filter_available_systems(STANDARD_BENCHMARK)
        elif systems_arg == 'extended':
            return filter_available_systems(EXTENDED_BENCHMARK)
        elif systems_arg == 'multi_basin':
            return get_multi_basin_systems()
        elif systems_arg == 'multi_attractor':
            return get_multi_attractor_systems()
        elif systems_arg == 'multi_scroll':
            return get_multiscroll_systems()
        elif systems_arg == 'full':
            return get_all_systems()
        else:
            raise ValueError(f"Unknown systems argument: {systems_arg}")
    except ImportError as e:
        print(f"Warning: Could not import system catalog: {e}")
        print("Using default quick test systems.")
        return ["Lorenz", "Rossler", "Chen", "Chua"]


def train_on_system(
    system_name: str,
    config_name: str,
    output_dir: Path,
    num_steps: int,
    target_size: Optional[int] = None,
    sparsity_coeff: Optional[float] = None,
    reconst_coeff: Optional[float] = None,
    pred_coeff: Optional[float] = None,
    lista_alpha: Optional[float] = None,
    pairwise: bool = False,
    standardize: bool = False,
    device: str = 'auto',
    seed: int = 0,
) -> Dict[str, Any]:
    """Train a model on a single dysts system.
    
    Args:
        system_name: Name of the dysts system (e.g., "Lorenz")
        config_name: Configuration preset name
        output_dir: Directory for logs and checkpoints
        num_steps: Number of training steps
        target_size: Latent dimension (optional override)
        sparsity_coeff: Sparsity coefficient (optional override)
        reconst_coeff: Reconstruction loss coefficient (optional override)
        pred_coeff: Prediction loss coefficient (optional override)
        lista_alpha: LISTA soft-threshold alpha (optional override)
        pairwise: Use pairwise (single-step) training
        standardize: Standardize dysts data (zero mean, unit variance)
        device: Device to train on
        seed: Random seed
        
    Returns:
        Dictionary with training results.
    """
    cfg = get_config(config_name)
    cfg.ENV.ENV_NAME = f"dysts:{system_name}"
    cfg.TRAIN.NUM_STEPS = num_steps
    cfg.SEED = seed
    
    # Apply overrides
    if target_size is not None:
        cfg.MODEL.TARGET_SIZE = target_size
    if sparsity_coeff is not None:
        cfg.MODEL.SPARSITY_COEFF = sparsity_coeff
    if reconst_coeff is not None:
        cfg.MODEL.RECONST_COEFF = reconst_coeff
    if pred_coeff is not None:
        cfg.MODEL.PRED_COEFF = pred_coeff
    if lista_alpha is not None:
        cfg.MODEL.ENCODER.LISTA.ALPHA = lista_alpha
    if pairwise:
        cfg.TRAIN.USE_SEQUENCE_LOSS = False
    if standardize:
        cfg.ENV.DYSTS.STANDARDIZE = True
    
    log_dir = output_dir / system_name
    
    result = {
        "system": system_name,
        "config": config_name,
        "num_steps": num_steps,
        "seed": seed,
        "status": "pending",
    }
    
    try:
        print(f"\n{'='*60}")
        print(f"Training on: {system_name}")
        print(f"Config: {config_name}, Steps: {num_steps}, Seed: {seed}")
        print('='*60)
        
        # Convert 'auto' to actual device
        actual_device = get_device(device)
        model = train(cfg, log_dir=str(log_dir), device=actual_device)
        
        result["status"] = "success"
        result["log_dir"] = str(log_dir)
        
        # Try to extract final metrics
        try:
            metrics_file = log_dir / "final_metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    result["final_metrics"] = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load final metrics: {e}")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"  FAILED: {e}")
    
    return result


def run_sweep(
    systems: List[str],
    config_name: str,
    output_dir: Path,
    num_steps: int,
    target_size: Optional[int] = None,
    sparsity_coeff: Optional[float] = None,
    reconst_coeff: Optional[float] = None,
    pred_coeff: Optional[float] = None,
    lista_alpha: Optional[float] = None,
    pairwise: bool = False,
    standardize: bool = False,
    device: str = 'auto',
    seeds: List[int] = [0],
) -> Dict[str, List[Dict[str, Any]]]:
    """Run training sweep across multiple systems.
    
    Args:
        systems: List of system names
        config_name: Configuration preset name
        output_dir: Base directory for outputs
        num_steps: Number of training steps per system
        target_size: Latent dimension (optional override)
        sparsity_coeff: Sparsity coefficient (optional override)
        reconst_coeff: Reconstruction loss coefficient (optional override)
        pred_coeff: Prediction loss coefficient (optional override)
        lista_alpha: LISTA soft-threshold alpha (optional override)
        pairwise: Use pairwise (single-step) training
        standardize: Standardize dysts data
        device: Device to train on
        seeds: List of random seeds for multiple runs
        
    Returns:
        Dictionary mapping system names to list of results (one per seed).
    """
    all_results = {}
    
    total_runs = len(systems) * len(seeds)
    current_run = 0
    
    for system_name in systems:
        system_results = []
        
        for seed in seeds:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Starting: {system_name} (seed={seed})")
            
            result = train_on_system(
                system_name=system_name,
                config_name=config_name,
                output_dir=output_dir / f"seed_{seed}",
                num_steps=num_steps,
                target_size=target_size,
                sparsity_coeff=sparsity_coeff,
                reconst_coeff=reconst_coeff,
                pred_coeff=pred_coeff,
                lista_alpha=lista_alpha,
                pairwise=pairwise,
                standardize=standardize,
                device=device,
                seed=seed,
            )
            system_results.append(result)
        
        all_results[system_name] = system_results
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Sweep training across multiple dysts systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  uv run python dysts_experiments/run_dysts_sweep.py --systems quick --config lista --num_steps 5000
  
  # Standard benchmark with multiple seeds
  uv run python dysts_experiments/run_dysts_sweep.py --systems standard --config lista --seeds 0 1 2
  
  # Multi-basin candidates
  uv run python dysts_experiments/run_dysts_sweep.py --systems multi_basin --config lista
  
  # Custom systems
  uv run python dysts_experiments/run_dysts_sweep.py --systems custom --custom_systems Lorenz Chua Chen
        """
    )
    
    # System selection
    parser.add_argument('--systems', type=str, default='quick',
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
    
    # Training configuration
    parser.add_argument('--config', type=str, default='lista',
                        choices=['generic', 'generic_sparse', 'lista', 'lista_nonlinear'],
                        help='Training configuration preset')
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='Number of training steps per system')
    parser.add_argument('--target_size', type=int, default=None,
                        help='Latent dimension (optional override)')
    parser.add_argument('--sparsity_coeff', type=float, default=None,
                        help='Sparsity coefficient (optional override)')
    parser.add_argument('--reconst_coeff', type=float, default=None,
                        help='Reconstruction loss coefficient (optional override)')
    parser.add_argument('--pred_coeff', type=float, default=None,
                        help='Prediction loss coefficient (optional override)')
    parser.add_argument('--lista_alpha', type=float, default=None,
                        help='LISTA soft-threshold alpha (optional override)')
    parser.add_argument('--pairwise', action='store_true',
                        help='Use pairwise (single-step) training instead of sequence training')
    parser.add_argument('--standardize', action='store_true',
                        help='Standardize dysts data (zero mean, unit variance). Recommended.')
    
    # Reproducibility
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='Random seeds for multiple runs')
    
    # Output
    parser.add_argument('--output_dir', type=Path, default=Path('runs/dysts_sweep'),
                        help='Base directory for outputs')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'mps', 'auto'],
                        help='Device to train on')
    
    # Utilities
    parser.add_argument('--dry_run', action='store_true',
                        help='Print systems and exit without training')
    
    args = parser.parse_args()
    
    # Get system list
    systems = get_system_list(args.systems, args.custom_systems)
    
    if not systems:
        print("No systems to train on!")
        return
    
    print(f"\n{'='*60}")
    print(f"DYSTS SWEEP CONFIGURATION")
    print('='*60)
    print(f"Systems ({len(systems)}): {systems}")
    print(f"Config: {args.config}")
    print(f"Steps per system: {args.num_steps}")
    print(f"Seeds: {args.seeds}")
    print(f"Total runs: {len(systems) * len(args.seeds)}")
    print(f"Output directory: {args.output_dir}")
    print('='*60)
    
    if args.dry_run:
        print("\n[DRY RUN] Would train on the above systems. Exiting.")
        return
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = args.output_dir / f"{args.config}_{args.systems}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sweep configuration
    sweep_config = {
        "systems": systems,
        "config": args.config,
        "num_steps": args.num_steps,
        "target_size": args.target_size,
        "sparsity_coeff": args.sparsity_coeff,
        "reconst_coeff": args.reconst_coeff,
        "pred_coeff": args.pred_coeff,
        "lista_alpha": args.lista_alpha,
        "pairwise": args.pairwise,
        "standardize": args.standardize,
        "seeds": args.seeds,
        "device": args.device,
        "timestamp": timestamp,
    }
    with open(sweep_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)
    
    # Run sweep
    results = run_sweep(
        systems=systems,
        config_name=args.config,
        output_dir=sweep_dir,
        num_steps=args.num_steps,
        target_size=args.target_size,
        sparsity_coeff=args.sparsity_coeff,
        reconst_coeff=args.reconst_coeff,
        pred_coeff=args.pred_coeff,
        lista_alpha=args.lista_alpha,
        pairwise=args.pairwise,
        standardize=args.standardize,
        device=args.device,
        seeds=args.seeds,
    )
    
    # Save results
    with open(sweep_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print('='*60)
    
    success_count = 0
    fail_count = 0
    
    for system_name, system_results in results.items():
        for result in system_results:
            if result["status"] == "success":
                success_count += 1
                print(f"  ✓ {system_name} (seed={result.get('seed', 0)})")
            else:
                fail_count += 1
                print(f"  ✗ {system_name} (seed={result.get('seed', 0)}): {result.get('error', 'Unknown error')}")
    
    print(f"\nSuccess: {success_count}/{success_count + fail_count}")
    print(f"Results saved to: {sweep_dir}")
    print('='*60)


if __name__ == '__main__':
    main()
