"""
Standalone script to evaluate trained checkpoints.

This script loads checkpoint.pt and last.pt from a training run directory
and runs the standardized evaluation suite on them.

Usage:
    python evaluate_checkpoints.py --run_dir runs/kae/20251114-111432 --system duffing
    python evaluate_checkpoints.py --run_dir runs/kae/20251114-111432 --system pendulum --device cpu
    python evaluate_checkpoints.py --run_dir runs/kae/20251114-111432 --system lorenz63 --checkpoints checkpoint.pt
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from skae.config import Config
from skae.dysts_cache_profiles import apply_dysts_cache_profile, default_dysts_cache_dir
from skae.data import make_env
from skae.model import make_model
from skae.evaluation import EvaluationSettings, evaluate_model


def remap_legacy_model_keys(eval_model: torch.nn.Module, state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remap known legacy checkpoint key prefixes to current model names."""
    model_keys = set(eval_model.state_dict().keys())
    has_encoder_in_model = any(key.startswith("encoder.") for key in model_keys)
    has_encoder_in_ckpt = any(key.startswith("encoder.") for key in state_dict.keys())
    has_lista_in_ckpt = any(key.startswith("lista.") for key in state_dict.keys())

    # Older LISTAKM checkpoints used `lista.*`; current code expects `encoder.*`.
    if has_encoder_in_model and has_lista_in_ckpt and not has_encoder_in_ckpt:
        remapped: Dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.startswith("lista."):
                remapped[f"encoder.{key[len('lista.') :]}"] = value
            else:
                remapped[key] = value
        print("  Applied legacy checkpoint remap: 'lista.*' -> 'encoder.*'")
        return remapped

    return state_dict


def get_device(device_arg: str) -> str:
    """Auto-detect the best available device.
    
    Priority order:
    1. Use explicitly requested device if available
    2. MPS (Metal Performance Shaders) on macOS
    3. CUDA on Linux/Windows
    4. CPU as fallback
    
    Args:
        device_arg: Requested device ('cpu', 'cuda', 'mps', or 'auto')
        
    Returns:
        Device string ('cpu', 'cuda', or 'mps')
    """
    # If explicitly CPU, use it
    if device_arg == 'cpu':
        return 'cpu'
    
    # If explicitly MPS, check availability
    if device_arg == 'mps':
        if torch.backends.mps.is_available():
            return 'mps'
        else:
            print("MPS not available, falling back to CPU")
            return 'cpu'
    
    # If explicitly CUDA, check availability
    if device_arg == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            print("CUDA not available, falling back to CPU")
            return 'cpu'
    
    # Auto-detect: prefer MPS on macOS, then CUDA, then CPU
    if device_arg == 'auto' or device_arg == 'cuda':
        # Check MPS first (macOS)
        if torch.backends.mps.is_available():
            return 'mps'
        # Then CUDA (Linux/Windows with GPU)
        elif torch.cuda.is_available():
            return 'cuda'
        # Fallback to CPU
        else:
            return 'cpu'
    
    return device_arg


def get_dt_from_config(cfg: Config) -> float:
    """Extract dt from environment config based on ENV_NAME."""
    env_name = cfg.ENV.ENV_NAME.lower()
    if env_name == 'duffing':
        return cfg.ENV.DUFFING.DT
    elif env_name == 'pendulum':
        return cfg.ENV.PENDULUM.DT
    elif env_name == 'lotka_volterra':
        return cfg.ENV.LOTKA_VOLTERRA.DT
    elif env_name == 'lorenz63':
        return cfg.ENV.LORENZ63.DT
    elif env_name == 'parabolic':
        return cfg.ENV.PARABOLIC.DT
    elif env_name == 'lyapunov':
        return cfg.ENV.LYAPUNOV.DT
    elif env_name == 'kuramoto':
        return cfg.ENV.KURAMOTO.DT
    elif env_name == 'hopfield':
        return cfg.ENV.HOPFIELD.DT
    elif env_name == 'competitive_lv':
        return cfg.ENV.COMPETITIVE_LV.DT
    else:
        return 0.01  # default fallback


def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint_name: str,
    cfg: Config,
    device: str,
    system: str,
    use_dysts_cache: bool = True,
    dysts_cache_profile: str = "full",
    dysts_cache_split: str = "test",
    dysts_cache_dir: Optional[str] = None,
    dysts_cache_num_workers: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint and evaluate it.
    
    Args:
        checkpoint_path: Path to checkpoint file
        checkpoint_name: Name identifier for this checkpoint (e.g., "best", "last")
        cfg: Configuration object
        device: Device to run evaluation on
        system: System/environment name to evaluate on
        output_dir: Optional directory to save evaluation results
        
    Returns:
        Evaluation results dictionary or None if checkpoint not found
    """
    if not checkpoint_path.exists():
        print(f"  Skipping {checkpoint_name}: checkpoint not found at {checkpoint_path}")
        return None
    
    print(f"\nEvaluating {checkpoint_name} checkpoint...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_step = checkpoint.get('step', 'unknown')
    print(f"  Loaded checkpoint (step={ckpt_step}). Building eval env/model...", flush=True)
    
    # Load config from checkpoint if available, otherwise use provided cfg
    # Keep original config for model creation (model architecture depends on training observation size)
    if 'config' in checkpoint:
        model_cfg = Config.from_dict(checkpoint['config'])
    else:
        model_cfg = cfg
    
    # Create model using original config (from training)
    model_env = make_env(model_cfg)
    eval_model = make_model(model_cfg, model_env.observation_size)
    
    # Create evaluation config with specified system for environment creation
    eval_cfg = Config.from_dict(model_cfg.to_dict())
    eval_cfg.ENV.ENV_NAME = system
    if system.lower().startswith("dysts:") and use_dysts_cache:
        eval_cfg.ENV.DYSTS.USE_NATIVE_CACHE = True
        eval_cfg.ENV.DYSTS.CACHE_REUSE = True
        eval_cfg.ENV.DYSTS.CACHE_SPLIT = dysts_cache_split
        apply_dysts_cache_profile(eval_cfg, dysts_cache_profile)
        if dysts_cache_dir:
            eval_cfg.ENV.DYSTS.CACHE_DIR = str(dysts_cache_dir)
        elif not eval_cfg.ENV.DYSTS.CACHE_DIR:
            eval_cfg.ENV.DYSTS.CACHE_DIR = default_dysts_cache_dir()
        if dysts_cache_num_workers is not None:
            eval_cfg.ENV.DYSTS.CACHE_NUM_WORKERS = int(dysts_cache_num_workers)
        print(
            "  Dysts eval cache enabled: "
            f"split={eval_cfg.ENV.DYSTS.CACHE_SPLIT}, "
            f"profile={dysts_cache_profile}, "
            f"dir='{eval_cfg.ENV.DYSTS.CACHE_DIR}'"
        )

    # Verify observation size compatibility and derive dt from instantiated env.
    eval_env = make_env(eval_cfg)
    dt = getattr(eval_env.unwrapped, "dt", get_dt_from_config(eval_cfg))
    if eval_env.observation_size != eval_model.observation_size:
        print(
            f"  WARNING: System '{system}' has observation size {eval_env.observation_size} "
            f"but model expects {eval_model.observation_size}. "
            f"Evaluation may fail or be skipped."
        )
    model_state_dict = remap_legacy_model_keys(eval_model, checkpoint['model_state_dict'])
    eval_model.load_state_dict(model_state_dict)
    eval_model = eval_model.to(device)
    eval_model.eval()
    eval_model.dt = dt
    
    # Create evaluation settings
    eval_settings = EvaluationSettings()
    eval_settings.systems = [system]
    
    # Evaluate
    if output_dir is None:
        output_dir = checkpoint_path.parent / f"evaluation_{checkpoint_name}"
    else:
        output_dir = output_dir / f"evaluation_{checkpoint_name}"
    
    print(f"  Calling evaluate_model() for system={system} ...", flush=True)
    eval_results = evaluate_model(
        model=eval_model,
        cfg=eval_cfg,
        device=device,
        settings=eval_settings,
        output_dir=output_dir,
    )
    print(f"  evaluate_model() finished for {checkpoint_name}.", flush=True)
    
    # Save results
    results_file = checkpoint_path.parent / f"evaluation_results_{checkpoint_name}.json"
    with open(results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Print summary
    print(f"  {checkpoint_name.upper()} - Evaluation summary:")
    system_metrics = eval_results.get(system)
    if system_metrics is not None:
        print(f"    System: {system}")
        for horizon in eval_settings.horizons:
            if system == "parabolic" and horizon > 100:
                continue
            horizon_key = str(horizon)
            modes = system_metrics.get("modes", {})
            no_re = modes.get("no_reencode", {}).get("horizons", {}).get(horizon_key)
            every = modes.get("every_step", {}).get("horizons", {}).get(horizon_key)
            best = system_metrics.get("best_periodic", {}).get(horizon_key)
            if no_re is None or every is None:
                continue
            best_str = "best-PR=N/A" if best is None else f"best-PR={best['mean']:.4e} ({best['mode']})"
            print(
                f"      Horizon {horizon}: "
                f"no-reencode={no_re['mean']:.4e}, "
                f"every-step={every['mean']:.4e}, "
                f"{best_str}"
            )
    
    print(f"  Evaluation artifacts saved to {output_dir}")
    return eval_results


def main():
    """Command-line interface for checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained Koopman Autoencoder checkpoints',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='Path to training run directory containing checkpoints'
    )
    
    # Optional system override
    parser.add_argument(
        '--system',
        type=str,
        default=None,
        help=(
            "System/environment to evaluate on. If omitted, uses the run config ENV_NAME "
            "(supports built-in and dysts systems such as 'dysts:Lorenz')."
        ),
    )
    
    # Optional arguments
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        default=['checkpoint.pt', 'last.pt'],
        help='Checkpoint files to evaluate (relative to run_dir)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'cuda', 'mps', 'auto'],
        help='Device to run evaluation on (auto: auto-detect best available)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for evaluation results (default: run_dir)'
    )
    parser.add_argument(
        '--disable_dysts_cache',
        action='store_true',
        help='Disable dysts trajectory cache during evaluation'
    )
    parser.add_argument(
        '--dysts_cache_profile',
        type=str,
        default='full',
        choices=['smoke', 'full'],
        help='Named dysts cache profile for evaluation'
    )
    parser.add_argument(
        '--dysts_cache_split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Cache split namespace for evaluation'
    )
    parser.add_argument(
        '--dysts_cache_dir',
        type=str,
        default=None,
        help='Optional cache directory (defaults to shared cache path)'
    )
    parser.add_argument(
        '--dysts_cache_num_workers',
        type=int,
        default=None,
        help='Parallel workers for cache build fallback'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    
    # Auto-detect device
    device = get_device(args.device)
    print(f"Using device: {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device == 'mps':
        print("  Using Metal Performance Shaders (MPS)")
    else:
        print("  Using CPU")
    
    # Load config from checkpoint or config.json
    cfg = None
    config_json_path = run_dir / 'config.json'
    if config_json_path.exists():
        print(f"Loading config from {config_json_path}")
        cfg = Config.from_json(str(config_json_path))
    else:
        # Try to load from first checkpoint
        first_checkpoint = run_dir / args.checkpoints[0]
        if first_checkpoint.exists():
            print(f"Loading config from checkpoint {first_checkpoint}")
            checkpoint = torch.load(first_checkpoint, map_location=device)
            if 'config' in checkpoint:
                cfg = Config.from_dict(checkpoint['config'])
            else:
                raise ValueError(
                    f"No config found in checkpoint and no config.json in {run_dir}. "
                    "Cannot determine model configuration."
                )
        else:
            raise ValueError(
                f"No config.json found and no checkpoint available. "
                f"Cannot determine model configuration."
            )
    
    eval_system = args.system if args.system else cfg.ENV.ENV_NAME
    print(f"Configuration loaded: {cfg.ENV.ENV_NAME} system, {cfg.MODEL.MODEL_NAME} model")
    print(f"Evaluation system: {eval_system}")
    if eval_system.lower().startswith("dysts:"):
        if args.disable_dysts_cache:
            print("Dysts cache: disabled")
        else:
            cache_dir = args.dysts_cache_dir if args.dysts_cache_dir else default_dysts_cache_dir()
            print(
                "Dysts cache: "
                f"enabled (split={args.dysts_cache_split}, profile={args.dysts_cache_profile}, dir='{cache_dir}')"
            )
    print("-" * 80)
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    
    # Evaluate each checkpoint
    all_results = {}
    for checkpoint_name in args.checkpoints:
        checkpoint_path = run_dir / checkpoint_name
        # Keep legacy naming for compatibility with collector tooling.
        if checkpoint_name == "checkpoint.pt":
            name_key = "best"
        elif checkpoint_name == "last.pt":
            name_key = "last"
        else:
            name_key = checkpoint_name.replace('.pt', '')
        
        results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            checkpoint_name=name_key,
            cfg=cfg,
            device=device,
            system=eval_system,
            use_dysts_cache=not args.disable_dysts_cache,
            dysts_cache_profile=args.dysts_cache_profile,
            dysts_cache_split=args.dysts_cache_split,
            dysts_cache_dir=args.dysts_cache_dir,
            dysts_cache_num_workers=args.dysts_cache_num_workers,
            output_dir=output_dir,
        )
        
        if results is not None:
            all_results[name_key] = results
    
    # Save combined summary
    if all_results:
        summary = {
            "run_dir": str(run_dir),
            "evaluated_checkpoints": list(all_results.keys()),
            "system": eval_system,
        }
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nCombined summary saved to {summary_file}")
    
    print("-" * 80)
    print(f"Evaluation complete! Results saved to {output_dir}")
    
    return all_results


if __name__ == '__main__':
    main()
