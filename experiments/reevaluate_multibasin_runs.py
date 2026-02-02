"""Re-evaluate multi-basin dysts checkpoints with long phase portraits."""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from skae.config import Config
from skae.model import make_model
from skae.data import make_env
from skae.evaluation import EvaluationSettings, evaluate_model


def get_device(device_arg: str) -> str:
    """Auto-detect the best available device."""
    if device_arg == 'cpu':
        return 'cpu'
    if device_arg == 'mps':
        if torch.backends.mps.is_available():
            return 'mps'
        print("MPS not available, falling back to CPU")
        return 'cpu'
    if device_arg == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        print("CUDA not available, falling back to CPU")
        return 'cpu'
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    return device_arg


def parse_dims(dims_arg: str) -> Sequence[int]:
    dims = []
    for item in dims_arg.split(","):
        item = item.strip()
        if not item:
            continue
        dims.append(int(item))
    return tuple(sorted(set(dims)))


def find_run_dirs(root: Path) -> List[Path]:
    run_dirs = []
    if not root.exists():
        return run_dirs
    
    for system_dir in sorted(root.iterdir()):
        if not system_dir.is_dir():
            continue
        for run_dir in sorted(system_dir.iterdir()):
            if (run_dir / "config.json").exists():
                run_dirs.append(run_dir)
    return run_dirs


def should_skip_evaluation(
    output_dir: Path,
    system: str,
    dims: Sequence[int],
) -> bool:
    system_dir = output_dir / system
    for plot_dim in dims:
        suffix = "2D" if plot_dim == 2 else "3D"
        portrait_path = system_dir / f"phase_portrait_plot_eval_{suffix}.png"
        if not portrait_path.exists():
            return False
    return True


def load_config(run_dir: Path, checkpoints: Sequence[str], device: str) -> Config:
    config_json_path = run_dir / 'config.json'
    if config_json_path.exists():
        return Config.from_json(str(config_json_path))
    
    for ckpt_name in checkpoints:
        ckpt_path = run_dir / ckpt_name
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=device)
            if 'config' in checkpoint:
                return Config.from_dict(checkpoint['config'])
    raise ValueError(f"No config found for run directory: {run_dir}")


def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint_name: str,
    cfg: Config,
    device: str,
    system: str,
    output_dir: Path,
    settings: EvaluationSettings,
    skip_existing: bool,
) -> None:
    if not checkpoint_path.exists():
        print(f"  Skipping {checkpoint_name}: checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'config' in checkpoint:
        model_cfg = Config.from_dict(checkpoint['config'])
    else:
        model_cfg = cfg
    
    model_env = make_env(model_cfg)
    eval_model = make_model(model_cfg, model_env.observation_size)
    
    eval_cfg = Config.from_dict(model_cfg.to_dict())
    eval_cfg.ENV.ENV_NAME = system
    
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    eval_model = eval_model.to(device)
    eval_model.eval()
    
    output_dir = output_dir / f"evaluation_{checkpoint_name}"
    if skip_existing and should_skip_evaluation(output_dir, system, settings.phase_portrait_dims):
        print(f"  Skipping {checkpoint_name}: plots already exist")
        return
    settings.systems = [system]
    evaluate_model(
        model=eval_model,
        cfg=eval_cfg,
        device=device,
        settings=settings,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate dysts multi-basin runs with long phase portraits."
    )
    parser.add_argument(
        '--runs_roots',
        type=str,
        nargs='+',
        default=[
            'runs/dysts_multi_basin_lista_nonlinear',
            'runs/dysts_multi_basin_generic_sparse',
        ],
        help='Root directories containing system subfolders with run timestamps',
    )
    parser.add_argument(
        '--checkpoints',
        type=str,
        nargs='+',
        default=['checkpoint.pt', 'last.pt'],
        help='Checkpoint files to evaluate (relative to run directory)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'cuda', 'mps', 'auto'],
        help='Device to run evaluation on',
    )
    parser.add_argument(
        '--phase_portrait_length',
        type=int,
        default=30000,
        help='Number of steps for phase portrait trajectories',
    )
    parser.add_argument(
        '--phase_portrait_batch_size',
        type=int,
        default=32,
        help='Batch size for phase portrait trajectories',
    )
    parser.add_argument(
        '--phase_portrait_dims',
        type=str,
        default='2,3',
        help='Comma-separated list of phase portrait dims to save (e.g. 2,3)',
    )
    parser.add_argument(
        '--output_tag',
        type=str,
        default='phase_30000',
        help='Tag for evaluation output directory name',
    )
    parser.add_argument(
        '--systems',
        type=str,
        nargs='+',
        default=None,
        help='Optional list of system names to evaluate (matches folder names)',
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Skip evaluation when phase portrait outputs already exist',
    )
    
    args = parser.parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    settings = EvaluationSettings()
    settings.phase_portrait_length = args.phase_portrait_length
    settings.phase_portrait_batch_size = args.phase_portrait_batch_size
    settings.phase_portrait_dims = parse_dims(args.phase_portrait_dims)
    
    for root_str in args.runs_roots:
        root = Path(root_str)
        run_dirs = find_run_dirs(root)
        if not run_dirs:
            print(f"No run directories found under {root}")
            continue
        
        print(f"Found {len(run_dirs)} runs under {root}")
        
        for run_dir in run_dirs:
            if args.systems is not None:
                if run_dir.parent.name not in args.systems:
                    continue
            print(f"\nEvaluating run: {run_dir}")
            cfg = load_config(run_dir, args.checkpoints, device)
            system = cfg.ENV.ENV_NAME
            output_dir = run_dir / f"reeval_{args.output_tag}"
            
            for ckpt in args.checkpoints:
                ckpt_path = run_dir / ckpt
                ckpt_name = ckpt.replace('.pt', '')
                evaluate_checkpoint(
                    checkpoint_path=ckpt_path,
                    checkpoint_name=ckpt_name,
                    cfg=cfg,
                    device=device,
                    system=system,
                    output_dir=output_dir,
                    settings=settings,
                    skip_existing=args.skip_existing,
                )
    
    print("Re-evaluation complete.")


if __name__ == '__main__':
    main()
