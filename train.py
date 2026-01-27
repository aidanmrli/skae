"""
Training script for Koopman Autoencoder models.

This script provides a complete training pipeline for learning Koopman operator
representations of dynamical systems using PyTorch.

Usage:
    python train.py --config generic_sparse --env duffing --num_steps 20000

Or use it programmatically:
    from train import train
    cfg = get_config("generic_sparse")
    cfg.ENV.ENV_NAME = "duffing"
    train(cfg, log_dir="./runs/experiment_001")
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

print("Loading torch...")
import torch
import torch.nn as nn
print("Torch loaded.")

print("Loading config...")
from config import Config, get_config
print("Config loaded.")

print("Loading data...")
from data import make_env, VectorWrapper, generate_trajectory, DystsTrajectoryCache
print("Data loaded.")

print("Loading model...")
from model import make_model
print("Model loaded.")

# Lazy import evaluation - only load when needed
print("All core imports loaded.")


class MetricsLogger:
    """Simple file-based metrics logger.
    
    Logs metrics to JSON files for later analysis or plotting.
    Can easily be replaced with wandb later.
    Uses buffered writes to reduce I/O overhead.
    """
    
    def __init__(self, log_dir: Path, flush_interval: int = 100):
        self.log_dir = log_dir
        self.metrics_file = log_dir / 'metrics_history.jsonl'
        self.metrics_history: List[Dict] = []
        self.buffer: List[str] = []
        self.flush_interval = flush_interval
        self.step_count = 0
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar metric."""
        entry = {
            'step': step,
            'name': name,
            'value': value,
        }
        # Buffer writes to reduce I/O overhead
        self.buffer.append(json.dumps(entry) + '\n')
        self.metrics_history.append(entry)
        self.step_count += 1
        
        # Flush buffer periodically
        if len(self.buffer) >= self.flush_interval:
            self.flush()
    
    def flush(self):
        """Flush buffered metrics to disk."""
        if self.buffer:
            with open(self.metrics_file, 'a') as f:
                f.writelines(self.buffer)
            self.buffer.clear()
    
    def log_dict(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log a dictionary of metrics."""
        for key, value in metrics.items():
            name = f"{prefix}/{key}" if prefix else key
            self.log_scalar(name, value, step)
    
    def close(self):
        """Save final summary and flush any remaining buffered writes."""
        # Flush any remaining buffered metrics
        self.flush()
        
        summary_file = self.log_dir / 'metrics_summary.json'
        
        # Compute summary statistics
        summary = {}
        metrics_by_name = {}
        for entry in self.metrics_history:
            name = entry['name']
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            metrics_by_name[name].append(entry['value'])
        
        for name, values in metrics_by_name.items():
            summary[name] = {
                'final': values[-1] if values else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None,
                'mean': sum(values) / len(values) if values else None,
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    nx: torch.Tensor,
    cfg: Config,
    dt: float,
    step: int = 0,
) -> Dict[str, float]:
    """Perform one training step.
    
    Args:
        model: Koopman machine model
        optimizer: PyTorch optimizer
        x: Current states [batch_size, observation_size] OR
           sequence [batch_size, seq_len, observation_size] if USE_SEQUENCE_LOSS=True
        nx: Next states [batch_size, observation_size] (unused if USE_SEQUENCE_LOSS=True)
        cfg: Configuration object
        dt: Time step for ODE integration
        step: Current training step (for StructuredLISTAKM exclusivity warmup)
        
    Returns:
        Dictionary of metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Compute loss
    if cfg.TRAIN.USE_SEQUENCE_LOSS:
        # x is a sequence: [batch_size, seq_len, observation_size]
        # StructuredLISTAKM needs step for exclusivity warmup
        if hasattr(model, 'get_exclusivity_weight'):
            loss, metrics = model.loss_sequence(x, dt, step=step)
        else:
            loss, metrics = model.loss_sequence(x, dt)
    else:
        # Standard single-step loss (StructuredLISTAKM needs step for warmup)
        if hasattr(model, 'get_exclusivity_weight'):
            loss, metrics = model.loss(x, nx, step=step)
        else:
            loss, metrics = model.loss(x, nx)
    
    # Backward pass
    loss.backward()
    optimizer.step()

    return metrics


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Create optimizer with a specific learning rate for the Koopman matrix.
    
    This constructs parameter groups so that parameters named with 'kmat' use
    cfg.TRAIN.K_MATRIX_LR while all other parameters use cfg.TRAIN.LR.
    """
    kmat_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'kmat' in name:
            kmat_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': cfg.TRAIN.LR,
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        })
    if kmat_params:
        param_groups.append({
            'params': kmat_params,
            'lr': cfg.TRAIN.K_MATRIX_LR,
            'weight_decay': 0.0,  # No weight decay on Koopman matrix
        })

    return torch.optim.AdamW(param_groups)


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    env_step_fn,
    num_steps: int = 50,
) -> Dict[str, Any]:
    """Quick evaluation helper used during training and unit tests."""
    
    # Lazy import to avoid loading evaluation module at startup
    from evaluation import rollout_every_step_reencode

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        true_traj = generate_trajectory(env_step_fn, x.cpu(), length=num_steps)
        pred_traj = rollout_every_step_reencode(model, x.to(device), num_steps)

        pred_traj_cpu = pred_traj.cpu()
        diff = pred_traj_cpu - true_traj
        nonfinite = ~torch.isfinite(pred_traj_cpu)
        if nonfinite.any():
            nonfinite_ratio = nonfinite.float().mean().item()
            print(
                "  Eval warning: non-finite predictions detected "
                f"({nonfinite_ratio:.2%} of entries). "
                "Using NaN-safe aggregation for errors."
            )

        # Per-step error: [time, batch] -> average over batch with NaN-safe mean.
        step_error = torch.norm(diff, dim=-1)
        step_error = torch.nanmean(step_error, dim=1)

        return {
            "true_trajectory": true_traj,
            "pred_trajectory": pred_traj_cpu,
            "pred_error": step_error,
            "mean_error": torch.nanmean(step_error).item(),
            "final_error": torch.nanmean(step_error[-1]).item(),
        }



def train(
    cfg: Config,
    log_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
) -> nn.Module:
    """Main training function.
    
    Args:
        cfg: Configuration object
        log_dir: Directory for tensorboard logs and checkpoints
        checkpoint_path: Path to checkpoint to resume from
        device: Device to train on ('cpu', 'cuda', 'mps')
        
    Returns:
        Trained model
    """
    print("Initializing training...")
    
    # Setup logging directory and save config
    if log_dir is None:
        if cfg.MODEL.MODEL_NAME == 'LISTAKM':
            log_dir = './runs/lista'
        elif cfg.MODEL.MODEL_NAME == 'HyperLISTAKM':
            log_dir = './runs/hyperlista'
        else:
            log_dir = './runs/kae'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(log_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(str(run_dir / 'config.json'))
    
    logger = MetricsLogger(run_dir)
    
    print("Setting random seed...")
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)
    # MPS doesn't have manual_seed, but manual_seed should be sufficient
    
    print("Creating environment...")
    base_env = make_env(cfg)
    env = VectorWrapper(base_env, cfg.TRAIN.BATCH_SIZE)
    
    # DYSTS-specific training caveat: without the cache we only see x->x+1 from freshly
    # reset initial conditions (i.e., mostly transients, not attractor distribution).
    # This often yields models that "look fine" on 1-step error but fail catastrophically
    # in long rollouts / phase portraits.
    try:
        from benchmarks.dysts_adapter import DystsEnv
        if isinstance(env.unwrapped, DystsEnv) and not cfg.ENV.DYSTS.USE_NATIVE_CACHE:
            print(
                "[warn] Training on a dysts system without trajectory cache. "
                "You are sampling only one-step transitions from reset() each step; "
                "for chaotic/multi-basin systems this often trains poorly for long rollouts. "
                "Consider adding --dysts_native_cache (and CACHE_WARMUP) and lowering "
                "--dysts_ic_noise_scale (e.g. 0.2).",
                flush=True,
            )
    except Exception:
        pass
    
    # Get dt from environment config for ODE integration
    env_name = cfg.ENV.ENV_NAME.lower()
    
    # Check if it's a dysts environment
    if env_name.startswith('dysts:') or hasattr(env.unwrapped, 'dt'):
        # For dysts environments, get dt from the unwrapped environment
        dt = getattr(env.unwrapped, 'dt', 0.01)
    elif env_name == 'duffing':
        dt = cfg.ENV.DUFFING.DT
    elif env_name == 'pendulum':
        dt = cfg.ENV.PENDULUM.DT
    elif env_name == 'lotka_volterra':
        dt = cfg.ENV.LOTKA_VOLTERRA.DT
    elif env_name == 'lorenz63':
        dt = cfg.ENV.LORENZ63.DT
    elif env_name == 'parabolic':
        dt = cfg.ENV.PARABOLIC.DT
    elif env_name == 'lyapunov':
        dt = cfg.ENV.LYAPUNOV.DT
    else:
        # Try to get dt from the unwrapped environment
        dt = getattr(env.unwrapped, 'dt', 0.01)
    
    print("Creating model...")
    model = make_model(cfg, env.observation_size)
    model = model.to(device)
    model.dt = dt  # Store dt in model for use in ODE integration
    
    print("Building optimizer...")
    optimizer = build_optimizer(model, cfg)
    
    start_step = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        print(f"Resumed from checkpoint at step {start_step}")
    
    # Pre-generate random number generators for data
    # Each batch gets a non-overlapping seed range to avoid collisions
    # Batch i uses seeds: cfg.SEED + i * BATCH_SIZE to cfg.SEED + (i+1) * BATCH_SIZE - 1
    num_batches = cfg.TRAIN.DATA_SIZE // cfg.TRAIN.BATCH_SIZE
    rngs = [torch.Generator().manual_seed(cfg.SEED + i * cfg.TRAIN.BATCH_SIZE) for i in range(num_batches)]

    # Generate fixed validation set for consistent evaluation
    # Optional dysts cache for faster data generation
    dysts_cache = None
    if cfg.ENV.DYSTS.USE_NATIVE_CACHE:
        try:
            from benchmarks.dysts_adapter import DystsEnv
            if isinstance(env.unwrapped, DystsEnv):
                print(
                    "Initializing dysts native trajectory cache "
                    f"(steps={cfg.ENV.DYSTS.CACHE_STEPS}, "
                    f"trajectories={cfg.ENV.DYSTS.CACHE_TRAJECTORIES}, "
                    f"warmup={cfg.ENV.DYSTS.CACHE_WARMUP}) ...",
                    flush=True,
                )
                cache_rng = torch.Generator().manual_seed(cfg.SEED + 123456)
                dysts_cache = DystsTrajectoryCache(env.unwrapped, cfg, cache_rng)
                print(
                    "Using dysts native trajectory cache "
                    f"(steps={cfg.ENV.DYSTS.CACHE_STEPS}, "
                    f"trajectories={cfg.ENV.DYSTS.CACHE_TRAJECTORIES})"
                )
            else:
                print("Warning: Dysts cache enabled but environment is not dysts.")
        except Exception as e:
            print(f"Warning: Failed to initialize dysts cache: {e}")
    
    print("Generating fixed validation set...")
    val_rng = torch.Generator().manual_seed(cfg.SEED + 999999) # Separate seed
    if cfg.TRAIN.USE_SEQUENCE_LOSS:
        val_seq = env.generate_sequence_batch(val_rng, window_length=cfg.TRAIN.SEQUENCE_LENGTH)
        val_x = val_seq[:16, 0, :].to(device) # Use 16 samples for better stability
    else:
        val_x = env.reset(val_rng)[:16].to(device)
    
    print(f"Training {cfg.MODEL.MODEL_NAME} on {cfg.ENV.ENV_NAME}")
    print(f"Device: {device}")
    print(f"Observation size: {env.observation_size}")
    print(f"Target size: {cfg.MODEL.TARGET_SIZE}")
    print(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    print(f"Total steps: {cfg.TRAIN.NUM_STEPS}")
    print(f"Log directory: {run_dir}")
    print("-" * 80)
    
    best_eval_final_error = float('inf')
    
    for step in range(start_step, cfg.TRAIN.NUM_STEPS):
        # Generate batch
        rng = rngs[step % num_batches]
        
        if cfg.TRAIN.USE_SEQUENCE_LOSS:
            if dysts_cache is not None:
                x_seq = dysts_cache.sample_sequence_batch(
                    rng,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    window_length=cfg.TRAIN.SEQUENCE_LENGTH,
                    device=device,
                )
            else:
                # Generate sequence windows
                x_seq = env.generate_sequence_batch(rng, window_length=cfg.TRAIN.SEQUENCE_LENGTH)
                # x_seq has shape [batch_size, seq_len+1, obs_size]
                x_seq = x_seq.to(device)
            nx = None  # Not used for sequence loss
            metrics = train_step(model, optimizer, x_seq, nx, cfg, dt, step=step)
        else:
            if dysts_cache is not None:
                x, nx = dysts_cache.sample_pair_batch(
                    rng,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    device=device,
                )
            else:
                # Generate single transitions (backward compatibility)
                x = env.reset(rng)
                nx = env.step(x)
                x = x.to(device)
                nx = nx.to(device)
            metrics = train_step(model, optimizer, x, nx, cfg, dt, step=step)
        
        logger.log_dict(metrics, step, prefix='train')
        
        if step % 100 == 0:
            if cfg.TRAIN.USE_SEQUENCE_LOSS:
                print(f"Step {step}/{cfg.TRAIN.NUM_STEPS} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Align: {metrics['alignment_loss']:.4f} | "
                      f"Recon: {metrics['reconst_loss']:.4f} | "
                      f"Pred: {metrics['prediction_loss']:.4f} | "
                      f"Sparsity: {metrics['sparsity_ratio']:.3f}")
            else:
                log_str = (f"Step {step}/{cfg.TRAIN.NUM_STEPS} | "
                           f"Loss: {metrics['loss']:.4f} | "
                           f"Res: {metrics['residual_loss']:.4f} | "
                           f"Recon: {metrics['reconst_loss']:.4f} | "
                           f"Pred: {metrics['prediction_loss']:.4f} | "
                           f"Sparsity: {metrics['sparsity_ratio']:.3f}")
                if 'homogeneous_loss' in metrics:
                    log_str += f" | Homog: {metrics['homogeneous_loss']:.4f}"
                print(log_str)
        
        # Periodic evaluation and checkpoint saving
        # Note: skip step=0 to avoid expensive eval before any learning happened.
        if (step > 0 and step % cfg.TRAIN.EVAL_EVERY == 0) or step == cfg.TRAIN.NUM_STEPS - 1:
            # Use fixed validation set
            eval_results = evaluate(
                model,
                val_x,
                lambda s: env.step(s),
                num_steps=cfg.TRAIN.EVAL_NUM_STEPS,
            )
            logger.log_scalar('eval/mean_error', eval_results['mean_error'], step)
            logger.log_scalar('eval/final_error', eval_results['final_error'], step)
            
            print(f"  Eval | Mean error: {eval_results['mean_error']:.4f} | "
                  f"Final error: {eval_results['final_error']:.4f}")
            
            # Save checkpoint
            checkpoint_dict = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg.to_dict(),
                'metrics': metrics,
            }
            
            # Save latest checkpoint
            torch.save(checkpoint_dict, run_dir / 'last.pt')
            
            # Save best checkpoint if eval error improved
            if eval_results['final_error'] < best_eval_final_error:
                best_eval_final_error = eval_results['final_error']
                torch.save(checkpoint_dict, run_dir / 'checkpoint.pt')
                print(f"  Saved best checkpoint (final eval error: {best_eval_final_error:.4f})")
    
    # Save final metrics and close logger
    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.close()

    # Plot training metrics
    print("-" * 80)
    print("Plotting training metrics...")
    from plot_training_metrics import plot_metrics
    try:
        plot_metrics(
            log_dir=run_dir,
            metrics_to_plot=None,  # Plot all metrics
            save_path=run_dir / 'training_metrics.png'
        )
        print(f"Training metrics plot saved to {run_dir / 'training_metrics.png'}")
    except Exception as e:
        print(f"Warning: Failed to plot training metrics: {e}")
        print("Continuing with evaluation...")

    print("-" * 80)
    print("Running standardized evaluation suite...")
    print("Loading evaluation module...")
    from evaluation import EvaluationSettings, evaluate_model
    
    def evaluate_checkpoint(checkpoint_path: Path, checkpoint_name: str):
        """Load a checkpoint and evaluate it."""
        if not checkpoint_path.exists():
            print(f"  Skipping {checkpoint_name}: checkpoint not found at {checkpoint_path}")
            return None
        
        print(f"\nEvaluating {checkpoint_name} checkpoint...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ckpt_step = checkpoint.get('step', 'unknown')
        print(f"  Loaded checkpoint (step={ckpt_step}). Building eval env/model...", flush=True)
        
        # Load model from checkpoint (use unwrapped env for observation_size)
        eval_env = make_env(cfg)
        eval_model = make_model(cfg, eval_env.observation_size)
        eval_model.load_state_dict(checkpoint['model_state_dict'])
        eval_model = eval_model.to(device)
        eval_model.eval()
        eval_model.dt = dt
        
        # Create evaluation settings
        eval_settings = EvaluationSettings()
        eval_settings.systems = [cfg.ENV.ENV_NAME]
        
        # Evaluate
        eval_dir = run_dir / f"evaluation_{checkpoint_name}"
        print(f"  Calling evaluate_model() for systems={eval_settings.systems} ...", flush=True)
        eval_results = evaluate_model(
            model=eval_model,
            cfg=cfg,
            device=device,
            settings=eval_settings,
            output_dir=eval_dir,
        )
        print(f"  evaluate_model() finished for {checkpoint_name}.", flush=True)
        
        # Save results
        results_file = run_dir / f"evaluation_results_{checkpoint_name}.json"
        with open(results_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        
        # Print summary
        primary_system = cfg.ENV.ENV_NAME
        primary_metrics = eval_results.get(primary_system)
        if primary_metrics is not None:
            print(f"  {checkpoint_name.upper()} - Primary system ({primary_system}) MSE summary:")
            is_dysts = primary_system.lower().startswith("dysts:")
            for horizon in eval_settings.horizons:
                if primary_system == "parabolic" and horizon > 100:
                    continue
                horizon_key = str(horizon)
                no_re = primary_metrics["modes"]["no_reencode"]["horizons"].get(horizon_key)
                every = primary_metrics["modes"]["every_step"]["horizons"].get(horizon_key)
                best = primary_metrics["best_periodic"].get(horizon_key)
                if no_re is None or every is None:
                    continue
                best_str = "best-PR=N/A" if best is None else f"best-PR={best['mean']:.4e} ({best['mode']})"
                print(
                    f"    Horizon {horizon}: "
                    f"no-reencode={no_re['mean']:.4e}, "
                    f"every-step={every['mean']:.4e}, "
                    f"{best_str}"
                )
            
            # For dysts systems, also print reencode @ 100, 200, 500, 1000 summary
            if is_dysts:
                print(f"  {checkpoint_name.upper()} - Periodic reencode summary (reencode @ 100, 200, 500, 1000):")
                for period in [100, 200, 500, 1000]:
                    mode_key = f"periodic_{period}"
                    mode_data = primary_metrics["modes"].get(mode_key)
                    if mode_data is None:
                        print(f"    reencode @ {period}: N/A")
                        continue
                    # Print horizon 100, 500, and 1000 MSE for each periodic mode
                    print(f"    reencode @ {period}:")
                    for horizon in [100, 500, 1000]:
                        horizon_key = str(horizon)
                        horizon_mse = mode_data["horizons"].get(horizon_key)
                        if horizon_mse is not None:
                            print(f"      H={horizon}: mean={horizon_mse['mean']:.4e}, std={horizon_mse['std']:.4e}")
        
        print(f"  Evaluation artifacts saved to {eval_dir}")
        return eval_results
    
    # Evaluate both checkpoints
    last_checkpoint = run_dir / 'last.pt'
    best_checkpoint = run_dir / 'checkpoint.pt'
    
    eval_results_last = evaluate_checkpoint(last_checkpoint, "last")
    eval_results_best = evaluate_checkpoint(best_checkpoint, "best")
    
    # Also save a combined summary
    if eval_results_last is not None or eval_results_best is not None:
        summary = {
            "last_checkpoint": eval_results_last is not None,
            "best_checkpoint": eval_results_best is not None,
        }
        summary_file = run_dir / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    # Run basin structure evaluation for StructuredLISTAKM models
    if cfg.MODEL.STRUCTURED.ENABLED:
        print("-" * 80)
        print("Running basin structure evaluation for StructuredLISTAKM...")
        try:
            from evaluate_basin_structure import (
                BasinLabeledDataset,
                BasinStructureAnalyzer,
                compute_basin_assignment_accuracy,
                plot_phase_portrait_basin_comparison,
                plot_confusion_matrix,
                plot_basin_norm_timeseries,
                plot_activation_distributions,
            )
            from model import StructuredLISTAKM
            from dataclasses import asdict

            # Load best checkpoint for basin analysis
            best_ckpt_path = run_dir / 'checkpoint.pt'
            if best_ckpt_path.exists():
                print(f"Loading best checkpoint for basin analysis...")
                ckpt = torch.load(best_ckpt_path, map_location=device)

                # Create model and load weights
                basin_eval_env = make_env(cfg)
                basin_eval_model = make_model(cfg, basin_eval_env.observation_size)
                basin_eval_model.load_state_dict(ckpt['model_state_dict'])
                basin_eval_model = basin_eval_model.to(device)
                basin_eval_model.eval()

                if isinstance(basin_eval_model, StructuredLISTAKM):
                    print(f"Model: {basin_eval_model.num_basins} model basins, "
                          f"d_global={basin_eval_model.d_global}, d_basin={basin_eval_model.d_basin}")

                    # Create basin-labeled dataset
                    # Only run for systems with known basins (duffing, lyapunov)
                    system_name = cfg.ENV.ENV_NAME.lower()
                    if system_name in ['duffing', 'lyapunov']:
                        print(f"Generating basin-labeled trajectories for {system_name}...")
                        basin_dataset = BasinLabeledDataset(
                            system=system_name,
                            cfg=cfg,
                            num_trajectories=100,
                            trajectory_length=500,
                            seed=cfg.SEED + 777777,  # Different seed for eval
                        )

                        # Run analysis
                        basin_output_dir = run_dir / "basin_structure_analysis"
                        basin_output_dir.mkdir(parents=True, exist_ok=True)

                        analyzer = BasinStructureAnalyzer(
                            basin_eval_model, basin_dataset, device=device
                        )
                        results = analyzer.run_full_analysis()

                        # Get best mapping for visualizations
                        _, confusion_matrix, best_mapping = compute_basin_assignment_accuracy(
                            analyzer.activations_list,
                            analyzer.ground_truth_basins,
                            basin_dataset.num_basins,
                            basin_eval_model.num_basins,
                        )

                        # Save results
                        results_path = basin_output_dir / 'analysis_results.json'
                        with open(results_path, 'w') as f:
                            json.dump(asdict(results), f, indent=2)
                        print(f"Saved basin analysis results to {results_path}")

                        # Generate visualizations
                        print("Generating basin structure visualizations...")

                        plot_phase_portrait_basin_comparison(
                            basin_dataset,
                            analyzer.activations_list,
                            best_mapping,
                            output_path=basin_output_dir / 'phase_portrait_comparison.png',
                        )

                        plot_confusion_matrix(
                            confusion_matrix,
                            basin_dataset.basin_names,
                            basin_eval_model.num_basins,
                            output_path=basin_output_dir / 'confusion_matrix.png',
                        )

                        # Basin norm timeseries for a few examples
                        for i in range(min(5, len(basin_dataset))):
                            plot_basin_norm_timeseries(
                                analyzer.activations_list[i],
                                basin_dataset.trajectories[i].final_basin,
                                title=f'Trajectory {i} (GT Basin: {basin_dataset.trajectories[i].final_basin})',
                                output_path=basin_output_dir / f'basin_norms_traj_{i}.png',
                            )

                        plot_activation_distributions(
                            analyzer.activations_list,
                            analyzer.ground_truth_basins,
                            basin_dataset.num_basins,
                            output_path=basin_output_dir / 'activation_distributions.png',
                        )

                        # Print summary
                        print("\n" + "=" * 60)
                        print("BASIN STRUCTURE ANALYSIS SUMMARY")
                        print("=" * 60)
                        print(f"System: {results.system_name}")
                        print(f"Ground-truth basins: {results.num_ground_truth_basins}")
                        print(f"Model basins: {results.num_model_basins}")
                        print("-" * 60)
                        print(f"Basin Assignment Accuracy: {results.basin_assignment_accuracy:.4f}")
                        print(f"Temporal Consistency: {results.temporal_consistency:.4f}")
                        print(f"Mean Activation Entropy: {results.mean_activation_entropy:.4f}")
                        print(f"Within-Basin Similarity: {results.within_basin_similarity:.4f}")
                        print(f"Cross-Basin Separation: {results.cross_basin_separation:.4f}")
                        print("-" * 60)
                        print("Per-Basin Accuracy:")
                        for gt_basin, acc in results.per_basin_accuracy.items():
                            basin_name = basin_dataset.basin_names[gt_basin] if gt_basin < len(basin_dataset.basin_names) else f"Basin {gt_basin}"
                            print(f"  {basin_name}: {acc:.4f}")
                        print("=" * 60)
                        print(f"Basin structure artifacts saved to {basin_output_dir}")
                    else:
                        print(f"Skipping basin structure analysis: system '{system_name}' "
                              "does not have known basin labels (supported: duffing, lyapunov)")
                else:
                    print("Warning: Model is not StructuredLISTAKM, skipping basin analysis")
            else:
                print("Warning: Best checkpoint not found, skipping basin structure analysis")
        except Exception as e:
            print(f"Warning: Basin structure evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 80)
    print(f"Training complete! Checkpoints saved to {run_dir}")
    
    return model


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


def main():
    """Command-line interface for training."""
    print("Starting train.py...")
    parser = argparse.ArgumentParser(
        description='Train Koopman Autoencoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on built-in environment
  python train.py --config generic_sparse --env duffing --num_steps 10000

  # Train on dysts chaotic system
  python train.py --config lista --env dysts:Lorenz --num_steps 10000
  python train.py --config lista --env dysts:Chua --target_size 1024

  # List available dysts systems
  python train.py --list-dysts
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='generic',
                        choices=['default', 'generic', 'generic_sparse', 
                                'generic_prediction', 'lista', 'lista_nonlinear', 'hyperlista'],
                        help='Training configuration preset')
    parser.add_argument('--env', type=str, default='duffing',
                        help='Environment name. Built-in: duffing, pendulum, lotka_volterra, '
                             'lorenz63, parabolic, lyapunov. '
                             'For dysts systems: use "dysts:SystemName" (e.g., "dysts:Lorenz", "dysts:Chua")')
    
    # Dysts utilities
    parser.add_argument('--list-dysts', action='store_true',
                        help='List all available dysts systems and exit')
    parser.add_argument('--standardize', action='store_true',
                        help='Standardize dysts data (zero mean, unit variance). Recommended for dysts systems.')
    parser.add_argument('--dysts_ic_noise_scale', type=float, default=None,
                        help='Dysts IC noise scale (perturbation around default IC). '
                             'Smaller values keep trajectories near the canonical attractor.')
    parser.add_argument('--dysts_native_cache', action='store_true',
                        help='Use native dysts trajectory cache for training data')
    parser.add_argument('--dysts_cache_steps', type=int, default=None,
                        help='Length of each cached dysts trajectory')
    parser.add_argument('--dysts_cache_trajectories', type=int, default=None,
                        help='Number of cached dysts trajectories')
    parser.add_argument('--dysts_cache_warmup', type=int, default=None,
                        help='Warmup steps to discard from cached trajectories')
    
    # Training
    parser.add_argument('--num_steps', type=int, default=20000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config default)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Model
    parser.add_argument('--target_size', type=int, default=None,
                        help='Latent dimension (overrides config default)')
    parser.add_argument('--sparsity_coeff', type=float, default=None,
                        help='Sparsity loss weight (overrides config default)')
    parser.add_argument('--reconst_coeff', type=float, default=None,
                        help='Reconstruction loss weight (overrides config default)')
    parser.add_argument('--pred_coeff', type=float, default=None,
                        help='Prediction loss weight (overrides config default)')
    parser.add_argument('--lista_alpha', type=float, default=None,
                        help='LISTA soft-threshold alpha (overrides config default)')
    parser.add_argument('--lista_num_loops', type=int, default=None,
                        help='Number of LISTA iterations (overrides config default)')

    # HyperLISTA (HyperLISTAKM) hyperparameters
    parser.add_argument('--hyperlista_c_theta', type=float, default=None,
                        help='HyperLISTA threshold scaling C_THETA (overrides config default)')
    parser.add_argument('--hyperlista_c_beta', type=float, default=None,
                        help='HyperLISTA momentum scaling C_BETA (overrides config default)')
    parser.add_argument('--hyperlista_c_ss', type=float, default=None,
                        help='HyperLISTA support-selection scaling C_SS (overrides config default)')
    
    # Structured latent space (StructuredLISTAKM)
    parser.add_argument('--structured', action='store_true',
                        help='Enable structured latent space with basin-aware Koopman')
    parser.add_argument('--d_global', type=int, default=None,
                        help='Global block dimension (default: 8)')
    parser.add_argument('--num_basins', type=int, default=None,
                        help='Number of basin slots (default: 20)')
    parser.add_argument('--d_basin', type=int, default=None,
                        help='Per-basin block dimension (default: 8)')
    parser.add_argument('--lambda_global', type=float, default=None,
                        help='Global sparsity weight (default: 1e-4)')
    parser.add_argument('--lambda_local', type=float, default=None,
                        help='Local sparsity weight (default: 1e-3)')
    parser.add_argument('--lambda_exclusivity', type=float, default=None,
                        help='Final exclusivity penalty weight (default: 1e-2)')
    parser.add_argument('--lambda_sparsity', type=float, default=None,
                        help='Explicit L1 sparsity weight on full z (default: 1e-3)')
    parser.add_argument('--lambda_entropy', type=float, default=None,
                        help='Entropy-based exclusivity weight (penalizes multiple active basins, default: 0)')
    parser.add_argument('--lambda_dominance', type=float, default=None,
                        help='Top-1 dominance loss weight (encourages one basin to dominate, default: 0)')
    parser.add_argument('--lambda_temporal', type=float, default=None,
                        help='Temporal consistency loss weight for sequence training (penalizes basin changes within trajectory, default: 0)')
    parser.add_argument('--excl_warmup_steps', type=int, default=None,
                        help='Steps to ramp exclusivity/sparsity from 0 to final (default: 1000)')
    
    # Training mode
    parser.add_argument('--pairwise', action='store_true',
                        help='Use pairwise (single-step) training instead of sequence training')
    parser.add_argument('--sequence', action='store_true',
                        help='Use sequence (multi-step ODE) training instead of pairwise')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Sequence length for sequence training (overrides config default)')
    parser.add_argument('--eval_every', type=int, default=None,
                        help='Evaluate every N steps during training (overrides config default)')
    parser.add_argument('--eval_num_steps', type=int, default=None,
                        help='Rollout horizon for the quick eval during training (overrides config default)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for logs and checkpoints (defaults to ./runs/kae, or ./runs/lista for LISTA configs)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'mps', 'auto'],
                        help='Device to train on (auto: auto-detect best available)')
    
    args = parser.parse_args()
    
    # Handle --list-dysts
    if args.list_dysts:
        print("\n" + "=" * 60)
        print("AVAILABLE DYSTS SYSTEMS")
        print("=" * 60)
        try:
            from data import get_available_environments
            envs = get_available_environments()
            
            print(f"\nBuilt-in environments ({len(envs['builtin'])}):")
            for env in envs['builtin']:
                print(f"  {env}")
            
            print(f"\nDysts systems ({len(envs['dysts'])}):")
            if envs['dysts']:
                # Print in columns
                systems = envs['dysts']
                for i in range(0, len(systems), 4):
                    row = systems[i:i+4]
                    print("  " + "  ".join(f"{s:<20}" for s in row))
                print(f"\nUsage: --env dysts:SystemName (e.g., --env dysts:Lorenz)")
            else:
                print("  (dysts library not available)")
                print("  Install with: pip install dysts")
        except Exception as e:
            print(f"Error listing environments: {e}")
        
        print("=" * 60 + "\n")
        return
    
    # Create config
    cfg = get_config(args.config)
    cfg.ENV.ENV_NAME = args.env
    cfg.TRAIN.NUM_STEPS = args.num_steps
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed
    
    # Override config with command-line args
    if args.lr is not None:
        cfg.TRAIN.LR = args.lr
    if args.target_size is not None:
        cfg.MODEL.TARGET_SIZE = args.target_size
    if args.sparsity_coeff is not None:
        cfg.MODEL.SPARSITY_COEFF = args.sparsity_coeff
    if args.reconst_coeff is not None:
        cfg.MODEL.RECONST_COEFF = args.reconst_coeff
    if args.pred_coeff is not None:
        cfg.MODEL.PRED_COEFF = args.pred_coeff
    if args.lista_alpha is not None:
        cfg.MODEL.ENCODER.LISTA.ALPHA = args.lista_alpha
    if args.lista_num_loops is not None:
        # Update both LISTA and HyperLISTA loop counts for convenience
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = args.lista_num_loops
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = args.lista_num_loops
    if args.hyperlista_c_theta is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = args.hyperlista_c_theta
    if args.hyperlista_c_beta is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.C_BETA = args.hyperlista_c_beta
    if args.hyperlista_c_ss is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.C_SS = args.hyperlista_c_ss
    
    # Dysts standardization
    if args.standardize:
        cfg.ENV.DYSTS.STANDARDIZE = True
        print("Using standardized dysts data (zero mean, unit variance)")
    if args.dysts_ic_noise_scale is not None:
        cfg.ENV.DYSTS.IC_NOISE_SCALE = float(args.dysts_ic_noise_scale)
        print(f"Using dysts IC noise scale: {cfg.ENV.DYSTS.IC_NOISE_SCALE}")
    if args.dysts_native_cache:
        cfg.ENV.DYSTS.USE_NATIVE_CACHE = True
        print("Using dysts native trajectory cache for training data")
    if args.dysts_cache_steps is not None:
        cfg.ENV.DYSTS.CACHE_STEPS = args.dysts_cache_steps
    if args.dysts_cache_trajectories is not None:
        cfg.ENV.DYSTS.CACHE_TRAJECTORIES = args.dysts_cache_trajectories
    if args.dysts_cache_warmup is not None:
        cfg.ENV.DYSTS.CACHE_WARMUP = args.dysts_cache_warmup
    
    # Training mode
    if args.pairwise and args.sequence:
        raise ValueError("Cannot specify both --pairwise and --sequence")
    if args.pairwise:
        cfg.TRAIN.USE_SEQUENCE_LOSS = False
        print("Using pairwise (single-step) training mode")
    if args.sequence:
        cfg.TRAIN.USE_SEQUENCE_LOSS = True
        print("Using sequence (multi-step ODE) training mode")
    if args.sequence_length is not None:
        cfg.TRAIN.SEQUENCE_LENGTH = args.sequence_length
    if args.eval_every is not None:
        cfg.TRAIN.EVAL_EVERY = int(args.eval_every)
    if args.eval_num_steps is not None:
        cfg.TRAIN.EVAL_NUM_STEPS = int(args.eval_num_steps)

    # Structured latent space config
    if args.structured:
        cfg.MODEL.STRUCTURED.ENABLED = True
        cfg.MODEL.MODEL_NAME = "StructuredLISTAKM"
        print("Enabling structured latent space with StructuredLISTAKM")
    if args.d_global is not None:
        cfg.MODEL.STRUCTURED.D_GLOBAL = args.d_global
    if args.num_basins is not None:
        cfg.MODEL.STRUCTURED.NUM_BASINS = args.num_basins
    if args.d_basin is not None:
        cfg.MODEL.STRUCTURED.D_BASIN = args.d_basin
    if args.lambda_global is not None:
        cfg.MODEL.STRUCTURED.LAMBDA_GLOBAL = args.lambda_global
    if args.lambda_local is not None:
        cfg.MODEL.STRUCTURED.LAMBDA_LOCAL = args.lambda_local
    if args.lambda_exclusivity is not None:
        cfg.MODEL.STRUCTURED.LAMBDA_EXCLUSIVITY = args.lambda_exclusivity
    if args.lambda_sparsity is not None:
        cfg.MODEL.STRUCTURED.LAMBDA_SPARSITY = args.lambda_sparsity
    if args.lambda_entropy is not None:
        cfg.MODEL.STRUCTURED.LAMBDA_ENTROPY = args.lambda_entropy
    if args.lambda_dominance is not None:
        cfg.MODEL.STRUCTURED.LAMBDA_DOMINANCE = args.lambda_dominance
    if args.lambda_temporal is not None:
        cfg.MODEL.STRUCTURED.LAMBDA_TEMPORAL = args.lambda_temporal
    if args.excl_warmup_steps is not None:
        cfg.MODEL.STRUCTURED.EXCL_WARMUP_STEPS = args.excl_warmup_steps

    # Auto-detect device
    device = get_device(args.device)
    print(f"Using device: {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device == 'mps':
        print("  Using Metal Performance Shaders (MPS)")
    else:
        print("  Using CPU")
    
    # Train
    train(cfg, log_dir=args.log_dir, checkpoint_path=args.checkpoint, device=device)


if __name__ == '__main__':
    main()

