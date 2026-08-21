"""
Reusable training runner and command-line interface for Koopman autoencoders.

This script provides a complete training pipeline for learning Koopman operator
representations of dynamical systems using PyTorch.

Usage:
    uv run skae-train --config generic_sparse --env duffing --num_steps 20000

Or use it programmatically:
    from skae.training import train
    cfg = get_config("generic_sparse")
    cfg.ENV.ENV_NAME = "duffing"
    train(cfg, log_dir="./runs/experiment_001")
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn

from skae.config import Config, apply_env_dt_override, get_config
from skae.dysts_cache_profiles import (
    DYSTS_CACHE_PROFILES,
    apply_dysts_cache_profile,
    default_dysts_cache_dir,
)

from skae.data import (
    DystsTrajectoryCache,
    VectorWrapper,
    generate_trajectory,
    make_env,
    wrap_training_env,
)

from skae.model import make_model
from skae.training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointManager,
    CheckpointSignalExit,
    SignalStopper,
    capture_rng_state,
    make_run_identity,
    restore_rng_state,
    validate_run_identity,
)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cuda_synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _cuda_profiler_boundary(name: str) -> None:
    """Call the CUDA profiler API and fail closed when it reports an error."""
    result = getattr(torch.cuda.cudart(), name)()
    status = result[0] if isinstance(result, tuple) else result
    if status not in (None, 0):
        raise RuntimeError(f"cuda profiler {name} failed with status {status}")


def _supports_device_sequence_generation(vector_env: VectorWrapper, device: str) -> bool:
    if device not in {"cuda", "mps"}:
        return False
    if bool(getattr(vector_env, "_device_sequence_generation_failed", False)):
        return False
    base_env = getattr(vector_env, "unwrapped", vector_env)
    module = str(base_env.__class__.__module__).lower()
    class_name = str(base_env.__class__.__name__).lower()
    if "dysts" in module or class_name == "dystsenv":
        return False
    return True


def generate_sequence_batch_for_device(
    vector_env: VectorWrapper,
    rng: torch.Generator,
    *,
    window_length: int,
    device: str,
) -> torch.Tensor:
    """Generate a training sequence directly on accelerator when safe.

    Most built-in environments are pure PyTorch, so their step functions can
    advance a CUDA batch once reset states are moved to the target device.
    Dysts and other NumPy-backed environments stay on the existing CPU path.
    """

    if _supports_device_sequence_generation(vector_env, device):
        try:
            init_states = vector_env.reset(rng).to(device)
            trajectories = generate_trajectory(
                vector_env.step,
                init_states,
                length=window_length,
            )
            if trajectories.device.type != torch.device(device).type:
                raise RuntimeError(
                    f"environment trajectory stayed on {trajectories.device}, expected {device}"
                )
            return torch.cat(
                [init_states.unsqueeze(0), trajectories],
                dim=0,
            ).transpose(0, 1)
        except Exception as exc:
            setattr(vector_env, "_device_sequence_generation_failed", True)
            print(
                "[train] accelerator-side sequence generation failed once; "
                f"falling back to CPU batches ({exc})",
                flush=True,
            )
    return vector_env.generate_sequence_batch(rng, window_length=window_length).to(device)


class MetricsLogger:
    """Simple file-based metrics logger.

    Logs metrics to JSON files for later analysis or plotting.
    Can easily be replaced with wandb later.
    Uses buffered writes to reduce I/O overhead.
    """

    def __init__(self, log_dir: Path, flush_interval: int = 100, save_history: bool = False):
        self.log_dir = log_dir
        self.save_history = bool(save_history)
        self.metrics_file = log_dir / 'metrics_history.jsonl'
        self.metrics_history: List[Dict] = []
        self.buffer: List[str] = []
        self.flush_interval = flush_interval
        self.step_count = 0
        self._summary_state: Dict[str, Dict[str, Any]] = {}

    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar metric."""
        entry = {
            'step': step,
            'name': name,
            'value': value,
        }
        if self.save_history:
            # Buffer writes to reduce I/O overhead.
            self.buffer.append(json.dumps(entry) + '\n')
            self.metrics_history.append(entry)
        self._update_summary(name, value)
        self.step_count += 1

        # Flush buffer periodically
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        """Flush buffered metrics to disk."""
        if self.save_history and self.buffer:
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
        summary = {
            name: {
                "final": state["final"],
                "min": state["min"],
                "max": state["max"],
                "mean": (
                    state["sum"] / state["count"]
                    if state["count"] > 0
                    else None
                ),
            }
            for name, state in self._summary_state.items()
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def state_dict(self) -> Dict[str, Any]:
        """Return logger state needed to continue a run without duplicate rows."""
        # A checkpoint boundary is also a durable logging boundary.  This
        # keeps the buffered history and the on-disk JSONL in agreement after
        # a requeue or signal-triggered exit.
        self.flush()
        return {
            "save_history": self.save_history,
            "metrics_history": list(self.metrics_history),
            "step_count": int(self.step_count),
            "summary_state": deepcopy(self._summary_state),
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore logger counters, summary aggregates, and buffered history."""
        if not state:
            return
        self.save_history = bool(state.get("save_history", self.save_history))
        self.metrics_history = list(state.get("metrics_history", []))
        self.buffer = []
        self.step_count = int(state.get("step_count", 0))
        self._summary_state = deepcopy(state.get("summary_state", {}))
        if self.save_history:
            expected_lines = "".join(
                json.dumps(entry) + "\n" for entry in self.metrics_history
            )
            current_lines = (
                self.metrics_file.read_text(encoding="utf-8")
                if self.metrics_file.exists()
                else ""
            )
            if current_lines != expected_lines:
                temporary = self.metrics_file.with_name(
                    f".{self.metrics_file.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    with temporary.open("w", encoding="utf-8") as handle:
                        handle.write(expected_lines)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, self.metrics_file)
                finally:
                    temporary.unlink(missing_ok=True)

    def _update_summary(self, name: str, value: float):
        state = self._summary_state.setdefault(
            name,
            {"final": None, "min": None, "max": None, "sum": 0.0, "count": 0},
        )
        state["final"] = value
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            numeric_value = float(value)
            state["sum"] += numeric_value
            state["count"] += 1
            state["min"] = (
                numeric_value
                if state["min"] is None
                else min(float(state["min"]), numeric_value)
            )
            state["max"] = (
                numeric_value
                if state["max"] is None
                else max(float(state["max"]), numeric_value)
            )


def parse_optional_bool(value: str) -> bool:
    """Parse a string boolean for optional CLI overrides."""
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Expected one of true/false, yes/no, 1/0."
    )


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_seq: torch.Tensor,
    step: int = 0,
) -> Dict[str, float]:
    """Perform one training step.

    Args:
        model: Koopman machine model
        optimizer: PyTorch optimizer
        x_seq: Sequence window [batch_size, horizon+1, observation_size]
        step: Current training step

    Returns:
        Dictionary of metrics
    """
    model.train()
    optimizer.zero_grad()

    if x_seq.ndim != 3 or x_seq.shape[1] < 2:
        raise ValueError("x_seq must have shape [batch, horizon+1, obs] with horizon >= 1")

    batch_size, seq_len, obs_size = x_seq.shape
    horizon = seq_len - 1

    x0 = x_seq[:, 0, :]
    x_true = x_seq[:, 1:, :]

    # Encode all states once, then rollout K-discrete dynamics from z0.
    z_all = model.encode(x_seq.reshape(batch_size * seq_len, obs_size)).reshape(batch_size, seq_len, -1)
    z0 = z_all[:, 0, :]
    z_true = z_all[:, 1:, :]
    z_pred = model.rollout_latent_discrete(z0, horizon=horizon)

    x_pred = model.decode(z_pred.reshape(batch_size * horizon, -1)).reshape(batch_size, horizon, obs_size)
    x_recon_true = model.decode(z_true.reshape(batch_size * horizon, -1)).reshape(batch_size, horizon, obs_size)

    block_losses: Optional[Dict[str, Any]] = None
    homogeneous_loss: Optional[torch.Tensor] = None

    if hasattr(model, "_block_loss_cfg") and getattr(model._block_loss_cfg, "ENABLED", False):
        z_block = torch.cat([z0.unsqueeze(1), z_true], dim=1).reshape(batch_size * seq_len, -1)
        one_block_loss, balance_loss, block_metrics = model._block_losses_from_z(z_block)
        block_losses = {
            "one_block_loss": one_block_loss,
            "balance_loss": balance_loss,
            "entropy": block_metrics["block_entropy"],
            "usage_entropy": block_metrics["block_usage_entropy"],
            "top1_gap": block_metrics["block_top1_gap"],
        }

    if getattr(model, "use_homogeneous", False) and hasattr(model, "get_homogeneous_coord"):
        c_hat = model.get_homogeneous_coord(z_pred.reshape(batch_size * horizon, -1))
        homogeneous_loss = torch.mean((c_hat - 1.0) ** 2)

    sparsity_target = getattr(model.cfg.MODEL, "SPARSITY_TARGET", "rollout")
    if sparsity_target == "rollout":
        sparsity_latent = z_pred
    elif sparsity_target == "encoded":
        sparsity_latent = z_true
    elif sparsity_target == "encoded_rollout":
        sparsity_latent = 0.5 * (z_true.abs() + z_pred.abs())
    else:
        raise ValueError(
            "Unknown MODEL.SPARSITY_TARGET "
            f"{sparsity_target!r}; expected rollout, encoded, or encoded_rollout"
        )

    loss, metrics = model.loss(
        x_pred=x_pred,
        x_true=x_true,
        x0=x0,
        z0=z0,
        z_pred=z_pred,
        z_true=z_true,
        reconstruction_error=torch.norm(x_true - x_recon_true, dim=-1).mean(),
        sparsity_latent=sparsity_latent,
        homogeneous_loss=homogeneous_loss,
        block_losses=block_losses,
        step=step,
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    return metrics


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Create optimizer with a specific learning rate for the Koopman matrix.

    This constructs parameter groups so that Koopman dynamics parameters use
    cfg.TRAIN.K_MATRIX_LR while all other parameters use cfg.TRAIN.LR.
    """
    kmat_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'kmat' in name or name.startswith('K_'):
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
    env_step_fn=None,
    num_steps: int = 50,
    true_trajectory: Optional[torch.Tensor] = None,
    rollout_mode: str = "every_step_reencode",
) -> Dict[str, Any]:
    """Quick evaluation helper used during training and unit tests."""

    # Lazy import to avoid loading evaluation module at startup
    from skae.evaluation import rollout_every_step_reencode, rollout_no_reencode

    if rollout_mode not in {"direct", "every_step_reencode"}:
        raise ValueError(f"Unknown rollout_mode: {rollout_mode}")

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        if true_trajectory is None:
            if env_step_fn is None:
                raise ValueError("env_step_fn is required when true_trajectory is absent")
            true_traj = generate_trajectory(env_step_fn, x.cpu(), length=num_steps)
        else:
            true_traj = true_trajectory.detach().cpu()
            expected_shape = (num_steps, x.shape[0], x.shape[1])
            if tuple(true_traj.shape) != expected_shape:
                raise ValueError(
                    f"true_trajectory shape {tuple(true_traj.shape)} != {expected_shape}"
                )
        rollout_fn = (
            rollout_no_reencode
            if rollout_mode == "direct"
            else rollout_every_step_reencode
        )
        pred_traj = rollout_fn(model, x.to(device), num_steps)

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
        obs_dim = max(1, int(diff.shape[-1]))
        step_error_per_dim = step_error / math.sqrt(float(obs_dim))
        squared_error = torch.sum(diff ** 2, dim=-1)
        full_horizon_finite = torch.isfinite(squared_error).all(dim=0)
        strict_values = squared_error[:, full_horizon_finite]
        strict_full_horizon_mse = (
            strict_values.mean().item()
            if strict_values.numel() > 0
            else float("nan")
        )
        full_horizon_finite_fraction = full_horizon_finite.float().mean().item()

        return {
            "true_trajectory": true_traj,
            "pred_trajectory": pred_traj_cpu,
            "pred_error": step_error,
            "pred_error_per_dim": step_error_per_dim,
            "mean_error": torch.nanmean(step_error).item(),
            "mean_error_per_dim": torch.nanmean(step_error_per_dim).item(),
            "final_error": torch.nanmean(step_error[-1]).item(),
            "final_error_per_dim": torch.nanmean(step_error_per_dim[-1]).item(),
            "strict_full_horizon_mse": strict_full_horizon_mse,
            "full_horizon_finite_fraction": full_horizon_finite_fraction,
        }



def train(
    cfg: Config,
    log_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
    skip_eval: bool = False,
    eval_profile: str = "full",
    eval_use_dynamics_prior: bool = False,
    eval_event_trigger_proj_threshold: Optional[float] = None,
    eval_event_trigger_ambiguity_threshold: Optional[float] = None,
    eval_event_trigger_spillover_threshold: Optional[float] = None,
    eval_event_trigger_support_margin_min_ratio: Optional[float] = None,
    eval_event_trigger_support_threshold: float = 1e-3,
    eval_event_trigger_min_dwell: int = 0,
    eval_event_trigger_max_interval: int = 0,
    save_metrics_history: bool = False,
    save_training_plot: bool = False,
    save_last_checkpoint: bool = False,
    save_eval_rollout_artifacts: bool = False,
    save_eval_plots: bool = False,
    save_eval_per_ic_values: bool = False,
    save_eval_error_curves: bool = False,
    pilot_warmup_steps: int = 0,
    pilot_measure_steps: int = 0,
    pilot_profile: bool = False,
    pilot_timing_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: Optional[int] = None,
    checkpoint_retention: int = 3,
    resume: bool = False,
    resume_if_available: bool = False,
    permanent_checkpoint_dir: Optional[str] = None,
) -> nn.Module:
    """Main training function.

    Args:
        cfg: Configuration object
        log_dir: Directory for tensorboard logs and checkpoints
        checkpoint_path: Path to checkpoint to resume from
        device: Device to train on ('cpu', 'cuda', 'mps')
        checkpoint_dir: Persistent directory for complete generation checkpoints.
            Supplying this option enables exact checkpoint/resume state;
            leaving it unset preserves the historical checkpoint behavior.
        checkpoint_interval: Completed training steps between complete saves.
        checkpoint_retention: Number of recent valid generations to retain.
        resume: Discover and load the newest valid generation in checkpoint_dir.
        resume_if_available: Resume when a valid generation exists, otherwise start fresh.
        permanent_checkpoint_dir: Optional durable copy destination for latest/best.
    Returns:
        Trained model
    """
    print("Initializing training...")
    pilot_warmup_steps = int(pilot_warmup_steps)
    pilot_measure_steps = int(pilot_measure_steps)
    if pilot_warmup_steps < 0 or pilot_measure_steps < 0:
        raise ValueError("pilot warmup/measure steps must be non-negative")
    pilot_mode = pilot_measure_steps > 0
    if pilot_profile and not pilot_mode:
        raise ValueError("pilot profiling requires a positive measured-step range")
    if pilot_mode and not str(device).startswith("cuda"):
        raise ValueError("the utilization pilot requires --device cuda")

    complete_checkpointing = checkpoint_dir is not None
    if resume and not complete_checkpointing:
        raise ValueError("resume=True requires checkpoint_dir")
    if resume_if_available and not complete_checkpointing:
        raise ValueError("resume_if_available=True requires checkpoint_dir")
    checkpoint_interval_value = (
        100 if checkpoint_interval is None else int(checkpoint_interval)
    )
    if complete_checkpointing and checkpoint_interval_value < 1:
        raise ValueError("checkpoint_interval must be >= 1")
    signal_stopper = SignalStopper() if complete_checkpointing else None
    if signal_stopper is not None:
        # Install before source inspection, cache construction, or model setup;
        # the handler only records the signal and checkpointing occurs later at
        # a safe completed-step boundary.
        signal_stopper.install()
    num_batches = cfg.TRAIN.DATA_SIZE // cfg.TRAIN.BATCH_SIZE
    if num_batches < 1:
        raise ValueError("TRAIN.DATA_SIZE must be at least TRAIN.BATCH_SIZE")

    checkpoint_manager = None
    resume_payload: Optional[Dict[str, Any]] = None
    if complete_checkpointing:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir,
            retention=checkpoint_retention,
            permanent_root=permanent_checkpoint_dir,
            checkpoint_interval=checkpoint_interval_value,
        )
        if resume or resume_if_available or checkpoint_path is not None:
            loaded = (
                checkpoint_manager.load_path(checkpoint_path, map_location=device)
                if checkpoint_path is not None
                else checkpoint_manager.load_newest_valid(map_location=device)
            )
            if loaded is None and (checkpoint_path is not None or not resume_if_available):
                raise CheckpointError("resume requested but no valid checkpoint was found")
            if loaded is not None:
                resume_payload = loaded["payload"]
                checkpoint_manager.run_id = str(
                    resume_payload.get("run_id", checkpoint_manager.run_id)
                )
        elif checkpoint_manager.load_newest_valid() is not None:
            raise CheckpointError(
                "checkpoint_dir already contains a valid run; pass resume=True to continue"
            )

    # Setup logging directory and save config.  A complete-checkpoint run uses
    # its persistent directory as the stable run directory across requeues.
    if complete_checkpointing:
        run_dir = Path(checkpoint_dir).expanduser()
    else:
        if log_dir is None:
            if cfg.MODEL.MODEL_NAME == 'LISTAKM':
                encoder_type = cfg.MODEL.ENCODER.ENCODER_TYPE.lower()
                if encoder_type == 'hyperlista':
                    log_dir = './runs/hyperlista'
                else:
                    log_dir = './runs/lista'
            else:
                log_dir = './runs/kae'
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(log_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    if not complete_checkpointing or resume_payload is None or not (run_dir / "config.json").exists():
        cfg.to_json(str(run_dir / 'config.json'))

    logger = MetricsLogger(run_dir, save_history=save_metrics_history or save_training_plot)
    run_identity = make_run_identity(
        cfg.to_dict(),
        device,
        batch_count=num_batches,
        logger_history=logger.save_history,
        source_root=Path.cwd(),
    )
    if resume_payload is not None:
        validate_run_identity(run_identity, resume_payload.get("run_identity", {}))

    print("Setting random seed...")
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        if complete_checkpointing:
            torch.cuda.manual_seed_all(cfg.SEED)
        else:
            torch.cuda.manual_seed(cfg.SEED)
    if complete_checkpointing:
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)
    # MPS doesn't have manual_seed, but manual_seed should be sufficient

    print("Creating environment...")
    base_env = make_env(cfg)
    train_env = VectorWrapper(wrap_training_env(base_env, cfg), cfg.TRAIN.BATCH_SIZE)
    eval_env = VectorWrapper(base_env, cfg.TRAIN.BATCH_SIZE)

    # DYSTS-specific training caveat: without the cache we only see x->x+1 from freshly
    # reset initial conditions (i.e., mostly transients, not attractor distribution).
    # This often yields models that "look fine" on 1-step error but fail catastrophically
    # in long rollouts / phase portraits.
    try:
        from skae.benchmarks.dysts_adapter import DystsEnv
        if isinstance(train_env.unwrapped, DystsEnv) and not cfg.ENV.DYSTS.USE_NATIVE_CACHE:
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

    print("Creating model...")
    model = make_model(cfg, base_env.observation_size)
    model = model.to(device)

    print("Building optimizer...")
    optimizer = build_optimizer(model, cfg)

    start_step = 0
    checkpoint = resume_payload
    if checkpoint is None and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_state = checkpoint.get('optimizer_state_dict')
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        else:
            print("Checkpoint has no optimizer state; resuming with a fresh optimizer.")
        if complete_checkpointing:
            start_step = int(checkpoint.get("next_step", 0))
        else:
            # Historical checkpoints stored the completed step.  Resuming at
            # step+1 fixes the duplicate-update off-by-one while accepting old
            # files that predate the complete-state schema.
            start_step = int(checkpoint.get('next_step', checkpoint.get('step', -1) + 1))
        print(f"Resumed from checkpoint at step {start_step}")

    if pilot_mode:
        if start_step != 0:
            raise ValueError("the utilization pilot does not accept a resumed checkpoint")
        expected_steps = pilot_warmup_steps + pilot_measure_steps
        if int(cfg.TRAIN.NUM_STEPS) != expected_steps:
            raise ValueError(
                "pilot NUM_STEPS must equal warmup + measured steps "
                f"({cfg.TRAIN.NUM_STEPS} != {expected_steps})"
            )
        pilot_measure_start = start_step + pilot_warmup_steps
        pilot_measure_end = pilot_measure_start + pilot_measure_steps
    else:
        pilot_measure_start = pilot_measure_end = -1

    resolved_config = json.loads(json.dumps(cfg.to_dict(), allow_nan=False))
    if pilot_mode:
        # Pilot-only launch lengths are not a scientific configuration change:
        # the unprofiled and NCU phases use different measured step counts.
        resolved_config.setdefault("TRAIN", {})["NUM_STEPS"] = (
            "pilot_step_count_excluded"
        )
    resolved_config_path = run_dir / "resolved_config.json"
    _atomic_json_write(resolved_config_path, resolved_config)
    resolved_config_sha256 = _stable_hash(resolved_config)
    architecture_identity = {
        "model_class": type(model).__name__,
        "observation_size": int(base_env.observation_size),
        "target_size": int(cfg.MODEL.TARGET_SIZE),
        "sequence_length": int(cfg.TRAIN.SEQUENCE_LENGTH),
        "parameter_shapes": {
            name: list(parameter.shape) for name, parameter in model.named_parameters()
        },
    }
    architecture_sha256 = _stable_hash(architecture_identity)

    # Pre-generate random number generators for data
    # Each batch gets a non-overlapping seed range to avoid collisions
    # Batch i uses seeds: cfg.SEED + i * BATCH_SIZE to cfg.SEED + (i+1) * BATCH_SIZE - 1
    rngs = [torch.Generator().manual_seed(cfg.SEED + i * cfg.TRAIN.BATCH_SIZE) for i in range(num_batches)]

    # Generate fixed validation set for consistent evaluation
    # Optional dysts cache for faster data generation
    dysts_cache = None
    val_dysts_cache = None
    if cfg.ENV.DYSTS.USE_NATIVE_CACHE:
        try:
            from skae.benchmarks.dysts_adapter import DystsEnv
            if isinstance(train_env.unwrapped, DystsEnv):
                print(
                    "Initializing dysts native trajectory cache "
                    f"(split={cfg.ENV.DYSTS.CACHE_SPLIT}, "
                    f"steps={cfg.ENV.DYSTS.CACHE_STEPS}, "
                    f"trajectories={cfg.ENV.DYSTS.CACHE_TRAJECTORIES}, "
                    f"warmup={cfg.ENV.DYSTS.CACHE_WARMUP}) ...",
                    flush=True,
                )
                dysts_cache = DystsTrajectoryCache(train_env.unwrapped, cfg)
                # Keep validation split separate from training split.
                val_cache_cfg = Config.from_dict(cfg.to_dict())
                val_cache_cfg.ENV.DYSTS.CACHE_SPLIT = "val"
                print(
                    "Initializing dysts validation cache "
                    f"(split={val_cache_cfg.ENV.DYSTS.CACHE_SPLIT}, "
                    f"steps={val_cache_cfg.ENV.DYSTS.CACHE_STEPS}, "
                    f"trajectories={val_cache_cfg.ENV.DYSTS.CACHE_TRAJECTORIES}, "
                    f"warmup={val_cache_cfg.ENV.DYSTS.CACHE_WARMUP}) ...",
                    flush=True,
                )
                val_dysts_cache = DystsTrajectoryCache(train_env.unwrapped, val_cache_cfg)
                print(
                    "Using dysts native trajectory cache "
                    f"(train_split={cfg.ENV.DYSTS.CACHE_SPLIT}, val_split={val_cache_cfg.ENV.DYSTS.CACHE_SPLIT}, "
                    f"steps={cfg.ENV.DYSTS.CACHE_STEPS}, trajectories={cfg.ENV.DYSTS.CACHE_TRAJECTORIES})"
                )
            else:
                print("Warning: Dysts cache enabled but environment is not dysts.")
        except Exception as e:
            if cfg.ENV.DYSTS.USE_NATIVE_CACHE:
                raise RuntimeError(
                    "Failed to initialize required Dysts native cache"
                ) from e
            print(f"Warning: Failed to initialize dysts cache: {e}")

    print("Generating fixed validation set...")
    val_rng = torch.Generator().manual_seed(cfg.SEED + 999999) # Separate seed
    val_window = max(1, cfg.TRAIN.SEQUENCE_LENGTH)
    if val_dysts_cache is not None:
        val_window = max(val_window, int(cfg.TRAIN.EVAL_NUM_STEPS))
    if val_dysts_cache is not None:
        val_seq = val_dysts_cache.sample_sequence_batch(
            val_rng,
            batch_size=16,
            window_length=val_window,
            device=device,
        )
    else:
        val_seq = generate_sequence_batch_for_device(
            eval_env,
            val_rng,
            window_length=val_window,
            device=device,
        )
    val_x = val_seq[:16, 0, :]
    val_true_trajectory = None
    if val_dysts_cache is not None:
        val_true_trajectory = val_seq[
            :16, 1 : int(cfg.TRAIN.EVAL_NUM_STEPS) + 1, :
        ].transpose(0, 1).contiguous()
    checkpoint_rollout_mode = (
        "direct" if val_dysts_cache is not None else "every_step_reencode"
    )
    print(f"Checkpoint selection rollout: {checkpoint_rollout_mode}")

    # Validation data is regenerated from its stable seed before restoring the
    # saved RNG snapshot.  This makes setup side effects invisible to the
    # resumed training stream while preserving the exact validation inputs.
    if complete_checkpointing and resume_payload is not None:
        restore_rng_state(
            resume_payload["rng_state"],
            rngs,
            validation_generator=val_rng,
        )
        logger.load_state_dict(resume_payload.get("logger_state"))

    print(f"Training {cfg.MODEL.MODEL_NAME} on {cfg.ENV.ENV_NAME}")
    print(f"Device: {device}")
    print(f"Observation size: {base_env.observation_size}")
    print(f"Target size: {cfg.MODEL.TARGET_SIZE}")
    print(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    print(f"Total steps: {cfg.TRAIN.NUM_STEPS}")
    if cfg.TRAIN.HARD_INIT_OVERSAMPLE.ENABLED:
        settings = cfg.TRAIN.HARD_INIT_OVERSAMPLE
        print(
            "[train] hard-init oversampling enabled "
            f"(fraction={settings.FRACTION:.2f}, pool={settings.POOL_SIZE}, "
            f"candidates={settings.NUM_CANDIDATES}, probe_steps={settings.PROBE_STEPS})"
        )
    print(f"Log directory: {run_dir}")
    print("-" * 80)

    best_eval_final_error = (
        float(resume_payload.get("best_eval_final_error", float("inf")))
        if resume_payload is not None
        else float("inf")
    )
    last_metrics: Dict[str, Any] = (
        dict(resume_payload.get("last_metrics", {})) if resume_payload is not None else {}
    )
    selection_metric = (
        resume_payload.get("checkpoint_selection_metric")
        if resume_payload is not None
        else None
    )
    selection_score = (
        resume_payload.get("checkpoint_selection_score")
        if resume_payload is not None
        else None
    )
    selection_finite_fraction = (
        resume_payload.get("checkpoint_selection_full_horizon_finite_fraction")
        if resume_payload is not None
        else None
    )

    def save_complete_checkpoint(next_step: int, *, is_best: bool = False) -> None:
        """Commit all state at a safe completed-step boundary."""
        if checkpoint_manager is None:
            return
        complete_state = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": checkpoint_manager.run_id,
            "config": cfg.to_dict(),
            "run_identity": run_identity,
            "storage_contract": {
                "scratch_root": str(checkpoint_manager.root),
                "permanent_root": (
                    None
                    if checkpoint_manager.permanent_root is None
                else str(checkpoint_manager.permanent_root)
                ),
                "retention": checkpoint_manager.retention,
                "checkpoint_interval": checkpoint_interval_value,
            },
            "step": int(next_step) - 1,
            "next_step": int(next_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "scaler_state_dict": None,
            "rng_state": capture_rng_state(rngs, validation_generator=val_rng),
            "data_order": {
                "num_batches": num_batches,
                "batch_size": int(cfg.TRAIN.BATCH_SIZE),
                "sequence_length": int(cfg.TRAIN.SEQUENCE_LENGTH),
                "seed": int(cfg.SEED),
                "generator_index": int(next_step % num_batches),
            },
            "last_metrics": dict(last_metrics),
            "logger_state": logger.state_dict(),
            "best_eval_final_error": float(best_eval_final_error),
            "checkpoint_selection_metric": selection_metric,
            "checkpoint_selection_score": selection_score,
            "checkpoint_selection_horizon": int(cfg.TRAIN.EVAL_NUM_STEPS),
            "checkpoint_selection_batch_size": int(val_x.shape[0]),
            "checkpoint_selection_split": (
                "val" if val_dysts_cache is not None else "generated_validation"
            ),
            "checkpoint_selection_full_horizon_finite_fraction": selection_finite_fraction,
        }
        checkpoint_manager.save(complete_state, next_step=next_step, is_best=is_best)

    if signal_stopper is not None and signal_stopper.requested:
        # A TERM during setup has no completed update to replay.  Persist the
        # initialized state at the exact current next_step before exiting.
        save_complete_checkpoint(start_step)
        logger.close()
        signal_stopper.restore()
        raise CheckpointSignalExit()

    last_checkpoint_dict: Optional[Dict[str, Any]] = None
    pilot_timer_start: Optional[float] = None
    pilot_wall_start: Optional[float] = None
    pilot_timing_written = False
    metrics: Dict[str, Any] = dict(last_metrics)
    for step in range(start_step, cfg.TRAIN.NUM_STEPS):
        if pilot_mode and step == pilot_measure_start:
            _cuda_synchronize(device)
            if pilot_profile:
                _cuda_profiler_boundary("cudaProfilerStart")
            pilot_timer_start = time.perf_counter()
            pilot_wall_start = time.time()
            print(
                "PILOT_MEASURE_START "
                f"step={pilot_measure_start} end_step_exclusive={pilot_measure_end} "
                f"warmup_steps={pilot_warmup_steps} measured_steps={pilot_measure_steps} "
                f"profiler_active={pilot_profile}",
                flush=True,
            )
        # Generate batch
        rng = rngs[step % num_batches]

        if dysts_cache is not None:
            x_seq = dysts_cache.sample_sequence_batch(
                rng,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                window_length=cfg.TRAIN.SEQUENCE_LENGTH,
                device=device,
            )
        else:
            x_seq = generate_sequence_batch_for_device(
                train_env,
                rng,
                window_length=cfg.TRAIN.SEQUENCE_LENGTH,
                device=device,
            )

        metrics = train_step(model, optimizer, x_seq, step=step)
        last_metrics = dict(metrics)

        in_measured_range = pilot_mode and pilot_measure_start <= step < pilot_measure_end
        if not in_measured_range:
            logger.log_dict(metrics, step, prefix='train')

        if step % 100 == 0 and not in_measured_range:
            sr_str = ""
            if 'spectral_radius' in metrics:
                sr_str = f" | SR: {metrics['spectral_radius']:.4f}"
            log_str = (f"Step {step}/{cfg.TRAIN.NUM_STEPS} | "
                       f"Loss: {metrics['loss']:.4f} | "
                       f"Align: {metrics['alignment_loss']:.4f} | "
                       f"Recon: {metrics['reconst_loss']:.4f} | "
                       f"Pred: {metrics['prediction_loss']:.4f} | "
                       f"Sparsity: {metrics['sparsity_ratio']:.3f}"
                       f"{sr_str}")
            if 'homogeneous_loss' in metrics:
                log_str += f" | Homog: {metrics['homogeneous_loss']:.4f}"
            print(log_str)

        # Periodic evaluation and checkpoint saving
        # Note: skip step=0 to avoid expensive eval before any learning happened.
        best_updated = False
        if (
            not pilot_mode
            and ((step > 0 and step % cfg.TRAIN.EVAL_EVERY == 0)
                 or step == cfg.TRAIN.NUM_STEPS - 1)
        ):
            # Use fixed validation set
            eval_results = evaluate(
                model,
                val_x,
                None if val_true_trajectory is not None else lambda s: eval_env.step(s),
                num_steps=cfg.TRAIN.EVAL_NUM_STEPS,
                true_trajectory=val_true_trajectory,
                rollout_mode=checkpoint_rollout_mode,
            )
            logger.log_scalar('eval/mean_error', eval_results['mean_error'], step)
            logger.log_scalar('eval/mean_error_per_dim', eval_results['mean_error_per_dim'], step)
            logger.log_scalar('eval/final_error', eval_results['final_error'], step)
            logger.log_scalar('eval/final_error_per_dim', eval_results['final_error_per_dim'], step)
            if checkpoint_rollout_mode == "direct":
                checkpoint_score = (
                    eval_results["strict_full_horizon_mse"]
                    if eval_results["full_horizon_finite_fraction"] == 1.0
                    else float("inf")
                )
                checkpoint_metric = (
                    "direct_strict_full_horizon_cumulative_state_summed_mse"
                )
            else:
                checkpoint_score = eval_results["final_error"]
                checkpoint_metric = "every_step_reencode_final_l2_error"

            selection_metric = checkpoint_metric
            selection_score = checkpoint_score
            selection_finite_fraction = eval_results['full_horizon_finite_fraction']

            print(
                f"  Eval | Mean error: {eval_results['mean_error']:.4f} "
                f"(per-dim {eval_results['mean_error_per_dim']:.4f}) | "
                f"Final error: {eval_results['final_error']:.4f} "
                f"(per-dim {eval_results['final_error_per_dim']:.4f}) | "
                f"Checkpoint score: {checkpoint_score:.4f} "
                f"(coverage {eval_results['full_horizon_finite_fraction']:.3f})"
            )

            # Save checkpoint
            checkpoint_dict = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if save_last_checkpoint else None,
                'config': cfg.to_dict(),
                'metrics': metrics,
                'checkpoint_selection_rollout': checkpoint_rollout_mode,
                'checkpoint_selection_metric': checkpoint_metric,
                'checkpoint_selection_score': checkpoint_score,
                'checkpoint_selection_horizon': int(cfg.TRAIN.EVAL_NUM_STEPS),
                'checkpoint_selection_batch_size': int(val_x.shape[0]),
                'checkpoint_selection_split': (
                    'val' if val_dysts_cache is not None else 'generated_validation'
                ),
                'checkpoint_selection_full_horizon_finite_fraction': (
                    eval_results['full_horizon_finite_fraction']
                ),
            }
            last_checkpoint_dict = checkpoint_dict

            if save_last_checkpoint:
                torch.save(checkpoint_dict, run_dir / 'last.pt')

            # Save best checkpoint if eval error improved
            if checkpoint_score < best_eval_final_error:
                best_eval_final_error = checkpoint_score
                best_updated = True
                torch.save(checkpoint_dict, run_dir / 'checkpoint.pt')
                print(f"  Saved best checkpoint ({checkpoint_metric}: {best_eval_final_error:.4f})")

        if pilot_mode and step == pilot_measure_end - 1:
            _cuda_synchronize(device)
            if pilot_timer_start is None or pilot_wall_start is None:
                raise RuntimeError("pilot timing started without a measured range")
            elapsed = time.perf_counter() - pilot_timer_start
            wall_end = time.time()
            if pilot_profile:
                _cuda_profiler_boundary("cudaProfilerStop")
            timing_payload = {
                "schema_version": 1,
                "status": "complete",
                "warmup_steps": pilot_warmup_steps,
                "measured_steps": pilot_measure_steps,
                "step_start": pilot_measure_start,
                "step_end_exclusive": pilot_measure_end,
                "elapsed_seconds": elapsed,
                "steps_per_second": pilot_measure_steps / elapsed,
                "cuda_synchronized_before_and_after": True,
                "profiler_range_active": bool(pilot_profile),
                "wall_start_unix": pilot_wall_start,
                "wall_end_unix": wall_end,
                "resolved_config_sha256": resolved_config_sha256,
                "resolved_config_path": str(resolved_config_path),
                "pilot_config_num_steps": int(cfg.TRAIN.NUM_STEPS),
                "architecture_sha256": architecture_sha256,
                "architecture_identity": architecture_identity,
            }
            if pilot_timing_path is None:
                raise RuntimeError("pilot timing path was not provided")
            _atomic_json_write(Path(pilot_timing_path), timing_payload)
            print(
                "PILOT_MEASURE_END "
                f"step={pilot_measure_end} elapsed_seconds={elapsed:.9f} "
                f"steps_per_second={timing_payload['steps_per_second']:.6f}",
                flush=True,
            )
            pilot_timing_written = True

        if checkpoint_manager is not None:
            signal_requested = bool(signal_stopper and signal_stopper.requested)
            checkpoint_due = (
                (step + 1) % checkpoint_interval_value == 0
                or step == cfg.TRAIN.NUM_STEPS - 1
                or best_updated
                or signal_requested
            )
            if checkpoint_due:
                save_complete_checkpoint(step + 1, is_best=best_updated)
            if signal_requested:
                logger.close()
                signal_stopper.restore()
                raise CheckpointSignalExit()

    if pilot_mode and not pilot_timing_written:
        raise RuntimeError("pilot measured range did not complete")

    if checkpoint_manager is None and not (run_dir / 'checkpoint.pt').exists() and last_checkpoint_dict is not None:
        torch.save(last_checkpoint_dict, run_dir / 'checkpoint.pt')
        print("  Saved final checkpoint to checkpoint.pt (no finite best eval was found)")

    if checkpoint_manager is not None:
        checkpoint_manager.materialize_legacy_aliases(run_dir)
        signal_stopper.restore()

    # Save final metrics and close logger
    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.close()

    if save_training_plot:
        print("-" * 80)
        print("Plotting training metrics...")
        from skae.training.plotting import plot_metrics
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

    if not skip_eval and not pilot_mode:
        print("-" * 80)
        print("Running standardized evaluation suite...")
        if eval_profile != "full":
            print(f"Using evaluation profile: {eval_profile}")
        print("Loading evaluation module...")
        from skae.evaluation import EvaluationSettings, evaluate_model

        def evaluate_checkpoint(checkpoint_path: Path, checkpoint_name: str):
            """Load a checkpoint and evaluate it."""
            if not checkpoint_path.exists():
                print(f"  Skipping {checkpoint_name}: checkpoint not found at {checkpoint_path}")
                return None

            print(f"\nEvaluating {checkpoint_name} checkpoint...", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            ckpt_step = checkpoint.get('step', 'unknown')
            print(f"  Loaded checkpoint (step={ckpt_step}). Building eval env/model...", flush=True)

            eval_cfg = Config.from_dict(cfg.to_dict())
            if eval_cfg.ENV.ENV_NAME.lower().startswith("dysts:") and eval_cfg.ENV.DYSTS.USE_NATIVE_CACHE:
                eval_cfg.ENV.DYSTS.CACHE_SPLIT = "test"
                if not eval_cfg.ENV.DYSTS.CACHE_DIR:
                    eval_cfg.ENV.DYSTS.CACHE_DIR = default_dysts_cache_dir()
                eval_cfg.ENV.DYSTS.CACHE_REUSE = True

            # Load model from checkpoint (use unwrapped env for observation_size)
            eval_env = make_env(eval_cfg)
            eval_model = make_model(eval_cfg, eval_env.observation_size)
            eval_model.load_state_dict(checkpoint['model_state_dict'])
            eval_model = eval_model.to(device)
            eval_model.eval()

            # Create evaluation settings
            eval_settings = EvaluationSettings()
            eval_settings.systems = [cfg.ENV.ENV_NAME]
            eval_settings.save_rollout_artifacts = bool(save_eval_rollout_artifacts)
            eval_settings.save_plots = bool(save_eval_plots)
            eval_settings.include_per_ic_values = bool(save_eval_per_ic_values)
            eval_settings.include_error_curves = bool(save_eval_error_curves)
            if eval_profile == "smoke":
                # Keep H1000 for compatibility checks, but reduce expensive eval breadth.
                eval_settings.batch_size = 32
                eval_settings.horizons = (100, 500, 1000)
                eval_settings.periodic_reencode_periods = (10, 25)
                eval_settings.dysts_periodic_reencode_periods = (10, 100, 500, 1000)
                eval_settings.phase_portrait_samples = 8
                eval_settings.phase_portrait_length = 100
                eval_settings.phase_portrait_batch_size = 64
                eval_settings.phase_portrait_reencode_periods = (0, 1, 10, 25)
                eval_settings.dysts_phase_portrait_reencode_periods = (0, 1, 100, 500, 1000)
            eval_settings.use_dynamics_prior = bool(eval_use_dynamics_prior)
            eval_settings.event_trigger_proj_threshold = (
                None
                if eval_event_trigger_proj_threshold is None
                else float(eval_event_trigger_proj_threshold)
            )
            eval_settings.event_trigger_ambiguity_threshold = (
                None
                if eval_event_trigger_ambiguity_threshold is None
                else float(eval_event_trigger_ambiguity_threshold)
            )
            eval_settings.event_trigger_spillover_threshold = (
                None
                if eval_event_trigger_spillover_threshold is None
                else float(eval_event_trigger_spillover_threshold)
            )
            eval_settings.event_trigger_support_margin_min_ratio = (
                None
                if eval_event_trigger_support_margin_min_ratio is None
                else float(eval_event_trigger_support_margin_min_ratio)
            )
            eval_settings.event_trigger_support_threshold = float(eval_event_trigger_support_threshold)
            eval_settings.event_trigger_min_dwell = int(eval_event_trigger_min_dwell)
            eval_settings.event_trigger_max_interval = int(eval_event_trigger_max_interval)

            # Evaluate
            eval_dir = run_dir / f"evaluation_{checkpoint_name}"
            print(f"  Calling evaluate_model() for systems={eval_settings.systems} ...", flush=True)
            eval_results = evaluate_model(
                model=eval_model,
                cfg=eval_cfg,
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
                print(f"  {checkpoint_name.upper()} - Primary system ({primary_system}) forecast summary:")
                is_dysts = primary_system.lower().startswith("dysts:")
                for horizon in eval_settings.horizons:
                    if primary_system == "parabolic" and horizon > 100:
                        continue
                    horizon_key = str(horizon)
                    no_re = primary_metrics["modes"]["no_reencode"]["horizons"].get(horizon_key)
                    every = primary_metrics["modes"]["every_step"]["horizons"].get(horizon_key)
                    best = primary_metrics["best_periodic"].get(horizon_key)
                    best_reset = primary_metrics.get("best_reset", {}).get(horizon_key)
                    if no_re is None or every is None:
                        continue
                    best_str = (
                        "best-PR=N/A"
                        if best is None
                        else (
                            f"best-PR={best['mean']:.4e} "
                            f"(per-dim {best.get('per_dim_mean', float('nan')):.4e}, {best['mode']})"
                        )
                    )
                    best_reset_str = (
                        "best-reset=N/A"
                        if best_reset is None
                        else (
                            f"best-reset={best_reset['mean']:.4e} "
                            f"(per-dim {best_reset.get('per_dim_mean', float('nan')):.4e}, {best_reset['mode']})"
                        )
                    )
                    print(
                        f"    Horizon {horizon}: "
                        f"no-reencode={no_re['mean']:.4e} "
                        f"(per-dim {no_re.get('per_dim_mean', float('nan')):.4e}), "
                        f"every-step={every['mean']:.4e} "
                        f"(per-dim {every.get('per_dim_mean', float('nan')):.4e}), "
                        f"{best_str}, "
                        f"{best_reset_str}"
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
                                print(
                                    f"      H={horizon}: mean={horizon_mse['mean']:.4e}, "
                                    f"per-dim={horizon_mse.get('per_dim_mean', float('nan')):.4e}, "
                                    f"std={horizon_mse['std']:.4e}"
                                )

            print(f"  Evaluation artifacts saved to {eval_dir}")
            return eval_results

        # Evaluate the compact best checkpoint by default. last.pt is optional
        # because it duplicates weights and evaluation output for most runs.
        last_checkpoint = run_dir / 'last.pt'
        best_checkpoint = run_dir / 'checkpoint.pt'

        # Some unstable runs can produce non-finite quick-eval errors at every
        # checkpoint, which means no "best" checkpoint ever gets recorded.
        # For benchmark completeness, fall back to the final weights so the
        # standardized evaluation still emits evaluation_results_best.json.
        if last_checkpoint.exists() and not best_checkpoint.exists():
            print(
                "Best checkpoint was never created during training. "
                "Copying last.pt to checkpoint.pt for fallback evaluation.",
                flush=True,
            )
            shutil.copy2(last_checkpoint, best_checkpoint)

        eval_results_last = None
        if save_last_checkpoint:
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
            raise RuntimeError(
                "Requested --device cuda, but CUDA is not available. "
                "Use --device auto or --device cpu for CPU fallback."
            )

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
    print("Starting skae-train...")
    parser = argparse.ArgumentParser(
        description='Train Koopman Autoencoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on built-in environment
  uv run skae-train --config generic_sparse --env duffing --num_steps 10000

  # Train on dysts chaotic system
  uv run skae-train --config lista --env dysts:Lorenz --num_steps 10000
  uv run skae-train --config lista --env dysts:Chua --target_size 1024

  # Train on an analytic multibasin system
  uv run skae-train --config generic_sparse --env analytic:cal_square_4 --num_steps 10000

  # List available environments
  uv run skae-train --list-envs
        """
    )

    # Configuration
    parser.add_argument('--config', type=str, default='generic',
                        choices=['default', 'generic', 'generic_sparse', 'generic_no_shrink',
                                'generic_prediction', 'lista', 'lista_nonlinear',
                                'lista_parity_generic_sparse', 'hyperlista',
                                'hyperlista_parity_generic_sparse'],
                        help='Training configuration preset')
    parser.add_argument('--env', type=str, default='duffing',
                        help='Environment name. Built-in: duffing, pendulum, lotka_volterra, '
                             'lorenz63, parabolic, lyapunov, blended, '
                             'multiwell, multiwell:<mode>, multiwell_*_hd. '
                             'For analytic systems: use "analytic:SystemName" '
                             '(e.g., "analytic:cal_square_4"). '
                             'For dysts systems: use "dysts:SystemName" (e.g., "dysts:Lorenz", "dysts:Chua")')
    parser.add_argument('--env_dt', type=float, default=None,
                        help='Override the integration timestep for the active environment')
    parser.add_argument('--lyapunov_dim', type=int, default=None,
                        help='Lyapunov state dimension (default: 2)')
    parser.add_argument('--lyapunov_num_basins', type=int, default=None,
                        help='Number of Lyapunov attractor centers (default: 13)')
    parser.add_argument('--lyapunov_points_mode', type=str, default=None,
                        choices=['fixed', 'random'],
                        help='Lyapunov center layout: fixed (canonical 2D grid) or random')
    parser.add_argument('--lyapunov_center_scale', type=float, default=None,
                        help='Lyapunov random center range scale (uniform in [-scale, scale])')
    parser.add_argument('--lyapunov_min_separation', type=float, default=None,
                        help='Minimum separation between Lyapunov random centers')
    parser.add_argument('--lyapunov_init_range', type=float, default=None,
                        help='Lyapunov reset range (uniform in [-r, r])')
    parser.add_argument('--lyapunov_extend_mode', type=str, default=None,
                        choices=['embed', 'full'],
                        help='Lyapunov dimension extension mode: embed (2D + decay) or full')
    parser.add_argument('--lyapunov_extra_decay', type=float, default=None,
                        help='Decay rate for extra dimensions in embed mode')

    # Dysts utilities
    parser.add_argument('--list-dysts', '--list-envs', dest='list_dysts', action='store_true',
                        help='List built-in, analytic, and dysts environments and exit')
    parser.add_argument('--standardize', action='store_true',
                        help='Standardize dysts data (zero mean, unit variance). Recommended for dysts systems.')
    parser.add_argument('--dysts_ic_noise_scale', type=float, default=None,
                        help='Dysts IC noise scale (perturbation around default IC). '
                             'Smaller values keep trajectories near the canonical attractor.')
    parser.add_argument('--dysts_native_cache', action='store_true',
                        help='Use native dysts trajectory cache for training data')
    parser.add_argument('--dysts_cache_profile', type=str, default=None,
                        choices=['smoke', 'full', 'long60'],
                        help='Named dysts cache profile (smoke, full, or long60)')
    parser.add_argument('--dysts_cache_steps', type=int, default=None,
                        help='Length of each cached dysts trajectory')
    parser.add_argument('--dysts_cache_trajectories', type=int, default=None,
                        help='Number of cached dysts trajectories')
    parser.add_argument('--dysts_cache_warmup', type=int, default=None,
                        help='Warmup steps to discard from cached trajectories')
    parser.add_argument('--dysts_cache_dir', type=str, default=None,
                        help='Optional cache directory for reusing cached dysts trajectories')
    parser.add_argument('--dysts_cache_reuse', action='store_true',
                        help='Reuse/load/save on-disk dysts cache when cache dir is set')
    parser.add_argument('--dysts_cache_split', type=str, default='train',
                        choices=['train', 'val', 'policy', 'test'],
                        help='Cache split namespace to use')
    parser.add_argument('--dysts_cache_num_workers', type=int, default=None,
                        help='Parallel workers for dysts native cache construction')
    parser.add_argument('--disable_dysts_auto_cache', action='store_true',
                        help='Disable automatic cache defaults for dysts environments')

    # Training
    parser.add_argument('--num_steps', type=int, default=20000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--hard_init_oversample', type=parse_optional_bool, default=None,
                        help='Enable training-time oversampling of hard initial conditions near separatrices')
    parser.add_argument('--hard_init_fraction', type=float, default=None,
                        help='Fraction of training resets drawn from the hard-initial-condition pool')
    parser.add_argument('--hard_init_pool_size', type=int, default=None,
                        help='Number of cached hard initial conditions retained for training-time oversampling')
    parser.add_argument('--hard_init_num_candidates', type=int, default=None,
                        help='Number of reset samples scored when building the hard-initial-condition pool')
    parser.add_argument('--hard_init_probe_steps', type=int, default=None,
                        help='Short rollout horizon used to score hard initial conditions')
    parser.add_argument('--hard_init_num_perturbations', type=int, default=None,
                        help='Number of local perturbations per candidate when scoring hard initial conditions')
    parser.add_argument('--hard_init_perturb_scale', type=float, default=None,
                        help='Relative perturbation scale used when scoring hard initial conditions')
    parser.add_argument('--hard_init_transient_window', type=int, default=None,
                        help='Late-window length used to score lingering transient motion')
    parser.add_argument('--hard_init_transient_weight', type=float, default=None,
                        help='Weight on the lingering-transient term in the hard-initial-condition score')
    parser.add_argument('--hard_init_jitter_scale', type=float, default=None,
                        help='Relative jitter applied when resampling from the hard-initial-condition pool')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config default)')
    parser.add_argument('--k_matrix_lr', type=float, default=None,
                        help='Koopman-matrix learning rate (overrides config default)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Weight decay (overrides config default)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # Model
    parser.add_argument('--target_size', type=int, default=None,
                        help='Latent dimension (overrides config default)')
    parser.add_argument('--sparsity_coeff', type=float, default=None,
                        help='Sparsity loss weight (overrides config default)')
    parser.add_argument('--sparsity_target', type=str, default=None,
                        choices=['rollout', 'encoded', 'encoded_rollout'],
                        help='Latent tensor for L1 sparsity: rollout, encoded, or average of both')
    parser.add_argument('--reconst_coeff', type=float, default=None,
                        help='Reconstruction loss weight (overrides config default)')
    parser.add_argument('--pred_coeff', type=float, default=None,
                        help='Prediction loss weight (overrides config default)')
    parser.add_argument('--res_coeff', type=float, default=None,
                        help='Residual/alignment loss weight (overrides config default)')
    parser.add_argument('--lista_alpha', type=float, default=None,
                        help='LISTA soft-threshold alpha (overrides config default)')
    parser.add_argument('--lista_num_loops', type=int, default=None,
                        help='Number of LISTA iterations (overrides config default)')
    parser.add_argument('--lista_use_momentum', type=parse_optional_bool, default=None,
                        help='Enable fixed-beta momentum in standard LISTA refinement')
    parser.add_argument('--lista_momentum_beta', type=float, default=None,
                        help='Momentum coefficient for standard LISTA refinement')
    parser.add_argument('--lista_linear_encoder', type=parse_optional_bool, default=None,
                        help='Use a linear LISTA pre-code instead of the default MLP pre-code')
    parser.add_argument('--lista_final_op', type=str, default=None,
                        choices=['shrink', 'relu', 'sign_split'],
                        help='LISTA final nonlinearity: shrink, relu, or sign_split')
    parser.add_argument('--lista_precode_mode', type=str, default=None,
                        choices=['auto', 'free_mlp', 'linear', 'dictionary_tied', 'hybrid'],
                        help='LISTA pre-code mode: auto, free_mlp, linear, dictionary_tied, or hybrid')
    parser.add_argument('--lista_precode_residual_scale', type=float, default=None,
                        help='Residual MLP scale for LISTA hybrid pre-code')
    parser.add_argument('--lista_adaptive_thresholds', type=parse_optional_bool, default=None,
                        help='Enable sample-dependent LISTA thresholds based on reconstruction/prior mismatch')
    parser.add_argument('--lista_alpha_residual_coeff', type=float, default=None,
                        help='LISTA adaptive-threshold coefficient on the reconstruction residual term')
    parser.add_argument('--lista_alpha_prior_coeff', type=float, default=None,
                        help='LISTA adaptive-threshold coefficient on the latent-prior mismatch term')
    parser.add_argument('--lista_groupwise_thresholds', type=parse_optional_bool, default=None,
                        help='Use separate learned LISTA base thresholds per inferred latent group')
    parser.add_argument('--encoder_group_shrinkage', type=parse_optional_bool, default=None,
                        help='Enable sparse-group shrinkage over inferred latent groups for LISTA/HyperLISTA encoders')
    parser.add_argument('--encoder_group_threshold_scale', type=float, default=None,
                        help='Group threshold multiplier relative to the elementwise LISTA/HyperLISTA threshold')
    parser.add_argument('--encoder_topk_groups', type=int, default=None,
                        help='Keep only the top-k latent groups before within-group thresholding (0 disables)')
    parser.add_argument('--decoder_coherence_weight', type=float, default=None,
                        help='Weight for the normalized decoder coherence penalty')
    parser.add_argument('--normalize_decoder_atoms', type=parse_optional_bool, default=None,
                        help='Normalize GenericKM linear decoder atoms at decode time')

    # Koopman matrix structure
    parser.add_argument('--k_structure', type=str, default=None,
                        choices=['dense', 'diagonal', 'block_diagonal'],
                        help='Koopman matrix structure (default: dense)')
    parser.add_argument('--k_block_size', type=int, default=None,
                        help='Block size for block_diagonal K (default: auto = target_size // 13)')
    parser.add_argument('--k_num_blocks', type=int, default=None,
                        help='Exact number of blocks for block_diagonal K (overrides k_block_size)')

    # Block activation losses (for block_diagonal K)
    parser.add_argument('--block_loss', action='store_true',
                        help='Enable block activation losses for block_diagonal K')
    parser.add_argument('--block_one_block_loss', type=str, default=None,
                        choices=['none', 'low_entropy', 'pairwise_overlap', 'top1_margin'],
                        help='Per-sample one-block loss type (default: none)')
    parser.add_argument('--block_one_block_weight', type=float, default=None,
                        help='Weight for per-sample one-block loss (default: 0)')
    parser.add_argument('--block_top1_margin', type=float, default=None,
                        help='Margin for top1_margin loss (default: 0.1)')
    parser.add_argument('--block_balance_loss', type=str, default=None,
                        choices=['none', 'usage_entropy', 'kl_uniform'],
                        help='Across-batch balance loss type (default: none)')
    parser.add_argument('--block_balance_weight', type=float, default=None,
                        help='Weight for across-batch balance loss (default: 0)')
    parser.add_argument('--block_energy_norm', type=str, default=None,
                        choices=['l1', 'l2'],
                        help='Energy norm for block activations (default: l2)')
    parser.add_argument('--soft_block', action='store_true',
                        help='Enable off-block penalty for dense soft block-sparse Koopman matrices')
    parser.add_argument('--soft_block_num_blocks', type=int, default=None,
                        help='Number of latent blocks used by the off-block penalty mask')
    parser.add_argument('--soft_block_weight', type=float, default=None,
                        help='Weight for dense-K off-block penalty')
    parser.add_argument('--soft_block_norm', type=str, default=None,
                        choices=['l1', 'fro'],
                        help='Norm for dense-K off-block penalty (default: l1)')

    # HyperLISTA hyperparameters (for LISTAKM with ENCODER_TYPE=hyperlista)
    parser.add_argument('--hyperlista_c_theta', type=float, default=None,
                        help='HyperLISTA threshold scaling C_THETA (overrides config default)')
    parser.add_argument('--hyperlista_c_beta', type=float, default=None,
                        help='HyperLISTA momentum scaling C_BETA (overrides config default)')
    parser.add_argument('--hyperlista_c_ss', type=float, default=None,
                        help='HyperLISTA support-selection scaling C_SS (overrides config default)')
    parser.add_argument('--hyperlista_step_scale', type=float, default=None,
                        help='HyperLISTA gradient step multiplier applied to 1/L')
    parser.add_argument('--hyperlista_use_ss', type=parse_optional_bool, default=None,
                        help='Enable or disable HyperLISTA support selection')
    parser.add_argument('--hyperlista_use_momentum', type=parse_optional_bool, default=None,
                        help='Enable or disable HyperLISTA momentum')
    parser.add_argument('--hyperlista_constrain_c_theta', type=parse_optional_bool, default=None,
                        help='Constrain HyperLISTA c_theta to stay strictly positive')
    parser.add_argument('--hyperlista_c_theta_min', type=float, default=None,
                        help='Minimum HyperLISTA c_theta value when constrained')

    parser.add_argument('--sequence_length', type=int, default=1,
                        help='Unified rollout horizon H (H=1 matches former pairwise training)')
    parser.add_argument('--pilot_warmup_steps', type=int, default=0,
                        help='Opt-in utilization pilot warmup optimizer steps (default: disabled)')
    parser.add_argument('--pilot_measure_steps', type=int, default=0,
                        help='Opt-in utilization pilot measured optimizer steps (default: disabled)')
    parser.add_argument('--pilot_profile', action='store_true',
                        help='Wrap the opt-in pilot measured range in CUDA profiler boundaries')
    parser.add_argument('--pilot_timing_path', type=str, default=None,
                        help='Atomic JSON receipt path for the opt-in pilot measured range')
    parser.add_argument('--eval_every', type=int, default=None,
                        help='Evaluate every N steps during training (overrides config default)')
    parser.add_argument('--eval_num_steps', type=int, default=None,
                        help='Rollout horizon for the quick eval during training (overrides config default)')
    parser.add_argument('--skip_eval', action='store_true',
                        help='Skip standardized evaluation suite after training')
    parser.add_argument('--eval_profile', type=str, default='full', choices=['full', 'smoke'],
                        help='Evaluation profile for post-training standardized evaluation')
    parser.add_argument('--eval_use_dynamics_prior', type=parse_optional_bool, default=None,
                        help='Warm-start reencoding with the predicted latent during standardized evaluation')
    parser.add_argument('--eval_event_trigger_proj_threshold', type=float, default=None,
                        help='Projection-gap threshold for event-triggered reencoding during standardized evaluation')
    parser.add_argument('--eval_event_trigger_ambiguity_threshold', type=float, default=None,
                        help='Group-ambiguity threshold for event-triggered reencoding during standardized evaluation')
    parser.add_argument('--eval_event_trigger_spillover_threshold', type=float, default=None,
                        help='Off-group spillover threshold for event-triggered reencoding during standardized evaluation')
    parser.add_argument('--eval_event_trigger_support_margin_min_ratio', type=float, default=None,
                        help='Minimum support-margin ratio before event-triggered reencoding fires during standardized evaluation')
    parser.add_argument('--eval_event_trigger_support_threshold', type=float, default=1e-3,
                        help='Support threshold used when computing support-margin trigger ratios during standardized evaluation')
    parser.add_argument('--eval_event_trigger_min_dwell', type=int, default=0,
                        help='Minimum number of steps between event-triggered resets during standardized evaluation')
    parser.add_argument('--eval_event_trigger_max_interval', type=int, default=0,
                        help='Maximum steps allowed between event-triggered resets during standardized evaluation (0 disables)')
    parser.add_argument('--save_metrics_history', action='store_true',
                        help='Write raw per-step metrics_history.jsonl. Off by default; metrics_summary.json is always written.')
    parser.add_argument('--save_training_plot', action='store_true',
                        help='Render training_metrics.png after training. Implies --save_metrics_history.')
    parser.add_argument('--save_last_checkpoint', action='store_true',
                        help='Also save last.pt for resumability/debugging. checkpoint.pt is always saved.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Persistent complete-state checkpoint directory; enables exact resume.')
    parser.add_argument('--checkpoint_interval', type=int, default=None,
                        help='Completed steps between complete-state checkpoint generations (default: 100).')
    parser.add_argument('--checkpoint_retention', type=int, default=3,
                        help='Number of recent valid complete checkpoint generations to retain (minimum 2).')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the newest valid complete checkpoint in --checkpoint_dir.')
    parser.add_argument('--resume_if_available', action='store_true',
                        help='Resume from a valid complete checkpoint when present; otherwise start fresh.')
    parser.add_argument('--permanent_checkpoint_dir', type=str, default=None,
                        help='Optional durable destination receiving atomic latest.pt/best.pt copies and manifests.')
    parser.add_argument('--save_eval_rollout_artifacts', action='store_true',
                        help='Save raw standardized-evaluation rollout tensors.')
    parser.add_argument('--save_eval_plots', action='store_true',
                        help='Render standardized-evaluation qualitative plots.')
    parser.add_argument('--save_eval_per_ic_values', action='store_true',
                        help='Include per-initial-condition metric arrays in evaluation JSON.')
    parser.add_argument('--save_eval_error_curves', action='store_true',
                        help='Include long per-step error curves in evaluation JSON.')

    # Logging
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for logs and checkpoints (defaults: ./runs/hyperlista for LISTAKM+hyperlista, ./runs/lista for other LISTAKM, ./runs/kae otherwise)')
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
        print("AVAILABLE ENVIRONMENTS")
        print("=" * 60)
        try:
            from skae.data import get_available_environments
            envs = get_available_environments()

            print(f"\nBuilt-in environments ({len(envs['builtin'])}):")
            for env in envs['builtin']:
                print(f"  {env}")

            print(f"\nAnalytic systems ({len(envs['analytic'])}):")
            if envs['analytic']:
                systems = envs['analytic']
                for i in range(0, len(systems), 4):
                    row = systems[i:i+4]
                    print("  " + "  ".join(f"{s:<20}" for s in row))
                print(
                    "\nUsage: --env analytic:SystemName "
                    '(e.g., --env analytic:cal_square_4)'
                )
            else:
                print("  (analytic systems not available)")

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

    # Lyapunov environment overrides
    if args.lyapunov_dim is not None:
        cfg.ENV.LYAPUNOV.DIM = args.lyapunov_dim
    if args.lyapunov_num_basins is not None:
        cfg.ENV.LYAPUNOV.NUM_BASINS = args.lyapunov_num_basins
    if args.lyapunov_points_mode is not None:
        cfg.ENV.LYAPUNOV.POINTS_MODE = args.lyapunov_points_mode
    if args.lyapunov_center_scale is not None:
        cfg.ENV.LYAPUNOV.CENTER_SCALE = args.lyapunov_center_scale
    if args.lyapunov_min_separation is not None:
        cfg.ENV.LYAPUNOV.MIN_SEPARATION = args.lyapunov_min_separation
    if args.lyapunov_init_range is not None:
        cfg.ENV.LYAPUNOV.INIT_RANGE = args.lyapunov_init_range
    if args.lyapunov_extend_mode is not None:
        cfg.ENV.LYAPUNOV.EXTEND_MODE = args.lyapunov_extend_mode
    if args.lyapunov_extra_decay is not None:
        cfg.ENV.LYAPUNOV.EXTRA_DECAY = args.lyapunov_extra_decay
    if args.env_dt is not None:
        apply_env_dt_override(cfg, dt=float(args.env_dt), env_name=args.env)
        print(f"Using environment dt override: {float(args.env_dt):.8g}")

    # Override config with command-line args
    if args.hard_init_oversample is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.ENABLED = args.hard_init_oversample
    if args.hard_init_fraction is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.FRACTION = float(args.hard_init_fraction)
    if args.hard_init_pool_size is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.POOL_SIZE = int(args.hard_init_pool_size)
    if args.hard_init_num_candidates is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.NUM_CANDIDATES = int(args.hard_init_num_candidates)
    if args.hard_init_probe_steps is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.PROBE_STEPS = int(args.hard_init_probe_steps)
    if args.hard_init_num_perturbations is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.NUM_PERTURBATIONS = int(args.hard_init_num_perturbations)
    if args.hard_init_perturb_scale is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.PERTURB_SCALE = float(args.hard_init_perturb_scale)
    if args.hard_init_transient_window is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.TRANSIENT_WINDOW = int(args.hard_init_transient_window)
    if args.hard_init_transient_weight is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.TRANSIENT_WEIGHT = float(args.hard_init_transient_weight)
    if args.hard_init_jitter_scale is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.JITTER_SCALE = float(args.hard_init_jitter_scale)

    if args.lr is not None:
        cfg.TRAIN.LR = args.lr
    if args.k_matrix_lr is not None:
        cfg.TRAIN.K_MATRIX_LR = args.k_matrix_lr
    if args.weight_decay is not None:
        cfg.TRAIN.WEIGHT_DECAY = args.weight_decay
    if args.target_size is not None:
        cfg.MODEL.TARGET_SIZE = args.target_size
    if args.sparsity_coeff is not None:
        cfg.MODEL.SPARSITY_COEFF = args.sparsity_coeff
    if args.reconst_coeff is not None:
        cfg.MODEL.RECONST_COEFF = args.reconst_coeff
    if args.pred_coeff is not None:
        cfg.MODEL.PRED_COEFF = args.pred_coeff
    if args.res_coeff is not None:
        cfg.MODEL.RES_COEFF = args.res_coeff
    if args.lista_alpha is not None:
        cfg.MODEL.ENCODER.LISTA.ALPHA = args.lista_alpha
    if args.lista_num_loops is not None:
        # Update both LISTA and HyperLISTA loop counts for convenience
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = args.lista_num_loops
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = args.lista_num_loops
    if args.lista_use_momentum is not None:
        cfg.MODEL.ENCODER.LISTA.USE_MOMENTUM = args.lista_use_momentum
    if args.lista_momentum_beta is not None:
        cfg.MODEL.ENCODER.LISTA.MOMENTUM_BETA = args.lista_momentum_beta
    if args.lista_linear_encoder is not None:
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = args.lista_linear_encoder
    if args.lista_final_op is not None:
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = args.lista_final_op
    if args.lista_precode_mode is not None:
        cfg.MODEL.ENCODER.LISTA.PRECODE_MODE = args.lista_precode_mode
    if args.lista_precode_residual_scale is not None:
        cfg.MODEL.ENCODER.LISTA.PRECODE_RESIDUAL_SCALE = args.lista_precode_residual_scale
    if args.lista_adaptive_thresholds is not None:
        cfg.MODEL.ENCODER.LISTA.ADAPTIVE_THRESHOLDS = args.lista_adaptive_thresholds
    if args.lista_alpha_residual_coeff is not None:
        cfg.MODEL.ENCODER.LISTA.ALPHA_RESIDUAL_COEFF = args.lista_alpha_residual_coeff
    if args.lista_alpha_prior_coeff is not None:
        cfg.MODEL.ENCODER.LISTA.ALPHA_PRIOR_COEFF = args.lista_alpha_prior_coeff
    if args.lista_groupwise_thresholds is not None:
        cfg.MODEL.ENCODER.LISTA.GROUPWISE_THRESHOLDS = args.lista_groupwise_thresholds
    if args.encoder_group_shrinkage is not None:
        cfg.MODEL.ENCODER.LISTA.GROUP_SHRINKAGE = args.encoder_group_shrinkage
        cfg.MODEL.ENCODER.HYPERLISTA.GROUP_SHRINKAGE = args.encoder_group_shrinkage
    if args.encoder_group_threshold_scale is not None:
        cfg.MODEL.ENCODER.LISTA.GROUP_THRESHOLD_SCALE = args.encoder_group_threshold_scale
        cfg.MODEL.ENCODER.HYPERLISTA.GROUP_THRESHOLD_SCALE = args.encoder_group_threshold_scale
    if args.encoder_topk_groups is not None:
        cfg.MODEL.ENCODER.LISTA.TOPK_GROUPS = args.encoder_topk_groups
        cfg.MODEL.ENCODER.HYPERLISTA.TOPK_GROUPS = args.encoder_topk_groups
    if args.decoder_coherence_weight is not None:
        cfg.MODEL.DECODER_COHERENCE_WEIGHT = args.decoder_coherence_weight
    if args.normalize_decoder_atoms is not None:
        cfg.MODEL.DECODER.NORMALIZE_ATOMS = args.normalize_decoder_atoms
    if args.sparsity_target is not None:
        cfg.MODEL.SPARSITY_TARGET = args.sparsity_target
    if args.k_structure is not None:
        cfg.MODEL.K_STRUCTURE = args.k_structure
        print(f"Using Koopman matrix structure: {args.k_structure}")
    if args.k_block_size is not None:
        cfg.MODEL.K_BLOCK_SIZE = args.k_block_size
    if args.k_num_blocks is not None:
        cfg.MODEL.K_NUM_BLOCKS = args.k_num_blocks

    # Block activation loss config
    if args.block_loss:
        cfg.MODEL.BLOCK_LOSS.ENABLED = True
    if args.block_one_block_loss is not None:
        cfg.MODEL.BLOCK_LOSS.ONE_BLOCK_LOSS = args.block_one_block_loss
        cfg.MODEL.BLOCK_LOSS.ENABLED = True
    if args.block_one_block_weight is not None:
        cfg.MODEL.BLOCK_LOSS.ONE_BLOCK_WEIGHT = args.block_one_block_weight
        cfg.MODEL.BLOCK_LOSS.ENABLED = True
    if args.block_top1_margin is not None:
        cfg.MODEL.BLOCK_LOSS.TOP1_MARGIN = args.block_top1_margin
        cfg.MODEL.BLOCK_LOSS.ENABLED = True
    if args.block_balance_loss is not None:
        cfg.MODEL.BLOCK_LOSS.BALANCE_LOSS = args.block_balance_loss
        cfg.MODEL.BLOCK_LOSS.ENABLED = True
    if args.block_balance_weight is not None:
        cfg.MODEL.BLOCK_LOSS.BALANCE_WEIGHT = args.block_balance_weight
        cfg.MODEL.BLOCK_LOSS.ENABLED = True
    if args.block_energy_norm is not None:
        cfg.MODEL.BLOCK_LOSS.ENERGY_NORM = args.block_energy_norm
        cfg.MODEL.BLOCK_LOSS.ENABLED = True

    if args.soft_block:
        cfg.MODEL.SOFT_BLOCK.ENABLED = True
    if args.soft_block_num_blocks is not None:
        cfg.MODEL.SOFT_BLOCK.NUM_BLOCKS = args.soft_block_num_blocks
        cfg.MODEL.SOFT_BLOCK.ENABLED = True
    if args.soft_block_weight is not None:
        cfg.MODEL.SOFT_BLOCK.WEIGHT = args.soft_block_weight
        cfg.MODEL.SOFT_BLOCK.ENABLED = args.soft_block_weight > 0.0 or cfg.MODEL.SOFT_BLOCK.ENABLED
    if args.soft_block_norm is not None:
        cfg.MODEL.SOFT_BLOCK.NORM = args.soft_block_norm
        cfg.MODEL.SOFT_BLOCK.ENABLED = True

    if args.hyperlista_c_theta is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = args.hyperlista_c_theta
    if args.hyperlista_c_beta is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.C_BETA = args.hyperlista_c_beta
    if args.hyperlista_c_ss is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.C_SS = args.hyperlista_c_ss
    if args.hyperlista_step_scale is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.STEP_SCALE = args.hyperlista_step_scale
    if args.hyperlista_use_ss is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.USE_SUPPORT_SELECTION = args.hyperlista_use_ss
    if args.hyperlista_use_momentum is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.USE_MOMENTUM = args.hyperlista_use_momentum
    if args.hyperlista_constrain_c_theta is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.CONSTRAIN_C_THETA = args.hyperlista_constrain_c_theta
    if args.hyperlista_c_theta_min is not None:
        cfg.MODEL.ENCODER.HYPERLISTA.C_THETA_MIN = args.hyperlista_c_theta_min

    # Dysts standardization
    is_dysts_env = cfg.ENV.ENV_NAME.lower().startswith("dysts:")
    if args.standardize:
        cfg.ENV.DYSTS.STANDARDIZE = True
        print("Using standardized dysts data (zero mean, unit variance)")
    if args.dysts_ic_noise_scale is not None:
        cfg.ENV.DYSTS.IC_NOISE_SCALE = float(args.dysts_ic_noise_scale)
        print(f"Using dysts IC noise scale: {cfg.ENV.DYSTS.IC_NOISE_SCALE}")
    cfg.ENV.DYSTS.CACHE_SPLIT = args.dysts_cache_split

    # Auto-defaults for dysts runs: use shared cache by default unless disabled.
    if is_dysts_env and not args.disable_dysts_auto_cache:
        cfg.ENV.DYSTS.USE_NATIVE_CACHE = True
        if not cfg.ENV.DYSTS.STANDARDIZE:
            cfg.ENV.DYSTS.STANDARDIZE = True
            print("Using standardized dysts data (zero mean, unit variance)")
        if args.dysts_cache_profile is None:
            no_manual_shape = (
                args.dysts_cache_steps is None
                and args.dysts_cache_trajectories is None
                and args.dysts_cache_warmup is None
            )
            if no_manual_shape:
                profile = apply_dysts_cache_profile(cfg, "full")
                print(
                    "Auto dysts cache profile 'full' "
                    f"(steps={profile['steps']}, trajectories={profile['trajectories']}, "
                    f"warmup={profile['warmup']})"
                )
        if args.dysts_cache_dir is None and not cfg.ENV.DYSTS.CACHE_DIR:
            cfg.ENV.DYSTS.CACHE_DIR = default_dysts_cache_dir()
        if not args.dysts_cache_reuse:
            cfg.ENV.DYSTS.CACHE_REUSE = True
        if args.dysts_cache_num_workers is None and cfg.ENV.DYSTS.CACHE_NUM_WORKERS < 2:
            cfg.ENV.DYSTS.CACHE_NUM_WORKERS = 2

    if args.dysts_cache_profile is not None:
        cfg.ENV.DYSTS.USE_NATIVE_CACHE = True
        profile = apply_dysts_cache_profile(cfg, args.dysts_cache_profile)
        print(
            f"Using dysts cache profile '{args.dysts_cache_profile}' "
            f"(steps={profile['steps']}, trajectories={profile['trajectories']}, "
            f"warmup={profile['warmup']})"
        )
    if args.dysts_native_cache:
        cfg.ENV.DYSTS.USE_NATIVE_CACHE = True
        print("Using dysts native trajectory cache for training data")
    if args.dysts_cache_steps is not None:
        cfg.ENV.DYSTS.CACHE_STEPS = args.dysts_cache_steps
    if args.dysts_cache_trajectories is not None:
        cfg.ENV.DYSTS.CACHE_TRAJECTORIES = args.dysts_cache_trajectories
    if args.dysts_cache_warmup is not None:
        cfg.ENV.DYSTS.CACHE_WARMUP = args.dysts_cache_warmup
    if args.dysts_cache_dir is not None:
        cfg.ENV.DYSTS.CACHE_DIR = args.dysts_cache_dir
    if args.dysts_cache_reuse:
        cfg.ENV.DYSTS.CACHE_REUSE = True
    if args.dysts_cache_num_workers is not None:
        cfg.ENV.DYSTS.CACHE_NUM_WORKERS = int(args.dysts_cache_num_workers)
    if is_dysts_env and cfg.ENV.DYSTS.USE_NATIVE_CACHE:
        print(
            "Dysts cache config: "
            f"split={cfg.ENV.DYSTS.CACHE_SPLIT}, "
            f"steps={cfg.ENV.DYSTS.CACHE_STEPS}, "
            f"trajectories={cfg.ENV.DYSTS.CACHE_TRAJECTORIES}, "
            f"warmup={cfg.ENV.DYSTS.CACHE_WARMUP}, "
            f"reuse={cfg.ENV.DYSTS.CACHE_REUSE}, "
            f"dir='{cfg.ENV.DYSTS.CACHE_DIR}'"
        )

    cfg.TRAIN.SEQUENCE_LENGTH = int(args.sequence_length)
    if cfg.TRAIN.SEQUENCE_LENGTH < 1:
        raise ValueError("--sequence_length must be >= 1")
    print(f"Using unified horizon-based training with sequence_length={cfg.TRAIN.SEQUENCE_LENGTH}")
    if args.eval_every is not None:
        cfg.TRAIN.EVAL_EVERY = int(args.eval_every)
    if args.eval_num_steps is not None:
        cfg.TRAIN.EVAL_NUM_STEPS = int(args.eval_num_steps)

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
    train(
        cfg,
        log_dir=args.log_dir,
        checkpoint_path=args.checkpoint,
        device=device,
        skip_eval=args.skip_eval,
        eval_profile=args.eval_profile,
        eval_use_dynamics_prior=bool(args.eval_use_dynamics_prior) if args.eval_use_dynamics_prior is not None else False,
        eval_event_trigger_proj_threshold=args.eval_event_trigger_proj_threshold,
        eval_event_trigger_ambiguity_threshold=args.eval_event_trigger_ambiguity_threshold,
        eval_event_trigger_spillover_threshold=args.eval_event_trigger_spillover_threshold,
        eval_event_trigger_support_margin_min_ratio=args.eval_event_trigger_support_margin_min_ratio,
        eval_event_trigger_support_threshold=args.eval_event_trigger_support_threshold,
        eval_event_trigger_min_dwell=args.eval_event_trigger_min_dwell,
        eval_event_trigger_max_interval=args.eval_event_trigger_max_interval,
        save_metrics_history=args.save_metrics_history,
        save_training_plot=args.save_training_plot,
        save_last_checkpoint=args.save_last_checkpoint,
        save_eval_rollout_artifacts=args.save_eval_rollout_artifacts,
        save_eval_plots=args.save_eval_plots,
        save_eval_per_ic_values=args.save_eval_per_ic_values,
        save_eval_error_curves=args.save_eval_error_curves,
        pilot_warmup_steps=args.pilot_warmup_steps,
        pilot_measure_steps=args.pilot_measure_steps,
        pilot_profile=args.pilot_profile,
        pilot_timing_path=args.pilot_timing_path,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_retention=args.checkpoint_retention,
        resume=args.resume,
        resume_if_available=args.resume_if_available,
        permanent_checkpoint_dir=args.permanent_checkpoint_dir,
    )


if __name__ == '__main__':
    main()
