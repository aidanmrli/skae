"""Train the minimal LISTA path on a spatialized reaction-diffusion dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from skae.benchmarks.spatialized_reaction_diffusion import flatten_fields, load_dataset, split_fields
from skae.config import Config, get_config
from skae.model import make_model

MIN_LATENT_STATE_RATIO = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a flattened-field LISTA Koopman model for the PDE smoke benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="lista_parity_generic_sparse")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k_matrix_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lista_num_loops", type=int, default=2)
    parser.add_argument("--lista_alpha", type=float, default=0.1)
    parser.add_argument("--reconst_coeff", type=float, default=0.5)
    parser.add_argument("--res_coeff", type=float, default=1.0)
    parser.add_argument("--pred_coeff", type=float, default=1.0)
    parser.add_argument("--sparsity_coeff", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_horizon", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return "cpu"
    return device_arg


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    kmat_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "kmat" in name or name.startswith("K_"):
            kmat_params.append(param)
        else:
            other_params.append(param)

    groups = []
    if other_params:
        groups.append(
            {
                "params": other_params,
                "lr": cfg.TRAIN.LR,
                "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
            }
        )
    if kmat_params:
        groups.append(
            {
                "params": kmat_params,
                "lr": cfg.TRAIN.K_MATRIX_LR,
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW(groups)


def sample_sequence_batch(
    fields: torch.Tensor,
    *,
    batch_size: int,
    window_length: int,
    generator: torch.Generator,
    device: str,
) -> torch.Tensor:
    traj_count, time_count, _obs = fields.shape
    max_start = time_count - (window_length + 1)
    if max_start < 0:
        raise ValueError(
            f"Dataset has {time_count} stored states, too short for sequence_length={window_length}."
        )
    traj_idx = torch.randint(0, traj_count, (batch_size,), generator=generator)
    time_idx = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    offsets = torch.arange(window_length + 1)
    seq_idx = time_idx[:, None] + offsets[None, :]
    return fields[traj_idx[:, None], seq_idx].to(device)


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, x_seq: torch.Tensor, step: int) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    if x_seq.ndim != 3 or x_seq.shape[1] < 2:
        raise ValueError("x_seq must have shape [batch, horizon+1, obs].")
    batch_size, seq_len, obs_size = x_seq.shape
    horizon = seq_len - 1

    x0 = x_seq[:, 0, :]
    x_true = x_seq[:, 1:, :]
    z_all = model.encode(x_seq.reshape(batch_size * seq_len, obs_size)).reshape(batch_size, seq_len, -1)
    z0 = z_all[:, 0, :]
    z_true = z_all[:, 1:, :]
    z_pred = model.rollout_latent_discrete(z0, horizon=horizon)
    x_pred = model.decode(z_pred.reshape(batch_size * horizon, -1)).reshape(batch_size, horizon, obs_size)
    x_recon_true = model.decode(z_true.reshape(batch_size * horizon, -1)).reshape(batch_size, horizon, obs_size)

    loss, metrics = model.loss(
        x_pred=x_pred,
        x_true=x_true,
        x0=x0,
        z0=z0,
        z_pred=z_pred,
        z_true=z_true,
        reconstruction_error=torch.norm(x_true - x_recon_true, dim=-1).mean(),
        sparsity_latent=z_pred,
        step=step,
    )
    loss.backward()
    optimizer.step()
    return metrics


def jsonable_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    """Convert scalar tensor/number metrics to JSON values without dropping diagnostics."""

    out: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                out[key] = float(value.detach().cpu().item())
            else:
                out[key] = value.detach().cpu().tolist()
            continue
        try:
            out[key] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            out[key] = value
    return out


@torch.no_grad()
def forecast_mse(
    model: nn.Module,
    fields: torch.Tensor,
    *,
    horizon: int,
    device: str,
    batch_size: int = 64,
) -> Tuple[float, float]:
    model.eval()
    horizon = min(int(horizon), int(fields.shape[1]) - 1)
    if horizon < 1:
        return float("nan"), float("nan")
    total_sse = 0.0
    total_count = 0
    total_final_sse = 0.0
    total_final_count = 0
    for start in range(0, fields.shape[0], batch_size):
        batch = fields[start : start + batch_size].to(device)
        x0 = batch[:, 0, :]
        truth = batch[:, 1 : horizon + 1, :]
        _z_pred, pred = model.rollout_observation_discrete(x0, horizon=horizon)
        diff = pred - truth
        total_sse += float(diff.square().sum().item())
        total_count += int(diff.numel())
        final_diff = diff[:, -1, :]
        total_final_sse += float(final_diff.square().sum().item())
        total_final_count += int(final_diff.numel())
    return total_sse / max(1, total_count), total_final_sse / max(1, total_final_count)


def configure_model(args: argparse.Namespace, metadata: Dict[str, object], observation_size: int) -> Config:
    cfg = get_config(args.config)
    cfg.SEED = int(args.seed if args.seed is not None else metadata.get("seed", 0))
    cfg.ENV.ENV_NAME = f"spatialized_reaction_diffusion:{metadata.get('source_system_name', 'unknown')}"
    cfg.TRAIN.NUM_STEPS = int(args.num_steps)
    cfg.TRAIN.BATCH_SIZE = int(args.batch_size)
    cfg.TRAIN.SEQUENCE_LENGTH = int(args.sequence_length)
    cfg.TRAIN.LR = float(args.lr)
    cfg.TRAIN.K_MATRIX_LR = float(args.k_matrix_lr)
    cfg.TRAIN.WEIGHT_DECAY = float(args.weight_decay)
    cfg.TRAIN.EVAL_EVERY = int(args.eval_every)
    min_target_size = int(math.ceil(MIN_LATENT_STATE_RATIO * int(observation_size)))
    requested_target_size = int(args.target_size)
    if requested_target_size <= 0:
        requested_target_size = min_target_size
    if requested_target_size < min_target_size:
        raise ValueError(
            "Spatialized PDE Koopman lifts must be overcomplete: "
            f"target_size={requested_target_size} is below {MIN_LATENT_STATE_RATIO:g} * "
            f"state_dim={observation_size} = {min_target_size}."
        )
    cfg.MODEL.TARGET_SIZE = requested_target_size
    cfg.MODEL.RES_COEFF = float(args.res_coeff)
    cfg.MODEL.RECONST_COEFF = float(args.reconst_coeff)
    cfg.MODEL.PRED_COEFF = float(args.pred_coeff)
    cfg.MODEL.SPARSITY_COEFF = float(args.sparsity_coeff)
    cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = int(args.lista_num_loops)
    cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = int(args.lista_num_loops)
    cfg.MODEL.ENCODER.LISTA.ALPHA = float(args.lista_alpha)
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.OBS_LOSS_DIM_NORMALIZATION = "sqrt_dim"
    return cfg


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    dataset_metadata: Dict[str, object],
    metrics: Dict[str, float],
    val_mse: float,
    val_final_mse: float,
) -> None:
    torch.save(
        {
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "dataset_metadata": dataset_metadata,
            "metrics": metrics,
            "val_mse": float(val_mse),
            "val_final_mse": float(val_final_mse),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    metadata = dataset["metadata"]
    train_fields = flatten_fields(split_fields(dataset, "train")).float()
    val_fields = flatten_fields(split_fields(dataset, "val")).float()
    observation_size = int(train_fields.shape[-1])
    cfg = configure_model(args, metadata, observation_size)
    device = resolve_device(args.device)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (run_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "run_dir": str(run_dir),
                "device": device,
                "train_shape": list(train_fields.shape),
                "val_shape": list(val_fields.shape),
                "observation_size": observation_size,
                "target_size": cfg.MODEL.TARGET_SIZE,
                "num_steps": cfg.TRAIN.NUM_STEPS,
                "sequence_length": cfg.TRAIN.SEQUENCE_LENGTH,
                "label_policy": metadata.get("training_label_policy"),
            },
            indent=2,
        ),
        flush=True,
    )

    model = make_model(cfg, observation_size).to(device)
    optimizer = build_optimizer(model, cfg)
    rng = torch.Generator().manual_seed(cfg.SEED + 12345)

    best_val = math.inf
    history_path = run_dir / "metrics_history.jsonl"
    if history_path.exists():
        history_path.unlink()
    last_metrics: Dict[str, object] = {}
    for step in range(cfg.TRAIN.NUM_STEPS):
        x_seq = sample_sequence_batch(
            train_fields,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            window_length=cfg.TRAIN.SEQUENCE_LENGTH,
            generator=rng,
            device=device,
        )
        last_metrics = train_step(model, optimizer, x_seq, step)

        if step % max(1, int(args.log_every)) == 0:
            print(
                "step={step} loss={loss:.6g} align={align:.6g} recon={recon:.6g} "
                "sparsity={sparsity:.4f} sr={sr:.4f}".format(
                    step=step,
                    loss=last_metrics.get("loss", float("nan")),
                    align=last_metrics.get("alignment_loss", float("nan")),
                    recon=last_metrics.get("reconst_loss", float("nan")),
                    sparsity=last_metrics.get("sparsity_ratio", float("nan")),
                    sr=last_metrics.get("spectral_radius", float("nan")),
                ),
                flush=True,
            )

        should_eval = (
            step == cfg.TRAIN.NUM_STEPS - 1
            or (step > 0 and step % max(1, int(args.eval_every)) == 0)
        )
        if should_eval:
            val_mse, val_final_mse = forecast_mse(
                model,
                val_fields,
                horizon=int(args.eval_horizon),
                device=device,
            )
            record = {
                "step": step,
                **jsonable_metrics(last_metrics),
                "val_mse": float(val_mse),
                "val_final_mse": float(val_final_mse),
                "eval_horizon": int(min(args.eval_horizon, val_fields.shape[1] - 1)),
            }
            with history_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            save_checkpoint(
                run_dir / "last.pt",
                step=step,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                dataset_metadata=metadata,
                metrics=last_metrics,
                val_mse=val_mse,
                val_final_mse=val_final_mse,
            )
            if val_mse < best_val:
                best_val = val_mse
                save_checkpoint(
                    run_dir / "checkpoint.pt",
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    cfg=cfg,
                    dataset_metadata=metadata,
                    metrics=last_metrics,
                    val_mse=val_mse,
                    val_final_mse=val_final_mse,
                )
            print(
                f"eval step={step} val_mse={val_mse:.6g} val_final_mse={val_final_mse:.6g} best_val_mse={best_val:.6g}",
                flush=True,
            )

    final_summary = {
        "status": "completed",
        "best_val_mse": float(best_val),
        "last_metrics": jsonable_metrics(last_metrics),
        "checkpoint": str(run_dir / "checkpoint.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
    }
    (run_dir / "training_summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    print(json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
