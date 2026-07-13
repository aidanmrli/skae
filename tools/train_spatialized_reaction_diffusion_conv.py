#!/usr/bin/env python3
"""Train convolutional Koopman models on spatialized reaction-diffusion fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from skae.benchmarks.spatialized_conv_koopman import (
    SpatialConvKoopman,
    SpatialConvKoopmanConfig,
    SpatialConvLossWeights,
    periodic_gradient_mse,
)
from skae.benchmarks.spatialized_reaction_diffusion import flatten_fields, load_dataset, split_fields

MIN_LATENT_STATE_RATIO = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a convolutional sparse/dense Koopman model for the spatialized PDE benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model_variant", default="conv_lista", choices=["conv_lista", "conv_dense", "conv_sparse_mlp"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--z_dim", type=int, default=0, help="Latent dimension. Values <=0 resolve to 4x the state dimension.")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument(
        "--conv_activation",
        default="",
        help=(
            "Override hidden conv-block activation for all variants. Empty keeps "
            "the default: tanh for dense and GELU for sparse/LISTA."
        ),
    )
    parser.add_argument("--lista_num_loops", type=int, default=2)
    parser.add_argument("--lista_alpha", type=float, default=1e-3)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument(
        "--train_observation_limit",
        type=int,
        default=0,
        help=(
            "If >0, sample training windows only from the first this many observation "
            "intervals. This allows long evaluation trajectories without training on "
            "their later frames."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k_matrix_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--prediction_weight", type=float, default=1.0)
    parser.add_argument("--reconstruction_weight", type=float, default=0.25)
    parser.add_argument("--latent_weight", type=float, default=0.1)
    parser.add_argument("--sparsity_weight", type=float, default=0.0)
    parser.add_argument("--k_stability_weight", type=float, default=1e-4)
    parser.add_argument("--gradient_weight", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_horizon", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--resume_from_latest", action="store_true", help="Resume from last.pt or checkpoint.pt in run_dir.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        return "cpu"
    return device_arg


def encoder_kind_for_variant(model_variant: str) -> str:
    if model_variant == "conv_lista":
        return "lista"
    if model_variant == "conv_dense":
        return "dense"
    if model_variant == "conv_sparse_mlp":
        return "sparse_mlp"
    raise ValueError(f"Unknown model_variant: {model_variant}")


def build_optimizer(model: SpatialConvKoopman, args: argparse.Namespace) -> torch.optim.Optimizer:
    kmat_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "kmat" in name:
            kmat_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})
    if kmat_params:
        groups.append({"params": kmat_params, "lr": float(args.k_matrix_lr), "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def _torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _find_resume_checkpoint(run_dir: Path) -> Path | None:
    candidates = [run_dir / "last.pt", run_dir / "checkpoint.pt"]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


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
        raise ValueError(f"Dataset has {time_count} states, too short for sequence_length={window_length}.")
    traj_idx = torch.randint(0, traj_count, (int(batch_size),), generator=generator)
    time_idx = torch.randint(0, max_start + 1, (int(batch_size),), generator=generator)
    offsets = torch.arange(int(window_length) + 1)
    seq_idx = time_idx[:, None] + offsets[None, :]
    return fields[traj_idx[:, None], seq_idx].to(device)


def jsonable_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().item()) if value.numel() == 1 else value.detach().cpu().tolist()
            continue
        try:
            out[key] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            out[key] = value
    return out


def train_step(
    model: SpatialConvKoopman,
    optimizer: torch.optim.Optimizer,
    x_seq: torch.Tensor,
    weights: SpatialConvLossWeights,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch_size, seq_len, obs_dim = x_seq.shape
    horizon = seq_len - 1

    z_all = model.encode(x_seq.reshape(batch_size * seq_len, obs_dim)).reshape(batch_size, seq_len, -1)
    z_roll = model.rollout_latent_discrete(z_all[:, 0], horizon=horizon)
    pred = model.decode(z_roll.reshape(batch_size * horizon, -1)).reshape(batch_size, horizon, obs_dim)
    recon = model.decode(z_all.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, obs_dim)

    prediction_loss = F.mse_loss(pred, x_seq[:, 1:])
    reconstruction_loss = F.mse_loss(recon, x_seq)
    latent_loss = F.mse_loss(z_roll, z_all[:, 1:].detach())
    sparsity_loss = z_all.abs().mean()
    gradient_loss = periodic_gradient_mse(
        pred,
        x_seq[:, 1:],
        grid_size=int(model.cfg.grid_size),
        channels=int(model.cfg.channels),
    )
    spectral_norm_proxy = torch.linalg.matrix_norm(model.kmat, ord="fro") / math.sqrt(model.kmat.shape[0])
    k_stability_loss = F.relu(spectral_norm_proxy - 1.25).square()
    loss = (
        weights.prediction * prediction_loss
        + weights.reconstruction * reconstruction_loss
        + weights.latent * latent_loss
        + weights.sparsity * sparsity_loss
        + weights.gradient * gradient_loss
        + weights.k_stability * k_stability_loss
    )
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        sparsity_ratio = (z_all.abs() <= 1e-4).float().mean()
    return {
        "loss": float(loss.detach().cpu()),
        "prediction_loss": float(prediction_loss.detach().cpu()),
        "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
        "latent_loss": float(latent_loss.detach().cpu()),
        "sparsity_loss": float(sparsity_loss.detach().cpu()),
        "gradient_loss": float(gradient_loss.detach().cpu()),
        "k_stability_loss": float(k_stability_loss.detach().cpu()),
        "sparsity_ratio_1e-4": float(sparsity_ratio.detach().cpu()),
        "k_fro_sqrt": float(spectral_norm_proxy.detach().cpu()),
    }


@torch.no_grad()
def forecast_mse(
    model: SpatialConvKoopman,
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
    for start in range(0, fields.shape[0], int(batch_size)):
        batch = fields[start : start + int(batch_size)].to(device)
        truth = batch[:, 1 : horizon + 1]
        _z_pred, pred = model.rollout_observation_discrete(batch[:, 0], horizon=horizon)
        diff = pred - truth
        total_sse += float(diff.square().sum().item())
        total_count += int(diff.numel())
        final_diff = diff[:, -1]
        total_final_sse += float(final_diff.square().sum().item())
        total_final_count += int(final_diff.numel())
    return total_sse / max(1, total_count), total_final_sse / max(1, total_final_count)


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: SpatialConvKoopman,
    optimizer: torch.optim.Optimizer,
    model_config: SpatialConvKoopmanConfig,
    loss_weights: SpatialConvLossWeights,
    dataset_metadata: Dict[str, object],
    metrics: Dict[str, float],
    val_mse: float,
    val_final_mse: float,
) -> None:
    torch.save(
        {
            "model_family": "spatial_conv_koopman",
            "step": int(step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config.to_dict(),
            "loss_weights": loss_weights.to_dict(),
            "dataset_metadata": dataset_metadata,
            "metrics": jsonable_metrics(metrics),
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
    original_train_shape = list(train_fields.shape)
    if int(args.train_observation_limit) > 0:
        max_frames = int(args.train_observation_limit) + 1
        if max_frames < int(args.sequence_length):
            raise ValueError(
                "train_observation_limit is too short for sequence_length: "
                f"train_observation_limit={int(args.train_observation_limit)}, "
                f"sequence_length={int(args.sequence_length)}."
            )
        if max_frames > train_fields.shape[1]:
            raise ValueError(
                "train_observation_limit exceeds available training trajectory: "
                f"train_observation_limit={int(args.train_observation_limit)}, "
                f"available_intervals={train_fields.shape[1] - 1}."
            )
        train_fields = train_fields[:, :max_frames, :]
    grid_size = int(metadata["grid_size"])
    observation_size = int(train_fields.shape[-1])
    min_z_dim = int(math.ceil(MIN_LATENT_STATE_RATIO * observation_size))
    if int(args.z_dim) <= 0:
        args.z_dim = min_z_dim
    if int(args.z_dim) < min_z_dim:
        raise ValueError(
            "Spatialized PDE Koopman lifts must be overcomplete: "
            f"z_dim={int(args.z_dim)} is below {MIN_LATENT_STATE_RATIO:g} * "
            f"state_dim={observation_size} = {min_z_dim}."
        )
    seed = int(args.seed if args.seed is not None else metadata.get("seed", 0))
    device = resolve_device(args.device)

    model_config = SpatialConvKoopmanConfig(
        grid_size=grid_size,
        channels=2,
        z_dim=int(args.z_dim),
        hidden_channels=int(args.hidden_channels),
        num_blocks=int(args.num_blocks),
        encoder_kind=encoder_kind_for_variant(args.model_variant),
        lista_loops=int(args.lista_num_loops),
        lista_alpha=float(args.lista_alpha),
        conv_activation=str(args.conv_activation),
    )
    loss_weights = SpatialConvLossWeights(
        prediction=float(args.prediction_weight),
        reconstruction=float(args.reconstruction_weight),
        latent=float(args.latent_weight),
        sparsity=float(args.sparsity_weight),
        k_stability=float(args.k_stability_weight),
        gradient=float(args.gradient_weight),
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_config.json").write_text(json.dumps(model_config.to_dict(), indent=2) + "\n")
    (run_dir / "loss_weights.json").write_text(json.dumps(loss_weights.to_dict(), indent=2) + "\n")
    (run_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    model = SpatialConvKoopman(model_config).to(device)
    optimizer = build_optimizer(model, args)
    rng = torch.Generator().manual_seed(seed + 12345)
    start_step = 0
    best_val = math.inf
    resume_path = _find_resume_checkpoint(run_dir) if bool(args.resume_from_latest) else None
    if resume_path is not None:
        payload = _torch_load(resume_path, device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_step = int(payload.get("step", -1)) + 1
        if (run_dir / "checkpoint.pt").exists():
            best_payload = _torch_load(run_dir / "checkpoint.pt", device)
            best_val = float(best_payload.get("val_mse", math.inf))
        else:
            best_val = float(payload.get("val_mse", math.inf))
        print(
            f"Resuming from {resume_path} at step={start_step} with best_val={best_val:.6g}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "run_dir": str(run_dir),
                "device": device,
                "model_variant": args.model_variant,
                "train_shape": list(train_fields.shape),
                "original_train_shape": original_train_shape,
                "train_observation_limit": int(args.train_observation_limit),
                "val_shape": list(val_fields.shape),
                "model_config": model_config.to_dict(),
                "loss_weights": loss_weights.to_dict(),
                "resume_path": str(resume_path) if resume_path is not None else None,
                "start_step": int(start_step),
                "label_policy": metadata.get("training_label_policy"),
            },
            indent=2,
        ),
        flush=True,
    )

    history_path = run_dir / "metrics_history.jsonl"
    if start_step <= 0 and history_path.exists():
        history_path.unlink()
    last_metrics: Dict[str, object] = {}
    for step in range(start_step, int(args.num_steps)):
        x_seq = sample_sequence_batch(
            train_fields,
            batch_size=int(args.batch_size),
            window_length=int(args.sequence_length),
            generator=rng,
            device=device,
        )
        last_metrics = train_step(model, optimizer, x_seq, loss_weights)
        if step % max(1, int(args.log_every)) == 0:
            print(
                "step={step} loss={loss:.6g} pred={pred:.6g} recon={recon:.6g} grad={grad:.6g} sparse={sparse:.4f}".format(
                    step=step,
                    loss=last_metrics.get("loss", float("nan")),
                    pred=last_metrics.get("prediction_loss", float("nan")),
                    recon=last_metrics.get("reconstruction_loss", float("nan")),
                    grad=last_metrics.get("gradient_loss", float("nan")),
                    sparse=last_metrics.get("sparsity_ratio_1e-4", float("nan")),
                ),
                flush=True,
            )
        should_eval = step == int(args.num_steps) - 1 or (step > 0 and step % max(1, int(args.eval_every)) == 0)
        if should_eval:
            val_mse, val_final_mse = forecast_mse(
                model,
                val_fields,
                horizon=int(args.eval_horizon),
                device=device,
            )
            record = {
                "step": int(step),
                **jsonable_metrics(last_metrics),
                "val_mse": float(val_mse),
                "val_final_mse": float(val_final_mse),
                "eval_horizon": int(min(args.eval_horizon, val_fields.shape[1] - 1)),
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            save_checkpoint(
                run_dir / "last.pt",
                step=step,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                loss_weights=loss_weights,
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
                    model_config=model_config,
                    loss_weights=loss_weights,
                    dataset_metadata=metadata,
                    metrics=last_metrics,
                    val_mse=val_mse,
                    val_final_mse=val_final_mse,
                )
            print(f"eval step={step} val_mse={val_mse:.6g} val_final_mse={val_final_mse:.6g} best={best_val:.6g}", flush=True)

    summary = {
        "status": "completed",
        "best_val_mse": float(best_val),
        "last_metrics": jsonable_metrics(last_metrics),
        "checkpoint": str(run_dir / "checkpoint.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "model_variant": args.model_variant,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
