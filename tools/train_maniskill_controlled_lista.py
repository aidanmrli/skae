"""Train the one-seed controlled LISTA/SKAE ManiSkill insertion model."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from skae.benchmarks.maniskill_controlled_lista import (
    ControlledLISTAConfig,
    ControlledLISTAKoopman,
    LossWeights,
    WindowSampler,
    compute_normalization,
    normalize_actions,
    normalize_observations,
    resolve_controlled_training_activation,
    save_checkpoint,
    train_step,
    validation_rollout_mse,
    write_json,
)
from skae.benchmarks.maniskill_insertion_dataset import load_compact_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Compact ManiSkill .npz dataset")
    parser.add_argument("--run_dir", type=Path, default=None, help="Output run directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--encoder_kind", default="lista", choices=("lista", "dense"))
    parser.add_argument("--activation", default="auto", choices=("auto", "relu", "tanh", "gelu"))
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--lista_loops", type=int, default=2)
    parser.add_argument("--lista_alpha", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--prediction_weight", type=float, default=1.0)
    parser.add_argument("--reconstruction_weight", type=float, default=0.1)
    parser.add_argument("--latent_weight", type=float, default=0.1)
    parser.add_argument("--sparsity_weight", type=float, default=1e-3)
    parser.add_argument("--k_stability_weight", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = load_compact_dataset(args.dataset)
    train_indices = dataset.indices_for_split("train")
    val_indices = dataset.indices_for_split("val")
    if val_indices.size == 0:
        val_indices = train_indices

    stats = compute_normalization(dataset, train_indices)
    observations = normalize_observations(dataset.observations, stats)
    actions = normalize_actions(dataset.actions, stats)
    activation = resolve_controlled_training_activation(
        args.encoder_kind,
        args.activation,
        args.sparsity_weight,
    )

    model_cfg = ControlledLISTAConfig(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        encoder_kind=args.encoder_kind,
        lista_loops=args.lista_loops,
        lista_alpha=args.lista_alpha,
        activation=activation,
    )
    weights = LossWeights(
        prediction=args.prediction_weight,
        reconstruction=args.reconstruction_weight,
        latent=args.latent_weight,
        sparsity=args.sparsity_weight,
        k_stability=args.k_stability_weight,
    )
    model = ControlledLISTAKoopman(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rng = np.random.default_rng(args.seed)
    train_sampler = WindowSampler(
        observations,
        actions,
        dataset.valid,
        train_indices,
        sequence_length=args.sequence_length,
        rng=rng,
    )
    val_sampler = WindowSampler(
        observations,
        actions,
        dataset.valid,
        val_indices,
        sequence_length=args.sequence_length,
        rng=np.random.default_rng(args.seed + 1),
    )

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs/maniskill_insertion") / f"{args.encoder_kind}_seed{args.seed}_{timestamp}"
    else:
        run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload: Dict[str, object] = {
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "device": str(device),
        "model_config": asdict(model_cfg),
        "loss_weights": asdict(weights),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "sequence_length": int(args.sequence_length),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_episode_count": int(train_indices.size),
        "val_episode_count": int(val_indices.size),
        "labels_used_for_training": False,
    }
    write_json(run_dir / "config.json", config_payload)

    history_path = run_dir / "metrics_history.jsonl"
    best_val = float("inf")
    best_metrics: Dict[str, float] = {}
    for step in range(int(args.num_steps)):
        x_seq, action_seq = train_sampler.sample(args.batch_size, device=device)
        metrics = train_step(model, optimizer, x_seq, action_seq, weights)
        metrics["step"] = float(step)
        if step % int(args.eval_every) == 0 or step == int(args.num_steps) - 1:
            val_mse = validation_rollout_mse(
                model,
                val_sampler,
                batch_size=max(1, min(args.batch_size, 128)),
                device=device,
            )
            metrics["val_rollout_mse"] = val_mse
            if val_mse < best_val:
                best_val = val_mse
                best_metrics = dict(metrics)
                save_checkpoint(
                    run_dir / "checkpoint.pt",
                    model=model,
                    optimizer=optimizer,
                    model_config=model_cfg,
                    loss_weights=weights,
                    normalization=stats,
                    step=step,
                    metrics=metrics,
                    metadata={
                        "dataset": str(args.dataset),
                        "labels_used_for_training": False,
                        "checkpoint_kind": "best_val_rollout_mse",
                    },
                )
            print(
                f"step={step} loss={metrics['loss']:.6f} "
                f"pred={metrics['prediction_loss']:.6f} val_mse={val_mse:.6f}",
                flush=True,
            )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    save_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        model_config=model_cfg,
        loss_weights=weights,
        normalization=stats,
        step=int(args.num_steps) - 1,
        metrics=metrics,
        metadata={
            "dataset": str(args.dataset),
            "labels_used_for_training": False,
            "checkpoint_kind": "last",
        },
    )
    write_json(
        run_dir / "final_metrics.json",
        {
            "best_val_rollout_mse": best_val,
            "best_metrics": best_metrics,
            "last_metrics": metrics,
            "checkpoint": str(run_dir / "checkpoint.pt"),
            "last_checkpoint": str(run_dir / "last.pt"),
        },
    )
    print(f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
