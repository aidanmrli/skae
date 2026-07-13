"""Train action-conditioned Koopman world models on compact control datasets."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from skae.benchmarks.control_world_model import (
    ControlWorldModel,
    ControlWorldModelConfig,
    LossWeights,
    WindowSampler,
    compute_normalization,
    evaluate_world_model,
    load_control_dataset,
    load_model_from_checkpoint,
    normalize_actions,
    normalize_observations,
    normalize_rewards,
    resolve_world_model_activation,
    save_checkpoint,
    select_fraction,
    train_step,
    write_json,
)


VARIANTS: Dict[str, Tuple[str, bool]] = {
    "sparse_additive": ("additive", True),
    "dense_additive": ("additive", False),
    "sparse_bilinear": ("bilinear", True),
    "dense_bilinear": ("bilinear", False),
    "mlp": ("mlp", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Compact control .npz dataset")
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--variant", default="sparse_additive", choices=sorted(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--activation", default="auto", choices=("auto", "relu", "tanh", "gelu"))
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--eval_horizons", default="1,5,10,20,50")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--prediction_weight", type=float, default=1.0)
    parser.add_argument("--reconstruction_weight", type=float, default=0.1)
    parser.add_argument("--latent_weight", type=float, default=0.1)
    parser.add_argument("--reward_weight", type=float, default=1.0)
    parser.add_argument("--continuation_weight", type=float, default=0.1)
    parser.add_argument("--k_sparsity_weight", type=float, default=1e-4)
    parser.add_argument("--k_stability_weight", type=float, default=1e-4)
    parser.add_argument("--density_threshold", type=float, default=1e-4)
    parser.add_argument("--planning_candidates", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = load_control_dataset(args.dataset)
    train_indices = dataset.indices_for_split("train")
    val_indices = dataset.indices_for_split("val")
    test_indices = dataset.indices_for_split("test")
    if val_indices.size == 0:
        val_indices = train_indices
    if test_indices.size == 0:
        test_indices = val_indices

    train_indices = select_fraction(train_indices, args.data_fraction, seed=args.seed)
    stats = compute_normalization(dataset, train_indices)
    observations = normalize_observations(dataset.observations, stats)
    actions = normalize_actions(dataset.actions, stats)
    rewards = normalize_rewards(dataset.rewards, stats)

    transition_kind, sparse_matrices = VARIANTS[args.variant]
    activation = resolve_world_model_activation(transition_kind, sparse_matrices, args.activation)
    model_cfg = ControlWorldModelConfig(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        transition_kind=transition_kind,
        sparse_matrices=sparse_matrices,
        activation=activation,
    )
    weights = LossWeights(
        prediction=args.prediction_weight,
        reconstruction=args.reconstruction_weight,
        latent=args.latent_weight,
        reward=args.reward_weight,
        continuation=args.continuation_weight,
        k_sparsity=args.k_sparsity_weight if sparse_matrices else 0.0,
        k_stability=args.k_stability_weight,
    )
    model = ControlWorldModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    eval_horizons = parse_int_csv(args.eval_horizons)
    max_eval_horizon = max(eval_horizons)
    train_sequence_length = int(args.sequence_length)
    eval_sequence_length = max(train_sequence_length, max_eval_horizon)

    train_sampler = WindowSampler(
        observations,
        actions,
        rewards,
        dataset.continuations,
        dataset.valid,
        train_indices,
        sequence_length=train_sequence_length,
        rng=np.random.default_rng(args.seed),
    )
    val_sampler = WindowSampler(
        observations,
        actions,
        rewards,
        dataset.continuations,
        dataset.valid,
        val_indices,
        sequence_length=eval_sequence_length,
        rng=np.random.default_rng(args.seed + 1),
    )
    test_sampler = WindowSampler(
        observations,
        actions,
        rewards,
        dataset.continuations,
        dataset.valid,
        test_indices,
        sequence_length=eval_sequence_length,
        rng=np.random.default_rng(args.seed + 2),
    )

    if args.run_dir is None:
        task = str(dataset.metadata.get("task", "control"))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path("runs/control_world_model") / task / args.variant / f"seed{args.seed}_{timestamp}"
    else:
        run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "dataset": str(args.dataset),
        "dataset_metadata": dataset.metadata,
        "variant": args.variant,
        "seed": int(args.seed),
        "device": str(device),
        "data_fraction": float(args.data_fraction),
        "model_config": asdict(model_cfg),
        "loss_weights": asdict(weights),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "sequence_length": int(args.sequence_length),
        "eval_horizons": eval_horizons,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "train_episode_count": int(train_indices.size),
        "val_episode_count": int(val_indices.size),
        "test_episode_count": int(test_indices.size),
        "labels_used_for_training": False,
    }
    write_json(run_dir / "config.json", config_payload)

    history_path = run_dir / "metrics_history.jsonl"
    best_val = float("inf")
    best_metrics: Dict[str, float] = {}
    last_metrics: Dict[str, float] = {}
    for step in range(int(args.num_steps)):
        x_seq, action_seq, reward_seq, continuation_seq = train_sampler.sample(args.batch_size, device=device)
        metrics = train_step(
            model,
            optimizer,
            x_seq,
            action_seq,
            reward_seq,
            continuation_seq,
            weights,
            density_threshold=args.density_threshold,
        )
        metrics["step"] = float(step)
        if step % int(args.eval_every) == 0 or step == int(args.num_steps) - 1:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            val_metrics = evaluate_world_model(
                model,
                val_sampler,
                batch_size=max(1, min(args.batch_size, 256)),
                device=device,
                horizons=eval_horizons,
                planning_candidates=args.planning_candidates,
                density_threshold=args.density_threshold,
            )
            metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
            selection_key = f"val/open_loop_mse_h{max_eval_horizon}"
            val_score = float(metrics[selection_key])
            if val_score < best_val:
                best_val = val_score
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
                        "variant": args.variant,
                        "checkpoint_kind": f"best_{selection_key}",
                        "labels_used_for_training": False,
                    },
                )
            print(
                f"step={step} loss={metrics['loss']:.6f} "
                f"pred={metrics['prediction_loss']:.6f} "
                f"{selection_key}={val_score:.6f}",
                flush=True,
            )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        last_metrics = metrics

    best_checkpoint_path = run_dir / "checkpoint.pt"
    eval_model = model
    if best_checkpoint_path.exists():
        eval_model, _, _ = load_model_from_checkpoint(best_checkpoint_path, device=device)
    test_metrics = evaluate_world_model(
        eval_model,
        test_sampler,
        batch_size=max(1, min(args.batch_size, 256)),
        device=device,
        horizons=eval_horizons,
        planning_candidates=args.planning_candidates,
        density_threshold=args.density_threshold,
    )
    save_checkpoint(
        run_dir / "last.pt",
        model=model,
        optimizer=optimizer,
        model_config=model_cfg,
        loss_weights=weights,
        normalization=stats,
        step=int(args.num_steps) - 1,
        metrics=last_metrics,
        metadata={
            "dataset": str(args.dataset),
            "variant": args.variant,
            "checkpoint_kind": "last",
            "labels_used_for_training": False,
        },
    )
    write_json(
        run_dir / "final_metrics.json",
        {
            "best_validation_score": best_val,
            "selection_horizon": max_eval_horizon,
            "best_metrics": best_metrics,
            "last_metrics": last_metrics,
            "test_metrics": test_metrics,
            "checkpoint": str(run_dir / "checkpoint.pt"),
            "last_checkpoint": str(run_dir / "last.pt"),
            "config": config_payload,
        },
    )
    print(f"run_dir={run_dir}", flush=True)


def parse_int_csv(value: str) -> Tuple[int, ...]:
    horizons = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not horizons:
        raise argparse.ArgumentTypeError("Expected at least one integer horizon")
    if min(horizons) <= 0:
        raise argparse.ArgumentTypeError("Horizons must be positive")
    return horizons


if __name__ == "__main__":
    main()
