"""Build TSV tasks for state-observation control world-model experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from skae.benchmarks.control_world_model import DMC_TASKS, write_json
from tools.train_control_world_model import VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_tsv", type=Path, required=True)
    parser.add_argument("--output_manifest_json", type=Path, required=True)
    parser.add_argument("--base_out", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--tasks_csv", default="cartpole_swingup,finger_spin,cheetah_run,walker_walk")
    parser.add_argument(
        "--variants_csv",
        default="sparse_additive,dense_additive,sparse_bilinear,dense_bilinear,mlp",
    )
    parser.add_argument("--seeds_csv", default="0,1,2")
    parser.add_argument("--data_fractions_csv", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--dataset_seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=256)
    parser.add_argument("--episode_length", type=int, default=250)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--eval_horizons", default="1,5,10,20,50")
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--planning_candidates", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = parse_csv(args.tasks_csv)
    variants = parse_csv(args.variants_csv)
    seeds = [int(value) for value in parse_csv(args.seeds_csv)]
    data_fractions = [float(value) for value in parse_csv(args.data_fractions_csv)]
    if not tasks:
        raise ValueError("tasks_csv must contain at least one task")
    if not variants:
        raise ValueError("variants_csv must contain at least one variant")
    if not seeds:
        raise ValueError("seeds_csv must contain at least one seed")
    if not data_fractions:
        raise ValueError("data_fractions_csv must contain at least one fraction")

    for task in tasks:
        if task not in DMC_TASKS:
            raise ValueError(f"Unknown task {task!r}; expected one of {sorted(DMC_TASKS)}")
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")

    rows: List[Dict[str, object]] = []
    task_id = 0
    train_tag = tagify_float(args.train_fraction)
    val_tag = tagify_float(args.val_fraction)
    for task in tasks:
        dataset_path = (
            args.dataset_root
            / (
                f"{task}_seed{args.dataset_seed}_e{args.num_episodes}"
                f"_h{args.episode_length}_tr{train_tag}_val{val_tag}.npz"
            )
        )
        dataset_summary = dataset_path.with_suffix(".summary.json")
        for variant in variants:
            for data_fraction in data_fractions:
                fraction_tag = tagify_float(data_fraction)
                for seed in seeds:
                    run_dir = args.base_out / task / variant / f"frac_{fraction_tag}" / f"seed_{seed}"
                    rows.append(
                        {
                            "task_id": task_id,
                            "task": task,
                            "variant": variant,
                            "seed": seed,
                            "data_fraction": data_fraction,
                            "dataset": dataset_path,
                            "dataset_summary": dataset_summary,
                            "run_dir": run_dir,
                            "dataset_seed": args.dataset_seed,
                            "num_episodes": args.num_episodes,
                            "episode_length": args.episode_length,
                            "train_fraction": args.train_fraction,
                            "val_fraction": args.val_fraction,
                            "num_steps": args.num_steps,
                            "batch_size": args.batch_size,
                            "sequence_length": args.sequence_length,
                            "eval_horizons": args.eval_horizons,
                            "z_dim": args.z_dim,
                            "hidden_dim": args.hidden_dim,
                            "lr": args.lr,
                            "weight_decay": args.weight_decay,
                            "eval_every": args.eval_every,
                            "planning_candidates": args.planning_candidates,
                        }
                    )
                    task_id += 1

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "task_count": len(rows),
        "tasks": tasks,
        "variants": variants,
        "seeds": seeds,
        "data_fractions": data_fractions,
        "dataset_seed": args.dataset_seed,
        "num_episodes": args.num_episodes,
        "episode_length": args.episode_length,
        "base_out": str(args.base_out),
        "dataset_root": str(args.dataset_root),
    }
    write_json(args.output_manifest_json, manifest)
    print(f"Wrote {len(rows)} tasks to {args.output_tsv}", flush=True)


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def tagify_float(value: float) -> str:
    raw = f"{float(value):g}"
    return raw.replace("-", "m").replace(".", "p").replace("+", "p")


if __name__ == "__main__":
    main()
