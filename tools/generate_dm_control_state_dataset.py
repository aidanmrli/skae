"""Generate compact state-observation DeepMind Control Suite datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from skae.benchmarks.control_world_model import DMC_TASKS, generate_dm_control_dataset, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(DMC_TASKS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--num_episodes", type=int, default=256)
    parser.add_argument("--episode_length", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--policy", default="random", choices=("random",))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dm_control_dataset(
        task=args.task,
        output_path=args.output,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        seed=args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        policy=args.policy,
    )
    summary = {
        "dataset": str(args.output),
        "task": args.task,
        "num_episodes": dataset.num_episodes,
        "max_transitions": dataset.max_transitions,
        "obs_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "train_episodes": int(dataset.indices_for_split("train").size),
        "val_episodes": int(dataset.indices_for_split("val").size),
        "test_episodes": int(dataset.indices_for_split("test").size),
        "metadata": dataset.metadata,
    }
    if args.summary_json is not None:
        write_json(args.summary_json, summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
