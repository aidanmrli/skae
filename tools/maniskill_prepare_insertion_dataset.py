"""Build a compact state-only ManiSkill insertion dataset.

Expected input is a ManiSkill replayed HDF5 file, usually produced from the raw
PegInsertionSide-v1 demonstrations with:

    uv run python -m mani_skill.trajectory.replay_trajectory \
      --traj-path demos/rigid_body/PegInsertionSide-v1/trajectory.h5 \
      -c pd_ee_delta_pose -o state --use-env-states --record-rewards --save-traj

The compact output keeps labels for evaluation only. Training scripts read only
state/action arrays, transition masks, and trajectory splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skae.benchmarks.maniskill_insertion_dataset import build_compact_dataset_from_h5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj_path", type=Path, required=True, help="Replayed ManiSkill .h5 trajectory path")
    parser.add_argument("--output", type=Path, required=True, help="Output compact .npz path")
    parser.add_argument(
        "--obs_key",
        default="obs",
        help="HDF5 key to flatten. Defaults to obs; falls back to env_states if absent.",
    )
    parser.add_argument(
        "--no_append_prev_action",
        action="store_true",
        help="Do not append previous action to each compact state vector.",
    )
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional cap for smoke datasets")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional per-episode transition cap")
    parser.add_argument("--min_steps", type=int, default=2, help="Minimum usable transitions per episode")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to <output>.summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_compact_dataset_from_h5(
        args.traj_path,
        args.output,
        obs_key=args.obs_key,
        append_prev_action=not args.no_append_prev_action,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    summary_path = args.summary
    if summary_path is None:
        summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    split_counts = {
        split_name: int((dataset.split == split_name).sum())
        for split_name in sorted(set(dataset.split.astype(str).tolist()))
    }
    summary = {
        "output": str(args.output),
        "source_traj_path": str(args.traj_path),
        "num_episodes": dataset.num_episodes,
        "max_transitions": dataset.max_transitions,
        "obs_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "split_counts": split_counts,
        "outcome_available": bool((dataset.outcome >= 0).any()),
        "contact_phase_available": dataset.contact_phase is not None,
        "metadata": dataset.metadata,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
