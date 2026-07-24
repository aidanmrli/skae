#!/usr/bin/env python3
"""Read checkpoint step indices only; never inspect losses or model outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.neurips_2026.global_k_dense_specificity import _tagify
from experiments.neurips_2026.global_k_dense_zero_wd_tasks import load_card, sha256_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--base_out", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    card, card_hash = load_card(args.card)
    with args.task_tsv.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = int(card["training"]["expected_run_count"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} task rows, found {len(rows)}")

    progress = []
    for row in rows:
        parent = (
            args.base_out
            / row["phase"]
            / row["model_variant"]
            / row["system_slug"]
            / f"dt_{_tagify(row['env_dt'])}"
            / f"seed_{row['seed']}"
        )
        candidates = sorted(
            path for path in parent.glob("20*")
            if (path / "checkpoint.pt").is_file()
        )
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one checkpoint under {parent}, found {candidates}")
        checkpoint_path = candidates[0] / "checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        step = int(checkpoint["step"])
        num_steps = int(checkpoint["config"]["TRAIN"]["NUM_STEPS"])
        progress.append(
            {
                "task_id": int(row["task_id"]),
                "pack_id": int(row["task_id"]) // 15,
                "system_key": row["system_key"],
                "seed": int(row["seed"]),
                "checkpoint_step": step,
                "completed_training_steps_lower_bound": step + 1,
                "configured_num_steps": num_steps,
                "checkpoint_progress_lower_bound": (step + 1) / num_steps,
                "checkpoint_mtime_epoch": checkpoint_path.stat().st_mtime,
                "checkpoint_path": str(checkpoint_path),
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "non_outcome_checkpoint_progress_only",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "card_sha256": card_hash,
        "task_tsv_sha256": sha256_path(args.task_tsv),
        "rows": progress,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
