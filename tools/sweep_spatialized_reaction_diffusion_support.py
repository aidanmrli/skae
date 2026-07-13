#!/usr/bin/env python3
"""Sweep support-family thresholds for a spatialized PDE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from skae.benchmarks.spatialized_reaction_diffusion import flatten_fields, load_dataset, split_fields
from tools.evaluate_spatialized_reaction_diffusion import load_model, resolve_device, support_alignment_metrics


def _parse_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--support_thresholds_csv", default="1e-4,1e-3,3e-3,1e-2,3e-2,5e-2,1e-1")
    parser.add_argument("--family_jaccards_csv", default="0.7,0.8,0.9,1.0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--deep_threshold", type=float, default=0.7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    val_fields_flat = flatten_fields(split_fields(dataset, "val")).float()
    test_fields_flat = flatten_fields(split_fields(dataset, "test")).float()
    observation_size = int(test_fields_flat.shape[-1])
    test_indices = dataset["split_indices"]["test"]
    test_global_labels = dataset["global_basin_labels"][test_indices]
    test_majority_fractions = dataset["majority_fractions"][test_indices]
    device = resolve_device(args.device)
    model, cfg_or_info, checkpoint = load_model(Path(args.checkpoint), observation_size, device)

    rows = []
    for threshold in _parse_floats(args.support_thresholds_csv):
        for family_jaccard in _parse_floats(args.family_jaccards_csv):
            support = support_alignment_metrics(
                model,
                val_fields_flat,
                test_fields_flat,
                test_global_labels=test_global_labels,
                test_majority_fractions=test_majority_fractions,
                threshold=float(threshold),
                family_jaccard=float(family_jaccard),
                max_validation_reps=1024,
                batch_size=int(args.batch_size),
                deep_threshold=float(args.deep_threshold),
                device=device,
            )
            rows.append(
                {
                    "support_threshold": float(threshold),
                    "family_jaccard": float(family_jaccard),
                    "validation_representative_count": support["validation_representative_count"],
                    "validation_unique_supports_before_cap": support["validation_unique_supports_before_cap"],
                    "validation_support_size_mean": support["validation_support_size_mean"],
                    "validation_support_size_median": support["validation_support_size_median"],
                    "test_support_size_mean": support["test_support_size_mean"],
                    "test_support_size_median": support["test_support_size_median"],
                    "all_test": support["all_test"],
                    "deep_test": support["deep_test"],
                }
            )

    results = {
        "status": "completed",
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "model_family": checkpoint.get("model_family", "skae_flat"),
        "model_config": cfg_or_info if isinstance(cfg_or_info, dict) else cfg_or_info.to_dict(),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
