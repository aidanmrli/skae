#!/usr/bin/env python3
"""Pre-generate and validate spatialized multibasin datasets before GPU jobs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from skae.benchmarks.spatialized_reaction_diffusion import (
    SpatialReactionDiffusionConfig,
    generate_dataset,
    load_dataset,
    save_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_tsv", required=True)
    parser.add_argument("--min_labels_per_split", type=int, default=2)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def _read_tasks(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _unique_dataset_tasks(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for row in rows:
        key = row["dataset_path"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _cfg_from_row(row: Dict[str, str]) -> SpatialReactionDiffusionConfig:
    return SpatialReactionDiffusionConfig(
        source_system=row["source_system"],
        grid_size=int(row["grid_size"]),
        diffusion=float(row["diffusion"]),
        rk4_dt=float(row["rk4_dt"]),
        substeps_per_observation=int(row["substeps_per_observation"]),
        trajectory_length=int(row["trajectory_length"]),
        label_extra_observations=int(row["label_extra_observations"]),
        train_trajectories=int(row["train_trajectories"]),
        val_trajectories=int(row["val_trajectories"]),
        test_trajectories=int(row["test_trajectories"]),
        seed=int(row["seed"]),
        laplacian_scaling=row["laplacian_scaling"],
    )


def _label_counts(bundle: Dict[str, object], split_name: str) -> Dict[str, int]:
    indices = bundle["split_indices"][split_name]  # type: ignore[index]
    labels = bundle["global_basin_labels"][indices]  # type: ignore[index]
    unique, counts = torch.unique(labels, return_counts=True)
    return {str(int(label.item())): int(count.item()) for label, count in zip(unique, counts)}


def _validate_bundle(bundle: Dict[str, object], *, min_labels_per_split: int) -> Tuple[bool, Dict[str, object]]:
    split_counts = {
        split: _label_counts(bundle, split)
        for split in ("train", "val", "test")
    }
    all_labels = bundle["global_basin_labels"]  # type: ignore[index]
    all_unique = torch.unique(all_labels)
    split_unique_counts = {split: len(counts) for split, counts in split_counts.items()}
    valid = int(all_unique.numel()) >= int(min_labels_per_split) and all(
        count >= int(min_labels_per_split) for count in split_unique_counts.values()
    )
    return valid, {
        "source_system": bundle["metadata"].get("source_system_name"),  # type: ignore[index]
        "grid_size": int(bundle["metadata"].get("grid_size")),  # type: ignore[index]
        "seed": int(bundle["metadata"].get("seed")),  # type: ignore[index]
        "all_label_count": int(all_unique.numel()),
        "split_label_counts": split_counts,
        "split_unique_counts": split_unique_counts,
        "majority_fraction_mean": float(bundle["majority_fractions"].float().mean().item()),  # type: ignore[index]
        "majority_fraction_min": float(bundle["majority_fractions"].float().min().item()),  # type: ignore[index]
        "invalid_value_count_total": int(bundle["invalid_value_counts"].sum().item()),  # type: ignore[index]
        "clipped_value_count_total": int(bundle["clipped_value_counts"].sum().item()),  # type: ignore[index]
    }


def main() -> None:
    args = parse_args()
    rows = _unique_dataset_tasks(_read_tasks(Path(args.task_tsv)))
    reports: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for row in rows:
        dataset_path = Path(row["dataset_path"])
        if dataset_path.exists():
            bundle = load_dataset(dataset_path)
            generated = False
        else:
            cfg = _cfg_from_row(row)
            bundle = generate_dataset(cfg)
            save_dataset(bundle, dataset_path)
            generated = True
        valid, report = _validate_bundle(bundle, min_labels_per_split=int(args.min_labels_per_split))
        item = {
            "dataset_path": str(dataset_path),
            "generated": generated,
            "valid_multibasin": valid,
            **report,
        }
        reports.append(item)
        if not valid:
            failures.append(item)
        summary_path = dataset_path.with_suffix(dataset_path.suffix + ".preflight.json")
        summary_path.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")
        print(json.dumps(item, sort_keys=True), flush=True)

    output = {
        "status": "completed" if not failures else "failed",
        "task_tsv": str(args.task_tsv),
        "dataset_count": len(reports),
        "failure_count": len(failures),
        "min_labels_per_split": int(args.min_labels_per_split),
        "datasets": reports,
        "failures": failures,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    if failures:
        raise SystemExit(
            "Preflight failed: at least one dataset did not contain multiple basin labels "
            "in every split. See " + str(args.output_json)
        )


if __name__ == "__main__":
    main()
