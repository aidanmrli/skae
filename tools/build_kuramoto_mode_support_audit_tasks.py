#!/usr/bin/env python3
"""Build task tables for the Kuramoto mode-support audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


MODEL_SPECS = {
    "generic_sparse": {
        "family": "generic",
        "root_label": "generic_sparse",
    },
    "lista_dense": {
        "family": "dense_lista",
        "root_label": "lista_dense",
    },
    "lista_blockdiag": {
        "family": "blockdiag_lista",
        "root_label": "lista_blockdiag",
    },
}


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_csv_strings(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _tagify(raw: str) -> str:
    return raw.replace("-", "m").replace(".", "p")


def _resolve_checkpoint(source_root: Path, model_variant: str, env_dt: float, seed: int) -> str:
    dt_tag = _tagify(f"{env_dt:g}")
    matches = sorted(
        source_root.glob(
            f"{model_variant}/kuramoto/dt_{dt_tag}/seed_{seed}/*/checkpoint.pt"
        )
    )
    if not matches:
        raise FileNotFoundError(
            "No checkpoint found for "
            f"variant={model_variant}, seed={seed}, env_dt={env_dt:g} under {source_root}"
        )
    return str(matches[-1])


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    model_variants = _parse_csv_strings(args.model_variants_csv)
    seeds = _parse_csv_ints(args.seeds_csv)
    sampling_strategies = _parse_csv_strings(args.sampling_strategies_csv)
    source_root = Path(args.source_root)
    scratch_root = Path(args.scratch_root)

    rows: List[Dict[str, object]] = []
    task_id = 0
    for model_variant in model_variants:
        if model_variant not in MODEL_SPECS:
            raise KeyError(f"Unknown model variant '{model_variant}'")
        spec = MODEL_SPECS[model_variant]
        for seed in seeds:
            checkpoint = _resolve_checkpoint(source_root, model_variant, args.env_dt, seed)
            for sampling_strategy in sampling_strategies:
                if sampling_strategy == "random":
                    num_trajectories = args.random_num_trajectories
                    trajectories_per_basin = ""
                    target_raw_labels_csv = ""
                    max_attempts = ""
                elif sampling_strategy == "balanced":
                    num_trajectories = ""
                    trajectories_per_basin = args.balanced_trajectories_per_basin
                    target_raw_labels_csv = args.balanced_target_raw_labels_csv
                    max_attempts = args.max_attempts
                else:
                    raise ValueError(
                        f"Unknown sampling strategy '{sampling_strategy}'."
                    )

                output_dir = scratch_root / "eval" / sampling_strategy / spec["family"] / f"seed_{seed}"
                rows.append(
                    {
                        "task_id": task_id,
                        "phase_label": args.phase_label,
                        "system": "kuramoto",
                        "family": spec["family"],
                        "root_label": spec["root_label"],
                        "model_variant": model_variant,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "output_dir": str(output_dir),
                        "sampling_strategy": sampling_strategy,
                        "num_trajectories": num_trajectories,
                        "trajectories_per_basin": trajectories_per_basin,
                        "target_raw_labels_csv": target_raw_labels_csv,
                        "trajectory_length": args.trajectory_length,
                        "long_rollout_steps": args.long_rollout_steps,
                        "support_threshold": args.support_threshold,
                        "support_modes_csv": args.support_modes_csv,
                        "threshold_sweep_modes_csv": args.threshold_sweep_modes_csv,
                        "thresholds_csv": args.thresholds_csv,
                        "max_attempts": max_attempts,
                        "device": args.device,
                    }
                )
                task_id += 1
    return rows


def _write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_label", default="kuramoto_mode_support_audit")
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_manifest_json", default=None)
    parser.add_argument(
        "--source_root",
        default="/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308/kuramoto_dt00625_200k",
    )
    parser.add_argument("--scratch_root", required=True)
    parser.add_argument("--model_variants_csv", default="generic_sparse,lista_dense,lista_blockdiag")
    parser.add_argument("--seeds_csv", default="0,1,2,3,4")
    parser.add_argument("--sampling_strategies_csv", default="random,balanced")
    parser.add_argument("--env_dt", type=float, default=0.00625)
    parser.add_argument("--random_num_trajectories", type=int, default=256)
    parser.add_argument("--balanced_trajectories_per_basin", type=int, default=16)
    parser.add_argument("--balanced_target_raw_labels_csv", default="-2,-1,0,1,2")
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--long_rollout_steps", type=int, default=5000)
    parser.add_argument("--support_threshold", type=float, default=1e-3)
    parser.add_argument("--support_modes_csv", default="mean,majority,modal")
    parser.add_argument("--threshold_sweep_modes_csv", default="mean,modal")
    parser.add_argument("--thresholds_csv", default="1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1")
    parser.add_argument("--max_attempts", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        payload = {
            "phase_label": args.phase_label,
            "source_root": args.source_root,
            "scratch_root": args.scratch_root,
            "model_variants": _parse_csv_strings(args.model_variants_csv),
            "seeds": _parse_csv_ints(args.seeds_csv),
            "sampling_strategies": _parse_csv_strings(args.sampling_strategies_csv),
            "env_dt": args.env_dt,
            "random_num_trajectories": args.random_num_trajectories,
            "balanced_trajectories_per_basin": args.balanced_trajectories_per_basin,
            "balanced_target_raw_labels": _parse_csv_ints(args.balanced_target_raw_labels_csv),
            "support_modes": _parse_csv_strings(args.support_modes_csv),
            "threshold_sweep_modes": _parse_csv_strings(args.threshold_sweep_modes_csv),
            "thresholds": [float(value) for value in args.thresholds_csv.split(",") if value.strip()],
            "max_attempts": args.max_attempts,
            "task_count": len(rows),
        }
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {len(rows)} Kuramoto mode-support audit tasks to {output_tsv}")


if __name__ == "__main__":
    main()
