"""Build task tables for standalone paper baseline suites.

The generated TSV drives classical state-space Koopman baselines and
unsupervised mixture/local-linear dynamics baselines. These runs do not depend
on trained sparse-KAE checkpoints and do not use basin labels for fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from experiments.neurips_2026.protocol import (
    CLASSICAL_BASELINE_METHOD_IDS,
    CONTROLLED_PAPER_PROTOCOL,
    LOCAL_LINEAR_BASELINE_METHOD_IDS,
    STANDALONE_BASELINE_SEEDS,
)

DEFAULT_SYSTEMS: Sequence[str] = CONTROLLED_PAPER_PROTOCOL.system_keys
DEFAULT_SEEDS: Sequence[int] = STANDALONE_BASELINE_SEEDS
CLASSICAL_METHODS: Sequence[str] = CLASSICAL_BASELINE_METHOD_IDS
MIXTURE_METHODS: Sequence[str] = LOCAL_LINEAR_BASELINE_METHOD_IDS
BASELINE_FAMILIES: Sequence[str] = (
    "classical_koopman",
    "mixture_local_linear",
)


def _parse_csv_strings(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str | None) -> List[int]:
    return [int(item) for item in _parse_csv_strings(raw)]


def _write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    systems = _parse_csv_strings(args.systems) or list(DEFAULT_SYSTEMS)
    seeds = _parse_csv_ints(args.seeds) or list(DEFAULT_SEEDS)
    baseline_families = _parse_csv_strings(args.baseline_families) or [
        "classical_koopman",
        "mixture_local_linear",
    ]

    rows: List[Dict[str, object]] = []
    task_id = 0
    for family in baseline_families:
        if family not in set(BASELINE_FAMILIES):
            raise ValueError(
                f"Unknown baseline family '{family}'. Expected one of {', '.join(BASELINE_FAMILIES)}."
            )
        if family == "classical_koopman":
            methods = CLASSICAL_METHODS
        else:
            methods = MIXTURE_METHODS
        for system in systems:
            for seed in seeds:
                rows.append(
                    {
                        "task_id": task_id,
                        "baseline_family": family,
                        "system": system,
                        "seed": seed,
                        "methods": ",".join(methods),
                        "horizons": args.horizons,
                        "num_trajectories": args.num_trajectories,
                        "trajectory_length": args.trajectory_length,
                        "train_fraction": args.train_fraction,
                        "ridge_lambda": args.ridge_lambda,
                        "edmd_degree": args.edmd_degree,
                        "kernel_centers": args.kernel_centers,
                        "kernel_gamma": args.kernel_gamma,
                        "max_train_pairs": args.max_train_pairs,
                        "num_components": args.num_components,
                        "component_mode": args.component_mode,
                        "env_dt": args.env_dt,
                        "dysts_dt_multiplier": args.dysts_dt_multiplier,
                        "dysts_standardize": int(args.dysts_standardize),
                        "config_name": args.config_name,
                        "torch_threads": args.torch_threads,
                    }
                )
                task_id += 1
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_manifest_json", default=None)
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument(
        "--baseline_families",
        default="classical_koopman,mixture_local_linear",
        help="Comma-separated subset of classical_koopman,mixture_local_linear.",
    )
    parser.add_argument("--horizons", default="100,500,1000")
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=1000)
    parser.add_argument("--train_fraction", type=float, default=0.6)
    parser.add_argument("--ridge_lambda", type=float, default=1e-6)
    parser.add_argument("--edmd_degree", type=int, default=3)
    parser.add_argument("--kernel_centers", type=int, default=128)
    parser.add_argument("--kernel_gamma", type=float, default=0.0)
    parser.add_argument("--max_train_pairs", type=int, default=0)
    parser.add_argument("--num_components", type=int, default=4)
    parser.add_argument(
        "--component_mode",
        default="fixed",
        choices=["fixed", "known_basin_count"],
        help="known_basin_count is an evaluation-diagnostic upper-bound component count, not basin-supervised fitting.",
    )
    parser.add_argument("--env_dt", type=float, default=0.0)
    parser.add_argument(
        "--dysts_dt_multiplier",
        type=float,
        default=0.0,
        help="If >0 for dysts:* systems, use multiplier times the intrinsic Dysts dt.",
    )
    parser.add_argument(
        "--dysts_standardize",
        type=int,
        default=0,
        help="Set to 1 to evaluate dysts:* systems in standardized coordinates.",
    )
    parser.add_argument("--config_name", default="default")
    parser.add_argument("--torch_threads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)
    if args.output_manifest_json:
        Path(args.output_manifest_json).write_text(
            json.dumps(
                {
                    "num_tasks": len(rows),
                    "systems": _parse_csv_strings(args.systems),
                    "seeds": _parse_csv_ints(args.seeds),
                    "baseline_families": _parse_csv_strings(args.baseline_families),
                    "classical_methods": list(CLASSICAL_METHODS),
                    "mixture_methods": list(MIXTURE_METHODS),
                    "config": vars(args),
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Wrote {len(rows)} paper baseline tasks to {output_tsv}")


if __name__ == "__main__":
    main()
