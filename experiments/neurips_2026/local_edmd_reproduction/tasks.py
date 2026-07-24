"""Build the immutable 75-task reproduction roster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Sequence

from experiments.neurips_2026.local_edmd_reproduction.contract import (
    BENCHMARKS,
    CONFIG_NAME,
    EDMD_DEGREE,
    ENV_DT,
    KERNEL_CENTERS,
    KERNEL_GAMMA,
    MAX_ABS_STATE_FOR_FIT,
    MAX_TRAIN_PAIRS,
    METHOD_ID,
    MIN_COMPONENT_TRANSITIONS,
    NUM_COMPONENTS_GRID,
    PROTOCOL_ID,
    RIDGE_LAMBDA,
    SEEDS,
    TORCH_THREADS,
    VALIDATION_FRACTION,
    expected_task_count,
)


FIELDNAMES = (
    "task_id",
    "benchmark",
    "system",
    "seed",
    "method",
    "horizons",
    "num_trajectories",
    "trajectory_length",
    "train_fraction",
    "ridge_lambda",
    "edmd_degree",
    "kernel_centers",
    "kernel_gamma",
    "max_train_pairs",
    "num_components_grid",
    "validation_fraction",
    "selection_horizons",
    "min_component_transitions",
    "max_abs_state_for_fit",
    "env_dt",
    "dysts_dt_multiplier",
    "dysts_standardize",
    "config_name",
    "torch_threads",
)


def build_rows() -> list[dict[str, object]]:
    """Construct tasks from the current frozen paper rosters."""

    rows: list[dict[str, object]] = []
    for benchmark, spec in BENCHMARKS.items():
        horizons = ",".join(map(str, spec.horizons))
        for system in spec.systems:
            for seed in SEEDS:
                rows.append(
                    {
                        "task_id": len(rows),
                        "benchmark": benchmark,
                        "system": system,
                        "seed": seed,
                        "method": METHOD_ID,
                        "horizons": horizons,
                        "num_trajectories": spec.num_trajectories,
                        "trajectory_length": spec.trajectory_length,
                        "train_fraction": spec.train_fraction,
                        "ridge_lambda": RIDGE_LAMBDA,
                        "edmd_degree": EDMD_DEGREE,
                        "kernel_centers": KERNEL_CENTERS,
                        "kernel_gamma": KERNEL_GAMMA,
                        "max_train_pairs": MAX_TRAIN_PAIRS,
                        "num_components_grid": ",".join(
                            map(str, NUM_COMPONENTS_GRID)
                        ),
                        "validation_fraction": VALIDATION_FRACTION,
                        "selection_horizons": horizons,
                        "min_component_transitions": MIN_COMPONENT_TRANSITIONS,
                        "max_abs_state_for_fit": MAX_ABS_STATE_FOR_FIT,
                        "env_dt": ENV_DT,
                        "dysts_dt_multiplier": spec.dysts_dt_multiplier,
                        "dysts_standardize": spec.dysts_standardize,
                        "config_name": CONFIG_NAME,
                        "torch_threads": TORCH_THREADS,
                    }
                )
    if len(rows) != expected_task_count():
        raise RuntimeError("Task roster size drifted")
    return rows


def _tsv_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def build_outputs(output_root: Path) -> dict[Path, bytes]:
    """Return deterministic task and manifest payloads."""

    rows = build_rows()
    task_payload = _tsv_bytes(rows)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "description": "Frozen known-outcome local polynomial EDMD reproduction roster.",
        "label_policy": (
            "No basin labels or counts enter fitting, selection, refitting, "
            "routing, or forecasting."
        ),
        "method": METHOD_ID,
        "num_tasks": len(rows),
        "task_tsv": {
            "name": "tasks.tsv",
            "rows": len(rows),
            "sha256": hashlib.sha256(task_payload).hexdigest(),
        },
        "benchmarks": {
            name: {
                "paper_protocol_id": spec.protocol_id,
                "systems": list(spec.systems),
                "seeds": list(SEEDS),
                "horizons": list(spec.horizons),
                "num_trajectories": spec.num_trajectories,
                "trajectory_length": spec.trajectory_length,
                "train_fraction": spec.train_fraction,
                "dysts_dt_multiplier": spec.dysts_dt_multiplier,
                "dysts_standardize": bool(spec.dysts_standardize),
            }
            for name, spec in BENCHMARKS.items()
        },
        "route_count_grid": list(NUM_COMPONENTS_GRID),
        "validation_fraction": VALIDATION_FRACTION,
        "selection_horizons": "same_as_report_horizons",
        "ridge_lambda": RIDGE_LAMBDA,
        "edmd_degree": EDMD_DEGREE,
        "minimum_component_transitions": MIN_COMPONENT_TRANSITIONS,
        "maximum_absolute_fit_state": MAX_ABS_STATE_FOR_FIT,
    }
    return {
        output_root / "tasks.tsv": task_payload,
        output_root / "task_manifest.json": (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }


def write_or_check(outputs: dict[Path, bytes], *, check: bool) -> None:
    stale = [
        path
        for path, payload in outputs.items()
        if not path.is_file() or path.read_bytes() != payload
    ]
    if check:
        if stale:
            raise RuntimeError("Stale reproduction inputs: " + ", ".join(map(str, stale)))
        print(f"Verified {len(outputs)} frozen reproduction inputs")
        return
    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_or_check(build_outputs(args.output_root), check=args.check)


if __name__ == "__main__":
    main()

