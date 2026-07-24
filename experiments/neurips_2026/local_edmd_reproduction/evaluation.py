"""Execute one frozen local-polynomial-EDMD reproduction task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from experiments.neurips_2026.baselines.classical import (
    _finite_mean,
    _finite_median,
    _generate_trajectories,
    _hist,
    _json_default,
    _make_env_for_system,
    _maybe_basin_labels,
    _split_trajectories,
    _stable_seed,
)
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
    RIDGE_LAMBDA,
    SEEDS,
    TORCH_THREADS,
    VALIDATION_FRACTION,
)
from experiments.neurips_2026.local_edmd_reproduction.model import (
    LocalEDMDModel,
    select_and_fit,
    split_fit_validation,
)
from experiments.neurips_2026.local_edmd_reproduction.source_lock import (
    verify_source_lock,
)


ROW_FIELDS = (
    "system",
    "seed",
    "method",
    "status",
    "skip_reason",
    "horizon",
    "endpoint_mse_mean",
    "endpoint_mse_median",
    "endpoint_mse_per_dim_mean",
    "cumulative_mse_mean",
    "cumulative_mse_median",
    "cumulative_mse_per_dim_mean",
    "finite_fraction",
    "selected_num_components",
    "validation_score",
    "num_components_grid",
    "selection_horizons",
    "validation_fraction",
    "fitted_component_count",
    "component_counts",
    "feature_method",
    "route_space",
    "feature_dim",
    "train_transitions",
    "env_dt",
    "state_dim",
    "train_trajectories",
    "validation_trajectories",
    "test_trajectories",
    "num_trajectories",
    "trajectory_length",
    "train_fraction",
    "ridge_lambda",
    "edmd_degree",
    "kernel_centers_requested",
    "kernel_centers_used",
    "kernel_gamma",
    "min_component_transitions",
    "max_abs_state_for_fit",
    "test_initial_basin_hist",
    "test_final_basin_hist",
    "candidate_scores_json",
)


def evaluate_rollout(
    model: LocalEDMDModel,
    trajectories: np.ndarray,
    horizons: Sequence[int],
) -> Dict[int, Dict[str, Optional[float]]]:
    """Apply the historical ordinary-through-horizon metric."""

    horizon_list = sorted({int(horizon) for horizon in horizons if int(horizon) > 0})
    horizon_list = [
        horizon
        for horizon in horizon_list
        if horizon <= trajectories.shape[1] - 1
    ]
    if not horizon_list:
        raise ValueError("No requested horizons fit the trajectories")
    max_horizon = max(horizon_list)
    predictions = model.rollout(trajectories[:, 0, :], horizon=max_horizon)
    targets = trajectories[:, 1 : max_horizon + 1, :]
    with np.errstate(over="ignore", invalid="ignore"):
        step_squared_error = ((predictions - targets) ** 2).sum(axis=2)
    state_dim = trajectories.shape[-1]
    metrics: Dict[int, Dict[str, Optional[float]]] = {}
    for horizon in horizon_list:
        endpoint = step_squared_error[:, horizon - 1]
        cumulative = step_squared_error[:, :horizon].mean(axis=1)
        finite = np.isfinite(cumulative)
        metrics[horizon] = {
            "endpoint_mse_mean": _finite_mean(endpoint),
            "endpoint_mse_median": _finite_median(endpoint),
            "endpoint_mse_per_dim_mean": _finite_mean(endpoint / state_dim),
            "cumulative_mse_mean": _finite_mean(cumulative),
            "cumulative_mse_median": _finite_median(cumulative),
            "cumulative_mse_per_dim_mean": _finite_mean(cumulative / state_dim),
            "finite_fraction": float(finite.mean()) if finite.size else None,
        }
    return metrics


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def read_task(path: Path, task_index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not 0 <= int(task_index) < len(rows):
        raise IndexError(f"Task index {task_index} outside 0..{len(rows) - 1}")
    return rows[int(task_index)]


def validate_task(row: dict[str, str], task_index: int) -> None:
    """Fail if a runtime task departs from the frozen scientific contract."""

    if any("label" in key or "basin" in key for key in row):
        raise ValueError("Training task must not contain labels or basin counts")
    benchmark = row.get("benchmark", "")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark!r}")
    spec = BENCHMARKS[benchmark]
    exact = {
        "task_id": str(task_index),
        "method": METHOD_ID,
        "horizons": ",".join(map(str, spec.horizons)),
        "num_trajectories": str(spec.num_trajectories),
        "trajectory_length": str(spec.trajectory_length),
        "ridge_lambda": str(RIDGE_LAMBDA),
        "edmd_degree": str(EDMD_DEGREE),
        "kernel_centers": str(KERNEL_CENTERS),
        "kernel_gamma": str(KERNEL_GAMMA),
        "max_train_pairs": str(MAX_TRAIN_PAIRS),
        "num_components_grid": ",".join(map(str, NUM_COMPONENTS_GRID)),
        "validation_fraction": str(VALIDATION_FRACTION),
        "selection_horizons": ",".join(map(str, spec.horizons)),
        "min_component_transitions": str(MIN_COMPONENT_TRANSITIONS),
        "max_abs_state_for_fit": str(MAX_ABS_STATE_FOR_FIT),
        "env_dt": str(ENV_DT),
        "dysts_dt_multiplier": str(spec.dysts_dt_multiplier),
        "dysts_standardize": str(spec.dysts_standardize),
        "config_name": CONFIG_NAME,
        "torch_threads": str(TORCH_THREADS),
    }
    mismatches = {
        key: (row.get(key), expected)
        for key, expected in exact.items()
        if row.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Task contract mismatch: {mismatches}")
    if row.get("system") not in spec.systems or int(row["seed"]) not in SEEDS:
        raise ValueError("Task system or seed is outside the frozen roster")
    if not math.isclose(float(row["train_fraction"]), spec.train_fraction):
        raise ValueError("Task train_fraction drifted")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ROW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, buffer.getvalue().encode())


def _task_output_dir(result_root: Path, system: str, seed: int) -> Path:
    slug = system.replace(":", "_").replace("/", "_").replace(".", "p").replace("-", "_")
    return result_root / "runs" / "local_edmd_koopman" / slug / f"seed_{seed}"


def _base_row(
    row: dict[str, str],
    *,
    env_dt: float,
    state_dim: int,
    train_count: int,
    validation_count: int,
    test_count: int,
    labels_initial: Optional[np.ndarray],
    labels_final: Optional[np.ndarray],
) -> dict[str, object]:
    return {
        "system": row["system"],
        "seed": int(row["seed"]),
        "method": METHOD_ID,
        "env_dt": env_dt,
        "state_dim": state_dim,
        "num_trajectories": int(row["num_trajectories"]),
        "trajectory_length": int(row["trajectory_length"]),
        "train_fraction": float(row["train_fraction"]),
        "train_trajectories": train_count,
        "validation_trajectories": validation_count,
        "test_trajectories": test_count,
        "ridge_lambda": RIDGE_LAMBDA,
        "edmd_degree": EDMD_DEGREE,
        "kernel_centers_requested": KERNEL_CENTERS,
        "num_components_grid": ",".join(map(str, NUM_COMPONENTS_GRID)),
        "selection_horizons": row["selection_horizons"],
        "validation_fraction": VALIDATION_FRACTION,
        "min_component_transitions": MIN_COMPONENT_TRANSITIONS,
        "max_abs_state_for_fit": MAX_ABS_STATE_FOR_FIT,
        "feature_method": "edmd_poly",
        "route_space": "state",
        "test_initial_basin_hist": _hist(labels_initial),
        "test_final_basin_hist": _hist(labels_final),
    }


def execute_task(row: dict[str, str]) -> list[dict[str, object]]:
    """Generate data, select k without labels/counts, refit, and forecast."""

    system, seed = row["system"], int(row["seed"])
    horizons = _parse_ints(row["horizons"])
    env, env_dt = _make_env_for_system(
        system,
        seed=seed,
        config_name=CONFIG_NAME,
        explicit_env_dt=None,
        dysts_dt_multiplier=float(row["dysts_dt_multiplier"]),
        dysts_standardize=bool(int(row["dysts_standardize"])),
    )
    trajectories = _generate_trajectories(
        env,
        system=system,
        seed=seed,
        num_trajectories=int(row["num_trajectories"]),
        trajectory_length=int(row["trajectory_length"]),
    )
    split_rng = np.random.default_rng(_stable_seed("split", system, seed))
    train_indices, test_indices = _split_trajectories(
        trajectories.shape[0], float(row["train_fraction"]), split_rng
    )
    train_trajectories = trajectories[train_indices]
    test_trajectories = trajectories[test_indices]
    count_rng = np.random.default_rng(
        _stable_seed("local-edmd-counts", system, seed)
    )
    _, validation_trajectories = split_fit_validation(
        train_trajectories,
        validation_fraction=VALIDATION_FRACTION,
        rng=count_rng,
    )
    # Diagnostics are computed only after the train/test split and never enter the fit.
    labels_initial = _maybe_basin_labels(env, test_trajectories[:, 0, :])
    labels_final = _maybe_basin_labels(env, test_trajectories[:, -1, :])
    base = _base_row(
        row,
        env_dt=env_dt,
        state_dim=int(env.observation_size),
        train_count=train_trajectories.shape[0],
        validation_count=validation_trajectories.shape[0],
        test_count=test_trajectories.shape[0],
        labels_initial=labels_initial,
        labels_final=labels_final,
    )
    model, selection = select_and_fit(
        train_trajectories,
        num_components_grid=NUM_COMPONENTS_GRID,
        validation_fraction=VALIDATION_FRACTION,
        selection_horizons=horizons,
        edmd_degree=EDMD_DEGREE,
        ridge_lambda=RIDGE_LAMBDA,
        max_train_pairs=MAX_TRAIN_PAIRS,
        min_component_transitions=MIN_COMPONENT_TRANSITIONS,
        max_abs_state_for_fit=MAX_ABS_STATE_FOR_FIT,
        seed=_stable_seed("select", system, seed, METHOD_ID),
        evaluator=evaluate_rollout,
    )
    metrics_by_horizon = evaluate_rollout(model, test_trajectories, horizons)
    rows: list[dict[str, object]] = []
    for horizon, metrics in metrics_by_horizon.items():
        result = dict(base)
        result.update(metrics)
        result.update(
            {
                "status": "ok",
                "skip_reason": "",
                "horizon": horizon,
                "feature_dim": model.feature_dim,
                "train_transitions": model.train_transitions,
                "selected_num_components": model.selected_num_components,
                "validation_score": selection["validation_score"],
                "candidate_scores_json": json.dumps(
                    selection["candidate_scores"], default=_json_default
                ),
                "fitted_component_count": model.fitted_component_count,
                "component_counts": json.dumps(model.component_counts),
                "kernel_centers_used": "",
                "kernel_gamma": "",
            }
        )
        rows.append(result)
    return rows


def run_task(
    task_tsv: Path, task_index: int, result_root: Path
) -> tuple[Path, Path]:
    """Run one authenticated array row and write raw provenance artifacts."""

    lock = verify_source_lock()
    row = read_task(task_tsv, task_index)
    validate_task(row, task_index)
    torch.set_num_threads(TORCH_THREADS)
    output_dir = _task_output_dir(result_root, row["system"], int(row["seed"]))
    rows = execute_task(row)
    rows_path = output_dir / "rows.csv"
    _write_rows(rows_path, rows)
    metadata = {
        "protocol_id": lock["protocol_id"],
        "task_index": task_index,
        "task": row,
        "source_lock_sha256": hashlib.sha256(
            Path(__file__).with_name("source_lock.json").read_bytes()
        ).hexdigest(),
        "row_file_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "label_use": "evaluation_diagnostics_only",
        "rollout_update": "reroute_each_predicted_state",
    }
    metadata_path = output_dir / "task_metadata.json"
    _atomic_write(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    )
    return rows_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-tsv", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_path, metadata_path = run_task(
        args.task_tsv, args.task_index, args.result_root
    )
    print(f"Wrote {rows_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
