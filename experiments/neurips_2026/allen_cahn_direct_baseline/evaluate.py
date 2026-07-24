"""Evaluate one frozen direct checkpoint on every report-always dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from experiments.neurips_2026.allen_cahn_direct_baseline.core import (
    DirectConfig,
    DirectResidualConv,
    duplicate_safe_json,
    load_field_splits,
    parameter_count,
    select_split,
    sha256_path,
    torch_load,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_direct_baseline.train import verify_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-lock", type=Path, required=True)
    parser.add_argument("--expected-task-lock-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def phase_marker(path: Path, phase: str) -> None:
    import time

    write_json_once(
        path,
        {
            "phase": phase,
            "utc": datetime.now(timezone.utc).isoformat(),
            "unix_time_seconds": time.time(),
        },
    )


def load_checkpoint(
    path: Path,
    *,
    seed: int,
    task_lock_sha256: str,
    lock: dict[str, Any],
    device: torch.device,
) -> tuple[DirectResidualConv, dict[str, Any], str]:
    checkpoint_sha256 = sha256_path(path)
    payload = torch_load(path)
    if payload.get("protocol_id") != lock["protocol_id"]:
        raise RuntimeError("Checkpoint protocol mismatch")
    if payload.get("model_family") != "spatial_conv_autoregressive":
        raise RuntimeError("Checkpoint model-family mismatch")
    if int(payload.get("model_seed", -1)) != int(seed):
        raise RuntimeError("Checkpoint seed mismatch")
    if payload.get("task_lock_sha256") != task_lock_sha256:
        raise RuntimeError("Checkpoint task-lock mismatch")
    training_record = lock["datasets"]["training_and_selector"]
    if payload.get("training_dataset_sha256") != training_record["sha256"]:
        raise RuntimeError("Checkpoint training-dataset mismatch")
    expected_config = lock["model_and_compute_match"]["direct_config"]
    if payload.get("model_config") != expected_config:
        raise RuntimeError("Checkpoint configuration mismatch")
    config = DirectConfig.from_mapping(payload["model_config"])
    model = DirectResidualConv(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if parameter_count(model) != int(
        lock["model_and_compute_match"]["direct_parameter_count"]
    ):
        raise RuntimeError("Checkpoint parameter count mismatch")
    model = model.float().to(device).eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Checkpoint is not all float32")
    return model, payload, checkpoint_sha256


@torch.inference_mode()
def evaluate_fields(
    model: DirectResidualConv,
    fields: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    horizon = 200
    squared_error = torch.zeros(horizon, dtype=torch.float64)
    persistence_squared_error = torch.zeros(horizon, dtype=torch.float64)
    count_per_step = 0
    for start in range(0, fields.shape[0], int(batch_size)):
        batch = fields[start : start + int(batch_size)].to(device)
        truth = batch[:, 1 : horizon + 1]
        prediction = model.rollout(batch[:, 0], horizon=horizon)
        if tuple(prediction.shape) != tuple(truth.shape):
            raise RuntimeError("Prediction/truth shape mismatch")
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("Nonfinite direct evaluation rollout")
        difference = prediction - truth
        persistence_difference = batch[:, :1] - truth
        squared_error += difference.square().sum(dim=(0, 2)).double().cpu()
        persistence_squared_error += (
            persistence_difference.square().sum(dim=(0, 2)).double().cpu()
        )
        count_per_step += int(difference.shape[0] * difference.shape[2])
    instantaneous = squared_error / count_per_step
    persistence_instantaneous = persistence_squared_error / count_per_step
    cumulative = instantaneous.cumsum(dim=0) / torch.arange(
        1, horizon + 1, dtype=torch.float64
    )
    persistence_cumulative = persistence_instantaneous.cumsum(dim=0) / torch.arange(
        1, horizon + 1, dtype=torch.float64
    )
    values = torch.cat(
        [instantaneous, cumulative, persistence_instantaneous, persistence_cumulative]
    )
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite direct evaluation metric")
    return {
        "trajectories": int(fields.shape[0]),
        "horizon_steps": horizon,
        "instantaneous_field_mse": instantaneous.tolist(),
        "through_horizon_field_mse": cumulative.tolist(),
        "persistence_instantaneous_field_mse": persistence_instantaneous.tolist(),
        "persistence_through_horizon_field_mse": persistence_cumulative.tolist(),
        "endpoints": {
            str(h): {
                "through_horizon_field_mse": float(cumulative[h - 1]),
                "terminal_field_mse": float(instantaneous[h - 1]),
                "persistence_through_horizon_field_mse": float(
                    persistence_cumulative[h - 1]
                ),
                "persistence_terminal_field_mse": float(
                    persistence_instantaneous[h - 1]
                ),
            }
            for h in (80, 120, 160, 200)
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    lock = verify_lock(args.task_lock, args.expected_task_lock_sha256)
    seeds = [int(value) for value in lock["scientific_protocol"]["model_seeds"]]
    if int(args.seed) not in seeds:
        raise ValueError("Seed is outside the frozen roster")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    training_summary = duplicate_safe_json(args.training_summary)
    if training_summary.get("protocol_id") != lock["protocol_id"]:
        raise RuntimeError("Training-summary protocol mismatch")
    if training_summary.get("artifact_role") != "scientific_training":
        raise RuntimeError("Training-summary role mismatch")
    if training_summary.get("status") != "training_completed":
        raise RuntimeError("Training did not complete")
    if training_summary.get("task_lock_sha256") != args.expected_task_lock_sha256:
        raise RuntimeError("Training-summary task-lock mismatch")
    if int(training_summary.get("seed", -1)) != int(args.seed):
        raise RuntimeError("Training-summary seed mismatch")
    if int(training_summary.get("completed_optimizer_updates", -1)) != 5500:
        raise RuntimeError("Training update budget mismatch")
    if int(training_summary.get("selector_candidate_count", -1)) != 15:
        raise RuntimeError("Training selector-candidate count mismatch")
    if sha256_path(args.checkpoint) != training_summary.get("checkpoint_sha256"):
        raise RuntimeError("Training summary does not bind the selected checkpoint")
    model, checkpoint, checkpoint_sha256 = load_checkpoint(
        args.checkpoint,
        seed=int(args.seed),
        task_lock_sha256=args.expected_task_lock_sha256,
        lock=lock,
        device=device,
    )
    if not math.isfinite(float(checkpoint["selector_score"])):
        raise RuntimeError("Checkpoint selector score is nonfinite")
    if int(checkpoint["completed_updates"]) != int(
        training_summary["best_completed_updates"]
    ):
        raise RuntimeError("Checkpoint update does not match training selection")

    loaded_fields: dict[str, tuple[dict[str, Any], torch.Tensor]] = {}
    for record in lock["datasets"]["report_always_evaluation"]:
        fields, splits = load_field_splits(
            Path(str(record["path"])),
            expected_sha256=str(record["sha256"]),
            expected_total_shape=tuple(int(v) for v in record["field_shape"]),
        )
        evaluation_fields = select_split(fields, splits, str(record["split"]))
        if tuple(evaluation_fields.shape) != (256, 201, 512):
            raise RuntimeError("Evaluation field shape drift")
        loaded_fields[str(record["dataset_id"])] = (record, evaluation_fields)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    phase_marker(args.output.parent / "evaluation_phase_start.json", "evaluation_loop_start")
    outputs: dict[str, Any] = {}
    for dataset_id, (record, evaluation_fields) in loaded_fields.items():
        outputs[dataset_id] = {
            "dataset_seed": int(record["dataset_seed"]),
            "role": str(record["role"]),
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
            "metrics": evaluate_fields(
                model,
                evaluation_fields,
                device=device,
                batch_size=int(args.batch_size),
            ),
        }
    phase_marker(args.output.parent / "evaluation_phase_end.json", "evaluation_loop_end")

    payload = {
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "task_lock_sha256": args.expected_task_lock_sha256,
        "model_seed": int(args.seed),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "training_summary_path": str(args.training_summary),
        "training_summary_sha256": sha256_path(args.training_summary),
        "checkpoint_completed_updates": int(checkpoint["completed_updates"]),
        "checkpoint_selector_score": float(checkpoint["selector_score"]),
        "rollout": "autonomous_observation_space_recurrence_from_x0_only",
        "label_firewall": "only fields and split_indices accessed",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_device_name": torch.cuda.get_device_name(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "datasets": outputs,
    }
    write_json_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": "evaluation_completed",
                "seed": int(args.seed),
                "output": str(args.output),
                "output_sha256": sha256_path(args.output),
                "checkpoint_sha256": checkpoint_sha256,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
