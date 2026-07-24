"""Train one frozen-seed matched direct Allen--Cahn baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from experiments.neurips_2026.allen_cahn_direct_baseline.core import (
    DirectConfig,
    DirectResidualConv,
    augment_periodic_symmetries,
    duplicate_safe_json,
    joint_endpoint_metrics,
    load_field_splits,
    parameter_count,
    sample_sequence_batch,
    select_split,
    sha256_path,
    verify_source_manifest,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_direct_baseline.execution import (
    CudaGraphTrainingStepper,
    eager_training_step,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-lock", type=Path, required=True)
    parser.add_argument("--expected-task-lock-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=80)
    return parser.parse_args()


def verify_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"Task-lock drift: {observed} != {expected_sha256}")
    lock = duplicate_safe_json(path)
    if lock.get("protocol_id") != "allen_cahn_matched_direct_baseline_v1":
        raise RuntimeError("Unexpected direct-baseline protocol")
    card_record = lock["prediction_card"]
    card_path = REPO_ROOT / str(card_record["path"])
    if sha256_path(card_path) != str(card_record["sha256"]):
        raise RuntimeError("Prediction-card hash mismatch")
    card = duplicate_safe_json(card_path)
    if card.get("protocol_id") != lock["protocol_id"]:
        raise RuntimeError("Card/lock protocol mismatch")
    manifest_record = lock["source_manifest"]
    manifest_path = REPO_ROOT / str(manifest_record["path"])
    if sha256_path(manifest_path) != str(manifest_record["sha256"]):
        raise RuntimeError("Source-manifest hash mismatch")
    verify_source_manifest(REPO_ROOT, manifest_path)
    return lock


def model_audit(
    model: DirectResidualConv,
    optimizer: torch.optim.Optimizer,
    lock: dict[str, Any],
) -> dict[str, Any]:
    forbidden = (
        torch.nn.ReLU,
        torch.nn.GELU,
        torch.nn.Softshrink,
        torch.nn.Dropout,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
    )
    forbidden_modules = [
        f"{name}:{module.__class__.__name__}"
        for name, module in model.named_modules()
        if isinstance(module, forbidden)
    ]
    count = parameter_count(model)
    expected = int(lock["model_and_compute_match"]["direct_parameter_count"])
    checks = {
        "parameter_count_matches_lock": count == expected,
        "activation_is_tanh": str(model.cfg.activation) == "tanh",
        "no_zero_inducing_or_dropout_module": not forbidden_modules,
        "optimizer_is_adam_not_adamw": isinstance(optimizer, torch.optim.Adam)
        and not isinstance(optimizer, torch.optim.AdamW),
        "optimizer_is_cuda_graph_capturable": all(
            bool(group.get("capturable", False)) for group in optimizer.param_groups
        ),
        "all_weight_decay_zero": all(
            float(group.get("weight_decay", 0.0)) == 0.0
            for group in optimizer.param_groups
        ),
        "all_parameters_trainable": all(
            bool(parameter.requires_grad) for parameter in model.parameters()
        ),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Direct-model audit failed: {failed}")
    return {
        "passed": True,
        "checks": checks,
        "forbidden_modules": forbidden_modules,
        "parameter_count": count,
    }


def save_checkpoint(
    path: Path,
    *,
    model: DirectResidualConv,
    config: DirectConfig,
    seed: int,
    completed_updates: int,
    selector_score: float,
    selector_endpoints: dict[str, dict[str, float]],
    task_lock_sha256: str,
    training_dataset_sha256: str,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": 1,
            "protocol_id": "allen_cahn_matched_direct_baseline_v1",
            "model_family": "spatial_conv_autoregressive",
            "model_state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "model_seed": int(seed),
            "completed_updates": int(completed_updates),
            "selector_score": float(selector_score),
            "selector_endpoints": selector_endpoints,
            "task_lock_sha256": task_lock_sha256,
            "training_dataset_sha256": training_dataset_sha256,
            "rollout_mode": "autonomous_observation_space_recurrence",
        },
        temporary,
    )
    temporary.replace(path)


def phase_marker(path: Path, phase: str) -> None:
    write_json_once(
        path,
        {
            "phase": phase,
            "utc": datetime.now(timezone.utc).isoformat(),
            "unix_time_seconds": time.time(),
        },
    )


def main() -> None:
    args = parse_args()
    lock = verify_lock(args.task_lock, args.expected_task_lock_sha256)
    scientific = lock["scientific_protocol"]
    if int(args.seed) not in [int(value) for value in scientific["model_seeds"]]:
        raise ValueError("Seed is outside the frozen roster")
    if args.run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.run_dir}")
    args.run_dir.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("The direct baseline requires CUDA")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    dataset_record = lock["datasets"]["training_and_selector"]
    fields, splits = load_field_splits(
        Path(str(dataset_record["path"])),
        expected_sha256=str(dataset_record["sha256"]),
        expected_total_shape=tuple(int(v) for v in dataset_record["field_shape"]),
    )
    train_fields = select_split(fields, splits, "train")
    validation_all = select_split(fields, splits, "val")
    validation_fields = validation_all[0::2].contiguous()
    expected_train_shape = tuple(scientific["training_field_shape"])
    expected_selector_shape = tuple(scientific["selector_field_shape"])
    if tuple(train_fields.shape) != expected_train_shape:
        raise ValueError(f"Training shape drift: {tuple(train_fields.shape)}")
    if tuple(validation_fields.shape) != expected_selector_shape:
        raise ValueError(f"Selector shape drift: {tuple(validation_fields.shape)}")

    config = DirectConfig(**lock["model_and_compute_match"]["direct_config"])
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    model = DirectResidualConv(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(scientific["learning_rate"]),
        weight_decay=0.0,
        capturable=True,
    )
    audit = model_audit(model, optimizer, lock)
    generator = torch.Generator().manual_seed(int(args.seed) + 12345)

    planned_updates = int(scientific["total_optimizer_updates"])
    warmup_updates = int(scientific["selector_warmup_updates"])
    checkpoint_every = int(scientific["checkpoint_every_updates"])
    batch_size = int(scientific["batch_size"])
    horizon = int(scientific["training_horizon_steps"])
    gradient_weight = float(scientific["gradient_weight"])
    smoke = bool(args.smoke)
    total_updates = int(args.smoke_steps) if smoke else planned_updates
    if smoke and total_updates != int(lock["telemetry"]["smoke_optimizer_updates"]):
        raise ValueError("Smoke update count does not match the frozen task lock")

    run_manifest = {
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "artifact_role": "non_scientific_gpu_smoke" if smoke else "scientific_training",
        "task_lock_sha256": args.expected_task_lock_sha256,
        "seed": int(args.seed),
        "model_config": config.to_dict(),
        "model_audit": audit,
        "training_dataset": dataset_record,
        "train_shape": list(train_fields.shape),
        "selector_shape": list(validation_fields.shape),
        "scientific_protocol": scientific,
        "realized_total_updates": total_updates,
        "device": str(device),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_device_name": torch.cuda.get_device_name(),
        "cuda_device_capability": list(torch.cuda.get_device_capability()),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "training_execution": (
            "one_eager_update_then_full_optimizer_step_cuda_graph_with_"
            "post_replay_batched_finiteness_reduction"
        ),
        "label_firewall": "only fields and split_indices accessed",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    write_json_once(args.run_dir / "run_manifest.json", run_manifest)
    (args.run_dir / "model_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)

    best_score = math.inf
    best_updates = -1
    history_path = args.run_dir / "metrics_history.jsonl"
    step_times: list[float] = []
    graph_replay_times: list[float] = []
    graph_stepper: CudaGraphTrainingStepper | None = None
    torch.cuda.reset_peak_memory_stats()
    phase_marker(args.run_dir / "training_phase_start.json", "optimizer_loop_start")
    for update_index in range(total_updates):
        sequence = sample_sequence_batch(
            train_fields,
            batch_size=batch_size,
            window_length=horizon,
            generator=generator,
        )
        sequence = augment_periodic_symmetries(sequence, generator=generator)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        sequence_device = sequence.to(device)
        if update_index == 0:
            metrics = eager_training_step(
                model,
                optimizer,
                sequence_device,
                gradient_weight=gradient_weight,
            )
        elif update_index == 1:
            graph_stepper = CudaGraphTrainingStepper(
                model,
                optimizer,
                sequence_device,
                gradient_weight=gradient_weight,
            )
            metrics = graph_stepper.last_metrics
        else:
            if graph_stepper is None:
                raise RuntimeError("CUDA graph stepper was not initialized")
            metrics = graph_stepper.step(sequence_device)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - start_time)
        if update_index >= 2:
            graph_replay_times.append(step_times[-1])
        completed_updates = update_index + 1
        if completed_updates == 1 or completed_updates % 25 == 0:
            print(
                f"updates={completed_updates} loss={metrics['loss']:.7g} "
                f"field={metrics['field_mse']:.7g} seconds={step_times[-1]:.4f}",
                flush=True,
            )
        if smoke:
            continue
        should_select = completed_updates == warmup_updates or (
            completed_updates > warmup_updates
            and (
                (
                    completed_updates - warmup_updates >= checkpoint_every + 1
                    and (completed_updates - warmup_updates - 1)
                    % checkpoint_every
                    == 0
                )
                or completed_updates == total_updates
            )
        )
        if not should_select:
            continue
        endpoints, selector_score = joint_endpoint_metrics(
            model,
            validation_fields,
            horizons=tuple(int(v) for v in scientific["checkpoint_horizons"]),
            device=device,
            batch_size=int(scientific["selector_batch_size"]),
        )
        record = {
            "completed_updates": completed_updates,
            "train": metrics,
            "selector_endpoints": endpoints,
            "selector_score": selector_score,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        if selector_score < best_score:
            best_score = selector_score
            best_updates = completed_updates
            save_checkpoint(
                args.run_dir / "checkpoint.pt",
                model=model,
                config=config,
                seed=int(args.seed),
                completed_updates=completed_updates,
                selector_score=selector_score,
                selector_endpoints=endpoints,
                task_lock_sha256=args.expected_task_lock_sha256,
                training_dataset_sha256=str(dataset_record["sha256"]),
            )
        print(
            f"selector updates={completed_updates} score={selector_score:.8g} "
            f"best={best_score:.8g}",
            flush=True,
        )

    phase_marker(args.run_dir / "training_phase_end.json", "optimizer_loop_end")

    timing = {
        "updates": total_updates,
        "step_seconds_mean_excluding_first": float(
            sum(step_times[1:]) / max(1, len(step_times) - 1)
        ),
        "step_seconds_min_excluding_first": float(min(step_times[1:])),
        "step_seconds_max_excluding_first": float(max(step_times[1:])),
        "graph_replay_updates": len(graph_replay_times),
        "graph_replay_seconds_mean": (
            float(sum(graph_replay_times) / len(graph_replay_times))
            if graph_replay_times
            else None
        ),
        "execution_mode": "one_eager_update_then_full_optimizer_step_cuda_graph",
        "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved()),
    }
    summary = {
        "status": "smoke_completed" if smoke else "training_completed",
        "protocol_id": lock["protocol_id"],
        "artifact_role": "non_scientific_gpu_smoke" if smoke else "scientific_training",
        "task_lock_sha256": args.expected_task_lock_sha256,
        "seed": int(args.seed),
        "completed_optimizer_updates": int(total_updates),
        "best_selector_score": None if smoke else float(best_score),
        "best_completed_updates": None if smoke else int(best_updates),
        "timing": timing,
        "run_manifest_sha256": sha256_path(args.run_dir / "run_manifest.json"),
        "model_audit_sha256": sha256_path(args.run_dir / "model_audit.json"),
        "training_phase_start_sha256": sha256_path(
            args.run_dir / "training_phase_start.json"
        ),
        "training_phase_end_sha256": sha256_path(
            args.run_dir / "training_phase_end.json"
        ),
    }
    if not smoke:
        checkpoint_path = args.run_dir / "checkpoint.pt"
        if best_updates < 0 or not checkpoint_path.is_file():
            raise RuntimeError("Scientific run completed without a selected checkpoint")
        summary["checkpoint_sha256"] = sha256_path(checkpoint_path)
        history = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_candidates = [2000] + list(range(2251, 5252, 250)) + [5500]
        observed_candidates = [int(row["completed_updates"]) for row in history]
        if observed_candidates != expected_candidates:
            raise RuntimeError(
                f"Selector cadence drift: {observed_candidates} != {expected_candidates}"
            )
        summary["selector_candidate_updates"] = observed_candidates
        summary["selector_candidate_count"] = len(observed_candidates)
        summary["metrics_history_sha256"] = sha256_path(history_path)
    write_json_once(args.run_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
