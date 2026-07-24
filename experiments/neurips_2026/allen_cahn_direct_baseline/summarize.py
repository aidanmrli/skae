"""Build the frozen, fail-closed report-always direct-baseline packet."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.neurips_2026.allen_cahn_direct_baseline.core import (
    duplicate_safe_json,
    sha256_path,
    torch_load,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_direct_baseline.evaluate import (
    load_checkpoint,
)
from experiments.neurips_2026.allen_cahn_direct_baseline.train import verify_lock


EXPECTED_CANDIDATES = [2000] + list(range(2251, 5252, 250)) + [5500]
DATASET_IDS = ("development_20260724", "new_ic_0", "new_ic_1", "new_ic_2")
NEW_IC_IDS = DATASET_IDS[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-lock", type=Path, required=True)
    parser.add_argument("--expected-task-lock-sha256", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def studentized_mean(values: np.ndarray) -> float:
    average = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation == 0.0:
        return math.copysign(math.inf, average) if average else 0.0
    return average / (standard_deviation / math.sqrt(len(values)))


def exact_one_sided_sign_flip(differences: np.ndarray) -> float:
    observed = studentized_mean(differences)
    exceedances = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = differences * np.asarray(signs, dtype=np.float64)
        if studentized_mean(permuted) >= observed:
            exceedances += 1
    return exceedances / (2 ** len(differences))


def paired_effect(
    direct: np.ndarray,
    comparator: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    direct_advantage = 1.0 - float(np.mean(direct)) / float(np.mean(comparator))
    comparator_advantage = 1.0 - float(np.mean(comparator)) / float(np.mean(direct))
    rng = np.random.default_rng(int(bootstrap_seed))
    samples = rng.integers(
        0, len(direct), size=(int(bootstrap_replicates), len(direct))
    )
    direct_boot = direct[samples].mean(axis=1)
    comparator_boot = comparator[samples].mean(axis=1)
    direct_effects = 1.0 - direct_boot / comparator_boot
    comparator_effects = 1.0 - comparator_boot / direct_boot
    direct_ci = np.quantile(direct_effects, (0.025, 0.975))
    comparator_ci = np.quantile(comparator_effects, (0.025, 0.975))
    return {
        "direct_mean": float(np.mean(direct)),
        "comparator_mean": float(np.mean(comparator)),
        "direct_advantage": direct_advantage,
        "direct_advantage_ci95": [float(direct_ci[0]), float(direct_ci[1])],
        "direct_seed_wins": int(np.sum(direct < comparator)),
        "direct_one_sided_exact_sign_flip_p": exact_one_sided_sign_flip(
            comparator - direct
        ),
        "comparator_advantage": comparator_advantage,
        "comparator_advantage_ci95": [
            float(comparator_ci[0]),
            float(comparator_ci[1]),
        ],
        "comparator_seed_wins": int(np.sum(comparator < direct)),
        "comparator_one_sided_exact_sign_flip_p": exact_one_sided_sign_flip(
            direct - comparator
        ),
    }


def descriptive_effect(direct: np.ndarray, comparator: np.ndarray) -> dict[str, Any]:
    return {
        "direct_mean": float(np.mean(direct)),
        "comparator_mean": float(np.mean(comparator)),
        "direct_advantage": 1.0
        - float(np.mean(direct)) / float(np.mean(comparator)),
        "direct_seed_wins": int(np.sum(direct < comparator)),
        "comparator_advantage": 1.0
        - float(np.mean(comparator)) / float(np.mean(direct)),
        "comparator_seed_wins": int(np.sum(comparator < direct)),
        "status": "descriptive_only_no_test_no_interval_no_rescue",
    }


def holm(p_by_name: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_by_name.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (total - rank) * float(value))
        adjusted[name] = min(1.0, running)
    return adjusted


def require_hash(path: Path, expected: object, label: str) -> None:
    observed = sha256_path(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash mismatch: {observed} != {expected}")


def validate_telemetry(
    path: Path,
    *,
    raw_path: Path,
    phase_start: Path,
    phase_end: Path,
    role: str,
    seed: int,
    task_lock_sha256: str,
) -> dict[str, Any]:
    payload = duplicate_safe_json(path)
    expected = {
        "protocol_id": "allen_cahn_matched_direct_baseline_v1",
        "artifact_role": role,
        "task_lock_sha256": task_lock_sha256,
        "model_seed": seed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Telemetry {key} mismatch")
    if payload.get("passed") is not True:
        raise RuntimeError("Telemetry did not pass")
    if not payload.get("checks") or not all(payload["checks"].values()):
        raise RuntimeError("Telemetry contains a failed or absent check")
    require_hash(raw_path, payload.get("telemetry_sha256"), "raw telemetry")
    require_hash(phase_start, payload.get("phase_start_sha256"), "phase start")
    require_hash(phase_end, payload.get("phase_end_sha256"), "phase end")
    return payload


def validate_curves(metrics: dict[str, Any]) -> None:
    expected_keys = {
        "trajectories",
        "horizon_steps",
        "instantaneous_field_mse",
        "through_horizon_field_mse",
        "persistence_instantaneous_field_mse",
        "persistence_through_horizon_field_mse",
        "endpoints",
    }
    if set(metrics) != expected_keys:
        raise RuntimeError("Evaluation metric schema drift")
    if int(metrics["trajectories"]) != 256 or int(metrics["horizon_steps"]) != 200:
        raise RuntimeError("Evaluation trajectory/horizon count drift")
    names = (
        "instantaneous_field_mse",
        "through_horizon_field_mse",
        "persistence_instantaneous_field_mse",
        "persistence_through_horizon_field_mse",
    )
    arrays = {name: np.asarray(metrics[name], dtype=np.float64) for name in names}
    for name, array in arrays.items():
        if array.shape != (200,) or not np.all(np.isfinite(array)) or np.any(array < 0):
            raise RuntimeError(f"Invalid evaluation curve {name}")
    denominator = np.arange(1.0, 201.0)
    if not np.allclose(
        arrays["through_horizon_field_mse"],
        np.cumsum(arrays["instantaneous_field_mse"]) / denominator,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Direct cumulative curve identity failed")
    if not np.allclose(
        arrays["persistence_through_horizon_field_mse"],
        np.cumsum(arrays["persistence_instantaneous_field_mse"]) / denominator,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError("Persistence cumulative curve identity failed")
    if set(metrics["endpoints"]) != {"80", "120", "160", "200"}:
        raise RuntimeError("Endpoint roster drift")
    for horizon in (80, 120, 160, 200):
        endpoint = metrics["endpoints"][str(horizon)]
        expected = {
            "through_horizon_field_mse": arrays[
                "through_horizon_field_mse"
            ][horizon - 1],
            "terminal_field_mse": arrays["instantaneous_field_mse"][horizon - 1],
            "persistence_through_horizon_field_mse": arrays[
                "persistence_through_horizon_field_mse"
            ][horizon - 1],
            "persistence_terminal_field_mse": arrays[
                "persistence_instantaneous_field_mse"
            ][horizon - 1],
        }
        if set(endpoint) != set(expected):
            raise RuntimeError("Endpoint schema drift")
        for key, value in expected.items():
            if not math.isclose(float(endpoint[key]), float(value), rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"Endpoint/curve identity failed for H{horizon} {key}")


def seed_directories(root: Path) -> set[int]:
    result = set()
    for path in root.glob("seed_*"):
        if path.is_dir():
            try:
                result.add(int(path.name.removeprefix("seed_")))
            except ValueError as exc:
                raise RuntimeError(f"Malformed seed directory {path}") from exc
    return result


def authenticate_seed(
    *,
    seed: int,
    training_root: Path,
    evaluation_root: Path,
    lock: dict[str, Any],
    task_lock_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_seed = training_root / f"seed_{seed}"
    model_root = train_seed / "model"
    evaluation_seed = evaluation_root / f"seed_{seed}"
    paths = {
        "run_manifest": model_root / "run_manifest.json",
        "model_audit": model_root / "model_audit.json",
        "training_summary": model_root / "training_summary.json",
        "metrics_history": model_root / "metrics_history.jsonl",
        "checkpoint": model_root / "checkpoint.pt",
        "train_phase_start": model_root / "training_phase_start.json",
        "train_phase_end": model_root / "training_phase_end.json",
        "train_raw_telemetry": train_seed / "raw_gpu_telemetry.csv",
        "train_telemetry": train_seed / "telemetry_audit.json",
        "evaluation": evaluation_seed / "evaluation.json",
        "eval_phase_start": evaluation_seed / "evaluation_phase_start.json",
        "eval_phase_end": evaluation_seed / "evaluation_phase_end.json",
        "eval_raw_telemetry": evaluation_seed / "raw_gpu_telemetry.csv",
        "eval_telemetry": evaluation_seed / "telemetry_audit.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing seed-{seed} artifacts: {missing}")

    manifest = duplicate_safe_json(paths["run_manifest"])
    audit = duplicate_safe_json(paths["model_audit"])
    summary = duplicate_safe_json(paths["training_summary"])
    if manifest.get("protocol_id") != lock["protocol_id"]:
        raise RuntimeError("Training manifest protocol mismatch")
    if manifest.get("artifact_role") != "scientific_training":
        raise RuntimeError("Training manifest role mismatch")
    if manifest.get("task_lock_sha256") != task_lock_sha256:
        raise RuntimeError("Training manifest task-lock mismatch")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError("Training manifest seed mismatch")
    if int(manifest.get("realized_total_updates", -1)) != 5500:
        raise RuntimeError("Training manifest update-budget mismatch")
    if manifest.get("model_config") != lock["model_and_compute_match"]["direct_config"]:
        raise RuntimeError("Training manifest model configuration mismatch")
    if manifest.get("training_dataset", {}).get("sha256") != lock["datasets"][
        "training_and_selector"
    ]["sha256"]:
        raise RuntimeError("Training manifest dataset mismatch")
    if audit.get("passed") is not True or not all(audit.get("checks", {}).values()):
        raise RuntimeError("Model audit failed")
    if int(audit.get("parameter_count", -1)) != int(
        lock["model_and_compute_match"]["direct_parameter_count"]
    ):
        raise RuntimeError("Model-audit parameter count mismatch")
    if manifest.get("model_audit") != audit:
        raise RuntimeError("Run manifest does not embed the exact model audit")

    expected_summary = {
        "protocol_id": lock["protocol_id"],
        "artifact_role": "scientific_training",
        "task_lock_sha256": task_lock_sha256,
        "status": "training_completed",
        "seed": seed,
        "completed_optimizer_updates": 5500,
        "selector_candidate_updates": EXPECTED_CANDIDATES,
        "selector_candidate_count": 15,
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Training summary {key} mismatch")
    require_hash(paths["run_manifest"], summary.get("run_manifest_sha256"), "run manifest")
    require_hash(paths["model_audit"], summary.get("model_audit_sha256"), "model audit")
    require_hash(paths["metrics_history"], summary.get("metrics_history_sha256"), "history")
    require_hash(paths["train_phase_start"], summary.get("training_phase_start_sha256"), "train phase start")
    require_hash(paths["train_phase_end"], summary.get("training_phase_end_sha256"), "train phase end")
    require_hash(paths["checkpoint"], summary.get("checkpoint_sha256"), "checkpoint")
    history = [
        json.loads(line)
        for line in paths["metrics_history"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [int(row["completed_updates"]) for row in history] != EXPECTED_CANDIDATES:
        raise RuntimeError("Training history selector cadence mismatch")
    scores = [float(row["selector_score"]) for row in history]
    if not all(math.isfinite(value) for value in scores):
        raise RuntimeError("Nonfinite selector score")
    best_index = int(np.argmin(np.asarray(scores)))
    if int(summary["best_completed_updates"]) != EXPECTED_CANDIDATES[best_index]:
        raise RuntimeError("Selected checkpoint update is not the earliest minimum")
    if not math.isclose(
        float(summary["best_selector_score"]), scores[best_index], rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError("Selected checkpoint score mismatch")
    checkpoint = torch_load(paths["checkpoint"])
    checkpoint_expected = {
        "protocol_id": lock["protocol_id"],
        "model_family": "spatial_conv_autoregressive",
        "model_seed": seed,
        "completed_updates": int(summary["best_completed_updates"]),
        "task_lock_sha256": task_lock_sha256,
        "training_dataset_sha256": lock["datasets"]["training_and_selector"]["sha256"],
        "model_config": lock["model_and_compute_match"]["direct_config"],
    }
    for key, value in checkpoint_expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"Checkpoint {key} mismatch")
    if not math.isclose(
        float(checkpoint["selector_score"]),
        float(summary["best_selector_score"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("Checkpoint selector score mismatch")
    loaded_model, loaded_payload, loaded_sha256 = load_checkpoint(
        paths["checkpoint"],
        seed=seed,
        task_lock_sha256=task_lock_sha256,
        lock=lock,
        device=torch.device("cpu"),
    )
    if loaded_sha256 != summary["checkpoint_sha256"] or loaded_payload[
        "completed_updates"
    ] != summary["best_completed_updates"]:
        raise RuntimeError("Strict checkpoint reload did not reconcile")
    del loaded_model, loaded_payload

    train_telemetry = validate_telemetry(
        paths["train_telemetry"],
        raw_path=paths["train_raw_telemetry"],
        phase_start=paths["train_phase_start"],
        phase_end=paths["train_phase_end"],
        role="scientific_training",
        seed=seed,
        task_lock_sha256=task_lock_sha256,
    )
    if not manifest.get("slurm_job_id") or train_telemetry["slurm_job_id"] != str(
        manifest["slurm_job_id"]
    ):
        raise RuntimeError("Training telemetry/manifest SLURM lineage mismatch")
    if str(manifest.get("slurm_array_task_id")) != str(seed - 64):
        raise RuntimeError("Training array-task/seed mapping mismatch")
    evaluation = duplicate_safe_json(paths["evaluation"])
    expected_evaluation = {
        "protocol_id": lock["protocol_id"],
        "task_lock_sha256": task_lock_sha256,
        "model_seed": seed,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "checkpoint_completed_updates": summary["best_completed_updates"],
        "training_summary_sha256": sha256_path(paths["training_summary"]),
        "rollout": "autonomous_observation_space_recurrence_from_x0_only",
        "label_firewall": "only fields and split_indices accessed",
        "slurm_array_task_id": str(seed - 64),
    }
    for key, value in expected_evaluation.items():
        if evaluation.get(key) != value:
            raise RuntimeError(f"Evaluation {key} mismatch")
    locked_datasets = {
        str(row["dataset_id"]): row
        for row in lock["datasets"]["report_always_evaluation"]
    }
    if tuple(evaluation.get("datasets", {}).keys()) != DATASET_IDS:
        raise RuntimeError("Evaluation dataset order/roster mismatch")
    for dataset_id, locked in locked_datasets.items():
        observed = evaluation["datasets"][dataset_id]
        for key in ("dataset_seed", "role", "path", "sha256"):
            if observed.get(key) != locked[key]:
                raise RuntimeError(f"Evaluation {dataset_id} {key} mismatch")
        validate_curves(observed["metrics"])
    eval_telemetry = validate_telemetry(
        paths["eval_telemetry"],
        raw_path=paths["eval_raw_telemetry"],
        phase_start=paths["eval_phase_start"],
        phase_end=paths["eval_phase_end"],
        role="scientific_evaluation",
        seed=seed,
        task_lock_sha256=task_lock_sha256,
    )
    if not evaluation.get("slurm_job_id") or eval_telemetry["slurm_job_id"] != str(
        evaluation["slurm_job_id"]
    ):
        raise RuntimeError("Evaluation telemetry/output SLURM lineage mismatch")
    authentication = {
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "training_summary_sha256": sha256_path(paths["training_summary"]),
        "evaluation_sha256": sha256_path(paths["evaluation"]),
        "training_telemetry_sha256": sha256_path(paths["train_telemetry"]),
        "evaluation_telemetry_sha256": sha256_path(paths["eval_telemetry"]),
        "training_gpu_uuid": train_telemetry["gpu_uuid"],
        "evaluation_gpu_uuid": eval_telemetry["gpu_uuid"],
        "selected_update": summary["best_completed_updates"],
        "selector_score": summary["best_selector_score"],
    }
    return evaluation, authentication

