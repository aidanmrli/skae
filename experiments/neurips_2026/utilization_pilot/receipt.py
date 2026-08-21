"""Final pilot receipt construction and storage-policy metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pilot import NCU_SMO_METRIC, SCHEMA_VERSION, rgu_accounting


def final_receipt(
    *,
    output: Path,
    status: str,
    identity: dict[str, Any],
    source_manifest: dict[str, Any],
    gpu_identity: dict[str, Any],
    measured_telemetry: dict[str, Any],
    profile_telemetry: dict[str, Any],
    phases: dict[str, Any],
    timing: dict[str, Any],
    smo: dict[str, Any],
    allocation_elapsed_seconds: float,
    attempt: int,
    continuation_validated: bool = False,
    continuation_receipt: str | None = None,
    storage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    measured_elapsed = float(timing.get("elapsed_seconds", 0.0))
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "utilization_pilot_final",
        "status": status,
        "identity_sha256": identity["task_identity_sha256"],
        "production_eligible": False,
        "diagnostic_only": True,
        "comparison": identity["comparison"],
        "task_identity": identity,
        "git": {
            "commit": source_manifest.get("git_commit"),
            "source_manifest": source_manifest,
            "working_tree_clean": source_manifest.get("working_tree_clean", False),
        },
        "resolved_configuration": {
            "sha256": timing.get("resolved_config_sha256"),
            "architecture_sha256": timing.get("architecture_sha256"),
            "architecture_identity": timing.get("architecture_identity"),
        },
        "gpu": gpu_identity,
        "telemetry": {
            "measured_phase": measured_telemetry,
            "profile_phase": profile_telemetry,
            "measured_phase_gpu_utilization_only": True,
            "raw_measured_csv": str(output / "nvidia_smi_unprofiled_1s.csv"),
            "raw_profile_csv": str(output / "nvidia_smi_profile_1s.csv"),
            "sampling_interval_seconds": 1,
        },
        "timing": {
            "unprofiled_measured_window": timing,
            "phase_receipts": phases,
            "allocation_wall_elapsed_seconds": allocation_elapsed_seconds,
            "measured_steps_per_second": timing.get("steps_per_second"),
            "startup_hard_init_validation_final_eval_checkpoint_excluded": True,
        },
        "sm_occupancy": {
            **smo,
            "metric": NCU_SMO_METRIC,
            "measurement_window": "verified optimizer-step range from CUDA profiler boundaries",
            "gpu_utilization_as_smo": False,
        },
        "rgu": rgu_accounting(
            gpu_count=1,
            measured_elapsed_seconds=measured_elapsed,
            allocation_elapsed_seconds=allocation_elapsed_seconds,
        ),
        "recovery": {
            "mode": "restart-progress",
            "progress_marker": str(output / "progress.json"),
            "attempt_receipt": str(output / f"attempt_{attempt:04d}.json"),
            "atomic_receipts": True,
            "signal": "SIGTERM",
            "requeue": True,
            "statefulness": "stateless benchmark; phase rerun is idempotent",
            "continuation_validated": continuation_validated,
            "continuation_receipt": continuation_receipt,
        },
        "storage": storage or {
            "active_tier": "SCRATCH",
            "retention": "disposable diagnostic pilot outputs; retain only receipts while quota permits",
            "permanent_copy": "not_applicable: no model checkpoint is claimed",
        },
        "skills_handoff": {
            "experiment": {"status": "recorded", "run_type": "GPU sanity pilot"},
            "checkpointing": {
                "status": "pending_real_resume_test",
                "recovery_mode": "restart-progress",
                "stateful_model_checkpoint": "not_applicable: benchmark is stateless",
            },
            "slurm_mila": {
                "status": "pending_adjudication",
                "three_hour_limit": True,
                "gpu_gate": "diagnostic pilot; production eligibility disabled",
            },
            "experiment_workflow": {
                "status": "pending_adjudication",
                "identity_and_continuation_validated": continuation_validated,
            },
        },
        "artifacts": {
            "task_identity": str(output / "task_identity.json"),
            "source_manifest": str(output / "source_manifest.sha256"),
            "gpu_identity": str(output / "gpu_identity.json"),
            "ncu_csv": str(output / "ncu_smo.csv"),
        },
    }
