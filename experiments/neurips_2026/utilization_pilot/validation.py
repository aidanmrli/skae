"""Durable receipt, timing, and storage validation for the utilization pilot."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .pilot import (
    SCHEMA_VERSION,
    MetricUnavailable,
    canonical_json,
    sha256_bytes,
    sha256_file,
    validate_task_identity,
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def existing_final(
    path: Path, identity_hash: str, source_manifest: dict[str, Any]
) -> dict[str, Any] | None:
    final = read_json(path)
    if final is None or final.get("identity_sha256") != identity_hash:
        return None
    try:
        validate_task_identity(final.get("task_identity", {}))
    except (AttributeError, TypeError, ValueError):
        return None
    git = final.get("git")
    saved_manifest = git.get("source_manifest") if isinstance(git, dict) else None
    saved_commit = git.get("commit") if isinstance(git, dict) else None
    saved_clean = git.get("working_tree_clean") if isinstance(git, dict) else None
    if (
        not isinstance(saved_manifest, dict)
        or saved_manifest.get("sha256") != source_manifest.get("sha256")
        or saved_commit != source_manifest.get("git_commit")
        or saved_clean is not True
    ):
        return None
    timing = final.get("timing")
    phase_receipts = timing.get("phase_receipts") if isinstance(timing, dict) else None
    if not isinstance(phase_receipts, dict):
        return None
    if final.get("status") == "complete" and not {"unprofiled", "profile"}.issubset(
        phase_receipts
    ):
        return None
    for phase, record in phase_receipts.items():
        if phase in {"allocation_elapsed_seconds", "storage"}:
            continue
        if not isinstance(record, dict):
            return None
        receipt_path = Path(str(record.get("receipt_path", "")))
        receipt = valid_phase_receipt(
            receipt_path,
            phase=str(record.get("phase", phase)),
            identity_hash=identity_hash,
            command=record.get("command", []),
            steps=record.get("steps"),
        )
        if receipt is None or record.get("artifacts") != receipt.get("artifacts"):
            return None
    if final.get("status") in {"complete", "ambiguous"}:
        return final
    return None


def validate_progress_hashes(progress: dict[str, Any]) -> None:
    values = [progress.get("resolved_config_sha256"), progress.get("architecture_sha256")]
    if (values[0] is None) != (values[1] is None):
        raise RuntimeError("progress marker has only one continuation identity hash")
    for value in values:
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError("progress marker has an invalid continuation identity hash")


def storage_metadata(output: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(output)
    minimum_free = 1 << 30
    if usage.free < minimum_free:
        raise RuntimeError(f"insufficient scratch free space: {usage.free} bytes")
    return {
        "active_tier": "SCRATCH",
        "free_bytes_before": usage.free,
        "minimum_free_bytes": minimum_free,
        "retention": "disposable diagnostic pilot outputs; retain receipts only while quota permits",
        "permanent_copy": "not_applicable: no model checkpoint is claimed",
        "checkpoint_claim": False,
    }


def allocation_segment(path: Path, attempt: int, started_unix: float) -> float:
    current = read_json(path) or {"schema_version": 1, "segments": []}
    segments = [item for item in current.get("segments", []) if isinstance(item, dict)]
    if not any(int(item.get("attempt", -1)) == attempt for item in segments):
        ended = time.time()
        segments.append(
            {
                "attempt": attempt,
                "started_unix": started_unix,
                "ended_unix": ended,
                "elapsed_seconds": max(0.0, ended - started_unix),
            }
        )
        from .pilot import atomic_write_json

        atomic_write_json(path, {"schema_version": 1, "segments": segments})
    return sum(float(item.get("elapsed_seconds", 0.0)) for item in segments)


def verified_timing(
    path: Path, *, warmup_steps: int, profiler_active: bool, measured_steps: int
) -> dict[str, Any]:
    timing = read_json(path)
    required = {
        "schema_version",
        "status",
        "warmup_steps",
        "measured_steps",
        "step_start",
        "step_end_exclusive",
        "elapsed_seconds",
        "steps_per_second",
        "cuda_synchronized_before_and_after",
        "profiler_range_active",
        "resolved_config_sha256",
        "resolved_config_path",
        "pilot_config_num_steps",
        "architecture_sha256",
    }
    if timing is None or not required.issubset(timing):
        raise MetricUnavailable(f"missing timing receipt: {path}")
    if (
        timing["schema_version"] != 1
        or timing["status"] != "complete"
        or timing["warmup_steps"] != warmup_steps
        or timing["measured_steps"] != measured_steps
        or timing["step_start"] != warmup_steps
        or timing["step_end_exclusive"] != warmup_steps + measured_steps
        or timing["cuda_synchronized_before_and_after"] is not True
        or timing["profiler_range_active"] is not profiler_active
        or timing["pilot_config_num_steps"] != warmup_steps + measured_steps
        or float(timing["elapsed_seconds"]) <= 0.0
        or float(timing["steps_per_second"]) <= 0.0
    ):
        raise MetricUnavailable(
            f"timing receipt does not match the exact measured range: {path}"
        )
    resolved_path = Path(str(timing["resolved_config_path"]))
    resolved_config = read_json(resolved_path)
    if resolved_config is None:
        raise MetricUnavailable(f"resolved config artifact is missing: {resolved_path}")
    if sha256_bytes(canonical_json(resolved_config).encode()) != timing[
        "resolved_config_sha256"
    ]:
        raise MetricUnavailable(f"resolved config hash mismatch: {resolved_path}")
    return timing


def valid_phase_receipt(
    path: Path,
    *,
    phase: str,
    identity_hash: str,
    command: list[str],
    steps: int,
) -> dict[str, Any] | None:
    receipt = read_json(path)
    if receipt is None:
        return None
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("receipt_type") != "utilization_pilot_phase"
        or receipt.get("phase") != phase
        or receipt.get("status") != "complete"
        or receipt.get("return_code") != 0
        or receipt.get("identity_sha256") != identity_hash
        or receipt.get("command") != command
        or receipt.get("steps") != steps
    ):
        return None
    artifacts = receipt.get("artifacts")
    expected_artifacts = {"stdout", "stderr", "telemetry", "timing", "resolved_config"}
    if phase == "profile":
        expected_artifacts.add("ncu")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        return None
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            return None
        artifact_path = Path(str(artifact.get("path", "")))
        try:
            valid_hash = artifact_path.is_file() and artifact.get("sha256") == sha256_file(
                artifact_path
            )
        except OSError:
            valid_hash = False
        if not valid_hash:
            return None
    timing = receipt.get("timing")
    if not isinstance(timing, dict):
        return None
    config_path = Path(str(timing.get("resolved_config_path", "")))
    config = read_json(config_path)
    if config is None or sha256_bytes(canonical_json(config).encode()) != timing.get(
        "resolved_config_sha256"
    ):
        return None
    return receipt


def newest_valid_phase_receipt(
    output: Path,
    *,
    phase: str,
    identity_hash: str,
    command: list[str],
    steps: int,
) -> dict[str, Any] | None:
    candidates = sorted(output.glob(f"{phase}_attempt_*.json"), reverse=True)
    for path in candidates:
        receipt = valid_phase_receipt(
            path,
            phase=phase,
            identity_hash=identity_hash,
            command=command,
            steps=steps,
        )
        if receipt is not None:
            return receipt
    return None
