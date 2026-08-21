"""Small runtime helpers kept separate from pilot orchestration."""

from __future__ import annotations

import subprocess
import time
import os
from pathlib import Path
from typing import Any, Mapping

from .pilot import NCU_SMO_METRIC, SCHEMA_VERSION, atomic_write_json, sha256_file


WARMUP_STEPS = 64
DEFAULT_TIMED_STEPS = 256
MIN_TIMED_STEPS = 256
MAX_TIMED_STEPS = 8192
TIMED_STEPS = DEFAULT_TIMED_STEPS
TOTAL_STEPS = WARMUP_STEPS + TIMED_STEPS
PROFILE_MEASURE_STEPS = 2
PROFILE_TOTAL_STEPS = WARMUP_STEPS + PROFILE_MEASURE_STEPS


def resolve_measure_steps(
    cli_value: int | None = None, *, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Resolve the operational unprofiled window without changing the task shape."""

    env = os.environ if environment is None else environment
    if cli_value is not None:
        raw_value: object = cli_value
        source = "cli"
    elif "PILOT_MEASURE_STEPS" in env:
        raw_value = env["PILOT_MEASURE_STEPS"]
        source = "env:PILOT_MEASURE_STEPS"
    else:
        raw_value = DEFAULT_TIMED_STEPS
        source = "default"
    try:
        steps = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("PILOT_MEASURE_STEPS must be an integer") from exc
    if not MIN_TIMED_STEPS <= steps <= MAX_TIMED_STEPS:
        raise ValueError(
            "PILOT_MEASURE_STEPS must be between "
            f"{MIN_TIMED_STEPS} and {MAX_TIMED_STEPS}"
        )
    return {"requested": steps, "actual": steps, "source": source}


def measurement_window_provenance(selection: Mapping[str, Any]) -> dict[str, Any]:
    actual = int(selection["actual"])
    return {
        "requested_unprofiled_measured_steps": int(selection["requested"]),
        "actual_unprofiled_measured_steps": actual,
        "unprofiled_step_start": WARMUP_STEPS,
        "unprofiled_step_end_exclusive": WARMUP_STEPS + actual,
        "profile_measured_steps": PROFILE_MEASURE_STEPS,
        "profile_step_start": WARMUP_STEPS,
        "profile_step_end_exclusive": WARMUP_STEPS + PROFILE_MEASURE_STEPS,
        "requested_from": str(selection["source"]),
    }


def train_args(
    num_steps: int,
    log_dir: Path,
    *,
    pilot_warmup_steps: int = 0,
    pilot_measure_steps: int = 0,
    pilot_profile: bool = False,
    pilot_timing_path: Path | None = None,
) -> list[str]:
    """Return the frozen scientific command; only phase steps/log vary."""

    command = [
        "uv", "run", "skae-train", "--config", "generic_sparse",
        "--env", "gated_transfer_linear", "--env_dt", "0.04",
        "--num_steps", str(num_steps), "--batch_size", "256",
        "--target_size", "256", "--sequence_length", "8",
        "--res_coeff", "1.0", "--reconst_coeff", "0.03",
        "--pred_coeff", "1.0", "--sparsity_coeff", "0.0",
        "--k_structure", "dense", "--lr", "5e-5",
        "--k_matrix_lr", "5e-6", "--weight_decay", "1e-4",
        "--hard_init_oversample", "true", "--hard_init_fraction", "0.5",
        "--hard_init_pool_size", "1024", "--hard_init_num_candidates", "4096",
        "--hard_init_probe_steps", "32", "--hard_init_num_perturbations", "4",
        "--hard_init_perturb_scale", "0.04", "--hard_init_transient_window", "8",
        "--hard_init_transient_weight", "0.5", "--hard_init_jitter_scale", "0.25",
        "--seed", "4", "--device", "cuda", "--log_dir", str(log_dir),
        "--skip_eval", "--save_last_checkpoint",
    ]
    if pilot_measure_steps:
        command.extend([
            "--pilot_warmup_steps", str(pilot_warmup_steps),
            "--pilot_measure_steps", str(pilot_measure_steps),
        ])
        if pilot_profile:
            command.append("--pilot_profile")
        if pilot_timing_path is None:
            raise ValueError("pilot timing path is required for measured commands")
        command.extend(["--pilot_timing_path", str(pilot_timing_path)])
    return command


def profile_command(log_dir: Path, ncu_output: Path, timing_path: Path) -> list[str]:
    return [
        "ncu", "--target-processes", "all", "--profile-from-start", "off",
        "--csv", "--metrics", NCU_SMO_METRIC, "--log-file", str(ncu_output),
        *train_args(
            PROFILE_TOTAL_STEPS,
            log_dir,
            pilot_warmup_steps=WARMUP_STEPS,
            pilot_measure_steps=PROFILE_MEASURE_STEPS,
            pilot_profile=True,
            pilot_timing_path=timing_path,
        ),
    ]


def start_telemetry(path: Path) -> subprocess.Popen[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit",
            "--format=csv,nounits", "-l", "1",
        ],
        stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    process._pilot_output_handle = handle  # type: ignore[attr-defined]
    return process


def stop_telemetry(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    handle = getattr(process, "_pilot_output_handle", None)
    if handle is not None:
        handle.close()


def phase_receipt(
    *, output: Path, phase: str, attempt: int, status: str,
    identity_hash: str, command: list[str], elapsed_seconds: float | None = None,
    steps: int | None = None, return_code: int | None = None,
    artifact_paths: dict[str, Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "receipt_type": "utilization_pilot_phase",
        "phase": phase, "attempt": attempt, "status": status,
        "identity_sha256": identity_hash, "command": command,
        "created_unix": time.time(),
    }
    if elapsed_seconds is not None:
        receipt["elapsed_seconds"] = elapsed_seconds
    if steps is not None:
        receipt["steps"] = steps
        if elapsed_seconds and elapsed_seconds > 0:
            receipt["steps_per_second"] = steps / elapsed_seconds
    if return_code is not None:
        receipt["return_code"] = return_code
    if artifact_paths:
        receipt["artifacts"] = {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
            if path.is_file()
        }
    if extra:
        receipt.update(extra)
    atomic_write_json(output, receipt)
    return receipt
