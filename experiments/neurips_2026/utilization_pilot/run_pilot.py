"""Run the stateless exact-shape base/candidate utilization pilot.

The pilot has three durable phases: an unprofiled warmup, an unprofiled
timed window, and a short Nsight Compute window.  Each phase has an atomic
receipt and is safe to re-run after a TERM/requeue.  A final receipt is always
diagnostic-only; it cannot authorize production work.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .pilot import (
    NCU_SMO_METRIC,
    SCHEMA_VERSION,
    MetricUnavailable,
    atomic_write_json,
    atomic_write_text,
    build_source_manifest,
    exact_task_identity,
    parse_ncu_smo_csv,
    phase_telemetry,
    validate_task_identity,
)
from .runtime import (
    PROFILE_MEASURE_STEPS,
    TIMED_STEPS,
    WARMUP_STEPS,
    TOTAL_STEPS,
    phase_receipt,
    profile_command,
    start_telemetry,
    stop_telemetry,
    train_args,
)
from .receipt import final_receipt
from .validation import (
    allocation_segment as _allocation_segment,
    existing_final as _existing_final,
    newest_valid_phase_receipt as _newest_valid_phase_receipt,
    read_json as _json,
    storage_metadata as _storage_metadata,
    validate_progress_hashes as _validate_progress_hashes,
    verified_timing as _verified_timing,
)


TERM_EXIT_CODE = 75

_TERM_REQUESTED = False
_ACTIVE_CHILD: subprocess.Popen[str] | None = None


def _handle_term(signum: int, _frame: Any) -> None:
    del signum
    global _TERM_REQUESTED
    _TERM_REQUESTED = True
    child = _ACTIVE_CHILD
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)

def _gpu_identity() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 6:
            continue
        rows.append(
            dict(
                zip(
                    ("index", "uuid", "name", "memory_total_mib", "driver", "compute_cap"),
                    values,
                )
            )
        )
    if len(rows) != 1:
        raise RuntimeError(f"pilot requires exactly one visible GPU, found {len(rows)}")
    if rows[0]["uuid"] == "" or not rows[0]["uuid"].startswith("GPU-"):
        raise RuntimeError("nvidia-smi did not report a GPU UUID")
    if "RTX 8000" not in rows[0]["name"] and "RTX8000" not in rows[0]["name"]:
        raise RuntimeError(f"pilot requires one RTX 8000, found {rows[0]['name']}")
    return rows

def _run_child(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int, float, float, float]:
    global _ACTIVE_CHILD
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    wall_started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        _ACTIVE_CHILD = child
        while child.poll() is None:
            time.sleep(0.2)
        return_code = child.returncode
    _ACTIVE_CHILD = None
    wall_ended = time.time()
    return return_code, time.monotonic() - started, wall_started, wall_ended

def _identity_hash(identity: dict[str, Any]) -> str:
    return str(identity["task_identity_sha256"])

def run_pilot(args: argparse.Namespace) -> int:
    global _TERM_REQUESTED
    root = Path(args.repo_root).resolve() if args.repo_root else _git_root()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = exact_task_identity(args.label, args.variant)
    validate_task_identity(identity)
    identity_hash = _identity_hash(identity)
    final_path = output / "final.json"
    source_manifest = build_source_manifest(root, output / "source_manifest.sha256")
    existing = _existing_final(final_path, identity_hash, source_manifest)
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0 if existing.get("status") == "complete" else 5

    attempt = int(args.restart_count)
    allocation_started_unix = time.time()
    storage = _storage_metadata(output)
    progress_path = output / "progress.json"
    progress = _json(progress_path) or {}
    _validate_progress_hashes(progress)
    completed = set(progress.get("completed_phases", []))
    if progress.get("identity_sha256") not in (None, identity_hash):
        raise RuntimeError("progress marker belongs to a different task identity")
    if progress.get("source_manifest_sha256") not in (None, source_manifest["sha256"]):
        raise RuntimeError("progress marker belongs to a different committed source")
    identity_path = output / "task_identity.json"
    atomic_write_json(identity_path, identity)
    atomic_write_json(output / "gpu_identity.json", {"gpus": _gpu_identity()})
    atomic_write_text(output / "identity.sha256", identity_hash + "\n")

    environment = os.environ.copy()
    environment["SKAE_UTILIZATION_PILOT_LABEL"] = args.label
    environment["SKAE_UTILIZATION_PILOT_VARIANT"] = args.variant
    environment["SKAE_UTILIZATION_PILOT_TASK_IDENTITY_SHA256"] = identity_hash

    attempt_path = output / f"attempt_{attempt:04d}.json"
    attempt_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "utilization_pilot_attempt",
        "attempt": attempt,
        "status": "running",
        "identity_sha256": identity_hash,
        "task_identity": identity,
        "source_manifest": source_manifest,
        "started_unix": time.time(),
        "recovery": {
            "mode": "restart-progress",
            "progress_marker": str(progress_path),
            "signal": "SIGTERM",
            "requeue": True,
        },
    }
    atomic_write_json(attempt_path, attempt_payload)
    _write_progress(
        progress_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "identity_sha256": identity_hash,
            "source_manifest_sha256": source_manifest["sha256"],
            "completed_phases": sorted(completed),
            "attempt": attempt,
            "resolved_config_sha256": progress.get("resolved_config_sha256"),
            "architecture_sha256": progress.get("architecture_sha256"),
        },
    )

    allocation_segments_path = output / "allocation_segments.json"
    telemetry_process: subprocess.Popen[str] | None = None
    phase_records: dict[str, Any] = {}

    current_phase = "starting"

    def run_phase(
        phase: str,
        steps: int,
        command: list[str],
        log_name: str,
        timing_path: Path,
        profiler_active: bool,
    ) -> None:
        nonlocal current_phase
        current_phase = phase
        existing = _newest_valid_phase_receipt(
            output, phase=phase, identity_hash=identity_hash,
            command=command, steps=steps,
        )
        if phase in completed and existing is not None:
            timing = _verified_timing(
                timing_path,
                warmup_steps=WARMUP_STEPS,
                profiler_active=profiler_active,
                measured_steps=steps,
            )
            for field in ("resolved_config_sha256", "architecture_sha256"):
                previous = progress.get(field)
                if previous not in (None, timing[field]):
                    raise RuntimeError(f"continuation {field} changed")
            phase_records[phase] = existing
            return
        completed.discard(phase)
        if _TERM_REQUESTED:
            raise InterruptedError
        stdout_path = output / f"{log_name}.out"
        stderr_path = output / f"{log_name}.err"
        telemetry_path = output / f"nvidia_smi_{phase}_1s.csv"
        telemetry_process = start_telemetry(telemetry_path)
        try:
            return_code, elapsed, _, _ = _run_child(
                command, cwd=root, environment=environment,
                stdout_path=stdout_path, stderr_path=stderr_path,
            )
        finally:
            stop_telemetry(telemetry_process)
        if _TERM_REQUESTED:
            raise InterruptedError
        try:
            timing = _verified_timing(
                timing_path,
                warmup_steps=WARMUP_STEPS,
                profiler_active=profiler_active,
                measured_steps=steps,
            )
        except MetricUnavailable as exc:
            if return_code != 0:
                raise RuntimeError(
                    f"{phase} child exited with status {return_code}; "
                    f"timing was unavailable: {exc}"
                ) from exc
            raise
        for field in ("resolved_config_sha256", "architecture_sha256"):
            previous = progress.get(field)
            if previous not in (None, timing[field]):
                raise RuntimeError(f"continuation {field} changed")
        telemetry = phase_telemetry(
            telemetry_path,
            phase=phase,
            child_return_code=return_code,
            start_unix=timing.get("wall_start_unix"),
            end_unix=timing.get("wall_end_unix"),
        )
        artifact_paths = {
            "stdout": stdout_path,
            "stderr": stderr_path,
            "telemetry": telemetry_path,
            "timing": timing_path,
            "resolved_config": Path(str(timing["resolved_config_path"])),
        }
        if phase == "profile":
            artifact_paths["ncu"] = output / "ncu_smo.csv"
        receipt_path = output / f"{phase}_attempt_{attempt:04d}.json"
        record = phase_receipt(
            output=receipt_path,
            phase=phase,
            attempt=attempt,
            status="complete" if return_code == 0 else "failed",
            identity_hash=identity_hash,
            command=command,
            elapsed_seconds=elapsed,
            steps=steps,
            return_code=return_code,
            artifact_paths=artifact_paths,
            extra={
                "timing": timing,
                "telemetry": telemetry,
                "receipt_path": str(receipt_path),
            },
        )
        phase_records[phase] = record
        if return_code != 0:
            if phase == "profile":
                raise MetricUnavailable(
                    f"Nsight Compute command failed with exit code {return_code}"
                )
            raise RuntimeError(f"{phase} command failed with exit code {return_code}")
        completed.add(phase)
        _write_progress(
            progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "identity_sha256": identity_hash,
                "source_manifest_sha256": source_manifest["sha256"],
                "completed_phases": sorted(completed),
                "last_phase": phase,
                "attempt": attempt,
                "phase_receipts": {
                    name: item.get("receipt_path") for name, item in phase_records.items()
                    if isinstance(item, dict) and item.get("receipt_path")
                },
                "resolved_config_sha256": timing["resolved_config_sha256"],
                "architecture_sha256": timing["architecture_sha256"],
            },
        )

    try:
        run_phase(
            "unprofiled", TIMED_STEPS,
            train_args(
                TOTAL_STEPS, output / "unprofiled_run",
                pilot_warmup_steps=WARMUP_STEPS,
                pilot_measure_steps=TIMED_STEPS,
                pilot_timing_path=output / "unprofiled_timing.json",
            ),
            f"unprofiled_attempt_{attempt:04d}",
            output / "unprofiled_timing.json",
            False,
        )
        if shutil.which("ncu") is None:
            raise MetricUnavailable("ncu is not available on this allocation")
        run_phase(
            "profile", PROFILE_MEASURE_STEPS,
            profile_command(
                output / "profile_run", output / "ncu_smo.csv",
                output / "profile_timing.json",
            ),
            f"profile_attempt_{attempt:04d}",
            output / "profile_timing.json",
            True,
        )
    except InterruptedError:
        allocation_elapsed = _allocation_segment(
            allocation_segments_path, attempt, allocation_started_unix
        )
        attempt_payload.update(
            {
                "status": "interrupted",
                "interrupted_phase": current_phase,
                "interrupted_unix": time.time(),
                "allocation_elapsed_seconds": allocation_elapsed,
            }
        )
        atomic_write_json(attempt_path, attempt_payload)
        _write_progress(
            progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "interrupted",
                "identity_sha256": identity_hash,
                "source_manifest_sha256": source_manifest["sha256"],
                "completed_phases": sorted(completed),
                "attempt": attempt,
                "resolved_config_sha256": progress.get("resolved_config_sha256"),
                "architecture_sha256": progress.get("architecture_sha256"),
            },
        )
        return TERM_EXIT_CODE
    except MetricUnavailable as exc:
        stop_telemetry(telemetry_process)
        telemetry_process = None
        phase_records["allocation_elapsed_seconds"] = _allocation_segment(
            allocation_segments_path, attempt, allocation_started_unix
        )
        phase_records["storage"] = storage
        final = final_receipt(
            output=output,
            status="ambiguous",
            identity=identity,
            source_manifest=source_manifest,
            gpu_identity=_json(output / "gpu_identity.json") or {},
            measured_telemetry=phase_records.get("unprofiled", {}).get("telemetry", {}),
            profile_telemetry=phase_records.get("profile", {}).get("telemetry", {}),
            phases=phase_records,
            timing=phase_records.get("unprofiled", {}).get("timing", {}),
            smo={"status": "missing", "error": str(exc), "metric": NCU_SMO_METRIC},
            allocation_elapsed_seconds=float(
                phase_records.get("allocation_elapsed_seconds", 0.0)
            ),
            attempt=attempt,
            storage=phase_records.get("storage"),
        )
        atomic_write_json(final_path, final)
        attempt_payload.update({"status": "ambiguous", "error": str(exc), "ended_unix": time.time()})
        atomic_write_json(attempt_path, attempt_payload)
        return 5
    except Exception as exc:
        stop_telemetry(telemetry_process)
        telemetry_process = None
        _allocation_segment(allocation_segments_path, attempt, allocation_started_unix)
        attempt_payload.update({"status": "failed", "error": str(exc), "ended_unix": time.time()})
        atomic_write_json(attempt_path, attempt_payload)
        raise
    finally:
        stop_telemetry(telemetry_process)

    try:
        smo = parse_ncu_smo_csv(output / "ncu_smo.csv")
    except MetricUnavailable as exc:
        smo = {"status": "missing", "error": str(exc), "metric": NCU_SMO_METRIC}
        status = "ambiguous"
    else:
        status = "complete"
    phase_records["allocation_elapsed_seconds"] = _allocation_segment(
        allocation_segments_path, attempt, allocation_started_unix
    )
    phase_records["storage"] = storage
    final = final_receipt(
        output=output,
        status=status,
        identity=identity,
        source_manifest=source_manifest,
        gpu_identity=_json(output / "gpu_identity.json") or {},
        measured_telemetry=phase_records.get("unprofiled", {}).get("telemetry", {}),
        profile_telemetry=phase_records.get("profile", {}).get("telemetry", {}),
        phases=phase_records,
        timing=phase_records.get("unprofiled", {}).get("timing", {}),
        smo=smo,
        allocation_elapsed_seconds=float(
            phase_records.get("allocation_elapsed_seconds", 0.0)
        ),
        attempt=attempt,
        storage=phase_records.get("storage"),
    )
    atomic_write_json(final_path, final)
    attempt_payload.update({"status": status, "ended_unix": time.time()})
    atomic_write_json(attempt_path, attempt_payload)
    _write_progress(
        progress_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "identity_sha256": identity_hash,
            "source_manifest_sha256": source_manifest["sha256"],
            "completed_phases": sorted(completed | {"profile"}),
            "attempt": attempt,
            "resolved_config_sha256": phase_records.get("unprofiled", {}).get("timing", {}).get("resolved_config_sha256"),
            "architecture_sha256": phase_records.get("unprofiled", {}).get("timing", {}).get("architecture_sha256"),
        },
    )
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if status == "complete" else 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--label", default=os.environ.get("PILOT_LABEL", "base"))
    parser.add_argument("--variant", default=os.environ.get("PILOT_VARIANT", "base"))
    parser.add_argument(
        "--restart-count",
        type=int,
        default=int(os.environ.get("SLURM_RESTART_COUNT", "0")),
    )
    return parser


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)
    try:
        return run_pilot(build_parser().parse_args())
    except Exception as exc:
        print(f"utilization pilot failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
