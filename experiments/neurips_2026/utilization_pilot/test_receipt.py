"""Atomic restart-progress receipts for the CPU-only pilot test allocation."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .pilot import SCHEMA_VERSION, atomic_write_json, sha256_bytes


def _commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def write_receipt(output: Path, phase: str, attempt: int, repo_root: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    commit = _commit(repo_root)
    identity = sha256_bytes(f"{commit}:tests/test_utilization_pilot.py".encode())
    progress_path = output / "progress.json"
    attempt_path = output / f"attempt_{attempt:04d}.json"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "utilization_pilot_test",
        "status": phase,
        "attempt": attempt,
        "identity_sha256": identity,
        "git_commit": commit,
        "command": "uv run pytest tests/test_utilization_pilot.py",
        "recovery": {
            "mode": "restart-progress",
            "progress_marker": str(progress_path),
            "atomic_receipts": True,
            "signal": "SIGTERM",
            "requeue": True,
        },
        "updated_unix": time.time(),
    }
    atomic_write_json(attempt_path, payload)
    if phase == "complete":
        payload["production_eligible"] = False
        atomic_write_json(output / "final.json", payload)
    atomic_write_json(
        progress_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": phase,
            "identity_sha256": identity,
            "attempt": attempt,
            "updated_unix": payload["updated_unix"],
        },
    )
    return 0


def _identity(repo_root: Path) -> tuple[str, str]:
    commit = _commit(repo_root)
    return commit, sha256_bytes(f"{commit}:tests/test_utilization_pilot.py".encode())


def hold(output: Path, attempt: int, repo_root: Path, seconds: float) -> int:
    output.mkdir(parents=True, exist_ok=True)
    commit, identity = _identity(repo_root)
    interrupted = output / "hold_interrupted.json"

    def on_term(_signum: int, _frame: Any) -> None:
        atomic_write_json(
            interrupted,
            {
                "schema_version": SCHEMA_VERSION,
                "receipt_type": "utilization_pilot_test_hold",
                "status": "interrupted",
                "return_code": 75,
                "attempt": attempt,
                "git_commit": commit,
                "identity_sha256": identity,
                "next_state": "resume_validate",
                "updated_unix": time.time(),
            },
        )
        raise SystemExit(75)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    deadline = time.monotonic() + float(seconds)
    while time.monotonic() < deadline:
        time.sleep(0.2)
    raise RuntimeError("controlled hold expired without the forced TERM")


def validate_hold(output: Path, repo_root: Path) -> int:
    _, identity = _identity(repo_root)
    with (output / "hold_interrupted.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "interrupted"
        or payload.get("return_code") != 75
        or payload.get("identity_sha256") != identity
        or payload.get("next_state") != "resume_validate"
    ):
        raise RuntimeError("forced TERM receipt is not a valid continuation marker")
    return 0


def validate_final(output: Path, repo_root: Path) -> int:
    _, identity = _identity(repo_root)
    with (output / "final.json").open("r", encoding="utf-8") as handle:
        final = json.load(handle)
    with (output / "progress.json").open("r", encoding="utf-8") as handle:
        progress = json.load(handle)
    if (
        final.get("schema_version") != SCHEMA_VERSION
        or final.get("status") != "complete"
        or final.get("command") != "uv run pytest tests/test_utilization_pilot.py"
        or final.get("production_eligible") is not False
        or final.get("identity_sha256") != identity
        or progress.get("status") != "complete"
        or progress.get("identity_sha256") != identity
    ):
        raise RuntimeError("CPU pilot final receipt failed JSON continuation validation")
    return 0


def write_continuation(output: Path, attempt: int, repo_root: Path) -> int:
    validate_hold(output, repo_root)
    _, identity = _identity(repo_root)
    atomic_write_json(
        output / "continuation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "receipt_type": "utilization_pilot_test_continuation",
            "status": "passed",
            "attempt": attempt,
            "identity_sha256": identity,
            "validated_previous_state": "interrupted",
            "next_state": "pytest",
            "updated_unix": time.time(),
        },
    )
    return 0


def write_forced_term(output: Path, attempt: int, repo_root: Path) -> int:
    commit, identity = _identity(repo_root)
    atomic_write_json(
        output / "forced_term.json",
        {
            "schema_version": SCHEMA_VERSION,
            "receipt_type": "utilization_pilot_test_forced_term",
            "status": "passed",
            "attempt": attempt,
            "git_commit": commit,
            "identity_sha256": identity,
            "next_state": "requeued_resume_validate",
            "updated_unix": time.time(),
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--phase",
        choices=["started", "complete", "interrupted", "failed", "hold", "validate-hold", "validate-final", "resume", "forced-term"],
        required=True,
    )
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "hold":
        return hold(args.output_root, args.attempt, args.repo_root, args.seconds)
    if args.phase == "validate-hold":
        return validate_hold(args.output_root, args.repo_root)
    if args.phase == "validate-final":
        return validate_final(args.output_root, args.repo_root)
    if args.phase == "resume":
        return write_continuation(args.output_root, args.attempt, args.repo_root)
    if args.phase == "forced-term":
        return write_forced_term(args.output_root, args.attempt, args.repo_root)
    return write_receipt(args.output_root, args.phase, args.attempt, args.repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
