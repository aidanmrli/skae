"""Complete-state checkpoint contracts for the training runner."""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import random
import signal
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from skae.training.checkpoint_store import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointManager,
)


CHECKPOINT_EXIT_CODE = 75


class CheckpointSignalExit(SystemExit):
    """Exit status used after a signal-safe checkpoint has been committed."""

    def __init__(self) -> None:
        super().__init__(CHECKPOINT_EXIT_CODE)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_without_total_steps(config: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    train = result.get("TRAIN")
    if isinstance(train, dict):
        # Extending a resumed run is intentional; every other math-affecting
        # configuration value remains part of the resume contract.
        train["NUM_STEPS"] = None
    return result


def source_identity(
    source_root: Optional[Path] = None,
    *,
    require_clean: bool = False,
) -> Dict[str, Any]:
    """Return a committed source identity, failing closed when requested."""
    root = Path(source_root or Path.cwd()).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        if require_clean:
            raise CheckpointError("managed checkpointing requires a Git source identity") from exc
        return {"git_commit": None, "git_dirty": None, "git_status_sha256": None}
    if require_clean and status.strip():
        raise CheckpointError(
            "managed checkpointing requires a clean committed source tree; "
            "commit or remove tracked and untracked changes first"
        )
    return {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def make_run_identity(
    config: Dict[str, Any],
    device: str,
    *,
    batch_count: int,
    logger_history: bool,
    source_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build stable and exact-continuation compatibility metadata."""
    return {
        "config": copy.deepcopy(config),
        "config_hash": _canonical_hash(config),
        "resume_config_hash": _canonical_hash(_config_without_total_steps(config)),
        "device": str(device),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "batch_count": int(batch_count),
        "logger_history": bool(logger_history),
        "source": source_identity(source_root, require_clean=True),
    }


def validate_run_identity(expected: Dict[str, Any], actual: Dict[str, Any]) -> None:
    """Reject a resume whose math, data order, source, or logger changed."""
    if not isinstance(actual, dict):
        raise CheckpointError("checkpoint has no run identity")
    for key in (
        "resume_config_hash",
        "device",
        "cuda_device_count",
        "batch_count",
        "logger_history",
    ):
        if actual.get(key) != expected.get(key):
            raise CheckpointError(
                f"incompatible checkpoint identity for {key}: "
                f"saved={actual.get(key)!r}, current={expected.get(key)!r}"
            )
    saved_source = actual.get("source")
    current_source = expected.get("source")
    if not isinstance(saved_source, dict) or not isinstance(current_source, dict):
        raise CheckpointError("checkpoint has no complete source identity")
    for key in ("git_commit", "git_dirty", "git_status_sha256"):
        if saved_source.get(key) != current_source.get(key):
            raise CheckpointError(f"checkpoint source identity differs for {key}")


def capture_rng_state(
    generators: Iterable[torch.Generator],
    validation_generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Capture Python, NumPy, CPU/CUDA, and every data-stream generator."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cpu_device": "cpu",
        "torch_cuda": [],
        "torch_cuda_devices": [],
        "batch_generators": [generator.get_state() for generator in generators],
        "batch_generator_devices": [str(generator.device) for generator in generators],
        "validation_generator": (
            None if validation_generator is None else validation_generator.get_state()
        ),
        "validation_generator_device": (
            None if validation_generator is None else str(validation_generator.device)
        ),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count())
        ]
        state["torch_cuda_devices"] = [
            f"cuda:{index}" for index in range(torch.cuda.device_count())
        ]
    return state


def restore_rng_state(
    state: Dict[str, Any],
    generators: Iterable[torch.Generator],
    validation_generator: Optional[torch.Generator] = None,
) -> None:
    """Restore RNG state and reject a changed generator/device layout."""
    if not isinstance(state, dict):
        raise CheckpointError("checkpoint has no RNG state")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
    except Exception as exc:
        raise CheckpointError("checkpoint RNG state is malformed") from exc
    cuda_states = list(state.get("torch_cuda") or [])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise CheckpointError("checkpoint CUDA RNG layout differs from current allocation")
        for index, cuda_state in enumerate(cuda_states):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=index)
    current_generators = list(generators)
    saved_generators = list(state.get("batch_generators") or [])
    saved_devices = list(state.get("batch_generator_devices") or [])
    if len(current_generators) != len(saved_generators):
        raise CheckpointError("checkpoint batch-generator count differs")
    if saved_devices and len(saved_devices) != len(saved_generators):
        raise CheckpointError("checkpoint batch-generator device metadata is malformed")
    for index, (generator, generator_state) in enumerate(zip(current_generators, saved_generators)):
        if saved_devices and str(generator.device) != saved_devices[index]:
            raise CheckpointError("checkpoint batch-generator device layout differs")
        generator.set_state(generator_state.cpu())
    validation_state = state.get("validation_generator")
    if validation_generator is not None and validation_state is not None:
        saved_device = state.get("validation_generator_device")
        if saved_device and str(validation_generator.device) != saved_device:
            raise CheckpointError("checkpoint validation-generator device differs")
        validation_generator.set_state(validation_state.cpu())


class SignalStopper:
    """Record termination signals and defer saving to a safe step boundary."""

    def __init__(self) -> None:
        self.requested = False
        self.signal_number: Optional[int] = None
        self._previous: Dict[int, Any] = {}
        self._installed = False

    def _handle(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.signal_number = int(signum)

    def install(self) -> None:
        if self._installed:
            return
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        self._installed = True
        atexit.register(self.restore)

    def restore(self) -> None:
        if not self._installed:
            return
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous.clear()
        self._installed = False
