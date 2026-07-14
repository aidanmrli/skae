"""Atomic artifact I/O and run discovery for staged ``F_abs`` training."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from skae.config import Config
from tools.staged_fabs_model import SourceTargetLocalMapBundle


def _save_checkpoint(
    path: Path,
    *,
    stage: str,
    next_step: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    bundle: Optional[SourceTargetLocalMapBundle],
    local_optimizer: Optional[torch.optim.Optimizer],
    best_eval_final_error: float,
    metrics: Dict[str, float],
    cfg: Config,
    route_metadata: Optional[Dict[str, object]] = None,
    route_codebook: Optional[Dict[str, object]] = None,
    target_centers: Optional[Dict[object, object]] = None,
    support_batches: Optional[Sequence[torch.Tensor]] = None,
    training_generators: Optional[Sequence[torch.Generator]] = None,
    include_optimizer_state: bool = False,
) -> None:
    payload: Dict[str, object] = {
        "checkpoint_schema_version": 3,
        "stage": stage,
        "next_step": int(next_step),
        "step": int(next_step) - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if include_optimizer_state and optimizer is not None
            else None
        ),
        "local_bundle_state_dict": bundle.state_dict() if bundle is not None else None,
        "local_optimizer_state_dict": (
            local_optimizer.state_dict()
            if include_optimizer_state and local_optimizer is not None
            else None
        ),
        "best_eval_final_error": float(best_eval_final_error),
        "metrics": dict(metrics),
        "config": cfg.to_dict(),
        "route_metadata": route_metadata or {},
        "route_codebook": route_codebook,
        "target_centers": target_centers,
        "support_batches": [batch.cpu() for batch in support_batches]
        if support_batches
        else [],
        "training_generator_states": [
            generator.get_state().cpu() for generator in training_generators
        ]
        if training_generators is not None
        else None,
        "torch_cpu_rng_state": torch.random.get_rng_state().cpu(),
        "torch_cuda_rng_states": [
            state.cpu() for state in torch.cuda.get_rng_state_all()
        ]
        if torch.cuda.is_available()
        else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_training_rng_states(
    payload: Dict[str, object],
    training_generators: Sequence[torch.Generator],
) -> bool:
    """Restore stochastic streams from a schema-3 checkpoint when present."""

    saved_generators = payload.get("training_generator_states")
    cpu_state = payload.get("torch_cpu_rng_state")
    if saved_generators is None or cpu_state is None:
        if int(payload.get("checkpoint_schema_version", 0)) >= 3:
            raise ValueError("Schema-3 checkpoint is missing RNG state.")
        return False
    if len(saved_generators) != len(training_generators):
        raise ValueError(
            "Checkpoint training-generator count does not match this run."
        )
    for generator, state in zip(training_generators, saved_generators):
        generator.set_state(state.cpu())
    torch.random.set_rng_state(cpu_state.cpu())
    cuda_states = payload.get("torch_cuda_rng_states", [])
    if torch.cuda.is_available() and cuda_states:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    return True


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_torch(path: Path, *, map_location: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _log_phase(run_dir: Path, phase: str, *, device: str, **payload: object) -> None:
    record: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        **payload,
    }
    if str(device) == "cuda" and torch.cuda.is_available():
        record["cuda_device_name"] = torch.cuda.get_device_name(0)
        record["cuda_memory_allocated_mb"] = round(
            torch.cuda.memory_allocated() / 1_000_000, 3
        )
        record["cuda_max_memory_allocated_mb"] = round(
            torch.cuda.max_memory_allocated() / 1_000_000, 3
        )
    path = run_dir / "phase_status.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    details = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"[phase] {phase} {details}".rstrip(), flush=True)


def _find_resume_run(seed_dir: Path) -> Optional[Path]:
    candidates = [
        path
        for path in seed_dir.glob("20*")
        if path.is_dir()
        and ((path / "last.pt").is_file() or (path / "checkpoint.pt").is_file())
        and not (path / "evaluation_results_best.json").is_file()
    ]
    return sorted(candidates, key=lambda path: (path.name, str(path)))[-1] if candidates else None


def _find_completed_run(seed_dir: Path) -> Optional[Path]:
    candidates = [
        path
        for path in seed_dir.glob("20*")
        if path.is_dir()
        and (path / "checkpoint.pt").is_file()
        and (path / "evaluation_results_best.json").is_file()
    ]
    return sorted(candidates, key=lambda path: (path.name, str(path)))[-1] if candidates else None
