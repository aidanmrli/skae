"""Train one run of the paper.

    python -m paper train --benchmark multibasin --system gated_local_linear --model lista --seed 0

The run directory (``paper.config.run_dir``) holds

* ``config.json``    the run configuration;
* ``last.pt``        restart state, saved every ``CHECKPOINT_EVERY`` updates and after the last:
  model, optimizer, Dysts batch generators, torch RNG and the training log;
* ``train_log.csv``  loss terms every ``metrics_every`` updates;
* ``final.pt``       the weights after the last update (the reported model);
* ``metrics.json``   its held-out forecasts (:func:`paper.training.evaluation.run_evaluation`).

:func:`train` is the entry point; :func:`run_updates` is its training loop.

Rerunning the same command resumes from ``last.pt``; on the same hardware the final model
is bitwise the one an uninterrupted run gives.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Tuple

import torch

from paper import config
from paper.config import RunConfig, TrainConfig
from paper.evaluate import fallback_and_radius
from paper.io import atomic_path, write_csv
from paper.systems import make_system
from paper.training.data import Batches, train_normalization
from paper.training.evaluation import run_evaluation
from skae.models import KoopmanAutoencoder, training_loss

CHECKPOINT_EVERY = 1000


def lr_factor(train: TrainConfig, step: int) -> float:
    """Learning-rate multiplier at update ``step``.

    ``constant`` (multibasin): 1. ``cosine_tail`` (Dysts): 1 for the first
    ``1 - LR_TAIL_FRACTION`` (75%) of the updates, then a cosine decay to
    ``LR_FINAL_FACTOR`` (1%) at the last update.
    """
    if train.lr_schedule == "constant":
        return 1.0
    tail, final = config.LR_TAIL_FRACTION, config.LR_FINAL_FACTOR
    phase = max(0.0, (step - (1.0 - tail) * train.steps) / max(1.0, tail * train.steps - 1.0))
    if phase <= 0.0:
        return 1.0
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, phase)))


def build_optimizer(model: KoopmanAutoencoder, train: TrainConfig) -> torch.optim.AdamW:
    """AdamW with two parameter groups: encoder/decoder (``lr``, weight decay) and K (``koopman_lr``, no decay)."""
    koopman = {id(parameter) for parameter in model.koopman_parameters()}
    return torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in koopman], "lr": train.lr,
         "weight_decay": train.weight_decay},
        {"params": model.koopman_parameters(), "lr": train.koopman_lr, "weight_decay": 0.0},
    ])


def build_run(cfg: RunConfig, device: torch.device, data_dir: Path) -> Tuple[KoopmanAutoencoder, Batches]:
    """The seeded initial model and the training batches; Dysts models normalize inputs with train-only moments."""
    torch.manual_seed(cfg.seed)
    system = make_system(cfg.benchmark, cfg.system)
    model = KoopmanAutoencoder(cfg.model, system.dim).to(device)
    batches = Batches(system, cfg, device, data_dir)
    if cfg.model.input_normalization:
        model.set_input_normalization(*train_normalization(batches.trajectories))
    return model, batches


def save(payload: dict, path: Path) -> None:
    """``torch.save`` that never leaves a partly written file (saving to a handle keeps the bytes name-independent)."""
    with atomic_path(path) as temporary, open(temporary, "wb") as handle:
        torch.save(payload, handle)


def save_restart(path: Path, step: int, model, optimizer, batches: Batches, log: List[dict]) -> None:
    """Everything an exact resume after update ``step`` needs (the learning rate is a function of the step)."""
    rng = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state_all()
    save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
          "generators": batches.state(), "rng": rng, "log": log}, path)


def load_restart(path: Path, model, optimizer, batches: Batches) -> Tuple[int, List[dict]]:
    """Restore a restart state; return the next update and the training log so far."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    batches.load_state(state["generators"])
    torch.set_rng_state(state["rng"]["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state["rng"]:
        torch.cuda.set_rng_state_all(state["rng"]["torch_cuda"])
    return state["step"] + 1, state["log"]


def run_updates(cfg: RunConfig, run_dir: Path, device: torch.device, data_dir: Path) -> None:
    """The paper's training loop: run (or resume) the ``cfg.train.steps`` updates, then write ``train_log.csv`` and
    ``final.pt``. (:func:`skae.fit` is the simpler loop of the model package.)"""
    model, batches = build_run(cfg, device, data_dir)
    optimizer = build_optimizer(model, cfg.train)
    start, log = 0, []
    if (run_dir / "last.pt").exists():
        start, log = load_restart(run_dir / "last.pt", model, optimizer, batches)
        print(f"[skae] resuming {run_dir} at update {start}", flush=True)
    total = cfg.train.steps
    model.train()
    for step in range(start, total):
        factor = lr_factor(cfg.train, step)
        for group, base_lr in zip(optimizer.param_groups, (cfg.train.lr, cfg.train.koopman_lr)):
            group["lr"] = base_lr * factor
        optimizer.zero_grad()
        loss, terms = training_loss(model, batches.get(step))
        loss.backward()
        optimizer.step()
        if step % cfg.train.metrics_every == 0:
            log.append({"step": step, "loss": loss.item(), **{k: float(v) for k, v in terms.items()},
                        "lr_factor": factor})
        if step % 1000 == 0 or step == total - 1:
            print(f"[skae] {cfg.run_name} update {step}/{total} loss {loss.item():.4e}", flush=True)
        if (step + 1) % CHECKPOINT_EVERY == 0 or step == total - 1:
            save_restart(run_dir / "last.pt", step, model, optimizer, batches, log)
    write_csv(run_dir / "train_log.csv", log)
    weights = {name: value.cpu() for name, value in model.state_dict().items()}
    fallback, radius = fallback_and_radius(batches.states())
    # final.pt keeps the largest training-state norm; load_run_model multiplies it by DIVERGENCE_RADIUS again
    # (exactly: the norm is a float32 value).
    save({"step": total - 1, "model": weights, "config": cfg.to_dict(), "state_mean": fallback.cpu(),
          "state_max_norm": radius / config.DIVERGENCE_RADIUS}, run_dir / "final.pt")


def train(cfg: RunConfig, run_dir: Path, device: str = "cuda", data_dir: Path = Path("data"),
          evaluate: bool = True) -> Path:
    """Train, resume or finish one run in ``run_dir`` (``cuda`` falls back to the CPU, with a warning); return
    ``run_dir``."""
    run_dir = Path(run_dir)
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA is not available; the run uses the CPU", stacklevel=2)
        device = "cpu"
    torch_device = torch.device(device)
    print(f"[skae] {cfg.run_name}: device {torch_device}", flush=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and RunConfig.load(config_path).to_dict() != cfg.to_dict():
        raise ValueError(f"{run_dir} holds a run with another configuration; use another output directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(config_path)
    if not (run_dir / "final.pt").exists():
        run_updates(cfg, run_dir, torch_device, Path(data_dir))
    if evaluate and not (run_dir / "metrics.json").exists():
        run_evaluation(run_dir, device=str(torch_device), data_dir=data_dir)
    return run_dir
