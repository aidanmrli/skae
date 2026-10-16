"""Dysts benchmark data: trajectory caches of ten chaotic three-dimensional flows.

Each split (``train``, ``val``, ``test``) of a flow holds 200 standardized
trajectories of 30,001 states, built with dysts 0.96. Slot ``i`` of a split
starts from the flow's default initial condition plus Gaussian jitter of 0.2
attractor standard deviations per coordinate, drawn from a torch generator
seeded by the split (0, 1, 2); only slot 0 of the train split starts exactly
at the default initial condition, so the splits share no initial condition.
dysts integrates the standardized flow with DOP853 (rtol = atol = 1e-12, its
defaults) and samples it every 5 native dysts steps; the first 2,000 samples
are a discarded warm-up. No observation noise is added.

Some Sakarya draws escape or become stiff. A Sakarya draw is rejected when its
standardized state leaves the ball of radius ``SAKARYA_BOUND`` or the solver
has evaluated the right-hand side more than ``SAKARYA_RHS_LIMIT`` times (both
checked after every solver step and at the end); its slot then takes the next
unused initial condition of the split's stream (200, 201, ...), slots in
order. ``SAKARYA_BOUND`` is ten times the largest standardized state norm of the
training split; ``SAKARYA_RHS_LIMIT`` is far above the roughly 55,000
evaluations of a typical trajectory.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.integrate import DOP853

from paper.systems import DYSTS_SYSTEMS

SPLIT_SEEDS = {"train": 0, "val": 1, "test": 2}
NUM_TRAJECTORIES = 200
NUM_STEPS = 30_000      # stored transitions per trajectory (30,001 states)
WARMUP = 2_000          # discarded samples before the first stored state
DT_MULTIPLIER = 5       # native dysts steps per stored step
JITTER = 0.2            # initial-condition jitter, in attractor standard deviations
SAKARYA_BOUND = 76.8667223899191
SAKARYA_RHS_LIMIT = 711_580


class DystsSystem:
    """A dysts flow whose ``dt`` is the stored step (``DT_MULTIPLIER`` native steps)."""

    def __init__(self, name: str):
        if name not in DYSTS_SYSTEMS:
            raise ValueError(f"unknown dysts system {name!r}; choose from {', '.join(DYSTS_SYSTEMS)}")
        from dysts import flows

        self.name = name
        self.flow = getattr(flows, name)()
        self.dim = int(self.flow.dimension)
        self.default_ic = np.asarray(self.flow.ic, dtype=np.float32)
        self.flow.dt = float(self.flow.dt) * DT_MULTIPLIER


def initial_conditions(system: DystsSystem, split: str, count: int) -> List[np.ndarray]:
    """The first ``count`` initial conditions (float32) of a split's stream."""
    generator = torch.Generator().manual_seed(SPLIT_SEEDS[split])
    std = np.asarray(system.flow.std, dtype=np.float32)
    ics = []
    for index in range(count):
        if split == "train" and index == 0:
            ics.append(system.default_ic)
        else:
            noise = torch.randn(system.dim, generator=generator).numpy()
            ics.append(system.default_ic + noise * std * JITTER)
    return ics


def _sakarya_rejects(solver) -> bool:
    return solver.nfev > SAKARYA_RHS_LIMIT or np.linalg.norm(solver.y) >= SAKARYA_BOUND


def _sakarya_solver():
    """A DOP853 subclass that stops at the first step after which the Sakarya rule
    rejects the draw, and the list that receives its instance."""
    instances = []

    class SakaryaDOP853(DOP853):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def step(self):
            message = super().step()
            if self.status != "failed" and _sakarya_rejects(self):
                self.status = "failed"
            return message

    return SakaryaDOP853, instances


def integrate(system: DystsSystem, ic: np.ndarray) -> Optional[torch.Tensor]:
    """Stored states ``[NUM_STEPS + 1, dim]`` (float32, standardized) from ``ic``.

    Returns None when the Sakarya rule rejects the draw.
    """
    flow = system.flow
    method, solvers = "DOP853", None
    if system.name == "Sakarya":
        # A start above the bound is rejected, a solver state at or above it (`_sakarya_rejects`); the two
        # differ only at exact equality and are kept as in the code that built the paper's caches.
        if np.linalg.norm((ic - flow.mean) / flow.std) > SAKARYA_BOUND:
            return None
        method, solvers = _sakarya_solver()
    with warnings.catch_warnings():  # DOP853 ignores the Jacobian that dysts passes
        warnings.filterwarnings("ignore", message=".*no effect for a chosen solver.*")
        states = flow.make_trajectory(WARMUP + NUM_STEPS + 1, init_cond=ic, resample=False, standardize=True,
                                      method=method)
    if solvers is not None and _sakarya_rejects(solvers[0]):  # also counts evaluations after the last step
        return None
    if states is None:
        raise RuntimeError(f"dysts failed to integrate {system.name} from {ic}")
    return torch.from_numpy(np.ascontiguousarray(states[WARMUP:], dtype=np.float32))


@lru_cache(maxsize=None)
def _system(name: str) -> DystsSystem:
    return DystsSystem(name)


def _integrate_in_worker(name: str, ic: np.ndarray) -> Optional[torch.Tensor]:
    return integrate(_system(name), ic)


def build_split(name: str, split: str, workers: int = 1) -> Tuple[torch.Tensor, List[int]]:
    """Trajectories ``[NUM_TRAJECTORIES, NUM_STEPS + 1, 3]`` of one split and the
    stream index of each slot's initial condition.

    Trajectories are independent, so ``workers`` processes give identical results.
    """
    system = DystsSystem(name)
    ics = initial_conditions(system, split, NUM_TRAJECTORIES)
    if workers > 1:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            states = list(pool.map(_integrate_in_worker, [name] * len(ics), ics))
    else:
        states = [integrate(system, ic) for ic in ics]
    streams = list(range(NUM_TRAJECTORIES))
    next_stream = NUM_TRAJECTORIES
    for slot in range(NUM_TRAJECTORIES):
        while states[slot] is None:  # rejected Sakarya draw: take the next unused initial condition
            streams[slot] = next_stream
            states[slot] = integrate(system, initial_conditions(system, split, next_stream + 1)[next_stream])
            next_stream += 1
    return torch.stack(states), streams


def cache_path(data_dir: Path, name: str, split: str) -> Path:
    """``<data_dir>/dysts/<name>_<split>.pt``."""
    return Path(data_dir) / "dysts" / f"{name}_{split}.pt"


def write_split(data_dir: Path, name: str, split: str, workers: int = 1) -> List[int]:
    """Build one split and save it to :func:`cache_path`; returns the slots' stream indices."""
    trajectories, streams = build_split(name, split, workers)
    path = cache_path(data_dir, name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:  # saving to a handle keeps the bytes independent of the file name
        torch.save({"system": name, "split": split, "warmup": WARMUP, "stream_indices": streams,
                    "trajectories": trajectories}, handle)
    partial.replace(path)
    return streams


def load_split(data_dir: Path, name: str, split: str) -> torch.Tensor:
    """Trajectories ``[200, 30_001, 3]`` (float32) of one split built by ``python -m paper data``."""
    path = cache_path(data_dir, name, split)
    if not path.is_file():
        raise FileNotFoundError(f"missing Dysts cache {path}; build it with `python -m paper data --data {data_dir}`")
    return torch.load(path, map_location="cpu", weights_only=True)["trajectories"]
