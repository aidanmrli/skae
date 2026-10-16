"""Training, validation and test data.

Multibasin systems
    Training windows are generated from ``BATCH_STREAMS`` (= 8) fixed
    streams: stream ``k`` resets state ``i`` with seed ``seed + 256 k + i``.
    Batches are therefore a fixed set of ``8 x 256 = 2048`` windows that the
    optimizer cycles through. Half of the initial states come from a pool of
    "hard" initial conditions (see :class:`HardInitSampler`).

Dysts systems
    Windows are sampled uniformly (trajectory, start time) from a cached split
    of 200 trajectories, using one stateful generator per stream.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from paper import config
from paper.config import RunConfig


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def reset_batch(reset_fn: Callable[[torch.Generator], torch.Tensor], base_seed: int, count: int) -> torch.Tensor:
    """Initial states ``reset_fn(Generator(base_seed + i))`` for ``i < count``."""
    return torch.stack([reset_fn(_generator(base_seed + index)) for index in range(count)], dim=0)


def rollout(step_fn: Callable[[torch.Tensor], torch.Tensor], states: torch.Tensor, steps: int) -> torch.Tensor:
    """``[steps, B, d]`` trajectory after ``states`` (excluding the start)."""
    out = []
    for _ in range(steps):
        states = step_fn(states)
        out.append(states)
    return torch.stack(out, dim=0)


class HardInitSampler:
    """Resets that oversample separatrix-adjacent, slowly settling states.

    A pool is built once per run from ``NUM_CANDIDATES`` uniform initial
    conditions. Each candidate is scored by (i) how far ``NUM_PERTURBATIONS``
    small perturbations have spread after ``PROBE_STEPS`` steps and (ii) how
    much it still moves over the last ``TRANSIENT_WINDOW`` steps. The
    ``POOL_SIZE`` highest-scoring candidates form the pool. With probability
    ``FRACTION`` a reset draws a pool state with a small Gaussian jitter;
    otherwise it falls back to the system's uniform reset. The score uses no
    basin labels. For run seed ``seed``, candidate ``i`` resets with seed
    ``seed + HARD_INIT_SEED_OFFSET + i`` and the perturbations come from one
    generator seeded ``seed + HARD_INIT_NOISE_SEED_OFFSET`` (``paper.config``).
    """

    FRACTION = 0.5
    POOL_SIZE = 1024
    NUM_CANDIDATES = 4096
    PROBE_STEPS = 32
    NUM_PERTURBATIONS = 4
    PERTURB_SCALE = 0.04
    TRANSIENT_WINDOW = 8
    TRANSIENT_WEIGHT = 0.5
    JITTER_SCALE = 0.25
    CHUNK_SIZE = 128

    def __init__(self, system, seed: int):
        self.system = system
        self.low, self.high = system.reset_bounds()
        self.pool = self._build_pool(seed)

    def _build_pool(self, seed: int) -> torch.Tensor:
        count = self.NUM_CANDIDATES
        pool_size = min(self.POOL_SIZE, count)
        window = min(self.TRANSIENT_WINDOW, self.PROBE_STEPS)
        span = (self.high - self.low).clamp_min(1e-6)
        box_diag = torch.linalg.vector_norm(span).clamp_min(1e-6)
        perturb_std = float(self.PERTURB_SCALE) * span
        perturb_norm = torch.linalg.vector_norm(perturb_std).clamp_min(1e-6)
        candidate_seed = seed + config.HARD_INIT_SEED_OFFSET
        candidates = torch.stack([self.system.reset(_generator(candidate_seed + i)).float() for i in range(count)],
                                 dim=0)
        low, high = self.low.view(1, 1, -1), self.high.view(1, 1, -1)
        noise_generator = _generator(seed + config.HARD_INIT_NOISE_SEED_OFFSET)
        scores = torch.empty(count, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, count, self.CHUNK_SIZE):
                chunk = candidates[start:start + self.CHUNK_SIZE]
                n, dim = chunk.shape
                noise = torch.randn(n, self.NUM_PERTURBATIONS, dim, generator=noise_generator, dtype=chunk.dtype)
                noise = noise * perturb_std.view(1, 1, -1).to(dtype=chunk.dtype)
                starts = torch.cat([chunk.unsqueeze(1), chunk.unsqueeze(1) + noise], dim=1)
                starts = torch.maximum(torch.minimum(starts, high), low)
                flat = starts.reshape(-1, dim)
                path = torch.cat([flat.unsqueeze(0), rollout(self.system.step, flat, self.PROBE_STEPS)], dim=0)
                path = path.reshape(self.PROBE_STEPS + 1, n, starts.shape[1], dim)
                end, tail = path[-1], path[-1 - window]
                sensitivity = torch.linalg.vector_norm(end[:, 1:] - end[:, :1], dim=-1).mean(dim=1) / perturb_norm
                late_motion = torch.linalg.vector_norm(end - tail, dim=-1).mean(dim=1) / box_diag
                scores[start:start + n] = sensitivity.float() + float(self.TRANSIENT_WEIGHT) * late_motion.float()
        top_index = torch.topk(scores, k=pool_size, largest=True, sorted=True).indices
        return candidates[top_index]

    def reset(self, generator: torch.Generator) -> torch.Tensor:
        """One initial state ``[d]``, drawn with ``generator`` (a drop-in for ``system.reset``)."""
        if torch.rand(1, generator=generator).item() >= self.FRACTION:
            return self.system.reset(generator)
        index = int(torch.randint(0, self.pool.shape[0], (1,), generator=generator).item())
        state = self.pool[index].clone()
        span = (self.high - self.low).clamp_min(1e-6)
        jitter_std = self.JITTER_SCALE * self.PERTURB_SCALE * span
        state = state + torch.randn(state.shape, generator=generator, dtype=state.dtype) * jitter_std.to(state.dtype)
        return torch.maximum(torch.minimum(state, self.high), self.low)


def multibasin_training_windows(system, cfg: RunConfig) -> torch.Tensor:
    """Fixed training windows ``[BATCH_STREAMS, batch, L + 1, 2]``."""
    train = cfg.train
    reset_fn = system.reset
    if train.hard_init:
        reset_fn = HardInitSampler(system, cfg.seed).reset
    batches = []
    for stream in range(config.BATCH_STREAMS):
        starts = reset_batch(reset_fn, cfg.seed + stream * train.batch_size, train.batch_size)
        path = torch.cat([starts.unsqueeze(0), rollout(system.step, starts, train.window)], dim=0)
        batches.append(path.transpose(0, 1))
    return torch.stack(batches, dim=0)


def sample_windows(trajectories: torch.Tensor, generator: torch.Generator, batch: int, window: int) -> torch.Tensor:
    """Uniformly sampled windows ``[batch, window + 1, d]`` from ``[N, T, d]``."""
    count, length, _ = trajectories.shape
    max_start = length - (window + 1)
    traj_index = torch.randint(0, count, (batch,), generator=generator)
    time_index = torch.randint(0, max_start + 1, (batch,), generator=generator)
    offsets = torch.arange(window + 1)
    return trajectories[traj_index[:, None], time_index[:, None] + offsets[None, :]]


def _forecast_set(system, trajectories: Optional[torch.Tensor], base_seed: int, count: int, horizon: int):
    """``count`` starts ``[count, d]`` and their next ``horizon`` true states ``[horizon, count, d]``.

    Multibasin: resets with seeds ``base_seed + i``, rolled out by the system.
    Dysts: windows of a cached split, drawn by :func:`sample_windows` from ``Generator(base_seed)``.
    """
    if trajectories is None:
        starts = reset_batch(system.reset, base_seed, count)
        with torch.no_grad():
            truth = rollout(system.step, starts, horizon)
        return starts, truth
    windows = sample_windows(trajectories, _generator(base_seed), count, horizon)
    return windows[:, 0].contiguous(), windows[:, 1:].transpose(0, 1).contiguous()


def test_set(system, cfg: RunConfig, test_trajectories: Optional[torch.Tensor] = None):
    """Held-out test starts (``TEST_STARTS`` = 100, seeds ``seed + TEST_SEED_OFFSET + i``) and their truth."""
    return _forecast_set(system, test_trajectories, cfg.seed + config.TEST_SEED_OFFSET, config.TEST_STARTS,
                         max(cfg.eval.horizons))


def validation_forecast_set(system, cfg: RunConfig, val_trajectories: Optional[torch.Tensor] = None):
    """Validation starts (``VALIDATION_STARTS`` = 100, seeds ``seed + VALIDATION_SEED_OFFSET + i``) and their
    truth for ``SELECTION_HORIZON`` steps."""
    return _forecast_set(system, val_trajectories, cfg.seed + config.VALIDATION_SEED_OFFSET, config.VALIDATION_STARTS,
                         config.SELECTION_HORIZON)
