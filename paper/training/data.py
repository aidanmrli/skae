"""Training data of a run: Dysts caches, train-only normalization and the batch streams."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch

from paper import config
from paper.config import RunConfig
from paper.data import multibasin_training_windows, sample_windows


def train_normalization(trajectories: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-coordinate mean and population standard deviation (floored at 1e-8) of ``[N, T, d]``.

    Computed in float64 over trajectories and time, returned as float32.
    """
    states = trajectories.detach().cpu().double()
    mean = states.mean(dim=(0, 1))
    scale = torch.sqrt(torch.mean((states - mean) ** 2, dim=(0, 1))).clamp_min(1e-8)
    return mean.float(), scale.float()


class Batches:
    """Training batches: fixed windows (multibasin) or windows sampled from the train cache (Dysts).

    Update ``step`` uses stream ``step % BATCH_STREAMS``. Multibasin streams are fixed
    batches; Dysts stream ``i`` has its own generator, seeded with ``seed + i * batch_size``.
    """

    def __init__(self, system, cfg: RunConfig, device: torch.device, data_dir: Path):
        self.cfg = cfg
        self.fixed = self.trajectories = None
        if cfg.benchmark == "multibasin":
            self.fixed = multibasin_training_windows(system, cfg).to(device)
        else:
            from paper.systems.dysts import load_split

            self.trajectories = load_split(data_dir, cfg.system, "train").to(device)
            self.generators = [torch.Generator().manual_seed(cfg.seed + i * cfg.train.batch_size)
                               for i in range(config.BATCH_STREAMS)]

    def states(self) -> torch.Tensor:
        """Every training state ``[n, d]``: the states of the fixed windows (multibasin) or of the trajectories."""
        data = self.fixed if self.fixed is not None else self.trajectories
        return data.reshape(-1, data.shape[-1])

    def get(self, step: int) -> torch.Tensor:
        """The windows ``[batch_size, window + 1, d]`` of update ``step``, on the training device.

        Multibasin: the fixed windows of stream ``step % BATCH_STREAMS``. Dysts: new windows drawn
        with that stream's generator, which advances.
        """
        stream = step % config.BATCH_STREAMS
        if self.fixed is not None:
            return self.fixed[stream]
        return sample_windows(self.trajectories, self.generators[stream], self.cfg.train.batch_size,
                              self.cfg.train.window)

    def state(self) -> Optional[List[torch.Tensor]]:
        """Generator states (Dysts) for a restart checkpoint."""
        return None if self.fixed is not None else [g.get_state() for g in self.generators]

    def load_state(self, states: Optional[List[torch.Tensor]]) -> None:
        """Restore the generator states saved by :meth:`state`."""
        if states is not None:
            for generator, state in zip(self.generators, states):
                generator.set_state(state)
