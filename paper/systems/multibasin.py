"""A multibasin system as the trainer and the analyses use it (``paper.systems.make_system``)."""

from __future__ import annotations

from typing import Tuple

import torch

from paper.systems import MULTIBASIN_SYSTEMS
from paper.systems.analytic import ANALYTIC_SYSTEMS
from paper.systems.gated import GatedLocalLinear, GatedTransferLinear

GATED_SYSTEMS = {"gated_local_linear": GatedLocalLinear, "gated_transfer_linear": GatedTransferLinear}
# name -> constructor taking the stored step ``dt``
IMPLEMENTATIONS = {**ANALYTIC_SYSTEMS, **GATED_SYSTEMS}


class MultibasinSystem:
    """One multibasin system.

    ``reset(generator)`` draws a float32 initial state uniformly from the box
    ``reset_bounds()``; ``step(states)`` advances a batch ``[B, 2]`` by one
    stored step ``dt``. Analytic systems are integrated in float64. An unknown
    name raises ``ValueError``.
    """

    dim = 2

    def __init__(self, name: str):
        if name not in MULTIBASIN_SYSTEMS:
            raise ValueError(f"unknown multibasin system {name!r}; choose from {', '.join(MULTIBASIN_SYSTEMS)}")
        if name not in IMPLEMENTATIONS:
            raise ValueError(f"multibasin system {name!r} has no entry in ANALYTIC_SYSTEMS or GATED_SYSTEMS")
        self.name = name
        self.spec = MULTIBASIN_SYSTEMS[name]
        self.dt = self.spec.dt
        self.num_basins = self.spec.num_basins
        self.analytic = name in ANALYTIC_SYSTEMS
        self.impl = IMPLEMENTATIONS[name](dt=self.dt)

    def reset(self, generator: torch.Generator) -> torch.Tensor:
        return self.impl.reset(generator).to(dtype=torch.float32)

    def step(self, states: torch.Tensor) -> torch.Tensor:
        # States are stepped one at a time under torch.vmap: this repeats the
        # floating-point operations that generated the paper's data (a batched
        # step can differ in the last bits). The transfer system selects its
        # regions with data-dependent branches, which vmap cannot trace, so it
        # is stepped as one batch.
        if self.name == "gated_transfer_linear":
            return self.impl.step(states)
        return torch.vmap(self._step_one)(states)

    def _step_one(self, state: torch.Tensor) -> torch.Tensor:
        if self.analytic:
            return self.impl.step(state.to(torch.float64)).to(state.dtype)
        return self.impl.step(state)

    def reset_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lower and upper corners of the initial-condition box."""
        box = self.impl.ic_box
        low = torch.tensor([float(lo) for lo, _ in box], dtype=torch.float32)
        high = torch.tensor([float(hi) for _, hi in box], dtype=torch.float32)
        return low, high
