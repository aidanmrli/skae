"""Base class and utilities for transition-rich dynamical system catalog."""

import torch
import math
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod


def rk4_step(dynamics_fn, state: torch.Tensor, dt: float) -> torch.Tensor:
    """Fourth-order Runge-Kutta integration step."""
    k1 = dynamics_fn(state)
    k2 = dynamics_fn(state + 0.5 * dt * k1)
    k3 = dynamics_fn(state + 0.5 * dt * k2)
    k4 = dynamics_fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class CatalogSystem(ABC):
    """Base class for transition-rich dynamical systems.

    Subclasses must define:
        - name (str): unique identifier
        - category (str): category letter (A-H)
        - description (str): human-readable description
        - dim (property): state-space dimension
        - dynamics(state): returns dx/dt
    """

    name: str = "unnamed"
    category: str = "?"
    description: str = ""

    def __init__(
        self,
        dt: float = 0.01,
        ic_box: Optional[List[Tuple[float, float]]] = None,
    ):
        self.dt = dt
        self._ic_box = ic_box

    @property
    def ic_box(self) -> List[Tuple[float, float]]:
        if self._ic_box is not None:
            return self._ic_box
        return [(-1.0, 1.0)] * self.dim

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """Return dx/dt given current state x."""
        ...

    def step(self, state: torch.Tensor) -> torch.Tensor:
        """Single RK4 integration step."""
        return rk4_step(self.dynamics, state, self.dt)

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample random initial condition from ic_box."""
        if rng is None:
            rng = torch.Generator()
        state = torch.zeros(self.dim, dtype=torch.float64)
        for i, (lo, hi) in enumerate(self.ic_box):
            state[i] = lo + (hi - lo) * torch.rand(1, generator=rng, dtype=torch.float64).item()
        return state

    def generate_trajectory(
        self, init_state: torch.Tensor, length: int
    ) -> torch.Tensor:
        """Generate trajectory of given length. Returns [length+1, dim]."""
        traj = torch.zeros(length + 1, self.dim, dtype=torch.float64)
        traj[0] = init_state.double()
        state = init_state.double()
        for t in range(length):
            state = self.step(state)
            # Clamp to prevent divergence
            state = torch.clamp(state, -1e6, 1e6)
            if torch.any(torch.isnan(state)) or torch.any(torch.isinf(state)):
                # Fill rest with last valid state
                traj[t + 1 :] = traj[t]
                break
            traj[t + 1] = state
        return traj

    def info(self) -> Dict[str, Any]:
        """Return metadata dictionary."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "dim": self.dim,
            "dt": self.dt,
            "ic_box": self.ic_box,
        }
