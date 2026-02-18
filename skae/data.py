"""
Data module for dynamical systems environments.

Provides PyTorch-based implementations of various nonlinear dynamical systems
structured as environments.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple
import time
import numpy as np
import torch
from skae.config import Config


# ---------------------------------------------------------------------------
# Base Environment Classes
# ---------------------------------------------------------------------------


class Env(ABC):
    """Base Environment class for dynamical systems."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg

    @abstractmethod
    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reset environment to initial state.
        
        Args:
            rng: Random number generator for reproducibility
            
        Returns:
            Initial state tensor
        """
        pass

    @abstractmethod
    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Take one step in the environment.
        
        Args:
            state: Current state tensor
            action: Optional action tensor (for controlled systems)
            
        Returns:
            Next state tensor
        """
        pass

    @property
    def observation_size(self) -> int:
        """Dimensionality of the state space."""
        rng = torch.Generator()
        rng.manual_seed(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.shape[-1]

    @property
    def action_size(self) -> int:
        """Dimensionality of the action space."""
        return 0

    @property
    def unwrapped(self) -> 'Env':
        """Return the unwrapped environment."""
        return self


class Wrapper(Env):
    """Base Wrapper class for environment modifications."""

    def __init__(self, env: Env):
        super().__init__(cfg=None)
        self.env = env

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.env.reset(rng)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped


class VectorWrapper(Wrapper):
    """Wrapper for vectorized/batched environment operations."""

    def __init__(self, env: Env, batch_size: int):
        super().__init__(env)
        self.batch_size = batch_size
        self._vmap_supported: Optional[bool] = None

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reset multiple environments in parallel with independent random seeds.
        
        Mimics JAX's jax.random.split behavior by creating independent generators
        for each environment in the batch to ensure diverse initial states.
        
        Args:
            rng: Random number generator (if None, creates a new one)
            
        Returns:
            Batch of initial states with shape [batch_size, state_dim]
        """
        if rng is None:
            rng = torch.Generator()
        
        # Create independent generators for each environment (like jax.random.split)
        base_seed = rng.initial_seed()
        states = []
        for i in range(self.batch_size):
            env_rng = torch.Generator().manual_seed(base_seed + i)
            states.append(self.env.reset(env_rng))
        return torch.stack(states, dim=0)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Step multiple environments in parallel.
        
        Args:
            state: Batch of states with shape [batch_size, state_dim]
            action: Optional batch of actions
            
        Returns:
            Batch of next states with shape [batch_size, state_dim]
        """
        # Try vmap for environments that support it (pure PyTorch)
        # Fall back to direct call for environments with numpy ops (e.g., DystsEnv)
        if self._vmap_supported is True:
            if action is None:
                return torch.vmap(lambda s: self.env.step(s, None))(state)
            return torch.vmap(lambda s, a: self.env.step(s, a))(state, action)
        if self._vmap_supported is False:
            return self.env.step(state, action)
        
        try:
            if action is None:
                result = torch.vmap(lambda s: self.env.step(s, None))(state)
            else:
                result = torch.vmap(lambda s, a: self.env.step(s, a))(state, action)
            self._vmap_supported = True
            return result
        except RuntimeError as e:
            if "storage" in str(e).lower() or "vmap" in str(e).lower():
                # Environment doesn't support vmap (uses numpy internally)
                # Fall back to letting the environment handle batching
                self._vmap_supported = False
                return self.env.step(state, action)
            raise
    
    def generate_sequence_batch(
        self, 
        rng: Optional[torch.Generator] = None,
        window_length: int = 10,
    ) -> torch.Tensor:
        """Generate a batch of sequence windows using vectorized operations.
        
        Args:
            rng: Random number generator
            window_length: Length of sequence (T steps after initial state)
            
        Returns:
            Batch of sequences with shape [batch_size, window_length+1, state_dim]
            Each sequence contains [x_t, x_{t+1}, ..., x_{t+T}]
        """
        init_states = self.reset(rng)  # [batch_size, state_dim]
        
        # Vectorized sequence generation: use generate_trajectory with batched step
        # This is much faster than the Python loop
        trajectories = generate_trajectory(
            self.step,
            init_states,
            length=window_length
        )  # [window_length, batch_size, state_dim]
        
        # Stack initial states with trajectories: [batch_size, window_length+1, state_dim]
        sequences = torch.cat([
            init_states.unsqueeze(0),  # [1, batch_size, state_dim]
            trajectories
        ], dim=0)  # [window_length+1, batch_size, state_dim]
        
        # Transpose to [batch_size, window_length+1, state_dim]
        return sequences.transpose(0, 1)


# ---------------------------------------------------------------------------
# Dysts trajectory cache for training
# ---------------------------------------------------------------------------

class DystsTrajectoryCache:
    """In-memory cache of dysts trajectories for fast batch sampling."""
    
    def __init__(
        self,
        env,
        cfg: Config,
        rng: Optional[torch.Generator] = None,
    ):
        self.env = env
        self.cfg = cfg
        self.cache_steps = cfg.ENV.DYSTS.CACHE_STEPS
        self.cache_trajectories = max(1, cfg.ENV.DYSTS.CACHE_TRAJECTORIES)
        self.cache_warmup = max(0, cfg.ENV.DYSTS.CACHE_WARMUP)
        self.standardize = cfg.ENV.DYSTS.STANDARDIZE
        self.resample = cfg.ENV.DYSTS.RESAMPLE
        self.pts_per_period = cfg.ENV.DYSTS.PTS_PER_PERIOD
        
        if self.cache_steps < 2:
            raise ValueError("Dysts cache requires CACHE_STEPS >= 2")
        
        if rng is None:
            rng = torch.Generator().manual_seed(0)
        self._rng = rng
        self._trajectories = self._build_cache()
    
    def _build_cache(self) -> torch.Tensor:
        """Generate cached trajectories using dysts native integrator."""
        base_ic = self.env._ic.detach().cpu().numpy()
        base_std = self.env._std.detach().cpu().numpy()
        original_ic = None
        if hasattr(self.env.system, "ic"):
            original_ic = np.array(self.env.system.ic, copy=True)
        
        trajectories: List[torch.Tensor] = []
        t0 = time.perf_counter()
        for i in range(self.cache_trajectories):
            if i == 0 or (i + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[dysts cache] building trajectory {i + 1}/{self.cache_trajectories} "
                    f"(elapsed={elapsed:.1f}s)",
                    flush=True,
                )
            if i == 0:
                ic = base_ic
            else:
                noise = torch.randn(self.env._dim, generator=self._rng).numpy()
                ic = base_ic + noise * base_std * float(self.env.ic_noise_scale)
            if hasattr(self.env.system, "ic"):
                self.env.system.ic = np.array(ic, dtype=np.float32)
            
            try:
                traj = self.env.make_trajectory_native(
                    n=self.cache_steps + self.cache_warmup,
                    resample=self.resample,
                    pts_per_period=self.pts_per_period,
                    standardize=self.standardize,
                )
            except Exception as e:
                if i == 0:
                    raise
                print(
                    f"Warning: Failed to build dysts trajectory {i + 1}/"
                    f"{self.cache_trajectories}: {e}. Using fewer trajectories."
                )
                break
            if self.cache_warmup > 0:
                traj = traj[self.cache_warmup:]
            traj = traj.float()
            trajectories.append(traj)
        
        if original_ic is not None:
            self.env.system.ic = original_ic
        
        elapsed = time.perf_counter() - t0
        print(
            f"[dysts cache] done: trajectories={len(trajectories)}, "
            f"steps={trajectories[0].shape[0] if trajectories else 0} "
            f"(total={elapsed:.1f}s)",
            flush=True,
        )
        
        if len(trajectories) == 1:
            return trajectories[0].unsqueeze(0)
        
        return torch.stack(trajectories, dim=0)
    
    @property
    def trajectories(self) -> torch.Tensor:
        return self._trajectories
    
    def sample_pair_batch(
        self,
        rng: torch.Generator,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sample a batch of (x_t, x_{t+1}) pairs."""
        traj_count, steps, _ = self._trajectories.shape
        max_start = steps - 1
        if max_start <= 0:
            raise ValueError("Cached trajectories are too short for pairwise sampling.")
        
        traj_idx = torch.randint(0, traj_count, (batch_size,), generator=rng)
        time_idx = torch.randint(0, max_start, (batch_size,), generator=rng)
        x = self._trajectories[traj_idx, time_idx]
        nx = self._trajectories[traj_idx, time_idx + 1]
        
        if device is not None:
            x = x.to(device)
            nx = nx.to(device)
        return x, nx
    
    def sample_sequence_batch(
        self,
        rng: torch.Generator,
        batch_size: int,
        window_length: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sample a batch of sequence windows [x_t, ..., x_{t+T}]."""
        traj_count, steps, _ = self._trajectories.shape
        max_start = steps - (window_length + 1)
        if max_start < 0:
            raise ValueError(
                "Cached trajectories are too short for sequence sampling. "
                "Increase CACHE_STEPS or reduce SEQUENCE_LENGTH."
            )
        
        traj_idx = torch.randint(0, traj_count, (batch_size,), generator=rng)
        time_idx = torch.randint(0, max_start + 1, (batch_size,), generator=rng)
        offsets = torch.arange(window_length + 1)
        seq_idx = time_idx[:, None] + offsets[None, :]
        sequences = self._trajectories[traj_idx[:, None], seq_idx]
        
        if device is not None:
            sequences = sequences.to(device)
        return sequences


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def integrate_euler(
    x: torch.Tensor,
    u: Optional[torch.Tensor],
    dt: float,
    dynamics_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
) -> torch.Tensor:
    """Euler integration step for ODE.
    
    Args:
        x: Current state
        u: Optional control input
        dt: Time step
        dynamics_fn: Function computing state derivatives
        
    Returns:
        Next state using Euler integration
    """
    return x + dt * dynamics_fn(x, u)


def integrate_rk4(
    x: torch.Tensor,
    u: Optional[torch.Tensor],
    dt: float,
    dynamics_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
) -> torch.Tensor:
    """Fourth-order Runge-Kutta (RK4) integration step for ODE.
    
    The RK4 method is a higher-order numerical integrator that provides
    better accuracy than Euler integration by using a weighted average
    of four slope estimates.
    
    Args:
        x: Current state
        u: Optional control input
        dt: Time step
        dynamics_fn: Function computing state derivatives
        
    Returns:
        Next state using RK4 integration
    """
    k1 = dynamics_fn(x, u)
    k2 = dynamics_fn(x + 0.5 * dt * k1, u)
    k3 = dynamics_fn(x + 0.5 * dt * k2, u)
    k4 = dynamics_fn(x + dt * k3, u)
    
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_trajectory(
    env_step: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    init_state: torch.Tensor,
    length: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    actions: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Generate a trajectory by repeatedly applying environment step.
    
    Args:
        env_step: Function that takes state (and optionally action) and returns next state
        init_state: Initial state tensor
        length: Number of steps (required if actions is None)
        rng: Random number generator (unused, for API compatibility)
        actions: Optional sequence of actions with shape [length, action_dim]
        
    Returns:
        Trajectory tensor with shape [length, state_dim] or [length, batch_size, state_dim]
    """
    if actions is None:
        assert length is not None, "Must provide either length or actions"
        states = []
        state = init_state
        for _ in range(length):
            state = env_step(state)
            states.append(state)
        return torch.stack(states, dim=0)
    else:
        states = []
        state = init_state
        for action in actions:
            state = env_step(state, action)
            states.append(state)
        return torch.stack(states, dim=0)


def generate_sequence_window(
    env_step: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    init_state: torch.Tensor,
    window_length: int,
) -> torch.Tensor:
    """Generate a sequence window including the initial state.
    
    Args:
        env_step: Function that takes state (and optionally action) and returns next state
        init_state: Initial state tensor
        window_length: Length of sequence window (T+1 total states including initial)
        
    Returns:
        Sequence tensor with shape [window_length+1, state_dim] or [window_length+1, batch_size, state_dim]
        This includes x_t, x_{t+1}, ..., x_{t+T}
    """
    states = [init_state]
    state = init_state
    for _ in range(window_length):
        state = env_step(state)
        states.append(state)
    return torch.stack(states, dim=0)


# ---------------------------------------------------------------------------
# Dynamical System Environments
# ---------------------------------------------------------------------------


class Pendulum(Env):
    """Pendulum model - freely swinging pole.
    
    State: [angle, angular_velocity]
    The angle x1 is measured in radians from the downward vertical position.
    Initial conditions sampled uniformly from [-π, π] × [-2, 2].
    
    Dynamics:
        dot(x1) = x2
        dot(x2) = -(g/L) * sin(x1)
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.g_over_l = 9.81 / 1.0
        self.dt = cfg.ENV.PENDULUM.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(-torch.pi, torch.pi, generator=rng)
        x2 = torch.empty(1).uniform_(-2.0, 2.0, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            x1, x2 = state[0], state[1]
            dx1 = x2
            dx2 = -self.g_over_l * torch.sin(x1)
            return torch.stack([dx1, dx2])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class Duffing(Env):
    """Duffing Oscillator - damped and force-driven particle model.
    
    Nonlinear second-order ODE: x'' = x - x^3
    
    Admits two center points at (x, x') = (±1, 0) and an unstable fixed point at origin.
    Initial conditions sampled uniformly from [-2, 2] × [-1, 1].
    
    Dynamics:
        dot(x1) = x2
        dot(x2) = x1 - x1^3
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.dt = cfg.ENV.DUFFING.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(-1.5, 1.5, generator=rng)
        x2 = torch.empty(1).uniform_(-1.0, 1.0, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            x1, x2 = state[0], state[1]
            dx1 = x2
            dx2 = x1 - x1**3
            return torch.stack([dx1, dx2])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class LotkaVolterra(Env):
    """Lotka-Volterra predator-prey model.
    
    Models population dynamics with predator-prey interactions.
    State: [prey_population, predator_population]
    
    Dynamics:
        dot(x1) = alpha * x1 - beta * x1 * x2
        dot(x2) = delta * x1 * x2 - gamma * x2
    
    Parameters: alpha = beta = gamma = delta = 0.2
    Fixed points: origin and center at (gamma/delta, alpha/beta) = (1, 1)
    Initial conditions sampled uniformly from [0.02, 3.0] × [0.02, 3.0]
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.alpha = 0.2
        self.beta = 0.2
        self.gamma = 0.2
        self.delta = 0.2
        self.dt = cfg.ENV.LOTKA_VOLTERRA.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(0.02, 3.0, generator=rng)
        x2 = torch.empty(1).uniform_(0.02, 3.0, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            prey, predator = state[0], state[1]
            dx1 = self.alpha * prey - self.beta * prey * predator
            dx2 = self.delta * prey * predator - self.gamma * predator
            return torch.stack([dx1, dx2])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class Lorenz63(Env):
    """Lorenz 63 system - chaotic three-dimensional system.
    
    Famous for the "butterfly effect" and sensitivity to initial conditions.
    Exhibits chaotic behavior with strange attractor.
    
    Dynamics:
        dot(x1) = sigma * (x2 - x1)
        dot(x2) = x1 * (rho - x3) - x2
        dot(x3) = x1 * x2 - beta * x3
    
    Standard parameters: sigma=10, rho=28, beta=8/3 (Lorenz 1963)
    Initial conditions: perturbations around (0, 1, 1.05) with std=1.0
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        self.dt = cfg.ENV.LORENZ63.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        base_point = torch.tensor([0.0, 1.0, 1.05], dtype=torch.float32)
        noise = torch.randn(3, generator=rng, dtype=torch.float32)
        return base_point + noise

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            x, y, z = state[0], state[1], state[2]
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            return torch.stack([dx, dy, dz])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class Parabolic(Env):
    """Parabolic Attractor - two-dimensional system with parabolic manifold.
    
    Dynamics admit solution asymptotically attracted to x2 = x1^2 for lambda < mu < 0.
    
    Dynamics:
        dot(x1) = mu * x1
        dot(x2) = lambda * (x2 - x1^2)
    
    The Koopman embedding z = [x1, x2, x1^2] has globally linear dynamics:
        dot(z) = [mu*z1, lambda*z2 - lambda*z3, 2*mu*z3]
    
    Parameters: lambda=-1.0, mu=-0.1
    Initial conditions sampled uniformly from [-1, 1] × [-1, 1]
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.const_lambda = cfg.ENV.PARABOLIC.LAMBDA
        self.const_mu = cfg.ENV.PARABOLIC.MU
        self.dt = cfg.ENV.PARABOLIC.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(-1.0, 1.0, generator=rng)
        x2 = torch.empty(1).uniform_(-1.0, 1.0, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            x1, x2 = state[0], state[1]
            dx1 = self.const_mu * x1
            dx2 = self.const_lambda * (x2 - x1**2)
            return torch.stack([dx1, dx2])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


# ---------------------------------------------------------------------------
# Lyapunov Multi-Attractor Environment (from Koopman_learning.ipynb)
# ---------------------------------------------------------------------------


class LyapunovMultiAttractor(Env):
    """Nonlinear 2D system with many exponentially stable equilibria.
    
    Implements the vector field described in the notebook, with equilibria at
    a fixed set of 13 points and dynamics built from Gaussian bump functions.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Configurable parameters (defaults match the notebook example)
        lyap_cfg = cfg.ENV.LYAPUNOV if hasattr(cfg.ENV, 'LYAPUNOV') else None
        self.dt = lyap_cfg.DT if lyap_cfg is not None else 0.05
        self.sigma = lyap_cfg.SIGMA if lyap_cfg is not None else 0.5
        self.dim = lyap_cfg.DIM if lyap_cfg is not None else 2
        self.num_basins = lyap_cfg.NUM_BASINS if lyap_cfg is not None else 13
        self.points_mode = lyap_cfg.POINTS_MODE if lyap_cfg is not None else "fixed"
        self.center_scale = lyap_cfg.CENTER_SCALE if lyap_cfg is not None else 2.0
        self.min_separation = lyap_cfg.MIN_SEPARATION if lyap_cfg is not None else 0.6
        self.init_range = lyap_cfg.INIT_RANGE if lyap_cfg is not None else 2.5
        self.extend_mode = lyap_cfg.EXTEND_MODE if lyap_cfg is not None else "embed"
        self.extra_decay = lyap_cfg.EXTRA_DECAY if lyap_cfg is not None else 1.0

        # Stable points (equilibria) - canonical 2D layout
        self._base_points_2d = torch.tensor([
            [-1.0, -1.0], [ 1.0, -1.0], [-1.0,  1.0], [ 1.0,  1.0],
            [ 0.0,  0.0],
            [-1.0, -2.0], [ 1.0, -2.0], [-1.0,  2.0], [ 1.0,  2.0],
            [-2.0, -1.0], [ 2.0, -1.0], [-2.0,  1.0], [ 2.0,  1.0],
        ], dtype=torch.float32)

        self.points_2d, self.points = self._init_points()
        self._sigma2 = float(self.sigma) * float(self.sigma)

    def _sample_points(
        self,
        num_points: int,
        dim: int,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Sample random centers with a minimum separation constraint."""
        points: List[torch.Tensor] = []
        max_tries = max(1000, num_points * 500)
        tries = 0
        while len(points) < num_points and tries < max_tries:
            candidate = (torch.rand(dim, generator=rng) * 2.0 - 1.0) * self.center_scale
            if not points:
                points.append(candidate)
            else:
                dists = torch.stack([torch.norm(candidate - p) for p in points])
                if torch.all(dists >= self.min_separation):
                    points.append(candidate)
            tries += 1
        if len(points) < num_points:
            # Fall back to unconstrained sampling for remaining points
            remaining = num_points - len(points)
            extra = (torch.rand(remaining, dim, generator=rng) * 2.0 - 1.0) * self.center_scale
            points.extend([p for p in extra])
        return torch.stack(points, dim=0)

    def _pad_points(self, points: torch.Tensor, dim: int) -> torch.Tensor:
        if points.shape[1] == dim:
            return points
        padded = torch.zeros(points.shape[0], dim, dtype=points.dtype)
        padded[:, :points.shape[1]] = points
        return padded

    def _init_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize attractor centers (points_2d, points_full)."""
        seed = getattr(self.cfg, 'SEED', 0)
        rng = torch.Generator().manual_seed(seed)

        # For embed mode, always generate centers in 2D and pad
        point_dim = 2 if self.extend_mode == "embed" else self.dim

        if self.points_mode == "fixed":
            points = self._base_points_2d.clone()
            if self.num_basins <= points.shape[0]:
                points = points[:self.num_basins]
            else:
                extra = self._sample_points(self.num_basins - points.shape[0], 2, rng)
                points = torch.cat([points, extra], dim=0)
        elif self.points_mode == "random":
            points = self._sample_points(self.num_basins, point_dim, rng)
        else:
            raise ValueError(f"Unknown POINTS_MODE '{self.points_mode}' (expected 'fixed' or 'random').")

        points_2d = points[:, :2] if points.shape[1] >= 2 else self._pad_points(points, 2)
        points_full = self._pad_points(points, self.dim)
        return points_2d, points_full

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        state = torch.empty(self.dim).uniform_(-self.init_range, self.init_range, generator=rng)
        return state.to(dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        sigma2 = self._sigma2

        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            if self.extend_mode == "embed" and self.dim > 2:
                # 2D Lyapunov dynamics + linear decay for extra dimensions
                state_2d = state[:2]
                diff = state_2d.unsqueeze(0) - self.points_2d  # [M, 2]
                r2 = (diff * diff).sum(dim=1)  # [M]
                normx2 = torch.dot(state_2d, state_2d)

                psi1 = normx2 * torch.exp(-r2 / sigma2)  # [M]
                term1 = (-2.0 / sigma2) * (psi1.unsqueeze(1) * diff).sum(dim=0)  # [2]

                psi2 = torch.exp(-r2 / sigma2)  # [M]
                term2 = -(psi2.unsqueeze(1) * diff).sum(dim=0)  # [2]

                dx2 = term1 + term2
                dx_extra = -self.extra_decay * state[2:]
                return torch.cat([dx2, dx_extra], dim=0)

            # Full-dimensional Lyapunov dynamics
            diff = state.unsqueeze(0) - self.points  # [M, d]
            r2 = (diff * diff).sum(dim=1)  # [M]
            normx2 = torch.dot(state, state)

            psi1 = normx2 * torch.exp(-r2 / sigma2)  # [M]
            term1 = (-2.0 / sigma2) * (psi1.unsqueeze(1) * diff).sum(dim=0)  # [d]

            psi2 = torch.exp(-r2 / sigma2)  # [M]
            term2 = -(psi2.unsqueeze(1) * diff).sum(dim=0)  # [d]

            return term1 + term2

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class MultiWellTransition(Env):
    """Multi-well potential system with deterministic transition mechanisms.

    Core 2D dynamics use smooth wells centered at configurable attractor points.
    Four deterministic variants are supported:
      - gradient: pure descent in the multi-well potential
      - rotational: adds a 90-degree rotated gradient component
      - energy: adds radial deterministic energy injection
      - strong_transition: combines rotational and radial terms

    For dim > 2:
      - embed mode: evolve only the 2D core and linearly decay extra dimensions
      - full mode: run multi-well dynamics in full dimension
    """

    _VALID_MODES = {"gradient", "rotational", "energy", "strong_transition"}

    def __init__(
        self,
        cfg: Config,
        dynamics_mode: Optional[str] = None,
        force_dim: Optional[int] = None,
        force_extend_mode: Optional[str] = None,
    ):
        super().__init__(cfg)
        mw_cfg = cfg.ENV.MULTIWELL if hasattr(cfg.ENV, "MULTIWELL") else None

        self.dt = mw_cfg.DT if mw_cfg is not None else 0.02
        self.sigma = mw_cfg.SIGMA if mw_cfg is not None else 0.7
        self.dim = mw_cfg.DIM if mw_cfg is not None else 2
        self.num_basins = mw_cfg.NUM_BASINS if mw_cfg is not None else 5
        self.points_mode = mw_cfg.POINTS_MODE if mw_cfg is not None else "fixed"
        self.center_scale = mw_cfg.CENTER_SCALE if mw_cfg is not None else 2.0
        self.min_separation = mw_cfg.MIN_SEPARATION if mw_cfg is not None else 0.6
        self.init_range = mw_cfg.INIT_RANGE if mw_cfg is not None else 2.5
        self.extend_mode = mw_cfg.EXTEND_MODE if mw_cfg is not None else "embed"
        self.extra_decay = mw_cfg.EXTRA_DECAY if mw_cfg is not None else 1.0

        base_mode = mw_cfg.DYNAMICS_MODE if mw_cfg is not None else "gradient"
        self.dynamics_mode = (dynamics_mode or base_mode).lower()
        self.alpha = mw_cfg.ALPHA if mw_cfg is not None else 0.6
        self.beta = mw_cfg.BETA if mw_cfg is not None else 1.2
        self.strong_alpha = mw_cfg.STRONG_ALPHA if mw_cfg is not None else 0.8
        self.strong_beta = mw_cfg.STRONG_BETA if mw_cfg is not None else 1.0

        if force_dim is not None:
            self.dim = force_dim
        if force_extend_mode is not None:
            self.extend_mode = force_extend_mode

        if self.dim < 2:
            raise ValueError("MultiWellTransition requires DIM >= 2.")
        if self.dynamics_mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown MULTIWELL.DYNAMICS_MODE '{self.dynamics_mode}'. "
                f"Expected one of {sorted(self._VALID_MODES)}."
            )

        # Canonical 5-point layout matching the user-provided system sketch.
        self._base_points_2d = torch.tensor([
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ], dtype=torch.float32)

        self.points_2d, self.points = self._init_points()
        self._sigma2 = float(self.sigma) * float(self.sigma)

    def _sample_points(
        self,
        num_points: int,
        dim: int,
        rng: torch.Generator,
    ) -> torch.Tensor:
        points: List[torch.Tensor] = []
        max_tries = max(1000, num_points * 500)
        tries = 0
        while len(points) < num_points and tries < max_tries:
            candidate = (torch.rand(dim, generator=rng) * 2.0 - 1.0) * self.center_scale
            if not points:
                points.append(candidate)
            else:
                dists = torch.stack([torch.norm(candidate - p) for p in points])
                if torch.all(dists >= self.min_separation):
                    points.append(candidate)
            tries += 1
        if len(points) < num_points:
            remaining = num_points - len(points)
            extra = (torch.rand(remaining, dim, generator=rng) * 2.0 - 1.0) * self.center_scale
            points.extend([p for p in extra])
        return torch.stack(points, dim=0)

    def _pad_points(self, points: torch.Tensor, dim: int) -> torch.Tensor:
        if points.shape[1] == dim:
            return points
        padded = torch.zeros(points.shape[0], dim, dtype=points.dtype)
        padded[:, :points.shape[1]] = points
        return padded

    def _init_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        seed = getattr(self.cfg, "SEED", 0)
        rng = torch.Generator().manual_seed(seed)
        point_dim = 2 if self.extend_mode == "embed" else self.dim

        if self.points_mode == "fixed":
            points = self._base_points_2d.clone()
            if self.num_basins <= points.shape[0]:
                points = points[:self.num_basins]
            else:
                extra = self._sample_points(self.num_basins - points.shape[0], 2, rng)
                points = torch.cat([points, extra], dim=0)
        elif self.points_mode == "random":
            points = self._sample_points(self.num_basins, point_dim, rng)
        else:
            raise ValueError(
                f"Unknown POINTS_MODE '{self.points_mode}' (expected 'fixed' or 'random')."
            )

        points_2d = points[:, :2] if points.shape[1] >= 2 else self._pad_points(points, 2)
        points_full = self._pad_points(points, self.dim)
        return points_2d, points_full

    def _potential_gradient(self, state: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        diff = state.unsqueeze(0) - points
        r2 = (diff * diff).sum(dim=1)
        weights = torch.exp(-r2 / self._sigma2)
        return 2.0 * (diff * weights.unsqueeze(1)).sum(dim=0)

    @staticmethod
    def _rot90(vec: torch.Tensor) -> torch.Tensor:
        rot = torch.zeros_like(vec)
        if vec.shape[0] >= 2:
            rot[0] = -vec[1]
            rot[1] = vec[0]
        return rot

    def _velocity(self, state: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if self.dynamics_mode == "gradient":
            return -grad
        if self.dynamics_mode == "rotational":
            return -grad + self.alpha * self._rot90(grad)
        if self.dynamics_mode == "energy":
            r = torch.norm(state)
            return -grad + self.beta * torch.sin(r) * state

        # strong_transition
        r = torch.norm(state)
        return (
            -grad
            + self.strong_alpha * self._rot90(grad)
            + self.strong_beta * torch.cos(2.0 * r) * state
        )

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        state = torch.empty(self.dim).uniform_(-self.init_range, self.init_range, generator=rng)
        return state.to(dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            if self.extend_mode == "embed" and self.dim > 2:
                core_state = state[:2]
                grad2 = self._potential_gradient(core_state, self.points_2d)
                dx2 = self._velocity(core_state, grad2)
                dx_extra = -self.extra_decay * state[2:]
                return torch.cat([dx2, dx_extra], dim=0)

            grad = self._potential_gradient(state, self.points)
            return self._velocity(state, grad)

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class BlendedLinearSystem(Env):
    """Synthetic 2D system with 3 basins having genuinely different local dynamics.

    This environment tests whether models learn dynamical regions vs geometric features.
    Unlike Lyapunov (identical Gaussian wells), each basin has distinct eigenstructure:

    Basin 1 (Spiral): Complex eigenvalues, oscillatory approach to equilibrium
        - Center: (2, 0)
        - Eigenvalues: -0.5 ± 3i (damped oscillation)

    Basin 2 (Slow Horizontal): Stiff system with slow X decay, fast Y decay
        - Center: (-2, 2)
        - Eigenvalues: -0.2 (slow), -5.0 (fast)
        - Trajectories look like horizontal lines approaching attractor

    Basin 3 (Fast Vertical): Stiff system with fast X decay, moderate Y decay
        - Center: (-2, -2)
        - Eigenvalues: -5.0 (fast), -2.0 (moderate)
        - Trajectories look like vertical lines approaching attractor

    The global dynamics are a softmax-weighted blend of local linear systems:
        dx/dt = sum_i w_i(x) * A_i * (x - c_i)
    where w_i(x) = softmax(-||x - c_i||^2 / (2*sigma^2))

    Key test: If a model learns geometric features (horizontal/vertical lines) rather
    than dynamical basins, it will confuse Basin 1's spiral (which passes through
    horizontal/vertical orientations) with Basins 2/3's manifold structure.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        # Timestep for integration
        self.dt = 0.05

        # Blending width (controls how sharply basins transition)
        self.sigma = 1.5

        # Basin centers
        self.centers = torch.tensor([
            [2.0, 0.0],    # Basin 1: Spiral (right)
            [-2.0, 2.0],   # Basin 2: Slow horizontal (top-left)
            [-2.0, -2.0],  # Basin 3: Fast vertical (bottom-left)
        ], dtype=torch.float32)

        # Local dynamics matrices A_i
        # Basin 1: Spiral (complex eigenvalues: -0.5 ± 3i)
        A1 = torch.tensor([
            [-0.5, -3.0],
            [3.0, -0.5]
        ], dtype=torch.float32)

        # Basin 2: Slow horizontal decay (eigenvalues: -0.2, -5.0)
        # Fast Y collapse, slow X drift -> horizontal manifold
        A2 = torch.tensor([
            [-0.2, 0.0],
            [0.0, -5.0]
        ], dtype=torch.float32)

        # Basin 3: Fast vertical decay (eigenvalues: -5.0, -2.0)
        # Fast X collapse, moderate Y decay -> vertical manifold
        A3 = torch.tensor([
            [-5.0, 0.0],
            [0.0, -2.0]
        ], dtype=torch.float32)

        self.matrices = torch.stack([A1, A2, A3], dim=0)  # [3, 2, 2]

        # Precompute sigma^2
        self._sigma2 = self.sigma ** 2

    @property
    def action_size(self) -> int:
        return 0

    def _get_weights(self, state: torch.Tensor) -> torch.Tensor:
        """Compute softmax blending weights based on distance to basin centers.

        Args:
            state: Current state [..., 2]

        Returns:
            Weights [..., 3] summing to 1
        """
        # Compute squared distances to each center
        # state: [..., 2], centers: [3, 2]
        diff = state.unsqueeze(-2) - self.centers  # [..., 3, 2]
        dist_sq = (diff ** 2).sum(dim=-1)  # [..., 3]

        # Gaussian weights (unnormalized)
        raw_weights = torch.exp(-dist_sq / (2 * self._sigma2))  # [..., 3]

        # Normalize to sum to 1
        return raw_weights / (raw_weights.sum(dim=-1, keepdim=True) + 1e-8)

    def _dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """Compute the blended vector field dx/dt.

        Args:
            state: Current state [..., 2]

        Returns:
            Velocity [..., 2]
        """
        weights = self._get_weights(state)  # [..., 3]

        # Compute local velocities for each basin
        # diff: [..., 3, 2] = state - center for each basin
        diff = state.unsqueeze(-2) - self.centers  # [..., 3, 2]

        # Local velocity: A_i @ (x - c_i) for each basin
        # matrices: [3, 2, 2], diff: [..., 3, 2]
        # We want: [..., 3, 2] where each [..., i, :] = A_i @ diff[..., i, :]
        local_vel = torch.einsum('ijk,...ik->...ij', self.matrices, diff)  # [..., 3, 2]

        # Weighted sum: sum_i w_i * v_i
        # weights: [..., 3], local_vel: [..., 3, 2]
        velocity = (weights.unsqueeze(-1) * local_vel).sum(dim=-2)  # [..., 2]

        return velocity

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reset to random initial state in [-4, 4]^2."""
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(-4.0, 4.0, generator=rng)
        x2 = torch.empty(1).uniform_(-4.0, 4.0, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Integrate one timestep using RK4."""
        def dynamics_fn(s: torch.Tensor, a: Optional[torch.Tensor] = None) -> torch.Tensor:
            return self._dynamics(s)

        return integrate_rk4(state, None, self.dt, dynamics_fn)

    def get_basin_label(self, state: torch.Tensor) -> torch.Tensor:
        """Get the dominant basin for each state (for evaluation).

        Args:
            state: States [..., 2]

        Returns:
            Basin indices [...] in {0, 1, 2}
        """
        weights = self._get_weights(state)
        return weights.argmax(dim=-1)

    def get_local_jacobian(self, state: torch.Tensor) -> torch.Tensor:
        """Get the dominant basin's Jacobian matrix (ground-truth local dynamics).

        This is useful for evaluating whether the learned K^(k) matches the true A_i.

        Args:
            state: States [..., 2]

        Returns:
            Jacobian matrices [..., 2, 2]
        """
        basin_idx = self.get_basin_label(state)  # [...]
        return self.matrices[basin_idx]  # [..., 2, 2]


# ---------------------------------------------------------------------------
# Registry and Factory
# ---------------------------------------------------------------------------


_ENV_REGISTRY = {
    "pendulum": Pendulum,
    "duffing": Duffing,
    "lotka_volterra": LotkaVolterra,
    "lorenz63": Lorenz63,
    "parabolic": Parabolic,
    "lyapunov": LyapunovMultiAttractor,
    "blended": BlendedLinearSystem,
    "multiwell": MultiWellTransition,
    "multiwell_gradient": lambda cfg: MultiWellTransition(cfg, dynamics_mode="gradient"),
    "multiwell_rotational": lambda cfg: MultiWellTransition(cfg, dynamics_mode="rotational"),
    "multiwell_energy": lambda cfg: MultiWellTransition(cfg, dynamics_mode="energy"),
    "multiwell_strong_transition": lambda cfg: MultiWellTransition(cfg, dynamics_mode="strong_transition"),
    # Fixed 8D aliases for parity with existing Lyapunov-HD experiments.
    "multiwell_gradient_hd": lambda cfg: MultiWellTransition(
        cfg, dynamics_mode="gradient", force_dim=8, force_extend_mode="embed"
    ),
    "multiwell_rotational_hd": lambda cfg: MultiWellTransition(
        cfg, dynamics_mode="rotational", force_dim=8, force_extend_mode="embed"
    ),
    "multiwell_energy_hd": lambda cfg: MultiWellTransition(
        cfg, dynamics_mode="energy", force_dim=8, force_extend_mode="embed"
    ),
    "multiwell_strong_transition_hd": lambda cfg: MultiWellTransition(
        cfg, dynamics_mode="strong_transition", force_dim=8, force_extend_mode="embed"
    ),
}


def make_env(cfg: Config) -> Env:
    """Factory function to create environment from configuration.
    
    Supports both built-in environments and dysts systems.
    
    For built-in environments:
        cfg.ENV.ENV_NAME should be one of: "duffing", "pendulum", "lotka_volterra",
        "lorenz63", "parabolic", "lyapunov", "blended", "multiwell",
        "multiwell_<variant>", "multiwell_<variant>_hd"
    
    For dysts systems:
        cfg.ENV.ENV_NAME should be "dysts:SystemName" (e.g., "dysts:Lorenz", "dysts:Chua")
        or just the system name directly (e.g., "Lorenz") if not in the built-in registry.
    
    Args:
        cfg: Configuration object with ENV.ENV_NAME specifying the system
        
    Returns:
        Environment instance
        
    Raises:
        ValueError: If ENV_NAME is not found in registry or dysts
    """
    env_name = cfg.ENV.ENV_NAME
    
    # Check for explicit dysts prefix
    if env_name.startswith("dysts:"):
        system_name = env_name.split(":", 1)[1]
        return _make_dysts_env(system_name, cfg)

    # Allow mode-specific multiwell syntax: "multiwell:<mode>"
    if env_name.startswith("multiwell:"):
        mode = env_name.split(":", 1)[1]
        return MultiWellTransition(cfg, dynamics_mode=mode)
    
    # Check built-in registry first
    if env_name in _ENV_REGISTRY:
        return _ENV_REGISTRY[env_name](cfg)
    
    # Try as a dysts system name (without prefix)
    # This allows using system names directly like "Lorenz" instead of "dysts:Lorenz"
    try:
        return _make_dysts_env(env_name, cfg)
    except (ImportError, ValueError):
        pass
    
    # Not found anywhere
    raise ValueError(
        f"Unknown environment '{env_name}'. "
        f"Built-in environments: {list(_ENV_REGISTRY.keys())}. "
        f"For dysts systems, use 'dysts:SystemName' (e.g., 'dysts:Lorenz')."
    )


def _make_dysts_env(system_name: str, cfg: Config):
    """Create a dysts environment with config settings.
    
    Args:
        system_name: Name of the dysts system (e.g., "Lorenz", "Chua")
        cfg: Configuration object
        
    Returns:
        DystsEnv instance
    """
    try:
        from skae.benchmarks.dysts_adapter import DystsEnv, is_dysts_available
    except ImportError:
        raise ImportError(
            "benchmarks module not found. Make sure the benchmarks/ directory exists."
        )
    
    if not is_dysts_available():
        raise ImportError(
            "dysts library is not available. Install with: pip install dysts "
            "or ensure the dysts submodule is present at ./dysts/"
        )
    
    # Get dysts-specific config
    dt_override = cfg.ENV.DYSTS.DT_OVERRIDE if cfg.ENV.DYSTS.DT_OVERRIDE > 0 else None
    ic_noise_scale = cfg.ENV.DYSTS.IC_NOISE_SCALE
    standardize = cfg.ENV.DYSTS.STANDARDIZE
    
    return DystsEnv(
        system_name=system_name,
        dt_override=dt_override,
        ic_noise_scale=ic_noise_scale,
        standardize=standardize,
    )


def get_available_environments() -> dict:
    """Get information about all available environments.
    
    Returns:
        Dictionary with 'builtin' and 'dysts' keys listing available systems.
    """
    result = {
        "builtin": list(_ENV_REGISTRY.keys()),
        "dysts": [],
    }
    
    try:
        from skae.benchmarks.dysts_adapter import get_dysts_systems, is_dysts_available
        if is_dysts_available():
            result["dysts"] = get_dysts_systems()
    except ImportError:
        pass
    
    return result
