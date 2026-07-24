"""Adapter to wrap dysts DynSys objects as SKAE Env instances.

This module provides a seamless bridge between the dysts library (135+ chaotic systems)
and the SKAE training/evaluation infrastructure.

Example:
    >>> from benchmarks.dysts_adapter import DystsEnv
    >>> env = DystsEnv("Lorenz")
    >>> x0 = env.reset()
    >>> x1 = env.step(x0)
"""

from functools import lru_cache
from typing import Optional, List, Dict, Any
import warnings

import torch
import numpy as np

# Try to import dysts - handle both installed package and local submodule
HAS_DYSTS = False
flows = None
DynSys = None
DynSysDelay = None
get_attractor_list = None
get_system_data = None

try:
    # First try: import as installed package
    from dysts import flows as _flows
    from dysts.base import DynSys as _DynSys, DynSysDelay as _DynSysDelay
    from dysts.systems import get_attractor_list as _get_attractor_list, get_system_data as _get_system_data
    flows = _flows
    DynSys = _DynSys
    DynSysDelay = _DynSysDelay
    get_attractor_list = _get_attractor_list
    get_system_data = _get_system_data
    HAS_DYSTS = True
except ImportError:
    try:
        # Second try: local submodule at ksae/dysts/dysts/
        import sys
        from pathlib import Path
        # The submodule is at ksae/dysts/ which contains the dysts/ package
        dysts_parent = Path(__file__).parent.parent / "dysts"
        if dysts_parent.exists() and (dysts_parent / "dysts").exists():
            sys.path.insert(0, str(dysts_parent))
            from dysts import flows as _flows
            from dysts.base import DynSys as _DynSys, DynSysDelay as _DynSysDelay
            from dysts.systems import get_attractor_list as _get_attractor_list, get_system_data as _get_system_data
            flows = _flows
            DynSys = _DynSys
            DynSysDelay = _DynSysDelay
            get_attractor_list = _get_attractor_list
            get_system_data = _get_system_data
            HAS_DYSTS = True
    except ImportError as e:
        # Failed to import - dysts not available
        pass


def is_dysts_available() -> bool:
    """Check if dysts library is available."""
    return HAS_DYSTS


def get_dysts_systems(include_delay: bool = False) -> List[str]:
    """Get list of available dysts systems.
    
    Args:
        include_delay: Whether to include delay differential equation systems.
        
    Returns:
        Sorted list of system names.
    """
    if not HAS_DYSTS:
        return []
    
    if include_delay:
        return get_attractor_list("continuous")
    else:
        return get_attractor_list("continuous_no_delay")


@lru_cache(maxsize=2)
def _cached_dysts_system_data(data_key: str) -> Dict[str, Any]:
    """Cache raw dysts metadata lookups to avoid repeated large loads."""
    return get_system_data(data_key)


def get_dysts_system_data(include_delay: bool = False) -> Dict[str, Any]:
    """Get metadata for all available dysts systems.
    
    Args:
        include_delay: Whether to include delay differential equation systems.
        
    Returns:
        Dictionary mapping system name to metadata.
    """
    if not HAS_DYSTS:
        return {}
    
    data_key = "continuous" if include_delay else "continuous_no_delay"
    return dict(_cached_dysts_system_data(data_key))


def get_dysts_system_metadata(system_name: str) -> Dict[str, Any]:
    """Get metadata for a dysts system.
    
    Args:
        system_name: Name of the dysts system (e.g., "Lorenz").
        
    Returns:
        Dictionary with system metadata (dimension, dt, period, parameters, etc.)
    """
    if not HAS_DYSTS:
        raise RuntimeError("dysts library is not available")
    
    all_data = _cached_dysts_system_data("continuous")
    if system_name in all_data:
        return all_data[system_name]
    
    # Fallback: instantiate the system to get metadata
    system_class = getattr(flows, system_name, None)
    if system_class is None:
        raise ValueError(f"Unknown dysts system: {system_name}")
    
    system = system_class()
    return {
        "dimension": system.dimension,
        "dt": getattr(system, "dt", 0.01),
        "period": getattr(system, "period", None),
        "parameters": system.params,
        "initial_conditions": system.ic.tolist() if hasattr(system, "ic") else None,
    }


def _to_float32_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert numpy output to a contiguous float32 tensor without torch-side casting."""
    out_np = np.asarray(array, dtype=np.float32)
    if not out_np.flags.c_contiguous:
        out_np = np.ascontiguousarray(out_np, dtype=np.float32)
    return torch.from_numpy(out_np)


class DystsEnv:
    """Wrapper that adapts a dysts DynSys to the SKAE Env interface.
    
    This class provides seamless integration between dysts chaotic systems
    and the SKAE training/evaluation infrastructure.
    
    Key adaptations:
    1. Uses dysts's default dt and initial conditions
    2. Converts between numpy (dysts) and PyTorch (SKAE)
    3. Implements step() using RK4 integration with dysts's rhs() method
    4. Provides consistent reset() with perturbations around default IC
    
    Args:
        system_name: Name of the dysts system (e.g., "Lorenz", "Rossler", "Chua")
        dt_override: Override for integration timestep (if None, uses dysts default)
        ic_noise_scale: Scale for perturbation noise around default initial condition
        
    Example:
        >>> env = DystsEnv("Lorenz")
        >>> print(f"Dimension: {env.observation_size}")
        Dimension: 3
        >>> x0 = env.reset()
        >>> x1 = env.step(x0)
    """
    
    def __init__(
        self, 
        system_name: str, 
        dt_override: Optional[float] = None,
        ic_noise_scale: float = 0.2,
        standardize: bool = False,
    ):
        if not HAS_DYSTS:
            raise RuntimeError(
                "dysts library is not available. Install it with: "
                "pip install dysts or ensure the dysts submodule is present."
            )
        
        # Get the system class and instantiate
        system_class = getattr(flows, system_name, None)
        if system_class is None:
            available = get_dysts_systems()[:10]
            raise ValueError(
                f"Unknown dysts system: '{system_name}'. "
                f"Available systems include: {available}... "
                f"Use get_dysts_systems() to see all {len(get_dysts_systems())} systems."
            )
        
        self.system: DynSys = system_class()
        self.system_name = system_name
        self.ic_noise_scale = ic_noise_scale
        self.standardize = standardize
        
        # Check if this is a delay system (not fully supported yet)
        if isinstance(self.system, DynSysDelay):
            warnings.warn(
                f"{system_name} is a delay differential equation system. "
                "DDE support is experimental and may not work correctly."
            )
        
        # Keep the wrapper and the native dysts integrator on the same sampling
        # interval.  ``DynSys.make_trajectory`` prefers ``self.dt`` over its
        # call-time ``dt`` argument, so updating only the wrapper would silently
        # generate native-dt trajectories while labeling them with the override.
        self.native_dt = self.system.dt
        self.dt = dt_override if dt_override is not None else self.native_dt
        if dt_override is not None:
            self.system.dt = float(dt_override)
        
        # Cache dimension
        self._dim = self.system.dimension
        
        # Store IC and statistics from dysts metadata
        self._ic = torch.tensor(self.system.ic, dtype=torch.float32)
        self._mean = torch.tensor(
            getattr(self.system, "mean", np.zeros(self._dim)), 
            dtype=torch.float32
        )
        self._std = torch.tensor(
            getattr(self.system, "std", np.ones(self._dim)), 
            dtype=torch.float32
        )
        
        # Period and Lyapunov exponent for optional analysis
        self.period = getattr(self.system, "period", None)
        self.lyapunov = getattr(self.system, "maximum_lyapunov_estimated", None)
        
        # Config placeholder for compatibility
        self.cfg = None
        
        # Cache whether batch RHS is supported
        self._batch_rhs_supported: Optional[bool] = None
    
    @property
    def observation_size(self) -> int:
        """Dimensionality of the state space."""
        return self._dim
    
    @property
    def action_size(self) -> int:
        """Dimensionality of the action space (always 0 for autonomous systems)."""
        return 0
    
    @property
    def unwrapped(self) -> "DystsEnv":
        """Return the unwrapped environment."""
        return self
    
    def _to_standardized(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw state to standardized (zero mean, unit variance)."""
        return (x - self._mean) / (self._std + 1e-8)
    
    def _from_standardized(self, x: torch.Tensor) -> torch.Tensor:
        """Convert standardized state back to raw coordinates."""
        return x * self._std + self._mean
    
    def _rhs_numpy(self, x_np: np.ndarray) -> np.ndarray:
        """Compute derivatives via dysts rhs using numpy inputs."""
        if x_np.ndim == 1:
            dx_np = self.system.rhs(x_np, t=0)
            return np.asarray(dx_np, dtype=np.float32)
        
        if self._batch_rhs_supported is False:
            return np.stack([self._rhs_numpy(x) for x in x_np], axis=0)
        
        if self._batch_rhs_supported is None:
            try:
                dx_np = self.system.rhs(x_np, t=0)
                dx_np = np.asarray(dx_np, dtype=np.float32)
                if dx_np.shape != x_np.shape:
                    raise ValueError("Batched rhs returned unexpected shape.")
                self._batch_rhs_supported = True
                return dx_np
            except Exception:
                self._batch_rhs_supported = False
                return np.stack([self._rhs_numpy(x) for x in x_np], axis=0)
        
        try:
            dx_np = self.system.rhs(x_np, t=0)
            dx_np = np.asarray(dx_np, dtype=np.float32)
            if dx_np.shape != x_np.shape:
                raise ValueError("Batched rhs returned unexpected shape.")
            return dx_np
        except Exception:
            self._batch_rhs_supported = False
            return np.stack([self._rhs_numpy(x) for x in x_np], axis=0)
    
    def _rk4_numpy(self, x_np: np.ndarray) -> np.ndarray:
        """RK4 integration using numpy arrays (supports batched input if rhs does)."""
        dt = self.dt
        k1 = self._rhs_numpy(x_np)
        k2 = self._rhs_numpy(x_np + 0.5 * dt * k1)
        k3 = self._rhs_numpy(x_np + 0.5 * dt * k2)
        k4 = self._rhs_numpy(x_np + dt * k3)
        return x_np + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reset to a random initial condition near the default IC.
        
        Args:
            rng: Random number generator for reproducibility.
            
        Returns:
            Initial state tensor of shape [observation_size].
            If standardize=True, returns standardized coordinates.
        """
        if rng is None:
            rng = torch.Generator()
        
        # Sample perturbation around default IC
        # Scale by std for appropriate spread relative to attractor size
        noise = torch.randn(self._dim, generator=rng) * self._std * self.ic_noise_scale
        state = self._ic + noise
        
        if self.standardize:
            return self._to_standardized(state)
        return state
    
    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advance state by one timestep using RK4 integration.
        
        Args:
            state: Current state tensor of shape [observation_size] or [batch, observation_size]
                   If standardize=True, expects standardized coordinates.
            action: Ignored (for API compatibility)
            
        Returns:
            Next state tensor with same shape as input.
            If standardize=True, returns standardized coordinates.
        """
        # Ensure CPU tensor for numpy-based integration
        state_cpu = state.detach().cpu().contiguous()
        
        # Convert from standardized to raw if needed
        if self.standardize:
            state_cpu = self._from_standardized(state_cpu)
        
        # Handle batched input
        if state_cpu.dim() == 1:
            result = self._step_single(state_cpu)
        else:
            result = self._step_batch(state_cpu)
        
        # Convert back to standardized if needed
        if self.standardize:
            result = self._to_standardized(result)
        
        return result
    
    def _step_single(self, state: torch.Tensor) -> torch.Tensor:
        """Single-state RK4 integration step."""
        state_cpu = state.detach().cpu().contiguous()
        out_np = self._rk4_numpy(state_cpu.numpy())
        return _to_float32_tensor(out_np)
    
    def _step_batch(self, state: torch.Tensor) -> torch.Tensor:
        """Batched RK4 integration step using numpy."""
        state_cpu = state.detach().cpu().contiguous()
        out_np = self._rk4_numpy(state_cpu.numpy())
        return _to_float32_tensor(out_np)
    
    def make_trajectory(self, n: int, init_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate a trajectory of n steps.
        
        This is a convenience method that wraps repeated step() calls.
        For long trajectories, consider using dysts's native make_trajectory()
        for better performance (uses scipy integrators).
        
        Args:
            n: Number of timesteps.
            init_cond: Initial condition. If None, uses default IC.
            
        Returns:
            Trajectory tensor of shape [n, observation_size].
        """
        if init_cond is None:
            init_cond = self._ic.clone()
        
        trajectory = [init_cond]
        state = init_cond
        
        for _ in range(n - 1):
            state = self._step_single(state)
            trajectory.append(state)
        
        return torch.stack(trajectory)
    
    def make_trajectory_native(
        self, 
        n: int, 
        resample: bool = True,
        pts_per_period: int = 100,
        standardize: bool = False,
        method: str = "Radau",
        rtol: float = 1e-12,
        atol: float = 1e-12,
    ) -> torch.Tensor:
        """Generate trajectory using dysts's native integrator.
        
        This uses scipy's adaptive integrators for potentially better accuracy
        on stiff systems. The trajectory is returned as a PyTorch tensor.
        
        Args:
            n: Number of points in trajectory.
            resample: Whether to resample to match dominant Fourier components.
            pts_per_period: Points per period if resampling.
            standardize: Whether to z-score standardize the trajectory.
            method: scipy ``solve_ivp`` integration method.
            rtol: Relative integration tolerance.
            atol: Absolute integration tolerance.
            
        Returns:
            Trajectory tensor of shape [n, observation_size].
        """
        traj_np = self.system.make_trajectory(
            n=n,
            dt=float(self.dt),
            resample=resample,
            pts_per_period=pts_per_period,
            standardize=standardize,
            method=str(method),
            rtol=float(rtol),
            atol=float(atol),
        )
        
        if traj_np is None:
            raise RuntimeError(f"Failed to generate trajectory for {self.system_name}")

        return _to_float32_tensor(traj_np)
    
    def __repr__(self) -> str:
        return (
            f"DystsEnv(system={self.system_name}, dim={self._dim}, "
            f"dt={self.dt:.4f}, period={self.period})"
        )
