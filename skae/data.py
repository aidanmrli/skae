"""
Data module for dynamical systems environments.

Provides PyTorch-based implementations of various nonlinear dynamical systems
structured as environments.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import hashlib
import json
import math
import os
from pathlib import Path
import signal
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


def _manual_seed_generator(seed: int) -> torch.Generator:
    """Return a CPU generator seeded deterministically from an integer."""
    return torch.Generator().manual_seed(int(seed))


def _infer_reset_bounds(env: Env) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer axis-aligned reset bounds for a black-box environment.

    The fixed transition-rich shortlist exposes either an `init_range` scalar or
    a Claude-catalog `ic_box`. For generic environments we fall back to an
    empirical bounding box from deterministic reset samples.
    """

    base_env = env.unwrapped
    if hasattr(base_env, "init_range"):
        init_range = float(getattr(base_env, "init_range"))
        dim = int(base_env.observation_size)
        low = torch.full((dim,), -init_range, dtype=torch.float32)
        high = torch.full((dim,), init_range, dtype=torch.float32)
        return low, high

    system = getattr(base_env, "system", None)
    if system is not None and hasattr(system, "ic_box"):
        ic_box = list(getattr(system, "ic_box"))
        low = torch.tensor([float(lo) for lo, _ in ic_box], dtype=torch.float32)
        high = torch.tensor([float(hi) for _, hi in ic_box], dtype=torch.float32)
        return low, high

    samples = []
    for sample_idx in range(64):
        samples.append(base_env.reset(_manual_seed_generator(10_000 + sample_idx)).float())
    batch = torch.stack(samples, dim=0)
    low = batch.min(dim=0).values
    high = batch.max(dim=0).values
    padding = 0.05 * (high - low).clamp_min(1e-3)
    return low - padding, high + padding


def build_hard_initial_condition_pool(
    env: Env,
    *,
    pool_size: int,
    num_candidates: int,
    probe_steps: int,
    num_perturbations: int,
    perturb_scale: float,
    transient_window: int,
    transient_weight: float,
    chunk_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a pool of hard initial conditions using a uniform black-box score.

    Each candidate is scored by:
    1. how much small perturbations spread apart after a short rollout
    2. how much motion remains late in that rollout (long transient proxy)

    This approximates separatrix proximity and slow settling without relying on
    training-time basin labels or basin counts.
    """

    candidate_count = max(1, int(num_candidates))
    pool_size = max(1, min(int(pool_size), candidate_count))
    probe_steps = max(1, int(probe_steps))
    num_perturbations = max(0, int(num_perturbations))
    transient_window = max(1, min(int(transient_window), probe_steps))
    chunk_size = max(1, int(chunk_size))
    transient_weight = float(transient_weight)

    base_env = env.unwrapped
    low, high = _infer_reset_bounds(base_env)
    span = (high - low).clamp_min(1e-6)
    box_diag = torch.linalg.vector_norm(span).clamp_min(1e-6)
    perturb_std = float(perturb_scale) * span
    perturb_norm = torch.linalg.vector_norm(perturb_std).clamp_min(1e-6)

    candidate_states = []
    base_seed = int(seed)
    for candidate_idx in range(candidate_count):
        candidate_states.append(
            base_env.reset(_manual_seed_generator(base_seed + candidate_idx)).float()
        )
    candidates = torch.stack(candidate_states, dim=0)

    scorer_env = VectorWrapper(base_env, batch_size=1)
    low_batch = low.unsqueeze(0).unsqueeze(0)
    high_batch = high.unsqueeze(0).unsqueeze(0)
    scores = torch.empty(candidate_count, dtype=torch.float32)
    noise_rng = _manual_seed_generator(base_seed + 1_000_000)

    with torch.no_grad():
        for start in range(0, candidate_count, chunk_size):
            stop = min(candidate_count, start + chunk_size)
            chunk = candidates[start:stop]
            batch_n, state_dim = chunk.shape

            if num_perturbations > 0:
                noise = torch.randn(
                    batch_n,
                    num_perturbations,
                    state_dim,
                    generator=noise_rng,
                    dtype=chunk.dtype,
                ) * perturb_std.view(1, 1, -1).to(dtype=chunk.dtype)
                rollout_init = torch.cat([chunk.unsqueeze(1), chunk.unsqueeze(1) + noise], dim=1)
            else:
                rollout_init = chunk.unsqueeze(1)
            rollout_init = torch.maximum(torch.minimum(rollout_init, high_batch), low_batch)
            flat_init = rollout_init.reshape(-1, state_dim)
            rollout = generate_sequence_window(
                scorer_env.step,
                flat_init,
                window_length=probe_steps,
            )
            rollout = rollout.reshape(probe_steps + 1, batch_n, rollout_init.shape[1], state_dim)
            endpoints = rollout[-1]
            tail_start = rollout[-1 - transient_window]

            if num_perturbations > 0:
                sensitivity = (
                    torch.linalg.vector_norm(endpoints[:, 1:] - endpoints[:, :1], dim=-1).mean(dim=1)
                    / perturb_norm
                )
            else:
                sensitivity = torch.zeros(batch_n, dtype=torch.float32)

            late_motion = (
                torch.linalg.vector_norm(endpoints - tail_start, dim=-1).mean(dim=1) / box_diag
            )
            scores[start:stop] = sensitivity.float() + transient_weight * late_motion.float()

    top_scores, top_indices = torch.topk(scores, k=pool_size, largest=True, sorted=True)
    return candidates[top_indices], top_scores


class HardInitialConditionWrapper(Wrapper):
    """Training-time oversampling wrapper for separatrix-adjacent initial states."""

    def __init__(self, env: Env, cfg: Config):
        super().__init__(env)
        settings = cfg.TRAIN.HARD_INIT_OVERSAMPLE
        self.enabled = bool(settings.ENABLED)
        self.fraction = float(settings.FRACTION)
        self.jitter_scale = float(settings.JITTER_SCALE)
        self.low, self.high = _infer_reset_bounds(env.unwrapped)
        self.hard_pool = torch.empty(
            (0, env.observation_size),
            dtype=torch.float32,
        )
        self.hard_scores = torch.empty(0, dtype=torch.float32)
        if not self.enabled:
            return
        self.hard_pool, self.hard_scores = build_hard_initial_condition_pool(
            env.unwrapped,
            pool_size=int(settings.POOL_SIZE),
            num_candidates=int(settings.NUM_CANDIDATES),
            probe_steps=int(settings.PROBE_STEPS),
            num_perturbations=int(settings.NUM_PERTURBATIONS),
            perturb_scale=float(settings.PERTURB_SCALE),
            transient_window=int(settings.TRANSIENT_WINDOW),
            transient_weight=float(settings.TRANSIENT_WEIGHT),
            chunk_size=int(settings.BUILD_CHUNK_SIZE),
            seed=int(cfg.SEED) + 2_000_000,
        )
        top = self.hard_scores[: min(5, self.hard_scores.numel())].tolist()
        print(
            "[hard-init] built pool "
            f"(size={self.hard_pool.shape[0]}, fraction={self.fraction:.2f}, "
            f"top_scores={[round(float(item), 3) for item in top]})",
            flush=True,
        )

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if (
            not self.enabled
            or self.hard_pool.numel() == 0
            or self.fraction <= 0.0
        ):
            return self.env.reset(rng)

        if rng is None:
            rng = torch.Generator()
        use_hard = torch.rand(1, generator=rng).item() < self.fraction
        if not use_hard:
            return self.env.reset(rng)

        index = int(
            torch.randint(
                low=0,
                high=self.hard_pool.shape[0],
                size=(1,),
                generator=rng,
            ).item()
        )
        state = self.hard_pool[index].clone()
        span = (self.high - self.low).clamp_min(1e-6)
        jitter_std = self.jitter_scale * 0.04 * span
        if self.jitter_scale > 0.0:
            jitter = torch.randn(
                state.shape,
                generator=rng,
                dtype=state.dtype,
            ) * jitter_std.to(dtype=state.dtype)
            state = state + jitter
        return torch.maximum(torch.minimum(state, self.high), self.low)


def wrap_training_env(env: Env, cfg: Config) -> Env:
    """Apply training-only environment wrappers without touching eval/reset defaults."""
    if bool(cfg.TRAIN.HARD_INIT_OVERSAMPLE.ENABLED):
        return HardInitialConditionWrapper(env, cfg)
    return env


# ---------------------------------------------------------------------------
# Dysts trajectory cache for training
# ---------------------------------------------------------------------------

class DystsTrajectoryTimeout(TimeoutError):
    """Raised when one native Dysts integration exceeds its wall budget."""


def _run_with_wall_timeout(callable_, timeout_seconds: float):
    """Run a worker-local integration with a POSIX wall-clock timeout."""

    timeout_seconds = float(timeout_seconds)
    if timeout_seconds <= 0.0:
        return callable_()
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise RuntimeError("Dysts trajectory timeouts require POSIX SIGALRM support")

    def _handle_timeout(_signum, _frame):
        raise DystsTrajectoryTimeout(
            f"Dysts trajectory exceeded {timeout_seconds:.1f}s wall time"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return callable_()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_dysts_native_trajectory(
    system_name: str,
    dt: float,
    ic: np.ndarray,
    n_steps: int,
    resample: bool,
    pts_per_period: int,
    standardize: bool,
    primary_method: str = "Radau",
    primary_timeout_seconds: float = 0.0,
    fallback_method: str = "",
    fallback_timeout_seconds: float = 0.0,
    trajectory_index: int = -1,
) -> Tuple[torch.Tensor, str]:
    """Worker helper for parallel dysts trajectory generation."""
    from skae.benchmarks.dysts_adapter import DystsEnv

    def _integrate(method: str, timeout_seconds: float) -> torch.Tensor:
        worker_env = DystsEnv(
            system_name=system_name,
            dt_override=float(dt),
            ic_noise_scale=0.0,
            standardize=standardize,
        )
        if hasattr(worker_env.system, "ic"):
            worker_env.system.ic = np.array(ic, dtype=np.float32)
        return _run_with_wall_timeout(
            lambda: worker_env.make_trajectory_native(
                n=n_steps,
                resample=resample,
                pts_per_period=pts_per_period,
                standardize=standardize,
                method=method,
                rtol=1e-12,
                atol=1e-12,
            ),
            timeout_seconds,
        )

    primary_method = str(primary_method or "Radau")
    try:
        return _integrate(primary_method, primary_timeout_seconds), primary_method
    except DystsTrajectoryTimeout:
        fallback_method = str(fallback_method or "").strip()
        if not fallback_method:
            raise
        print(
            "[dysts cache] trajectory "
            f"{trajectory_index} timed out under {primary_method}; "
            f"retrying the same IC with {fallback_method}",
            flush=True,
        )
        return (
            _integrate(fallback_method, fallback_timeout_seconds),
            fallback_method,
        )


def _init_dysts_cache_worker() -> None:
    """Keep torch CPU threading minimal inside spawned cache workers.

    Forked PyTorch workers can deadlock in OpenMP-backed copy paths when they
    inherit a parent process that already imported torch. Spawning fresh workers
    plus forcing single-threaded torch in the worker avoids that class of hangs.
    """
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


class DystsTrajectoryCache:
    """In-memory cache of dysts trajectories for fast batch sampling."""

    _CACHE_SCHEMA_VERSION = 3

    # Cache IC construction is intentionally deterministic and independent of model seed
    # so cache files can be reused across runs with different training seeds.
    _CACHE_BUILD_SEED = 0

    @staticmethod
    def _split_seed_offset(split: str) -> int:
        """Deterministic offset per cache split namespace."""
        normalized = (split or "train").strip().lower()
        if normalized == "train":
            return 0
        if normalized == "val":
            return 1
        if normalized == "test":
            return 2
        if normalized == "policy":
            return 3
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 1_000_000

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
        self.cache_dir = str(getattr(cfg.ENV.DYSTS, "CACHE_DIR", "")).strip()
        self.cache_reuse = bool(getattr(cfg.ENV.DYSTS, "CACHE_REUSE", False))
        self.cache_split = str(getattr(cfg.ENV.DYSTS, "CACHE_SPLIT", "train")).strip().lower() or "train"
        self.cache_num_workers = max(1, int(getattr(cfg.ENV.DYSTS, "CACHE_NUM_WORKERS", 1)))
        self.cache_primary_method = str(
            getattr(cfg.ENV.DYSTS, "CACHE_PRIMARY_METHOD", "Radau") or "Radau"
        )
        self.cache_trajectory_timeout_seconds = max(
            0.0,
            float(
                getattr(
                    cfg.ENV.DYSTS,
                    "CACHE_TRAJECTORY_TIMEOUT_SECONDS",
                    0.0,
                )
            ),
        )
        self.cache_timeout_fallback_method = str(
            getattr(cfg.ENV.DYSTS, "CACHE_TIMEOUT_FALLBACK_METHOD", "") or ""
        ).strip()
        self.cache_fallback_timeout_seconds = max(
            0.0,
            float(
                getattr(
                    cfg.ENV.DYSTS,
                    "CACHE_FALLBACK_TIMEOUT_SECONDS",
                    0.0,
                )
            ),
        )
        self._cache_integration_method_counts: dict[str, int] = {}

        if self.cache_steps < 2:
            raise ValueError("Dysts cache requires CACHE_STEPS >= 2")

        # Keep constructor arg for backward compatibility; cache generation itself
        # is intentionally seed-free from the training run perspective.
        _ = rng
        split_seed = self._CACHE_BUILD_SEED + self._split_seed_offset(self.cache_split)
        self._cache_build_rng = torch.Generator().manual_seed(split_seed)
        self._trajectories = self._build_cache()

    def _cache_path(self, base_ic: np.ndarray, base_std: np.ndarray) -> Optional[Path]:
        if not self.cache_dir or not self.cache_reuse:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        payload = {
            "version": self._CACHE_SCHEMA_VERSION,
            "system": str(getattr(self.env, "system_name", "unknown")),
            "dt": float(getattr(self.env, "dt", 0.0)),
            "cache_split": self.cache_split,
            "cache_steps": int(self.cache_steps),
            "cache_trajectories": int(self.cache_trajectories),
            "cache_warmup": int(self.cache_warmup),
            "standardize": bool(self.standardize),
            "resample": bool(self.resample),
            "pts_per_period": int(self.pts_per_period),
            "ic_noise_scale": float(getattr(self.env, "ic_noise_scale", 0.0)),
            "base_ic": np.asarray(base_ic, dtype=np.float32).round(6).tolist(),
            "base_std": np.asarray(base_std, dtype=np.float32).round(6).tolist(),
        }
        digest = hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        safe_system = str(payload["system"]).replace(":", "_")
        safe_split = str(payload["cache_split"]).replace(":", "_")
        return Path(self.cache_dir) / f"{safe_system}_{safe_split}_{digest}.pt"

    def _load_cached_trajectories(self, cache_path: Optional[Path]) -> Optional[torch.Tensor]:
        if cache_path is None or not cache_path.exists():
            return None
        try:
            payload = torch.load(cache_path, map_location="cpu")
            if not isinstance(payload, dict):
                print(f"[dysts cache] legacy payload at {cache_path}; rebuilding.")
                return None
            meta = payload.get("meta", {})
            expected_dt = float(getattr(self.env, "dt", 0.0))
            if int(meta.get("schema_version", -1)) != self._CACHE_SCHEMA_VERSION:
                print(f"[dysts cache] stale schema at {cache_path}; rebuilding.")
                return None
            if not math.isclose(
                float(meta.get("dt", float("nan"))),
                expected_dt,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                print(f"[dysts cache] timestep mismatch at {cache_path}; rebuilding.")
                return None
            expected_meta = {
                "system": str(getattr(self.env, "system_name", "unknown")),
                "cache_split": self.cache_split,
                "cache_steps": int(self.cache_steps),
                "cache_trajectories": int(self.cache_trajectories),
                "cache_warmup": int(self.cache_warmup),
                "standardize": bool(self.standardize),
                "resample": bool(self.resample),
                "pts_per_period": int(self.pts_per_period),
                "ic_noise_scale": float(getattr(self.env, "ic_noise_scale", 0.0)),
            }
            for key, expected in expected_meta.items():
                if meta.get(key) != expected:
                    print(
                        f"[dysts cache] metadata mismatch for {key} at "
                        f"{cache_path}; rebuilding.",
                        flush=True,
                    )
                    return None
            trajectories = payload.get("trajectories")
            if not isinstance(trajectories, torch.Tensor):
                print(f"[dysts cache] invalid tensor payload at {cache_path}; rebuilding.")
                return None
            expected_shape = (
                self.cache_trajectories,
                self.cache_steps + 1,
                int(self.env.observation_size),
            )
            if tuple(trajectories.shape) != expected_shape:
                print(
                    f"[dysts cache] invalid tensor shape {tuple(trajectories.shape)} "
                    f"!= {expected_shape} at {cache_path}; rebuilding."
                )
                return None
            if trajectories.dtype != torch.float32:
                print(f"[dysts cache] non-float32 payload at {cache_path}; rebuilding.")
                return None
            if not torch.isfinite(trajectories).all():
                print(f"[dysts cache] nonfinite payload at {cache_path}; rebuilding.")
                return None
            print(
                f"[dysts cache] cache hit: loaded {trajectories.shape[0]} trajectories "
                f"from {cache_path}",
                flush=True,
            )
            return trajectories.float()
        except Exception as e:
            print(f"[dysts cache] failed loading {cache_path}: {e}. Rebuilding.")
            return None

    def _save_cached_trajectories(
        self,
        cache_path: Optional[Path],
        trajectories: torch.Tensor,
    ) -> None:
        if cache_path is None:
            return
        tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp-{os.getpid()}")
        try:
            torch.save(
                {
                    "trajectories": trajectories.cpu(),
                    "meta": {
                        "schema_version": self._CACHE_SCHEMA_VERSION,
                        "system": str(getattr(self.env, "system_name", "unknown")),
                        "dt": float(getattr(self.env, "dt", 0.0)),
                        "cache_split": self.cache_split,
                        "cache_steps": int(self.cache_steps),
                        "cache_trajectories": int(self.cache_trajectories),
                        "cache_warmup": int(self.cache_warmup),
                        "standardize": bool(self.standardize),
                        "resample": bool(self.resample),
                        "pts_per_period": int(self.pts_per_period),
                        "ic_noise_scale": float(
                            getattr(self.env, "ic_noise_scale", 0.0)
                        ),
                        "integration_primary_method": str(
                            getattr(self, "cache_primary_method", "Radau")
                        ),
                        "integration_primary_timeout_seconds": float(
                            getattr(
                                self,
                                "cache_trajectory_timeout_seconds",
                                0.0,
                            )
                        ),
                        "integration_fallback_method": str(
                            getattr(self, "cache_timeout_fallback_method", "")
                        ),
                        "integration_fallback_timeout_seconds": float(
                            getattr(
                                self,
                                "cache_fallback_timeout_seconds",
                                0.0,
                            )
                        ),
                        "integration_method_counts": dict(
                            getattr(self, "_cache_integration_method_counts", {})
                        ),
                    },
                },
                tmp_path,
            )
            os.replace(tmp_path, cache_path)
            print(f"[dysts cache] saved cache to {cache_path}", flush=True)
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"Failed to save Dysts cache at {cache_path}") from e

    def _initial_conditions(self, base_ic: np.ndarray, base_std: np.ndarray) -> List[np.ndarray]:
        ic_list: List[np.ndarray] = []
        for _ in range(self.cache_trajectories):
            # Every split receives its own deterministically seeded perturbations.
            # Keeping trajectory zero at the exact default IC made train/val/test
            # share one identical rollout despite separate cache namespaces.
            noise = torch.randn(self.env._dim, generator=self._cache_build_rng).numpy()
            ic = base_ic + noise * base_std * float(self.env.ic_noise_scale)
            ic_list.append(np.array(ic, dtype=np.float32))
        return ic_list

    def _build_cache_parallel(
        self,
        ic_list: List[np.ndarray],
        t0: float,
    ) -> List[torch.Tensor]:
        total_steps = self.cache_steps + self.cache_warmup + 1
        trajectories: List[Optional[torch.Tensor]] = [None] * len(ic_list)
        method_counts: dict[str, int] = {}
        completed = 0
        print(
            f"[dysts cache] using parallel workers={self.cache_num_workers} "
            f"for {len(ic_list)} trajectories",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=self.cache_num_workers,
            mp_context=get_context("spawn"),
            initializer=_init_dysts_cache_worker,
        ) as executor:
            futures = {}
            for idx, ic in enumerate(ic_list):
                futures[executor.submit(
                    _build_dysts_native_trajectory,
                    str(getattr(self.env, "system_name", "")),
                    float(getattr(self.env, "dt", 0.0)),
                    ic,
                    total_steps,
                    bool(self.resample),
                    int(self.pts_per_period),
                    bool(self.standardize),
                    str(self.cache_primary_method),
                    float(self.cache_trajectory_timeout_seconds),
                    str(self.cache_timeout_fallback_method),
                    float(self.cache_fallback_timeout_seconds),
                    int(idx),
                )] = idx
            for future in as_completed(futures):
                idx = futures[future]
                completed += 1
                try:
                    traj, integration_method = future.result()
                    method_counts[integration_method] = (
                        method_counts.get(integration_method, 0) + 1
                    )
                    if self.cache_warmup > 0:
                        traj = traj[self.cache_warmup:]
                    if traj.dtype != torch.float32:
                        traj = traj.to(dtype=torch.float32)
                    trajectories[idx] = traj
                except Exception as e:
                    if idx == 0:
                        raise
                    print(
                        f"Warning: Failed to build dysts trajectory {idx + 1}/"
                        f"{len(ic_list)}: {e}.",
                        flush=True,
                    )
                if completed == 1 or completed % 10 == 0 or completed == len(ic_list):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[dysts cache] completed {completed}/{len(ic_list)} "
                        f"(elapsed={elapsed:.1f}s)",
                        flush=True,
                    )
        missing = [idx for idx, traj in enumerate(trajectories) if traj is None]
        if missing:
            raise RuntimeError(
                "Dysts cache construction failed for trajectory indices: "
                + ",".join(str(idx) for idx in missing)
            )
        self._cache_integration_method_counts = method_counts
        return [traj for traj in trajectories if traj is not None]

    def _build_cache(self) -> torch.Tensor:
        """Generate cached trajectories using dysts native integrator."""
        base_ic = self.env._ic.detach().cpu().numpy()
        base_std = self.env._std.detach().cpu().numpy()
        cache_path = self._cache_path(base_ic, base_std)
        cached = self._load_cached_trajectories(cache_path)
        if cached is not None:
            return cached

        ic_list = self._initial_conditions(base_ic, base_std)
        original_ic = None
        if hasattr(self.env.system, "ic"):
            original_ic = np.array(self.env.system.ic, copy=True)

        trajectories: List[torch.Tensor] = []
        t0 = time.perf_counter()

        can_parallel = (
            self.cache_num_workers > 1
            and self.cache_trajectories > 1
            and hasattr(self.env, "system_name")
        )
        if can_parallel:
            trajectories = self._build_cache_parallel(ic_list, t0)
        else:
            for i, ic in enumerate(ic_list):
                if i == 0 or (i + 1) % 10 == 0:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[dysts cache] building trajectory {i + 1}/{self.cache_trajectories} "
                        f"(elapsed={elapsed:.1f}s)",
                        flush=True,
                    )
                if hasattr(self.env.system, "ic"):
                    self.env.system.ic = np.array(ic, dtype=np.float32)
                try:
                    traj = self.env.make_trajectory_native(
                        n=self.cache_steps + self.cache_warmup + 1,
                        resample=self.resample,
                        pts_per_period=self.pts_per_period,
                        standardize=self.standardize,
                        method=self.cache_primary_method,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to build dysts trajectory {i + 1}/"
                        f"{self.cache_trajectories}"
                    ) from e
                if self.cache_warmup > 0:
                    traj = traj[self.cache_warmup:]
                if traj.dtype != torch.float32:
                    traj = traj.to(dtype=torch.float32)
                trajectories.append(traj)
            self._cache_integration_method_counts = {
                self.cache_primary_method: len(trajectories)
            }

        if original_ic is not None:
            self.env.system.ic = original_ic

        elapsed = time.perf_counter() - t0
        print(
            f"[dysts cache] done: trajectories={len(trajectories)}, "
            f"steps={trajectories[0].shape[0] if trajectories else 0} "
            f"(total={elapsed:.1f}s)",
            flush=True,
        )

        if len(trajectories) != self.cache_trajectories:
            raise RuntimeError(
                f"Expected {self.cache_trajectories} Dysts trajectories, "
                f"built {len(trajectories)}"
            )

        if len(trajectories) == 1:
            stacked = trajectories[0].unsqueeze(0)
        else:
            stacked = torch.stack(trajectories, dim=0)

        expected_shape = (
            self.cache_trajectories,
            self.cache_steps + 1,
            int(self.env.observation_size),
        )
        if tuple(stacked.shape) != expected_shape:
            raise RuntimeError(
                f"Dysts cache shape {tuple(stacked.shape)} != {expected_shape}"
            )
        if stacked.dtype != torch.float32 or not torch.isfinite(stacked).all():
            raise RuntimeError("Dysts cache must be finite float32")

        self._save_cached_trajectories(cache_path, stacked)
        return stacked

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


class KuramotoOscillators(Env):
    """Coupled Kuramoto oscillators with scalable intrinsic dimension.

    State:
        theta in R^N, one phase per oscillator.

    Dynamics:
        dtheta_i/dt = omega_i + (K / N) * sum_j sin(theta_j - theta_i)
    with either ring (nearest-neighbor) or all-to-all coupling.
    """

    _VALID_TOPOLOGIES = {"ring", "all_to_all"}
    _VALID_OMEGA_MODES = {"identical", "uniform_spread", "random"}

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        k_cfg = cfg.ENV.KURAMOTO
        self.num_oscillators = int(k_cfg.NUM_OSCILLATORS)
        self.dt = float(k_cfg.DT)
        self.k_coupling = float(k_cfg.K_COUPLING)
        self.topology = str(k_cfg.TOPOLOGY).lower()
        self.omega_mode = str(k_cfg.OMEGA_MODE).lower()
        self.omega_spread = float(k_cfg.OMEGA_SPREAD)

        if self.num_oscillators < 2:
            raise ValueError("KURAMOTO.NUM_OSCILLATORS must be >= 2.")
        if self.topology not in self._VALID_TOPOLOGIES:
            raise ValueError(
                f"Unknown KURAMOTO.TOPOLOGY '{self.topology}'. "
                f"Expected one of {sorted(self._VALID_TOPOLOGIES)}."
            )
        if self.omega_mode not in self._VALID_OMEGA_MODES:
            raise ValueError(
                f"Unknown KURAMOTO.OMEGA_MODE '{self.omega_mode}'. "
                f"Expected one of {sorted(self._VALID_OMEGA_MODES)}."
            )

        self.omega = self._init_natural_frequencies()

    @property
    def action_size(self) -> int:
        return 0

    def _init_natural_frequencies(self) -> torch.Tensor:
        if self.omega_mode == "identical":
            return torch.zeros(self.num_oscillators, dtype=torch.float32)

        if self.omega_mode == "uniform_spread":
            return torch.linspace(
                -self.omega_spread,
                self.omega_spread,
                self.num_oscillators,
                dtype=torch.float32,
            )

        seed = int(getattr(self.cfg, "SEED", 0)) + 13
        rng = torch.Generator().manual_seed(seed)
        return torch.empty(self.num_oscillators, dtype=torch.float32).uniform_(
            -self.omega_spread,
            self.omega_spread,
            generator=rng,
        )

    def _coupling_term(self, theta: torch.Tensor) -> torch.Tensor:
        if self.topology == "ring":
            right = torch.roll(theta, shifts=-1, dims=-1)
            left = torch.roll(theta, shifts=1, dims=-1)
            return torch.sin(right - theta) + torch.sin(left - theta)

        # all_to_all
        phase_diffs = theta.unsqueeze(-2) - theta.unsqueeze(-1)
        return torch.sin(phase_diffs).sum(dim=-1)

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        return torch.empty(self.num_oscillators, dtype=torch.float32).uniform_(
            -torch.pi,
            torch.pi,
            generator=rng,
        )

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        omega = self.omega.to(device=state.device, dtype=state.dtype)
        coupling_scale = self.k_coupling / float(self.num_oscillators)

        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            coupling = self._coupling_term(state)
            return omega + coupling_scale * coupling

        return integrate_rk4(state, None, self.dt, dynamics_fn)

    def basin_label(self, state: torch.Tensor):
        """Return the winding-number basin label (signed integer)."""
        phase_diffs = torch.roll(state, shifts=-1, dims=-1) - state
        wrapped_diffs = torch.atan2(torch.sin(phase_diffs), torch.cos(phase_diffs))
        winding = torch.round(wrapped_diffs.sum(dim=-1) / (2.0 * torch.pi)).to(torch.int64)
        if winding.ndim == 0:
            return int(winding.item())
        return winding


class ContinuousHopfield(Env):
    """Continuous Hopfield network with configurable stored patterns."""

    _VALID_PATTERN_MODES = {"random", "orthogonal"}

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        h_cfg = cfg.ENV.HOPFIELD
        self.num_neurons = int(h_cfg.NUM_NEURONS)
        self.num_patterns = int(h_cfg.NUM_PATTERNS)
        self.dt = float(h_cfg.DT)
        self.beta = float(h_cfg.BETA)
        self.tau = float(h_cfg.TAU)
        self.pattern_mode = str(h_cfg.PATTERN_MODE).lower()
        self.init_scale = float(h_cfg.INIT_SCALE)

        if self.num_neurons < 2:
            raise ValueError("HOPFIELD.NUM_NEURONS must be >= 2.")
        if self.num_patterns < 1:
            raise ValueError("HOPFIELD.NUM_PATTERNS must be >= 1.")
        if self.tau <= 0:
            raise ValueError("HOPFIELD.TAU must be positive.")
        if self.pattern_mode not in self._VALID_PATTERN_MODES:
            raise ValueError(
                f"Unknown HOPFIELD.PATTERN_MODE '{self.pattern_mode}'. "
                f"Expected one of {sorted(self._VALID_PATTERN_MODES)}."
            )
        if self.pattern_mode == "orthogonal" and self.num_patterns > self.num_neurons:
            raise ValueError("HOPFIELD orthogonal patterns require NUM_PATTERNS <= NUM_NEURONS.")

        self.patterns = self._init_patterns()
        self.weight_matrix = (self.patterns.T @ self.patterns) / float(self.num_patterns)
        self.weight_matrix.fill_diagonal_(0.0)
        self.bias = torch.zeros(self.num_neurons, dtype=torch.float32)
        self.num_basins = self.num_patterns

    @property
    def action_size(self) -> int:
        return 0

    def _init_patterns(self) -> torch.Tensor:
        seed = int(getattr(self.cfg, "SEED", 0)) + 29
        rng = torch.Generator().manual_seed(seed)

        if self.pattern_mode == "random":
            patterns = torch.randint(
                0,
                2,
                (self.num_patterns, self.num_neurons),
                generator=rng,
                dtype=torch.int64,
            )
            return patterns.to(dtype=torch.float32) * 2.0 - 1.0

        basis = torch.randn(self.num_neurons, self.num_neurons, generator=rng, dtype=torch.float32)
        q, _ = torch.linalg.qr(basis)
        # Orthogonal mode starts from orthonormal directions, then binarizes to
        # +/-1 so stored patterns stay in the same discrete space as random mode.
        patterns = torch.sign(q[:, :self.num_patterns].T)
        patterns[patterns == 0] = 1.0
        return patterns

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        return torch.randn(self.num_neurons, generator=rng, dtype=torch.float32) * self.init_scale

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        weights = self.weight_matrix.to(device=state.device, dtype=state.dtype)
        bias = self.bias.to(device=state.device, dtype=state.dtype)

        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            activation = torch.tanh(self.beta * state)
            recurrent_drive = torch.matmul(activation, weights.T) + bias
            return (-state + recurrent_drive) / self.tau

        return integrate_rk4(state, None, self.dt, dynamics_fn)

    def basin_label(self, state: torch.Tensor):
        """Return the index of the stored pattern with maximal overlap."""
        patterns = self.patterns.to(device=state.device, dtype=state.dtype)
        sign_state = torch.where(state >= 0, torch.ones_like(state), -torch.ones_like(state))
        overlaps = torch.matmul(sign_state, patterns.T) / float(self.num_neurons)
        label = overlaps.argmax(dim=-1).to(torch.int64)
        if label.ndim == 0:
            return int(label.item())
        return label


class CompetitiveLotkaVolterra(Env):
    """Competitive N-species Lotka-Volterra with configurable interaction matrix."""

    _VALID_INTERACTION_MODES = {"symmetric", "asymmetric", "block_diagonal"}
    _VALID_R_MODES = {"uniform", "heterogeneous"}

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        lv_cfg = cfg.ENV.COMPETITIVE_LV
        self.num_species = int(lv_cfg.NUM_SPECIES)
        self.dt = float(lv_cfg.DT)
        self.interaction_mode = str(lv_cfg.INTERACTION_MODE).lower()
        self.r_mode = str(lv_cfg.R_MODE).lower()
        self.interaction_scale = float(lv_cfg.INTERACTION_SCALE)
        self.system_seed = getattr(lv_cfg, "SYSTEM_SEED", None)
        self.positivity_clip = bool(lv_cfg.POSITIVITY_CLIP)
        self.survival_threshold = float(lv_cfg.SURVIVAL_THRESHOLD)
        self.init_min = float(lv_cfg.INIT_MIN)
        self.init_max = float(lv_cfg.INIT_MAX)

        if self.num_species < 2:
            raise ValueError("COMPETITIVE_LV.NUM_SPECIES must be >= 2.")
        if self.interaction_mode not in self._VALID_INTERACTION_MODES:
            raise ValueError(
                f"Unknown COMPETITIVE_LV.INTERACTION_MODE '{self.interaction_mode}'. "
                f"Expected one of {sorted(self._VALID_INTERACTION_MODES)}."
            )
        if self.r_mode not in self._VALID_R_MODES:
            raise ValueError(
                f"Unknown COMPETITIVE_LV.R_MODE '{self.r_mode}'. "
                f"Expected one of {sorted(self._VALID_R_MODES)}."
            )
        if self.init_min <= 0 or self.init_max <= self.init_min:
            raise ValueError("COMPETITIVE_LV requires 0 < INIT_MIN < INIT_MAX.")

        self.growth_rates = self._init_growth_rates()
        self.interaction_matrix = self._init_interaction_matrix()

    @property
    def action_size(self) -> int:
        return 0

    def _make_rng(self, offset: int = 0) -> torch.Generator:
        base_seed = self.system_seed
        if base_seed is None:
            base_seed = int(getattr(self.cfg, "SEED", 0))
        seed = int(base_seed) + 43 + offset
        return torch.Generator().manual_seed(seed)

    def _init_growth_rates(self) -> torch.Tensor:
        if self.r_mode == "uniform":
            return torch.ones(self.num_species, dtype=torch.float32)

        rng = self._make_rng(offset=1)
        noise = torch.empty(self.num_species, dtype=torch.float32).uniform_(-0.2, 0.2, generator=rng)
        return torch.clamp(1.0 + noise, min=0.05)

    def _init_interaction_matrix(self) -> torch.Tensor:
        rng = self._make_rng(offset=2)
        n = self.num_species

        if self.interaction_mode == "symmetric":
            off_diag = torch.empty((n, n), dtype=torch.float32).uniform_(
                0.0,
                self.interaction_scale,
                generator=rng,
            )
            off_diag = 0.5 * (off_diag + off_diag.T)
        elif self.interaction_mode == "asymmetric":
            off_diag = torch.empty((n, n), dtype=torch.float32).uniform_(
                0.0,
                self.interaction_scale,
                generator=rng,
            )
        else:
            block_id = torch.arange(n) % 2
            same_block = block_id.unsqueeze(0) == block_id.unsqueeze(1)
            noise = torch.rand((n, n), generator=rng, dtype=torch.float32)
            intra = 0.5 * self.interaction_scale
            inter = 1.5 * self.interaction_scale
            off_diag = torch.where(
                same_block,
                intra + 0.3 * self.interaction_scale * noise,
                inter + 0.3 * self.interaction_scale * noise,
            )
            off_diag = 0.5 * (off_diag + off_diag.T)

        off_diag.fill_diagonal_(1.0)
        return off_diag

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        return torch.empty(self.num_species, dtype=torch.float32).uniform_(
            self.init_min,
            self.init_max,
            generator=rng,
        )

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        growth = self.growth_rates.to(device=state.device, dtype=state.dtype)
        interaction = self.interaction_matrix.to(device=state.device, dtype=state.dtype)

        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            competition = torch.matmul(state, interaction.T)
            return state * (growth - competition)

        next_state = integrate_rk4(state, None, self.dt, dynamics_fn)
        if self.positivity_clip:
            next_state = torch.clamp(next_state, min=0.0)
        return next_state

    def basin_label(self, state: torch.Tensor):
        """Return species-survival support encoded as a bitmask integer."""
        alive = (state > self.survival_threshold).to(torch.int64)
        bits = (1 << torch.arange(self.num_species, device=state.device, dtype=torch.int64))
        label = (alive * bits).sum(dim=-1)
        if label.ndim == 0:
            return int(label.item())
        return label


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


def _as_batch_2d(state: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """View a single 2D state as a size-1 batch for vectorized logic."""
    if state.ndim == 1:
        return state.unsqueeze(0), True
    return state, False


class GatedLocalLinear(Env):
    """Three-basin local-linear system with shared gate sectors between basins."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.dt = (
            float(cfg.ENV.GATED_LOCAL_LINEAR.DT)
            if hasattr(cfg.ENV, "GATED_LOCAL_LINEAR")
            else 0.04
        )
        self.num_basins = 3
        self.center_radius = 1.75
        self.basin_radius = 1.05
        self.init_range = 2.6
        self.gate_contraction = 1.35
        self.gate_swirl = 0.9

        self.center_angles = torch.linspace(
            0.0, 2.0 * torch.pi, steps=self.num_basins + 1, dtype=torch.float32
        )[:-1]
        self.points_2d = self._build_centers(radius=self.center_radius)
        self.points = self.points_2d
        self.basin_matrices = self._build_basin_matrices()
        self.gate_matrices = self._build_gate_matrices()

    @staticmethod
    def _rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
        c = torch.cos(angle)
        s = torch.sin(angle)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])], dim=0)

    def _build_centers(self, radius: float) -> torch.Tensor:
        return torch.stack(
            [radius * torch.cos(self.center_angles), radius * torch.sin(self.center_angles)],
            dim=1,
        )

    def _build_basin_matrices(self) -> torch.Tensor:
        templates = torch.stack(
            [
                torch.tensor([[-0.9, -1.2], [1.2, -0.9]], dtype=torch.float32),
                torch.tensor([[-1.35, 0.2], [-0.3, -0.7]], dtype=torch.float32),
                torch.tensor([[-0.7, -0.1], [0.5, -1.2]], dtype=torch.float32),
            ],
            dim=0,
        )
        matrices = []
        for basin_index in range(self.num_basins):
            base = templates[basin_index % templates.shape[0]]
            rot = self._rotation_matrix(self.center_angles[basin_index])
            matrices.append(rot @ base @ rot.transpose(0, 1))
        return torch.stack(matrices, dim=0)

    def _build_gate_matrices(self) -> torch.Tensor:
        base = torch.tensor(
            [
                [-self.gate_contraction, -self.gate_swirl],
                [self.gate_swirl, -self.gate_contraction],
            ],
            dtype=torch.float32,
        )
        return torch.stack([base.clone() for _ in range(self.num_basins)], dim=0)

    def _sector_index(self, state_2d: torch.Tensor) -> torch.Tensor:
        angles = torch.atan2(state_2d[..., 1], state_2d[..., 0])
        deltas = (
            torch.remainder(angles.unsqueeze(-1) - self.center_angles + torch.pi, 2.0 * torch.pi)
            - torch.pi
        )
        return deltas.abs().argmin(dim=-1).to(dtype=torch.long)

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        return torch.empty(2, dtype=torch.float32).uniform_(
            -self.init_range, self.init_range, generator=rng
        )

    def region_label(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch_2d(state)
        diff = batch.unsqueeze(1) - self.points_2d.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        nearest_basin = dist.argmin(dim=-1).to(dtype=torch.long)
        in_basin = dist.min(dim=-1).values <= self.basin_radius
        gate_sector = self._sector_index(batch)
        labels = torch.where(in_basin, nearest_basin, self.num_basins + gate_sector)
        return labels[0] if squeezed else labels

    def basin_label(self, state: torch.Tensor) -> torch.Tensor:
        region = self.region_label(state)
        if isinstance(region, torch.Tensor) and region.ndim == 0:
            return region if int(region.item()) < self.num_basins else region - self.num_basins
        return torch.where(region < self.num_basins, region, region - self.num_basins)

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch_2d(state)
        basin = self.basin_label(batch)
        region = self.region_label(batch)
        centers = self.points_2d[basin]
        delta = batch - centers
        basin_mats = self.basin_matrices[basin]
        gate_mats = self.gate_matrices[basin]
        matrices = torch.where((region < self.num_basins)[..., None, None], basin_mats, gate_mats)
        velocity = torch.einsum("...ij,...j->...i", matrices, delta)
        return velocity[0] if squeezed else velocity

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return integrate_rk4(state, None, self.dt, lambda s, _a=None: self.dynamics(s))


class GatedTransferLinear(Env):
    """Three-basin transfer system with explicit source, exit, and channel regions."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.dt = (
            float(cfg.ENV.GATED_TRANSFER_LINEAR.DT)
            if hasattr(cfg.ENV, "GATED_TRANSFER_LINEAR")
            else 0.04
        )
        self.num_basins = 3
        self.center_radius = 1.85
        self.center_phase = 0.0
        self.core_radius = 0.30
        self.source_radius = 0.80
        self.handoff_radius = 0.45
        self.exit_min_radius = 0.60
        self.channel_half_width = 0.22
        self.channel_lane_offset = 0.28
        self.exit_half_angle = 0.72
        self.init_range = 2.8
        self.return_rate_scale = 0.65
        self.background_rate_scale = 0.50
        self.exit_forward_rate = 1.0
        self.exit_transverse_rate = 1.8
        self.exit_handoff_offset = 0.40
        self.channel_speed = 1.55
        self.channel_transverse_contraction = 2.8

        self.center_angles = (
            torch.linspace(0.0, 2.0 * torch.pi, steps=self.num_basins + 1, dtype=torch.float32)[:-1]
            + self.center_phase
        )
        self.points_2d = self._build_centers(radius=self.center_radius)
        self.points = self.points_2d
        self.outward_directions = self._normalize(self.points_2d)
        self.ordered_pairs = [
            (source, dest)
            for source in range(self.num_basins)
            for dest in range(self.num_basins)
            if dest != source
        ]
        self.pair_to_index = {pair: index for index, pair in enumerate(self.ordered_pairs)}
        self.source_pair_indices_by_basin = [
            torch.tensor(
                [self.pair_to_index[(source, dest)] for dest in range(self.num_basins) if dest != source],
                dtype=torch.long,
            )
            for source in range(self.num_basins)
        ]
        self.exit_directions_2d = torch.stack(
            [
                self._normalize((self.points_2d[dest] - self.points_2d[source]).unsqueeze(0))[0]
                for source, dest in self.ordered_pairs
            ],
            dim=0,
        )
        self.exit_normals_2d = self._rot90(self.exit_directions_2d)
        (
            self.channel_directions_2d,
            self.channel_normals_2d,
            self.channel_entry_points_2d,
            self.channel_handoff_points_2d,
            self.channel_handoff_targets_2d,
            self.channel_lengths,
            self.channel_destination_basins,
        ) = self._build_channels()
        self.basin_matrices = self._build_basin_matrices()
        self.return_matrices = self.return_rate_scale * self.basin_matrices
        self.background_matrices = self.background_rate_scale * self.basin_matrices
        self.core_offset = 0
        self.return_offset = self.num_basins
        self.exit_offset = self.return_offset + self.num_basins
        self.channel_offset = self.exit_offset + len(self.ordered_pairs)
        self.num_regions = self.channel_offset + len(self.ordered_pairs)

    @staticmethod
    def _normalize(vectors: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)
        return vectors / norms

    @staticmethod
    def _rot90(vec: torch.Tensor) -> torch.Tensor:
        rot = torch.zeros_like(vec)
        rot[..., 0] = -vec[..., 1]
        rot[..., 1] = vec[..., 0]
        return rot

    @staticmethod
    def _rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
        c = torch.cos(angle)
        s = torch.sin(angle)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])], dim=0)

    def _build_centers(self, radius: float) -> torch.Tensor:
        return torch.stack(
            [radius * torch.cos(self.center_angles), radius * torch.sin(self.center_angles)],
            dim=1,
        )

    def _build_basin_matrices(self) -> torch.Tensor:
        templates = torch.stack(
            [
                torch.tensor([[-1.0, -1.1], [1.1, -1.0]], dtype=torch.float32),
                torch.tensor([[-1.4, 0.2], [-0.2, -0.7]], dtype=torch.float32),
                torch.tensor([[-0.8, -0.3], [0.5, -1.3]], dtype=torch.float32),
            ],
            dim=0,
        )
        matrices = []
        for basin_index in range(self.num_basins):
            base = templates[basin_index % templates.shape[0]]
            rot = self._rotation_matrix(self.center_angles[basin_index])
            matrices.append(rot @ base @ rot.transpose(0, 1))
        return torch.stack(matrices, dim=0)

    def _build_channels(self):
        directions = []
        normals = []
        entry_points = []
        handoff_points = []
        handoff_targets = []
        lengths = []
        destination_basins = []
        for pair_index, (source, dest) in enumerate(self.ordered_pairs):
            source_to_dest = self.exit_directions_2d[pair_index]
            source_exit_normal = self.exit_normals_2d[pair_index]
            destination_outward = self.outward_directions[dest]
            destination_tangent = self._rot90(destination_outward)
            incoming_delta = float(torch.sin(self.center_angles[source] - self.center_angles[dest]))
            tangent_sign = 1.0 if incoming_delta >= 0.0 else -1.0

            entry = (
                self.points_2d[source]
                + self.source_radius * source_to_dest
                + (tangent_sign * self.channel_lane_offset) * source_exit_normal
            )
            handoff = (
                self.points_2d[dest]
                + self.handoff_radius * destination_outward
                + (tangent_sign * self.channel_lane_offset) * destination_tangent
            )
            direction = self._normalize((handoff - entry).unsqueeze(0))[0]
            normal = self._rot90(direction)
            handoff_target = entry + self.exit_handoff_offset * direction
            directions.append(direction)
            normals.append(normal)
            entry_points.append(entry)
            handoff_points.append(handoff)
            handoff_targets.append(handoff_target)
            lengths.append(torch.dot(direction, handoff - entry))
            destination_basins.append(dest)
        return (
            torch.stack(directions, dim=0),
            torch.stack(normals, dim=0),
            torch.stack(entry_points, dim=0),
            torch.stack(handoff_points, dim=0),
            torch.stack(handoff_targets, dim=0),
            torch.stack(lengths, dim=0),
            torch.tensor(destination_basins, dtype=torch.long),
        )

    def _nearest_basin(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        diff = batch.unsqueeze(1) - self.points_2d.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        nearest_dist, nearest_basin = dist.min(dim=1)
        return nearest_basin.to(dtype=torch.long), nearest_dist

    def _channel_membership(
        self,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel = batch.unsqueeze(1) - self.channel_entry_points_2d.unsqueeze(0)
        longitudinal = (rel * self.channel_directions_2d.unsqueeze(0)).sum(dim=-1)
        transverse = (rel * self.channel_normals_2d.unsqueeze(0)).sum(dim=-1)
        mask = (
            (longitudinal >= 0.0)
            & (longitudinal <= self.channel_lengths.unsqueeze(0))
            & (torch.abs(transverse) <= self.channel_half_width)
        )
        has_channel = mask.any(dim=1)
        masked_distance = torch.where(
            mask,
            torch.abs(transverse),
            torch.full_like(transverse, float("inf")),
        )
        closest_match = masked_distance.argmin(dim=1)
        channel_index = torch.where(
            has_channel,
            closest_match.to(dtype=torch.long),
            torch.full_like(closest_match, -1, dtype=torch.long),
        )
        return channel_index, longitudinal, transverse

    def _source_neighborhood_label_flat(
        self,
        batch: torch.Tensor,
        nearest_basin: torch.Tensor,
        nearest_dist: torch.Tensor,
        channel_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if channel_index is None:
            channel_index, _, _ = self._channel_membership(batch)
        return torch.where(
            (nearest_dist <= self.source_radius) & (channel_index < 0),
            nearest_basin,
            torch.full_like(nearest_basin, -1),
        )

    def _core_basin_label_flat(
        self,
        nearest_basin: torch.Tensor,
        nearest_dist: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            nearest_dist <= self.core_radius,
            nearest_basin,
            torch.full_like(nearest_basin, -1),
        )

    def _exit_pair_index_flat(
        self,
        batch: torch.Tensor,
        source_basin: torch.Tensor,
    ) -> torch.Tensor:
        exit_pair = torch.full((batch.shape[0],), -1, dtype=torch.long, device=batch.device)
        min_cosine = math.cos(self.exit_half_angle)
        for basin in range(self.num_basins):
            basin_mask = source_basin == basin
            if not bool(basin_mask.any()):
                continue
            local_states = batch[basin_mask]
            rel = local_states - self.points_2d[basin]
            rel_radius = torch.norm(rel, dim=-1)
            rel_unit = self._normalize(rel)
            pair_indices = self.source_pair_indices_by_basin[basin].to(device=batch.device)
            pair_dirs = self.exit_directions_2d[pair_indices]
            cosine = rel_unit @ pair_dirs.transpose(0, 1)
            best_cosine, best_local = cosine.max(dim=1)
            selected_pairs = pair_indices[best_local]
            exit_pair[basin_mask] = torch.where(
                (best_cosine >= min_cosine) & (rel_radius >= self.exit_min_radius),
                selected_pairs,
                torch.full_like(selected_pairs, -1),
            )
        return exit_pair

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        return torch.empty(2, dtype=torch.float32).uniform_(
            -self.init_range, self.init_range, generator=rng
        )

    def source_neighborhood_label(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch_2d(state)
        nearest_basin, nearest_dist = self._nearest_basin(batch)
        channel_index, _, _ = self._channel_membership(batch)
        labels = self._source_neighborhood_label_flat(batch, nearest_basin, nearest_dist, channel_index)
        return labels[0] if squeezed else labels

    def region_label(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch_2d(state)
        nearest_basin, nearest_dist = self._nearest_basin(batch)
        core_basin = self._core_basin_label_flat(nearest_basin, nearest_dist)
        channel_index, _, _ = self._channel_membership(batch)
        source_basin = self._source_neighborhood_label_flat(batch, nearest_basin, nearest_dist, channel_index)
        exit_pair = self._exit_pair_index_flat(batch, source_basin)
        labels = self.return_offset + nearest_basin
        channel_mask = channel_index >= 0
        labels = torch.where(channel_mask, self.channel_offset + channel_index, labels)
        exit_mask = (source_basin >= 0) & (core_basin < 0) & (channel_index < 0) & (exit_pair >= 0)
        labels = torch.where(exit_mask, self.exit_offset + exit_pair, labels)
        core_mask = core_basin >= 0
        labels = torch.where(core_mask, core_basin, labels)
        return labels[0] if squeezed else labels

    def basin_label(self, state: torch.Tensor) -> torch.Tensor:
        region = self.region_label(state)
        if not isinstance(region, torch.Tensor):
            region = torch.tensor(region)
        squeezed = region.ndim == 0
        region_batch = region.unsqueeze(0) if squeezed else region
        basin = torch.full_like(region_batch, -1)
        core_mask = region_batch < self.num_basins
        basin = torch.where(core_mask, region_batch, basin)
        return_mask = (region_batch >= self.return_offset) & (region_batch < self.exit_offset)
        basin = torch.where(return_mask, region_batch - self.return_offset, basin)
        exit_mask = (region_batch >= self.exit_offset) & (region_batch < self.channel_offset)
        if bool(exit_mask.any()):
            exit_pair = region_batch[exit_mask] - self.exit_offset
            basin[exit_mask] = self.channel_destination_basins[exit_pair]
        channel_mask = region_batch >= self.channel_offset
        if bool(channel_mask.any()):
            channel_pair = region_batch[channel_mask] - self.channel_offset
            basin[channel_mask] = self.channel_destination_basins[channel_pair]
        return basin[0] if squeezed else basin

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch_2d(state)
        nearest_basin, nearest_dist = self._nearest_basin(batch)
        core_basin = self._core_basin_label_flat(nearest_basin, nearest_dist)
        channel_index, _, _ = self._channel_membership(batch)
        source_basin = self._source_neighborhood_label_flat(batch, nearest_basin, nearest_dist, channel_index)
        exit_pair = self._exit_pair_index_flat(batch, source_basin)
        derivatives = torch.zeros_like(batch)

        channel_mask = channel_index >= 0
        if bool(channel_mask.any()):
            idx = channel_index[channel_mask]
            rel = batch[channel_mask] - self.channel_entry_points_2d[idx]
            transverse = (rel * self.channel_normals_2d[idx]).sum(dim=-1, keepdim=True)
            derivatives[channel_mask] = (
                self.channel_speed * self.channel_directions_2d[idx]
                - self.channel_transverse_contraction * transverse * self.channel_normals_2d[idx]
            )

        non_channel_mask = ~channel_mask
        if bool(non_channel_mask.any()):
            active_indices = non_channel_mask.nonzero(as_tuple=False).reshape(-1)
            active_states = batch[active_indices]
            active_nearest = nearest_basin[active_indices]
            active_source = source_basin[active_indices]
            active_core = core_basin[active_indices]
            active_exit = exit_pair[active_indices]

            matrices = self.background_matrices[active_nearest].clone()
            centers = self.points_2d[active_nearest].clone()
            in_source_mask = active_source >= 0
            if bool(in_source_mask.any()):
                source_basins = active_source[in_source_mask]
                matrices[in_source_mask] = self.return_matrices[source_basins]
                centers[in_source_mask] = self.points_2d[source_basins]
            core_mask = active_core >= 0
            if bool(core_mask.any()):
                core_basins = active_core[core_mask]
                matrices[core_mask] = self.basin_matrices[core_basins]
                centers[core_mask] = self.points_2d[core_basins]

            exit_mask = (active_source >= 0) & (active_core < 0) & (active_exit >= 0)
            if bool(exit_mask.any()):
                exit_indices = active_exit[exit_mask]
                source_centers = self.points_2d[active_source[exit_mask]]
                exit_directions = self.exit_directions_2d[exit_indices]
                exit_normals = self.exit_normals_2d[exit_indices]
                rel = active_states[exit_mask] - source_centers
                transverse = (rel * exit_normals).sum(dim=-1, keepdim=True)
                exit_derivatives = (
                    self.exit_forward_rate * exit_directions
                    - self.exit_transverse_rate * transverse * exit_normals
                )
                derivatives[active_indices[exit_mask]] = exit_derivatives
                non_exit_keep_mask = ~exit_mask
                if bool(non_exit_keep_mask.any()):
                    kept_indices = active_indices[non_exit_keep_mask]
                    derivatives[kept_indices] = torch.einsum(
                        "...ij,...j->...i",
                        matrices[non_exit_keep_mask],
                        active_states[non_exit_keep_mask] - centers[non_exit_keep_mask],
                    )
            else:
                derivatives[active_indices] = torch.einsum(
                    "...ij,...j->...i",
                    matrices,
                    active_states - centers,
                )

        return derivatives[0] if squeezed else derivatives

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return integrate_rk4(state, None, self.dt, lambda s, _a=None: self.dynamics(s))


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

        b_cfg = getattr(cfg.ENV, "BLENDED", None)
        # Timestep for integration
        self.dt = float(b_cfg.DT) if b_cfg is not None else 0.05

        # Blending width (controls how sharply basins transition)
        self.sigma = float(b_cfg.SIGMA) if b_cfg is not None else 1.5

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


class AnalyticMultibasinEnv(Env):
    """Expose a retained analytic multibasin system through the Env API."""

    def __init__(self, cfg: Config, system_name: str):
        super().__init__(cfg)
        from skae.dynamics.analytic import ensure_catalog_registered, get_system

        ensure_catalog_registered()
        dt_override = float(cfg.ENV.CLAUDE_CATALOG.DT)
        kwargs = {"dt": dt_override} if dt_override > 0.0 else {}
        self.system_name = system_name
        self.system = get_system(system_name, **kwargs)

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.system.reset(rng).to(dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if state.ndim > 1:
            batch = state.to(dtype=torch.float64)
            try:
                next_state = torch.vmap(self.system.step)(batch)
            except RuntimeError:
                next_state = torch.stack([self.system.step(s) for s in batch], dim=0)
            return next_state.to(dtype=state.dtype if state.is_floating_point() else torch.float32)
        next_state = self.system.step(state.to(dtype=torch.float64))
        return next_state.to(dtype=state.dtype if state.is_floating_point() else torch.float32)


# Public compatibility alias used by historical code and serialized naming.
ClaudeCatalogEnv = AnalyticMultibasinEnv


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
    "gated_local_linear": GatedLocalLinear,
    "gated_transfer_linear": GatedTransferLinear,
    "kuramoto": KuramotoOscillators,
    "hopfield": ContinuousHopfield,
    "competitive_lv": CompetitiveLotkaVolterra,
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
        "kuramoto", "hopfield", "competitive_lv", "analytic:SystemName",
        "multiwell_<variant>", "multiwell_<variant>_hd"

    For analytic paper systems:
        cfg.ENV.ENV_NAME should be "analytic:SystemName"
        (e.g., "analytic:cal_square_4"). Historical ``claude:`` keys remain
        accepted for frozen run compatibility.

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

    if env_name.startswith(("analytic:", "claude:")):
        system_name = env_name.split(":", 1)[1]
        return AnalyticMultibasinEnv(cfg, system_name)

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
        "For analytic systems, use 'analytic:SystemName' "
        "(e.g., 'analytic:cal_square_4'). "
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
        Dictionary with 'builtin', 'analytic', and 'dysts' keys. The historical
        'claude_catalog' key mirrors 'analytic' for compatibility.
        available systems.
    """
    result = {
        "builtin": sorted(_ENV_REGISTRY.keys()),
        "analytic": [],
        "claude_catalog": [],
        "dysts": [],
    }

    try:
        from skae.dynamics.analytic import ensure_catalog_registered, list_systems

        ensure_catalog_registered()
        analytic_systems = list_systems()
        result["analytic"] = analytic_systems
        result["claude_catalog"] = analytic_systems
    except ImportError:
        pass

    try:
        from skae.benchmarks.dysts_adapter import get_dysts_systems, is_dysts_available
        if is_dysts_available():
            result["dysts"] = get_dysts_systems()
    except ImportError:
        pass

    return result
