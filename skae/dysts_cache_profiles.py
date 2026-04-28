"""Shared dysts cache profiles and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


DYSTS_CACHE_PROFILES: Dict[str, Dict[str, int]] = {
    "smoke": {
        "steps": 5000,
        "trajectories": 32,
        "warmup": 200,
    },
    "full": {
        "steps": 30000,
        "trajectories": 200,
        "warmup": 2000,
    },
    "long60": {
        "steps": 60000,
        "trajectories": 200,
        "warmup": 2000,
    },
}

_DEFAULT_SHARED_CACHE_DIR = Path("/network/scratch/l/lia/skae/dysts_native_cache")


def default_dysts_cache_dir() -> str:
    """Pick a sensible cache root for the current machine."""
    if _DEFAULT_SHARED_CACHE_DIR.parent.exists():
        return str(_DEFAULT_SHARED_CACHE_DIR)
    return "runs/dysts_native_cache"


def apply_dysts_cache_profile(cfg, profile_name: str) -> Dict[str, int]:
    """Apply a named dysts cache profile to a config object."""
    if profile_name not in DYSTS_CACHE_PROFILES:
        raise ValueError(
            f"Unknown dysts cache profile '{profile_name}'. "
            f"Available: {sorted(DYSTS_CACHE_PROFILES.keys())}"
        )
    profile = DYSTS_CACHE_PROFILES[profile_name]
    cfg.ENV.DYSTS.CACHE_STEPS = int(profile["steps"])
    cfg.ENV.DYSTS.CACHE_TRAJECTORIES = int(profile["trajectories"])
    cfg.ENV.DYSTS.CACHE_WARMUP = int(profile["warmup"])
    return profile
