"""Shared dysts cache profiles and helpers."""

from __future__ import annotations

import os
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

def default_dysts_cache_dir() -> str:
    """Pick a cache root without embedding a contributor-specific path."""

    configured = os.environ.get("DYSTS_CACHE_DIR")
    if configured:
        return str(Path(configured).expanduser())
    scratch = os.environ.get("SKAE_SCRATCH_ROOT")
    if scratch:
        return str(Path(scratch).expanduser() / "dysts_native_cache")
    user = os.environ.get("USER", "user")
    user_scratch = Path("/network/scratch") / user[:1] / user
    if user_scratch.exists():
        return str(user_scratch / "skae" / "dysts_native_cache")
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
