"""Portable runtime storage resolution shared by Python entry points."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_scratch_root(*, fallback: Path) -> Path:
    """Match the documented cluster scratch precedence without user literals."""

    configured = os.environ.get("SKAE_SCRATCH_ROOT")
    if configured:
        return Path(configured).expanduser()

    cluster_scratch = os.environ.get("SCRATCH")
    if cluster_scratch:
        return Path(cluster_scratch).expanduser() / "skae"

    user = os.environ.get("USER", "").strip()
    if user:
        user_root = Path("/network/scratch") / user[:1] / user
        if user_root.is_dir():
            return user_root / "skae"

    return fallback.expanduser()


__all__ = ["resolve_scratch_root"]
