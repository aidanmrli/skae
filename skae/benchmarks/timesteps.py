"""Timestep lookup helpers shared by benchmark evaluation tools."""

from __future__ import annotations

from skae.config import Config, get_env_dt


def resolve_system_default_dt(system_key: str) -> float:
    """Resolve a configured or Dysts-native timestep from an environment key."""
    if system_key.lower().startswith("dysts:"):
        from skae.benchmarks.dysts_adapter import get_dysts_system_metadata

        dysts_name = system_key.split(":", 1)[1]
        metadata = get_dysts_system_metadata(dysts_name)
        dt = metadata.get("dt")
        if dt is None:
            raise ValueError(f"Dysts metadata for '{dysts_name}' does not define dt")
        return float(dt)

    cfg = Config()
    cfg.ENV.ENV_NAME = system_key
    return float(get_env_dt(cfg, system_key))
