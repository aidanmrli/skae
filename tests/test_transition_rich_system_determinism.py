"""Determinism checks for transition-rich native systems."""

from __future__ import annotations

import torch

from skae.config import Config
from skae.data import make_env


def _build_env(env_name: str):
    cfg = Config()
    cfg.ENV.ENV_NAME = env_name
    return make_env(cfg)


def test_transition_rich_reset_is_seed_deterministic():
    for env_name in ("multiwell_strong_transition", "gated_local_linear", "gated_transfer_linear"):
        env = _build_env(env_name)
        rng1 = torch.Generator().manual_seed(123)
        rng2 = torch.Generator().manual_seed(123)
        assert torch.allclose(env.reset(rng1), env.reset(rng2))


def test_transition_rich_step_is_deterministic_for_fixed_state():
    for env_name in ("multiwell_strong_transition", "gated_local_linear", "gated_transfer_linear"):
        env = _build_env(env_name)
        state = torch.tensor([0.25, -0.5], dtype=torch.float32)
        next_a = env.step(state)
        next_b = env.step(state)
        assert torch.allclose(next_a, next_b)
