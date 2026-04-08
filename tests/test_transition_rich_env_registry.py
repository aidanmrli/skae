"""Tests for transition-rich native environment registration."""

from __future__ import annotations

import torch

from skae.config import Config
from skae.data import get_available_environments, make_env


def test_transition_rich_native_envs_construct_through_factory():
    for env_name in ("gated_local_linear", "gated_transfer_linear"):
        cfg = Config()
        cfg.ENV.ENV_NAME = env_name
        env = make_env(cfg)

        rng = torch.Generator().manual_seed(0)
        state = env.reset(rng)
        next_state = env.step(state)

        assert state.shape == (2,)
        assert next_state.shape == (2,)
        assert torch.all(torch.isfinite(next_state))


def test_transition_rich_native_envs_appear_in_environment_listing():
    envs = get_available_environments()
    assert "gated_local_linear" in envs["builtin"]
    assert "gated_transfer_linear" in envs["builtin"]
