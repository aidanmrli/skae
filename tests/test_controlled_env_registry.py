"""Tests for controlled-paper environment registration."""

from __future__ import annotations

import torch

from experiments.neurips_2026.controlled import (
    controlled_systems,
    resolve_controlled_default_dt,
)
from experiments.neurips_2026.protocol import CONTROLLED_PAPER_PROTOCOL
from skae.config import Config
from skae.data import get_available_environments, make_env


def test_controlled_native_envs_construct_through_factory():
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


def test_controlled_native_envs_appear_in_environment_listing():
    envs = get_available_environments()
    assert "gated_local_linear" in envs["builtin"]
    assert "gated_transfer_linear" in envs["builtin"]


def test_catalog_registers_only_the_thirteen_retained_paper_systems():
    envs = get_available_environments()
    expected = {
        key.removeprefix("claude:")
        for key in CONTROLLED_PAPER_PROTOCOL.system_keys
        if key.startswith("claude:")
    }

    assert set(envs["analytic"]) == expected
    assert set(envs["claude_catalog"]) == expected


def test_controlled_roster_envs_construct_with_benchmark_dt():
    for system in controlled_systems():
        cfg = Config()
        cfg.ENV.ENV_NAME = system.env_name
        cfg.ENV.CLAUDE_CATALOG.DT = resolve_controlled_default_dt(system.system_key)

        env = make_env(cfg)
        rng = torch.Generator().manual_seed(0)
        state = env.reset(rng)
        next_state = env.step(state)

        assert state.shape == next_state.shape
        assert torch.all(torch.isfinite(next_state))
