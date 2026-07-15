"""Tests for forecasting-root collection helpers."""

from __future__ import annotations

from experiments.neurips_2026.workflows.controlled_collection import (
    _read_env_dt_from_cfg,
)


def test_read_env_dt_from_cfg_handles_transition_rich_native_envs():
    cfg_data = {
        "ENV": {
            "ENV_NAME": "gated_local_linear",
            "GATED_LOCAL_LINEAR": {"DT": 0.04},
            "GATED_TRANSFER_LINEAR": {"DT": 0.02},
        }
    }

    assert _read_env_dt_from_cfg(cfg_data) == 0.04


def test_read_env_dt_from_cfg_handles_claude_catalog_envs():
    cfg_data = {
        "ENV": {
            "ENV_NAME": "claude:cal_asymmetric_3",
            "CLAUDE_CATALOG": {"DT": 0.03},
        }
    }

    assert _read_env_dt_from_cfg(cfg_data) == 0.03


def test_read_env_dt_from_cfg_handles_analytic_envs():
    cfg_data = {
        "ENV": {
            "ENV_NAME": "analytic:cal_asymmetric_3",
            "CLAUDE_CATALOG": {"DT": 0.03},
        }
    }

    assert _read_env_dt_from_cfg(cfg_data) == 0.03
