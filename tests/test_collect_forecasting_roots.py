"""Tests for forecasting-root collection helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tools.collect_forecasting_roots import _collect_rows, _read_env_dt_from_cfg


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


def _write_run(run_dir: Path, env_name: str, dt: float, metric: float) -> None:
    run_dir.mkdir(parents=True)
    config = {
        "ENV": {
            "ENV_NAME": env_name,
            env_name.upper(): {"DT": dt},
        },
        "MODEL": {"MODEL_NAME": "GenericKM", "TARGET_SIZE": 16},
        "TRAIN": {"SEQUENCE_LENGTH": 8, "NUM_STEPS": 100},
    }
    eval_data = {
        env_name: {
            "modes": {
                "no_reencode": {"horizons": {"100": {"mean": metric + 2.0}}},
                "every_step": {"horizons": {"100": {"mean": metric + 1.0}}},
            },
            "best_periodic": {"100": {"mean": metric, "mode": "periodic_10"}},
            "best_reset": {"100": {"mean": metric, "mode": "reset_10"}},
        }
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "evaluation_results_best.json").write_text(json.dumps(eval_data))


def test_collect_latest_keeps_distinct_systems_under_dimension_wrapper(tmp_path):
    root = tmp_path / "root"
    _write_run(
        root / "n_16" / "hopfield_n16_p16" / "dt_0p00625" / "seed_0" / "20260101-000000",
        env_name="hopfield",
        dt=0.00625,
        metric=1.0,
    )
    _write_run(
        root / "n_16" / "kuramoto_n16_identical" / "dt_0p00625" / "seed_0" / "20260102-000000",
        env_name="kuramoto",
        dt=0.00625,
        metric=2.0,
    )

    rows = _collect_rows(
        root_specs=[("root", root)],
        horizons=[100],
        eval_file_name="evaluation_results_best.json",
        select="latest",
    )

    assert {(row["system_name"], row["system_key"]) for row in rows} == {
        ("hopfield_n16_p16", "hopfield"),
        ("kuramoto_n16_identical", "kuramoto"),
    }
