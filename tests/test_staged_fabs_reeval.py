"""Safety contracts for staged/global wide-periodic reevaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skae.config import apply_env_dt_override, get_config
from experiments.neurips_2026.local_operators.reevaluate import (
    _best_periodic_metrics,
    _evaluation_settings,
    _raise_for_error_rows,
    _validate_pair_config_identity,
    _write_csv,
    _write_summary,
)
from experiments.neurips_2026.local_operators.reevaluation_io import (
    _cache_fingerprint,
    _discover_runs,
)


ROOT = Path(__file__).resolve().parents[1]


def _completed_run(
    root: Path,
    *,
    system: str = "duffing",
    seed: int = 0,
    stamp: str = "20260101-000000",
) -> Path:
    run = root / system / "dt_0p01" / f"seed_{seed}" / stamp
    run.mkdir(parents=True)
    (run / "checkpoint.pt").write_bytes(b"checkpoint")
    (run / "config.json").write_text(
        json.dumps({"SEED": seed, "ENV": {"ENV_NAME": system}})
    )
    (run / "evaluation_results_best.json").write_text("{}")
    return run


def test_discovery_uses_only_completed_unambiguous_config_identical_runs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    completed = _completed_run(root)
    incomplete = _completed_run(root, seed=1)
    (incomplete / "evaluation_results_best.json").unlink()
    assert _discover_runs(root) == {("duffing", 0): completed}

    _completed_run(root, stamp="20260102-000000")
    with pytest.raises(RuntimeError, match="Ambiguous completed runs"):
        _discover_runs(root)


def test_discovery_rejects_path_config_identity_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run = _completed_run(root)
    (run / "config.json").write_text(
        json.dumps({"SEED": 9, "ENV": {"ENV_NAME": "duffing"}})
    )
    with pytest.raises(ValueError, match="path/config identity mismatch"):
        _discover_runs(root)


def test_cache_fingerprint_covers_checkpoint_artifact_and_eval_contract(
    tmp_path: Path,
) -> None:
    run = _completed_run(tmp_path / "runs")

    def fingerprint(**overrides):
        values = {
            "kind": "staged",
            "system": "duffing",
            "seed": 0,
            "run_dir": run,
            "horizons": (100, 500, 1000),
            "periods": (1, 2, 5, 10, 20, 25, 50, 100),
            "batch_size": 100,
            "support_definition": "absolute:0.001",
            "family_jaccard_threshold": 0.4,
        }
        values.update(overrides)
        return _cache_fingerprint(**values)

    baseline = fingerprint()
    assert baseline["stage2_artifacts_sha256"] is None
    (run / "stage2_artifacts.pt").write_bytes(b"route-a")
    with_artifact = fingerprint()
    assert with_artifact != baseline
    (run / "stage2_artifacts.pt").write_bytes(b"route-b")
    assert fingerprint() != with_artifact
    (run / "checkpoint.pt").write_bytes(b"different-checkpoint")
    assert fingerprint() != with_artifact
    assert fingerprint(horizons=(100,)) != fingerprint()
    assert fingerprint(periods=(1, 10)) != fingerprint()
    assert fingerprint(batch_size=32) != fingerprint()
    assert fingerprint(family_jaccard_threshold=0.5) != fingerprint()


def test_pair_identity_requires_equal_system_seed_and_observation_dt() -> None:
    staged = get_config("generic_sparse")
    baseline = get_config("generic_sparse")
    staged.ENV.ENV_NAME = baseline.ENV.ENV_NAME = "duffing"
    staged.SEED = baseline.SEED = 3
    apply_env_dt_override(staged, 0.02)
    apply_env_dt_override(baseline, 0.02)
    assert _validate_pair_config_identity(
        staged, baseline, system="duffing", seed=3
    ) == pytest.approx(0.02)
    apply_env_dt_override(baseline, 0.03)
    with pytest.raises(ValueError, match="observation dt mismatch"):
        _validate_pair_config_identity(
            staged, baseline, system="duffing", seed=3
        )


def test_every_step_is_reused_as_period_one_without_duplicate_rollout() -> None:
    cfg = get_config("generic_sparse")
    settings = _evaluation_settings(
        cfg=cfg,
        horizons=(100,),
        periods=(1, 2, 5),
        batch_size=100,
    )
    assert settings.periodic_reencode_periods == (1, 2, 5)
    result = {
        cfg.ENV.ENV_NAME: {
            "best_reset": {
                "100": {
                    "mode": "every_step",
                    "mean": 0.25,
                    "per_dim_mean": 0.125,
                }
            }
        }
    }
    metrics = _best_periodic_metrics(result, cfg.ENV.ENV_NAME, (100,))
    assert metrics["h100_best_periodic_mode"] == "periodic_1"
    assert metrics["h100_best_periodic_mean"] == pytest.approx(0.25)


def test_mixed_error_rows_are_written_then_fail(tmp_path: Path) -> None:
    rows = [
        {"system_key": "duffing", "seed": 0, "status": "ok", "wins_all_horizons": True},
        {"system_key": "duffing", "seed": 1, "status": "error", "error": "boom"},
    ]
    csv_path = tmp_path / "rows.csv"
    summary_path = tmp_path / "summary.md"
    _write_csv(csv_path, rows)
    _write_summary(summary_path, rows, (100,), (1, 2))
    assert "error" in csv_path.read_text()
    assert "N/A" in summary_path.read_text()
    with pytest.raises(SystemExit, match="1 error row"):
        _raise_for_error_rows(rows)


def test_reevaluation_launcher_has_gpu_guard_and_fixed_grid() -> None:
    text = (
        ROOT / "scripts/neurips_2026/local_operators/reevaluate.sh"
    ).read_text()
    assert "source scripts/common/gpu_guard.sh" in text
    assert "gpu_guard_assert_cuda_visible" in text
    assert "gpu_guard_start_sampler" in text
    assert "1,2,5,10,20,25,50,100" in text
    loader_text = (
        ROOT / "experiments/neurips_2026/local_operators/reevaluate.py"
    ).read_text()
    assert "learn_target_centers=True" in loader_text
