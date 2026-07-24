from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from experiments.neurips_2026.allen_cahn_periodic_reencoding.p200_one_refresh import (
    summarize_optional_p200_one_refresh,
    summarize_p200_one_refresh,
    validate_p200_h400_rows,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import DIRECT


GRID = (DIRECT, 1, 2, 5, 10, 20, 25, 50, 100)
MODELS = tuple(range(64, 74))
TEST_DATASETS = (201, 202, 203)


def _card() -> dict:
    return {
        "roster": {"arms": ["dense", "sparse"], "model_seeds": list(MODELS)},
        "cadence_selection": {"cadence_grid": list(GRID)},
        "prospective_datasets": {
            "validation": [
                {"index": index, "seed": seed}
                for index, seed in enumerate((101, 102, 103))
            ],
            "test": [
                {"index": index, "seed": seed}
                for index, seed in enumerate(TEST_DATASETS)
            ],
        },
        "system": {
            "stored_dt": 0.1,
            "validation_horizon_steps": 200,
            "test_horizon_steps": 400,
        },
    }


def _instantaneous(
    arm: str,
    model_seed: int,
    dataset_seed: int,
    *,
    cadence: str | int,
) -> np.ndarray:
    offset = 0.01 * (model_seed - 64) + 0.02 * (dataset_seed - 201)
    early = (2.0 if arm == "dense" else 1.8) + offset
    if cadence == DIRECT:
        tail = (8.0 if arm == "dense" else 6.5) + offset
    elif cadence == 200:
        tail = (4.0 if arm == "dense" else 3.0) + offset
    else:
        raise AssertionError("fixture cadence is unsupported")
    return np.asarray([early] * 200 + [tail] * 200, dtype=np.float64)


def _row(
    arm: str,
    model_seed: int,
    dataset_seed: int,
    *,
    cadence: str | int,
    horizon: int,
) -> dict:
    full = _instantaneous(
        arm,
        model_seed,
        dataset_seed,
        cadence=cadence,
    )
    instantaneous = full[:horizon]
    cumulative = np.cumsum(instantaneous) / np.arange(1, horizon + 1)
    return {
        "arm": arm,
        "model_seed": model_seed,
        "dataset_seed": dataset_seed,
        "cadence": cadence,
        "horizon_steps": horizon,
        "instantaneous_field_mse": instantaneous.tolist(),
        "cumulative_field_mse": cumulative.tolist(),
    }


def _rows(*, cadence: str | int, horizon: int) -> list[dict]:
    return [
        _row(arm, model_seed, dataset_seed, cadence=cadence, horizon=horizon)
        for arm in ("dense", "sparse")
        for model_seed in MODELS
        for dataset_seed in TEST_DATASETS
    ]


def test_p200_summary_reports_one_refresh_and_all_descriptive_endpoints() -> None:
    p200 = _rows(cadence=200, horizon=400)
    direct_h400 = _rows(cadence=DIRECT, horizon=400)
    direct_h200 = _rows(cadence=DIRECT, horizon=200)
    result = summarize_p200_one_refresh(
        p200,
        direct_h400,
        direct_h200,
        _card(),
        bootstrap_replicates=2_000,
        bootstrap_seed=19,
    )
    assert result["status"] == "complete"
    assert result["refresh_boundary_step"] == 200
    assert result["refresh_physical_time"] == pytest.approx(20.0)
    assert result["refresh_count"] == 1
    assert result["encoder_calls"] == 2
    assert result["validation_selection_eligible"] is False
    assert result["can_rescue_h200_primary"] is False
    assert result["can_rescue_full_grid_h400"] is False
    assert result["prefix_integrity"][
        "p200_matches_independent_h200_direct"
    ] is True
    assert result["prefix_integrity"][
        "h400_direct_matches_independent_h200_direct"
    ] is True

    endpoint_names = {
        "h400_cumulative_field_mse",
        "h201_h400_tail_field_mse",
        "h400_terminal_field_mse",
    }
    fixed = result["fixed_p200_sparse_vs_dense"]["endpoints"]
    assert set(fixed) == endpoint_names
    for endpoint in fixed.values():
        assert endpoint["relative_reduction_of_arm_means"] > 0.0
        assert endpoint["sparse_seed_wins"] == 10
        assert endpoint["paired_ratio_bootstrap"]["ci95_lower"] > 0.0
        assert endpoint["inference_role"].startswith("optional_descriptive")

    within = result["within_arm_p200_vs_direct"]
    assert set(within) == {"dense", "sparse"}
    for arm in ("dense", "sparse"):
        assert set(within[arm]["endpoints"]) == endpoint_names
        for endpoint in within[arm]["endpoints"].values():
            assert endpoint["relative_reduction_of_arm_means"] > 0.0
            assert endpoint["selected_seed_wins"] == 10
            assert endpoint["paired_ratio_bootstrap"]["ci95_lower"] > 0.0


def test_p200_validator_requires_exact_finite_cross_and_matching_prefix() -> None:
    p200 = _rows(cadence=200, horizon=400)
    direct_h200 = _rows(cadence=DIRECT, horizon=200)
    validate_p200_h400_rows(p200, direct_h200, _card())

    with pytest.raises(ValueError, match="exact frozen cross"):
        validate_p200_h400_rows(p200[:-1], direct_h200, _card())

    nonfinite = deepcopy(p200)
    nonfinite[0]["instantaneous_field_mse"][300] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        validate_p200_h400_rows(nonfinite, direct_h200, _card())

    mismatched = deepcopy(p200)
    mismatched[0]["instantaneous_field_mse"][199] += 1e-6
    with pytest.raises(AssertionError, match="H200 prefix mismatch"):
        validate_p200_h400_rows(mismatched, direct_h200, _card())

    cumulative_mismatch = deepcopy(p200)
    cumulative_mismatch[0]["cumulative_field_mse"][50] += 1e-6
    with pytest.raises(AssertionError, match="cumulative.*prefix mismatch"):
        validate_p200_h400_rows(cumulative_mismatch, direct_h200, _card())

    direct_h400_mismatch = _rows(cadence=DIRECT, horizon=400)
    direct_h400_mismatch[0]["instantaneous_field_mse"][0] += 1e-6
    with pytest.raises(AssertionError, match="H400 direct.*prefix mismatch"):
        summarize_p200_one_refresh(
            p200,
            direct_h400_mismatch,
            direct_h200,
            _card(),
            bootstrap_replicates=10,
        )


def test_p200_is_forbidden_from_the_validation_grid() -> None:
    card = _card()
    card["cadence_selection"]["cadence_grid"].append(200)
    with pytest.raises(ValueError, match="outside the validation cadence grid"):
        validate_p200_h400_rows(
            _rows(cadence=200, horizon=400),
            _rows(cadence=DIRECT, horizon=200),
            card,
        )


def test_optional_p200_absence_or_failure_suppresses_only_this_diagnostic() -> None:
    absent = summarize_optional_p200_one_refresh(
        None,
        _rows(cadence=DIRECT, horizon=400),
        _rows(cadence=DIRECT, horizon=200),
        _card(),
        bootstrap_replicates=10,
    )
    assert absent["status"] == "suppressed"
    assert absent["invalidates_h200_primary"] is False
    assert absent["invalidates_fixed_or_full_grid_h400"] is False

    failed = _rows(cadence=200, horizon=400)
    failed[0]["cumulative_field_mse"][399] = np.inf
    suppressed = summarize_optional_p200_one_refresh(
        failed,
        _rows(cadence=DIRECT, horizon=400),
        _rows(cadence=DIRECT, horizon=200),
        _card(),
        bootstrap_replicates=10,
    )
    assert suppressed["status"] == "suppressed"
    assert suppressed["reason"] == "optional_rows_failed_strict_validation"
    assert suppressed["error_type"] == "FloatingPointError"
    assert suppressed["invalidates_h200_primary"] is False
    assert suppressed["invalidates_fixed_or_full_grid_h400"] is False
