from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from experiments.neurips_2026.allen_cahn_periodic_reencoding.h400_full_grid import (
    summarize_full_grid_h400_pipeline,
    validate_full_grid_h400_rows,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import DIRECT


GRID = (DIRECT, 1, 2, 5, 10, 20, 25, 50, 100)
MODELS = tuple(range(64, 74))
VALIDATION_DATASETS = (101, 102, 103)
TEST_DATASETS = (201, 202, 203)


def _card() -> dict:
    return {
        "roster": {"arms": ["dense", "sparse"], "model_seeds": list(MODELS)},
        "cadence_selection": {"cadence_grid": list(GRID)},
        "prospective_datasets": {
            "validation": [
                {"index": index, "seed": seed}
                for index, seed in enumerate(VALIDATION_DATASETS)
            ],
            "test": [
                {"index": index, "seed": seed}
                for index, seed in enumerate(TEST_DATASETS)
            ],
        },
    }


def _validation_risk(arm: str, cadence: str | int) -> float:
    if arm == "dense":
        return {DIRECT: 1.0, 1: 2.0}.get(cadence, 5.0)
    return {DIRECT: 2.0, 1: 1.0}.get(cadence, 5.0)


def _test_risk(cadence: str | int) -> tuple[float, float]:
    # Arm-independent risks make the exact swap calculation transparent.
    early = {DIRECT: 8.0, 1: 4.0}.get(cadence, 20.0)
    tail = {DIRECT: 2.0, 1: 1.0}.get(cadence, 5.0)
    return early, tail


def _validation_rows() -> list[dict]:
    rows = []
    for arm in ("dense", "sparse"):
        for model in MODELS:
            for dataset in VALIDATION_DATASETS:
                for cadence in GRID:
                    value = _validation_risk(arm, cadence)
                    rows.append(
                        {
                            "arm": arm,
                            "model_seed": model,
                            "dataset_seed": dataset,
                            "cadence": cadence,
                            "horizon_steps": 200,
                            "instantaneous_field_mse": [value] * 200,
                            "cumulative_field_mse": [value] * 200,
                        }
                    )
    return rows


def _h400_rows() -> list[dict]:
    rows = []
    for arm in ("dense", "sparse"):
        for model in MODELS:
            for dataset in TEST_DATASETS:
                for cadence in GRID:
                    early, tail = _test_risk(cadence)
                    instantaneous = np.asarray([early] * 200 + [tail] * 200)
                    cumulative = np.cumsum(instantaneous) / np.arange(1, 401)
                    rows.append(
                        {
                            "arm": arm,
                            "model_seed": model,
                            "dataset_seed": dataset,
                            "cadence": cadence,
                            "horizon_steps": 400,
                            "instantaneous_field_mse": instantaneous.tolist(),
                            "cumulative_field_mse": cumulative.tolist(),
                        }
                    )
    return rows


def _frequency(payload: dict, arm: str, cadence: str | int) -> int:
    return next(
        row["count"]
        for row in payload["selection_frequencies"][arm]
        if row["cadence"] == cadence
    )


def test_h400_pipeline_uses_h200_selector_for_both_endpoints() -> None:
    result = summarize_full_grid_h400_pipeline(
        _validation_rows(),
        _h400_rows(),
        _card(),
        bootstrap_replicates=2_000,
        bootstrap_seed=17,
        bootstrap_chunk_size=257,
        permutation_chunk_size=73,
    )
    assert result["selected_cadences_from_h200_validation"] == {
        "dense": DIRECT,
        "sparse": 1,
    }
    assert result["h400_outcomes_used_for_cadence_selection"] is False
    expected = {
        "h400_cumulative_field_mse": (5.0, 2.5),
        "h201_h400_tail_field_mse": (2.0, 1.0),
    }
    for name, (dense_mean, sparse_mean) in expected.items():
        endpoint = result["endpoints"][name]
        assert endpoint["point_test"]["dense_mean"] == pytest.approx(dense_mean)
        assert endpoint["point_test"]["sparse_mean"] == pytest.approx(sparse_mean)
        assert endpoint["point_test"][
            "relative_reduction_of_arm_means"
        ] == pytest.approx(0.5)
        frontier = endpoint["descriptive_accuracy_compute_frontier"]
        assert [row["cadence"] for row in frontier] == list(GRID)
        expected_refreshes = [0] + [399 // int(period) for period in GRID[1:]]
        assert [row["refresh_count"] for row in frontier] == expected_refreshes
        assert [row["encoder_calls"] for row in frontier] == [
            1 + value for value in expected_refreshes
        ]
        assert all(
            row["inference_role"] == "descriptive_only_not_used_for_selection"
            for row in frontier
        )
        direct = frontier[0]
        assert direct["dense_arm_mean_mse"] == pytest.approx(dense_mean)
        assert direct["sparse_arm_mean_mse"] == pytest.approx(dense_mean)
        assert direct["refresh_count"] == 0
        assert direct["encoder_calls"] == 1
        assert direct["fixed_same_cadence_relative_reduction_of_arm_means"] == 0.0
        period_one = frontier[1]
        assert period_one["dense_arm_mean_mse"] == pytest.approx(sparse_mean)
        assert period_one["sparse_arm_mean_mse"] == pytest.approx(sparse_mean)
        assert period_one["refresh_count"] == 399
        assert period_one["encoder_calls"] == 400
        period_two = frontier[2]
        assert period_two["refresh_count"] == 199
        assert period_two["encoder_calls"] == 200
        period_five = frontier[3]
        assert period_five["refresh_count"] == 79
        assert period_five["encoder_calls"] == 80
        period_100 = frontier[-1]
        assert period_100["refresh_count"] == 3
        assert period_100["encoder_calls"] == 4
        exact = endpoint["selection_aware_pipeline_inference"][
            "exact_one_sided_studentized_arm_swap"
        ]
        assert exact["one_sided_exact_p"] == pytest.approx(386.0 / 1024.0)
        assert exact["selector_rerun_for_every_swap"] is True
        assert exact["swaps_changing_point_selection"] == 638
        assert exact["null_hypothesis"] == (
            "sharp_null_of_joint_dense_sparse_arm_exchangeability"
        )
        assert exact["sharp_null_unit"] == "complete_paired_model_seed_pipeline"
        assert "conditional" in exact["fixed_three_test_panel_qualification"]
        assert "three_frozen_h200" in exact[
            "fixed_validation_panel_qualification"
        ]
        bootstrap = endpoint["selection_aware_pipeline_inference"][
            "paired_seed_bootstrap"
        ]
        assert bootstrap["ci95_lower"] == pytest.approx(0.5)
        assert bootstrap["ci95_upper"] == pytest.approx(0.5)
        assert bootstrap["selector_rerun_for_every_replicate"] is True
        assert _frequency(bootstrap, "dense", DIRECT) == 2_000
        assert _frequency(bootstrap, "sparse", 1) == 2_000
        policy = endpoint["selection_aware_within_arm_selected_vs_direct"]
        assert policy["dense"]["selected_cadence"] == DIRECT
        assert policy["dense"]["heldout_point_relative_reduction"] == 0.0
        assert policy["sparse"]["selected_cadence"] == 1
        assert policy["sparse"]["heldout_point_relative_reduction"] == pytest.approx(0.5)
        assert policy["sparse"]["selection_aware_bootstrap_ci95_lower"] == pytest.approx(0.5)
        assert policy["sparse"]["selector_rerun_for_every_replicate"] is True


def test_h400_outcomes_cannot_change_the_validation_choices() -> None:
    rows = _h400_rows()
    for row in rows:
        oracle = (
            row["arm"] == "dense" and row["cadence"] == 1
        ) or (
            row["arm"] == "sparse" and row["cadence"] == DIRECT
        )
        if oracle:
            row["instantaneous_field_mse"] = [0.001] * 400
            row["cumulative_field_mse"] = [0.001] * 400
    result = summarize_full_grid_h400_pipeline(
        _validation_rows(),
        rows,
        _card(),
        bootstrap_replicates=211,
        bootstrap_seed=4,
    )
    assert result["selected_cadences_from_h200_validation"] == {
        "dense": DIRECT,
        "sparse": 1,
    }
    for endpoint in result["endpoints"].values():
        bootstrap = endpoint["selection_aware_pipeline_inference"][
            "paired_seed_bootstrap"
        ]
        assert _frequency(bootstrap, "dense", DIRECT) == 211
        assert _frequency(bootstrap, "sparse", 1) == 211


def test_h400_validator_requires_the_complete_finite_nine_cadence_cross() -> None:
    rows = _h400_rows()
    validate_full_grid_h400_rows(rows, _card())
    with pytest.raises(ValueError, match="exact frozen cross"):
        validate_full_grid_h400_rows(rows[:-1], _card())

    corrupted = deepcopy(rows)
    corrupted[0]["instantaneous_field_mse"][399] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        validate_full_grid_h400_rows(corrupted, _card())

    wrong_horizon = deepcopy(rows)
    wrong_horizon[0]["horizon_steps"] = 200
    with pytest.raises(ValueError, match="wrong horizon"):
        validate_full_grid_h400_rows(wrong_horizon, _card())

    wrong_grid = _card()
    wrong_grid["cadence_selection"]["cadence_grid"] = list(GRID[:-1])
    with pytest.raises(ValueError, match="nine frozen cadences"):
        validate_full_grid_h400_rows(rows, wrong_grid)


def test_h400_validator_rejects_nonfinite_unselected_cadence() -> None:
    rows = _h400_rows()
    unselected = next(
        row
        for row in rows
        if row["arm"] == "dense" and row["cadence"] == 100
    )
    unselected["cumulative_field_mse"][250] = np.inf
    with pytest.raises(FloatingPointError, match="nonfinite"):
        summarize_full_grid_h400_pipeline(
            _validation_rows(),
            rows,
            _card(),
            bootstrap_replicates=10,
        )
