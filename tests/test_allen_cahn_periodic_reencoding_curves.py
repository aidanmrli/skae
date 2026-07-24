from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.neurips_2026.allen_cahn_periodic_reencoding.curve_summary import (
    combined_accuracy_refresh_frontier,
    summarize_curve_panel,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import DIRECT


MODELS = tuple(range(64, 74))
DATASETS = (201, 202, 203)


def _card() -> dict:
    return {
        "roster": {"arms": ["dense", "sparse"], "model_seeds": list(MODELS)},
        "cadence_selection": {"cadence_grid": [DIRECT, 2]},
        "prospective_datasets": {
            "validation": [
                {"index": index, "seed": 100 + index} for index in range(3)
            ],
            "test": [
                {"index": index, "seed": seed}
                for index, seed in enumerate(DATASETS)
            ],
        },
        "system": {"stored_dt": 0.1},
    }


def _rows() -> list[dict]:
    rows = []
    for arm in ("dense", "sparse"):
        for model_seed in MODELS:
            for dataset_seed in DATASETS:
                for cadence in (DIRECT, 2):
                    value = 2.0 if arm == "dense" else 1.0
                    persistence = 4.0
                    rows.append(
                        {
                            "arm": arm,
                            "model_seed": model_seed,
                            "dataset_seed": dataset_seed,
                            "cadence": cadence,
                            "horizon_steps": 4,
                            "instantaneous_field_mse": [value] * 4,
                            "cumulative_field_mse": [value] * 4,
                            "instantaneous_persistence_mse": [persistence] * 4,
                            "cumulative_persistence_mse": [persistence] * 4,
                            "instantaneous_model_over_persistence": [
                                value / persistence
                            ] * 4,
                            "cumulative_model_over_persistence": [
                                value / persistence
                            ] * 4,
                        }
                    )
    return rows


def test_curve_panel_is_exact_balanced_and_cost_annotated() -> None:
    result = summarize_curve_panel(
        _rows(), _card(), cadences=[DIRECT, 2], horizon=4, tier="unit"
    )
    assert len(result["records"]) == 4
    dense_direct = next(
        row for row in result["records"]
        if row["arm"] == "dense" and row["cadence"] == DIRECT
    )
    assert dense_direct["mean_cumulative_field_mse"] == [2.0] * 4
    assert dense_direct["mean_cumulative_model_over_persistence"] == [0.5] * 4
    assert len(dense_direct["per_seed_cumulative_field_mse"]) == 10
    period_two = next(
        row for row in result["records"] if row["cadence"] == 2
    )
    assert period_two["refresh_count"] == 1
    assert period_two["encoder_calls"] == 2

    with pytest.raises(ValueError, match="exact requested cross"):
        summarize_curve_panel(
            _rows()[:-1], _card(), cadences=[DIRECT, 2], horizon=4, tier="unit"
        )
    corrupted = deepcopy(_rows())
    corrupted[0]["cumulative_field_mse"][2] = float("nan")
    with pytest.raises(FloatingPointError, match="invalid"):
        summarize_curve_panel(
            corrupted, _card(), cadences=[DIRECT, 2], horizon=4, tier="unit"
        )


def test_combined_frontier_appends_schema_identical_p200_row() -> None:
    grid_row = {
        "cadence": DIRECT,
        "dense_arm_mean_mse": 2.0,
        "sparse_arm_mean_mse": 1.0,
        "fixed_same_cadence_sparse_over_dense_ratio_of_arm_means": 0.5,
        "fixed_same_cadence_relative_reduction_of_arm_means": 0.5,
        "refresh_count": 0,
        "encoder_calls": 1,
        "rollout_horizon_steps": 400,
        "aggregation": "balanced_mean_over_ten_models_and_three_fixed_test_panels",
        "inference_role": "descriptive_only_not_used_for_selection",
    }
    full = {
        "endpoints": {
            "h400_cumulative_field_mse": {
                "descriptive_accuracy_compute_frontier": [grid_row]
            }
        }
    }
    p200 = {
        "status": "complete",
        "fixed_p200_sparse_vs_dense": {
            "endpoints": {
                "h400_cumulative_field_mse": {
                    "dense_mean": 1.5,
                    "sparse_mean": 0.9,
                    "sparse_over_dense_ratio_of_arm_means": 0.6,
                    "relative_reduction_of_arm_means": 0.4,
                }
            }
        },
    }
    result = combined_accuracy_refresh_frontier(full, p200)
    rows = result["endpoints"]["h400_cumulative_field_mse"]
    assert len(rows) == 2
    assert set(rows[0]) == set(rows[1])
    assert rows[1]["cadence"] == 200
    assert rows[1]["refresh_count"] == 1
    assert result["p200_included"] is True
