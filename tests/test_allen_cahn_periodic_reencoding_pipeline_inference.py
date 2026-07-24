from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.neurips_2026.allen_cahn_periodic_reencoding.h200_sensitivities import (
    summarize_same_cadence_h200,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.pipeline_inference import (
    selection_aware_paired_bootstrap,
    summarize_pipeline_inference,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.policy_pipeline import (
    summarize_policy_pipeline_bootstrap,
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


def _risk(arm: str, cadence: str | int, *, role: str) -> float:
    if role == "validation":
        if arm == "dense":
            return {DIRECT: 1.0, 1: 2.0}.get(cadence, 5.0)
        return {DIRECT: 2.0, 1: 1.0}.get(cadence, 5.0)
    # Identical arm-wise test vectors make the exact effect of rerunning the
    # selectors under seed-wise arm swaps analytically transparent.
    return {DIRECT: 2.0, 1: 1.0}.get(cadence, 5.0)


def _rows(*, role: str) -> list[dict]:
    datasets = VALIDATION_DATASETS if role == "validation" else TEST_DATASETS
    rows = []
    for arm in ("dense", "sparse"):
        for model in MODELS:
            for dataset in datasets:
                for cadence in GRID:
                    value = _risk(arm, cadence, role=role)
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


def _seed_risks(*, role: str) -> np.ndarray:
    result = np.empty((2, 10, len(GRID)), dtype=np.float64)
    for arm_index, arm in enumerate(("dense", "sparse")):
        for cadence_index, cadence in enumerate(GRID):
            result[arm_index, :, cadence_index] = _risk(
                arm, cadence, role=role
            )
    return result


def _frequency(payload: dict, arm: str, cadence: str | int) -> int:
    return next(
        row["count"]
        for row in payload["selection_frequencies"][arm]
        if row["cadence"] == cadence
    )


def test_pipeline_exact_test_reruns_selection_and_differs_from_naive() -> None:
    summary = summarize_pipeline_inference(
        _rows(role="validation"),
        _rows(role="test"),
        _card(),
        bootstrap_replicates=2_000,
        bootstrap_seed=17,
        bootstrap_chunk_size=257,
        permutation_chunk_size=73,
    )
    assert summary["selected_cadences_from_validation"] == {
        "dense": DIRECT,
        "sparse": 1,
    }
    point = summary["primary_point_test"]
    assert point["relative_reduction_of_arm_means"] == pytest.approx(0.5)
    assert point["sparse_seed_wins"] == 10
    assert len(point["per_dataset_effects"]) == 3
    assert all(
        row["relative_reduction_of_arm_means"] == pytest.approx(0.5)
        for row in point["per_dataset_effects"]
    )

    naive = summary["conditional_fixed_selection_inference"][
        "exact_one_sided_studentized_sign_flip"
    ]
    pipeline = summary["selection_aware_pipeline_inference"][
        "exact_one_sided_studentized_sign_flip"
    ]
    assert naive["one_sided_exact_p"] == pytest.approx(1.0 / 1024.0)
    assert pipeline["one_sided_exact_p"] == pytest.approx(386.0 / 1024.0)
    assert pipeline["exceedances_literal_greater_equal"] == 386
    assert pipeline["swaps_changing_point_selection"] == 638
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    assert _frequency(pipeline, "dense", DIRECT) == 638
    assert _frequency(pipeline, "dense", 1) == 386
    assert _frequency(pipeline, "sparse", DIRECT) == 638
    assert _frequency(pipeline, "sparse", 1) == 386

    bootstrap = summary["selection_aware_pipeline_inference"][
        "paired_seed_bootstrap"
    ]
    assert bootstrap["relative_reduction_of_arm_means"] == pytest.approx(0.5)
    assert bootstrap["ci95_lower"] == pytest.approx(0.5)
    assert bootstrap["ci95_upper"] == pytest.approx(0.5)
    assert _frequency(bootstrap, "dense", DIRECT) == 2_000
    assert _frequency(bootstrap, "sparse", 1) == 2_000

    policy = summary["within_arm_selected_vs_direct_h200"]
    assert policy["dense"]["identity_comparison_selected_is_direct"] is True
    assert policy["dense"]["relative_reduction_of_arm_means"] == 0.0
    assert policy["dense"]["exact_one_sided_studentized_sign_flip"][
        "one_sided_exact_p"
    ] == 1.0
    assert policy["sparse"]["relative_reduction_of_arm_means"] == pytest.approx(0.5)
    assert policy["sparse"]["selected_seed_wins"] == 10


def test_pipeline_bootstrap_is_deterministic_across_chunking() -> None:
    validation = _seed_risks(role="validation")
    test = _seed_risks(role="test")
    first = selection_aware_paired_bootstrap(
        validation, test, GRID, replicates=2_137, seed=99, chunk_size=113
    )
    second = selection_aware_paired_bootstrap(
        validation, test, GRID, replicates=2_137, seed=99, chunk_size=401
    )
    assert first == second


def test_primary_test_outcomes_cannot_choose_the_cadence() -> None:
    validation_rows = _rows(role="validation")
    test_rows = _rows(role="test")
    # Make each arm's validation-selected cadence look worse than another test
    # cadence.  The output must retain the validation choices anyway.
    for row in test_rows:
        oracle_test_choice = (
            row["arm"] == "dense" and row["cadence"] == 1
        ) or (row["arm"] == "sparse" and row["cadence"] == DIRECT)
        if oracle_test_choice:
            row["instantaneous_field_mse"] = [0.01] * 200
            row["cumulative_field_mse"] = [0.01] * 200
    summary = summarize_pipeline_inference(
        validation_rows,
        test_rows,
        _card(),
        bootstrap_replicates=211,
        bootstrap_seed=4,
    )
    assert summary["selected_cadences_from_validation"] == {
        "dense": DIRECT,
        "sparse": 1,
    }
    bootstrap = summary["selection_aware_pipeline_inference"][
        "paired_seed_bootstrap"
    ]
    assert _frequency(bootstrap, "dense", DIRECT) == 211
    assert _frequency(bootstrap, "sparse", 1) == 211
    assert summary["test_selection_forbidden"] is True


def test_pipeline_requires_the_full_finite_nine_cadence_cross() -> None:
    validation_rows = _rows(role="validation")
    test_rows = _rows(role="test")
    with pytest.raises(ValueError, match="exact frozen cross"):
        summarize_pipeline_inference(
            validation_rows[:-1],
            test_rows,
            _card(),
            bootstrap_replicates=10,
        )

    corrupted = deepcopy(validation_rows)
    corrupted[0]["cumulative_field_mse"][8] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        summarize_pipeline_inference(
            corrupted,
            test_rows,
            _card(),
            bootstrap_replicates=10,
        )


def test_fixed_h200_sensitivities_report_every_cadence_without_selection() -> None:
    result = summarize_same_cadence_h200(
        _rows(role="test"), _card(), bootstrap_replicates=101, bootstrap_seed=9
    )
    assert set(result) == {"direct", *(f"period_{value}" for value in GRID[1:])}
    assert result["direct"]["relative_reduction_of_arm_means"] == 0.0
    assert result["period_1"]["relative_reduction_of_arm_means"] == 0.0
    assert all(
        row["inference_role"] == "mandatory_descriptive_sensitivity"
        for row in result.values()
    )


def test_within_arm_pipeline_bootstrap_reruns_validation_selector() -> None:
    result = summarize_policy_pipeline_bootstrap(
        _rows(role="validation"),
        _rows(role="test"),
        _card(),
        bootstrap_replicates=313,
        bootstrap_seed=11,
        chunk_size=37,
    )
    dense, sparse = result["arms"]["dense"], result["arms"]["sparse"]
    assert dense["selected_cadence"] == DIRECT
    assert dense["heldout_point_relative_reduction"] == 0.0
    assert sparse["selected_cadence"] == 1
    assert sparse["heldout_point_relative_reduction"] == pytest.approx(0.5)
    assert sparse["selection_aware_bootstrap_ci95_lower"] == pytest.approx(0.5)
    assert sparse["selection_aware_bootstrap_ci95_upper"] == pytest.approx(0.5)
    assert sum(row["count"] for row in sparse["selection_frequencies"]) == 313
    assert result["selector_rerun_for_every_replicate"] is True
