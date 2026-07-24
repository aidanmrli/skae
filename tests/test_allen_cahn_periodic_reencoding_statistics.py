from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.neurips_2026.allen_cahn_periodic_reencoding.policy_statistics import (
    summarize_selected_vs_direct,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    DIRECT,
    exact_one_sided_studentized_sign_flip,
    paired_ratio_bootstrap,
    select_recipe_cadences,
    summarize_test_rows,
    validate_primary_test_rows,
    validate_test_rows,
    validate_validation_rows,
    validation_candidate_scores,
)


def _card() -> dict:
    return {
        "cadence_grid": [DIRECT, 1, 2, 5, 10, 20, 25, 50, 100],
        "model_seeds": list(range(64, 74)),
        "validation_seeds": [101, 102, 103],
        "test_seeds": [201, 202, 203],
    }


def _validation_rows() -> list[dict]:
    card = _card()
    dense_scores = {
        DIRECT: 1.0,
        1: 3.0,
        2: 2.9,
        5: 2.8,
        10: 2.7,
        20: 2.6,
        25: 2.5,
        50: 2.0,
        100: 1.0,
    }
    sparse_scores = {
        DIRECT: 2.0,
        1: 0.5,
        2: 2.9,
        5: 2.8,
        10: 2.7,
        20: 2.6,
        25: 1.0,
        50: 2.0,
        100: 1.0,
    }
    rows = []
    for arm, scores in (("dense", dense_scores), ("sparse", sparse_scores)):
        for model_seed in card["model_seeds"]:
            for dataset_seed in card["validation_seeds"]:
                for cadence in card["cadence_grid"]:
                    score = scores[cadence]
                    instantaneous = np.full(200, score, dtype=np.float64)
                    cumulative = np.full(200, score, dtype=np.float64)
                    rows.append(
                        {
                            "arm": arm,
                            "model_seed": model_seed,
                            "dataset_seed": dataset_seed,
                            "cadence": cadence,
                            "horizon_steps": 200,
                            "instantaneous_field_mse": instantaneous.tolist(),
                            "cumulative_field_mse": cumulative.tolist(),
                        }
                    )
    return rows


def _test_rows(selected: dict[str, str | int]) -> list[dict]:
    card = _card()
    needed = {DIRECT, selected["dense"], selected["sparse"]}
    cadences = [value for value in card["cadence_grid"] if value in needed]
    rows = []
    for arm in ("dense", "sparse"):
        for model_index, model_seed in enumerate(card["model_seeds"]):
            for dataset_index, dataset_seed in enumerate(card["test_seeds"]):
                scale = 1.0 + 0.01 * model_index + 0.02 * dataset_index
                for cadence in cadences:
                    if arm == "dense":
                        factor = 2.0
                    elif cadence == 100:
                        factor = 1.0
                    else:
                        # Direct sparse happens to win even more strongly on
                        # test.  The reducer must nevertheless retain the
                        # validation-frozen cadence 100 for its primary result.
                        factor = 0.25
                    step_scale = 1.0 + 0.001 * np.arange(400)
                    instantaneous = factor * scale * step_scale
                    cumulative = np.cumsum(instantaneous) / np.arange(1, 401)
                    rows.append(
                        {
                            "arm": arm,
                            "model_seed": model_seed,
                            "dataset_seed": dataset_seed,
                            "cadence": cadence,
                            "horizon_steps": 400,
                            "instantaneous_field_mse": instantaneous.tolist(),
                            "cumulative_field_mse": cumulative.tolist(),
                        }
                    )
    return rows


def _primary_test_rows() -> list[dict]:
    card = _card()
    rows = []
    for arm in ("dense", "sparse"):
        for model_seed in card["model_seeds"]:
            for dataset_seed in card["test_seeds"]:
                for cadence in card["cadence_grid"]:
                    instantaneous = np.ones(200, dtype=np.float64)
                    rows.append(
                        {
                            "arm": arm,
                            "model_seed": model_seed,
                            "dataset_seed": dataset_seed,
                            "cadence": cadence,
                            "horizon_steps": 200,
                            "instantaneous_field_mse": instantaneous.tolist(),
                            "cumulative_field_mse": instantaneous.tolist(),
                        }
                    )
    return rows


def test_recipe_selection_uses_exact_global_minimum_and_frozen_ties() -> None:
    rows = _validation_rows()
    validate_validation_rows(rows, _card())
    scores = validation_candidate_scores(rows, _card())
    assert all(row["eligible"] for arm in scores.values() for row in arm)
    assert all(row["nonfinite_value_count"] == 0 for arm in scores.values() for row in arm)

    # Direct beats period 100 on an exact dense tie. Sparse period 1 is the
    # literal minimum; no candidate may be silently dropped.
    assert select_recipe_cadences(rows, _card()) == {
        "dense": DIRECT,
        "sparse": 1,
    }


def test_any_nonfinite_validation_value_invalidates_entire_packet() -> None:
    rows = _validation_rows()
    rows[0]["cumulative_field_mse"][31] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite test value"):
        validate_validation_rows(rows, _card())
    with pytest.raises(FloatingPointError, match="nonfinite test value"):
        validation_candidate_scores(rows, _card())
    with pytest.raises(FloatingPointError, match="nonfinite test value"):
        select_recipe_cadences(rows, _card())


def test_primary_test_validator_requires_complete_finite_full_grid() -> None:
    rows = _primary_test_rows()
    validate_primary_test_rows(rows, _card())
    with pytest.raises(ValueError, match="exact frozen cross"):
        validate_primary_test_rows(rows[:-1], _card())
    corrupted = deepcopy(rows)
    corrupted[0]["instantaneous_field_mse"][5] = float("inf")
    with pytest.raises(FloatingPointError, match="nonfinite test value"):
        validate_primary_test_rows(corrupted, _card())


def test_canonical_nested_card_schema_is_consumed() -> None:
    flat = _card()
    nested = {
        "roster": {
            "arms": ["dense", "sparse"],
            "model_seeds": flat["model_seeds"],
        },
        "cadence_selection": {"cadence_grid": flat["cadence_grid"]},
        "prospective_datasets": {
            "validation": [
                {"index": index, "seed": seed}
                for index, seed in enumerate(flat["validation_seeds"])
            ],
            "test": [
                {"index": index, "seed": seed}
                for index, seed in enumerate(flat["test_seeds"])
            ],
        },
    }
    assert select_recipe_cadences(_validation_rows(), nested) == {
        "dense": DIRECT,
        "sparse": 1,
    }
    nested["roster"]["arms"] = ["sparse", "dense"]
    with pytest.raises(ValueError, match="roster.arms"):
        validate_validation_rows(_validation_rows(), nested)


def test_validation_roster_requires_the_equal_complete_grid() -> None:
    rows = _validation_rows()
    with pytest.raises(ValueError, match="exact frozen cross"):
        validate_validation_rows(rows[:-1], _card())

    duplicate = rows + [deepcopy(rows[0])]
    with pytest.raises(ValueError, match="Duplicate forecast row"):
        validate_validation_rows(duplicate, _card())


def test_exact_sign_flip_is_literal_and_enumerates_every_vector() -> None:
    result = exact_one_sided_studentized_sign_flip(np.ones(10))
    assert result["enumerated_sign_vectors"] == 1024
    assert result["one_sided_exact_p"] == pytest.approx(1.0 / 1024.0)
    assert result["comparison"] == "T_perm >= T_observed_literal_no_tolerance"
    assert result["observed_studentized_statistic"] is None
    assert result["observed_studentized_statistic_status"] == "positive_infinity"
    assert json.loads(json.dumps(result, allow_nan=False)) == result

    ties = exact_one_sided_studentized_sign_flip(np.zeros(10))
    assert ties["one_sided_exact_p"] == 1.0
    assert ties["observed_studentized_statistic"] == 0.0
    assert ties["observed_studentized_statistic_status"] == "finite"


def test_paired_ratio_bootstrap_is_deterministic_and_ratio_of_means() -> None:
    dense = np.arange(1.0, 11.0)
    sparse = 0.5 * dense
    first = paired_ratio_bootstrap(dense, sparse, replicates=2_000, seed=17)
    second = paired_ratio_bootstrap(dense, sparse, replicates=2_000, seed=17)
    assert first == second
    assert first["relative_reduction_of_arm_means"] == pytest.approx(0.5)
    assert first["ci95_lower"] == pytest.approx(0.5)
    assert first["ci95_upper"] == pytest.approx(0.5)


def test_test_summary_uses_three_dataset_seed_means_and_all_frozen_endpoints() -> None:
    selected = {"dense": DIRECT, "sparse": 100}
    rows = _test_rows(selected)
    validate_test_rows(rows, _card(), selected)
    summary = summarize_test_rows(
        rows,
        _card(),
        selected,
        bootstrap_replicates=2_000,
        bootstrap_seed=41,
    )
    assert summary["selected_cadences_from_validation"] == selected
    primary = summary["selected_recipe_comparison"]
    assert primary["dense_cadence"] == DIRECT
    assert primary["sparse_cadence"] == 100
    assert set(primary["endpoints"]) == {
        "h200_cumulative_field_mse",
        "h400_cumulative_field_mse",
        "h201_h400_tail_field_mse",
        "h400_terminal_field_mse",
    }
    for endpoint in primary["endpoints"].values():
        assert endpoint["relative_reduction_of_arm_means"] == pytest.approx(0.5)
        assert endpoint["sparse_over_dense_ratio_of_arm_means"] == pytest.approx(0.5)
        assert endpoint["sparse_seed_wins"] == 10
        assert endpoint["paired_ratio_bootstrap"]["ci95_lower"] == pytest.approx(0.5)
        assert endpoint["paired_ratio_bootstrap"]["ci95_upper"] == pytest.approx(0.5)
        assert len(endpoint["dense_paired_seed_values"]) == 10
        assert len(endpoint["sparse_paired_seed_values"]) == 10
        assert len(endpoint["per_dataset_effects"]) == 3
        assert all(
            effect["relative_reduction_of_arm_means"] == pytest.approx(0.5)
            for effect in endpoint["per_dataset_effects"]
        )

    # Test performance would favor sparse direct, but it appears only as a
    # named sensitivity and cannot replace the primary validation choice.
    direct = summary["same_cadence_sensitivity"][DIRECT]
    assert direct["endpoints"]["h200_cumulative_field_mse"][
        "relative_reduction_of_arm_means"
    ] == pytest.approx(0.875)


def test_selected_vs_direct_reports_policy_gain_and_exact_identity() -> None:
    selected = {"dense": DIRECT, "sparse": 100}
    rows = _test_rows(selected)
    # Make sparse direct twice the error of sparse period 100 at every cell.
    # Dense selected=direct remains an exact identity comparison.
    for row in rows:
        if row["arm"] == "sparse" and row["cadence"] == DIRECT:
            row["instantaneous_field_mse"] = (
                8.0 * np.asarray(row["instantaneous_field_mse"])
            ).tolist()
            row["cumulative_field_mse"] = (
                8.0 * np.asarray(row["cumulative_field_mse"])
            ).tolist()
    summary = summarize_selected_vs_direct(
        rows,
        _card(),
        selected,
        bootstrap_replicates=2_000,
        bootstrap_seed=53,
    )
    dense = summary["arms"]["dense"]
    sparse = summary["arms"]["sparse"]
    assert dense["selected_cadence"] == DIRECT
    assert sparse["selected_cadence"] == 100
    for endpoint in dense["endpoints"].values():
        assert endpoint["identity_comparison_selected_is_direct"] is True
        assert endpoint["relative_reduction_of_arm_means"] == 0.0
        assert endpoint["selected_over_direct_ratio_of_arm_means"] == 1.0
        assert endpoint["selected_seed_wins"] == 0
        assert endpoint["paired_ratio_bootstrap"]["ci95_lower"] == 0.0
        assert endpoint["paired_ratio_bootstrap"]["ci95_upper"] == 0.0
        assert endpoint["exact_one_sided_studentized_sign_flip"][
            "one_sided_exact_p"
        ] == 1.0
    for endpoint in sparse["endpoints"].values():
        assert endpoint["identity_comparison_selected_is_direct"] is False
        assert endpoint["relative_reduction_of_arm_means"] == pytest.approx(0.5)
        assert endpoint["selected_over_direct_ratio_of_arm_means"] == pytest.approx(0.5)
        assert endpoint["selected_seed_wins"] == 10
        assert endpoint["paired_ratio_bootstrap"]["ci95_lower"] == pytest.approx(0.5)
        assert endpoint["paired_ratio_bootstrap"]["ci95_upper"] == pytest.approx(0.5)
        assert endpoint["exact_one_sided_studentized_sign_flip"][
            "one_sided_exact_p"
        ] == pytest.approx(1.0 / 1024.0)


def test_test_roster_and_finiteness_are_strict() -> None:
    selected = {"dense": DIRECT, "sparse": 100}
    rows = _test_rows(selected)
    with pytest.raises(ValueError, match="exact frozen cross"):
        validate_test_rows(rows[:-1], _card(), selected)

    corrupted = deepcopy(rows)
    corrupted[0]["instantaneous_field_mse"][17] = np.inf
    with pytest.raises(FloatingPointError, match="nonfinite test value"):
        validate_test_rows(corrupted, _card(), selected)


def test_selection_map_and_card_partitions_are_frozen() -> None:
    selected = {"dense": DIRECT, "sparse": 100}
    with pytest.raises(ValueError, match="exactly dense and sparse"):
        validate_test_rows(_test_rows(selected), _card(), {"dense": DIRECT})

    overlapping = _card()
    overlapping["test_seeds"][0] = overlapping["validation_seeds"][0]
    with pytest.raises(ValueError, match="must be disjoint"):
        validate_validation_rows(_validation_rows(), overlapping)
