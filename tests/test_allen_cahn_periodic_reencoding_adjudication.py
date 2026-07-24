from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.neurips_2026.allen_cahn_periodic_reencoding.adjudication import (
    adjudicate,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.forecast_skill import (
    deployment_cost,
    summarize_selected_absolute_skill,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.lineage import (
    canonical_digest,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    DIRECT,
    select_recipe_cadences,
    validation_candidate_scores,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.summary_integrity import (
    verify_h400_failure_lineage,
    verify_selection_lineage,
)


MODELS = tuple(range(64, 74))
VALIDATION_DATASETS = (101, 102, 103)
TEST_DATASETS = (201, 202, 203)
TAIL = "h201_h400_tail_field_mse"


def _endpoint(
    reduction: float = 0.10,
    *,
    ci_lower: float = 0.03,
    p_value: float = 0.01,
    wins: int = 10,
) -> dict:
    return {
        "relative_reduction_of_arm_means": reduction,
        "paired_ratio_bootstrap": {"ci95_lower": ci_lower},
        "exact_one_sided_studentized_sign_flip": {
            "one_sided_exact_p": p_value
        },
        "sparse_seed_wins": wins,
        "selected_seed_wins": wins,
        "per_dataset_effects": [
            {"relative_reduction_of_arm_means": reduction} for _ in range(3)
        ],
    }


def _absolute(
    *, ratio: float = 0.5, all_three: bool = True, h400: bool = False
) -> dict:
    name = TAIL if h400 else "h200_cumulative_field_mse"
    return {
        "endpoints": {
            name: {
                "sparse": {
                    "model_over_x0_persistence": ratio,
                    "all_three_dataset_ratios_below_one": all_three,
                }
            }
        }
    }


def _policy_h200(
    *, dense: str | int = DIRECT, sparse: str | int = DIRECT
) -> dict:
    def arm(cadence: str | int) -> dict:
        periodic = cadence != DIRECT
        return {
            "selected_cadence": cadence,
            "heldout_point_relative_reduction": 0.20 if periodic else 0.0,
            "selection_aware_bootstrap_ci95_lower": 0.05 if periodic else 0.0,
        }

    return {
        "conditional": {"dense": {}, "sparse": {}},
        "selection_aware": {"dense": arm(dense), "sparse": arm(sparse)},
    }


def _stress(
    tail: dict | None = None,
) -> tuple[dict, dict, dict]:
    selected_tail = _endpoint() if tail is None else tail
    summary = {
        "selected_recipe_comparison": {"endpoints": {TAIL: selected_tail}},
        "same_cadence_sensitivity": {
            DIRECT: {"endpoints": {TAIL: _endpoint()}}
        },
    }
    policy = {
        "arms": {
            arm: {"endpoints": {TAIL: _endpoint()}}
            for arm in ("dense", "sparse")
        }
    }
    return summary, policy, selected_tail


def _truth(*ratios: float) -> list[dict]:
    assert len(ratios) == 3
    return [
        {"late_over_early_one_step_truth_change_ratio": value}
        for value in ratios
    ]


def _decision(
    *,
    primary: dict | None = None,
    policy_h200: dict | None = None,
    tail: dict | None = None,
    stress_available: bool = True,
    failures: list[dict] | None = None,
    truth: list[dict] | None = None,
    h200_absolute: dict | None = None,
    h400_absolute: dict | None = None,
    h400_policy: dict | None = None,
) -> dict:
    summary, stress_policy, selection_aware_tail = _stress(tail)
    if not stress_available:
        summary = stress_policy = selection_aware_tail = None
    return adjudicate(
        primary=_endpoint() if primary is None else primary,
        direct_h200=_endpoint(),
        policy_h200=_policy_h200() if policy_h200 is None else policy_h200,
        absolute_skill_h200=_absolute() if h200_absolute is None else h200_absolute,
        absolute_skill_h400=(
            _absolute(h400=True) if h400_absolute is None else h400_absolute
        )
        if stress_available
        else None,
        stress_summary=summary,
        stress_policy=stress_policy,
        selection_aware_h400_tail=selection_aware_tail,
        stress_failures=[] if failures is None else failures,
        truth_difficulty=_truth(0.2, 0.3, 0.4) if truth is None else truth,
        truth_threshold=0.05,
        selection_aware_h400_policy=h400_policy,
    )


def test_both_direct_and_supported_periodic_policies_have_distinct_branches() -> None:
    direct = _decision()
    assert direct["branch"] == (
        "selection_aware_h400_durability_with_direct_selected_for_both_arms"
    )
    assert all(
        row["status"] == "direct_selected_no_periodic_policy_claim"
        for row in direct["periodic_selection_generalization_by_arm"].values()
    )

    periodic = _decision(policy_h200=_policy_h200(sparse=20))
    assert periodic["branch"] == (
        "selection_aware_h400_durability_with_refresh_dependent_policies"
    )
    assert periodic["periodic_selection_generalization_by_arm"]["sparse"][
        "status"
    ] == "positive_heldout_gain_with_selection_aware_interval_above_zero"
    assert periodic["periodic_tail_support_by_arm"]["sparse"][
        "passed"
    ]


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        (
            _endpoint(0.10, ci_lower=-0.01),
            "positive_but_inconclusive_h200_sparse_recipe_advantage",
        ),
        (
            _endpoint(-0.10, ci_lower=-0.20, wins=0),
            "selected_sparse_recipe_no_heldout_h200_advantage",
        ),
    ],
)
def test_h200_uncertainty_is_not_conflated_with_direction_reversal(
    primary: dict, expected: str
) -> None:
    assert _decision(primary=primary)["branch"] == expected


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        (
            _endpoint(0.10, ci_lower=-0.01),
            "positive_h201_h400_effect_but_selection_aware_durability_inconclusive",
        ),
        (
            _endpoint(-0.10, ci_lower=-0.20, wins=0),
            "strong_h200_advantage_reverses_in_h201_h400",
        ),
    ],
)
def test_h400_uncertainty_is_not_conflated_with_direction_reversal(
    tail: dict, expected: str
) -> None:
    assert _decision(tail=tail)["branch"] == expected


def test_required_h400_failure_suppresses_stress_claim_without_erasing_h200() -> None:
    failures = [{"arm": "sparse", "cadence": 20, "reason": "nonfinite"}]
    decision = _decision(stress_available=False, failures=failures)
    assert decision["branch"] == "strong_h200_sparse_recipe_h400_stress_unavailable"
    assert decision["strong_selection_aware_h200_selected_policy"]["passed"]
    assert decision["h201_h400_selected_policy"] is None
    assert decision["h400_stress_failures"] == failures


def test_truth_gate_distinguishes_all_three_stationary_from_heterogeneous() -> None:
    stationary = _decision(truth=_truth(0.01, 0.02, 0.03))
    assert stationary["branch"] == "strong_h200_with_h400_pattern_retention_only"
    scope = stationary["h400_truth_dynamics_language_gate"]
    assert scope["all_three_near_stationary"] is True
    assert scope["permitted_h400_language"] == "attractor_or_pattern_retention_only"

    heterogeneous = _decision(truth=_truth(0.01, 0.20, 0.30))
    assert heterogeneous["branch"] == (
        "strong_h200_with_h400_dataset_specific_dynamics_only"
    )
    scope = heterogeneous["h400_truth_dynamics_language_gate"]
    assert scope["all_three_near_stationary"] is False
    assert scope["heterogeneous_near_stationarity"] is True
    assert scope["permitted_h400_language"] == (
        "heterogeneous_late_dynamics_dataset_specific_claims_only"
    )


def test_selection_aware_h400_within_arm_policy_gate() -> None:
    supported = {
        "dense": {},
        "sparse": {
            "heldout_point_relative_reduction": 0.2,
            "selection_aware_bootstrap_ci95_lower": 0.05,
            "heldout_point_selected_seed_wins": 9,
        },
    }
    result = _decision(
        policy_h200=_policy_h200(sparse=20), h400_policy=supported
    )
    assert result["branch"] == (
        "selection_aware_h400_durability_with_refresh_dependent_policies"
    )
    assert result["periodic_tail_support_by_arm"]["sparse"]["passed"]
    uncertain = deepcopy(supported)
    uncertain["sparse"]["selection_aware_bootstrap_ci95_lower"] = -0.01
    result = _decision(
        policy_h200=_policy_h200(sparse=20), h400_policy=uncertain
    )
    assert result["branch"].endswith("periodic_policy_benefit_not_supported")


@pytest.mark.parametrize(
    "absolute",
    [_absolute(ratio=1.01), _absolute(ratio=0.80, all_three=False)],
)
def test_absolute_persistence_is_a_required_h200_claim_gate(absolute: dict) -> None:
    decision = _decision(h200_absolute=absolute)
    checks = decision["strong_selection_aware_h200_selected_policy"]
    assert checks["passed"] is False
    assert decision["branch"] == "positive_h200_sparse_recipe_effect_below_full_strong_gate"


def test_absolute_persistence_is_also_required_for_h400_durability() -> None:
    decision = _decision(h400_absolute=_absolute(ratio=1.01, h400=True))
    checks = decision["h201_h400_selected_policy"]
    assert checks["sparse_beats_x0_persistence"] is False
    assert checks["passed"] is False
    assert decision["branch"] == (
        "positive_h201_h400_effect_but_selection_aware_durability_inconclusive"
    )


def test_absolute_skill_uses_selected_cross_and_all_dataset_persistence_gate() -> None:
    card = _card()
    selected = {"dense": DIRECT, "sparse": 20}
    rows = _absolute_rows(selected)
    result = summarize_selected_absolute_skill(rows, card, selected, horizon=200)
    endpoint = result["endpoints"]["h200_cumulative_field_mse"]
    assert endpoint["dense"]["model_over_x0_persistence"] == pytest.approx(0.75)
    assert endpoint["sparse"]["model_over_x0_persistence"] == pytest.approx(0.50)
    assert endpoint["sparse"]["all_three_dataset_ratios_below_one"] is True

    for row in rows:
        if row["arm"] == "sparse" and row["dataset_seed"] == TEST_DATASETS[-1]:
            row["instantaneous_field_mse"] = [1.1] * 200
    result = summarize_selected_absolute_skill(rows, card, selected, horizon=200)
    sparse = result["endpoints"]["h200_cumulative_field_mse"]["sparse"]
    assert sparse["model_over_x0_persistence"] == pytest.approx(0.7)
    assert sparse["all_three_dataset_ratios_below_one"] is False


def test_deployment_cost_counts_only_used_boundary_refreshes() -> None:
    assert deployment_cost(DIRECT, 400) == {
        "horizon_steps": 400,
        "refresh_count": 0,
        "encoder_calls": 1,
        "decoder_calls": 400,
        "latent_k_steps": 400,
    }
    assert deployment_cost(1, 400)["refresh_count"] == 399
    assert deployment_cost(20, 200)["refresh_count"] == 9
    assert deployment_cost(100, 400)["refresh_count"] == 3
    assert deployment_cost(400, 400)["encoder_calls"] == 1
    with pytest.raises(ValueError, match="positive"):
        deployment_cost(0, 400)


def _card() -> dict:
    return {
        "protocol_id": "unit-periodic",
        "cadence_grid": [DIRECT, 20],
        "model_seeds": list(MODELS),
        "validation_seeds": list(VALIDATION_DATASETS),
        "test_seeds": list(TEST_DATASETS),
    }


def _absolute_rows(selected: dict[str, str | int]) -> list[dict]:
    rows = []
    for arm in ("dense", "sparse"):
        model_value = 0.75 if arm == "dense" else 0.50
        for model_seed in MODELS:
            for dataset_seed in TEST_DATASETS:
                rows.append(
                    {
                        "arm": arm,
                        "model_seed": model_seed,
                        "dataset_seed": dataset_seed,
                        "cadence": selected[arm],
                        "horizon_steps": 200,
                        "instantaneous_field_mse": [model_value] * 200,
                        "instantaneous_persistence_mse": [1.0] * 200,
                    }
                )
    return rows


def _validation_rows() -> list[dict]:
    rows = []
    for arm in ("dense", "sparse"):
        for model_seed in MODELS:
            for dataset_seed in VALIDATION_DATASETS:
                for cadence in (DIRECT, 20):
                    value = 1.0 if (arm == "dense") == (cadence == DIRECT) else 2.0
                    rows.append(
                        {
                            "arm": arm,
                            "model_seed": model_seed,
                            "dataset_seed": dataset_seed,
                            "cadence": cadence,
                            "horizon_steps": 200,
                            "instantaneous_field_mse": [value] * 200,
                            "cumulative_field_mse": [value] * 200,
                        }
                    )
    return rows


def _sealed_selection(rows: list[dict], card: dict) -> tuple[dict, dict]:
    selected = select_recipe_cadences(rows, card)
    decision = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": "card-hash",
        "source_manifest_sha256": "source-hash",
        "selection_endpoint": "H200 cumulative field MSE",
        "selection_scope": "one recipe-level cadence per arm",
        "selected_cadences": selected,
        "candidate_scores": validation_candidate_scores(rows, card),
        "validation_rows_sha256": canonical_digest(rows),
    }
    return decision, {"selected_cadences": selected}


def test_selection_lineage_rejects_digest_score_choice_and_payload_tampering() -> None:
    card, rows = _card(), _validation_rows()
    selection, scientific = _sealed_selection(rows, card)
    verified = verify_selection_lineage(
        selection,
        scientific,
        rows,
        card,
        card_hash="card-hash",
        source_hash="source-hash",
    )
    assert verified["candidate_scores_recomputed_exactly"] is True

    altered_rows = deepcopy(rows)
    altered_rows[0]["cumulative_field_mse"][0] = 1.5
    with pytest.raises(RuntimeError, match="digest"):
        verify_selection_lineage(
            selection,
            scientific,
            altered_rows,
            card,
            card_hash="card-hash",
            source_hash="source-hash",
        )

    altered_selection = deepcopy(selection)
    altered_selection["candidate_scores"]["dense"][0][
        "h200_cumulative_field_mse"
    ] = 9.0
    with pytest.raises(RuntimeError, match="candidate scores"):
        verify_selection_lineage(
            altered_selection,
            scientific,
            rows,
            card,
            card_hash="card-hash",
            source_hash="source-hash",
        )

    altered_selection = deepcopy(selection)
    altered_selection["selected_cadences"]["sparse"] = DIRECT
    with pytest.raises(RuntimeError, match="cadence choice"):
        verify_selection_lineage(
            altered_selection,
            scientific,
            rows,
            card,
            card_hash="card-hash",
            source_hash="source-hash",
        )

    altered_scientific = deepcopy(scientific)
    altered_scientific["selected_cadences"]["sparse"] = DIRECT
    with pytest.raises(RuntimeError, match="Scientific payload"):
        verify_selection_lineage(
            selection,
            altered_scientific,
            rows,
            card,
            card_hash="card-hash",
            source_hash="source-hash",
        )


def test_h400_failure_lineage_authenticates_nested_tiers() -> None:
    card = {
        "roster": {"model_seeds": list(MODELS)},
        "cadence_selection": {"cadence_grid": [DIRECT, 20]},
    }
    failure = {
        "arm": "dense",
        "model_seed": 64,
        "cadence": DIRECT,
        "status": "whole_h400_policy_nonfinite",
        "error_type": "FloatingPointError",
        "finite_prefix_scored": False,
    }
    scientific = {
        "stress_failures": [failure],
        "required_stress_failures": [failure],
        "grid_stress_failures": [failure],
        "p200_failures": [],
    }
    result = verify_h400_failure_lineage(
        scientific, card, {"dense": DIRECT, "sparse": 20}
    )
    assert result == {"all": 1, "required": 1, "full_grid": 1, "p200": 0}

    tampered = deepcopy(scientific)
    tampered["p200_failures"] = [failure]
    with pytest.raises(RuntimeError, match="p200 H400 failure classification"):
        verify_h400_failure_lineage(
            tampered, card, {"dense": DIRECT, "sparse": 20}
        )
