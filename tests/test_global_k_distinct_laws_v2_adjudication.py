import json
from pathlib import Path

import numpy as np
import pytest

from experiments.neurips_2026.summarize_global_k_distinct_laws_v2 import (
    _exact_sign_p,
    _safe_ratio,
    adjudicate,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = json.loads((
    ROOT / "experiments/neurips_2026/global_k_distinct_laws_v2_card.json"
).read_text())
CARD_HASH = "synthetic-card"
TASK_HASH = "synthetic-task"


def _law(value: float) -> dict:
    costs = [[value if row == column else 1.0 for column in range(3)] for row in range(3)]
    return {
        "cost_matrix": costs,
        "max_own_relative_error": value,
        "max_own_over_nearest_wrong": value,
        "identity_over_best_nonidentity": value,
        "identity_is_unique_optimum": True,
        "own_over_nearest_wrong_by_basin": [value] * 3,
    }


def _result(
    arm: str, *, joint=True, h=True, g=True, h_global=True,
    closure_valid=True, closure_pass=True, affine=True, finite=True,
    residual=True, kink=(True, True, True), value=None,
) -> dict:
    metric = (0.1 if arm == "sparse" else 0.3) if value is None else value
    kink_rows = [
        {"basin": basin, "passed_both_estimands_both_epsilons": passed}
        for basin, passed in enumerate(kink)
    ]
    radius = [
        {
            "basin": basin,
            "H": [
                {"autograd_agreement": 0.1 if finite else 0.5,
                 "own_law_is_nearest": finite,
                 "normalized_linear_fit_residual": 0.1 if residual else 0.5}
                for _ in range(5)
            ],
            "G": [
                {"autograd_agreement": 0.1 if finite else 0.5,
                 "own_law_is_nearest": finite,
                 "normalized_linear_fit_residual": 0.1 if residual else 0.5}
                for _ in range(5)
            ],
        }
        for basin in range(3)
    ]
    direct = {
        "per_seed_pass": closure_valid and closure_pass,
        "observed_max_change_normalized_leakage": 0.1,
        "median_null_max": 0.5,
        "observed_over_median_null": 0.2,
        "rows": [
            {"basin": basin, "point_count": 200, "change_normalized_leakage": 0.1}
            for basin in range(3)
        ],
    }
    return {
        "status": "eligible", "law_valid": True,
        "direct_closure_valid": closure_valid,
        "dense_secondary_closure_valid": closure_valid if arm == "dense" else None,
        "failure_reasons": [] if closure_valid else ["direct_closure_denominator_or_value_invalid"],
        "routing": {"family_valid": True},
        "H_block": {"law_identification": _law(metric)},
        "G_block": {"law_identification": _law(metric)},
        "H_global": {"law_identification": _law(metric), "positive_control_pass": h_global},
        "G_global_diagnostic": {"law_identification": _law(metric)},
        "closure": {"maximum": 0.1},
        "kink_guard": {
            "rows": kink_rows, "complete_seed_pass": all(kink),
        },
        "coordinate_null": {
            "H": {"median_assignment": 0.5, "observed_over_median": metric / 0.5},
            "G": {"median_assignment": 0.5, "observed_over_median": metric / 0.5},
        },
        "direct_active_code_cloud_closure": direct,
        "center_forecast_guards": [
            {
                "restricted_forecast": 0.1 if affine else 0.5,
                "k_induced_update": 0.1 if affine else 0.5,
            }
            for _ in range(3)
        ],
        "finite_radius_robustness": {"by_basin": radius},
        "per_seed_joint_h_g_pass": arm == "sparse" and joint and all(kink) and closure_valid,
        "per_seed_g_only_pass": arm == "sparse" and g and all(kink) and closure_valid,
        "per_seed_h_only_pass": arm == "sparse" and h and all(kink),
    }


def _rows(**sparse_options) -> list[dict]:
    rows = []
    for task_id, (arm, seed) in enumerate(
        [("sparse", seed) for seed in range(100, 110)]
        + [("dense", seed) for seed in range(100, 110)]
    ):
        result = _result(arm, **(sparse_options if arm == "sparse" else {}))
        rows.append({
            "task_id": task_id, "arm": arm, "seed": seed,
            "card_sha256": CARD_HASH, "task_tsv_sha256": TASK_HASH,
            "result": result,
            "provenance": {"selected_checkpoint_sha256": f"checkpoint-{task_id}"},
        })
    return rows


def _audit(rows, passed=True) -> dict:
    return {
        "status": "passed" if passed else "failed",
        "protocol_id": CARD["protocol_id"], "card_sha256": CARD_HASH,
        "task_tsv_sha256": TASK_HASH, "task_count": 20,
        "passed_count": 20 if passed else 19,
        "arm_counts": {"sparse": 10, "dense": 10},
        "parameter_counts_by_arm": {"sparse": {"total": 10}, "dense": {"total": 9}},
        "rows": [
            {
                "task_id": row["task_id"], "arm": row["arm"],
                "seed": row["seed"],
                "checkpoint_sha256": row["provenance"]["selected_checkpoint_sha256"],
            }
            for row in rows
        ],
    }


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, "finite_neighborhood_local_laws"),
        ({"finite": False}, "affine_fixed_point_local_laws"),
        ({"residual": False}, "affine_fixed_point_local_laws"),
        ({"affine": False}, "distinct_predictor_and_k_induced_jacobians_only"),
        ({"joint": False, "h": False}, "distinct_k_induced_update_jacobians_only"),
        ({"joint": False, "g": False}, "restricted_predictor_jacobians_mechanism_unresolved"),
        ({"joint": False, "h": False, "g": False}, "global_predictor_only"),
        ({"joint": False, "h": False, "g": False, "h_global": False}, "failed"),
    ],
)
def test_complete_mechanism_state_machine(options, expected):
    rows = _rows(**options)
    assert adjudicate(rows, CARD, _audit(rows))["mechanism_tier"] == expected


def test_invalid_checkpoint_audit_forces_invalid_tier():
    rows = _rows()
    assert adjudicate(rows, CARD, _audit(rows, passed=False))["mechanism_tier"] == "invalid"


def test_closure_invalid_sparse_row_remains_h_evaluable_but_not_g_or_joint():
    rows = _rows(joint=False, g=False)
    rows[0]["result"] = _result(
        "sparse", joint=False, g=False, h=True, closure_valid=False
    )
    decision = adjudicate(rows, CARD, _audit(rows))
    assert decision["mechanism_tier"] == "restricted_predictor_jacobians_mechanism_unresolved"
    assert decision["sparse_gates"]["H_seed_passes"] == 10
    assert decision["validity"]["sparse_direct_closure_evaluable"] == 9


def test_dense_secondary_closure_invalid_does_not_remove_h_specificity():
    rows = _rows()
    for row in rows[10:]:
        row["result"] = _result("dense", closure_valid=False)
    decision = adjudicate(rows, CARD, _audit(rows))
    assert decision["specificity"]["valid"] is True
    assert decision["specificity"]["passed"] is True


def test_relative_specificity_cannot_pass_when_sparse_h_recovery_fails():
    rows = _rows(joint=False, h=False, g=False, value=0.85)
    for row in rows[10:]:
        row["result"] = _result("dense", value=0.95)
    decision = adjudicate(rows, CARD, _audit(rows))
    assert decision["sparse_gates"]["H_only_reporting"] is False
    assert decision["specificity"]["valid"] is True
    assert decision["specificity"]["passed"] is False
    assert decision["relative_specificity_tier"] == "not_sparse_recipe_specific"


def test_kink_filter_prevents_disjoint_law_and_extension_rows():
    rows = _rows()
    for index, row in enumerate(rows[:10]):
        kink = (False, True, True) if index < 3 else (True, True, True)
        row["result"] = _result("sparse", kink=kink)
    decision = adjudicate(rows, CARD, _audit(rows))
    assert decision["sparse_gates"]["H_nearest_rows"] == 27
    assert decision["sparse_gates"]["kink_valid_rows_considered"] == 27
    assert decision["specificity"]["paired_seeds"] == list(range(103, 110))
    assert decision["specificity"]["valid"] is False


def test_frozen_sign_denominator_and_zero_tie_conventions():
    assert _exact_sign_p(9, 10) == pytest.approx(11 / 1024)
    assert _safe_ratio(0.0, 0.0) == 1.0
    assert _safe_ratio(1.0, 0.0) > 1.0
    assert not (0.0 < 0.0)
