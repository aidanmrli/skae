import json
from pathlib import Path

import pytest
import torch

from experiments.neurips_2026.global_k_distinct_laws_v2_supplemental_audit import (
    _adverse_specificity_guard,
    _canonical_bytes,
    _checkpoint_hashes_authenticate,
    _finite_radius_integrity,
    _finite_trajectory_counts,
    _load_supplemental_lock,
    _per_basin_counts,
    _reproduce_decision,
    _serialized_selector_score,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = (
    ROOT
    / "experiments/neurips_2026/global_k_distinct_laws_v2_supplemental_audit_card.json"
)
LOCK = (
    ROOT
    / "experiments/neurips_2026/global_k_distinct_laws_v2_supplemental_audit_lock.json"
)
V2_CARD = ROOT / "experiments/neurips_2026/global_k_distinct_laws_v2_card.json"


def test_finite_trajectory_counts_require_every_value() -> None:
    initial = torch.zeros(16, 2)
    truth = torch.zeros(200, 16, 2)
    prediction = torch.zeros_like(truth)
    complete = _finite_trajectory_counts(initial, truth, prediction)
    assert complete["joint_finite_count"] == 16

    prediction[57, 9, 1] = float("nan")
    incomplete = _finite_trajectory_counts(initial, truth, prediction)
    assert incomplete["prediction_finite_count"] == 15
    assert incomplete["joint_finite_count"] == 15
    assert incomplete["joint_finite_by_trajectory"][9] is False


def test_canonical_decision_bytes_are_exact_not_whitespace_tolerant() -> None:
    payload = {"b": [1, 2], "a": True}
    canonical = _canonical_bytes(payload)
    assert json.loads(canonical) == payload
    assert canonical != canonical.rstrip(b"\n")


def test_locked_evaluator_hash_mismatch_fails_before_adjudication(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="evaluator/source-lock mismatch"):
        _reproduce_decision(
            rows=[{"provenance": {"evaluator_sha256": "wrong"}}],
            v2_card={},
            audit_summary={},
            decision_path=tmp_path / "does_not_need_to_exist.json",
            card_hash="card",
            task_hash="task",
            source_lock_hash="lock",
            evaluator_hash="expected",
            summarizer_hash="summarizer",
        )


def _law() -> dict:
    return {
        "law_identification": {
            "cost_matrix": [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            "own_over_nearest_wrong_by_basin": [0.1, 0.1, 0.1],
        }
    }


def _finite_records(passed: bool) -> list[dict]:
    return [
        {
            "radius": radius,
            "autograd_agreement": 0.1 if passed else 0.5,
            "own_law_is_nearest": passed,
            "normalized_linear_fit_residual": 0.1 if passed else 0.5,
        }
        for radius in [0.01, 0.03, 0.06, 0.12, 0.18]
    ]


def _row(seed: int, basin_passes: list[bool]) -> dict:
    h, g = _law(), _law()
    h["law_identification"]["own_over_nearest_wrong_by_basin"] = [
        0.1 if passed else 1.0 for passed in basin_passes
    ]
    g["law_identification"]["own_over_nearest_wrong_by_basin"] = [
        0.1 if passed else 1.0 for passed in basin_passes
    ]
    return {
        "arm": "sparse",
        "seed": seed,
        "result": {
            "status": "eligible",
            "H_block": h,
            "G_block": g,
            "kink_guard": {
                "rows": [
                    {"passed_both_estimands_both_epsilons": True} for _ in range(3)
                ]
            },
            "direct_closure_valid": True,
            "direct_active_code_cloud_closure": {
                "rows": [
                    {"change_normalized_leakage": 0.1 if passed else 0.9}
                    for passed in basin_passes
                ]
            },
            "center_forecast_guards": [
                {
                    "restricted_forecast": 0.1 if passed else 0.5,
                    "k_induced_update": 0.1 if passed else 0.5,
                }
                for passed in basin_passes
            ],
            "finite_radius_robustness": {
                "by_basin": [
                    {
                        "basin": basin_index,
                        "H": _finite_records(passed),
                        "G": _finite_records(passed),
                    }
                    for basin_index, passed in enumerate(basin_passes)
                ]
            },
        },
    }


def test_per_basin_guard_rejects_aggregate_10_10_4() -> None:
    card = json.loads(V2_CARD.read_text())
    rows = [
        _row(seed, [True, True, seed < 4])
        for seed in range(10)
    ]
    result = _per_basin_counts(
        rows, card, "finite_neighborhood_local_laws", minimum=8
    )
    assert result["counts_by_basin"][2]["finite_every_gate"] == 4
    assert result["blanket_three_law_wording_permitted"] is False


def test_per_basin_guard_accepts_eight_replicates_each() -> None:
    card = json.loads(V2_CARD.read_text())
    rows = [
        _row(seed, [seed < 8, seed < 8, seed < 8])
        for seed in range(10)
    ]
    result = _per_basin_counts(
        rows, card, "finite_neighborhood_local_laws", minimum=8
    )
    assert [row["finite_every_gate"] for row in result["counts_by_basin"]] == [
        8,
        8,
        8,
    ]
    assert result["blanket_three_law_wording_permitted"] is True


def test_finite_radius_integrity_accepts_exact_paired_order() -> None:
    card = json.loads(V2_CARD.read_text())
    result = _finite_radius_integrity([_row(100, [True, True, True])], card)
    assert result["passed"] is True
    assert result["checked_H_G_basin_pairs"] == 3
    assert all(row["H_G_radius_index_alignment"] for row in result["rows"])


@pytest.mark.parametrize(
    ("case", "estimand", "radii"),
    [
        ("permuted", "G", [0.03, 0.01, 0.06, 0.12, 0.18]),
        ("wrong", "H", [0.01, 0.03, 0.06, 0.12, 0.19]),
        ("duplicate", "G", [0.01, 0.03, 0.06, 0.12, 0.12]),
        ("nan", "H", [0.01, 0.03, float("nan"), 0.12, 0.18]),
    ],
)
def test_finite_radius_integrity_rejects_adversarial_sequences(
    case: str, estimand: str, radii: list[float],
) -> None:
    card = json.loads(V2_CARD.read_text())
    row = _row(100, [True, True, True])
    records = row["result"]["finite_radius_robustness"]["by_basin"][1][
        estimand
    ]
    for record, radius in zip(records, radii):
        record["radius"] = radius
    result = _finite_radius_integrity([row], card)
    assert result["passed"] is False, case
    failed = [item for item in result["rows"] if not item["passed"]]
    assert len(failed) == 1
    assert failed[0]["basin_index"] == 1
    assert failed[0]["H_G_radius_index_alignment"] is False


def test_per_basin_finite_gate_rejects_radius_misalignment() -> None:
    card = json.loads(V2_CARD.read_text())
    rows = [_row(seed, [seed < 8, seed < 8, seed < 8]) for seed in range(10)]
    records = rows[0]["result"]["finite_radius_robustness"]["by_basin"][0]["G"]
    records[0]["radius"], records[1]["radius"] = (
        records[1]["radius"], records[0]["radius"]
    )
    result = _per_basin_counts(
        rows, card, "finite_neighborhood_local_laws", minimum=8
    )
    assert result["counts_by_basin"][0]["finite_every_gate"] == 7
    assert result["blanket_three_law_wording_permitted"] is False


def test_finite_radius_integrity_authenticates_parent_card_sequence() -> None:
    card = json.loads(V2_CARD.read_text())
    card["finite_radius_robustness_not_selection"]["radii"] = [
        0.01, 0.03, 0.06, 0.18, 0.12
    ]
    result = _finite_radius_integrity([_row(100, [True, True, True])], card)
    assert result["passed"] is False
    assert result["parent_card_radius_check"]["passed"] is False


def test_absent_historical_selector_score_is_explicitly_unavailable() -> None:
    value, source = _serialized_selector_score(
        {"step": 2000}, {"checkpoint_selector": {"metric": "final_error"}}
    )
    assert value is None
    assert source is None


def test_checkpoint_authentication_rejects_any_mutated_source() -> None:
    assert _checkpoint_hashes_authenticate("same", "same", "same")
    assert not _checkpoint_hashes_authenticate("mutated", "same", "same")
    assert not _checkpoint_hashes_authenticate("same", "mutated", "same")
    assert not _checkpoint_hashes_authenticate("same", "same", "mutated")


def _specificity_decision(ratios: list[float], wins: int) -> dict:
    return {
        "specificity": {
            "paired_seeds": list(range(100, 100 + len(ratios))),
            "ratios_by_seed": {
                "H_row": ratios,
                "H_assignment": ratios,
            },
            "H_sparse_better_both_count": wins,
            "missing_or_ineligible_pairs_counted_as_sign_failures": True,
            "passed": True,
        }
    }


def test_adverse_completion_closes_odd_sample_median_loophole() -> None:
    card = json.loads(V2_CARD.read_text())
    observed = [0.1, 0.1, 0.1, 0.1, 0.89, 0.99, 0.99, 0.99, 0.99]
    assert float(torch.tensor(observed).median()) == pytest.approx(0.89)
    guard = _adverse_specificity_guard(
        _specificity_decision(observed, wins=9), card
    )
    assert guard["H_row"]["adverse_fixed_ten_median"] == pytest.approx(0.94)
    assert guard["adverse_completion_passed"] is False
    assert guard["positive_relative_specificity_claim_permitted"] is False


def test_adverse_completion_can_pass_with_nine_uniformly_strong_pairs() -> None:
    card = json.loads(V2_CARD.read_text())
    guard = _adverse_specificity_guard(
        _specificity_decision([0.1] * 9, wins=9), card
    )
    assert guard["H_row"]["adverse_fixed_ten_median"] == pytest.approx(0.1)
    assert guard["adverse_completion_passed"] is True


def test_protocol_documents_conservative_ineligible_limitation() -> None:
    card = json.loads(CARD.read_text())
    limitation = card["limitations"]["conservative_ineligible_numerical_mismatch"]
    assert "cannot create one" in limitation
    assert "not automatically conservative for median-based relative specificity" in limitation
    assert "does not repair" in limitation


def test_supplemental_lock_authenticates_every_execution_source() -> None:
    lock, card = _load_supplemental_lock(LOCK, CARD)
    assert lock["protocol_id"] == card["protocol_id"]
    assert len(lock["supplemental_sources"]) == 5
