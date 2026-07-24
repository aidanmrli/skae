"""Tests for the frozen global-K support-closure evidence packet."""

import json

import numpy as np
import pandas as pd

from experiments.neurips_2026.evidence.global_k_support_closure import (
    CARD,
    DECISION,
    GUARD_RUN_ROWS,
    GUARD_SOURCE_ROSTER,
    GUARD_SYSTEM_ROWS,
    PROVENANCE,
    RUN_ROWS,
    SYSTEM_ROWS,
    check,
    recompute_decision,
    recompute_system_rows,
    render_table,
    verify_full_packet,
    verify_packet,
)
from experiments.neurips_2026.evidence.global_k_support_closure_guard import (
    recompute_guard_system_rows,
)
from experiments.neurips_2026.evidence.global_k_support_closure_rendering import (
    PNG_DPI,
)


def test_packet_reduces_to_internally_frozen_partial_closure_decision():
    system_rows, decision, provenance = verify_packet()

    assert decision["decision"] == "partial_closure"
    assert decision["eligible_run_count"] == 45
    assert decision["eligible_system_count"] == 15
    assert decision["activity_leakage_system_wins"] == 15
    assert decision["activity_change_leakage_system_wins"] == 15
    assert decision["restricted_residual_system_wins"] == 15
    assert [name for name, passed in decision["checks"].items() if not passed] == [
        "operator_differentiation_guard"
    ]
    assert system_rows.shape[0] == 15
    assert provenance["result"]["dense_specificity_control"] == "pending"
    correction = provenance["result"]["executed_decision_string_correction"]
    assert "overstrong" in correction
    assert "not an invariant chart or invariant subspace" in correction


def test_compact_rows_independently_reproduce_system_rows_and_decision():
    run_rows = pd.read_csv(RUN_ROWS, float_precision="round_trip")
    expected_system_rows = pd.read_csv(
        SYSTEM_ROWS, float_precision="round_trip"
    )
    card = json.loads(CARD.read_text())
    expected_decision = json.loads(DECISION.read_text())

    actual_system_rows = recompute_system_rows(run_rows)
    assert actual_system_rows["system_key"].tolist() == expected_system_rows[
        "system_key"
    ].tolist()
    numeric = expected_system_rows.select_dtypes(include=np.number).columns
    np.testing.assert_allclose(
        actual_system_rows[numeric],
        expected_system_rows[numeric],
        rtol=1e-13,
        atol=1e-15,
    )
    assert recompute_decision(run_rows, actual_system_rows, card) == expected_decision


def test_scope_and_claim_boundaries_are_fail_closed():
    provenance = json.loads(PROVENANCE.read_text())
    protocol = provenance["protocol"]
    boundary = provenance["claim_boundary"]

    assert protocol["observed_system_dimension"] == 2
    assert protocol["latent_dimension"] == 256
    assert "current and next supports" in protocol["scope"]
    assert "z @ P @ K @ (I-P)" in protocol["row_vector_convention"]
    assert protocol["posthoc_restriction"].startswith("P @ K @ P")
    assert "sign-split ReLU" in boundary["sign_split_caveat"]
    exclusions = " ".join(boundary["not_supported"])
    assert "invariant subspaces" in exclusions
    assert "distinct local laws" in exclusions
    assert "multistep forecasting" in exclusions


def test_all_current_guard_is_recomputed_without_promoting_it_to_primary():
    _primary, _decision, provenance, rows, guard = verify_full_packet()

    assert guard["status"].startswith("post-hoc reduction")
    assert guard == provenance["all_current_guard"]
    assert guard["eligible_run_count"] == 45
    assert guard["eligible_system_count"] == 15
    assert guard["added_transition_count"] == 3527
    assert np.isclose(guard["added_transition_percentage_points"], 0.9567599826388888)
    medians = guard["system_medians"]
    assert np.isclose(medians["activity_leakage_true_over_null"], 0.022294791176291174)
    assert np.isclose(
        medians["activity_change_leakage_true_over_null"], 0.4333614756929085
    )
    assert np.isclose(medians["matrix_leakage_true_over_null"], 0.2399255482792354)
    assert np.isclose(
        medians["matrix_change_leakage_true_over_null"], 0.699185907524442
    )
    assert np.isclose(
        medians["restricted_inside_residual_true_over_null"],
        0.05325424019276353,
    )
    assert guard["system_wins"]["activity_leakage"] == 15
    assert guard["system_wins"]["activity_change_leakage"] == 15
    assert guard["system_wins"]["matrix_leakage"] == 15
    assert guard["system_wins"]["matrix_change_leakage"] == 15
    assert guard["system_wins"]["restricted_inside_residual"] == 15
    assert not guard["reference_checks_not_a_second_frozen_decision"][
        "operator_differentiation_guard"
    ]
    assert rows.shape[0] == 15

    table = render_table(_decision, guard).decode("utf-8")
    assert "Raw-$K$ matrix Frobenius leakage" in table
    assert "$(K-I)$ matrix Frobenius leakage" in table


def test_all_current_compact_rows_and_source_roster_are_self_consistent():
    run_rows = pd.read_csv(GUARD_RUN_ROWS, float_precision="round_trip")
    expected = pd.read_csv(GUARD_SYSTEM_ROWS, float_precision="round_trip")
    actual = recompute_guard_system_rows(run_rows)
    numeric = expected.select_dtypes(include=np.number).columns
    np.testing.assert_allclose(actual[numeric], expected[numeric], rtol=1e-13, atol=1e-15)
    roster = json.loads(GUARD_SOURCE_ROSTER.read_text())
    assert roster["shard_count"] == 45
    assert roster["portable_digest"] == (
        "cd3c708ba7314f91074bf5b4c84a93e5736c8a6930f2a3869d1970d49b8f9836"
    )
    assert "not a public preregistration" in roster["status"]


def test_generated_table_and_figures_are_authenticated_and_reproducible():
    assert PNG_DPI == 320
    check()
