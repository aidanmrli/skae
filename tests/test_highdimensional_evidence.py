"""Tests for the independent high-dimensional evidence packet."""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.neurips_2026.evidence.allen_cahn_contingency import (
    DEEP_THRESHOLD,
    fate_row_labels,
    outcome_independent_family_order,
    ordered_contingency,
)
from experiments.neurips_2026.evidence.highdimensional import (
    ALLEN_FIXED_RECORDS,
    ALLEN_SUPPORT,
    COMBINED_PROVENANCE,
    LORENZ_PROVENANCE,
    PAPER_EVIDENCE_DIR,
    TEMPORAL_PROVENANCE,
    combined_provenance,
    resolve_child_evidence_path,
    sha256,
    verify_compact_inputs,
    verify_declared_provenance,
)


def test_active_highdimensional_provenance_hashes_resolve():
    verify_compact_inputs()
    verify_declared_provenance()


def test_temporal_child_provenance_checks_data_and_parent_display_records():
    provenance = json.loads(TEMPORAL_PROVENANCE.read_text())
    expected_displays = {
        "main_display_pdf",
        "main_display_png",
        "support_contingency_pdf",
        "support_contingency_png",
        "snapshot_pdf",
    }

    assert expected_displays.issubset(provenance["evidence"])
    for record in provenance["evidence"].values():
        path = resolve_child_evidence_path(TEMPORAL_PROVENANCE, record["path"])
        assert path.is_file()
        assert sha256(path) == record["sha256"]


def test_temporal_child_provenance_rejects_packet_escape():
    with pytest.raises(ValueError, match="escapes the paper packet"):
        resolve_child_evidence_path(TEMPORAL_PROVENANCE, "../../../outside")


def test_active_highdimensional_provenance_is_derived_from_current_children():
    declared = json.loads(COMBINED_PROVENANCE.read_text())

    assert declared == combined_provenance(PAPER_EVIDENCE_DIR)


def test_lorenz96_confirmation_is_fail_closed_as_direct_repeated_k_rollout():
    provenance = json.loads(LORENZ_PROVENANCE.read_text())
    protocol = provenance["protocol"]

    assert protocol["rollout_mode"] == "no_reencode_repeated_K"
    assert protocol["rollout_implementation"].endswith(
        "GenericKoopmanModel.rollout_observation_discrete"
    )
    assert len(protocol["model_source_sha256_at_training_commit"]) == 64
    assert len(protocol["evaluator_source_sha256_at_training_commit"]) == 64


def test_predeclared_deep_interior_supports_are_exactly_aligned():
    records = pd.read_csv(ALLEN_FIXED_RECORDS)
    sparse = records.loc[records["model"] == "sparse"]
    deep = sparse.loc[sparse["majority_fraction"] >= DEEP_THRESHOLD]
    shared_order = outcome_independent_family_order(sparse)

    matrix, basin_ids, family_ids = ordered_contingency(
        deep, family_order=shared_order
    )

    assert deep.shape[0] == 130
    assert basin_ids == [0, 1, 2, 3]
    assert family_ids == list(range(9)) + [-1]
    np.testing.assert_allclose(
        matrix[:, :4],
        np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(matrix[:, 4:], 0.0)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)


def test_contingency_row_labels_report_panel_specific_final_state_counts():
    records = pd.read_csv(ALLEN_FIXED_RECORDS)
    sparse = records.loc[records["model"] == "sparse"]
    selected = sparse.loc[sparse["majority_fraction"] >= DEEP_THRESHOLD]

    assert fate_row_labels(sparse, [0, 1, 2, 3]) == [
        "fate 1 (n=74)",
        "fate 2 (n=55)",
        "fate 3 (n=65)",
        "fate 4 (n=62)",
    ]
    assert sum("n=" in label for label in fate_row_labels(selected, [0, 1, 2, 3])) == 4


def test_contingencies_share_raw_outcome_independent_family_identity():
    all_records = pd.DataFrame(
        {
            "global_basin_label": [0, 0, 1, 1, 0],
            "transferred_family": [7, 7, 2, 2, -1],
        }
    )
    selected = all_records.iloc[[0, 2]].copy()
    shared_order = outcome_independent_family_order(all_records)

    all_matrix, all_basins, all_ids = ordered_contingency(
        all_records, family_order=shared_order
    )
    selected_matrix, selected_basins, selected_ids = ordered_contingency(
        selected, family_order=shared_order
    )

    assert shared_order == [2, 7, -1]
    assert all_ids == selected_ids == shared_order
    assert all_basins == selected_basins == [0, 1]
    np.testing.assert_allclose(selected_matrix, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert not np.array_equal(selected_matrix[:, :2], np.eye(2))


def test_deep_interior_alignment_and_dense_contract_hold_for_all_ten_seeds():
    rows = pd.read_csv(ALLEN_SUPPORT)
    deep = rows.loc[
        (rows["model"] == "sparse")
        & (rows["scope"] == "final")
        & (rows["slice"] == "deep_test")
    ]

    assert deep["seed"].tolist() == list(range(21, 31))
    assert set(deep["num_trajectories"]) == {130}
    assert set(deep["num_families"]) == {4}
    np.testing.assert_allclose(
        deep[
            [
                "normalized_h_basin_given_family",
                "normalized_h_family_given_basin",
            ]
        ],
        0.0,
    )
    np.testing.assert_allclose(deep[["purity", "nmi", "ari"]], 1.0)

    provenance = json.loads(TEMPORAL_PROVENANCE.read_text())
    dense = provenance["dense_no_sparsity_contract"]
    assert dense["audits_passed"] == 10
    assert dense["hidden_activation"] == "tanh"
    assert dense["forbidden_modules_present"] == 0
    assert dense["sparsity_coefficient"] == 0
    assert dense["temporal_group_sparsity_coefficient"] == 0
    assert dense["weight_decay"] == 0
    assert dense["holdout_active_density_at_1e-4"] > 0.99


def test_fixed_seed_all_trajectory_contingency_retains_primary_fragmentation():
    records = pd.read_csv(ALLEN_FIXED_RECORDS)
    sparse = records.loc[records["model"] == "sparse"]

    matrix, _basin_ids, family_ids = ordered_contingency(sparse)

    assert sum(family >= 0 for family in family_ids) == 9
    assert -1 in family_ids
    assert np.min(matrix.max(axis=1)) >= 0.83
