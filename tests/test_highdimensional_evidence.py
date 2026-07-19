"""Tests for the independent high-dimensional evidence packet."""

import json

import numpy as np
import pandas as pd

from experiments.neurips_2026.evidence.allen_cahn_contingency import (
    DEEP_THRESHOLD,
    ordered_contingency,
)
from experiments.neurips_2026.evidence.highdimensional import (
    ALLEN_FIXED_RECORDS,
    ALLEN_SUPPORT,
    TEMPORAL_PROVENANCE,
    verify_compact_inputs,
    verify_declared_provenance,
)


def test_active_highdimensional_provenance_hashes_resolve():
    verify_compact_inputs()
    verify_declared_provenance()


def test_predeclared_deep_interior_supports_are_exactly_aligned():
    records = pd.read_csv(ALLEN_FIXED_RECORDS)
    sparse = records.loc[records["model"] == "sparse"]
    deep = sparse.loc[sparse["majority_fraction"] >= DEEP_THRESHOLD]

    matrix, basin_ids, family_ids = ordered_contingency(deep)

    assert deep.shape[0] == 130
    assert basin_ids == [0, 1, 2, 3]
    assert len(family_ids) == 4
    np.testing.assert_allclose(matrix, np.eye(4))


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
