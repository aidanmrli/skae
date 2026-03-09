"""Tests for the dense-LISTA easy-system parity Stage-2 task builder."""

from __future__ import annotations

from argparse import Namespace

from tools.build_dense_lista_easy_parity_stage2_tasks import (
    DEFAULT_BASE_ARM_SPECS,
    DEFAULT_PRED_COEFFS,
    DEFAULT_RECONST_COEFFS,
    DEFAULT_SPARSITY_COEFFS,
    HOLDOUT_SYSTEM_KEYS,
    _build_rows,
)


def _base_args() -> Namespace:
    return Namespace(
        output_tsv="unused.tsv",
        output_manifest_json=None,
        phase_label="stage2",
        systems_csv=None,
        seeds_csv=None,
        base_arms_csv=None,
        sparsity_coeffs_csv=None,
        reconst_coeffs_csv=None,
        pred_coeffs_csv=None,
        eval_profile="full",
    )


def test_default_stage2_matrix_shape():
    rows, arm_specs = _build_rows(_base_args())

    # Baseline plus two off-baseline values per coefficient family.
    unique_coeff_variants_per_base = 1 + 2 + 2 + 2
    assert len(arm_specs) == len(DEFAULT_BASE_ARM_SPECS) * unique_coeff_variants_per_base
    assert len(rows) == len(HOLDOUT_SYSTEM_KEYS) * len(DEFAULT_BASE_ARM_SPECS) * unique_coeff_variants_per_base * 3
    assert {row["system_key"] for row in rows} == set(HOLDOUT_SYSTEM_KEYS)
    assert {row["num_steps"] for row in rows} == {100000, 200000}
    assert {row["lr"] for row in rows} == {5e-5}
    assert {row["k_matrix_lr"] for row in rows} == {5e-6}
    assert {row["weight_decay"] for row in rows} == {1e-4}
    assert {row["sparsity_coeff"] for row in rows} == set(DEFAULT_SPARSITY_COEFFS)
    assert {row["reconst_coeff"] for row in rows} == set(DEFAULT_RECONST_COEFFS)
    assert {row["pred_coeff"] for row in rows} == set(DEFAULT_PRED_COEFFS)


def test_custom_stage2_subset_builds_only_requested_tasks():
    args = _base_args()
    args.systems_csv = "duffing"
    args.seeds_csv = "2"
    args.base_arms_csv = "100000:5e-5:5e-6:1e-4"
    args.sparsity_coeffs_csv = "0.006"
    args.reconst_coeffs_csv = "0.03"
    args.pred_coeffs_csv = "1.0,2.0"

    rows, arm_specs = _build_rows(args)

    # Baseline + one off-baseline pred variant.
    assert len(arm_specs) == 2
    assert len(rows) == 2
    assert {row["system_key"] for row in rows} == {"duffing"}
    assert {row["seed"] for row in rows} == {2}
    assert {row["num_steps"] for row in rows} == {100000}
    assert {row["pred_coeff"] for row in rows} == {1.0, 2.0}
