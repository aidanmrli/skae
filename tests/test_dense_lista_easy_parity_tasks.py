"""Tests for the dense-LISTA easy-system parity stage-1 task builder."""

from __future__ import annotations

from argparse import Namespace

from tools.build_dense_lista_easy_parity_tasks import (
    DEFAULT_WEIGHT_DECAY,
    EASY_SYSTEM_KEYS,
    STAGE1_LR_PAIRS,
    STAGE1_NUM_STEPS,
    _build_rows,
)


def _base_args() -> Namespace:
    return Namespace(
        output_tsv="unused.tsv",
        output_manifest_json=None,
        phase_label="stage1",
        systems_csv=None,
        seeds_csv=None,
        num_steps_csv=None,
        lr_pairs_csv=None,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        eval_profile="full",
    )


def test_default_stage1_matrix_shape():
    rows, arm_specs = _build_rows(_base_args())

    assert len(arm_specs) == len(STAGE1_NUM_STEPS) * len(STAGE1_LR_PAIRS)
    assert len(rows) == len(EASY_SYSTEM_KEYS) * len(STAGE1_NUM_STEPS) * len(STAGE1_LR_PAIRS) * 3
    assert {row["system_key"] for row in rows} == set(EASY_SYSTEM_KEYS)
    assert {row["num_steps"] for row in rows} == set(STAGE1_NUM_STEPS)
    assert {row["lr"] for row in rows} == {pair[0] for pair in STAGE1_LR_PAIRS}
    assert {row["k_matrix_lr"] for row in rows} == {pair[1] for pair in STAGE1_LR_PAIRS}
    assert {row["weight_decay"] for row in rows} == {DEFAULT_WEIGHT_DECAY}


def test_custom_subset_builds_only_requested_tasks():
    args = _base_args()
    args.systems_csv = "duffing,blended"
    args.seeds_csv = "1"
    args.num_steps_csv = "100000"
    args.lr_pairs_csv = "1e-4:1e-5"

    rows, arm_specs = _build_rows(args)

    assert len(arm_specs) == 1
    assert len(rows) == 2
    assert {row["system_key"] for row in rows} == {"duffing", "blended"}
    assert {row["seed"] for row in rows} == {1}
    assert {row["num_steps"] for row in rows} == {100000}
