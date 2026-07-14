"""Tests for controlled-multibasin paper task generation."""

from __future__ import annotations

import csv
from argparse import Namespace

import pytest

from skae.benchmarks.paper_protocol import (
    CONTROLLED_MODEL_ROW_IDS,
    CONTROLLED_PAPER_PROTOCOL,
)
from tools.build_transition_rich_basin_partition_tasks import (
    _build_rows,
    _manifest_payload,
    _selected_model_specs,
    _selected_system_specs,
)


def _base_args() -> Namespace:
    return Namespace(
        output_tsv="unused.tsv",
        output_manifest_json=None,
        phase_label="transition_rich_basin_partition",
        systems_csv=None,
        model_variants_csv=None,
        seeds_csv=None,
        paper_protocol=True,
        eval_profile="full",
        num_steps_override=None,
        dt_table=None,
        dt_column="requested_dt",
    )


def test_default_matrix_is_exactly_the_reported_protocol():
    rows = _build_rows(_base_args())

    assert len(rows) == 15 * 6 * 15
    assert list(dict.fromkeys(row["system_key"] for row in rows)) == list(
        CONTROLLED_PAPER_PROTOCOL.system_keys
    )
    assert list(dict.fromkeys(row["model_variant"] for row in rows)) == list(
        CONTROLLED_MODEL_ROW_IDS
    )
    assert {row["seed"] for row in rows} == set(range(15))
    assert {row["num_steps"] for row in rows} == {200_000}
    assert {row["batch_size"] for row in rows} == {256}
    assert {row["target_size"] for row in rows} == {256}
    assert {row["sequence_length"] for row in rows} == {8}

    by_variant = {row["model_variant"]: row for row in rows}
    assert by_variant[CONTROLLED_MODEL_ROW_IDS[0]]["soft_block"] == 0
    block_rows = [
        row for row in rows if row["model_variant"] == CONTROLLED_MODEL_ROW_IDS[1]
    ]
    soft_rows = [
        row for row in rows if row["model_variant"] == CONTROLLED_MODEL_ROW_IDS[2]
    ]
    assert {row["k_num_blocks"] for row in block_rows} == {3, 4, 5, 6, 8}
    assert all(row["soft_block"] == 1 for row in soft_rows)
    assert {row["soft_block_num_blocks"] for row in soft_rows} == {3, 4, 5, 6, 8}
    dense = by_variant[CONTROLLED_MODEL_ROW_IDS[-1]]
    assert dense["config_name"] == "generic_no_shrink"
    assert dense["sparsity_coeff"] == 0.0


def test_subsets_and_seed_overrides_support_targeted_repairs():
    args = _base_args()
    args.systems_csv = "gated_local_linear,claude:cal_hexagon_6"
    args.model_variants_csv = (
        "lista_blockdiag_signsplit_hardinit_basin_partition,"
        "mlp_sparse_hardinit_basin_partition_control"
    )
    args.seeds_csv = "2,14"
    args.num_steps_override = 10

    rows = _build_rows(args)

    assert len(rows) == 8
    assert {row["seed"] for row in rows} == {2, 14}
    assert {row["num_steps"] for row in rows} == {10}
    block_rows = [
        row for row in rows if row["model_variant"].startswith("lista_blockdiag")
    ]
    assert {row["k_num_blocks"] for row in block_rows} == {3, 6}


def test_retired_systems_and_models_are_rejected():
    args = _base_args()
    args.systems_csv = "claude:checkerboard_potential"
    with pytest.raises(KeyError, match="Unknown controlled paper system"):
        _build_rows(args)

    args = _base_args()
    args.model_variants_csv = "lista_dense_basin_partition"
    with pytest.raises(KeyError, match="Unknown controlled paper model"):
        _build_rows(args)


def test_dt_table_can_repair_only_selected_arms(tmp_path):
    dt_table = tmp_path / "requested_dt.tsv"
    with dt_table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_variant", "system_key", "requested_dt"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_variant": CONTROLLED_MODEL_ROW_IDS[0],
                "system_key": "gated_local_linear",
                "requested_dt": "0.02",
            }
        )

    args = _base_args()
    args.seeds_csv = "0"
    args.dt_table = str(dt_table)
    rows = _build_rows(args)

    assert len(rows) == 1
    assert rows[0]["env_dt"] == 0.02


def test_manifest_payload_records_count_semantics():
    args = _base_args()
    args.systems_csv = "gated_local_linear"
    args.model_variants_csv = CONTROLLED_MODEL_ROW_IDS[1]
    payload = _manifest_payload(
        phase_label=args.phase_label,
        systems=_selected_system_specs(args),
        models=_selected_model_specs(args),
        seeds=[0],
        task_count=1,
        eval_profile=args.eval_profile,
        num_steps=200_000,
    )

    assert payload["protocol_id"] == CONTROLLED_PAPER_PROTOCOL.protocol_id
    assert payload["systems"] == ["gated_local_linear"]
    assert "no basin labels" in payload["count_semantics"]
