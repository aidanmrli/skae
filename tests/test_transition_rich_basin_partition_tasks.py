"""Tests for transition-rich basin-partition task generation."""

from __future__ import annotations

from argparse import Namespace

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
        eval_profile="full",
    )


def test_transition_rich_basin_partition_default_matrix():
    rows = _build_rows(_base_args())

    assert len(rows) == 17 * 2 * 3
    assert {row["model_variant"] for row in rows} == {
        "lista_dense_basin_partition",
        "lista_blockdiag_basin_partition",
    }
    assert {row["target_size"] for row in rows} == {256}
    assert {row["sequence_length"] for row in rows} == {8}


def test_transition_rich_basin_partition_blockdiag_uses_system_basin_count():
    args = _base_args()
    args.systems_csv = "gated_local_linear,claude:cal_hexagon_6"
    args.model_variants_csv = "lista_blockdiag_basin_partition"
    args.seeds_csv = "1"

    rows = _build_rows(args)

    assert len(rows) == 2
    by_system = {row["system_key"]: row for row in rows}
    assert by_system["gated_local_linear"]["k_num_blocks"] == 3
    assert by_system["claude:cal_hexagon_6"]["k_num_blocks"] == 6


def test_transition_rich_basin_partition_manifest_payload_tracks_metadata():
    args = _base_args()
    args.systems_csv = "gated_local_linear,claude:transition_routes_4"
    args.model_variants_csv = "lista_dense_basin_partition,lista_blockdiag_basin_partition"
    args.seeds_csv = "0,2"

    payload = _manifest_payload(
        phase_label=args.phase_label,
        systems=_selected_system_specs(args),
        models=_selected_model_specs(args),
        seeds=[0, 2],
        task_count=8,
        eval_profile=args.eval_profile,
    )

    assert payload["systems"] == ["gated_local_linear", "claude:transition_routes_4"]
    assert payload["models"] == [
        "lista_dense_basin_partition",
        "lista_blockdiag_basin_partition",
    ]
    assert payload["selected_systems"][0]["basin_count"] == 3
    assert payload["selected_models"][1]["use_basin_count_for_blocks"]
