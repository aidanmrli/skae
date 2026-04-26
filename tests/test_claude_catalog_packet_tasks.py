"""Tests for Claude-catalog packet task generation."""

from __future__ import annotations

from argparse import Namespace

from tools.build_claude_catalog_packet_tasks import _build_rows, _manifest_payload, _selected_model_specs, _selected_system_specs


def _base_args() -> Namespace:
    return Namespace(
        output_tsv="unused.tsv",
        output_manifest_json=None,
        phase_label="claude_catalog_packet",
        systems_csv=None,
        model_variants_csv=None,
        seeds_csv=None,
        include_second_wave=False,
        eval_profile="full",
    )


def test_claude_catalog_packet_default_matrix():
    """The default Claude packet should build a 6 systems x 3 models x 3 seeds matrix."""
    rows = _build_rows(_base_args())

    assert len(rows) == 6 * 3 * 3
    assert {row["model_variant"] for row in rows} == {
        "generic_sparse_ns20k_best",
        "generic_sparse_sc0_ns20k_best",
        "lista_dense_promoted_stage4",
    }
    assert {row["system_group"] for row in rows} == {"strict_core"}
    assert {row["sequence_length"] for row in rows} == {8}
    assert {row["target_size"] for row in rows} == {256}


def test_claude_catalog_packet_custom_subset_uses_requested_systems_and_models():
    """System/model overrides should narrow the task table without changing recipe fields."""
    args = _base_args()
    args.systems_csv = "claude:cal_triangle_3,claude:transition_routes_4"
    args.model_variants_csv = "generic_sparse_sc0_ns20k_best"
    args.seeds_csv = "1"

    rows = _build_rows(args)

    assert len(rows) == 2
    assert {row["system_key"] for row in rows} == {
        "claude:cal_triangle_3",
        "claude:transition_routes_4",
    }
    assert {row["model_variant"] for row in rows} == {"generic_sparse_sc0_ns20k_best"}
    assert {row["seed"] for row in rows} == {1}
    assert {row["sparsity_coeff"] for row in rows} == {0.0}
    assert {row["env_dt"] for row in rows} == {0.03}


def test_claude_catalog_packet_manifest_payload_tracks_selected_metadata():
    """Manifest payload should preserve descriptive metadata for the queued subset."""
    args = _base_args()
    args.systems_csv = "claude:cal_triangle_3,claude:transition_routes_4"
    args.model_variants_csv = "generic_sparse_ns20k_best,lista_dense_promoted_stage4"
    args.seeds_csv = "0,2"

    payload = _manifest_payload(
        phase_label=args.phase_label,
        systems=_selected_system_specs(args),
        models=_selected_model_specs(args),
        seeds=[0, 2],
        task_count=8,
        include_second_wave=args.include_second_wave,
        eval_profile=args.eval_profile,
    )

    assert payload["systems"] == [
        "claude:cal_triangle_3",
        "claude:transition_routes_4",
    ]
    assert payload["models"] == [
        "generic_sparse_ns20k_best",
        "lista_dense_promoted_stage4",
    ]
    assert payload["packet_recipe"]["sequence_length"] == 8
    assert payload["selected_systems"][0]["paper_role"] == "minimal symmetric control"
    assert payload["selected_systems"][1]["resolved_default_dt"] == 0.03
    assert payload["selected_models"][1]["k_structure"] == "dense"
