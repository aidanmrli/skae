"""Tests for transition-rich basin-partition task generation."""

from __future__ import annotations

import csv
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
        num_steps_override=None,
        dt_table=None,
        dt_column="requested_dt",
    )


def test_transition_rich_basin_partition_default_matrix():
    rows = _build_rows(_base_args())

    assert len(rows) == 17 * 46 * 3
    assert {row["model_variant"] for row in rows} == {
        "lista_dense_basin_partition",
        "lista_dense_projgap_trigger_basin_partition",
        "lista_dense_dyna_projgap_trigger_basin_partition",
        "lista_dense_softblock_hybrid_trigger_basin_partition",
        "lista_dense_softblock_basin_partition",
        "lista_dense_softblock_sparsegroup_basin_partition",
        "lista_dense_softblock_dict_tied_precode_basin_partition",
        "lista_dense_softblock_hybrid_precode_basin_partition",
        "lista_dense_softblock_strong_basin_partition",
        "lista_dense_softblock_signsplit_basin_partition",
        "lista_dense_softblock_signsplit_loops4_basin_partition",
        "lista_dense_softblock_signsplit_doubleblocks_basin_partition",
        "lista_dense_softblock_signsplit_p128_basin_partition",
        "lista_dense_softblock_signsplit_p64_basin_partition",
        "lista_dense_softblock_signsplit_p64_hardinit_basin_partition",
        "lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition",
        "lista_dense_softblock_signsplit_p64_softblock1em3_basin_partition",
        "lista_dense_softblock_signsplit_p64_momentum_basin_partition",
        "lista_dense_softblock_signsplit_linear_encoder_basin_partition",
        "lista_dense_softblock_signsplit_coherence_basin_partition",
        "lista_dense_softblock_signsplit_linear_encoder_coherence_basin_partition",
        "lista_blockdiag_basin_partition",
        "lista_blockdiag_loops4_basin_partition",
        "lista_blockdiag_double_basin_partition",
        "lista_blockdiag_top1_balance_basin_partition",
        "lista_blockdiag_sparsegroup_basin_partition",
        "lista_blockdiag_adaptive_groupwise_threshold_basin_partition",
        "lista_blockdiag_signsplit_basin_partition",
        "lista_blockdiag_signsplit_hardinit_basin_partition",
        "lista_blockdiag_signsplit_momentum_basin_partition",
        "lista_blockdiag_signsplit_loops4_basin_partition",
        "lista_blockdiag_signsplit_double_basin_partition",
        "lista_blockdiag_signsplit_linear_encoder_basin_partition",
        "lista_blockdiag_signsplit_coherence_basin_partition",
        "lista_blockdiag_signsplit_linear_encoder_coherence_basin_partition",
        "hyperlista_dense_basin_partition",
        "hyperlista_dense_dyna_projgap_trigger_basin_partition",
        "hyperlista_blockdiag_basin_partition",
        "hyperlista_blockdiag_hybrid_trigger_basin_partition",
        "hyperlista_blockdiag_sparsegroup_top2_basin_partition",
        "hyperlista_blockdiag_no_ss_basin_partition",
        "structured_lista_temporal_basin_partition",
        "structured_lista_entropy_temporal_basin_partition",
        "structured_lista_dominance_temporal_basin_partition",
        "mlp_sparse_basin_partition_control",
        "mlp_zero_sparse_basin_partition_control",
    }
    assert {row["target_size"] for row in rows} == {64, 128, 256}
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
    args.model_variants_csv = "lista_dense_basin_partition,mlp_sparse_basin_partition_control"
    args.seeds_csv = "0,2"

    payload = _manifest_payload(
        phase_label=args.phase_label,
        systems=_selected_system_specs(args),
        models=_selected_model_specs(args),
        seeds=[0, 2],
        task_count=8,
        eval_profile=args.eval_profile,
        num_steps=20_000,
    )

    assert payload["systems"] == ["gated_local_linear", "claude:transition_routes_4"]
    assert payload["models"] == [
        "lista_dense_basin_partition",
        "mlp_sparse_basin_partition_control",
    ]
    assert payload["selected_systems"][0]["basin_count"] == 3
    assert payload["selected_models"][1]["config_name"] == "generic_sparse"


def test_transition_rich_basin_partition_can_override_num_steps_globally():
    args = _base_args()
    args.systems_csv = "gated_local_linear"
    args.model_variants_csv = (
        "lista_dense_softblock_signsplit_p64_hardinit_basin_partition,"
        "mlp_zero_sparse_basin_partition_control"
    )
    args.seeds_csv = "0,1"
    args.num_steps_override = 200000

    rows = _build_rows(args)

    assert len(rows) == 4
    assert {row["num_steps"] for row in rows} == {200000}


def test_transition_rich_basin_partition_emits_block_loss_columns():
    args = _base_args()
    args.systems_csv = "gated_local_linear"
    args.model_variants_csv = "lista_blockdiag_top1_balance_basin_partition"
    args.seeds_csv = "0"

    rows = _build_rows(args)

    assert len(rows) == 1
    row = rows[0]
    assert row["block_loss"] == 1
    assert row["block_one_block_loss"] == "top1_margin"
    assert row["block_one_block_weight"] == 0.3
    assert row["block_top1_margin"] == 0.05
    assert row["block_balance_loss"] == "kl_uniform"
    assert row["block_balance_weight"] == 1.0


def test_transition_rich_basin_partition_emits_hyperlista_and_scaled_block_columns():
    args = _base_args()
    args.systems_csv = "gated_local_linear"
    args.model_variants_csv = (
        "hyperlista_blockdiag_no_ss_basin_partition,"
        "lista_blockdiag_double_basin_partition"
    )
    args.seeds_csv = "0"

    rows = _build_rows(args)

    assert len(rows) == 2
    by_variant = {row["model_variant"]: row for row in rows}

    hyper_row = by_variant["hyperlista_blockdiag_no_ss_basin_partition"]
    assert hyper_row["config_name"] == "hyperlista_parity_generic_sparse"
    assert hyper_row["hyperlista_use_ss"] == "false"
    assert hyper_row["hyperlista_use_momentum"] == "true"
    assert hyper_row["hyperlista_c_theta"] == 1e-2

    double_row = by_variant["lista_blockdiag_double_basin_partition"]
    assert double_row["k_num_blocks"] == 6


def test_transition_rich_basin_partition_emits_eval_reset_policy_columns():
    args = _base_args()
    args.systems_csv = "gated_local_linear"
    args.model_variants_csv = (
        "lista_dense_projgap_trigger_basin_partition,"
        "lista_dense_dyna_projgap_trigger_basin_partition"
    )
    args.seeds_csv = "0"

    rows = _build_rows(args)

    assert len(rows) == 2
    by_variant = {row["model_variant"]: row for row in rows}

    trigger_row = by_variant["lista_dense_projgap_trigger_basin_partition"]
    assert trigger_row["eval_use_dynamics_prior"] == "false"
    assert trigger_row["eval_event_trigger_proj_threshold"] == 0.05
    assert trigger_row["eval_event_trigger_min_dwell"] == 10
    assert trigger_row["eval_event_trigger_max_interval"] == 25

    dyna_row = by_variant["lista_dense_dyna_projgap_trigger_basin_partition"]
    assert dyna_row["eval_use_dynamics_prior"] == "true"
    assert dyna_row["eval_event_trigger_proj_threshold"] == 0.05


def test_transition_rich_basin_partition_emits_group_aware_encoder_columns():
    args = _base_args()
    args.systems_csv = "gated_local_linear"
    args.model_variants_csv = (
        "lista_blockdiag_sparsegroup_basin_partition,"
        "hyperlista_blockdiag_sparsegroup_top2_basin_partition,"
        "lista_dense_softblock_sparsegroup_basin_partition"
    )
    args.seeds_csv = "0"

    rows = _build_rows(args)

    assert len(rows) == 3
    by_variant = {row["model_variant"]: row for row in rows}

    lista_row = by_variant["lista_blockdiag_sparsegroup_basin_partition"]
    assert lista_row["encoder_group_shrinkage"] == "true"
    assert lista_row["encoder_group_threshold_scale"] == 1.0
    assert lista_row["encoder_topk_groups"] == ""
    assert lista_row["k_num_blocks"] == 3

    hyper_row = by_variant["hyperlista_blockdiag_sparsegroup_top2_basin_partition"]
    assert hyper_row["encoder_group_shrinkage"] == "true"
    assert hyper_row["encoder_group_threshold_scale"] == 1.0
    assert hyper_row["encoder_topk_groups"] == 2
    assert hyper_row["k_num_blocks"] == 3

    soft_row = by_variant["lista_dense_softblock_sparsegroup_basin_partition"]
    assert soft_row["encoder_group_shrinkage"] == "true"
    assert soft_row["soft_block_num_blocks"] == 3


def test_transition_rich_basin_partition_emits_lista_precode_and_threshold_columns():
    args = _base_args()
    args.systems_csv = "gated_local_linear"
    args.model_variants_csv = (
        "lista_blockdiag_adaptive_groupwise_threshold_basin_partition,"
        "lista_dense_softblock_dict_tied_precode_basin_partition,"
        "lista_dense_softblock_hybrid_precode_basin_partition"
    )
    args.seeds_csv = "0"

    rows = _build_rows(args)

    assert len(rows) == 3
    by_variant = {row["model_variant"]: row for row in rows}

    adaptive_row = by_variant["lista_blockdiag_adaptive_groupwise_threshold_basin_partition"]
    assert adaptive_row["lista_adaptive_thresholds"] == "true"
    assert adaptive_row["lista_alpha_residual_coeff"] == 0.25
    assert adaptive_row["lista_alpha_prior_coeff"] == 0.25
    assert adaptive_row["lista_groupwise_thresholds"] == "true"
    assert adaptive_row["eval_use_dynamics_prior"] == "true"
    assert adaptive_row["k_num_blocks"] == 3

    tied_row = by_variant["lista_dense_softblock_dict_tied_precode_basin_partition"]
    assert tied_row["lista_precode_mode"] == "dictionary_tied"
    assert tied_row["lista_precode_residual_scale"] == ""
    assert tied_row["soft_block_num_blocks"] == 3

    hybrid_row = by_variant["lista_dense_softblock_hybrid_precode_basin_partition"]
    assert hybrid_row["lista_precode_mode"] == "hybrid"
    assert hybrid_row["lista_precode_residual_scale"] == 0.1
    assert hybrid_row["soft_block_num_blocks"] == 3


def test_transition_rich_basin_partition_emits_soft_block_and_structured_columns():
    args = _base_args()
    args.systems_csv = "gated_local_linear,claude:cal_octagon_8"
    args.model_variants_csv = (
        "lista_dense_softblock_basin_partition,"
        "structured_lista_entropy_temporal_basin_partition"
    )
    args.seeds_csv = "0"

    rows = _build_rows(args)

    assert len(rows) == 4
    by_key = {(row["model_variant"], row["system_key"]): row for row in rows}

    soft_row = by_key[("lista_dense_softblock_basin_partition", "gated_local_linear")]
    assert soft_row["soft_block"] == 1
    assert soft_row["soft_block_num_blocks"] == 3
    assert soft_row["soft_block_weight"] == 1e-4
    assert soft_row["soft_block_norm"] == "l1"

    structured_row = by_key[("structured_lista_entropy_temporal_basin_partition", "claude:cal_octagon_8")]
    assert structured_row["structured"] == 1
    assert structured_row["structured_d_global"] == 16
    assert structured_row["structured_num_basins"] == 8
    assert structured_row["structured_d_basin"] == 30
    assert structured_row["lambda_entropy"] == 1e-2
    assert structured_row["lambda_temporal"] == 1e-2
    assert structured_row["excl_warmup_steps"] == 2000


def test_transition_rich_basin_partition_emits_signsplit_loop_and_target_sweep_columns():
    args = _base_args()
    args.systems_csv = "gated_local_linear"
    args.model_variants_csv = (
        "lista_blockdiag_signsplit_momentum_basin_partition,"
        "lista_blockdiag_signsplit_double_basin_partition,"
        "lista_dense_softblock_signsplit_loops4_basin_partition,"
        "lista_dense_softblock_signsplit_p64_basin_partition,"
        "lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition,"
        "lista_dense_softblock_signsplit_p64_softblock1em3_basin_partition,"
        "lista_dense_softblock_signsplit_p64_momentum_basin_partition"
    )
    args.seeds_csv = "0"

    rows = _build_rows(args)

    assert len(rows) == 7
    by_variant = {row["model_variant"]: row for row in rows}

    momentum_row = by_variant["lista_blockdiag_signsplit_momentum_basin_partition"]
    assert momentum_row["lista_final_op"] == "sign_split"
    assert momentum_row["lista_use_momentum"] == "true"
    assert momentum_row["lista_momentum_beta"] == 0.25
    assert momentum_row["k_num_blocks"] == 3

    blockdiag_row = by_variant["lista_blockdiag_signsplit_double_basin_partition"]
    assert blockdiag_row["lista_final_op"] == "sign_split"
    assert blockdiag_row["lista_num_loops"] == 2
    assert blockdiag_row["k_num_blocks"] == 6

    loops4_row = by_variant["lista_dense_softblock_signsplit_loops4_basin_partition"]
    assert loops4_row["lista_final_op"] == "sign_split"
    assert loops4_row["lista_num_loops"] == 4
    assert loops4_row["soft_block_num_blocks"] == 3

    p64_row = by_variant["lista_dense_softblock_signsplit_p64_basin_partition"]
    assert p64_row["target_size"] == 64
    assert p64_row["soft_block_num_blocks"] == 3

    p64_strong_row = by_variant["lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition"]
    assert p64_strong_row["target_size"] == 64
    assert p64_strong_row["soft_block_weight"] == 5e-4

    p64_stronger_row = by_variant["lista_dense_softblock_signsplit_p64_softblock1em3_basin_partition"]
    assert p64_stronger_row["target_size"] == 64
    assert p64_stronger_row["soft_block_weight"] == 1e-3

    p64_momentum_row = by_variant["lista_dense_softblock_signsplit_p64_momentum_basin_partition"]
    assert p64_momentum_row["target_size"] == 64
    assert p64_momentum_row["lista_use_momentum"] == "true"
    assert p64_momentum_row["lista_momentum_beta"] == 0.25


def test_transition_rich_basin_partition_emits_linear_encoder_and_coherence_columns():
    args = _base_args()
    args.systems_csv = "gated_local_linear"
    args.model_variants_csv = (
        "lista_blockdiag_signsplit_linear_encoder_basin_partition,"
        "lista_dense_softblock_signsplit_linear_encoder_coherence_basin_partition"
    )
    args.seeds_csv = "0"

    rows = _build_rows(args)

    assert len(rows) == 2
    by_variant = {row["model_variant"]: row for row in rows}

    linear_row = by_variant["lista_blockdiag_signsplit_linear_encoder_basin_partition"]
    assert linear_row["lista_linear_encoder"] == "true"
    assert linear_row["decoder_coherence_weight"] == ""
    assert linear_row["k_num_blocks"] == 3

    combo_row = by_variant["lista_dense_softblock_signsplit_linear_encoder_coherence_basin_partition"]
    assert combo_row["lista_linear_encoder"] == "true"
    assert combo_row["decoder_coherence_weight"] == 5e-4
    assert combo_row["soft_block_num_blocks"] == 3


def test_transition_rich_basin_partition_emits_hard_init_columns():
    args = _base_args()
    args.systems_csv = "gated_transfer_linear"
    args.model_variants_csv = (
        "lista_blockdiag_signsplit_hardinit_basin_partition,"
        "lista_dense_softblock_signsplit_p64_hardinit_basin_partition"
    )
    args.seeds_csv = "0"

    rows = _build_rows(args)

    assert len(rows) == 2
    by_variant = {row["model_variant"]: row for row in rows}

    block_row = by_variant["lista_blockdiag_signsplit_hardinit_basin_partition"]
    assert block_row["hard_init_oversample"] == "true"
    assert block_row["hard_init_fraction"] == 0.5
    assert block_row["hard_init_pool_size"] == 1024
    assert block_row["hard_init_num_candidates"] == 4096
    assert block_row["hard_init_probe_steps"] == 32
    assert block_row["hard_init_num_perturbations"] == 4
    assert block_row["hard_init_perturb_scale"] == 0.04
    assert block_row["hard_init_transient_window"] == 8
    assert block_row["hard_init_transient_weight"] == 0.5
    assert block_row["hard_init_jitter_scale"] == 0.25
    assert block_row["k_num_blocks"] == 3

    dense_row = by_variant["lista_dense_softblock_signsplit_p64_hardinit_basin_partition"]
    assert dense_row["hard_init_oversample"] == "true"
    assert dense_row["target_size"] == 64
    assert dense_row["soft_block_num_blocks"] == 3


def test_transition_rich_basin_partition_dt_table_filters_to_requested_arms(tmp_path):
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
                "model_variant": "lista_dense_basin_partition",
                "system_key": "gated_local_linear",
                "requested_dt": "0.02",
            }
        )
        writer.writerow(
            {
                "model_variant": "lista_blockdiag_basin_partition",
                "system_key": "claude:transition_routes_4",
                "requested_dt": "0.01",
            }
        )

    args = _base_args()
    args.seeds_csv = "0"
    args.dt_table = str(dt_table)

    rows = _build_rows(args)

    assert len(rows) == 2
    assert {(row["model_variant"], row["system_key"]) for row in rows} == {
        ("lista_dense_basin_partition", "gated_local_linear"),
        ("lista_blockdiag_basin_partition", "claude:transition_routes_4"),
    }
    assert {row["env_dt"] for row in rows} == {0.02, 0.01}
