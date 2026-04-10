"""Tests for the fixed transition-rich basin-partition manifest."""

from __future__ import annotations

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE,
    TRANSITION_RICH_BASIN_PARTITION_H1000_THRESHOLD,
    TRANSITION_RICH_BASIN_PARTITION_MAX_HALVINGS,
    TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS,
    TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH,
    TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE,
    get_transition_rich_basin_count,
    get_transition_rich_basin_partition_model,
    resolve_transition_rich_default_dt,
    transition_rich_dt_halving_schedule,
    transition_rich_basin_partition_manifest_jsonable,
    transition_rich_basin_partition_models,
    transition_rich_basin_partition_systems,
)


def test_transition_rich_basin_partition_manifest_shape():
    systems = transition_rich_basin_partition_systems()
    models = transition_rich_basin_partition_models()

    assert len(systems) == 17
    assert len(models) == 46
    assert TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS == 20_000
    assert TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE == 256
    assert TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE == 256
    assert TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH == 8
    assert TRANSITION_RICH_BASIN_PARTITION_H1000_THRESHOLD == 50.0
    assert TRANSITION_RICH_BASIN_PARTITION_MAX_HALVINGS == 6


def test_transition_rich_basin_partition_known_defaults():
    assert get_transition_rich_basin_count("gated_local_linear") == 3
    assert resolve_transition_rich_default_dt("claude:cal_pentagon_5") == 0.03
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_basin_partition"
    ).use_basin_count_for_blocks
    assert get_transition_rich_basin_partition_model(
        "mlp_sparse_basin_partition_control"
    ).config_name == "generic_sparse"
    assert get_transition_rich_basin_partition_model(
        "mlp_zero_sparse_basin_partition_control"
    ).config_name == "generic_sparse"
    assert get_transition_rich_basin_partition_model(
        "mlp_zero_sparse_basin_partition_control"
    ).sparsity_coeff == 0.0
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_top1_balance_basin_partition"
    ).block_balance_loss == "kl_uniform"
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_sparsegroup_basin_partition"
    ).encoder_group_shrinkage is True
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_adaptive_groupwise_threshold_basin_partition"
    ).lista_adaptive_thresholds is True
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_adaptive_groupwise_threshold_basin_partition"
    ).lista_groupwise_thresholds is True
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_double_basin_partition"
    ).basin_count_block_multiplier == 2
    assert get_transition_rich_basin_partition_model(
        "hyperlista_blockdiag_no_ss_basin_partition"
    ).hyperlista_use_ss is False
    assert get_transition_rich_basin_partition_model(
        "lista_dense_dyna_projgap_trigger_basin_partition"
    ).eval_use_dynamics_prior is True
    assert get_transition_rich_basin_partition_model(
        "lista_dense_projgap_trigger_basin_partition"
    ).eval_event_trigger_proj_threshold == 0.05
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_basin_partition"
    ).soft_block_weight == 1e-4
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_sparsegroup_basin_partition"
    ).encoder_group_threshold_scale == 1.0
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_dict_tied_precode_basin_partition"
    ).lista_precode_mode == "dictionary_tied"
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_hybrid_precode_basin_partition"
    ).lista_precode_residual_scale == 0.1
    assert get_transition_rich_basin_partition_model(
        "hyperlista_blockdiag_sparsegroup_top2_basin_partition"
    ).encoder_topk_groups == 2
    assert get_transition_rich_basin_partition_model(
        "structured_lista_temporal_basin_partition"
    ).structured
    assert get_transition_rich_basin_partition_model(
        "structured_lista_entropy_temporal_basin_partition"
    ).lambda_entropy == 1e-2
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_signsplit_basin_partition"
    ).lista_final_op == "sign_split"
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_signsplit_hardinit_basin_partition"
    ).hard_init_oversample is True
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_signsplit_hardinit_basin_partition"
    ).hard_init_fraction == 0.5
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_signsplit_momentum_basin_partition"
    ).lista_use_momentum is True
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_signsplit_momentum_basin_partition"
    ).lista_momentum_beta == 0.25
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_signsplit_double_basin_partition"
    ).basin_count_block_multiplier == 2
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_signsplit_doubleblocks_basin_partition"
    ).soft_block_num_blocks_multiplier == 2
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_signsplit_p64_basin_partition"
    ).target_size == 64
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_signsplit_p64_hardinit_basin_partition"
    ).hard_init_num_candidates == 4096
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_signsplit_p64_softblock5em4_basin_partition"
    ).soft_block_weight == 5e-4
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_signsplit_p64_softblock1em3_basin_partition"
    ).soft_block_weight == 1e-3
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_signsplit_p64_momentum_basin_partition"
    ).lista_use_momentum is True
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_signsplit_linear_encoder_basin_partition"
    ).lista_linear_encoder is True
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_signsplit_coherence_basin_partition"
    ).decoder_coherence_weight == 5e-4
    assert get_transition_rich_basin_partition_model(
        "lista_dense_softblock_signsplit_linear_encoder_coherence_basin_partition"
    ).lista_linear_encoder is True
    assert transition_rich_dt_halving_schedule("gated_local_linear", max_halvings=2) == [0.04, 0.02, 0.01]


def test_transition_rich_basin_partition_manifest_jsonable():
    payload = transition_rich_basin_partition_manifest_jsonable()

    assert payload["num_steps"] == 20_000
    assert len(payload["systems"]) == 17
    assert len(payload["models"]) == 46
    assert payload["h1000_threshold"] == 50.0
    assert any(item["system_key"] == "gated_transfer_linear" for item in payload["systems"])
