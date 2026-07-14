"""Contracts for the controlled multibasin paper manifest."""

from __future__ import annotations

import pytest

from experiments.neurips_2026.protocol import (
    CONTROLLED_MODEL_ROW_IDS,
    CONTROLLED_PAPER_PROTOCOL,
)
from experiments.neurips_2026.controlled import (
    TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE,
    TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS,
    TRANSITION_RICH_BASIN_PARTITION_SEEDS,
    TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH,
    TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE,
    get_transition_rich_basin_count,
    get_transition_rich_basin_partition_model,
    resolve_transition_rich_default_dt,
    transition_rich_basin_partition_manifest_jsonable,
    transition_rich_basin_partition_models,
    transition_rich_basin_partition_systems,
)


def test_manifest_is_exactly_the_frozen_paper_matrix():
    systems = transition_rich_basin_partition_systems()
    models = transition_rich_basin_partition_models()

    assert tuple(system.system_key for system in systems) == (
        CONTROLLED_PAPER_PROTOCOL.system_keys
    )
    assert tuple(model.variant for model in models) == CONTROLLED_MODEL_ROW_IDS
    assert TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS == 200_000
    assert TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE == 256
    assert TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE == 256
    assert TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH == 8
    assert tuple(TRANSITION_RICH_BASIN_PARTITION_SEEDS) == tuple(range(15))


def test_retained_recipes_preserve_reported_architecture_differences():
    dense_lista = get_transition_rich_basin_partition_model(
        "lista_dense_signsplit_p256_hardinit_basin_partition"
    )
    block_lista = get_transition_rich_basin_partition_model(
        "lista_blockdiag_signsplit_hardinit_basin_partition"
    )
    soft_lista = get_transition_rich_basin_partition_model(
        "lista_dense_softblock_signsplit_p256_hardinit_basin_partition"
    )
    dense_mlp = get_transition_rich_basin_partition_model(
        "mlp_zero_sparse_hardinit_basin_partition_control"
    )

    assert dense_lista.lista_num_loops == 2
    assert dense_lista.lista_final_op == "sign_split"
    assert dense_lista.target_size == 256
    assert block_lista.use_basin_count_for_blocks
    assert soft_lista.soft_block
    assert soft_lista.use_basin_count_for_soft_block_num_blocks
    assert dense_mlp.config_name == "generic_no_shrink"
    assert dense_mlp.sparsity_coeff == 0.0
    assert all(
        model.hard_init_num_candidates == 4096
        for model in transition_rich_basin_partition_models()
    )


def test_system_metadata_and_default_dt_remain_available_to_evaluators():
    assert get_transition_rich_basin_count("gated_local_linear") == 3
    assert resolve_transition_rich_default_dt("claude:cal_pentagon_5") == 0.03
    with pytest.raises(KeyError, match="Unknown controlled paper system"):
        get_transition_rich_basin_count("claude:checkerboard_potential")


def test_manifest_jsonable_has_no_exploratory_rows():
    payload = transition_rich_basin_partition_manifest_jsonable()

    assert payload["protocol_id"] == CONTROLLED_PAPER_PROTOCOL.protocol_id
    assert len(payload["systems"]) == 15
    assert len(payload["models"]) == 6
    assert "h1000_threshold" not in payload
    assert {item["system_key"] for item in payload["systems"]} == set(
        CONTROLLED_PAPER_PROTOCOL.system_keys
    )
