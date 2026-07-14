"""Contract tests for the frozen paper-facing benchmark protocol."""

from experiments.neurips_2026.protocol import (
    CONTROLLED_MODEL_ROW_IDS,
    CONTROLLED_PAPER_PROTOCOL,
    DYSTS_MODEL_ROW_IDS,
    DYSTS_PAPER_PROTOCOL,
    DYSTS_PAPER_ROW_OVERRIDES,
    PAPER_MODEL_ROWS,
    PAPER_SEEDS,
)


def test_paper_protocol_matrix_is_frozen():
    assert len(CONTROLLED_PAPER_PROTOCOL.system_keys) == 15
    assert len(DYSTS_PAPER_PROTOCOL.system_keys) == 10
    assert PAPER_SEEDS == tuple(range(15))
    assert CONTROLLED_PAPER_PROTOCOL.seeds == PAPER_SEEDS
    assert DYSTS_PAPER_PROTOCOL.seeds == PAPER_SEEDS

    assert CONTROLLED_PAPER_PROTOCOL.num_steps == 200_000
    assert CONTROLLED_PAPER_PROTOCOL.sequence_length == 8
    assert DYSTS_PAPER_PROTOCOL.num_steps == 100_000
    assert DYSTS_PAPER_PROTOCOL.sequence_length == 10
    assert DYSTS_PAPER_PROTOCOL.dt_multiplier == 30.0
    assert DYSTS_PAPER_ROW_OVERRIDES[0].variant == "lista_sb"
    assert DYSTS_PAPER_ROW_OVERRIDES[0].lista_num_loops == 2
    assert DYSTS_PAPER_ROW_OVERRIDES[0].lista_final_op == "sign_split"
    assert DYSTS_PAPER_ROW_OVERRIDES[0].source_campaign_system_count == 12
    assert DYSTS_PAPER_ROW_OVERRIDES[0].retained_paper_system_count == 10


def test_six_display_rows_map_one_to_one_across_benchmarks():
    assert len(PAPER_MODEL_ROWS) == 6
    assert len(set(CONTROLLED_MODEL_ROW_IDS)) == 6
    assert len(set(DYSTS_MODEL_ROW_IDS)) == 6
    assert CONTROLLED_MODEL_ROW_IDS[-1] == (
        "mlp_zero_sparse_hardinit_basin_partition_control"
    )
    assert DYSTS_MODEL_ROW_IDS[-1] == "dense_mlp_tanh"
