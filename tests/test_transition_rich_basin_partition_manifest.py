"""Tests for the fixed transition-rich basin-partition manifest."""

from __future__ import annotations

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE,
    TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS,
    TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH,
    TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE,
    get_transition_rich_basin_count,
    get_transition_rich_basin_partition_model,
    resolve_transition_rich_default_dt,
    transition_rich_basin_partition_manifest_jsonable,
    transition_rich_basin_partition_models,
    transition_rich_basin_partition_systems,
)


def test_transition_rich_basin_partition_manifest_shape():
    systems = transition_rich_basin_partition_systems()
    models = transition_rich_basin_partition_models()

    assert len(systems) == 17
    assert len(models) == 2
    assert TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS == 200_000
    assert TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE == 256
    assert TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE == 256
    assert TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH == 8


def test_transition_rich_basin_partition_known_defaults():
    assert get_transition_rich_basin_count("gated_local_linear") == 3
    assert resolve_transition_rich_default_dt("claude:cal_pentagon_5") == 0.03
    assert get_transition_rich_basin_partition_model(
        "lista_blockdiag_basin_partition"
    ).use_basin_count_for_blocks


def test_transition_rich_basin_partition_manifest_jsonable():
    payload = transition_rich_basin_partition_manifest_jsonable()

    assert payload["num_steps"] == 200_000
    assert len(payload["systems"]) == 17
    assert len(payload["models"]) == 2
    assert any(item["system_key"] == "gated_transfer_linear" for item in payload["systems"])
