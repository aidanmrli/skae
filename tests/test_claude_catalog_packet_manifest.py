"""Tests for the first Claude-catalog packet manifest."""

from skae.benchmarks.claude_catalog_packet_manifest import (
    CLAUDE_CATALOG_PACKET_BATCH_SIZE,
    CLAUDE_CATALOG_PACKET_NUM_STEPS,
    CLAUDE_CATALOG_PACKET_SEQUENCE_LENGTH,
    CLAUDE_CATALOG_PACKET_TARGET_SIZE,
    claude_catalog_packet_manifest_jsonable,
    claude_catalog_packet_models,
    claude_catalog_packet_systems,
    resolve_claude_catalog_packet_dt,
)


def test_claude_catalog_packet_manifest_shape():
    """The default Claude packet should keep a fixed 6x3 structure."""
    systems = claude_catalog_packet_systems()
    models = claude_catalog_packet_models()

    assert len(systems) == 6
    assert len(claude_catalog_packet_systems(include_second_wave=True)) == 9
    assert len(models) == 3
    assert CLAUDE_CATALOG_PACKET_NUM_STEPS == 200_000
    assert CLAUDE_CATALOG_PACKET_BATCH_SIZE == 256
    assert CLAUDE_CATALOG_PACKET_TARGET_SIZE == 256
    assert CLAUDE_CATALOG_PACKET_SEQUENCE_LENGTH == 8


def test_claude_catalog_packet_known_defaults():
    """The packet should resolve the underlying catalog dt correctly."""
    assert resolve_claude_catalog_packet_dt("claude:cal_triangle_3") == 0.03


def test_claude_catalog_packet_manifest_jsonable():
    """Manifest JSON snapshot should expose resolved defaults for all systems."""
    payload = claude_catalog_packet_manifest_jsonable()

    assert payload["num_steps"] == 200_000
    assert len(payload["systems"]) == 6
    assert len(payload["models"]) == 3
    assert any(item["system_key"] == "claude:transition_routes_4" for item in payload["systems"])
