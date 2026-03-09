"""Tests for the canonical paper benchmark manifest."""

from skae.benchmarks.paper_benchmark_manifest import (
    PAPER_BENCHMARK_BATCH_SIZE,
    PAPER_BENCHMARK_NUM_STEPS,
    PAPER_BENCHMARK_SEQUENCE_LENGTH,
    PAPER_BENCHMARK_TARGET_SIZE,
    get_paper_benchmark_model,
    paper_benchmark_manifest_jsonable,
    paper_benchmark_models,
    paper_benchmark_systems,
    resolve_system_default_dt,
)


def test_paper_benchmark_manifest_shape():
    """The paper benchmark should keep a fixed system/model matrix."""
    systems = paper_benchmark_systems()
    models = paper_benchmark_models()

    assert len(systems) == 29
    assert len(models) == 4
    assert PAPER_BENCHMARK_NUM_STEPS == 50_000
    assert PAPER_BENCHMARK_BATCH_SIZE == 256
    assert PAPER_BENCHMARK_TARGET_SIZE == 256
    assert PAPER_BENCHMARK_SEQUENCE_LENGTH == 8


def test_paper_benchmark_known_defaults():
    """Built-in defaults and model variants should be stable."""
    assert resolve_system_default_dt("duffing") == 0.01
    assert get_paper_benchmark_model("lista_diagonal").k_structure == "diagonal"


def test_paper_benchmark_manifest_jsonable():
    """Manifest JSON snapshot should expose resolved defaults for all systems."""
    payload = paper_benchmark_manifest_jsonable()

    assert payload["num_steps"] == 50_000
    assert len(payload["systems"]) == 29
    assert len(payload["models"]) == 4
    assert any(item["system_key"] == "dysts:LorenzCoupled" for item in payload["systems"])
