"""Focused provenance tests for the staged Allen--Cahn v4 release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.neurips_2026.allen_cahn_support_subspaces.io import (
    sha256_path,
    verify_path,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.summarize_gpu_telemetry import (
    scope_timing_checks,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.validate_canary import (
    LINEAGE_CHECK_KEYS,
    TELEMETRY_CHECK_KEYS,
    validate_release_receipt,
)


def _receipt() -> dict:
    return {
        "status": "passed",
        "mechanism_metrics_deserialized": False,
        "mechanism_metrics_used_for_release": False,
        "seed": 64,
        "card_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "profile_decision_sha256": "c" * 64,
        "checks": {
            "lineage": dict.fromkeys(LINEAGE_CHECK_KEYS, True),
            "telemetry": dict.fromkeys(TELEMETRY_CHECK_KEYS, True),
        },
    }


def test_release_receipt_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = _receipt()
    path.write_text(json.dumps(payload))
    assert validate_release_receipt(
        path, card_hash="a" * 64, source_hash="b" * 64, profile_hash="c" * 64
    )["status"] == "passed"
    payload["mechanism_metrics_deserialized"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="release receipt"):
        validate_release_receipt(
            path, card_hash="a" * 64, source_hash="b" * 64,
            profile_hash="c" * 64,
        )


@pytest.mark.parametrize("group", ["lineage", "telemetry"])
def test_release_receipt_rejects_empty_or_wrong_check_roster(
    tmp_path: Path, group: str
) -> None:
    path = tmp_path / "receipt.json"
    for invalid in ({}, {"unexpected": True}):
        payload = _receipt()
        payload["checks"][group] = invalid
        path.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="release receipt"):
            validate_release_receipt(
                path, card_hash="a" * 64, source_hash="b" * 64,
                profile_hash="c" * 64,
            )


def test_canary_never_deserializes_scientific_shard_and_queue_is_staged() -> None:
    validator = Path(
        "experiments/neurips_2026/allen_cahn_support_subspaces/validate_canary.py"
    ).read_text()
    assert "json.loads(shard_path" not in validator
    assert "sha256_path(shard_path)" in validator
    queue = Path(
        "scripts/neurips_2026/allen_cahn_support_subspaces/queue_science.sh"
    ).read_text()
    assert queue.index("--array=0") < queue.index("CANARY_CHECK_JOB")
    assert queue.index("CANARY_CHECK_JOB") < queue.index("--array=1-9%8")
    assert '--dependency=afterok:"${CANARY_CHECK_JOB}"' in queue


def test_pinned_input_hash_check_rejects_byte_drift(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"frozen")
    digest = sha256_path(path)
    verify_path(path, digest)
    path.write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_path(path, digest)


def test_v4_gpu_scope_excludes_boundaries_and_rejects_slow_sampling() -> None:
    start = {"event": "gpu_compute_start", "unix_time": 100.0, "seed": 64}
    done = {"event": "gpu_compute_done", "unix_time": 104.0, "seed": 64}
    valid, timing = scope_timing_checks(
        start, done, [101.0, 102.0, 103.0], seed=64,
        maximum_median_interval=1.5,
    )
    assert all(valid.values())
    assert timing["median_interval_seconds"] == 1.0
    boundary, _ = scope_timing_checks(
        start, done, [100.0, 101.0, 102.0], seed=64,
        maximum_median_interval=1.5,
    )
    assert boundary["samples_within_gpu_interval"] is False
    slow, _ = scope_timing_checks(
        start, done, [100.5, 102.5, 103.9], seed=64,
        maximum_median_interval=1.5,
    )
    assert slow["sampling_interval"] is False


def test_v4_worker_and_evaluator_use_evaluator_owned_gpu_markers() -> None:
    evaluator = Path(
        "experiments/neurips_2026/allen_cahn_support_subspaces/evaluate.py"
    ).read_text()
    worker = Path(
        "scripts/neurips_2026/allen_cahn_support_subspaces/run_array.sh"
    ).read_text()
    assert evaluator.index('"gpu_compute_start"') < evaluator.index(
        "BEGIN FROZEN V3 SCIENTIFIC COMPUTATION"
    )
    assert evaluator.index("BEGIN FROZEN V3 SCIENTIFIC COMPUTATION") < evaluator.index(
        "historical_forecasts ="
    ) < evaluator.index("END FROZEN V3 SCIENTIFIC COMPUTATION")
    assert evaluator.index("END FROZEN V3 SCIENTIFIC COMPUTATION") < evaluator.index(
        '"gpu_compute_done"'
    ) < evaluator.index("payload = {")
    assert worker.count('[[ -e "${GPU_DONE_FILE}" ]] && break') >= 3
    assert worker.index("sleep 1") < worker.index("SAMPLE=$(nvidia-smi")
    assert worker.index("SAMPLE=$(nvidia-smi") < worker.index(
        "printf '%s\\n'"
    )
    card = json.loads(Path(
        "experiments/neurips_2026/allen_cahn_support_subspaces/prediction_card.json"
    ).read_text())
    assert card["protocol_id"].endswith("_v4")
    assert card["scientific_hardware_gates"]["telemetry_interval_seconds"] == 1
    assert "921ec1bcb2eab90f" in card["v3_failed_attempt_provenance"]
    assert "never deserialized" in card["v3_failed_attempt_provenance"]
