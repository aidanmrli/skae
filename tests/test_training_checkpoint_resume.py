"""Tests for complete-state checkpointing and exact training continuation."""

from __future__ import annotations

import json
import hashlib
import os
import random
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from skae.config import get_config
from skae.training.checkpointing import (
    CHECKPOINT_EXIT_CODE,
    CheckpointManager,
    CheckpointSignalExit,
    SignalStopper,
    capture_rng_state,
    restore_rng_state,
)
from skae.training.checkpoint_validation import valid_complete_payload
from skae.training.runner import train


def _load_payload(manager: CheckpointManager):
    recovered = manager.load_newest_valid()
    assert recovered is not None
    return recovered["payload"]


def _manager_state(token: str):
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cpu_device": "cpu",
        "torch_cuda": [],
        "torch_cuda_devices": [],
        "batch_generators": [torch.get_rng_state()],
        "batch_generator_devices": ["cpu"],
        "validation_generator": None,
        "validation_generator_device": None,
    }
    return {
        "token": token,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "rng_state": rng_state,
        "run_identity": {
            "config": {},
            "config_hash": "e" * 64,
            "resume_config_hash": "a" * 64,
            "device": "cpu",
            "cuda_device_count": 0,
            "batch_count": 1,
            "logger_history": False,
            "source": {
                "git_commit": "c" * 40,
                "git_dirty": False,
                "git_status_sha256": "d" * 64,
            },
        },
        "logger_state": {
            "save_history": False,
            "metrics_history": [],
            "step_count": 0,
            "summary_state": {},
        },
        "config": {},
        "data_order": {
            "num_batches": 1,
            "batch_size": 1,
            "sequence_length": 1,
            "seed": 0,
            "generator_index": 0,
        },
        "storage_contract": {
            "retention": 2,
            "checkpoint_interval": 1,
            "scratch_root": "scratch",
            "permanent_root": None,
        },
        "last_metrics": {},
        "scheduler_state_dict": None,
        "scaler_state_dict": None,
        "best_eval_final_error": float("inf"),
        "checkpoint_selection_metric": None,
        "checkpoint_selection_score": None,
    }


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _small_config(num_steps: int):
    cfg = get_config("generic")
    cfg.SEED = 19
    cfg.MODEL.TARGET_SIZE = 8
    cfg.MODEL.ENCODER.LAYERS = [8]
    cfg.TRAIN.NUM_STEPS = num_steps
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TRAIN.DATA_SIZE = 8
    # Evaluate at every completed step so a split at step 3 has the same
    # evaluation/metric history as the uninterrupted six-step run.
    cfg.TRAIN.EVAL_EVERY = 1
    cfg.TRAIN.EVAL_NUM_STEPS = 2
    return cfg


def test_newest_valid_checkpoint_skips_corrupt_generation(tmp_path):
    manager = CheckpointManager(tmp_path / "scratch", retention=2)
    manager.save(_manager_state("old"), next_step=4)
    newest = manager.save(_manager_state("new"), next_step=8)

    checkpoint_path = manager.root / newest["checkpoint_file"]
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"corrupt")

    recovered = manager.load_newest_valid()
    assert recovered is not None
    assert recovered["payload"]["token"] == "old"
    assert recovered["payload"]["next_step"] == 4
    assert json.loads((manager.root / "checkpoint_receipt.json").read_text())["next_step"] == 8


def test_incomplete_newest_generation_is_skipped(tmp_path):
    manager = CheckpointManager(tmp_path / "scratch", retention=2)
    manager.save(_manager_state("old"), next_step=4)
    newest = manager.save(_manager_state("incomplete"), next_step=8)
    checkpoint_path = manager.root / newest["checkpoint_file"]
    malformed = torch.load(checkpoint_path, weights_only=False)
    malformed["run_identity"]["batch_count"] = "malformed"
    torch.save(malformed, checkpoint_path)
    manifest_path = manager.root / "checkpoint-00000002.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest["size_bytes"] = checkpoint_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))

    recovered = manager.load_newest_valid()
    assert recovered is not None
    assert recovered["payload"]["token"] == "old"


def test_incomplete_newest_rng_metadata_is_skipped(tmp_path):
    manager = CheckpointManager(tmp_path / "scratch", retention=2)
    manager.save(_manager_state("old"), next_step=4)
    newest = manager.save(_manager_state("bad-rng"), next_step=8)
    checkpoint_path = manager.root / newest["checkpoint_file"]
    malformed = torch.load(checkpoint_path, weights_only=False)
    malformed["rng_state"]["batch_generator_devices"] = ["not-a-device"]
    torch.save(malformed, checkpoint_path)
    manifest_path = manager.root / "checkpoint-00000002.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest["size_bytes"] = checkpoint_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))

    recovered = manager.load_newest_valid()
    assert recovered is not None
    assert recovered["payload"]["token"] == "old"


def test_complete_payload_accepts_logger_metadata_scalar():
    payload = _manager_state("logger-metadata")
    payload.update({
        "schema_version": 1,
        "run_id": "run-id",
        "generation": 1,
        "next_step": 1,
    })
    payload["run_identity"]["logger_history"] = True
    payload["logger_state"] = {
        "save_history": True,
        "metrics_history": [
            {"step": 0, "name": "train/metadata", "value": "sqrt_dim"}
        ],
        "step_count": 1,
        "summary_state": {
            "train/metadata": {
                "final": "sqrt_dim",
                "min": None,
                "max": None,
                "sum": 0.0,
                "count": 0,
            }
        },
    }
    assert valid_complete_payload(payload)


def test_permanent_latest_fallback_and_progress_selection(tmp_path):
    scratch = tmp_path / "scratch"
    permanent = tmp_path / "permanent"
    manager = CheckpointManager(scratch, permanent_root=permanent, retention=2)
    manager.save(_manager_state("permanent"), next_step=4)
    scratch_only = CheckpointManager(scratch, retention=2)
    scratch_only.save(_manager_state("scratch-newer"), next_step=8)

    recovered = CheckpointManager(scratch, permanent_root=permanent).load_newest_valid()
    assert recovered is not None
    assert recovered["payload"]["token"] == "scratch-newer"
    for path in scratch.iterdir():
        path.unlink()
    recovered = CheckpointManager(scratch, permanent_root=permanent).load_newest_valid()
    assert recovered is not None
    assert recovered["payload"]["token"] == "permanent"


def test_permanent_newer_state_beats_older_scratch_state(tmp_path):
    scratch = tmp_path / "scratch"
    permanent = tmp_path / "permanent"
    manager = CheckpointManager(scratch, permanent_root=permanent, retention=2)
    manager.save(_manager_state("scratch-old"), next_step=4)
    newest = manager.save(_manager_state("permanent-newer"), next_step=8)
    (scratch / newest["checkpoint_file"]).unlink()
    (scratch / "checkpoint-00000002.manifest.json").unlink()

    recovered = CheckpointManager(scratch, permanent_root=permanent).load_newest_valid()
    assert recovered is not None
    assert recovered["payload"]["token"] == "permanent-newer"


def test_rng_snapshot_restores_all_training_streams():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    generators = [torch.Generator().manual_seed(21), torch.Generator().manual_seed(22)]
    validation = torch.Generator().manual_seed(23)
    state = capture_rng_state(generators, validation)

    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        [torch.rand(3, generator=generator) for generator in generators],
        torch.rand(3, generator=validation),
    )
    restore_rng_state(state, generators, validation)
    actual = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
        [torch.rand(3, generator=generator) for generator in generators],
        torch.rand(3, generator=validation),
    )
    assert expected[0:2] == pytest.approx(actual[0:2])
    for expected_tensor, actual_tensor in zip(expected[2:], actual[2:]):
        if isinstance(expected_tensor, list):
            for expected_item, actual_item in zip(expected_tensor, actual_tensor):
                assert torch.equal(expected_item, actual_item)
        else:
            assert torch.equal(expected_tensor, actual_tensor)


def test_signal_stopper_uses_checkpoint_exit_status():
    stopper = SignalStopper()
    stopper.install()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        assert stopper.requested is True
        assert stopper.signal_number == signal.SIGTERM
        with pytest.raises(CheckpointSignalExit) as exc_info:
            raise CheckpointSignalExit()
        assert exc_info.value.code == CHECKPOINT_EXIT_CODE
    finally:
        stopper.restore()


def test_checkpoint_launcher_signals_actual_task_pid():
    launcher = (
        Path(__file__).parents[1]
        / "scripts/neurips_2026/checkpoint_resume_test/run.sh"
    )
    text = launcher.read_text()
    assert 'TASK_PID_FILE="$CHECKPOINT_DIR/task-${RESTART_COUNT}.pid"' in text
    assert 'temporary="${pid_file}.tmp.$$"' in text
    assert 'exec "$@"' in text
    assert "wait_for_task_pid" in text
    assert 'kill -TERM "$TASK_PID"' in text
    assert 'kill -TERM -- "-$TRAIN_PID"' not in text
    assert 'wait "$TRAIN_PID"' in text
    assert '[[ "$train_rc" -ne 75 ]]' in text


def test_split_resume_matches_uninterrupted_training_exactly(tmp_path):
    uninterrupted_dir = tmp_path / "uninterrupted"
    split_dir = tmp_path / "split"

    uninterrupted_model = train(
        _small_config(6),
        device="cpu",
        skip_eval=True,
        save_metrics_history=True,
        checkpoint_dir=str(uninterrupted_dir),
        checkpoint_interval=1,
    )

    train(
        _small_config(3),
        device="cpu",
        skip_eval=True,
        save_metrics_history=True,
        checkpoint_dir=str(split_dir),
        checkpoint_interval=1,
    )
    resumed_model = train(
        _small_config(6),
        device="cpu",
        skip_eval=True,
        save_metrics_history=True,
        checkpoint_dir=str(split_dir),
        checkpoint_interval=1,
        resume=True,
    )

    for name, parameter in uninterrupted_model.state_dict().items():
        assert torch.equal(parameter, resumed_model.state_dict()[name]), name

    uninterrupted_payload = _load_payload(CheckpointManager(uninterrupted_dir, retention=2))
    resumed_payload = _load_payload(CheckpointManager(split_dir, retention=2))
    _assert_nested_equal(
        uninterrupted_payload["optimizer_state_dict"],
        resumed_payload["optimizer_state_dict"],
    )
    _assert_nested_equal(uninterrupted_payload["rng_state"], resumed_payload["rng_state"])
    assert uninterrupted_payload["next_step"] == resumed_payload["next_step"] == 6
    assert uninterrupted_payload["logger_state"] == resumed_payload["logger_state"]
    assert uninterrupted_payload["data_order"] == resumed_payload["data_order"]
    assert uninterrupted_payload["best_eval_final_error"] == resumed_payload["best_eval_final_error"]
    assert uninterrupted_payload["checkpoint_selection_metric"] == resumed_payload["checkpoint_selection_metric"]
    assert uninterrupted_payload["checkpoint_selection_score"] == resumed_payload["checkpoint_selection_score"]
    for key in (
        "resume_config_hash",
        "device",
        "cuda_device_count",
        "batch_count",
        "logger_history",
        "source",
    ):
        assert uninterrupted_payload["run_identity"][key] == resumed_payload["run_identity"][key]

    for directory in (uninterrupted_dir, split_dir):
        assert (directory / "latest.json").exists()
        assert (directory / "last.pt").exists()
        assert (directory / "checkpoint_receipt.json").exists()
        latest = json.loads((directory / "latest.json").read_text())
        receipt = json.loads((directory / "checkpoint_receipt.json").read_text())
        assert latest["next_step"] == 6
        assert latest["save_duration_seconds"] >= 0.0
        assert receipt["checkpoint_interval"] == 1
