"""Serialized gated-transfer cache continuation through the real runner."""

from __future__ import annotations

import math
import os
import signal
from typing import Any

import numpy as np
import pytest
import torch

import skae.training.runner as runner_module
from skae.config import Config, get_config
from skae.training.checkpointing import CheckpointManager, CheckpointSignalExit


def _gated_transfer_config(num_steps: int) -> Config:
    cfg = get_config("generic_sparse")
    cfg.SEED = 17
    cfg.ENV.ENV_NAME = "gated_transfer_linear"
    cfg.ENV.GATED_TRANSFER_LINEAR.DT = 0.04
    cfg.TRAIN.NUM_STEPS = num_steps
    cfg.TRAIN.BATCH_SIZE = 4
    cfg.TRAIN.DATA_SIZE = 8
    cfg.TRAIN.SEQUENCE_LENGTH = 2
    cfg.TRAIN.EVAL_EVERY = 1000
    cfg.TRAIN.EVAL_NUM_STEPS = 2
    settings = cfg.TRAIN.HARD_INIT_OVERSAMPLE
    settings.ENABLED = True
    settings.FRACTION = 0.5
    settings.POOL_SIZE = 8
    settings.NUM_CANDIDATES = 16
    settings.PROBE_STEPS = 3
    settings.NUM_PERTURBATIONS = 1
    settings.TRANSIENT_WINDOW = 2
    settings.BUILD_CHUNK_SIZE = 8
    settings.JITTER_SCALE = 0.0
    cfg.MODEL.TARGET_SIZE = 8
    cfg.MODEL.ENCODER.LAYERS = [8]
    cfg.MODEL.DECODER.LAYERS = []
    return cfg


def _assert_nested_close(left: Any, right: Any, *, atol: float, rtol: float) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        if atol == 0.0 and rtol == 0.0:
            assert torch.equal(left, right)
        else:
            assert torch.allclose(left, right, atol=atol, rtol=rtol)
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key], atol=atol, rtol=rtol)
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_close(left_item, right_item, atol=atol, rtol=rtol)
        return
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) or math.isnan(right):
            assert math.isnan(left) and math.isnan(right)
        elif math.isfinite(left) and math.isfinite(right):
            assert math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
        else:
            assert left == right
        return
    assert left == right


def _load_payload(path) -> dict:
    recovered = CheckpointManager(path, retention=3).load_newest_valid()
    assert recovered is not None
    return recovered["payload"]


def _assert_continuation_payloads(
    uninterrupted: dict,
    resumed: dict,
    *,
    atol: float,
    rtol: float,
) -> None:
    assert uninterrupted["next_step"] == resumed["next_step"]
    assert uninterrupted["data_order"] == resumed["data_order"]
    assert uninterrupted["config"] == resumed["config"]
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "last_metrics",
        "logger_state",
        "best_eval_final_error",
        "checkpoint_selection_metric",
        "checkpoint_selection_score",
    ):
        _assert_nested_close(
            uninterrupted[key],
            resumed[key],
            atol=atol,
            rtol=rtol,
        )
    for key in (
        "resume_config_hash",
        "device",
        "cuda_device_count",
        "batch_count",
        "logger_history",
        "source",
    ):
        assert uninterrupted["run_identity"][key] == resumed["run_identity"][key]


@pytest.mark.parametrize(
    ("device", "atol", "rtol"),
    [
        ("cpu", 0.0, 0.0),
        pytest.param(
            "cuda",
            1e-6,
            1e-6,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_gated_transfer_serialized_resume_rebuilds_cache_and_matches_uninterrupted(
    tmp_path,
    monkeypatch,
    device: str,
    atol: float,
    rtol: float,
):
    """A signal-saved real checkpoint resumes the exact cached data stream."""
    total_steps = 6
    split_steps = 3
    uninterrupted_dir = tmp_path / "uninterrupted"
    split_dir = tmp_path / "split"
    cache_instances = []
    real_cache = runner_module._DeterministicSequenceCache

    class RecordingCache(real_cache):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.first_get_size = None
            cache_instances.append(self)

        def get(self, key):
            if self.first_get_size is None:
                self.first_get_size = len(self)
            return super().get(key)

    monkeypatch.setattr(runner_module, "_DeterministicSequenceCache", RecordingCache)
    real_train_step = runner_module.train_step
    uninterrupted_windows = []

    def record_uninterrupted(model, optimizer, x_seq, step=0):
        uninterrupted_windows.append(x_seq.detach().cpu().clone())
        return real_train_step(model, optimizer, x_seq, step=step)

    monkeypatch.setattr(runner_module, "train_step", record_uninterrupted)
    uninterrupted_model = runner_module.train(
        _gated_transfer_config(total_steps),
        device=device,
        skip_eval=True,
        save_metrics_history=True,
        checkpoint_dir=str(uninterrupted_dir),
        checkpoint_interval=1,
    )

    interrupted_windows = []

    def interrupt_after_save_boundary(model, optimizer, x_seq, step=0):
        interrupted_windows.append(x_seq.detach().cpu().clone())
        metrics = real_train_step(model, optimizer, x_seq, step=step)
        if step == split_steps - 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return metrics

    monkeypatch.setattr(runner_module, "train_step", interrupt_after_save_boundary)
    with pytest.raises(CheckpointSignalExit):
        runner_module.train(
            _gated_transfer_config(total_steps),
            device=device,
            skip_eval=True,
            save_metrics_history=True,
            checkpoint_dir=str(split_dir),
            checkpoint_interval=1,
        )

    interrupted_payload = _load_payload(split_dir)
    assert interrupted_payload["next_step"] == split_steps
    assert "training_sequence_cache" not in interrupted_payload

    resumed_windows = []

    def record_resumed(model, optimizer, x_seq, step=0):
        resumed_windows.append(x_seq.detach().cpu().clone())
        return real_train_step(model, optimizer, x_seq, step=step)

    monkeypatch.setattr(runner_module, "train_step", record_resumed)
    resumed_model = runner_module.train(
        _gated_transfer_config(total_steps),
        device=device,
        skip_eval=True,
        save_metrics_history=True,
        checkpoint_dir=str(split_dir),
        checkpoint_interval=1,
        resume=True,
    )

    assert len(uninterrupted_windows) == total_steps
    assert len(interrupted_windows) == split_steps
    assert len(resumed_windows) == total_steps - split_steps
    for expected, actual in zip(uninterrupted_windows[:split_steps], interrupted_windows):
        assert torch.equal(expected, actual)
    for expected, actual in zip(uninterrupted_windows[split_steps:], resumed_windows):
        assert torch.equal(expected, actual)

    uninterrupted_payload = _load_payload(uninterrupted_dir)
    resumed_payload = _load_payload(split_dir)
    _assert_continuation_payloads(
        uninterrupted_payload,
        resumed_payload,
        atol=atol,
        rtol=rtol,
    )
    _assert_nested_close(
        uninterrupted_model.state_dict(),
        resumed_model.state_dict(),
        atol=atol,
        rtol=rtol,
    )

    assert len(cache_instances) == 3
    assert cache_instances[1] is not cache_instances[2]
    assert all(cache.first_get_size == 0 for cache in cache_instances)
    assert all(len(cache) > 0 for cache in cache_instances)
