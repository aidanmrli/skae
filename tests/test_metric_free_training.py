"""Equivalence and continuation contracts for scheduled metric collection."""

from __future__ import annotations

import copy
import math
import os
import signal
from typing import Any

import numpy as np
import pytest
import torch

import skae.training.runner as runner_module
from skae.config import Config, get_config
from skae.data import make_env
from skae.model import make_model
from skae.training.checkpointing import CheckpointError, CheckpointManager, CheckpointSignalExit
from skae.training.runner import build_optimizer, train_step


def _gated_config() -> Config:
    cfg = get_config("generic_sparse")
    cfg.SEED = 29
    cfg.ENV.ENV_NAME = "gated_transfer_linear"
    cfg.ENV.GATED_TRANSFER_LINEAR.DT = 0.04
    cfg.MODEL.TARGET_SIZE = 8
    cfg.MODEL.ENCODER.LAYERS = [8]
    cfg.MODEL.DECODER.LAYERS = []
    cfg.TRAIN.BATCH_SIZE = 4
    cfg.TRAIN.DATA_SIZE = 8
    cfg.TRAIN.SEQUENCE_LENGTH = 2
    cfg.TRAIN.METRICS_EVERY = 3
    cfg.TRAIN.EIGEN_METRICS_EVERY = 2
    return cfg


def _lista_penalty_config() -> Config:
    cfg = _gated_config()
    cfg.MODEL.MODEL_NAME = "LISTAKM"
    cfg.MODEL.K_STRUCTURE = "block_diagonal"
    cfg.MODEL.K_NUM_BLOCKS = 2
    cfg.MODEL.K_BLOCK_SIZE = 4
    cfg.MODEL.BLOCK_LOSS.ENABLED = True
    cfg.MODEL.BLOCK_LOSS.ONE_BLOCK_LOSS = "low_entropy"
    cfg.MODEL.BLOCK_LOSS.ONE_BLOCK_WEIGHT = 0.1
    cfg.MODEL.BLOCK_LOSS.BALANCE_LOSS = "usage_entropy"
    cfg.MODEL.BLOCK_LOSS.BALANCE_WEIGHT = 0.1
    cfg.MODEL.DECODER_COHERENCE_WEIGHT = 0.05
    return cfg


def _assert_nested_close(left: Any, right: Any, *, atol: float, rtol: float) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
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
        assert math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
        return
    assert left == right


def _loss_inputs(model: torch.nn.Module, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, seq_len, obs_size = x_seq.shape
    horizon = seq_len - 1
    x0 = x_seq[:, 0, :]
    x_true = x_seq[:, 1:, :]
    z_all = model.encode(x_seq.reshape(batch_size * seq_len, obs_size)).reshape(
        batch_size, seq_len, -1
    )
    z0 = z_all[:, 0, :]
    z_true = z_all[:, 1:, :]
    z_pred = model.rollout_latent_discrete(z0, horizon=horizon)
    x_pred = model.decode(z_pred.reshape(batch_size * horizon, -1)).reshape(
        batch_size, horizon, obs_size
    )
    x_recon_true = model.decode(z_true.reshape(batch_size * horizon, -1)).reshape_as(x_true)
    return {
        "x_pred": x_pred,
        "x_true": x_true,
        "x0": x0,
        "z0": z0,
        "z_pred": z_pred,
        "z_true": z_true,
        "reconstruction_error": torch.norm(x_true - x_recon_true, dim=-1).mean(),
        "sparsity_latent": z_pred,
    }


def _run_pair(cfg: Config, *, device: str, monkeypatch) -> None:
    torch.manual_seed(123)
    metric_model = make_model(cfg, observation_size=2).to(device)
    free_model = copy.deepcopy(metric_model).to(device)
    metric_optimizer = build_optimizer(metric_model, cfg)
    free_optimizer = build_optimizer(free_model, cfg)
    free_optimizer.load_state_dict(copy.deepcopy(metric_optimizer.state_dict()))

    x_seq = torch.randn(cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.SEQUENCE_LENGTH + 1, 2, device=device)
    metric_inputs = _loss_inputs(metric_model, x_seq)
    free_inputs = _loss_inputs(free_model, x_seq)
    metric_loss, metric_snapshot = metric_model.loss(
        **metric_inputs,
        step=0,
        collect_metrics=True,
        metric_eigen_every=cfg.TRAIN.EIGEN_METRICS_EVERY,
    )
    free_loss, free_snapshot = free_model.loss(
        **free_inputs,
        step=0,
        collect_metrics=False,
        metric_eigen_every=cfg.TRAIN.EIGEN_METRICS_EVERY,
    )
    tolerance = 1e-6 if device.startswith("cuda") else 0.0
    assert torch.allclose(metric_loss, free_loss, atol=tolerance, rtol=tolerance)
    assert free_snapshot == {}
    assert "loss" in metric_snapshot

    eigen_called = False
    original_eigen = free_model._k_eigen_metrics

    def fail_eigen():
        nonlocal eigen_called
        eigen_called = True
        raise AssertionError("metric-free step unexpectedly computed eigendecomposition")

    for step in range(4):
        if step == 1:
            free_model._k_eigen_metrics = fail_eigen
        elif step == 2:
            free_model._k_eigen_metrics = original_eigen
        metric_metrics = train_step(
            metric_model,
            metric_optimizer,
            x_seq,
            step=step,
            collect_metrics=True,
            metric_eigen_every=cfg.TRAIN.EIGEN_METRICS_EVERY,
        )
        free_metrics = train_step(
            free_model,
            free_optimizer,
            x_seq,
            step=step,
            collect_metrics=step in (0, 2),
            metric_eigen_every=cfg.TRAIN.EIGEN_METRICS_EVERY,
        )
        if step in (0, 2):
            assert free_metrics.keys() == metric_metrics.keys()
            for key in free_metrics:
                left, right = free_metrics[key], metric_metrics[key]
                if isinstance(left, (float, int)) and isinstance(right, (float, int)):
                    assert math.isclose(left, right, abs_tol=tolerance, rel_tol=tolerance)
                else:
                    assert left == right
        else:
            assert free_metrics == {}
        for left, right in zip(metric_model.parameters(), free_model.parameters()):
            assert torch.allclose(left, right, atol=tolerance, rtol=tolerance)
            assert left.grad is not None and right.grad is not None
            assert torch.allclose(left.grad, right.grad, atol=tolerance, rtol=tolerance)
        _assert_nested_close(
            metric_optimizer.state_dict(),
            free_optimizer.state_dict(),
            atol=tolerance,
            rtol=tolerance,
        )
    assert not eigen_called


def test_metric_free_gated_transfer_cpu_is_exact(monkeypatch):
    _run_pair(_gated_config(), device="cpu", monkeypatch=monkeypatch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_metric_free_gated_transfer_cuda_matches_scheduled_metrics(monkeypatch):
    _run_pair(_gated_config(), device="cuda", monkeypatch=monkeypatch)


def test_metric_free_lista_preserves_configured_penalties(monkeypatch):
    _run_pair(_lista_penalty_config(), device="cpu", monkeypatch=monkeypatch)


def _checkpoint_config(num_steps: int = 6) -> Config:
    cfg = _gated_config()
    cfg.TRAIN.NUM_STEPS = num_steps
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.TRAIN.DATA_SIZE = 8
    cfg.TRAIN.SEQUENCE_LENGTH = 1
    cfg.TRAIN.EVAL_EVERY = 1000
    cfg.TRAIN.METRICS_EVERY = 3
    cfg.TRAIN.EIGEN_METRICS_EVERY = 2
    return cfg


def _checkpoint_payload(path) -> dict:
    recovered = CheckpointManager(path, retention=3).load_newest_valid()
    assert recovered is not None
    return recovered["payload"]


def test_checkpoint_resume_spans_metric_free_steps_and_validates_contract(tmp_path, monkeypatch):
    total_steps = 6
    split_step = 3
    uninterrupted_dir = tmp_path / "uninterrupted"
    split_dir = tmp_path / "split"

    uninterrupted_model = runner_module.train(
        _checkpoint_config(total_steps),
        device="cpu",
        skip_eval=True,
        checkpoint_dir=str(uninterrupted_dir),
        checkpoint_interval=3,
    )

    real_train_step = runner_module.train_step

    def interrupt_after_checkpoint_boundary(
        model, optimizer, x_seq, step=0, collect_metrics=True, metric_eigen_every=100
    ):
        metrics = real_train_step(
            model,
            optimizer,
            x_seq,
            step=step,
            collect_metrics=collect_metrics,
            metric_eigen_every=metric_eigen_every,
        )
        if step == split_step - 1:
            os.kill(os.getpid(), signal.SIGTERM)
        return metrics

    monkeypatch.setattr(runner_module, "train_step", interrupt_after_checkpoint_boundary)
    with pytest.raises(CheckpointSignalExit):
        runner_module.train(
            _checkpoint_config(total_steps),
            device="cpu",
            skip_eval=True,
            checkpoint_dir=str(split_dir),
            checkpoint_interval=3,
        )
    interrupted = _checkpoint_payload(split_dir)
    assert interrupted["next_step"] == split_step
    assert interrupted["last_metrics_step"] == split_step - 1
    assert interrupted["metric_contract"] == {
        "metrics_every": 3,
        "eigen_metrics_every": 2,
        "history_mode": "scheduled",
    }

    monkeypatch.setattr(runner_module, "train_step", real_train_step)
    resumed_model = runner_module.train(
        _checkpoint_config(total_steps),
        device="cpu",
        skip_eval=True,
        checkpoint_dir=str(split_dir),
        checkpoint_interval=3,
        resume=True,
    )
    uninterrupted = _checkpoint_payload(uninterrupted_dir)
    resumed = _checkpoint_payload(split_dir)
    assert uninterrupted["next_step"] == resumed["next_step"] == total_steps
    assert uninterrupted["last_metrics_step"] == resumed["last_metrics_step"] == total_steps - 1
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "data_order",
        "last_metrics",
        "logger_state",
        "metric_contract",
    ):
        _assert_nested_close(uninterrupted[key], resumed[key], atol=0.0, rtol=0.0)
    for name, parameter in uninterrupted_model.state_dict().items():
        assert torch.equal(parameter, resumed_model.state_dict()[name]), name

    changed = _checkpoint_config(total_steps)
    changed.TRAIN.METRICS_EVERY = 1
    with pytest.raises(CheckpointError, match="resume_config_hash"):
        runner_module.train(
            changed,
            device="cpu",
            skip_eval=True,
            checkpoint_dir=str(split_dir),
            checkpoint_interval=3,
            resume=True,
        )
