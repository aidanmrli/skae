"""Paired short training and resume equivalence checks for sequence caching."""

import copy
import math

import pytest
import torch

from skae.config import Config, get_config
from skae.data import (
    HardInitialConditionWrapper,
    VectorWrapper,
    make_env,
    wrap_training_env,
)
from skae.model import make_model
from skae.training.runner import (
    _DeterministicSequenceCache,
    build_optimizer,
    generate_sequence_batch_for_device,
    train_step,
)


def _rng(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _paired_training_config() -> Config:
    cfg = get_config("generic_sparse")
    cfg.SEED = 17
    cfg.ENV.ENV_NAME = "gated_transfer_linear"
    cfg.ENV.GATED_TRANSFER_LINEAR.DT = 0.04
    cfg.TRAIN.BATCH_SIZE = 4
    cfg.TRAIN.DATA_SIZE = 8
    cfg.TRAIN.SEQUENCE_LENGTH = 2
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


def _canonical_training_env(cfg: Config) -> VectorWrapper:
    return VectorWrapper(wrap_training_env(make_env(cfg), cfg), cfg.TRAIN.BATCH_SIZE)


def _assert_nested_equal(left, right, *, atol: float, rtol: float):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        if atol == 0.0 and rtol == 0.0:
            assert torch.equal(left, right)
        else:
            assert torch.allclose(left, right, atol=atol, rtol=rtol)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key], atol=atol, rtol=rtol)
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item, atol=atol, rtol=rtol)
        return
    assert left == right


def _assert_training_states_equal(
    fast_model,
    legacy_model,
    fast_optimizer,
    legacy_optimizer,
    fast_metrics,
    legacy_metrics,
    *,
    atol: float,
    rtol: float,
):
    assert fast_metrics.keys() == legacy_metrics.keys()
    for key in fast_metrics:
        left, right = fast_metrics[key], legacy_metrics[key]
        if isinstance(left, float):
            if atol == 0.0 and rtol == 0.0:
                assert left == right
            else:
                assert math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
        else:
            assert left == right

    fast_parameters = dict(fast_model.named_parameters())
    legacy_parameters = dict(legacy_model.named_parameters())
    assert fast_parameters.keys() == legacy_parameters.keys()
    for name in fast_parameters:
        fast_parameter = fast_parameters[name]
        legacy_parameter = legacy_parameters[name]
        if atol == 0.0 and rtol == 0.0:
            assert torch.equal(fast_parameter, legacy_parameter)
        else:
            assert torch.allclose(fast_parameter, legacy_parameter, atol=atol, rtol=rtol)
        assert fast_parameter.grad is not None
        assert legacy_parameter.grad is not None
        if atol == 0.0 and rtol == 0.0:
            assert torch.equal(fast_parameter.grad, legacy_parameter.grad)
        else:
            assert torch.allclose(fast_parameter.grad, legacy_parameter.grad, atol=atol, rtol=rtol)

    _assert_nested_equal(
        fast_optimizer.state_dict(),
        legacy_optimizer.state_dict(),
        atol=atol,
        rtol=rtol,
    )


def _paired_window(
    step: int,
    fast_env: VectorWrapper,
    legacy_env: VectorWrapper,
    fast_rngs,
    legacy_rngs,
    cache: _DeterministicSequenceCache,
    cfg: Config,
    monkeypatch,
    device: str,
):
    batch_index = step % (cfg.TRAIN.DATA_SIZE // cfg.TRAIN.BATCH_SIZE)
    monkeypatch.delenv("SKAE_DISABLE_BATCH_FASTPATH", raising=False)
    fast_window = generate_sequence_batch_for_device(
        fast_env,
        fast_rngs[batch_index],
        window_length=cfg.TRAIN.SEQUENCE_LENGTH,
        device=device,
        sequence_cache=cache,
    )
    monkeypatch.setenv("SKAE_DISABLE_BATCH_FASTPATH", "1")
    legacy_window = generate_sequence_batch_for_device(
        legacy_env,
        legacy_rngs[batch_index],
        window_length=cfg.TRAIN.SEQUENCE_LENGTH,
        device=device,
    )
    monkeypatch.delenv("SKAE_DISABLE_BATCH_FASTPATH", raising=False)
    return fast_window, legacy_window


def _run_paired_training_equivalence(monkeypatch, *, device: str, atol: float, rtol: float):
    cfg = _paired_training_config()
    fast_env = _canonical_training_env(cfg)
    legacy_env = _canonical_training_env(cfg)
    assert isinstance(fast_env.env, HardInitialConditionWrapper)
    assert isinstance(legacy_env.env, HardInitialConditionWrapper)

    torch.manual_seed(123)
    fast_model = make_model(cfg, observation_size=2).to(device)
    legacy_model = copy.deepcopy(fast_model).to(device)
    fast_optimizer = build_optimizer(fast_model, cfg)
    legacy_optimizer = build_optimizer(legacy_model, cfg)
    legacy_optimizer.load_state_dict(copy.deepcopy(fast_optimizer.state_dict()))
    fast_rngs = [_rng(cfg.SEED + i * cfg.TRAIN.BATCH_SIZE) for i in range(2)]
    legacy_rngs = [_rng(cfg.SEED + i * cfg.TRAIN.BATCH_SIZE) for i in range(2)]
    assert [rng.initial_seed() for rng in fast_rngs] == [17, 21]
    cache = _DeterministicSequenceCache(max_entries=2)

    for step in range(2):
        fast_window, legacy_window = _paired_window(
            step,
            fast_env,
            legacy_env,
            fast_rngs,
            legacy_rngs,
            cache,
            cfg,
            monkeypatch,
            device,
        )
        if atol == 0.0 and rtol == 0.0:
            assert torch.equal(fast_window, legacy_window)
        else:
            assert torch.allclose(fast_window, legacy_window, atol=atol, rtol=rtol)
        read_only_snapshot = fast_window.clone()
        fast_metrics = train_step(fast_model, fast_optimizer, fast_window, step=step)
        assert torch.equal(fast_window, read_only_snapshot)
        legacy_metrics = train_step(legacy_model, legacy_optimizer, legacy_window, step=step)
        _assert_training_states_equal(
            fast_model,
            legacy_model,
            fast_optimizer,
            legacy_optimizer,
            fast_metrics,
            legacy_metrics,
            atol=atol,
            rtol=rtol,
        )

    checkpoint_model = copy.deepcopy(fast_model)
    checkpoint_optimizer_state = copy.deepcopy(fast_optimizer.state_dict())
    resumed_optimizer = build_optimizer(checkpoint_model, cfg)
    resumed_optimizer.load_state_dict(checkpoint_optimizer_state)
    resumed_env = _canonical_training_env(cfg)
    resumed_rngs = [_rng(cfg.SEED + i * cfg.TRAIN.BATCH_SIZE) for i in range(2)]
    resumed_cache = _DeterministicSequenceCache(max_entries=2)
    assert len(resumed_cache) == 0

    resumed_window, legacy_window = _paired_window(
        2,
        resumed_env,
        legacy_env,
        resumed_rngs,
        legacy_rngs,
        resumed_cache,
        cfg,
        monkeypatch,
        device,
    )
    if atol == 0.0 and rtol == 0.0:
        assert torch.equal(resumed_window, legacy_window)
    else:
        assert torch.allclose(resumed_window, legacy_window, atol=atol, rtol=rtol)
    resumed_metrics = train_step(checkpoint_model, resumed_optimizer, resumed_window, step=2)
    legacy_metrics = train_step(legacy_model, legacy_optimizer, legacy_window, step=2)
    _assert_training_states_equal(
        checkpoint_model,
        legacy_model,
        resumed_optimizer,
        legacy_optimizer,
        resumed_metrics,
        legacy_metrics,
        atol=atol,
        rtol=rtol,
    )
    assert len(resumed_cache) == 1


def test_cached_and_legacy_training_are_cpu_bitwise_equivalent_and_resume_rebuilds_cache(
    monkeypatch,
):
    _run_paired_training_equivalence(monkeypatch, device="cpu", atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cached_and_legacy_training_match_on_cuda_with_tight_tolerance(monkeypatch):
    _run_paired_training_equivalence(monkeypatch, device="cuda", atol=1e-6, rtol=1e-6)
