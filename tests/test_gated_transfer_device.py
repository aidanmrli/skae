"""Focused device and cache tests for the gated-transfer data fast path."""

import pytest
import torch

from skae.config import Config
from skae.data import GatedLocalLinear, GatedTransferLinear, VectorWrapper
from skae.training.runner import (
    _DeterministicSequenceCache,
    generate_sequence_batch_for_device,
)


def _config(environment: str = "gated_transfer_linear") -> Config:
    cfg = Config()
    cfg.ENV.ENV_NAME = environment
    cfg.ENV.GATED_TRANSFER_LINEAR.DT = 0.04
    cfg.ENV.GATED_LOCAL_LINEAR.DT = 0.04
    return cfg


def _rng(seed: int = 1234) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_cache_population_is_bitwise_equal_to_legacy_cpu_sequence():
    cfg = _config()
    cached_env = VectorWrapper(GatedTransferLinear(cfg), batch_size=8)
    legacy_env = VectorWrapper(GatedTransferLinear(cfg), batch_size=8)
    cache = _DeterministicSequenceCache(max_entries=4)

    cached = generate_sequence_batch_for_device(
        cached_env,
        _rng(),
        window_length=8,
        device="cpu",
        sequence_cache=cache,
    )
    legacy = generate_sequence_batch_for_device(
        legacy_env,
        _rng(),
        window_length=8,
        device="cpu",
        legacy_cpu_sequence=True,
    )

    assert cached.is_contiguous()
    assert torch.equal(cached, legacy)


def test_cache_is_bounded_and_reuses_the_same_contiguous_tensor(monkeypatch):
    cfg = _config()
    env = VectorWrapper(GatedTransferLinear(cfg), batch_size=4)
    cache = _DeterministicSequenceCache(max_entries=1)

    first = generate_sequence_batch_for_device(
        env,
        _rng(10),
        window_length=2,
        device="cpu",
        sequence_cache=cache,
    )
    second = generate_sequence_batch_for_device(
        env,
        _rng(10),
        window_length=2,
        device="cpu",
        sequence_cache=cache,
    )
    assert first is second
    assert first.is_contiguous()
    assert len(cache) == 1

    generate_sequence_batch_for_device(
        env,
        _rng(11),
        window_length=2,
        device="cpu",
        sequence_cache=cache,
    )
    assert len(cache) == 1

    monkeypatch.setenv("SKAE_DISABLE_BATCH_FASTPATH", "1")
    opt_out_cache = _DeterministicSequenceCache(max_entries=4)
    opt_out = generate_sequence_batch_for_device(
        env,
        _rng(10),
        window_length=2,
        device="cpu",
        sequence_cache=opt_out_cache,
    )
    assert len(opt_out_cache) == 0
    assert torch.equal(opt_out, first)


def test_non_target_environment_does_not_enter_target_cache_path():
    cfg = _config("gated_local_linear")
    env = VectorWrapper(GatedLocalLinear(cfg), batch_size=4)
    cache = _DeterministicSequenceCache(max_entries=4)

    sequence = generate_sequence_batch_for_device(
        env,
        _rng(),
        window_length=2,
        device="cpu",
        sequence_cache=cache,
    )

    assert sequence.shape == (4, 3, 2)
    assert len(cache) == 0
    assert not env.supports_batched_step


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_cuda_sequence_matches_legacy_cpu_sequence_and_uses_device_views():
    cfg = _config()
    native_env = VectorWrapper(GatedTransferLinear(cfg), batch_size=8)
    legacy_env = VectorWrapper(GatedTransferLinear(cfg), batch_size=8)

    native = generate_sequence_batch_for_device(
        native_env,
        _rng(),
        window_length=8,
        device="cuda",
    )
    legacy = generate_sequence_batch_for_device(
        legacy_env,
        _rng(),
        window_length=8,
        device="cpu",
        legacy_cpu_sequence=True,
    ).to("cuda")

    assert native.device.type == "cuda"
    assert torch.allclose(native, legacy, atol=5e-6, rtol=5e-6)

    device_views = native_env.unwrapped._device_view_cache[
        (native.device, native.dtype)
    ]
    assert all(value.device == native.device for value in device_views.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_batched_step_matches_legacy_step_on_cuda():
    cfg = _config()
    env = GatedTransferLinear(cfg)
    states = torch.stack([env.reset(_rng(100 + i)) for i in range(8)], dim=0)
    cpu_reference = env.step(states)
    cuda_result = env.step(states.cuda()).cpu()

    assert torch.allclose(cuda_result, cpu_reference, atol=5e-6, rtol=5e-6)
