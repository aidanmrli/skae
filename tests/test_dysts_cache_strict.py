from types import SimpleNamespace

import numpy as np
import torch

from skae.data import (
    DystsTrajectoryCache,
    DystsTrajectoryTimeout,
    _build_dysts_native_trajectory,
)


def _cache(tmp_path, split="train"):
    cache = DystsTrajectoryCache.__new__(DystsTrajectoryCache)
    cache.env = SimpleNamespace(
        system_name="Example",
        dt=0.3,
        observation_size=3,
        _dim=3,
        ic_noise_scale=0.1,
    )
    cache.cache_split = split
    cache.cache_steps = 4
    cache.cache_trajectories = 2
    cache.cache_warmup = 1
    cache.standardize = True
    cache.resample = False
    cache.pts_per_period = 100
    cache.cache_dir = str(tmp_path)
    cache.cache_reuse = True
    seed = cache._CACHE_BUILD_SEED + cache._split_seed_offset(split)
    cache._cache_build_rng = torch.Generator().manual_seed(seed)
    return cache


def test_initial_conditions_are_disjoint_across_splits(tmp_path):
    base = np.zeros(3, dtype=np.float32)
    std = np.ones(3, dtype=np.float32)
    train = _cache(tmp_path, "train")._initial_conditions(base, std)
    val = _cache(tmp_path, "val")._initial_conditions(base, std)
    policy = _cache(tmp_path, "policy")._initial_conditions(base, std)
    test = _cache(tmp_path, "test")._initial_conditions(base, std)
    fingerprints = {
        tuple(value.tolist())
        for group in (train, val, policy, test)
        for value in group
    }
    assert len(fingerprints) == 8
    assert tuple(base.tolist()) not in fingerprints


def test_cache_reload_requires_exact_schema_metadata_shape_and_finiteness(tmp_path):
    cache = _cache(tmp_path)
    base = np.zeros(3, dtype=np.float32)
    std = np.ones(3, dtype=np.float32)
    path = cache._cache_path(base, std)
    tensor = torch.zeros(2, 5, 3, dtype=torch.float32)
    cache._save_cached_trajectories(path, tensor)
    assert torch.equal(cache._load_cached_trajectories(path), tensor)

    payload = torch.load(path, weights_only=False)
    payload["meta"]["cache_split"] = "test"
    torch.save(payload, path)
    assert cache._load_cached_trajectories(path) is None

    cache._save_cached_trajectories(path, tensor)
    payload = torch.load(path, weights_only=False)
    payload["trajectories"][0, 0, 0] = float("nan")
    torch.save(payload, path)
    assert cache._load_cached_trajectories(path) is None


def test_cache_records_solver_fallback_provenance(tmp_path):
    cache = _cache(tmp_path)
    cache.cache_primary_method = "Radau"
    cache.cache_trajectory_timeout_seconds = 300.0
    cache.cache_timeout_fallback_method = "DOP853"
    cache.cache_fallback_timeout_seconds = 1200.0
    cache._cache_integration_method_counts = {"Radau": 197, "DOP853": 3}
    base = np.zeros(3, dtype=np.float32)
    std = np.ones(3, dtype=np.float32)
    path = cache._cache_path(base, std)

    cache._save_cached_trajectories(
        path,
        torch.zeros(2, 5, 3, dtype=torch.float32),
    )
    meta = torch.load(path, weights_only=False)["meta"]

    assert meta["integration_primary_method"] == "Radau"
    assert meta["integration_primary_timeout_seconds"] == 300.0
    assert meta["integration_fallback_method"] == "DOP853"
    assert meta["integration_fallback_timeout_seconds"] == 1200.0
    assert meta["integration_method_counts"] == {"Radau": 197, "DOP853": 3}


def test_timeout_fallback_retries_exact_same_initial_condition(monkeypatch):
    calls = []

    class FakeSystem:
        def __init__(self):
            self.ic = np.zeros(3, dtype=np.float32)

    class FakeEnv:
        def __init__(self, **_kwargs):
            self.system = FakeSystem()

        def make_trajectory_native(self, *, n, method, **_kwargs):
            calls.append((method, self.system.ic.copy()))
            if method == "Radau":
                raise DystsTrajectoryTimeout("synthetic timeout")
            return torch.ones(n, 3, dtype=torch.float32)

    monkeypatch.setattr("skae.benchmarks.dysts_adapter.DystsEnv", FakeEnv)
    initial_condition = np.array([1.0, -2.0, 3.0], dtype=np.float32)

    trajectory, method = _build_dysts_native_trajectory(
        "Sakarya",
        0.3,
        initial_condition,
        8,
        False,
        100,
        True,
        primary_method="Radau",
        primary_timeout_seconds=300.0,
        fallback_method="DOP853",
        fallback_timeout_seconds=1200.0,
        trajectory_index=17,
    )

    assert method == "DOP853"
    assert trajectory.shape == (8, 3)
    assert [call[0] for call in calls] == ["Radau", "DOP853"]
    assert np.array_equal(calls[0][1], initial_condition)
    assert np.array_equal(calls[1][1], initial_condition)
