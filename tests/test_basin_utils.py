"""Tests for basin labeling utilities on high-dimensional benchmark systems."""

import torch

from skae.basin_utils import BasinLabeledDataset
from skae.config import Config


def test_hopfield_basin_dataset_uses_pattern_labels():
    cfg = Config()
    cfg.SEED = 0
    cfg.ENV.ENV_NAME = "hopfield"
    cfg.ENV.HOPFIELD.NUM_NEURONS = 8
    cfg.ENV.HOPFIELD.NUM_PATTERNS = 3
    cfg.ENV.HOPFIELD.PATTERN_MODE = "random"

    dataset = BasinLabeledDataset(
        system="hopfield",
        cfg=cfg,
        num_trajectories=16,
        trajectory_length=20,
        long_rollout_steps=40,
        seed=3,
    )

    assert dataset.num_basins == 3
    assert len(dataset.basin_names) == 3
    assert all(0 <= item.final_basin < dataset.num_basins for item in dataset.trajectories)


def test_kuramoto_basin_dataset_remaps_winding_labels():
    cfg = Config()
    cfg.SEED = 0
    cfg.ENV.ENV_NAME = "kuramoto"
    cfg.ENV.KURAMOTO.NUM_OSCILLATORS = 8
    cfg.ENV.KURAMOTO.OMEGA_MODE = "identical"
    cfg.ENV.KURAMOTO.TOPOLOGY = "ring"

    dataset = BasinLabeledDataset(
        system="kuramoto",
        cfg=cfg,
        num_trajectories=20,
        trajectory_length=25,
        long_rollout_steps=60,
        seed=5,
    )

    assert dataset.num_basins >= 1
    assert len(dataset.basin_names) == dataset.num_basins
    assert all(name.startswith("Winding q=") for name in dataset.basin_names)
    assert all(0 <= item.final_basin < dataset.num_basins for item in dataset.trajectories)
    assert all(item.raw_final_basin is not None for item in dataset.trajectories)
    assert sorted(dataset.raw_to_mapped_label.keys()) == dataset.raw_basin_labels


class _FakeBasinEnv:
    num_basins = 0

    def reset(self, rng):
        return torch.zeros(1, dtype=torch.float32)

    def step(self, state):
        return state

    def basin_label(self, state):
        return 0


def test_kuramoto_balanced_sampling_preserves_raw_labels(monkeypatch):
    cfg = Config()
    cfg.SEED = 0
    cfg.ENV.ENV_NAME = "kuramoto"

    labels = iter([0, 2, 0, 2])

    monkeypatch.setattr("skae.basin_utils.make_env", lambda cfg: _FakeBasinEnv())
    monkeypatch.setattr(
        "skae.basin_utils.generate_trajectory",
        lambda step_fn, init_state, length: init_state.repeat(length, 1),
    )
    monkeypatch.setattr(
        "skae.basin_utils.identify_env_basin",
        lambda env, trajectory, long_rollout_steps: next(labels),
    )

    dataset = BasinLabeledDataset(
        system="kuramoto",
        cfg=cfg,
        num_trajectories=99,
        trajectory_length=8,
        long_rollout_steps=10,
        seed=11,
        sampling_strategy="balanced",
        target_raw_labels=[0, 2],
        trajectories_per_basin=2,
        max_attempts=4,
    )

    assert dataset.num_trajectories == 4
    assert dataset.num_basins == 2
    assert dataset.raw_basin_labels == [0, 2]
    assert dataset.raw_to_mapped_label == {0: 0, 2: 1}
    assert dataset.raw_basin_distribution == {0: 2, 2: 2}
    assert dataset.mapped_basin_distribution == {0: 2, 1: 2}
    assert [traj.raw_final_basin for traj in dataset.trajectories] == [0, 2, 0, 2]
    assert [traj.final_basin for traj in dataset.trajectories] == [0, 1, 0, 1]


def test_kuramoto_balanced_sampling_raises_when_quota_missing(monkeypatch):
    cfg = Config()
    cfg.SEED = 0
    cfg.ENV.ENV_NAME = "kuramoto"

    monkeypatch.setattr("skae.basin_utils.make_env", lambda cfg: _FakeBasinEnv())
    monkeypatch.setattr(
        "skae.basin_utils.generate_trajectory",
        lambda step_fn, init_state, length: init_state.repeat(length, 1),
    )
    monkeypatch.setattr(
        "skae.basin_utils.identify_env_basin",
        lambda env, trajectory, long_rollout_steps: 0,
    )

    try:
        BasinLabeledDataset(
            system="kuramoto",
            cfg=cfg,
            num_trajectories=10,
            trajectory_length=8,
            long_rollout_steps=10,
            seed=11,
            sampling_strategy="balanced",
            target_raw_labels=[0, 1],
            trajectories_per_basin=1,
            max_attempts=3,
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected balanced Kuramoto sampling to raise")

    assert "missing" in message
    assert "1" in message


def test_competitive_lv_basin_dataset_uses_support_masks():
    cfg = Config()
    cfg.SEED = 0
    cfg.ENV.ENV_NAME = "competitive_lv"
    cfg.ENV.COMPETITIVE_LV.NUM_SPECIES = 6
    cfg.ENV.COMPETITIVE_LV.INTERACTION_MODE = "symmetric"

    dataset = BasinLabeledDataset(
        system="competitive_lv",
        cfg=cfg,
        num_trajectories=18,
        trajectory_length=20,
        long_rollout_steps=40,
        seed=7,
    )

    assert dataset.num_basins >= 1
    assert len(dataset.basin_names) == dataset.num_basins
    assert all(name.startswith("Survivor mask ") for name in dataset.basin_names)
    assert all(0 <= item.final_basin < dataset.num_basins for item in dataset.trajectories)
