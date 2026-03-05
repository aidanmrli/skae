"""Tests for basin labeling utilities on high-dimensional benchmark systems."""

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
