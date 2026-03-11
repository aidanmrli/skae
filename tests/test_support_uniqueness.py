"""Tests for support-uniqueness metrics and degeneracy diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from skae.basin_utils import BasinLabeledTrajectory
from tools.evaluate_support_uniqueness import _support_from_latents, compute_support_uniqueness


class _DummyModel:
    def __init__(self, latents_by_index):
        self.latents_by_index = latents_by_index
        self.target_size = next(iter(latents_by_index.values())).shape[-1]

    def eval(self):
        return self

    def encode(self, states):
        return self.latents_by_index[int(states[0, 0].item())]


def test_modal_support_uses_most_common_timestep_pattern():
    latents = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    support = _support_from_latents(latents, threshold=0.5, mode="modal")

    assert support.tolist() == [1, 0, 0]


def test_compute_support_uniqueness_reports_mode_degeneracy_and_hamming_ratio():
    latents_by_index = {
        0: torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        1: torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        2: torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32),
        3: torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
    }
    model = _DummyModel(latents_by_index)
    dataset = SimpleNamespace(
        system="kuramoto",
        num_basins=2,
        sampling_strategy="balanced",
        raw_basin_labels=[-1, 1],
        raw_to_mapped_label={-1: 0, 1: 1},
        raw_basin_distribution={-1: 2, 1: 2},
        mapped_basin_distribution={0: 2, 1: 2},
        trajectories=[
            BasinLabeledTrajectory(
                states=torch.tensor([[0.0]], dtype=torch.float32),
                final_basin=0,
                raw_final_basin=-1,
            ),
            BasinLabeledTrajectory(
                states=torch.tensor([[1.0]], dtype=torch.float32),
                final_basin=0,
                raw_final_basin=-1,
            ),
            BasinLabeledTrajectory(
                states=torch.tensor([[2.0]], dtype=torch.float32),
                final_basin=1,
                raw_final_basin=1,
            ),
            BasinLabeledTrajectory(
                states=torch.tensor([[3.0]], dtype=torch.float32),
                final_basin=1,
                raw_final_basin=1,
            ),
        ],
    )

    result = compute_support_uniqueness(
        model,
        dataset,
        device="cpu",
        support_threshold=0.5,
        support_mode="mean",
    )

    assert result.unique_mode_supports == 2
    assert result.per_basin_mode_count == {0: 2, 1: 1}
    assert result.per_basin_mode_tie_count == {0: 1, 1: 2}
    assert result.per_basin_unique_support_count == {0: 1, 1: 2}
    assert result.trajectory_unique_support_count == 3
    assert result.trajectory_unique_support_rate == 0.75
    assert result.mean_within_basin_hamming == 0.5
    assert result.mean_between_basin_hamming == 2.5
    assert result.between_over_within_hamming_ratio == 5.0
