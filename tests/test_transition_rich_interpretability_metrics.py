"""Tests for the study-plan transition-rich interpretability reducer helpers."""

from __future__ import annotations

import numpy as np
import torch

from tools.reduce_transition_rich_interpretability_metrics import (
    _dominant_group_labels,
    canonical_support_masks_by_basin,
    conditional_entropy,
    dominant_support_mass_per_basin,
    freeze_support_rollout_metrics,
    jacobian_distance_summary,
    normalized_mutual_information,
    operator_distance_summary,
    support_family_labels,
    support_projection_metrics,
    support_transition_metrics,
    switch_timing_metrics,
)


def test_information_metrics_are_perfect_for_deterministic_support_basin_mapping():
    supports = [b"a", b"a", b"b", b"b"]
    basins = [0, 0, 1, 1]

    assert np.isclose(conditional_entropy(basins, supports), 0.0)
    assert np.isclose(conditional_entropy(supports, basins), 0.0)
    assert np.isclose(normalized_mutual_information(supports, basins), 1.0)
    assert np.isclose(dominant_support_mass_per_basin(supports, basins), 1.0)


def test_support_transition_metrics_separate_persistence_from_switch_alignment():
    support_mask = np.asarray(
        [
            [
                [True, False, False],
                [True, False, False],
                [False, True, False],
            ]
        ],
        dtype=bool,
    )
    support_keys = np.asarray([[b"a", b"a", b"b"]], dtype=object)
    basin_labels = np.asarray([[0, 0, 1]], dtype=np.int64)

    metrics = support_transition_metrics(support_mask, support_keys, basin_labels)

    assert np.isclose(metrics["support_persistence"], 1.0)
    assert np.isclose(metrics["support_switch_off_basin_switch"], 0.0)
    assert np.isclose(metrics["support_switch_on_basin_switch"], 1.0)
    assert np.isclose(metrics["basin_switch_fraction"], 0.5)
    assert np.isclose(metrics["support_jaccard_mean"], 1.0)


def test_operator_distance_summary_prefers_between_basin_separation():
    latents = np.asarray(
        [
            [[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            [[1.0, 0.0], [2.1, 0.0], [4.41, 0.0]],
            [[0.0, 1.0], [0.0, 3.0], [0.0, 9.0]],
        ],
        dtype=np.float64,
    )
    basin_labels = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    class_labels = np.asarray(
        [
            ["a1", "a1", "a1"],
            ["a2", "a2", "a2"],
            ["b1", "b1", "b1"],
        ],
        dtype=object,
    )

    metrics = operator_distance_summary(
        latents,
        basin_labels,
        class_labels,
        ridge_lambda=1e-6,
        min_transitions=2,
    )

    assert metrics["operator_class_count"] == 3.0
    assert metrics["operator_support_vs_basin_fro_mean"] is not None
    assert metrics["operator_within_basin_fro_mean"] is not None
    assert metrics["operator_between_basin_fro_mean"] is not None
    assert metrics["operator_between_basin_fro_mean"] > metrics["operator_within_basin_fro_mean"]
    assert metrics["operator_between_over_within"] is not None
    assert metrics["operator_between_over_within"] > 1.0


def test_dominant_group_labels_can_ignore_structured_global_block():
    latents = np.asarray(
        [
            [[9.0, 3.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 4.0, 0.0]],
        ],
        dtype=np.float64,
    )

    labels = _dominant_group_labels(latents, [2, 2], offset=1)

    assert labels.tolist() == [[0, 1]]


def test_support_family_labels_merge_nearby_exact_supports_by_jaccard():
    support_mask = np.asarray(
        [
            [
                [True, True, False, False],
                [True, True, True, False],
                [False, False, True, True],
            ]
        ],
        dtype=bool,
    )

    labels = support_family_labels(support_mask, min_jaccard=0.5)

    assert labels[0, 0] == labels[0, 1]
    assert labels[0, 2] != labels[0, 0]


def test_canonical_support_masks_by_basin_use_candidate_subset_only():
    support_mask = np.asarray(
        [
            [[True, False], [True, False], [False, True]],
            [[False, True], [False, True], [True, False]],
        ],
        dtype=bool,
    )
    basin_labels = np.asarray(
        [
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )
    candidate_mask = np.asarray([True, True, False, False, True, True], dtype=bool)

    canonical = canonical_support_masks_by_basin(support_mask, basin_labels, candidate_mask)

    assert canonical[0].tolist() == [True, False]
    assert canonical[1].tolist() == [False, True]


def test_support_projection_metrics_favor_own_basin_support():
    class IdentityModel:
        def step_latent(self, z: torch.Tensor) -> torch.Tensor:
            return z

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return z

    trajectories = torch.tensor(
        [
            [[1.0, 0.5], [1.0, 0.0]],
            [[0.25, 2.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    latents = trajectories.numpy()
    basin_labels = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
    support_templates = {
        0: np.asarray([True, False], dtype=bool),
        1: np.asarray([False, True], dtype=bool),
    }
    subset_mask = np.ones(trajectories.shape[:2], dtype=bool).reshape(-1)

    metrics = support_projection_metrics(
        IdentityModel(),
        latents,
        trajectories,
        basin_labels,
        support_templates,
        subset_mask,
        device="cpu",
    )

    assert metrics["support_projection_state_count"] == 2.0
    assert metrics["support_projection_base_mse"] is not None
    assert metrics["support_projection_self_mse"] is not None
    assert metrics["support_projection_wrong_mse"] is not None
    assert metrics["support_projection_self_mse"] < metrics["support_projection_base_mse"]
    assert metrics["support_projection_wrong_mse"] > metrics["support_projection_self_mse"]
    assert metrics["support_projection_self_over_base"] is not None
    assert metrics["support_projection_wrong_over_base"] is not None
    assert metrics["support_projection_self_over_base"] < 1.0
    assert metrics["support_projection_wrong_over_base"] > 1.0


def test_switch_timing_metrics_capture_delay_false_switches_and_chatter():
    class_labels = np.asarray(
        [
            ["a", "b", "a", "c", "c", "d"],
            ["u", "u", "u", "v", "v", "v"],
        ],
        dtype=object,
    )
    basin_labels = np.asarray(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 1, 2, 2, 2, 2],
        ],
        dtype=np.int64,
    )

    metrics = switch_timing_metrics(class_labels, basin_labels)

    assert metrics["switch_trajectory_count"] == 2.0
    assert metrics["switch_detected_fraction"] == 1.0
    assert metrics["switch_miss_fraction"] == 0.0
    assert metrics["switch_false_switches_mean"] == 1.0
    assert metrics["switch_delay_mean"] == 0.5
    assert metrics["switch_delay_abs_mean"] == 0.5
    assert metrics["switch_chatter_mean"] == 0.5
    assert metrics["switch_pre_dwell_mean"] == 2.0
    assert metrics["switch_post_dwell_mean"] == 2.5


def test_freeze_support_rollout_metrics_measure_longer_horizon_support_freeze():
    class IdentityModel:
        def step_latent(self, z: torch.Tensor) -> torch.Tensor:
            return z

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return z

    trajectories = torch.tensor(
        [
            [[1.0, 0.2], [1.0, 0.0], [1.0, 0.0]],
            [[0.25, 2.0], [0.0, 2.0], [0.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    latents = trajectories.numpy()
    basin_labels = np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.int64)
    support_templates = {
        0: np.asarray([True, False], dtype=bool),
        1: np.asarray([False, True], dtype=bool),
    }
    subset_mask = np.ones(trajectories.shape[:2], dtype=bool).reshape(-1)

    metrics = freeze_support_rollout_metrics(
        IdentityModel(),
        latents,
        trajectories,
        basin_labels,
        support_templates,
        subset_mask,
        device="cpu",
        horizons=(1, 2),
        max_states_per_horizon=16,
        sample_seed=0,
    )

    assert metrics["support_freeze_template_count"] == 2.0
    assert metrics["support_freeze_state_count_h1"] == 4.0
    assert metrics["support_freeze_state_count_h2"] == 2.0
    assert metrics["support_freeze_self_over_base_h1"] is not None
    assert metrics["support_freeze_self_over_base_h2"] is not None
    assert metrics["support_freeze_wrong_over_base_h1"] is not None
    assert metrics["support_freeze_wrong_over_base_h2"] is not None
    assert metrics["support_freeze_self_over_base_h1"] < 1.0
    assert metrics["support_freeze_self_over_base_h2"] < 1.0
    assert metrics["support_freeze_wrong_over_base_h1"] > 1.0
    assert metrics["support_freeze_wrong_over_base_h2"] > 1.0
    assert metrics["support_freeze_longest_horizon"] == 2.0
    assert metrics["support_freeze_longest_self_over_base"] is not None
    assert metrics["support_freeze_longest_self_over_base"] < 1.0


def test_jacobian_distance_summary_prefers_between_basin_and_can_compare_to_true():
    jacobians = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.1, 0.0], [0.0, 0.9]],
            [[3.0, 0.0], [0.0, 3.0]],
            [[2.9, 0.0], [0.0, 3.1]],
        ],
        dtype=np.float32,
    )
    true_jacobians = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[3.0, 0.0], [0.0, 3.0]],
            [[3.0, 0.0], [0.0, 3.0]],
        ],
        dtype=np.float32,
    )
    basin_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    class_labels = np.asarray(["a1", "a2", "b1", "b2"], dtype=object)

    metrics = jacobian_distance_summary(
        jacobians,
        basin_labels,
        class_labels,
        min_states=1,
        true_jacobians=true_jacobians,
    )

    assert metrics["jacobian_state_count"] == 4.0
    assert metrics["jacobian_class_count"] == 4.0
    assert metrics["jacobian_support_vs_basin_fro_mean"] is not None
    assert metrics["jacobian_within_basin_fro_mean"] is not None
    assert metrics["jacobian_between_basin_fro_mean"] is not None
    assert metrics["jacobian_between_basin_fro_mean"] > metrics["jacobian_within_basin_fro_mean"]
    assert metrics["jacobian_between_over_within"] is not None
    assert metrics["jacobian_between_over_within"] > 1.0
    assert metrics["jacobian_support_vs_true_fro_mean"] is not None
    assert metrics["jacobian_basin_vs_true_fro_mean"] is not None
