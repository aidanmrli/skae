"""Tests for label-free local EDMD/Koopman baseline helpers."""

from __future__ import annotations

from argparse import Namespace

import numpy as np

from tools.build_paper_baseline_tasks import _build_rows
from tools.evaluate_local_edmd_koopman_baselines import (
    _evaluate_rollout,
    _select_and_fit,
)


def _piecewise_affine_trajectories(num_per_side: int = 24, length: int = 8) -> np.ndarray:
    trajectories = []
    left_center = np.asarray([-1.0, 0.0], dtype=np.float64)
    right_center = np.asarray([1.0, 0.0], dtype=np.float64)
    left_a = np.asarray([[0.72, 0.18], [-0.08, 0.62]], dtype=np.float64)
    right_a = np.asarray([[0.45, -0.24], [0.16, 0.70]], dtype=np.float64)

    for index in range(num_per_side):
        offset = 0.02 * index
        state = left_center + np.asarray([-0.35 - offset, 0.15 + 0.01 * index])
        rows = [state.copy()]
        for _ in range(length):
            state = left_center + left_a @ (state - left_center)
            rows.append(state.copy())
        trajectories.append(np.stack(rows, axis=0))

    for index in range(num_per_side):
        offset = 0.02 * index
        state = right_center + np.asarray([0.35 + offset, -0.15 - 0.01 * index])
        rows = [state.copy()]
        for _ in range(length):
            state = right_center + right_a @ (state - right_center)
            rows.append(state.copy())
        trajectories.append(np.stack(rows, axis=0))

    return np.stack(trajectories, axis=0)


def test_validation_selected_local_edmd_prefers_two_piecewise_regimes():
    trajectories = _piecewise_affine_trajectories()

    model, selection = _select_and_fit(
        "local_edmd_poly_kmeans",
        trajectories,
        num_components_grid=[1, 2],
        validation_fraction=0.25,
        selection_horizons=[5],
        edmd_degree=1,
        kernel_centers=8,
        kernel_gamma=0.0,
        ridge_lambda=1e-8,
        max_train_pairs=0,
        min_component_transitions=1,
        max_abs_state_for_fit=1e6,
        seed=123,
    )

    assert selection["selected_num_components"] == 2
    assert model.selected_num_components == 2
    assert model.fitted_component_count == 2

    metrics = _evaluate_rollout(model, trajectories, [5])
    assert metrics[5]["finite_fraction"] == 1.0
    assert metrics[5]["cumulative_mse_mean"] is not None
    assert metrics[5]["cumulative_mse_mean"] < 1e-6


def test_paper_baseline_tasks_include_local_edmd_family_fields():
    args = Namespace(
        output_tsv="unused.tsv",
        output_manifest_json=None,
        systems="gated_local_linear",
        seeds="0,1",
        baseline_families="local_edmd_koopman",
        horizons="10,20",
        num_trajectories=32,
        trajectory_length=40,
        train_fraction=0.6,
        ridge_lambda=1e-6,
        edmd_degree=2,
        kernel_centers=16,
        kernel_gamma=0.0,
        max_train_pairs=128,
        num_components=4,
        component_mode="fixed",
        local_num_components_grid="1,2,4",
        local_validation_fraction=0.2,
        local_selection_horizons="10",
        local_min_component_transitions=8,
        local_max_abs_state_for_fit=123.0,
        env_dt=0.0,
        dysts_dt_multiplier=0.0,
        dysts_standardize=0,
        config_name="default",
        torch_threads=1,
    )

    rows = _build_rows(args)

    assert len(rows) == 2
    assert {row["baseline_family"] for row in rows} == {"local_edmd_koopman"}
    assert {
        row["methods"] for row in rows
    } == {"local_edmd_poly_kmeans,local_rbf_edmd_kmeans"}
    assert {row["local_num_components_grid"] for row in rows} == {"1,2,4"}
    assert {row["local_validation_fraction"] for row in rows} == {0.2}
    assert {row["local_selection_horizons"] for row in rows} == {"10"}
    assert {row["local_min_component_transitions"] for row in rows} == {8}
    assert {row["local_max_abs_state_for_fit"] for row in rows} == {123.0}
