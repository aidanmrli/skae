"""Tests for the frozen controlled basin/support alignment reducer."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments.neurips_2026.alignment import (
    ENDPOINT_ROLLOUT_STEPS,
    ENTROPY_UNITS,
    FAMILY_JACCARD_THRESHOLD,
    FAMILY_COUNT_SEMANTICS,
    NATIVE_LABEL_SOURCE,
    OUTPUT_COLUMNS,
    PROXY_LABEL_SOURCE,
    SUPPORT_SCHEME,
    _scored_alignment_metrics,
    _label_sequences_and_centers,
    _assign_nearest_centers,
    _kmeans_centers,
    alignment_protocol_metadata,
    absolute_support_mask,
    conditional_entropy,
    support_family_labels,
    tie_inclusive_high_center_margin_mask,
)
from experiments.neurips_2026.workflows.alignment_reduction import (
    _load_latest_specs,
    _write_csv,
)
from experiments.neurips_2026.workflows import alignment_reduction as reducer


def test_active_output_schema_is_exact() -> None:
    assert SUPPORT_SCHEME == "absolute:0.001"
    assert FAMILY_JACCARD_THRESHOLD == 0.5
    assert OUTPUT_COLUMNS == (
        "root_label",
        "system_name",
        "seed",
        "support_scheme",
        "subset",
        "num_states",
        "observed_label_count",
        "family_jaccard_threshold",
        "family_h_basin_given_family",
        "family_unique_count",
    )


def test_alignment_protocol_records_label_margin_and_metric_semantics() -> None:
    protocol = alignment_protocol_metadata()
    assert protocol["num_evaluation_trajectories"] == 128
    assert protocol["trajectory_transitions"] == 128
    assert protocol["states_per_trajectory"] == 129
    assert protocol["evaluation_seed"] == 42
    assert protocol["proxy_basin_count_source"] == (
        "known_benchmark_count_for_evaluation_only"
    )
    assert protocol["proxy_endpoint_rollout_steps"] == ENDPOINT_ROLLOUT_STEPS == 5000
    assert protocol["center_margin_definition"] == (
        "second_nearest_center_distance_minus_nearest"
    )
    assert protocol["entropy_units"] == ENTROPY_UNITS == "nats"
    assert protocol["family_count_semantics"] == FAMILY_COUNT_SEMANTICS
    assert protocol["mask_visit_order"] == (
        "descending_frequency_then_ascending_packbits_bytes"
    )
    assert protocol["family_assignment_tie_break"] == "earliest_created_family"
    assert protocol["kmeans_farthest_tie_break"] == "first_endpoint_index"
    assert protocol["kmeans_assignment_tie_break"] == "first_center_index"
    assert protocol["kmeans_empty_cluster_rule"] == "retain_previous_center"


def test_native_gated_path_uses_env_labels_and_points() -> None:
    class NativeEnv:
        points = torch.tensor([[0.0], [5.0], [10.0]])

        @staticmethod
        def basin_label(states):
            return torch.cdist(states, NativeEnv.points).argmin(dim=1)

    trajectories = torch.tensor([[[0.0], [1.0]], [[9.0], [10.0]]])
    labels, centers, source = _label_sequences_and_centers(
        NativeEnv(),
        trajectories,
        system_key="gated_local_linear",
        endpoint_rollout_steps=0,
    )
    assert source == NATIVE_LABEL_SOURCE
    assert torch.equal(centers, NativeEnv.points)
    assert labels.tolist() == [[0, 0], [2, 2]]


def test_catalog_path_uses_known_count_endpoint_centers_and_proxy_labels() -> None:
    class ProxyEnv:
        @staticmethod
        def step(states):
            return states

    trajectories = torch.tensor(
        [[[0.0], [0.0]], [[5.0], [5.0]], [[10.0], [10.0]]]
    )
    labels, centers, source = _label_sequences_and_centers(
        ProxyEnv(),
        trajectories,
        system_key="claude:cal_asymmetric_3",
        endpoint_rollout_steps=0,
    )
    assert source == PROXY_LABEL_SOURCE
    assert centers.shape == (3, 1)
    assert sorted(centers[:, 0].tolist()) == [0.0, 5.0, 10.0]
    assert labels[:, 0].unique().numel() == 3


def test_absolute_support_uses_strict_paper_threshold() -> None:
    latents = np.asarray([[[-0.0011, -0.001, 0.0, 0.001, 0.0011]]])
    assert absolute_support_mask(latents).tolist() == [
        [[True, False, False, False, True]]
    ]


def test_support_family_labels_preserve_frozen_greedy_jaccard_rule() -> None:
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
    labels = support_family_labels(support_mask)
    assert labels[0, 0] == labels[0, 1]
    assert labels[0, 2] != labels[0, 0]


def test_support_family_ties_use_frequency_bytes_then_earliest_family() -> None:
    supports = np.asarray(
        [[
            [True, False],
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [True, True],
        ]],
        dtype=bool,
    )
    labels = support_family_labels(supports)
    assert labels[0, -1] == labels[0, 0]

    equal_frequency = np.asarray(
        [[[True, False], [False, True]]], dtype=bool
    )
    ordered = support_family_labels(equal_frequency)
    assert ordered.tolist() == [[1, 0]]


def test_kmeans_and_nearest_center_ties_use_first_index() -> None:
    points = torch.tensor([[0.0], [-1.0], [1.0]])
    initial = _kmeans_centers(points, num_centers=2, num_iters=0)
    assert initial.tolist() == [[0.0], [-1.0]]
    labels = _assign_nearest_centers(
        torch.tensor([[[0.0]]]), torch.tensor([[-1.0], [1.0]])
    )
    assert labels.item() == 0
    retained = _kmeans_centers(torch.zeros(3, 1), num_centers=2)
    assert retained.tolist() == [[0.0], [0.0]]


def test_high_center_margin_mask_keeps_margin_at_or_above_each_q75() -> None:
    states = torch.tensor(
        [[[0.0], [1.0], [2.0], [3.0]], [[6.0], [7.0], [8.0], [10.0]]]
    )
    centers = torch.tensor([[0.0], [10.0]])
    basins = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]])
    score_mask = tie_inclusive_high_center_margin_mask(states, centers, basins)
    assert score_mask.tolist() == [True, False, False, False, False, False, False, True]


def test_high_center_margin_mask_includes_all_q75_ties() -> None:
    states = torch.tensor([[[2.0], [2.0], [2.0], [2.0]]])
    centers = torch.tensor([[0.0], [10.0]])
    basins = np.asarray([[0, 0, 0, 0]])
    score_mask = tie_inclusive_high_center_margin_mask(states, centers, basins)
    assert score_mask.tolist() == [True, True, True, True]


def test_scored_alignment_metrics_match_conditional_entropy_definition() -> None:
    families = np.asarray([[0, 0], [0, 0]])
    basins = np.asarray([[0, 0], [1, 1]])
    deep = np.ones(4, dtype=bool)
    entropy, count = _scored_alignment_metrics(families, basins, deep)
    assert np.isclose(entropy, math.log(2.0))
    assert count == 1.0
    assert np.isclose(conditional_entropy([0, 0, 1, 1], [0, 0, 0, 0]), entropy)


def test_non_deep_states_influence_family_fit_before_deep_scoring() -> None:
    bridge = [True, True, True]
    left = [True, True, False]
    right = [False, True, True]
    supports = np.asarray([[bridge, bridge, bridge, left, right]], dtype=bool)
    basins = np.asarray([[0, 0, 0, 0, 1]])
    deep = np.asarray([False, False, False, True, True])

    all_state_families = support_family_labels(supports)
    all_fit_entropy, all_fit_count = _scored_alignment_metrics(
        all_state_families,
        basins,
        deep,
    )
    deep_only_families = support_family_labels(supports[:, -2:, :])
    deep_only_entropy, deep_only_count = _scored_alignment_metrics(
        deep_only_families,
        basins[:, -2:],
        np.ones(2, dtype=bool),
    )

    assert all_fit_count == 1.0
    assert np.isclose(all_fit_entropy, math.log(2.0))
    assert deep_only_count == 2.0
    assert np.isclose(deep_only_entropy, 0.0)


def test_latest_specs_select_newest_timestamped_run(tmp_path: Path) -> None:
    rows = tmp_path / "forecasting_rows.csv"
    rows.write_text(
        "root_label,system_key,system_name,seed,run_dir\n"
        "model,system,System,2,/runs/20260101-000000\n"
        "model,system,System,2,/runs/20260102-000000\n"
    )
    specs = _load_latest_specs(rows, ["model"], [], [])
    assert len(specs) == 1
    assert specs[0].run_dir == "/runs/20260102-000000"


def test_csv_writer_emits_schema_for_empty_shard(tmp_path: Path) -> None:
    output = tmp_path / "rows.csv"
    _write_csv(output, [])
    with output.open(newline="") as handle:
        reader = csv.reader(handle)
        assert tuple(next(reader)) == OUTPUT_COLUMNS
        assert list(reader) == []


def test_failed_run_flushes_failed_manifest_and_exits_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = "lista_dense_signsplit_p256_hardinit_basin_partition"
    rows = tmp_path / "forecasting_rows.csv"
    rows.write_text(
        "root_label,system_key,system_name,seed,run_dir\n"
        f"{root},gated_local_linear,gated_local_linear,0,{tmp_path / 'run'}\n"
    )
    output = tmp_path / "output"

    def fail_run(*args, **kwargs):
        raise RuntimeError("expected failure")

    monkeypatch.setattr(reducer, "reduce_run", fail_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reduce_transition_rich_interpretability_metrics.py",
            "--rows_csv",
            str(rows),
            "--output_dir",
            str(output),
            "--root_labels",
            root,
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        reducer.main()
    assert exc_info.value.code == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["num_failures"] == 1
    assert json.loads((output / "failures.json").read_text())[0][
        "error"
    ].startswith("RuntimeError")


def test_retired_reducer_surfaces_are_absent() -> None:
    source = Path(
        "experiments/neurips_2026/workflows/alignment_reduction.py"
    ).read_text() + Path("experiments/neurips_2026/alignment.py").read_text()
    for retired in (
        "relative_thresholds",
        "topk_values",
        "freeze_support",
        "support_projection",
        "operator_distance",
        "jacobian",
        "save_visuals",
    ):
        assert retired not in source
