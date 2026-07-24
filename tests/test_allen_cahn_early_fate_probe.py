from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.neurips_2026.allen_cahn_early_fate_probe.features import (
    field_summary,
    matched_topk_masks,
    modal_well_labels,
    well_area_fractions,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe.io import (
    CARD_PATH,
    REPO_ROOT,
    load_card,
    load_task_manifest,
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe.probes import (
    fit_probe,
    require_class_counts,
    stratified_folds,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe.statistics import (
    contrast_summary,
    holm_adjust,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe.reduction_utils import (
    relative_pass,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe.telemetry import (
    summarize_scope,
)


def test_card_task_and_source_lock_are_exact() -> None:
    card, _ = load_card()
    task, task_sha = load_task_manifest(card)
    assert task_sha == card["inputs"]["task_manifest"]["sha256"]
    assert task["gpu_tasks"][0]["model_seeds"] == list(range(64, 74))
    assert task["gpu_tasks"][0]["dataset_seeds"] == [1775404171, 74732421, 293789188]
    verify_source_manifest(card)
    text = "\n".join(
        (REPO_ROOT / path).read_text()
        for path in card["source_lock"]["required_manifest_paths"]
        if str(path).endswith((".py", ".sh", ".json"))
    )
    assert ("TASK_MANIFEST_SHA256_" + "PLACEHOLDER") not in text
    assert ("distinct_laws_" + "v2") not in text.lower()


def test_card_has_redteam_claim_boundaries() -> None:
    card = json.loads(CARD_PATH.read_text())
    assert card["validity"]["minimum_test_count_per_class_per_dataset"] >= 32
    assert card["validity"]["require_all_four_classes_in_every_training_fold"]
    assert card["primary_gate"]["minimum_model_seed_wins"] == 9
    assert card["primary_gate"]["minimum_dataset_seed_wins"] == 3
    assert set(card["primary_gate"]["claim_tiers"]) == {
        "support_information",
        "coordinate_identity",
        "better_than_dense_representation",
        "accessibility_beyond_physical_state",
        "future_information_beyond_initial_occupancy",
    }
    assert "conditional robustness" in card["statistics"]["two_way_bootstrap"]["estimand"]


def test_feature_definitions_and_stable_topk() -> None:
    fields = torch.zeros(4, 512)
    reshaped = fields.reshape(4, 16, 16, 2)
    reshaped[0, :, :, 0] = 1.5
    reshaped[1, :, :, 1] = 1.5
    reshaped[2, :, :, 0] = -1.5
    reshaped[3, :, :, 1] = -1.5
    assert modal_well_labels(fields).tolist() == [0, 1, 2, 3]
    assert torch.allclose(well_area_fractions(fields), torch.eye(4, dtype=torch.float64))
    assert field_summary(fields).shape == (4, 11)
    dense = np.array([[2.0, -2.0, 1.0, 0.0], [1.0, 3.0, 2.0, 4.0]])
    sparse = np.array([[True, False, True, False], [False, False, False, True]])
    masks = matched_topk_masks(dense, sparse)
    assert masks.tolist() == [[True, True, False, False], [False, False, False, True]]
    assert np.array_equal(masks.sum(1), sparse.sum(1))


def test_class_and_fold_gates_fail_closed() -> None:
    with pytest.raises(ValueError):
        require_class_counts(np.array([0] * 40 + [1] * 40 + [2] * 40), minimum=1)
    labels = np.repeat(np.arange(4), 10)
    folds = stratified_folds(labels, n_splits=5, seed=20260721)
    assert len(folds) == 5
    for fold in folds:
        assert set(labels[fold]) == {0, 1, 2, 3}


def test_probe_selection_is_training_only_and_test_estimand_is_fixed() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(4), 12)
    features = np.eye(4)[labels] + 0.05 * rng.normal(size=(48, 4))
    test_labels = np.repeat(np.arange(4), 8)
    test_features = np.eye(4)[test_labels]
    result = fit_probe(
        features,
        labels,
        [test_features],
        [test_labels],
        alphas=[0.01, 1.0, 100.0],
        n_splits=3,
        split_seed=9,
        minimum_test_count=8,
    )
    permuted = np.roll(test_labels, 1)
    second = fit_probe(
        features,
        labels,
        [test_features],
        [permuted],
        alphas=[0.01, 1.0, 100.0],
        n_splits=3,
        split_seed=9,
        minimum_test_count=8,
    )
    assert result.alpha == second.alpha
    assert result.cv_scores == second.cv_scores
    with pytest.raises(ValueError):
        fit_probe(
            features,
            labels,
            [test_features[:24]],
            [test_labels[:24]],
            alphas=[1.0],
            n_splits=3,
            split_seed=9,
            minimum_test_count=8,
        )


def test_nine_model_and_three_dataset_gate_is_not_relaxed() -> None:
    differences = np.full((10, 3), 0.10)
    differences[-2:, :] = -0.01
    differences[:, -1] -= 0.20
    summary = contrast_summary(differences, bootstrap_replicates=1000, bootstrap_seed=3)
    assert summary["model_seed_wins"] <= 8 or summary["dataset_seed_wins"] <= 2
    assert not relative_pass(summary, 0.01)
    assert holm_adjust([0.01, 0.02, 0.03, 0.04]) == pytest.approx(
        [0.04, 0.06, 0.06, 0.06]
    )


def test_telemetry_reports_allocation_and_active_utilization() -> None:
    records = [
        {
            "epoch": float(index),
            "uuid": "gpu",
            "name": "A100",
            "utilization": value,
            "memory_used": 10.0,
            "memory_total": 100.0,
        }
        for index, value in enumerate([0.0, 90.0, 100.0, 80.0, 0.0])
    ]
    result = summarize_scope(records, 0.0, 4.0)
    assert result["sample_count"] == 5
    assert result["active_sample_count"] == 3
    assert result["zero_utilization_fraction"] == pytest.approx(0.4)
    assert result["mean_active_gpu_utilization_percent"] == pytest.approx(90.0)
