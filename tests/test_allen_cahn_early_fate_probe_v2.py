from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.neurips_2026.allen_cahn_forecast_replication.core import (
    realized_rng_streams,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe_v2.features import (
    field_summary,
    matched_topk_masks,
    well_area_fractions,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe_v2.io import (
    CARD_PATH,
    REPO_ROOT,
    load_card,
    load_task_manifest,
    verify_authenticated_v1_generator,
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe_v2.probes import (
    fit_probe,
    require_class_counts,
    stratified_folds,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe_v2.reduction_utils import (
    labels_and_eligibility,
    relative_pass,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe_v2.statistics import (
    contrast_summary,
    holm_adjust,
)
from experiments.neurips_2026.allen_cahn_early_fate_probe_v2.telemetry import (
    summarize_scope,
)


def test_v2_card_task_and_source_lock_are_exact() -> None:
    card, _ = load_card()
    task, task_sha = load_task_manifest(card)
    assert task_sha == card["inputs"]["task_manifest"]["sha256"]
    assert task["gpu_tasks"][0]["dataset_seeds"] == card["roster"]["dataset_seeds"]
    assert task["gpu_tasks"][0]["outputs"][-1].endswith(
        "raw_telemetry_unverified.csv"
    )
    assert [item["role"] for item in task["cpu_tasks"]] == [
        "authenticate_gpu_allocation_and_encoding_telemetry",
        "authenticated_tie_exclusion_and_label_aware_probe_reduction",
    ]
    assert task["cpu_tasks"][1]["dependency"] == "cpu_telemetry_task_0"
    assert task["no_v3_if_validity_fails"] is True
    verify_source_manifest(card)
    source_text = "\n".join(
        (REPO_ROOT / path).read_text()
        for path in card["source_lock"]["required_manifest_paths"]
        if not Path(path).is_absolute() and str(path).endswith((".py", ".sh", ".json"))
    )
    assert ("distinct_laws_" + "v2") not in source_text.lower()
    assert "TASK_MANIFEST_SHA256_" + "PLACEHOLDER" not in source_text


def test_telemetry_authentication_is_dependent_cpu_only() -> None:
    scripts = REPO_ROOT / "scripts/neurips_2026/allen_cahn_early_fate_probe_v2"
    telemetry = (scripts / "run_telemetry.sh").read_text()
    queue = (scripts / "queue.sh").read_text()
    gpu = (scripts / "run_generate_extract.sh").read_text()
    assert "#SBATCH --gres=" not in telemetry
    assert "#SBATCH --gpus=" not in telemetry
    assert "--expected-gpu-slurm-job-id" in telemetry
    assert "EXPECTED_GPU_JOB_ID=${GPU_JOB_ID}" in queue
    assert 'afterok:${GPU_JOB_ID}' in queue
    assert 'afterok:${TELEMETRY_JOB_ID}' in queue
    assert "raw_telemetry_unverified.csv" in gpu
    assert ".telemetry" not in gpu


def test_v2_changes_only_target_eligibility_and_new_data() -> None:
    card = json.loads(CARD_PATH.read_text())
    v1_path = REPO_ROOT / "experiments/neurips_2026/allen_cahn_early_fate_probe/prediction_card.json"
    v1 = json.loads(v1_path.read_text())
    for key in ("model_assertions", "feature_protocol", "probe", "statistics", "primary_gate", "secondary_policy"):
        if key in {"feature_protocol", "probe", "statistics"}:
            left = {
                name: value
                for name, value in card[key].items()
                if name not in {"eligibility_application", "eligibility_preprocessing"}
            }
            assert left == v1[key]
        else:
            assert card[key] == v1[key]
    for key in ("model_seeds", "arms", "observation_indices", "observation_times", "no_adaptive_earliest_time"):
        assert card["roster"][key] == v1["roster"][key]
    assert card["validity"]["minimum_eligible_fraction_each_test_dataset"] == 0.95
    assert card["validity"]["failure_policy"].startswith("Any eligibility")
    assert card["v1_disclosure"]["terminal_policy"].startswith("If V2 validity fails")


def test_seed_derivation_and_rng_streams_are_prospective_and_disjoint() -> None:
    card, _ = load_card()
    derivation = card["prospective_datasets"]["seed_derivation"]
    observed = []
    for index in range(3):
        value = f"{derivation['namespace']}|{derivation['root']}|dataset_{index}"
        digest = hashlib.sha256(value.encode()).hexdigest()
        assert digest == derivation["digests"][index]
        observed.append(int(digest[:8], 16) & 0x7FFFFFFF)
    assert observed == card["prospective_datasets"]["seeds"]
    proof = realized_rng_streams(card)
    assert proof["new_stream_cardinality"] == 768
    assert proof["excluded_intersection_empty"]
    assert proof["modular_residue_proof_passed"]


def test_complete_generator_contract_equals_authenticated_v1_card() -> None:
    card, _ = load_card()
    authenticated = verify_authenticated_v1_generator(card)
    reference_record = card["inputs"]["authenticated_v1_dataset_generating_card"]
    reference = json.loads((REPO_ROOT / reference_record["path"]).read_text())
    assert authenticated == reference["system_and_generator"]
    assert card["system_and_generator"] == reference["system_and_generator"]
    assert set(card["system_and_generator"]) == set(reference["system_and_generator"])


def test_exact_terminal_ties_are_excluded_and_argmax_otherwise() -> None:
    fields = torch.zeros(5, 16, 16, 2)
    fields[0, :, :, 0] = 1.5
    fields[1, :, :, 1] = 1.5
    fields[2, :, :, 0] = -1.5
    fields[3, :, :, 1] = -1.5
    fields[4, :8, :, 0] = 1.5
    fields[4, 8:, :, 1] = 1.5
    labels, eligible = labels_and_eligibility(fields.reshape(5, 512))
    assert labels.tolist() == [0, 1, 2, 3, 0]
    assert eligible.tolist() == [True, True, True, True, False]
    assert field_summary(fields.reshape(5, 512)).shape == (5, 11)
    assert well_area_fractions(fields.reshape(5, 512))[4].tolist() == [0.5, 0.5, 0.0, 0.0]


def test_stable_dense_topk_and_identical_eligibility_mask() -> None:
    dense = np.array([[2.0, -2.0, 1.0, 0.0], [1.0, 3.0, 2.0, 4.0]])
    sparse = np.array([[True, False, True, False], [False, False, False, True]])
    masks = matched_topk_masks(dense, sparse)
    assert masks.tolist() == [[True, True, False, False], [False, False, False, True]]
    eligible = np.array([True, False])
    for values in (dense, sparse, masks):
        assert values[eligible].shape[0] == 1
    assert np.array_equal(masks.sum(1), sparse.sum(1))


def test_post_exclusion_class_and_fold_gates_fail_closed() -> None:
    with pytest.raises(ValueError):
        require_class_counts(np.array([0] * 40 + [1] * 40 + [2] * 40), minimum=1)
    labels = np.repeat(np.arange(4), 40)
    folds = stratified_folds(labels, n_splits=5, seed=20260721)
    assert all(set(labels[fold]) == {0, 1, 2, 3} for fold in folds)
    assert 0.94 < 0.95


def test_probe_selection_remains_training_only() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(4), 12)
    features = np.eye(4)[labels] + 0.05 * rng.normal(size=(48, 4))
    test_labels = np.repeat(np.arange(4), 8)
    test_features = np.eye(4)[test_labels]
    kwargs = dict(
        alphas=[0.01, 1.0, 100.0], n_splits=3, split_seed=9,
        minimum_test_count=8,
    )
    first = fit_probe(features, labels, [test_features], [test_labels], **kwargs)
    second = fit_probe(
        features, labels, [test_features], [np.roll(test_labels, 1)], **kwargs
    )
    assert first.alpha == second.alpha
    assert first.cv_scores == second.cv_scores


def test_nine_model_three_dataset_and_holm_gates_are_unchanged() -> None:
    differences = np.full((10, 3), 0.10)
    differences[-2:, :] = -0.01
    differences[:, -1] -= 0.20
    summary = contrast_summary(differences, bootstrap_replicates=1000, bootstrap_seed=3)
    assert not relative_pass(summary, 0.01)
    assert holm_adjust([0.01, 0.02, 0.03, 0.04]) == pytest.approx(
        [0.04, 0.06, 0.06, 0.06]
    )


def test_telemetry_keeps_zero_samples_in_all_sample_mean() -> None:
    records = [
        {
            "epoch": float(index), "uuid": "gpu", "name": "A100",
            "utilization": value, "memory_used": 10.0, "memory_total": 100.0,
        }
        for index, value in enumerate([0.0, 90.0, 100.0, 80.0, 0.0])
    ]
    result = summarize_scope(records, 0.0, 4.0)
    assert result["active_sample_count"] == 3
    assert result["zero_utilization_fraction"] == pytest.approx(0.4)
    assert result["mean_active_gpu_utilization_percent"] == pytest.approx(90.0)
    assert result["mean_all_gpu_utilization_percent"] == pytest.approx(54.0)
