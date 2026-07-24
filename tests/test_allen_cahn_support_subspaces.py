from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.neurips_2026.allen_cahn_support_subspaces.evaluation_helpers import (
    REQUIRED_SOURCE_PATHS,
    family_summary,
    forecast_kernel_discrepancy,
    historical_forecast_reproduction_metrics,
    load_profile_decision,
    matrix_for_row_vectors,
    verify_forecast_reproduction,
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.family_reduction import (
    family_decision,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.io import (
    assert_no_forbidden_mapping_access,
    load_card,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.metrics import (
    closure_metrics,
    decoded_rollout_metrics,
    fit_codebook,
    matched_topk_masks,
    matrix_leakage_metrics,
    operator_distance,
    operator_signature_distance,
    ordinary_permutations,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.reduction_statistics import (
    exact_max_t_adjusted_p,
    finite_tree,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.summarize import select_decision
from experiments.neurips_2026.allen_cahn_support_subspaces.select_profile import read_telemetry


class ToyKoopman(torch.nn.Module):
    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.kmat = torch.nn.Parameter(matrix.clone())

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def step_latent(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.kmat)

    def rollout_observation_discrete(self, x0: torch.Tensor, *, horizon: int):
        states = []
        state = self.encode(x0)
        for _ in range(horizon):
            state = self.step_latent(state)
            states.append(state)
        latent = torch.stack(states, dim=1)
        decoded = self.decode(latent.reshape(-1, latent.shape[-1])).reshape_as(latent)
        return latent, decoded


class BatchShapeToy(ToyKoopman):
    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value + value.shape[0] * 1e-6


def test_row_operator_uses_transpose_for_nonsymmetric_k() -> None:
    model = ToyKoopman(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    row_operator = matrix_for_row_vectors(model)
    expected = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    probe = torch.tensor([[5.0, 7.0]])
    assert torch.equal(row_operator, expected)
    assert torch.equal(model.step_latent(probe), probe @ expected)


def test_decoded_rollout_modes_have_frozen_semantics() -> None:
    model = ToyKoopman(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    fields = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]])
    result = decoded_rollout_metrics(
        model, fields, torch.tensor([[True, False]]), horizons=[1, 2], batch_size=1
    )
    assert result["1"]["full"]["field_mse"] == pytest.approx(73.0)
    assert result["1"]["mask_once"]["field_mse"] == pytest.approx(5.0)
    assert result["1"]["restricted"]["field_mse"] == pytest.approx(0.5)
    assert result["2"]["full"]["field_mse"] == pytest.approx(1089.0)
    assert result["2"]["mask_once"]["field_mse"] == pytest.approx(71.0)
    assert result["2"]["restricted"]["field_mse"] == pytest.approx(0.5)
    assert result["2"]["full"]["terminal_field_mse"] == pytest.approx(2105.0)


def test_identity_projector_has_zero_closure_leakage() -> None:
    latents = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2) + 1.0
    masks = torch.ones(3, 2, dtype=torch.bool)
    matrix = torch.tensor([[1.0, 0.2], [0.3, 1.0]])
    result = closure_metrics(latents, masks, matrix, horizon=2, state_batch_size=4)
    matrix_result = matrix_leakage_metrics(masks, matrix)
    assert result["activity_k_leakage_rms"] == 0.0
    assert result["activity_kminusI_leakage_rms"] == 0.0
    assert matrix_result["matrix_k_leakage_fro"] == 0.0
    assert matrix_result["matrix_kminusI_leakage_fro"] == 0.0


def test_off_block_operator_has_positive_matrix_leakage() -> None:
    matrix = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
    mask = torch.tensor([[True, False]])
    result = matrix_leakage_metrics(mask, matrix)
    assert result["matrix_k_leakage_fro"] == pytest.approx(2.0 / np.sqrt(5.0))
    assert result["matrix_kminusI_leakage_fro"] == pytest.approx(1.0)


def test_signature_sees_same_law_up_to_orthogonal_similarity() -> None:
    first = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    angle = np.pi / 4.0
    rotation = torch.tensor([
        [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]
    ], dtype=torch.float64)
    second = rotation.T @ first @ rotation
    change = torch.block_diag(first, second)
    matrix = torch.eye(4, dtype=torch.float64) + change
    supports = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool)
    assert operator_signature_distance(supports, matrix) == pytest.approx(0.0, abs=1e-12)
    assert operator_distance(supports, matrix) > 0.0


def test_synthetic_distinct_signatures_and_correct_routing_have_power() -> None:
    symmetric = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)
    skew = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    background = torch.diag(torch.linspace(0.02, 0.07, 6, dtype=torch.float64))
    first_block = torch.block_diag(symmetric, background)
    second_block = torch.block_diag(skew, background)
    change = torch.zeros(64, 64, dtype=torch.float64)
    change[:8, :8] = first_block
    change[8:16, 8:16] = second_block
    change[16:, 16:] = 0.04 * torch.eye(48, dtype=torch.float64)
    matrix = torch.eye(64, dtype=torch.float64) + change
    supports = torch.zeros(2, 64, dtype=torch.bool)
    supports[0, :8] = True
    supports[1, 8:16] = True
    assert matrix_leakage_metrics(supports, matrix)["matrix_kminusI_leakage_fro"] == 0.0
    observed = operator_signature_distance(supports, matrix)
    assert observed is not None and observed > 0.0
    null = [
        operator_signature_distance(supports[:, torch.as_tensor(permutation)], matrix)
        for permutation in ordinary_permutations(64, 16, 20260727)
    ]
    null_median = float(np.median(null))
    assert np.isfinite(null_median) and null_median > 0.0
    assert observed / null_median >= 1.1

    model = ToyKoopman(matrix.T.float())
    x0 = torch.zeros(2, 64)
    x0[0, :2] = torch.tensor([1.0, 0.5])
    x0[1, 8:10] = torch.tensor([1.0, -0.5])
    step1 = x0 @ matrix.float()
    step2 = step1 @ matrix.float()
    fields = torch.stack((x0, step1, step2), dim=1)
    correct = decoded_rollout_metrics(
        model, fields, supports, horizons=[2], batch_size=2
    )
    wrong = decoded_rollout_metrics(
        model, fields, supports.flip(0), horizons=[2], batch_size=2
    )
    assert correct["2"]["restricted"]["field_mse"] == pytest.approx(0.0, abs=1e-12)
    assert wrong["2"]["restricted"]["field_mse"] > 0.0


def test_synthetic_off_block_mixing_cannot_reach_low_leakage_branch() -> None:
    matrix = torch.eye(4)
    matrix[0, 2] = 3.0
    matrix[2, 0] = -3.0
    supports = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool)
    leakage = matrix_leakage_metrics(supports, matrix)["matrix_kminusI_leakage_fro"]
    assert leakage is not None and leakage > 0.5
    decision = select_decision(
        validity=True,
        exact_closure={"passed": False, "activity_weighted_passed": True},
        forecast={"passed": True, "projected_vs_dense_full_passed": True},
        family={
            "family_passed": True,
            "signature_differentiation_passed": True,
            "routing_specificity_passed": True,
        },
    )
    assert decision == "activity_weighted_closure_only"


def test_identity_k_undefined_change_metric_is_fail_closed() -> None:
    latents = torch.ones(1, 2, 2)
    result = closure_metrics(
        latents, torch.tensor([[True, False]]), torch.eye(2), horizon=1,
        state_batch_size=1,
    )
    assert result["activity_kminusI_leakage_rms"] is None
    assert not finite_tree(result)


def test_matched_topk_is_cardinality_exact_and_stable_on_ties() -> None:
    dense = np.asarray([[3.0, -3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    sparse = np.asarray([[True, False, True, False], [True, True, True, False]])
    masks = matched_topk_masks(dense, sparse)
    assert np.array_equal(masks.sum(axis=1), sparse.sum(axis=1))
    assert np.array_equal(masks[0], [True, True, False, False])
    assert np.array_equal(masks[1], [True, True, True, False])


def test_coordinate_permutations_preserve_cardinality_and_are_reproducible() -> None:
    first = ordinary_permutations(8, 4, 123)
    second = ordinary_permutations(8, 4, 123)
    mask = np.asarray([True, False, True, False, False, True, False, False])
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert all(mask[permutation].sum() == mask.sum() for permutation in first)


def test_codebook_is_deterministic_and_capped_without_forced_assignment() -> None:
    masks = np.eye(4, dtype=bool)
    first = fit_codebook(masks, min_jaccard=0.4, max_representatives=2, min_fit_count=1)
    second = fit_codebook(masks, min_jaccard=0.4, max_representatives=2, min_fit_count=1)
    assert np.array_equal(first.representatives, second.representatives)
    assert first.representatives.shape == (2, 4)
    assert first.fit_counts.sum() == 2


def test_family_pair_is_frozen_by_train_counts_not_score_frequency() -> None:
    card, _ = load_card()
    card = copy.deepcopy(card)
    card["support"]["min_fit_trajectories"] = 1
    card["support"]["min_score_trajectories_for_per_family_estimates"] = 1
    card["family_gates"]["minimum_score_coverage"] = 0.0
    card["family_gates"]["minimum_qualified_family_coverage"] = 0.0
    train = np.asarray([[1, 0, 0]] * 5 + [[0, 1, 0]] * 3 + [[0, 0, 1]], dtype=bool)
    score_a = np.asarray([[0, 1, 0]] * 8 + [[1, 0, 0]] + [[0, 0, 1]], dtype=bool)
    score_b = np.asarray([[1, 0, 0]] * 8 + [[0, 1, 0]] + [[0, 0, 1]], dtype=bool)
    summary_a, _, reps_a, _ = family_summary(train, score_a, card)
    summary_b, _, reps_b, _ = family_summary(train, score_b, card)
    assert np.array_equal(reps_a, reps_b)
    assert summary_a["fit_frozen_top_two_family_indices"] == [0, 1]
    assert summary_b["fit_frozen_top_two_family_indices"] == [0, 1]


def test_empty_family_reduction_fails_without_bootstrap_crash() -> None:
    card, _ = load_card()
    result = family_decision([], card, closure_reducer=lambda *_args, **_kwargs: {})
    assert not result["family_passed"]
    assert not result["routing_specificity_passed"]
    assert all(not cell["passed"] for cell in result["routing_specificity"].values())


def test_exact_sign_flip_uses_count_over_full_enumeration() -> None:
    values = [float(index) for index in range(1, 11)]
    result = exact_max_t_adjusted_p({"cell": values})
    assert result["cell"] == pytest.approx(1.0 / 1024.0)


def test_full_forecast_reproduction_is_fail_closed() -> None:
    card, _ = load_card()
    forecasts = {
        arm: {
            str(horizon): {
                "full": {"field_mse": 0.1, "terminal_field_mse": 0.2}
            }
            for horizon in card["roster"]["horizons"]
        }
        for arm in card["roster"]["arms"]
    }
    references = {
        (arm, 64, int(horizon)): {"field_mse": 0.1, "terminal_field_mse": 0.2}
        for arm in card["roster"]["arms"] for horizon in card["roster"]["horizons"]
    }
    assert verify_forecast_reproduction(
        forecasts, references, seed=64, card=card
    )["passed"]
    references[("sparse", 64, 200)]["field_mse"] = 1.0
    with pytest.raises(AssertionError):
        verify_forecast_reproduction(forecasts, references, seed=64, card=card)


def test_historical_provenance_kernel_is_separate_from_three_mode_kernel() -> None:
    model = BatchShapeToy(torch.eye(2))
    fields = torch.zeros(2, 3, 2, dtype=torch.float32)
    historical = historical_forecast_reproduction_metrics(
        model, fields, horizons=[2], batch_size=2
    )
    scientific = decoded_rollout_metrics(
        model, fields, torch.ones(2, 2, dtype=torch.bool),
        horizons=[2], batch_size=2,
    )
    assert historical["2"]["full"]["field_mse"] == pytest.approx(16e-12)
    assert scientific["2"]["full"]["field_mse"] == pytest.approx(36e-12)
    discrepancy = forecast_kernel_discrepancy(scientific, historical)
    assert discrepancy["descriptive_only_not_a_scientific_gate"] is True
    assert discrepancy["maximum_relative_difference"] > 0.0


def test_decision_precedence_never_calls_activity_only_invariant() -> None:
    forecast = {"passed": True, "projected_vs_dense_full_passed": True}
    family = {
        "family_passed": True,
        "signature_differentiation_passed": True,
        "routing_specificity_passed": True,
    }
    assert select_decision(
        validity=True,
        exact_closure={"passed": False, "activity_weighted_passed": True},
        forecast=forecast,
        family=family,
    ) == "activity_weighted_closure_only"
    family["routing_specificity_passed"] = False
    assert select_decision(
        validity=True,
        exact_closure={"passed": True, "activity_weighted_passed": True},
        forecast=forecast,
        family=family,
    ) == "distinct_signatures_without_routing_specificity"


def test_information_firewall_and_route_lock_order() -> None:
    card, _ = load_card()
    assert_no_forbidden_mapping_access(["fields", "split_indices"], card)
    with pytest.raises(AssertionError):
        assert_no_forbidden_mapping_access(["basin_labels"], card)
    source = Path(
        "experiments/neurips_2026/allen_cahn_support_subspaces/evaluate.py"
    ).read_text()
    assert source.index("historical_forecasts =") < source.index("train_z0 =")
    assert source.index("score_z0 =") < source.index("routing objects are now locked")
    assert source.index("routing objects are now locked") < source.index("future = encode_states")


def test_card_and_active_files_obey_protocol_guards() -> None:
    card, _ = load_card()
    assert card["inputs"]["training_dataset"]["expected_selected_trajectories"] == 512
    assert card["inputs"]["score_dataset"]["expected_selected_trajectories"] == 256
    strong = card["decision_branches"][
        "strong_routed_low_leakage_charts_with_distinct_signatures"
    ]
    assert "wrong-family" in strong
    canary = Path("experiments/neurips_2026/allen_cahn_support_subspaces/validate_canary.py").read_text()
    assert "json.loads(shard_path" not in canary and all(f'shard["{key}"]' not in canary for key in ("closure", "forecast", "family", "sparse_family"))
    queue = Path("scripts/neurips_2026/allen_cahn_support_subspaces/queue_science.sh").read_text()
    assert "--array=0" in queue and "--array=1-9%8" in queue and "CANARY_CHECK_JOB" in queue
    directory = Path("experiments/neurips_2026/allen_cahn_support_subspaces")
    assert all(
        len(path.read_text().splitlines()) <= 500
        for path in directory.iterdir() if path.is_file()
    )


def test_prediction_card_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version": 1, "schema_version": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_card(duplicate)


def test_source_manifest_requires_exact_roster_and_hashes(tmp_path: Path) -> None:
    lines = [f"{sha256_path(Path(path))}  {path}" for path in sorted(REQUIRED_SOURCE_PATHS)]
    manifest = tmp_path / "source_manifest.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    verify_source_manifest(manifest)
    manifest.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="roster mismatch"):
        verify_source_manifest(manifest)
    bad = lines.copy()
    bad[0] = "0" * 64 + "  " + sorted(REQUIRED_SOURCE_PATHS)[0]
    manifest.write_text("\n".join(bad) + "\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_source_manifest(manifest)


def test_repository_source_manifest_and_queue_roots_of_trust() -> None:
    manifest = Path(
        "experiments/neurips_2026/allen_cahn_support_subspaces/source_manifest.sha256"
    )
    manifest_hash = verify_source_manifest(manifest)
    card_path = Path(
        "experiments/neurips_2026/allen_cahn_support_subspaces/prediction_card.json"
    )
    card_hash = sha256_path(card_path)
    scripts = Path("scripts/neurips_2026/allen_cahn_support_subspaces")
    for name in ("queue_profile.sh", "queue_science.sh"):
        source = (scripts / name).read_text()
        assert f"EXPECTED_CARD_SHA256={card_hash}" in source
        assert f"EXPECTED_SOURCE_MANIFEST_SHA256={manifest_hash}" in source


def test_telemetry_rejects_multiple_gpu_uuids_and_workers_pin_selector(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "GPU-a, 2026/01/01, 90, 100, 1000\n"
        "GPU-b, 2026/01/01, 90, 100, 1000\n"
    )
    with pytest.raises(RuntimeError, match="multiple GPU UUIDs"):
        read_telemetry(telemetry)
    scripts = Path("scripts/neurips_2026/allen_cahn_support_subspaces")
    for name in ("run_profile.sh", "run_array.sh"):
        source = (scripts / name).read_text()
        assert "ALLOCATED_GPU_IDS=${SLURM_JOB_GPUS:?missing SLURM_JOB_GPUS}" in source
        assert '[[ "${ALLOCATED_GPU_IDS}" != *,* ]]' in source
        assert "mapfile -t VISIBLE_GPU_UUIDS" in source
        assert "nvidia-smi --query-gpu=uuid --format=csv,noheader" in source
        assert '[[ "${#VISIBLE_GPU_UUIDS[@]}" -eq 1 ]]' in source
        assert "GPU_SELECTOR=${VISIBLE_GPU_UUIDS[0]}" in source
        assert '[[ "${GPU_SELECTOR}" == GPU-* ]]' in source
        assert 'nvidia-smi -i "${GPU_SELECTOR}"' in source
        assert "GPU_SELECTOR=${SLURM_JOB_GPUS" not in source


def test_profile_decision_is_independently_recomputed_and_smallest(tmp_path: Path) -> None:
    card, card_hash = load_card()
    source_hash = "a" * 64
    candidates = []
    for batch_size in card["hardware_profile"]["candidate_batch_sizes"]:
        profile_path = tmp_path / f"batch_{batch_size}.json"
        telemetry_path = tmp_path / f"batch_{batch_size}_nvidia_smi.csv"
        telemetry_path.write_text("".join(
            f"GPU-test, 2026/01/01, 90, 1000, 81920\n" for _ in range(25)
        ))
        telemetry = read_telemetry(telemetry_path)
        profile = {
            "status": "completed",
            "synthetic_inputs_only": True,
            "outcomes_accessed": False,
            "datasets_opened": False,
            "batch_size": batch_size,
            "profile_seconds": 50.1,
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "slurm_job_id": "123",
            "slurm_job_gpus": "0",
            "visible_cuda_device_count": 1,
            "device_name": "NVIDIA A100-SXM4-80GB",
            "device_uuid": "GPU-test",
            "resident_model_count": 2,
            "closure_state_batch_size": 8192,
            "historical_provenance_kernel_profiled": True,
            "historical_reproduction_batch_size": 128,
            "historical_reproduction_horizons": [80, 120, 160, 200],
            "device_total_memory_bytes": 81920 * 1024 * 1024,
            "peak_reserved_bytes": 10 * 1024 * 1024 * 1024,
        }
        profile_path.write_text(json.dumps(profile))
        gates = {
            "integrity": True,
            "duration": True,
            "active_samples": True,
            "mean_active_utilization": True,
            "mean_all_utilization": True,
            "peak_memory": True,
            "profile_peak_memory": True,
        }
        candidates.append({
            "batch_size": batch_size,
            "passed": True,
            "gates": gates,
            "profile": profile,
            "telemetry": telemetry,
            "profile_filename": profile_path.name,
            "telemetry_filename": telemetry_path.name,
            "profile_sha256": sha256_path(profile_path),
            "telemetry_sha256": sha256_path(telemetry_path),
        })
    decision = {
        "status": "passed",
        "selected_batch_size": 128,
        "selection_rule": "smallest passing frozen candidate",
        "synthetic_inputs_only": True,
        "outcomes_quarantined": True,
        "telemetry_interval_seconds": 2,
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "candidate_batch_sizes": [128, 256],
        "candidates": candidates,
    }
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision))
    load_profile_decision(
        decision_path, 128, card=card, card_hash=card_hash,
        source_manifest_hash=source_hash,
    )
    decision["selected_batch_size"] = 256
    decision_path.write_text(json.dumps(decision))
    with pytest.raises(RuntimeError, match="independent contract validation"):
        load_profile_decision(
            decision_path, 256, card=card, card_hash=card_hash,
            source_manifest_hash=source_hash,
        )
