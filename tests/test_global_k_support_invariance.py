from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.neurips_2026.global_k_support_invariance import (
    assert_sign_split_layout,
    assign_families,
    evaluate_regime,
    fit_family_codebook,
    matrix_metrics,
    sign_pair_permutations,
    transition_metrics,
)
from experiments.neurips_2026.summarize_global_k_support_invariance import METRICS, decide


def _block_problem(repeats: int = 32):
    representatives = np.zeros((2, 8), dtype=bool)
    representatives[0, [0, 1, 4, 5]] = True
    representatives[1, [2, 3, 6, 7]] = True
    labels = np.repeat(np.arange(2), repeats)
    rng = np.random.default_rng(3)
    z = rng.normal(size=(labels.size, 8)).astype(np.float32) * representatives[labels]
    k_matrix = np.zeros((8, 8), dtype=np.float32)
    for support in representatives:
        indices = np.flatnonzero(support)
        k_matrix[np.ix_(indices, indices)] = np.ones((indices.size, indices.size)) / indices.size
    z_next = z @ k_matrix
    return z, z_next, labels, representatives, k_matrix


def test_greedy_codebook_is_fit_only_and_routes_unseen_masks():
    fit = np.asarray([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ], dtype=bool)
    codebook = fit_family_codebook(fit, min_jaccard=0.5)
    assert codebook.representatives.shape == (2, 4)
    score = np.asarray([[1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 1, 0]], dtype=bool)
    labels = assign_families(score, codebook, min_jaccard=0.5)
    assert labels.tolist()[:2] == [0, 1]
    assert labels[2] == -1


def test_aligned_block_k_has_zero_raw_and_posthoc_leakage():
    z, z_next, labels, representatives, k_matrix = _block_problem()
    result = evaluate_regime(
        z, z_next, labels, representatives, np.asarray([100, 100]), k_matrix, min_family_score=1
    )
    aggregate = result["aggregate"]
    assert aggregate["activity_k_leakage_rms"] == pytest.approx(0.0)
    assert aggregate["matrix_k_leakage_fro_activity_weighted_mean"] == pytest.approx(0.0)
    assert aggregate["activity_k_change_leakage_rms"] == pytest.approx(0.0)
    assert aggregate["matrix_k_change_leakage_fro_activity_weighted_mean"] == pytest.approx(0.0)
    assert aggregate["posthoc_pkp_inside_residual_rms"] == pytest.approx(0.0)
    assert aggregate["closure_pythagorean_error"] < 1e-12


def test_pair_permutation_null_breaks_coordinate_alignment():
    z, z_next, labels, representatives, k_matrix = _block_problem(repeats=64)
    leakages = []
    for permutation in sign_pair_permutations(4, count=16, seed=20260722):
        result = evaluate_regime(
            z[:, permutation], z_next[:, permutation], labels, representatives[:, permutation],
            np.asarray([100, 100]), k_matrix, min_family_score=1,
        )
        leakages.append(result["aggregate"]["activity_k_leakage_rms"])
    assert np.median(leakages) > 0.1


def test_dense_mixing_k_fails_support_closure():
    rng = np.random.default_rng(4)
    z = np.zeros((64, 8), dtype=np.float32)
    z[:, :2] = rng.normal(size=(64, 2))
    mask = np.zeros_like(z, dtype=bool)
    mask[:, :2] = True
    k_matrix = np.ones((8, 8), dtype=np.float32) / np.sqrt(8.0)
    z_next = z @ k_matrix
    metrics = transition_metrics(z, z_next, mask, k_matrix)
    assert metrics["activity_k_leakage_rms"] > 0.8
    assert metrics["activity_k_change_leakage_rms"] > 0.5
    assert matrix_metrics(mask[0], k_matrix)["matrix_k_leakage_fro"] > 0.8
    assert matrix_metrics(mask[0], k_matrix)["matrix_k_change_leakage_fro"] > 0.5


def test_identity_k_cannot_satisfy_change_normalized_guard():
    z = np.ones((8, 6), dtype=np.float32)
    mask = np.zeros_like(z, dtype=bool)
    mask[:, :3] = True
    k_matrix = np.eye(6, dtype=np.float32)
    metrics = transition_metrics(z, z, mask, k_matrix)
    assert metrics["activity_k_leakage_rms"] == pytest.approx(0.0)
    assert metrics["activity_k_change_leakage_rms"] is None
    assert matrix_metrics(mask[0], k_matrix)["matrix_k_change_leakage_fro"] is None


def test_pair_permutations_preserve_values_cardinality_and_sign_pairing():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(7, 10)).astype(np.float32)
    mask = np.abs(z) > 0.4
    for permutation in sign_pair_permutations(5, count=4, seed=7):
        np.testing.assert_allclose(np.sort(z[:, permutation], axis=1), np.sort(z, axis=1))
        np.testing.assert_array_equal(mask[:, permutation].sum(axis=1), mask.sum(axis=1))
        np.testing.assert_array_equal(permutation[5:] - 5, permutation[:5])


class _FakeEncoder:
    final_op = "sign_split"
    base_zdim = 4

    @staticmethod
    def _split_sign(value):
        return torch.cat([F.relu(value), F.relu(-value)], dim=-1)


class _FakeModel:
    _uses_sign_split_latent = True
    encoder = _FakeEncoder()
    dict = torch.zeros(8, 2)


def _fake_config(final_op: str = "sign_split"):
    lista = SimpleNamespace(FINAL_OP=final_op)
    encoder = SimpleNamespace(ENCODER_TYPE="lista", LISTA=lista)
    return SimpleNamespace(MODEL=SimpleNamespace(ENCODER=encoder, TARGET_SIZE=8))


def test_sign_split_order_is_dynamically_verified():
    assert assert_sign_split_layout(_fake_config(), _FakeModel()) == 4
    with pytest.raises(AssertionError):
        assert_sign_split_layout(_fake_config("relu"), _FakeModel())


def _strong_system_row(index):
    row = {
        "system_key": f"system_{index}",
        "system_name": f"system_{index}",
        "run_count": 3,
        "eligible_seed_count": 3,
        "system_eligible": True,
    }
    for display in METRICS:
        row[f"{display}_true"] = 0.2
        row[f"{display}_null"] = 0.5
        row[f"{display}_true_over_null"] = 0.4
    row["global_over_identity_true"] = 0.8
    row["operator_distance_true"] = 0.5
    row["operator_distance_null"] = 0.25
    row["operator_distance_true_over_null"] = 2.0
    return row


def test_strong_decision_requires_change_normalized_gate():
    from experiments.neurips_2026.global_k_support_invariance import load_card

    card, _hash = load_card()
    run_rows = [{"status": "eligible"} for _ in range(45)]
    systems = [_strong_system_row(index) for index in range(15)]
    assert decide(run_rows, systems, card)["decision"] == "strong_direct_sum"
    for row in systems:
        row["activity_change_leakage_true_over_null"] = 0.9
    result = decide(run_rows, systems, card)
    assert result["decision"] == "failed"
    assert not result["checks"]["activity_change_leakage_null_ratio"]
