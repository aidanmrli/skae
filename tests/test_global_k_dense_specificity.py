"""Tests for the preregistered dense zero-WD specificity control."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.neurips_2026.global_k_dense_specificity import (
    assign_dense_families,
    assert_exact_dense_control,
    fit_dense_family_codebook,
    matched_topk_masks,
)
from experiments.neurips_2026.global_k_dense_zero_wd_tasks import build_rows, load_card
from experiments.neurips_2026.global_k_support_invariance import (
    assign_families,
    fit_family_codebook,
)
from experiments.neurips_2026.summarize_global_k_dense_specificity import decide
from skae.config import get_config
from skae.model import make_model


def test_matched_topk_preserves_each_sparse_cardinality() -> None:
    dense = np.asarray([[4.0, 1.0, 3.0, 2.0], [0.0, -2.0, 5.0, 1.0]])
    sparse = np.asarray([[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 4.0]])
    masks = matched_topk_masks(dense, sparse, threshold=0.5)
    assert masks.sum(axis=1).tolist() == [2, 2]
    assert masks[0].tolist() == [True, False, True, False]
    assert masks[1].tolist() == [False, True, True, False]


def test_matched_topk_breaks_ties_by_coordinate_index() -> None:
    dense = np.ones((1, 4))
    sparse = np.asarray([[1.0, 1.0, 0.0, 0.0]])
    assert matched_topk_masks(dense, sparse, threshold=0.5).tolist() == [
        [True, True, False, False]
    ]


def test_packed_dense_codebook_is_exactly_reference_equivalent() -> None:
    mask = np.random.default_rng(7).random((80, 19)) > 0.7
    reference = fit_family_codebook(mask, min_jaccard=0.5)
    packed = fit_dense_family_codebook(mask, min_jaccard=0.5)
    np.testing.assert_array_equal(packed.representatives, reference.representatives)
    np.testing.assert_array_equal(packed.fit_counts, reference.fit_counts)
    assert packed.exact_key_to_family == reference.exact_key_to_family
    score = np.random.default_rng(8).random((30, 19)) > 0.7
    np.testing.assert_array_equal(
        assign_dense_families(score, packed, min_jaccard=0.5),
        assign_families(score, reference, min_jaccard=0.5),
    )


def test_frozen_full_task_rows_are_exact_dense_controls() -> None:
    card, _ = load_card()
    rows = build_rows(card, "full")
    assert len(rows) == 45
    assert {(row["system_key"], int(row["seed"])) for row in rows} == {
        (system, seed)
        for system in card["training"]["systems"]
        for seed in card["training"]["seeds"]
    }
    assert all(row["config_name"] == "generic_no_shrink" for row in rows)
    assert all(float(row["weight_decay"]) == 0.0 for row in rows)
    assert all(float(row["sparsity_coeff"]) == 0.0 for row in rows)
    assert all(row["k_structure"] == "dense" for row in rows)
    assert all(row["basin_count"] == "" for row in rows)


def _exact_dense_model():
    card, _ = load_card()
    train = card["training"]
    cfg = get_config("generic_no_shrink")
    cfg.MODEL.TARGET_SIZE = train["latent_dim"]
    cfg.MODEL.RES_COEFF = train["residual_coefficient"]
    cfg.MODEL.RECONST_COEFF = train["reconstruction_coefficient"]
    cfg.MODEL.PRED_COEFF = train["prediction_coefficient"]
    cfg.MODEL.K_STRUCTURE = train["koopman_structure"]
    cfg.TRAIN.NUM_STEPS = train["num_steps"]
    cfg.TRAIN.BATCH_SIZE = train["batch_size"]
    cfg.TRAIN.LR = train["learning_rate"]
    cfg.TRAIN.K_MATRIX_LR = train["koopman_learning_rate"]
    cfg.TRAIN.WEIGHT_DECAY = 0.0
    cfg.TRAIN.SEQUENCE_LENGTH = train["sequence_length"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.ENABLED = True
    return card, cfg, make_model(cfg, observation_size=2)


def test_dense_checkpoint_audit_reconstructs_only_zero_wd_groups() -> None:
    card, cfg, model = _exact_dense_model()
    audit = assert_exact_dense_control(
        cfg, model, card, {"optimizer_state_dict": None}
    )
    assert audit["compact_best_checkpoint_omits_optimizer_state"] is True
    assert audit["reconstructed_param_group_weight_decays"] == [0.0, 0.0]


def test_dense_checkpoint_audit_rejects_serialized_nonzero_wd() -> None:
    card, cfg, model = _exact_dense_model()
    checkpoint = {"optimizer_state_dict": {"param_groups": [{"weight_decay": 1e-4}]}}
    with pytest.raises(AssertionError, match="serialized_optimizer"):
        assert_exact_dense_control(cfg, model, card, checkpoint)


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        (lambda cfg: setattr(cfg.MODEL.DECODER, "USE_BIAS", True), "decoder_bias_absent"),
        (
            lambda cfg: setattr(cfg.MODEL.DECODER, "NORMALIZE_ATOMS", True),
            "decoder_atom_normalization_off",
        ),
        (lambda cfg: setattr(cfg.MODEL, "USE_HOMOGENEOUS", True), "homogeneous_coordinates_off"),
        (lambda cfg: setattr(cfg.MODEL, "NORM_FN", "ball"), "identity_latent_normalization"),
        (lambda cfg: setattr(cfg.TRAIN, "EVAL_EVERY", 250), "eval_every"),
    ],
)
def test_dense_checkpoint_audit_rejects_hidden_control_drift(mutation, failure) -> None:
    card, cfg, _model = _exact_dense_model()
    mutation(cfg)
    model = make_model(cfg, observation_size=2)
    with pytest.raises(AssertionError, match=failure):
        assert_exact_dense_control(cfg, model, card, {"optimizer_state_dict": None})


def _system_rows(activity: float, residual: float, count: int = 15):
    return [
        {
            "system_eligible": True,
            "activity_leakage_true": 0.2,
            "activity_leakage_null": 0.4,
            "activity_leakage_true_over_null": activity,
            "restricted_residual_true": 0.2,
            "restricted_residual_null": 0.4,
            "restricted_residual_true_over_null": residual,
        }
        for _ in range(count)
    ]


def test_specificity_decision_requires_both_frozen_ratios() -> None:
    card, _ = load_card()
    run_rows = [{"status": "eligible"} for _ in range(45)]
    passing = decide(run_rows, _system_rows(activity=0.5, residual=0.5), card)
    assert passing["decision"] == "sparse_support_specific"
    failing = decide(run_rows, _system_rows(activity=0.01, residual=0.01), card)
    assert failing["decision"] == "not_sparse_specific"


def test_specificity_decision_is_invalid_when_dense_routing_is_ineligible() -> None:
    card, _ = load_card()
    run_rows = [{"status": "eligible"} for _ in range(35)]
    result = decide(run_rows, _system_rows(activity=0.5, residual=0.5, count=11), card)
    assert result["decision"] == "invalid_dense_control"
