import json
from pathlib import Path

import numpy as np

from experiments.neurips_2026.global_k_distinct_laws import (
    fit_centered_local_matrix,
    law_cost_summary,
    load_protocol_card,
    match_families_to_basins,
    rk4_step_matrix,
    sample_centered_disk,
)
from experiments.neurips_2026.summarize_global_k_distinct_laws import adjudicate
from experiments.neurips_2026.build_global_k_distinct_laws_packet import (
    sparse_phase_decision,
)


ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "experiments/neurips_2026/global_k_distinct_laws_card.json"
V2_CARD = ROOT / "experiments/neurips_2026/global_k_distinct_laws_v2_card.json"


def test_centered_local_fit_recovers_column_vector_matrix_and_intercept():
    offsets = sample_centered_disk(512, 0.18, 7).astype(np.float64)
    matrix = np.asarray([[-0.04, -0.03], [0.05, -0.02]])
    intercept = np.asarray([0.001, -0.002])
    updates = offsets @ matrix.T + intercept
    fit = fit_centered_local_matrix(offsets, updates)
    np.testing.assert_allclose(fit.matrix, matrix, atol=1e-12)
    np.testing.assert_allclose(fit.intercept, intercept, atol=1e-12)
    assert fit.relative_residual < 1e-12


def test_rk4_step_matrix_matches_fourth_order_polynomial():
    matrix = np.asarray([[-0.9, -1.2], [1.2, -0.9]])
    dt = 0.04
    scaled = dt * matrix
    expected = (
        np.eye(2)
        + scaled
        + scaled @ scaled / 2
        + scaled @ scaled @ scaled / 6
        + scaled @ scaled @ scaled @ scaled / 24
    )
    np.testing.assert_allclose(rk4_step_matrix(matrix, dt), expected)


def test_family_matching_is_one_to_one_and_reports_basin_rates():
    assignments = np.asarray(
        [
            [4] * 8 + [8] * 2,
            [7] * 9 + [4],
            [8] * 7 + [7] * 3,
        ]
    )
    retained = np.zeros(10, dtype=bool)
    retained[[4, 7, 8]] = True
    mapping, rates, counts = match_families_to_basins(assignments, retained, 3)
    assert mapping.tolist() == [4, 7, 8]
    np.testing.assert_allclose(rates, [0.8, 0.9, 0.7])
    assert counts.shape == (3, 3)


def test_law_cost_primary_uses_fixed_predictions_and_rejects_permutation():
    true = np.asarray(
        [
            [[-0.04, -0.05], [0.05, -0.04]],
            [[-0.06, 0.01], [-0.02, -0.03]],
            [[-0.03, -0.01], [0.02, -0.05]],
        ]
    )
    predicted = true + 0.01 * np.asarray(
        [np.eye(2), -np.eye(2), np.diag([1.0, -1.0])]
    )
    summary = law_cost_summary(predicted, true)
    assert summary["optimal_assignment"] == [0, 1, 2]
    assert summary["identity_is_unique_optimum"]
    assert len(summary["assignment_costs"]) == 6
    assert summary["identity_over_best_nonidentity"] < 1.0


def test_card_forbids_trivial_wrong_source_and_normalizes_k_minus_i():
    card = json.loads(CARD.read_text())
    primary = card["primary_estimand"]
    controls = card["controls"]
    assert "K=I" in primary["near_identity_guard"]
    assert "J_true-I" in primary["near_identity_guard"]
    assert primary["unchanged_global_K"] is True
    assert "erasing source activity" in controls["wrong_support_excluded"]
    assert controls["coordinate_null"]["role"].startswith("secondary")
    geometry = card["evaluation_geometry"]
    assert geometry["calibration_and_score_relation"].startswith("independently seeded")
    specificity = card["sparse_specificity_gate"]
    assert "max_own_over_nearest_wrong" in specificity["row_metric_per_seed"]
    assert "identity_over_best_nonidentity" in specificity["assignment_metric_per_seed"]
    assert "all three" in specificity["seed_aggregation"]


def test_protocol_card_authenticates_referenced_cards():
    card, digest = load_protocol_card(CARD)
    assert card["protocol_id"] == "global_k_distinct_laws_gated_local_linear_v1"
    assert len(digest) == 64


def _synthetic_shard(arm, seed, row_ratio, assignment_ratio, strong=True):
    law = {
        "max_own_over_nearest_wrong": row_ratio,
        "identity_over_best_nonidentity": assignment_ratio,
        "max_own_relative_error": 0.2,
    }
    return {
        "arm": arm,
        "seed": seed,
        "result": {
            "status": "eligible",
            "strong_distinct_law_pass": strong,
            "block": {"law_identification": law},
            "global": {"law_identification": law},
        },
    }


def test_adjudication_requires_all_seeds_and_dense_specificity():
    card = json.loads(CARD.read_text())
    by_arm = {
        "sparse": [_synthetic_shard("sparse", seed, 0.3, 0.3) for seed in range(3)],
        "dense": [_synthetic_shard("dense", seed, 0.6, 0.6) for seed in range(3)],
    }
    decision = adjudicate(by_arm, card)
    assert decision["decision"] == "strong_distinct_laws_sparse_specific"
    assert decision["gates"]["sparse_strong_3_of_3"]

    by_arm["sparse"][2]["result"]["strong_distinct_law_pass"] = False
    decision = adjudicate(by_arm, card)
    assert decision["decision"] == "global_map_only"


def test_sparse_packet_keeps_dense_specificity_pending():
    card = json.loads(CARD.read_text())
    sparse = [_synthetic_shard("sparse", seed, 0.3, 0.3) for seed in range(3)]
    decision = sparse_phase_decision(sparse, card)
    assert decision["decision"] == "strong_distinct_laws_pending_dense_specificity"
    assert decision["dense_specificity"] == "pending"


def test_v2_is_new_seed_jacobian_replication_not_v1_reclassification():
    card = json.loads(V2_CARD.read_text())
    assert card["status"] == "preregistered_before_new_seed_training_or_checkpoint_evaluation"
    assert card["v1_provenance_and_boundary"]["v1_formal_decision"] == "invalid"
    assert card["v1_provenance_and_boundary"]["reuse_of_v1_model_seeds_or_checkpoints"] is False
    seeds = card["new_seed_contract"]["scientific_seeds"]
    assert seeds == list(range(100, 110))
    assert set(seeds).isdisjoint(range(15))
    primary = card["primary_autograd_estimand"]
    assert "autograd" in primary["primary_law_matrix"]
    assert "-x" in primary["primary_restricted_predictor_change"]
    assert "D(E(x)P_f)" in primary["required_k_induced_co_estimand"]
    assert "H_b=G_b" in primary["reconstruction_derivative_guard"]
    assert primary["unchanged_global_K"] is True
    assert primary["no_latent_dynamics_fit"] is True


def test_v2_dense_is_exact_tanh_control_and_paired_before_outcomes():
    card = json.loads(V2_CARD.read_text())
    dense = card["training_arms"]["dense"]
    assert dense["encoder"]["hidden_activation"] == "tanh"
    assert dense["weight_decay"] == 0.0
    assert dense["loss_weights"]["sparsity"] == 0.0
    assert dense["all_sparse_or_zero_inducing_features_disabled"] is True
    schedule = card["gpu_utilization_and_schedule"]
    assert schedule["scientific_training_after_smoke_pass"]["pack_concurrency"] == 20
    assert "concurrently" in schedule["dense_schedule"]


def test_v2_separates_pointwise_decision_from_intercept_and_radius_robustness():
    card = json.loads(V2_CARD.read_text())
    intercept = card["center_forecast_guards_and_affine_law_gate"]
    robustness = card["finite_radius_robustness_not_selection"]
    assert intercept["pointwise_jacobian_decisions_independent_of_affine_gate"] is True
    for metric in (
        "support_reconstruction",
        "k_induced_update",
        "restricted_forecast",
        "full_reconstruction",
        "full_forecast",
    ):
        assert metric in intercept
    assert "24" in intercept["affine_local_law_gate"]
    assert robustness["primary_autograd_decision_independent_of_sweep"] is True
    assert robustness["radii"] == [0.01, 0.03, 0.06, 0.12, 0.18]
    aggregate = card["aggregate_sparse_gate"]
    assert aggregate["minimum_joint_h_g_seed_passes"] == 8
    assert aggregate["minimum_h_of_30_basin_rows_with_own_law_nearest"] == 27
    assert aggregate["minimum_g_of_30_basin_rows_with_own_law_nearest"] == 27
    assert card["dense_recipe_specificity_gate"]["minimum_paired_seeds_sparse_better_on_both_h_metrics"] == 9


def test_v2_kink_guard_and_full_global_pass_are_explicit():
    card = json.loads(V2_CARD.read_text())
    kink = card["autograd_differentiability_kink_guard"]
    assert kink["coordinate_symmetric_epsilons"] == [0.0005, 0.0015]
    assert kink["maximum_disagreement_each_epsilon"] == 0.1
    assert kink["aggregate_minimum_seed_basin_pairs_passing_both_epsilons"] == 27
    assert card["per_seed_sparse_gate"]["kink_guard_all_three_basins_required"] is True
    assert "both h_f^block and g_f^block" in kink["method"]
    assert card["per_seed_sparse_gate"]["h_global_positive_control_required"] is True
    full = card["three_law_identification"]["full_global_pass"]
    assert "H_global" in full
    assert "e_s^global<=0.8" in full
    assert "r_s^global<=0.8" in full
    assert "a_s^global<=0.8" in full
    assert "uniquely" in full


def test_v2_dense_specificity_is_complete_recipe_not_sparsity_causation():
    card = json.loads(V2_CARD.read_text())
    specificity = card["dense_recipe_specificity_gate"]
    assert specificity["decision_metric"].startswith("H_block")
    assert "G_block" in specificity["g_secondary_report"]
    assert "complete sparse recipe" in specificity["complete_recipe_caveat"]
    decision = card["decision_structure"]
    assert "No positive branch" in decision["mandatory_positive_claim_caveat"]
    assert "does not assert identical" in card["new_seed_contract"]["pairing_scope"]


def test_v2_gpu_smoke_must_pass_before_scientific_pack():
    card = json.loads(V2_CARD.read_text())
    smoke = card["gpu_utilization_and_schedule"]["smoke"]
    assert smoke["outcomes_quarantined"] is True
    assert smoke["pack_size"] == smoke["pack_concurrency"] == 20
    assert smoke["minimum_mean_active_gpu_utilization_percent"] >= 90
    assert smoke["minimum_p10_active_gpu_utilization_percent"] >= 80
    assert smoke["maximum_peak_memory_fraction"] <= 0.8


def test_v2_pre_run_amendment_closes_reconstruction_jacobian_loophole():
    card = json.loads(V2_CARD.read_text())
    amendment = card["pre_run_estimand_amendment"]
    assert amendment["superseded_card_sha256"] == (
        "0efcb4a97f07965af9c4748814337027ebe49b12cc22a9051c6aa8ebf664a07d"
    )
    assert "before source/task freeze" in amendment["timing"]
    laws = card["three_law_identification"]
    assert "Both H_block and G_block" in laws["restricted_one_k_law_pass"]
    finite = card["finite_radius_robustness_not_selection"]
    assert "For M in {H,G}" in finite["agreement"]
    assert "for each of H and G" in finite["finite_neighborhood_extension_gate"]
    decisions = card["decision_structure"]["mechanism_claim_tiers"]
    assert "distinct_k_induced_update_jacobians_only" in decisions
    assert "actual predictor" in decisions["distinct_k_induced_update_jacobians_only"]
