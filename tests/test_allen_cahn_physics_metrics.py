from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.neurips_2026.allen_cahn_forecast_replication.io import (
    load_pinned_module,
    pinned_source,
)
from experiments.neurips_2026.allen_cahn_physics_metrics.core import (
    HORIZON,
    METRIC_NAMES,
    energy_components,
    score_candidate,
    validate_score_record,
)
from experiments.neurips_2026.allen_cahn_physics_metrics.io import (
    CARD_PATH,
    assert_paths_sealed,
    authenticated_prior,
    load_card,
)
from experiments.neurips_2026.allen_cahn_physics_metrics.statistics import (
    arm_mean_effect_summary,
    classify_secondary_pattern,
    exact_one_sided_sign_flip,
    holm_adjust,
    validate_bootstrap_seed_schedule,
)


def _centers() -> torch.Tensor:
    angles = torch.linspace(0.0, 2.0 * torch.pi, 5)[:-1]
    return 1.5 * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)


def _uniform(well: int, trajectories: int = 2) -> torch.Tensor:
    center = _centers()[well]
    return center.view(1, 1, 1, 1, 2).expand(
        trajectories, HORIZON, 16, 16, 2
    ).clone()


def test_card_freezes_every_requested_metric_and_authenticated_panel() -> None:
    card, _ = load_card()
    names = tuple(row["name"] for row in card["metric_contract"]["mandatory_metrics"])
    assert names == METRIC_NAMES
    assert card["matching_contract"]["time_policy"].startswith("Every stored observation")
    assert [row["dataset_seed"] for row in card["authenticated_inputs"]["datasets"]] == [
        1775404171,
        74732421,
        293789188,
    ]
    assert card["execution"]["scientific_jobs_submitted"] == 0
    assert card["execution"]["physics_outcomes_inspected"] == 0
    assert card["visualization_contract"]["paper_display_dataset_index"] == 0
    assert card["visualization_contract"]["archived_snapshot_dataset_indices"] == [0, 1, 2]
    expected_bootstrap_seeds = {
        name: 2026072101 + index for index, name in enumerate(METRIC_NAMES)
    }
    assert card["inference_and_reporting"]["bootstrap_seeds_by_metric"] == (
        expected_bootstrap_seeds
    )
    assert validate_bootstrap_seed_schedule(expected_bootstrap_seeds) == (
        expected_bootstrap_seeds
    )


def test_authenticated_prior_checks_metadata_only_and_exact_rosters() -> None:
    card, _ = load_card()
    prior = authenticated_prior(card)
    assert len(prior["datasets"]) == 3
    assert len(prior["checkpoint_roster"]) == 20
    assert {(row["arm"], row["seed"]) for row in prior["checkpoint_roster"]} == {
        (arm, seed) for arm in ("dense", "sparse") for seed in range(64, 74)
    }


def test_sealed_artifact_tokens_fail_before_access() -> None:
    with pytest.raises(AssertionError, match="sealed"):
        assert_paths_sealed([Path("/tmp/early_fate_probe/fields.pt")])
    with pytest.raises(AssertionError, match="sealed"):
        assert_paths_sealed([Path("/tmp/seed20260725.pt")])


def test_identical_fields_have_zero_error_and_perfect_modal_accuracy() -> None:
    truth = _uniform(0)
    record = score_candidate(
        truth.clone(),
        truth,
        _centers(),
        beta=8.0,
        reaction_strength=1.0,
        diffusion=0.005,
    )
    for name in METRIC_NAMES:
        values = torch.tensor(record["curves"][name]["instantaneous"])
        expected = 1.0 if name == "modal_well_accuracy" else 0.0
        torch.testing.assert_close(values, torch.full_like(values, expected))
    validate_score_record(record)


def test_uniform_wrong_well_penalizes_phase_but_not_invents_interface() -> None:
    truth = _uniform(0)
    prediction = _uniform(1)
    record = score_candidate(
        prediction,
        truth,
        _centers(),
        beta=8.0,
        reaction_strength=1.0,
        diffusion=0.005,
    )
    curves = record["curves"]
    assert curves["nearest_well_pixel_disagreement"]["instantaneous"][0] == 1.0
    assert curves["modal_well_accuracy"]["instantaneous"][0] == 0.0
    assert curves["well_area_fraction_tv_error"]["instantaneous"][0] == 1.0
    assert curves["interface_edge_disagreement"]["instantaneous"][0] == 0.0


def test_periodic_boundary_shift_is_detected() -> None:
    truth = _uniform(0, trajectories=1)
    prediction = truth.clone()
    truth[:, :, :, 8:] = _centers()[1]
    prediction[:, :, :, 9:] = _centers()[1]
    record = score_candidate(
        prediction,
        truth,
        _centers(),
        beta=8.0,
        reaction_strength=1.0,
        diffusion=0.005,
    )
    assert record["curves"]["interface_edge_disagreement"]["instantaneous"][0] > 0


def test_energy_is_exact_sum_of_frozen_components() -> None:
    fields = _uniform(0, trajectories=1)
    fields[:, :, :, 8:] = _centers()[1]
    energy = energy_components(
        fields,
        _centers(),
        beta=8.0,
        reaction_strength=1.0,
        diffusion=0.005,
    )
    torch.testing.assert_close(energy["free"], energy["potential"] + energy["gradient"])
    assert bool((energy["gradient"] > 0).all())


def test_negative_discrete_energy_gradient_matches_pinned_pde_rhs() -> None:
    card, _ = load_card()
    prior = authenticated_prior(card)
    source_card = prior["source_card"]
    module = load_pinned_module(
        pinned_source(source_card, "physics_and_initial_conditions")
    )
    generator_config = source_card["system_and_generator"]
    coefficients = card["metric_contract"]["energy_coefficients"]
    pinned_config = module.SpatialReactionDiffusionConfig(
        source_system="allen_cahn_4",
        grid_size=16,
        spatial_extent=1.0,
        laplacian_scaling="continuum",
        diffusion=float(coefficients["diffusion"]),
        allen_cahn_beta=float(coefficients["beta"]),
        allen_cahn_reaction_strength=float(coefficients["reaction_strength"]),
        allen_cahn_center_radius=float(generator_config["allen_cahn_center_radius"]),
    )
    system = module.get_source_system("allen_cahn_4", pinned_config)
    centers = module.extract_attractor_centers(system).to(torch.float64)
    generator = torch.Generator().manual_seed(2026072108)
    field = (0.25 * torch.randn(16, 16, 2, generator=generator, dtype=torch.float64))
    field.requires_grad_(True)
    energy = energy_components(
        field[None, None],
        centers,
        beta=float(coefficients["beta"]),
        reaction_strength=float(coefficients["reaction_strength"]),
        diffusion=float(coefficients["diffusion"]),
    )["free"].sum()
    euclidean_gradient = torch.autograd.grad(energy, field)[0]
    negative_discrete_l2_gradient = -euclidean_gradient / (1.0 / 16.0) ** 2
    pinned_rhs = module._pde_rhs(system, field, pinned_config)
    torch.testing.assert_close(
        negative_discrete_l2_gradient,
        pinned_rhs,
        rtol=1e-11,
        atol=1e-11,
    )


def test_curve_validation_rejects_a_single_omitted_or_changed_time() -> None:
    truth = _uniform(0)
    record = score_candidate(
        truth,
        truth,
        _centers(),
        beta=8.0,
        reaction_strength=1.0,
        diffusion=0.005,
    )
    broken = copy.deepcopy(record)
    broken["curves"][METRIC_NAMES[0]]["instantaneous"] = broken["curves"][METRIC_NAMES[0]][
        "instantaneous"
    ][:-1]
    with pytest.raises(AssertionError, match="H200"):
        validate_score_record(broken)
    broken = copy.deepcopy(record)
    broken["curves"][METRIC_NAMES[0]]["cumulative"][99] = 0.25
    with pytest.raises(AssertionError):
        validate_score_record(broken)

    sorted_key_roundtrip = json.loads(json.dumps(record, sort_keys=True))
    validate_score_record(sorted_key_roundtrip)


def test_exact_test_and_holm_are_deterministic_and_complete() -> None:
    improvements = np.arange(1.0, 11.0)
    result = exact_one_sided_sign_flip(improvements)
    assert result["enumerated_sign_vectors"] == 1024
    assert 0.0 <= result["one_sided_exact_p"] <= 1.0
    raw = {name: 0.01 + 0.01 * index for index, name in enumerate(METRIC_NAMES)}
    adjusted = holm_adjust(raw)
    assert tuple(adjusted) == METRIC_NAMES
    assert all(raw[name] <= adjusted[name] <= 1.0 for name in METRIC_NAMES)


def test_pattern_branches_are_mutually_exclusive() -> None:
    all_families = {"phase_assignment", "interface_geometry", "thermodynamics"}
    assert classify_secondary_pattern(2, {"phase_assignment"}) == (
        "little_or_reversed_secondary_translation"
    )
    assert classify_secondary_pattern(3, all_families) == "mixed_secondary_translation"
    assert classify_secondary_pattern(5, {"phase_assignment", "thermodynamics"}) == (
        "mixed_secondary_translation"
    )
    assert classify_secondary_pattern(5, all_families) == "broad_secondary_concordance"


def test_zero_dense_error_uses_direction_correct_absolute_fallback() -> None:
    effect = arm_mean_effect_summary(0.0, 0.25, "lower")
    assert effect == {
        "label": "oriented_absolute_error_improvement_dense_minus_sparse",
        "value": -0.25,
    }
    accuracy = arm_mean_effect_summary(0.0, 0.25, "higher")
    assert accuracy == {
        "label": "accuracy_percentage_point_difference",
        "value": 25.0,
    }


def test_prediction_card_has_no_duplicate_json_keys() -> None:
    def reject(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(key)
            result[key] = value
        return result

    json.loads(CARD_PATH.read_text(), object_pairs_hook=reject)
