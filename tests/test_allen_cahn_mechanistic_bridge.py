"""Synthetic and static tests for the dormant Allen--Cahn bridge packet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.aggregation import (
    _routing_aggregate,
    decide,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.conditional_guard import (
    validate_mechanism_decision,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.families import (
    alignment_metrics,
    allen_cahn_centers,
    jaccard_rows,
    modal_well_fates,
    truth_difficulty,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.integrity import (
    REQUIRED_SOURCE_PATHS,
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import (
    CARD_PATH,
    _duplicate_safe_json,
    assert_field_only_keys,
    finite_tree,
    load_card,
    load_dataset_manifest,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.probes import (
    fit_nested_ridge,
    score_fitted,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.rollouts import (
    initial_projection_controls,
    rollout_full,
    rollout_projected_modes,
    rollout_restricted,
    rollout_support_contrast,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.statistics import (
    difference_summary,
    ratio_summary,
    two_way_bootstrap,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.metrics import (
    assign_codebook,
    fit_codebook,
    matched_topk_masks,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.wrong_supports import (
    build_wrong_support_codebook,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = REPO_ROOT / "experiments/neurips_2026/allen_cahn_mechanistic_bridge"
SCRIPT_ROOT = REPO_ROOT / "scripts/neurips_2026/allen_cahn_mechanistic_bridge"
SOURCE_MANIFEST = BRIDGE_ROOT / "source_manifest.sha256"


def _field_for_fates(fates: list[int]) -> torch.Tensor:
    centers = allen_cahn_centers().float()
    return torch.stack(
        [centers[index].view(1, 1, 2).expand(16, 16, 2) for index in fates]
    ).reshape(len(fates), -1)


def test_card_is_duplicate_safe_and_exactly_reserves_new_seeds(tmp_path: Path) -> None:
    card, _ = load_card()
    assert card["new_datasets"]["seeds"] == [20260729, 20260730, 20260731]
    assert card["new_datasets"]["reserved_forbidden_seed"] == 20260725
    assert card["roster"]["latent_dim"] >= 4 * 512
    assert card["roster"]["rollout"].startswith("direct repeated global K")
    assert card["conditional_launch"]["decision_path"].endswith("v4/summary/decision.json")
    assert card["new_datasets"]["output_root"].endswith("_v2")
    assert card["field_only_stage"]["label_firewall"].startswith("torch.load may")
    assert card["primary_gates"]["probe"]["minimum_model_seed_wins"] == 8
    assert card["primary_gates"]["probe"]["minimum_dataset_seed_wins"] == 2
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"same": 1, "same": 2}\n')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        _duplicate_safe_json(duplicate)


def test_conditional_guard_requires_every_mechanism_qualification() -> None:
    card, _ = load_card()
    contract = card["conditional_launch"]
    decision = {
        "card_sha256": contract["required_card_sha256"],
        "source_manifest_sha256": contract["required_source_manifest_sha256"],
        "validity": {"passed": True, "provenance_and_firewall": True},
        "family": {
            "family_checks": {"eligible_seeds": True},
            "qualification": {"eligible_seed_count": 8},
        },
    }
    assert validate_mechanism_decision(decision, card)["passed"]
    decision["family"]["qualification"]["eligible_seed_count"] = 7
    assert not validate_mechanism_decision(decision, card)["passed"]


def test_field_only_key_firewall_fails_closed() -> None:
    card, _ = load_card()
    assert_field_only_keys(["fields", "split_indices"], card)
    with pytest.raises(AssertionError, match="forbidden"):
        assert_field_only_keys(["fields", "global_basin_labels"], card)
    with pytest.raises(AssertionError, match="non-whitelisted"):
        assert_field_only_keys(["fields", "metadata"], card)


def test_train_only_codebook_and_dense_topk_are_deterministic() -> None:
    train = np.asarray(
        [[1, 1, 0, 0]] * 20 + [[0, 0, 1, 1]] * 18 + [[1, 0, 1, 0]],
        dtype=bool,
    )
    codebook = fit_codebook(
        train, min_jaccard=0.4, max_representatives=8, min_fit_count=16
    )
    assert codebook.representatives.shape == (2, 4)
    labels, similarities = assign_codebook(
        np.asarray([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 1]], bool),
        codebook.representatives,
        min_jaccard=0.4,
    )
    assert labels[:2].tolist() == [0, 1]
    assert similarities[:2].tolist() == [1.0, 1.0]
    dense = np.asarray([[3.0, -3.0, 2.0, -2.0], [0.0, 4.0, 2.0, 1.0]])
    sparse = np.asarray([[1, 0, 1, 0], [0, 1, 0, 0]], bool)
    topk = matched_topk_masks(dense, sparse)
    assert np.array_equal(topk.sum(1), sparse.sum(1))
    assert topk.tolist() == [[True, True, False, False], [False, True, False, False]]
    train_labels, _ = assign_codebook(
        train, codebook.representatives, min_jaccard=0.4
    )
    wrong = build_wrong_support_codebook(
        train, train_labels, codebook.representatives, codebook.fit_counts
    )
    assert bool(wrong["valid_and_distinct"].all())
    assert torch.equal(
        wrong["target_cardinalities"], wrong["wrong_cardinalities"]
    )
    assert not bool(torch.any(torch.all(
        wrong["representatives"] == torch.from_numpy(codebook.representatives), dim=1
    )))


def test_fates_alignment_and_unknown_coverage_are_well_defined() -> None:
    fates = np.asarray([0, 1, 2, 3] * 4)
    fields = _field_for_fates(fates.tolist())
    assert modal_well_fates(fields).tolist() == fates.tolist()
    perfect = alignment_metrics(fates, fates)
    assert perfect["coverage"] == 1.0
    assert perfect["ari"] == pytest.approx(1.0)
    assert perfect["purity"] == pytest.approx(1.0)
    unknown = alignment_metrics(np.full(fates.shape, -1), fates)
    assert unknown["coverage"] == 0.0
    assert unknown["defined_on_covered_rows"] is False
    assert unknown["ari"] == 0.0


def test_truth_difficulty_distinguishes_dynamic_from_asymptotic() -> None:
    x0 = _field_for_fates([0, 0])
    x200 = _field_for_fates([1, 1])
    dynamic = truth_difficulty(x0, x200, _field_for_fates([2, 2]))
    stable = truth_difficulty(x0, x200, x200.clone())
    assert dynamic["dynamic_temporal_extrapolation"] is True
    assert dynamic["modal_fate_change_fraction"] == 1.0
    assert stable["dynamic_temporal_extrapolation"] is False
    assert stable["continued_change_ratio"] == 0.0


def test_nested_probe_recovers_four_classes_and_ties_choose_larger_alpha() -> None:
    rng = np.random.default_rng(1201)
    labels = np.repeat(np.arange(4), 16)
    features = np.eye(4)[labels] + 0.02 * rng.normal(size=(labels.size, 4))
    fitted, audit = fit_nested_ridge(
        features,
        labels,
        alphas=[0.01, 0.1, 1.0],
        outer_folds=4,
        inner_folds=3,
        final_folds=4,
        seed=99,
    )
    score = score_fitted(fitted, features, labels)
    assert audit["outer_oof"]["balanced_accuracy"] > 0.95
    assert score["test"]["balanced_accuracy"] > 0.95
    constant, tie_audit = fit_nested_ridge(
        np.zeros((64, 2)),
        labels,
        alphas=[0.1, 10.0],
        outer_folds=4,
        inner_folds=3,
        final_folds=4,
        seed=99,
    )
    assert constant.selected_alpha == 10.0
    assert tie_audit["selected_alpha"] == 10.0


class _IdentityModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def step_latent(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return value


def test_direct_rollouts_never_reencode_and_fixed_support_changes_prediction() -> None:
    model = _IdentityModel()
    fields = torch.tensor([[[1.0, 2.0]] * 4, [[3.0, 4.0]] * 4])
    masks = torch.tensor([[True, False], [True, False]])
    full = rollout_full(model, fields, horizons=[2, 3], batch_size=1)
    projected = rollout_projected_modes(
        model, fields, masks, horizons=[2, 3], batch_size=2
    )
    restricted = rollout_restricted(
        model, fields, masks, horizons=[2, 3], batch_size=2
    )
    wrong_masks = ~masks
    contrast = rollout_support_contrast(
        model, fields, masks, wrong_masks, horizons=[2, 3], batch_size=2
    )
    initial = initial_projection_controls(
        model, fields, masks, wrong_masks, batch_size=2
    )
    assert torch.equal(full["2"]["through_mse"], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(projected["full"]["3"]["through_mse"], full["3"]["through_mse"])
    assert torch.equal(
        projected["restricted"]["3"]["through_mse"],
        restricted["3"]["through_mse"],
    )
    assert bool((restricted["3"]["through_mse"] > 0).all())
    assert set(contrast) == {
        "correct_mask_once", "correct_restricted", "wrong_mask_once",
        "wrong_restricted",
    }
    assert torch.equal(
        contrast["correct_mask_once"]["3"]["through_mse"],
        contrast["correct_restricted"]["3"]["through_mse"],
    )
    assert torch.equal(initial["correct_cardinality"], initial["wrong_cardinality"])


def test_crossed_statistics_resample_both_independent_axes() -> None:
    candidate = np.full((10, 3), 0.8)
    control = np.ones((10, 3))
    draws = two_way_bootstrap(candidate - control, replicates=103, seed=4)
    difference = difference_summary(control, candidate, replicates=103, seed=5)
    ratio = ratio_summary(candidate, control, replicates=103, seed=6)
    assert draws.shape == (103,)
    assert np.allclose(draws, -0.2)
    assert difference["model_seed_candidate_wins"] == 10
    assert ratio["ratio_of_cell_means"] == pytest.approx(0.8)
    assert ratio["dataset_seed_candidate_wins"] == 3
    assert two_way_bootstrap(np.ones((8, 3)), replicates=5, seed=2).shape == (5,)
    with pytest.raises(ValueError, match="8--10 model"):
        two_way_bootstrap(np.ones((7, 3)), replicates=5, seed=1)
    with pytest.raises(ValueError, match="10 model, 3 dataset"):
        two_way_bootstrap(np.ones((30, 1)), replicates=5, seed=1)


def test_routing_reduction_freezes_complete_case_models_without_outcomes() -> None:
    rows = []
    for model in range(64, 74):
        for dataset in (20260729, 20260730, 20260731):
            horizons = {}
            for horizon in ("160", "200", "400"):
                horizons[horizon] = {
                    "full": {"through_mse": 0.75, "terminal_mse": 0.85},
                    "correct": {
                        "mask_once": {"through_mse": 0.9, "terminal_mse": 1.0},
                        "restricted": {
                            "through_mse": 0.8, "terminal_mse": 0.9,
                            "modal_fate_accuracy": 0.8,
                        },
                    },
                    "wrong": {
                        "mask_once": {"through_mse": 0.9, "terminal_mse": 1.0},
                        "restricted": {
                            "through_mse": 1.0, "terminal_mse": 1.1,
                            "modal_fate_accuracy": 0.6,
                        },
                    },
                }
            rows.append({
                "model_seed": model,
                "dataset_seed": dataset,
                "routing": {
                    "wrong_control_count": 0 if model == 73 else 256,
                    "paired_cardinality_exact": True,
                    "same_subset_for_all_modes": True,
                    "wrong_control_coverage": 1.0,
                    "horizons": horizons,
                    "initial_projection": {
                        "correct_capture_fraction": 0.99,
                        "wrong_capture_fraction": 0.8,
                        "correct_reconstruction_mse": 0.1,
                        "wrong_reconstruction_mse": 0.2,
                    },
                },
            })
    result = _routing_aggregate(
        rows, list(range(64, 74)), [20260729, 20260730, 20260731],
        replicates=17, seed=9, minimum_models=8,
    )
    assert result["all_cells_available"] is True
    assert result["evaluable_model_seed_count"] == 9
    assert result["excluded_model_seeds"] == [73]


def test_dataset_manifest_is_exact_and_hash_locked(tmp_path: Path) -> None:
    card, _ = load_card()
    card = json.loads(json.dumps(card))
    card["new_datasets"]["output_root"] = str(tmp_path)
    records = []
    for seed, relative in zip(
        card["new_datasets"]["seeds"], card["new_datasets"]["paths"]
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"data-{seed}".encode())
        summary = Path(str(path) + ".summary.json")
        summary.write_text(json.dumps({"seed": seed}))
        records.append({
            "seed": seed,
            "path": str(path),
            "sha256": sha256_path(path),
            "summary_sha256": sha256_path(summary),
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"datasets": records}))
    assert len(load_dataset_manifest(manifest, card)["datasets"]) == 3
    records[0]["seed"] += 1
    manifest.write_text(json.dumps({"datasets": records}))
    with pytest.raises(RuntimeError, match="seed roster"):
        load_dataset_manifest(manifest, card)


def test_source_roster_launchers_and_static_firewalls_are_exact() -> None:
    card, _ = load_card()
    assert "experiments/neurips_2026/allen_cahn_support_subspaces/select_profile.py" in REQUIRED_SOURCE_PATHS
    assert verify_source_manifest(SOURCE_MANIFEST) == sha256_path(SOURCE_MANIFEST)
    extract = (BRIDGE_ROOT / "extract_field_only.py").read_text()
    assert "load_training_fates" not in extract
    assert "global_basin_labels" not in extract
    assert '"requested_dataset_keys": ["fields", "split_indices"]' in extract
    assert '"label_tensors_may_have_been_deserialized": True' in extract
    main_offset = extract.index("def main()")
    assert extract.index("    features = {", main_offset) < extract.index(
        "    stability = _support_stability(", main_offset
    )
    assert (
        '"x0_probe_features_materialized_before_future_encoding": True'
        in extract
    )
    generate_worker = (SCRIPT_ROOT / "run_generate.sh").read_text()
    assert "generation_telemetry" in generate_worker
    assert "sleep 2" in generate_worker
    assert '"gpu_invocation_start"' in generate_worker
    assert '"gpu_invocation_done"' in generate_worker
    assert '--gpu_start_file "${START}" --gpu_done_file "${DONE}"' in generate_worker
    assert card["generation_hardware"][
        "maximum_median_sample_interval_seconds"
    ] == 3.0
    extract_worker = (SCRIPT_ROOT / "run_extract.sh").read_text()
    assert '--release_file "${RELEASE}"' in extract_worker
    assert '--gpu_start_file "${START}" --gpu_done_file "${DONE}"' in extract_worker
    assert '[[ -e "${DONE}" ]] && break' in extract_worker
    extract = (BRIDGE_ROOT / "extract_field_only.py").read_text()
    assert '"event": "gpu_compute_start"' in extract
    assert '"event": "gpu_compute_done"' in extract
    assert '"preload_and_serialization_excluded": True' in extract
    assert card["hardware"]["telemetry_interval_seconds"] == 1
    assert card["hardware"]["maximum_median_sample_interval_seconds"] == 1.5
    for script in SCRIPT_ROOT.glob("*.sh"):
        text = script.read_text()
        assert "/network/scratch/l/lia" not in text
        assert "/home/mila/l/lia" not in text
        assert "scripts/common/cluster_env.sh" in text
        assert '${SLURM_SUBMIT_DIR:-$PWD}' in text
    for path in list(BRIDGE_ROOT.glob("*")) + list(SCRIPT_ROOT.glob("*.sh")):
        if path.name not in {"prediction_card.json", "io.py"} and path.is_file():
            assert "20260725" not in path.read_text()
    for queue in SCRIPT_ROOT.glob("queue_*.sh"):
        text = queue.read_text()
        assert "__FREEZE_CARD__" not in text
        assert "__FREEZE_SOURCE__" not in text
        assert "__FREEZE_AFTER_" in text
    for relative in REQUIRED_SOURCE_PATHS:
        assert (REPO_ROOT / relative).is_file()


def test_all_bridge_authored_files_respect_line_cap_and_finite_tree() -> None:
    paths = [path for path in BRIDGE_ROOT.glob("*") if path.is_file()]
    paths += list(SCRIPT_ROOT.glob("*.sh"))
    paths += [Path(__file__)]
    assert max(len(path.read_text().splitlines()) for path in paths) <= 500
    assert finite_tree({"tensor": torch.ones(2), "array": np.ones(2), "x": 1.0})
    assert not finite_tree({"missing": None})
    assert not finite_tree({"tensor": torch.tensor([float("inf")])})
    assert not finite_tree({"array": np.asarray([float("nan")])})


def test_jaccard_empty_rows_follow_frozen_identity_convention() -> None:
    left = np.asarray([[0, 0, 0], [1, 0, 1]], bool)
    right = np.asarray([[0, 0, 0], [1, 1, 0]], bool)
    assert jaccard_rows(left, right).tolist() == [1.0, 1.0 / 3.0]


def test_decision_requires_probe_dataset_wins_and_routing_interaction() -> None:
    card, _ = load_card()
    ratio = {
        "ratio_of_cell_means": 0.8, "bootstrap_interval": [0.7, 0.9],
        "model_seed_candidate_wins": 8, "dataset_seed_candidate_wins": 2,
    }
    difference = {
        "difference_mean": 0.2, "bootstrap_interval": [0.1, 0.3],
        "model_seed_candidate_wins": 8, "dataset_seed_candidate_wins": 2,
    }
    alignment = {
        **difference, "sparse_coverage_mean": 0.9,
        "minimum_dataset_sparse_coverage": 0.85, "sparse_ari_mean": 0.5,
    }
    probe = {**difference, "sparse_support_balanced_accuracy_mean": 0.8}
    aggregate = {
        "forecast": {
            horizon: {"through_mse": dict(ratio)} for horizon in ("160", "200")
        },
        "alignment": {"primary_h200": alignment},
        "probe": probe,
        "routing": {
            "all_cells_available": True,
            "minimum_dataset_coverage": 0.85,
            "by_horizon": {"200": {"through_mse": {
                "correct_over_wrong_restricted": dict(ratio),
                "correct_restricted_over_full": dict(ratio),
                "restriction_interaction": dict(ratio),
            }}},
        },
    }
    assert decide(aggregate, card)["branch"] == "same_checkpoint_mechanistic_bridge"
    aggregate["probe"]["model_seed_candidate_wins"] = 7
    assert decide(aggregate, card)["checks"]["support_probe"] is False
