from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = (
    ROOT
    / "experiments"
    / "neurips_2026"
    / "allen_cahn_forecast_replication"
    / "prediction_card.json"
)
CARD = json.loads(CARD_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_phase_two_is_preregistered_and_attests_zero_scientific_runs() -> None:
    assert CARD["schema_version"] == 1
    assert CARD["status"].endswith("no_generation_or_evaluation_authorized")
    gate = CARD["implementation_and_launch_gate"]
    assert "do not submit" in gate["phase_2_scope"]
    assert gate["scientific_jobs_submitted"] == 0
    assert gate["datasets_generated"] == 0
    assert gate["checkpoints_evaluated"] == 0


def test_phase_two_amendment_preserves_phase_one_root_and_zero_run_chronology() -> None:
    amendments = CARD["amendment_history"]
    assert len(amendments) == 2
    amendment = amendments[0]
    assert amendment["superseded_phase_1_card_sha256"] == (
        "bdca8a8b89635415af22f714a629e179c9b9fc8fb825f13a0835641e90c2ba46"
    )
    assert "Independent adversarial review" in amendment["reason"]
    chronology = amendment["chronology_attestation"]
    for event in (
        "phase-2 implementation",
        "source freeze",
        "synthetic smoke tests",
        "trajectory generation",
        "checkpoint evaluation",
    ):
        assert event in chronology
    assert amendment["zero_run_attestation_at_amendment"] == {
        "scientific_jobs_submitted": 0,
        "datasets_generated": 0,
        "checkpoints_evaluated": 0,
        "scientific_outcomes_accessed": False,
    }
    assert "never silently replaces" in amendment["phase_1_preservation_policy"]
    telemetry_repair = amendments[1]
    assert telemetry_repair["superseded_phase_2_card_sha256"] == (
        "fb8540f9591dedde3a0f3688e207d2081d28cc297e7f6e074249b4ec70ae17dc"
    )
    assert telemetry_repair["superseded_phase_2_source_manifest_sha256"] == (
        "ebc7a453eeb2cc6eea6df218752d4fc44283253ec7a41b1a53a8d171935e5df5"
    )
    for defect in ("utilization==0", "matmul precision", "sign-flip"):
        assert defect in telemetry_repair["reason"]
    assert telemetry_repair["zero_run_attestation_at_amendment"] == {
        "scientific_jobs_submitted": 0,
        "datasets_generated": 0,
        "checkpoints_evaluated": 0,
        "scientific_outcomes_accessed": False,
    }


def test_dataset_seeds_follow_frozen_hash_derivation_and_are_disjoint() -> None:
    datasets = CARD["prospective_datasets"]
    derivation = datasets["seed_derivation"]
    seeds = []
    digests = []
    for index in range(3):
        key = (
            f"{derivation['namespace']}|{derivation['source_protocol_sha256']}|"
            f"dataset_{index}"
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        digests.append(digest)
        seeds.append(int(digest[:8], 16) & 0x7FFFFFFF)
    assert digests == derivation["digests"]
    assert seeds == datasets["seeds"]
    assert len(seeds) == len(set(seeds)) == 3
    assert not set(seeds) & set(datasets["excluded_dataset_seeds"])
    assert datasets["reserved_forbidden_seed"] not in seeds
    for seed, relative in zip(seeds, datasets["paths"], strict=True):
        assert str(seed) in relative
        assert "20260725" not in relative
    assert "20260725" not in datasets["output_root"]


def test_collision_audit_and_fail_closed_overwrite_policy_are_frozen() -> None:
    audit = CARD["prospective_datasets"]["collision_audit"]
    assert audit["performed_before_card_creation"] is True
    assert audit["repository_text_matches_per_seed"] == [0, 0, 0]
    assert audit["scratch_filename_matches_per_seed"] == [0, 0, 0]
    assert audit["output_root_existed"] is False
    assert "Fail closed" in audit["launch_policy"]
    assert "never overwrite" in audit["launch_policy"]
    phase_two = CARD["prospective_datasets"]["phase_2_pre_execution_collision_audit"]
    assert phase_two["output_root_existed"] is False
    assert phase_two["exact_dataset_targets_existed"] == [False, False, False]
    assert phase_two["scratch_filename_matches_per_seed_through_depth_four"] == [
        0,
        0,
        0,
    ]
    assert phase_two["unexpected_repository_text_matches"] == 0
    assert phase_two["reserved_path_accessed"] is False
    assert phase_two["scientific_jobs_submitted"] == 0
    assert phase_two["datasets_generated"] == 0
    assert phase_two["checkpoints_evaluated"] == 0


def test_exact_system_generator_and_field_only_contract_are_locked() -> None:
    generator = CARD["system_and_generator"]
    expected = {
        "grid_size": 16,
        "channels": 2,
        "state_dim": 512,
        "latent_dim": 2048,
        "diffusion": 0.005,
        "rk4_dt": 0.005,
        "substeps_per_observation": 20,
        "stored_dt": 0.1,
        "trajectory_length": 200,
        "stored_states": 201,
        "physical_horizon": 20.0,
        "trajectories_per_dataset": 256,
        "label_extra_observations": 0,
        "min_regions": 2,
        "max_regions": 3,
        "mask_temperature": 0.65,
        "low_frequency_cutoff": 3,
        "noise_scale": 0.03,
        "require_min_area_fraction": 0.08,
        "allen_cahn_beta": 8.0,
        "allen_cahn_reaction_strength": 1.0,
        "allen_cahn_center_radius": 1.5,
        "generator_batch_shape": [3, 256],
        "compile_step": False,
    }
    assert {key: generator[key] for key in expected} == expected
    assert generator["latent_dim"] >= 4 * generator["state_dim"]
    firewall = CARD["field_only_firewall"]
    assert firewall["label_or_count_inputs_used"] is False
    assert firewall["saved_field_shape"] == [256, 201, 16, 16, 2]
    assert firewall["exact_dataset_top_level_keys"] == ["fields", "split_indices", "metadata"]
    assert firewall["allowed_evaluator_inputs"] == ["fields", "split_indices.val"]
    assert {"label", "basin", "fate", "well", "center", "count"}.issubset(
        firewall["forbidden_key_fragments"]
    )


def test_all_compact_inputs_and_pinned_reference_sources_match_hashes() -> None:
    inputs = CARD["frozen_inputs"]
    for key in ("source_protocol", "source_statistics", "artifact_roster", "architecture_audit"):
        record = inputs[key]
        assert _sha256(ROOT / record["path"]) == record["sha256"]
    for record in inputs["pinned_sources"]:
        path = Path(record["path"])
        assert path.is_file()
        assert _sha256(path) == record["sha256"]
        assert record["historical_commit"] == "376de761431fe85f37f003ae878ba6efea8afb96"


def test_twenty_checkpoint_roster_exactly_matches_audited_csv() -> None:
    roster = CARD["checkpoint_roster"]
    with (ROOT / CARD["frozen_inputs"]["artifact_roster"]["path"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        source_rows = list(csv.DictReader(handle))
    expected = {
        (
            row["arm"],
            int(row["seed"]),
            int(row["checkpoint_step"]),
            row["checkpoint_path"],
            row["checkpoint_sha256"],
        )
        for row in source_rows
    }
    actual = {
        (
            row["arm"],
            int(row["seed"]),
            int(row["checkpoint_step"]),
            row["path"],
            row["sha256"],
        )
        for row in roster["runs"]
    }
    assert len(source_rows) == len(roster["runs"]) == len(expected) == len(actual) == 20
    assert actual == expected
    assert roster["existence_audit"]["files_present"] == 20
    assert roster["existence_audit"]["files_missing"] == 0
    assert set(roster["model_seeds"]) == set(range(64, 74))
    assert {(row["arm"], row["seed"]) for row in roster["runs"]} == {
        (arm, seed) for arm in ("dense", "sparse") for seed in range(64, 74)
    }


def test_primary_estimand_uses_paired_model_seeds_and_direct_rollout() -> None:
    evaluation = CARD["evaluation"]
    inference = CARD["estimand_and_inference"]
    assert evaluation["primary_horizon"] == 200
    assert evaluation["secondary_horizons"] == [160]
    assert "direct repeated-global-K" in evaluation["rollout"]
    assert "no periodic" in evaluation["rollout"]
    assert "nested prefix" in evaluation["rollout"]
    assert "ten paired model seeds" in inference["inference_unit"]
    assert inference["no_dataset_bootstrap"] is True
    assert "2^10" in inference["exact_test"]
    assert "literal floating comparison" in inference["exact_test"]
    assert "no added numerical tolerance" in inference["exact_test"]
    assert "100000" in inference["bootstrap"]
    assert "seed 20260720" in inference["bootstrap"]
    assert len(inference["strong_replication_gate"]) == 6
    assert "No secondary endpoint or curve point may rescue" in evaluation[
        "secondary_policy"
    ]
    curve_contract = evaluation["curve_contract"]
    assert "first average the three fixed datasets" in curve_contract[
        "visualization_reduction"
    ]
    assert "Never treat the 30" in curve_contract["visualization_reduction"]
    assert "50000" in curve_contract["pointwise_bootstrap_bands"]
    assert "seed 20260721" in curve_contract["pointwise_bootstrap_bands"]
    assert "not simultaneous" in curve_contract["inference_policy"]
    numerics = evaluation["numerics"]
    assert numerics["float32_matmul_precision"] == "high"
    assert numerics["expected_cuda_matmul_allow_tf32"] is True
    assert numerics["expected_cudnn_allow_tf32"] is True
    assert "torch.set_float32_matmul_precision('high')" in numerics[
        "precision_setup"
    ]


def test_every_result_branch_preserves_original_failure_disclosure() -> None:
    required = (
        "original four-cell gate remains failed",
        "both original terminal 95% paired-seed CIs crossed zero",
        "H200 terminal reduction was 3.22% with 7/10 seed wins",
        "missing both 5% and 8/10 gates",
        "outcome-aware, endpoint-specific prospective replication",
        "not a reclassification",
    )
    branches = CARD["decision_branches"]
    assert set(branches) == {
        "strong_replication",
        "directional_but_below_strong_gate",
        "null_or_reversal",
        "invalid",
    }
    for text in branches.values():
        for fragment in required:
            assert fragment in text


def test_gpu_plan_is_packed_and_has_strict_utilization_gates() -> None:
    hardware = CARD["hardware_plan"]
    assert hardware["partition"] == "long"
    assert "[3,256,201,16,16,2]" in hardware["gpu_job"]
    assert "all 20" in hardware["gpu_job"]
    assert hardware["boundary_samples_excluded_per_side"] == 1
    assert hardware["minimum_all_window_samples_before_boundary_exclusion"] == 12
    assert hardware["minimum_retained_all_window_samples"] == 10
    assert hardware["minimum_mean_retained_all_window_gpu_utilization_percent"] == 90.0
    assert hardware[
        "strict_p10_retained_all_window_gpu_utilization_percent_above"
    ] == 80.0
    assert hardware["minimum_median_sample_cadence_seconds"] == 0.5
    assert hardware["maximum_median_sample_cadence_seconds"] == 1.5
    assert hardware["maximum_sample_gap_seconds"] == 1.5
    assert hardware["maximum_marker_edge_gap_seconds"] == 1.5
    assert "including utilization==0" in hardware["telemetry_scope"]
    assert "one invariant GPU UUID" in hardware["telemetry_scope"]
    assert "same SLURM job ID" in hardware["telemetry_scope"]
    assert "No utilization-dependent filtering" not in hardware["telemetry_failure"]
    assert "no utilization-dependent filtering" in hardware["telemetry_failure"]
    assert "synthetic packed-versus-sequential" in hardware["outcome_free_smoke"]
    assert "Never add dummy" in hardware["no_padding"]


def test_runtime_receipt_requires_explicit_roster_gpu_and_job_lineage() -> None:
    lock = CARD["source_and_outcome_lock"]
    assert "explicit 20-row checkpoint" in lock["environment_receipt"]
    assert "single telemetry GPU UUID" in lock["environment_receipt"]
    assert "available matching SLURM job" in lock["environment_receipt"]
    assert "explicit 20-row checkpoint hash roster" in lock["outcome_guard"]
    assert "one GPU UUID" in lock["outcome_guard"]
    assert "SLURM job lineage" in lock["outcome_guard"]
