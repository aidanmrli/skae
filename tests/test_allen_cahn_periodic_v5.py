"""Outcome-blind regression tests for periodic-reencoding execution V5."""

from __future__ import annotations

from pathlib import Path

from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    duplicate_safe_json,
    parse_source_manifest,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v4 import guard as v4_guard
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v5 import guard
from experiments.neurips_2026.allen_cahn_periodic_reencoding_v5.telemetry_policy import (
    REQUIRED_HARDWARE_PLAN,
    gate_checks,
    install,
    window_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    ROOT / f"experiments/neurips_2026/allen_cahn_periodic_reencoding{suffix}"
    for suffix in ("", "_v2", "_v3", "_v4", "_v5")
]
V1, V2, V3, V4, V5 = PACKAGES
SCRIPTS = ROOT / "scripts/neurips_2026/allen_cahn_periodic_reencoding_v5"


def _sample(epoch: float, utilization: float, memory_fraction: float = 0.25) -> dict:
    return {
        "epoch_seconds": epoch,
        "gpu_uuid": "GPU-49925033-cf14-40cd-4a4f-12fad21d1629",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "utilization_percent": utilization,
        "memory_used_mib": 81920.0 * memory_fraction,
        "memory_total_mib": 81920.0,
    }


def _plan() -> dict:
    return duplicate_safe_json(V5 / "prediction_card.json")["hardware_plan"]


def test_v1_through_v4_remain_byte_frozen() -> None:
    expected = {
        V1: (
            "97716bb6dc2c0e6a4f8389d362c06ad67045ac2a85574c794f1966a388ce9e17",
            "a4729c04d1981c031d1531304668dc01b4165c1d5b1610d4d780d9730d123c6f",
        ),
        V2: (
            "e5af3746ae9a537f4c7860221228c9f39fc92acd7ce6442b85cd48d1f35cab4f",
            "0fa8b1994c1634164224441c3baf19f48314fb9db9ae89472c2802d491d82265",
        ),
        V3: (
            "a8636606f3248135759efe18f82b6dd62c95ad0d731f979611bf2093bdee5f48",
            "6e738758bcb1fc6d3041e0cd46696588a668085df53e722e7a5203af54ad8a68",
        ),
        V4: (
            "9d2374f986164941771af076576046358bce6bdbf19501a10a079d84081bc6f7",
            "015a305244ba0fc037b5af8d09fc3e15b633991f6f628d3ee7d27ac6a28dd593",
        ),
    }
    for package, (card_hash, source_hash) in expected.items():
        assert sha256_path(package / "prediction_card.json") == card_hash
        assert sha256_path(package / "source_manifest.sha256") == source_hash


def test_v5_scientific_protocol_is_exactly_v4() -> None:
    v4 = duplicate_safe_json(V4 / "prediction_card.json")
    v5 = duplicate_safe_json(V5 / "prediction_card.json")
    scientific_keys = (
        "status", "experiment_type", "question", "hypothesis",
        "scientific_distinction", "prior_outcome_aware_context", "frozen_parent",
        "roster", "system", "cadence_selection", "test_evaluation",
        "estimand_and_inference", "interpretation_branches", "visualization_plan",
    )
    assert all(v5[key] == v4[key] for key in scientific_keys)
    assert all(
        key == "output_root" or v5["prospective_datasets"][key] == value
        for key, value in v4["prospective_datasets"].items()
    )
    non_monitoring_hardware = {
        "partition", "device_name", "visible_gpu_count", "host_memory", "cpus",
        "time_limit", "validation_batch", "heldout_batch",
        "maximum_latent_segment", "maximum_peak_memory_fraction", "no_padding",
        "telemetry_windows",
    }
    assert all(v5["hardware_plan"][key] == v4["hardware_plan"][key]
               for key in non_monitoring_hardware)
    assert v5["outcome_free_smoke"]["workload"] == v4["outcome_free_smoke"]["workload"]
    assert v5["outcome_free_smoke"]["forbidden"] == v4["outcome_free_smoke"]["forbidden"]
    history = v5["outcome_blind_v4_operational_history"]
    assert history["scientific_choices_changed_in_v5"] is False
    assert history["scientific_jobs_submitted"] == 0
    assert history["scientific_outcomes_accessed"] is False


def test_v5_monitoring_contract_is_sparse_natural_and_non_brittle() -> None:
    plan = _plan()
    assert plan["telemetry_interval_seconds"] == 60
    assert plan["boundary_samples_excluded_per_side"] == 0
    assert plan["minimum_all_window_samples_before_boundary_exclusion"] == 3
    assert plan["minimum_retained_all_window_samples"] == 3
    assert plan["minimum_mean_retained_all_window_gpu_utilization_percent"] == 90
    assert plan["maximum_peak_memory_fraction"] == 0.8
    assert plan["no_padding"] is True
    forbidden = {
        "strict_p10_retained_all_window_gpu_utilization_percent_above",
        "minimum_median_sample_cadence_seconds",
        "maximum_median_sample_cadence_seconds",
        "maximum_sample_gap_seconds",
        "maximum_marker_edge_gap_seconds",
    }
    assert forbidden.isdisjoint(plan)


def test_window_keeps_every_natural_sample_and_records_descriptives() -> None:
    samples = [_sample(5.0, 100.0), _sample(65.0, 100.0), _sample(305.0, 70.0)]
    window = window_statistics(samples, start=0.0, end=600.0)
    assert window["all_window_samples"] == 3
    assert window["retained_all_window_samples"] == 3
    assert window["boundary_samples_excluded_per_side"] == 0
    assert window["mean_retained_all_window_gpu_utilization_percent"] == 90.0
    assert window["p10_retained_all_window_gpu_utilization_percent"] < 80.0
    assert window["maximum_sample_gap_seconds"] == 240.0
    checks = gate_checks(window, _plan())
    assert all(checks.values())
    assert set(checks) == {
        "exact_unconditional_boundary_exclusion", "no_utilization_filter",
        "minimum_all_window_samples", "minimum_retained_all_window_samples",
        "minimum_mean_retained_utilization", "strict_peak_memory_fraction",
    }
    assert "strict_p10_retained_utilization" not in checks
    assert "median_sample_cadence" not in checks
    assert "maximum_sample_gap" not in checks
    assert "leading_marker_edge_coverage" not in checks
    assert "trailing_marker_edge_coverage" not in checks


def test_monitoring_still_fails_low_count_low_mean_and_memory() -> None:
    low_count = window_statistics(
        [_sample(10.0, 100.0), _sample(70.0, 100.0)], start=0.0, end=100.0
    )
    assert gate_checks(low_count, _plan())["minimum_all_window_samples"] is False
    low_mean = window_statistics(
        [_sample(10.0, 100.0), _sample(70.0, 80.0), _sample(90.0, 80.0)],
        start=0.0,
        end=100.0,
    )
    assert gate_checks(low_mean, _plan())["minimum_mean_retained_utilization"] is False
    high_memory = window_statistics(
        [_sample(10.0, 100.0, 0.8), _sample(70.0, 100.0), _sample(90.0, 100.0)],
        start=0.0,
        end=100.0,
    )
    assert gate_checks(high_memory, _plan())["strict_peak_memory_fraction"] is False


def test_v5_guard_reuses_the_strict_duplicate_safe_v4_guard() -> None:
    assert guard.validate_smoke_artifacts is v4_guard.validate_smoke_artifacts
    assert guard.validate_outcome_guard is v4_guard.validate_outcome_guard


def test_v5_wrappers_install_every_policy_hook() -> None:
    from experiments.neurips_2026.allen_cahn_periodic_reencoding import (
        smoke_audit as base_smoke_audit,
        telemetry as base_telemetry,
    )

    old_required = base_telemetry.REQUIRED_HARDWARE_PLAN
    old_telemetry_gate = base_telemetry.gate_checks
    old_telemetry_window = base_telemetry.window_statistics
    old_smoke_gate = base_smoke_audit.gate_checks
    old_smoke_window = base_smoke_audit.window_statistics
    try:
        install()
        assert base_telemetry.REQUIRED_HARDWARE_PLAN is REQUIRED_HARDWARE_PLAN
        assert base_telemetry.gate_checks is gate_checks
        assert base_telemetry.window_statistics is window_statistics
        assert base_smoke_audit.gate_checks is gate_checks
        assert base_smoke_audit.window_statistics is window_statistics
    finally:
        base_telemetry.REQUIRED_HARDWARE_PLAN = old_required
        base_telemetry.gate_checks = old_telemetry_gate
        base_telemetry.window_statistics = old_telemetry_window
        base_smoke_audit.gate_checks = old_smoke_gate
        base_smoke_audit.window_statistics = old_smoke_window


def test_v5_scripts_are_portable_fail_closed_and_science_reusing() -> None:
    sources = {path.name: path.read_text(encoding="utf-8") for path in SCRIPTS.glob("*.sh")}
    assert set(sources) == {
        "queue.sh", "run_generate_select_evaluate.sh", "run_smoke.sh", "run_summary.sh"
    }
    assert all("jq" not in source for source in sources.values())
    assert all("pmon" not in source and "process_account" not in source
               for source in sources.values())
    assert all("sleep 2" not in source for source in sources.values())
    assert all("#SBATCH --partition=long" in source for source in sources.values())
    gpu = sources["run_generate_select_evaluate.sh"]
    smoke = sources["run_smoke.sh"]
    assert "--loop=60" in gpu and "--loop=60" in smoke
    assert gpu.count("--loop=60") == 1 and smoke.count("--loop=60") == 1
    assert "allen_cahn_periodic_reencoding_v3.run" in gpu
    assert "allen_cahn_periodic_reencoding_v3.uuid_probe" in smoke
    assert "allen_cahn_periodic_reencoding_v5.telemetry" in gpu
    assert "allen_cahn_periodic_reencoding_v5.smoke_audit" in smoke
    assert all("allen_cahn_periodic_reencoding_v5.guard" in sources[name]
               for name in ("queue.sh", "run_generate_select_evaluate.sh", "run_smoke.sh", "run_summary.sh"))


def test_v5_roots_are_unique_and_lifecycle_is_fail_closed() -> None:
    cards = [duplicate_safe_json(package / "prediction_card.json") for package in PACKAGES]
    science = [Path(card["prospective_datasets"]["output_root"]) for card in cards]
    smoke = [Path(card["outcome_free_smoke"]["output_root"]) for card in cards]
    assert len(set(science + smoke)) == 10
    assert science[-1].name.endswith("_v5") and smoke[-1].name.endswith("_v5")
    for name in ("queue.sh", "run_generate_select_evaluate.sh", "run_smoke.sh"):
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert 'if [[ -e "${OUTPUT_ROOT}" ]]' in source


def test_v5_manifest_is_exact_v4_transitive_extension() -> None:
    v4_entries = set(parse_source_manifest(V4 / "source_manifest.sha256"))
    additions = {
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v4/source_manifest.sha256",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v5/__init__.py",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v5/guard.py",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v5/smoke_audit.py",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v5/telemetry.py",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v5/telemetry_policy.py",
        "experiments/neurips_2026/allen_cahn_periodic_reencoding_v5/prediction_card.json",
        "scripts/neurips_2026/allen_cahn_periodic_reencoding_v5/queue.sh",
        "scripts/neurips_2026/allen_cahn_periodic_reencoding_v5/run_generate_select_evaluate.sh",
        "scripts/neurips_2026/allen_cahn_periodic_reencoding_v5/run_smoke.sh",
        "scripts/neurips_2026/allen_cahn_periodic_reencoding_v5/run_summary.sh",
        "tests/test_allen_cahn_periodic_v5.py",
        "docs/archive/allen_cahn_periodic_reencoding_v4_smoke_telemetry_failure_20260721.md",
    }
    expected = v4_entries | additions
    card = duplicate_safe_json(V5 / "prediction_card.json")
    assert set(card["source_and_outcome_guard"]["required_manifest_paths"]) == expected
    assert set(parse_source_manifest(V5 / "source_manifest.sha256")) == expected
    locked = parse_source_manifest(V5 / "source_manifest.sha256")
    for relative, digest in locked.items():
        if relative.startswith("tests/"):
            continue
        assert sha256_path(ROOT / relative) == digest
