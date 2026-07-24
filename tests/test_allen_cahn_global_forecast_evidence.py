"""Contracts for the forecast-optimized Allen--Cahn evidence packet."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.neurips_2026.evidence.allen_cahn_global_forecast import (
    PACKET_ID,
    PROTOCOL,
    ROWS,
    artifact_manifest,
    load_inputs,
    validate_rows,
)
from experiments.neurips_2026.evidence.allen_cahn_global_forecast_statistics import (
    exact_max_t_sensitivity,
    summarize,
)


def test_active_packet_recomputes_reported_effects_and_failed_gate():
    rows, protocol, artifacts = load_inputs()
    statistics = summarize(rows, protocol, packet_id=PACKET_ID)
    cells = statistics["comparison"]["cells"]

    assert statistics["status"] == "confirmation_gate_failed_secondary_evidence"
    assert statistics["decision"] == "terminate_allen_cahn_tuning"
    assert statistics["sealed_holdout"] == "not_generated_or_opened"
    assert statistics["comparison"]["passed"] is False
    assert statistics["comparison"]["all_four_means_lower"] is True
    assert statistics["comparison"]["all_four_ci95_lower_bounds_above_zero"] is False

    np.testing.assert_allclose(
        cells["h160_field_mse"]["relative_reduction_of_means"],
        0.06304563287888199,
    )
    np.testing.assert_allclose(
        cells["h200_field_mse"]["relative_reduction_of_means"],
        0.05482836605708319,
    )
    assert cells["h160_field_mse"]["sparse_seed_wins"] == 8
    assert cells["h200_field_mse"]["sparse_seed_wins"] == 8
    assert cells["h160_final_field_mse"]["sparse_seed_wins"] == 7
    assert cells["h200_final_field_mse"]["sparse_seed_wins"] == 7
    assert protocol["arms"]["sparse_lista_alpha"] == 0.15
    assert protocol["arms"]["sparse_elementwise_sparsity_weight"] == 0.01
    assert protocol["arms"]["sparse_temporal_group_sparsity_weight"] == 0
    assert protocol["training"]["sequence_length"] == 200
    assert protocol["training"]["checkpoint_horizons"] == [160, 200]
    assert len(artifacts) == 20
    assert protocol["frozen_compact_evidence"]["architecture_audit_sha256"] == (
        "f414cffce5c37144891e93292dbf9a6d0c66165170b9a20c4cbd3a7674ff2421"
    )
    manifest = artifact_manifest(artifacts)
    assert len(manifest["runs"]) == 20
    assert all(len(run["checkpoint"]["sha256"]) == 64 for run in manifest["runs"])
    assert all(len(run["evaluation"]["sha256"]) == 64 for run in manifest["runs"])
    replacement = artifacts.loc[
        (artifacts["arm"] == "sparse") & (artifacts["seed"] == 69)
    ].iloc[0]
    assert replacement["slurm_job_id"] == 10157586


def test_packet_recomputes_multiplicity_and_qualification_guards():
    rows, protocol, _artifacts = load_inputs()
    statistics = summarize(rows, protocol, packet_id=PACKET_ID)
    max_t = statistics["max_t_sensitivity"]
    qualification = statistics["qualification"]

    np.testing.assert_allclose(
        max_t["H160 through-horizon mean"]["one_sided_max_t_fwer_adjusted_p"],
        0.013671875,
    )
    np.testing.assert_allclose(
        max_t["H200 through-horizon mean"]["one_sided_max_t_fwer_adjusted_p"],
        0.0185546875,
    )
    assert all(cell["swaps"] == 1024 for cell in max_t.values())
    assert qualification["all_runs_qualified"] is True
    assert qualification["dense_no_sparsity_audit_passed"] is True
    assert qualification["sparsity_guard_passed"] is True
    assert qualification["gpu_utilization_guard_passed"] is True
    assert qualification["sparse_near_zero_fraction_minimum"] > 0.39
    assert qualification["sparse_near_zero_fraction_mean"] > 0.43
    assert qualification["dense_near_zero_fraction_mean"] < 0.003
    assert qualification["gpu_utilization_minimum_percent"] > 90


def test_packet_requires_exact_arm_seed_horizon_completeness():
    rows, protocol, _artifacts = load_inputs()
    with pytest.raises(ValueError, match="one row per arm/seed/horizon"):
        validate_rows(rows.iloc[:-1], protocol)


def test_frozen_rows_hash_fails_closed(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    tampered = tmp_path / ROWS.name
    tampered.write_bytes(ROWS.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_inputs(tampered, PROTOCOL)


def test_exact_max_t_never_reports_adjusted_p_below_raw_p():
    rows, _protocol, _artifacts = load_inputs()
    result = exact_max_t_sensitivity(rows)
    assert all(
        cell["one_sided_max_t_fwer_adjusted_p"]
        >= cell["one_sided_exact_sign_flip_p"]
        for cell in result.values()
    )
