"""Contracts for the authenticated Allen--Cahn new-IC evidence packet."""

from __future__ import annotations

import csv
import copy
import json

import numpy as np
import pytest

from experiments.neurips_2026.evidence.allen_cahn_forecast_replication import (
    DATASET_ROWS,
    DATASET_SEEDS,
    EVIDENCE_MANIFEST,
    EXPECTED_HASHES,
    FIGURE_PDF,
    FIGURE_PNG,
    MODEL_SEEDS,
    SEED_ROWS,
    SUMMARY,
    check_packet,
    validate_compact,
)
from experiments.neurips_2026.evidence.allen_cahn_forecast_replication_curves import (
    CURVE_ROWS,
    CURVE_SUMMARY,
    FULL_HORIZON_FIGURE_PDF,
    FULL_HORIZON_FIGURE_PNG,
    validate_curve_compact,
)


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _packet():
    return json.loads(SUMMARY.read_text()), _read_csv(SEED_ROWS), _read_csv(DATASET_ROWS)


def test_primary_new_ic_result_is_strong_and_recomputes_from_paired_seeds():
    summary, seed_rows, dataset_rows = _packet()
    primary = summary["primary"]
    dense = np.asarray([float(row["dense_h200_cumulative_field_mse"]) for row in seed_rows])
    sparse = np.asarray([float(row["sparse_h200_cumulative_field_mse"]) for row in seed_rows])

    assert [int(row["model_seed"]) for row in seed_rows] == list(MODEL_SEEDS)
    assert [int(row["dataset_seed"]) for row in dataset_rows] == list(DATASET_SEEDS)
    assert np.isclose(1.0 - sparse.mean() / dense.mean(), 0.058574119441734296)
    assert np.isclose(primary["ci95_lower"], 0.024028248853500533)
    assert np.isclose(primary["ci95_upper"], 0.09208249660573453)
    assert np.isclose(primary["one_sided_exact_sign_flip_p"], 0.0087890625)
    assert int((sparse < dense).sum()) == primary["paired_model_seed_wins"] == 8
    assert all(primary["strong_gate_checks"].values())
    np.testing.assert_allclose(
        [float(row["relative_reduction"]) for row in dataset_rows],
        [0.046778320592200995, 0.058791275800516574, 0.07030232235157208],
    )


def test_terminal_and_development_results_remain_separate_and_descriptive():
    summary, seed_rows, dataset_rows = _packet()
    validate_compact(summary, seed_rows, dataset_rows)

    terminal = summary["secondary"]["h200_terminal"]
    assert terminal["status"] == "descriptive_only_no_test_no_interval_no_rescue"
    assert np.isclose(terminal["relative_reduction_of_arm_means"], 0.03092028422002624)
    assert terminal["paired_model_seed_wins"] == 7
    development = summary["development_context"]
    assert development["dataset_seed"] == 20260724
    assert development["h200_cumulative_reduction_card_rounded"] == 0.0548
    assert np.isclose(development["h200_terminal_reduction"], 0.03224233607265487)
    assert development["four_cell_gate_passed"] is False
    assert "never pool" in development["comparison_policy"]
    assert "terminal superiority" in summary["not_supported"]
    assert "support-alignment mediation" in summary["not_supported"]


def test_source_chain_gpu_audit_and_claim_boundary_are_frozen():
    summary, _seed_rows, _dataset_rows = _packet()
    assert summary["authentication"]["source_artifact_sha256"] == EXPECTED_HASHES
    assert summary["authentication"]["crossed_model_by_dataset_cells"] == 60
    gpu = summary["gpu_evaluation"]
    assert gpu["status"] == "passed"
    assert np.isclose(gpu["mean_retained_utilization_percent"], 98.62406015037594)
    assert gpu["p10_retained_utilization_percent"] == 100.0
    assert gpu["retained_samples"] == 133
    assert gpu["zero_utilization_samples_retained"] == 1
    assert gpu["no_padding"] is True
    boundary = summary["claim_boundary"]
    assert "not retraining" in boundary
    assert "not physics or system generalization" in boundary
    assert "connect support alignment causally" in boundary


def test_compact_validator_rejects_terminal_promotion():
    summary, seed_rows, dataset_rows = _packet()
    tampered = copy.deepcopy(summary)
    tampered["secondary"]["h200_terminal"]["status"] = "confirmatory"
    with pytest.raises(ValueError, match="Terminal claim boundary"):
        validate_compact(tampered, seed_rows, dataset_rows)


def test_full_horizon_companion_preserves_curve_and_inference_boundaries():
    summary = json.loads(CURVE_SUMMARY.read_text())
    rows = _read_csv(CURVE_ROWS)
    validate_curve_compact(summary, rows)

    assert [int(row["horizon_step"]) for row in rows] == list(range(1, 201))
    np.testing.assert_allclose(
        [float(row["physical_time"]) for row in rows],
        0.1 * np.arange(1, 201),
    )
    h200 = rows[-1]
    assert "dense_mean_cumulative_field_mse" not in h200
    assert "sparse_mean_cumulative_field_mse" not in h200
    assert np.isclose(
        float(h200["dense_mean_through_horizon_mean_field_mse"]),
        0.04902859565677742,
    )
    assert np.isclose(
        float(h200["sparse_mean_through_horizon_mean_field_mse"]),
        0.04615678883871684,
    )
    assert np.isclose(float(h200["relative_reduction_of_arm_means"]), 0.058574119441734296)
    assert np.isclose(float(h200["relative_reduction_pointwise_ci95_lower"]), 0.023522097826444446)
    assert np.isclose(float(h200["relative_reduction_pointwise_ci95_upper"]), 0.09195290754542729)
    assert "not_simultaneous" in summary["curve"]["pointwise_interval_role"]
    assert summary["primary_h200"]["ci95_lower"] == 0.024028248853500533
    assert summary["primary_h200"]["ci95_upper"] == 0.09208249660573453
    assert "outcome_aware" in summary["primary_h200"]["role"]
    assert "separate_from_curve_bands" in summary["primary_h200"]["role"]
    assert "preregistered" not in json.dumps(summary).lower()
    metric = summary["metric_schema"]["through_horizon_mean_field_mse"]
    assert "divided by h" in metric["definition"]
    assert metric["not_an_unnormalized_cumulative_sum"] is True
    dense_instantaneous = np.asarray(
        [float(row["dense_mean_instantaneous_field_mse"]) for row in rows]
    )
    sparse_instantaneous = np.asarray(
        [float(row["sparse_mean_instantaneous_field_mse"]) for row in rows]
    )
    signed_gap = dense_instantaneous - sparse_instantaneous
    trace = summary["full_trace_descriptive"]
    assert trace["sparse_arm_mean_lower_count"] == int((signed_gap > 0).sum()) == 200
    assert trace["sparse_arm_mean_lower_at_all_evaluated_times"] is True
    assert np.isclose(trace["signed_gap_sum_steps_1_through_200"], 0.5743613636121154)
    assert np.isclose(trace["signed_gap_sum_after_physical_time_2"], 0.39174209764848156)
    assert np.isclose(
        trace["signed_gap_share_after_physical_time_2"],
        signed_gap[20:].sum() / signed_gap.sum(),
    )
    assert np.isclose(
        trace["signed_gap_share_after_physical_time_2"],
        0.6820481363593904,
    )
    assert "four-cell gate failed" in summary["figure_disclosure"]
    assert "same-checkpoint new-IC robustness check" in summary["figure_title"]
    assert "replication" not in summary["figure_title"].lower()
    assert summary["source"]["scientific_payload_sha256"] == (
        "4c536871e71f47fd055db057da8c1c4a1213a0aceee9687ddb1c14dbc8963cf0"
    )


def test_manifest_and_publication_rendering_are_authenticated_and_deterministic():
    manifest = json.loads(EVIDENCE_MANIFEST.read_text())
    assert manifest["source_artifacts"] == EXPECTED_HASHES
    assert set(manifest["outputs"]) == {
        "summary",
        "seed_rows",
        "dataset_rows",
        "figure_pdf",
        "figure_png",
        "curve_summary",
        "curve_rows",
        "full_horizon_figure_pdf",
        "full_horizon_figure_png",
    }
    assert FIGURE_PDF.stat().st_size > 20_000
    assert FIGURE_PNG.stat().st_size > 100_000
    assert FULL_HORIZON_FIGURE_PDF.stat().st_size > 15_000
    assert FULL_HORIZON_FIGURE_PNG.stat().st_size > 100_000
    check_packet()
