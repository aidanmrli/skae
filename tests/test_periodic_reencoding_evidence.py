"""Outcome-agnostic contracts for the Allen--Cahn periodic evidence builder."""

from __future__ import annotations

import ast
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from experiments.neurips_2026.cli import COMMANDS
from experiments.neurips_2026.evidence import (
    allen_cahn_periodic_reencoding_contract as contract,
)
from experiments.neurips_2026.evidence.allen_cahn_periodic_reencoding import (
    build,
    check_packet,
)
from experiments.neurips_2026.evidence.allen_cahn_periodic_reencoding_contract import (
    CARD,
    EXPECTED_CARD_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    SOURCE_MANIFEST,
    TABLE_ROW_IDS,
    VISUALIZATION_PLAN,
    authenticate_summary,
    read_json,
    sha256_path,
    verify_frozen_sources,
)
from experiments.neurips_2026.evidence.allen_cahn_periodic_reencoding_reduction import (
    build_compact,
    validate_compact,
)
from experiments.neurips_2026.evidence.allen_cahn_periodic_reencoding_rendering import (
    comparison_table_bytes,
    render_periodic_figure,
)


def _cross(dense: float, sparse: float) -> dict[str, object]:
    dense_values = [dense * (0.91 + 0.02 * index) for index in range(10)]
    sparse_values = [sparse * (0.91 + 0.02 * index) for index in range(10)]
    reduction = 1.0 - sum(sparse_values) / sum(dense_values)
    return {
        "dense_paired_seed_values": dense_values,
        "sparse_paired_seed_values": sparse_values,
        "dense_mean": sum(dense_values) / 10,
        "sparse_mean": sum(sparse_values) / 10,
        "relative_reduction_of_arm_means": reduction,
        "sparse_seed_wins": sum(sparse < dense for sparse, dense in zip(sparse_values, dense_values)),
        "paired_ratio_bootstrap": {
            "ci95_lower": reduction - 0.02, "ci95_upper": reduction + 0.02,
        },
        "exact_one_sided_studentized_sign_flip": {"one_sided_exact_p": 0.01},
    }


def _within(direct: float, selected: float) -> dict[str, object]:
    direct_values = [direct * (0.91 + 0.02 * index) for index in range(10)]
    selected_values = [selected * (0.91 + 0.02 * index) for index in range(10)]
    reduction = 1.0 - sum(selected_values) / sum(direct_values)
    return {
        "direct_paired_seed_values": direct_values,
        "selected_paired_seed_values": selected_values,
        "direct_mean": sum(direct_values) / 10,
        "selected_mean": sum(selected_values) / 10,
        "relative_reduction_of_arm_means": reduction,
        "selected_seed_wins": sum(selected < direct for selected, direct in zip(selected_values, direct_values)),
        "paired_ratio_bootstrap": {
            "ci95_lower": reduction - 0.02, "ci95_upper": reduction + 0.02,
        },
    }


def _aware(effect: float, cadence: int) -> dict[str, object]:
    return {
        "selected_cadence": cadence,
        "heldout_point_relative_reduction": effect,
        "heldout_point_selected_seed_wins": 9,
        "selection_aware_bootstrap_ci95_lower": effect - 0.02,
        "selection_aware_bootstrap_ci95_upper": effect + 0.02,
    }


def _full_endpoint(dense: float, sparse: float) -> dict[str, object]:
    point = _cross(dense, sparse)
    reduction = float(point["relative_reduction_of_arm_means"])
    return {
        "point_test": point,
        "selection_aware_pipeline_inference": {
            "paired_seed_bootstrap": {
                "ci95_lower": reduction - 0.025,
                "ci95_upper": reduction + 0.025,
            },
            "exact_one_sided_studentized_arm_swap": {"one_sided_exact_p": 0.02},
        },
        "selection_aware_within_arm_selected_vs_direct": {
            "dense": _aware(0.04, 10), "sparse": _aware(0.07, 20),
        },
    }


def _scores(selected: int, scale: float) -> list[dict[str, object]]:
    cadences = ["direct", 1, 2, 5, 10, 20, 25, 50, 100]
    return [
        {"cadence": cadence, "h200_cumulative_field_mse": scale + (0 if cadence == selected else 0.01 + 0.001 * index)}
        for index, cadence in enumerate(cadences)
    ]


def _source_summary() -> dict[str, object]:
    selected = {"dense": 10, "sparse": 20}
    h400, tail = _cross(0.30, 0.27), _cross(0.40, 0.35)
    direct_h400, direct_tail = _cross(0.32, 0.30), _cross(0.43, 0.40)
    conditional = {
        "dense": {"endpoints": {"h201_h400_tail_field_mse": _within(0.43, 0.40)}},
        "sparse": {"endpoints": {"h201_h400_tail_field_mse": _within(0.40, 0.35)}},
    }
    return {
        "selection_lineage": {"selected_cadences": selected},
        "confirmatory_h200_endpoint": _cross(0.20, 0.18),
        "same_cadence_h200_sensitivities": {"direct": _cross(0.22, 0.205)},
        "pipeline_inference_h200": {
            "validation_candidate_scores": {
                "dense": _scores(10, 0.20), "sparse": _scores(20, 0.18),
            },
            "within_arm_selected_vs_direct_h200": {
                "dense": _within(0.22, 0.20), "sparse": _within(0.205, 0.18),
            },
        },
        "periodic_policy_h200": {"selection_aware_pipeline_bootstrap": {"arms": {
            "dense": _aware(1.0 - 0.20 / 0.22, 10),
            "sparse": _aware(1.0 - 0.18 / 0.205, 20),
        }}},
        "conditional_h400_stress_summary": {
            "selected_recipe_comparison": {
                "dense_cadence": 10, "sparse_cadence": 20,
                "endpoints": {
                    "h400_cumulative_field_mse": h400,
                    "h201_h400_tail_field_mse": tail,
                },
            },
            "same_cadence_sensitivity": {"direct": {
                "dense_cadence": "direct", "sparse_cadence": "direct",
                "endpoints": {
                    "h400_cumulative_field_mse": direct_h400,
                    "h201_h400_tail_field_mse": direct_tail,
                },
            }},
        },
        "conditional_h400_selected_vs_direct": {"arms": conditional},
        "selection_aware_full_grid_h400": {"endpoints": {
            "h400_cumulative_field_mse": _full_endpoint(0.30, 0.27),
            "h201_h400_tail_field_mse": {
                **_full_endpoint(0.40, 0.35),
                "selection_aware_within_arm_selected_vs_direct": {
                    "dense": _aware(1.0 - 0.40 / 0.43, 10),
                    "sparse": _aware(1.0 - 0.35 / 0.40, 20),
                },
            },
        }},
        "decision": {"branch": "synthetic_full_branch", "claim_boundary": "joint recipe only"},
        "truth_difficulty": {"status": "synthetic"},
    }


def _curves(full: bool = True) -> dict[str, object]:
    def tier(horizon: int) -> dict[str, object]:
        records = []
        for arm, base in (("dense", 0.20), ("sparse", 0.18)):
            for cadence in ("direct", 10 if arm == "dense" else 20):
                values = [base + 0.0002 * step + (0.01 if cadence == "direct" else 0) for step in range(horizon)]
                records.append({"arm": arm, "cadence": cadence, "mean_cumulative_field_mse": values})
        return {
            "tier": f"synthetic_h{horizon}", "horizon_steps": horizon,
            "time_step": 0.1, "records": records,
            "complete_curve_names": ["cumulative_field_mse"],
        }
    frontier = {
        "endpoints": {"h400_cumulative_field_mse": [
            {
                "cadence": cadence,
                "refresh_count": 0 if cadence == "direct" else (399 // int(cadence)),
                "dense_arm_mean_mse": 0.30 + 0.0001 * index,
                "sparse_arm_mean_mse": 0.27 + 0.0001 * index,
            }
            for index, cadence in enumerate(["direct", 1, 2, 5, 10, 20, 25, 50, 100])
        ]}
    }
    return {"h200": tier(200), "h400": tier(400) if full else None, "accuracy_refresh_frontier_h400": frontier if full else None}


def _bundle(full: bool = True) -> dict[str, object]:
    source = _source_summary()
    if not full:
        source["conditional_h400_stress_summary"] = None
        source["conditional_h400_selected_vs_direct"] = None
        source["selection_aware_full_grid_h400"] = None
    digest = "a" * 64
    return {
        "summary": source, "curves": _curves(full),
        "telemetry": {
            "status": "passed", "both_disjoint_windows_passed": True,
            "zeros_retained": True, "no_padding": True, "windows": {},
        },
        "receipt_sha256": "b" * 64,
        "receipt": {
            "outcome_guard_receipt_sha256": digest,
            "scientific_payload_sha256": digest, "summary_sha256": digest,
            "decision_sha256": digest, "seed_rows_sha256": digest,
            "forecast_curve_sha256": digest,
        },
        "guard": {
            "telemetry_audit_sha256": digest, "runtime_lineage_sha256": digest,
            "raw_telemetry_sha256": digest,
        },
        "source_paths": {"summary": "/scratch/frozen/summary/summary.json"},
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_authenticated_source(
    root: Path, smoke_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> str:
    card = deepcopy(read_json(CARD))
    card["prospective_datasets"]["output_root"] = str(root)
    card["outcome_free_smoke"]["output_root"] = str(smoke_root)
    monkeypatch.setattr(
        contract, "verify_frozen_sources", lambda: (card, {"synthetic": "fixture"})
    )
    smoke = smoke_root / "smoke_receipt.json"
    _write_json(
        smoke,
        {
            "status": "passed_outcome_free_gpu_smoke",
            "card_sha256": EXPECTED_CARD_SHA256,
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "scientific_outcomes_accessed": False,
        },
    )
    source_files = {}
    for name in (
        "selection_decision", "validation_data_manifest", "test_data_manifest",
        "scientific_payload", "runtime_lineage",
    ):
        path = root / f"{name}.json"
        _write_json(path, {"fixture": name})
        source_files[name] = path
    scientific_sha = sha256_path(source_files["scientific_payload"])
    raw = root / "raw_gpu_telemetry.csv"
    raw.write_text("timestamp,utilization\n0,95\n", encoding="utf-8")
    window = {
        "boundary_samples_excluded_per_side": 0,
        "utilization_filter_applied": False,
        "all_window_samples": 3,
        "retained_all_window_samples": 3,
        "mean_retained_all_window_gpu_utilization_percent": 95.0,
        "p10_retained_all_window_gpu_utilization_percent": 0.0,
        "median_sample_cadence_seconds": 75.0,
        "maximum_sample_gap_seconds": 180.0,
        "leading_marker_edge_gap_seconds": 30.0,
        "trailing_marker_edge_gap_seconds": 45.0,
        "peak_memory_fraction": 0.5,
        "zero_utilization_retained_samples": 0,
    }
    windows = {name: dict(window) for name in contract.WINDOW_NAMES}
    checks = {
        name: contract._telemetry_checks(value, card["hardware_plan"])
        for name, value in windows.items()
    }
    audit = {
        "status": "passed",
        "card_sha256": EXPECTED_CARD_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "scientific_payload_opened": False,
        "forecast_outcomes_accessed": False,
        "both_disjoint_windows_required": True,
        "disjoint_window_ordering_verified": True,
        "evaluation_end_epoch_identity_verified": True,
        "every_retained_sample_including_zero_used": True,
        "slurm_job_id": "10170004",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000004",
        "validity_windows": windows,
        "validity_checks": checks,
    }
    audit_path = root / "telemetry_audit.json"
    _write_json(audit_path, audit)
    guard = {
        "status": "authorized_for_dependent_cpu_summary",
        "card_sha256": EXPECTED_CARD_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "scientific_payload_sha256": scientific_sha,
        "scientific_payload_opened": False,
        "forecast_outcomes_accessed": False,
        "both_disjoint_windows_required": True,
        "disjoint_window_ordering_verified": True,
        "evaluation_end_epoch_identity_verified": True,
        "slurm_job_id": audit["slurm_job_id"], "gpu_uuid": audit["gpu_uuid"],
        "validity_windows": windows, "validity_checks": checks,
        "checkpoint_roster": [
            {"arm": arm, "seed": seed}
            for arm in ("dense", "sparse") for seed in range(64, 74)
        ],
        "smoke_receipt_path": str(smoke),
        "smoke_receipt_sha256": sha256_path(smoke),
        "telemetry_audit_path": str(audit_path),
        "telemetry_audit_sha256": sha256_path(audit_path),
        "raw_telemetry_path": str(raw), "raw_telemetry_sha256": sha256_path(raw),
    }
    for name, path in source_files.items():
        guard[f"{name}_path"] = str(path)
        guard[f"{name}_sha256"] = (
            scientific_sha if name == "scientific_payload" else sha256_path(path)
        )
    guard_path = root / "outcome_guard_receipt.json"
    _write_json(guard_path, guard)
    guard_sha = sha256_path(guard_path)
    curves = _curves()
    curves.update({
        "schema_version": 1, "card_sha256": EXPECTED_CARD_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "outcome_guard_receipt_sha256": guard_sha,
        "scientific_payload_sha256": scientific_sha,
    })
    summary_dir = root / "summary"
    curve_path = summary_dir / "forecast_curves_and_frontier.json"
    _write_json(curve_path, curves)
    summary = _source_summary()
    summary.update({
        "card_sha256": EXPECTED_CARD_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "outcome_guard_receipt_sha256": guard_sha,
        "forecast_curve_artifact": {
            "path": str(curve_path), "sha256": sha256_path(curve_path),
        },
    })
    summary_path, decision_path = summary_dir / "summary.json", summary_dir / "decision.json"
    _write_json(summary_path, summary)
    _write_json(decision_path, {
        "selected_cadences": summary["selection_lineage"]["selected_cadences"],
        "confirmatory_h200_endpoint": summary["confirmatory_h200_endpoint"],
        "decision": summary["decision"],
    })
    seed_path = summary_dir / "seed_rows.csv"
    endpoints = {"h200_selection_aware_primary": summary["confirmatory_h200_endpoint"]}
    endpoints.update(summary["conditional_h400_stress_summary"]["selected_recipe_comparison"]["endpoints"])
    with seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model_seed", "endpoint", "dense_mse", "sparse_mse", "reduction"])
        for endpoint_name, endpoint in endpoints.items():
            for offset, (dense, sparse) in enumerate(zip(
                endpoint["dense_paired_seed_values"], endpoint["sparse_paired_seed_values"], strict=True,
            )):
                writer.writerow([64 + offset, endpoint_name, format(dense, ".17g"), format(sparse, ".17g"), format(1.0 - sparse / dense, ".17g")])
    receipt = {
        "schema_version": 1, "card_sha256": EXPECTED_CARD_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "outcome_guard_receipt_sha256": guard_sha,
        "scientific_payload_sha256": scientific_sha,
    }
    for name, path in (
        ("summary", summary_path), ("decision", decision_path),
        ("seed_rows", seed_path), ("forecast_curve", curve_path),
    ):
        receipt[f"{name}_path"] = str(path)
        receipt[f"{name}_sha256"] = sha256_path(path)
    receipt_path = summary_dir / "summary_receipt.json"
    _write_json(receipt_path, receipt)
    return sha256_path(receipt_path)


def test_v5_source_hashes_v4_rejection_and_cli_registration() -> None:
    card, manifest = verify_frozen_sources()
    assert card["execution_attempt_id"] == "allen_cahn_periodic_reencoding_clean_rerun_v5"
    assert CARD.parent.name == "allen_cahn_periodic_reencoding_v5"
    assert SOURCE_MANIFEST.parent == CARD.parent
    assert card["visualization_plan"] == list(VISUALIZATION_PLAN)
    assert len(manifest) == len(card["source_and_outcome_guard"]["required_manifest_paths"])
    assert COMMANDS[("build", "allen-cahn-periodic-reencoding")].endswith(
        "evidence.allen_cahn_periodic_reencoding"
    )
    assert "<v5-root>" in (CARD.parents[1] / "README.md").read_text()
    v4 = read_json(
        CARD.parents[1] / "allen_cahn_periodic_reencoding_v4/prediction_card.json"
    )
    with pytest.raises(ValueError, match="frozen V5 card"):
        authenticate_summary(Path(v4["prospective_datasets"]["output_root"]), "0" * 64)


def test_authentication_boundary_imports_no_scientific_modules() -> None:
    evidence_dir = CARD.parents[1] / "evidence"
    for name in (
        "allen_cahn_periodic_reencoding.py",
        "allen_cahn_periodic_reencoding_contract.py",
        "allen_cahn_periodic_reencoding_reduction.py",
        "allen_cahn_periodic_reencoding_rendering.py",
    ):
        tree = ast.parse((evidence_dir / name).read_text(encoding="utf-8"))
        modules = {
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(
            module.startswith(
                "experiments.neurips_2026.allen_cahn_periodic_reencoding"
            )
            for module in modules
        )


def test_complete_branch_has_exact_four_panels_and_ten_table_rows(tmp_path: Path) -> None:
    summary, rows = build_compact(_bundle())
    assert summary["source_authentication"]["card_sha256"] == EXPECTED_CARD_SHA256
    assert summary["source_authentication"]["source_manifest_sha256"] == EXPECTED_SOURCE_MANIFEST_SHA256
    assert summary["fixed_display_contract"]["outcome_dependent_row_or_panel_selection"] is False
    assert [row["row_id"] for row in rows] == list(TABLE_ROW_IDS)
    assert all(row["status"] != "unavailable_frozen_tier" for row in rows)
    assert rows[0]["inference_role"] == "confirmatory_selection_aware_primary"
    assert all(row["one_sided_p"] is None for row in rows[6:])
    assert all(len(values) == 10 for values in summary["display_payload"]["paired_sparse_over_dense_ratios"].values())
    table = comparison_table_bytes(rows)
    assert table.count(b" \\\\") == 12
    pdf, png = tmp_path / "first.pdf", tmp_path / "first.png"
    pdf_again, png_again = tmp_path / "second.pdf", tmp_path / "second.png"
    render_periodic_figure(summary, pdf, png)
    render_periodic_figure(summary, pdf_again, png_again)
    assert pdf.read_bytes() == pdf_again.read_bytes()
    assert png.read_bytes() == png_again.read_bytes()
    assert pdf.stat().st_size > 10_000 and png.stat().st_size > 50_000


def test_authenticate_build_and_portable_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, smoke = tmp_path / "source", tmp_path / "smoke"
    receipt_sha = _write_authenticated_source(source, smoke, monkeypatch)
    authenticated = authenticate_summary(source, receipt_sha)
    assert authenticated["receipt_sha256"] == receipt_sha
    assert authenticated["telemetry"]["both_disjoint_windows_passed"] is True
    assert "maximum_sample_gap" in authenticated["telemetry"]["descriptive_only_metrics"]
    data, figures, tables = (
        tmp_path / "data", tmp_path / "figures", tmp_path / "tables"
    )
    outputs = build(source, receipt_sha, data, figures, tables)
    check_packet(data, figures, tables)
    manifest = read_json(outputs["manifest"])
    assert "reduction_sha256" in manifest["builder_sources"]
    assert manifest["source_authentication"] == read_json(outputs["summary"])[
        "source_authentication"
    ]


def test_h400_failure_keeps_roster_and_marks_unavailable(tmp_path: Path) -> None:
    summary, rows = build_compact(_bundle(full=False))
    assert [row["row_id"] for row in rows] == list(TABLE_ROW_IDS)
    unavailable = [row["row_id"] for row in rows if row["status"] == "unavailable_frozen_tier"]
    assert unavailable == list(TABLE_ROW_IDS[2:6]) + list(TABLE_ROW_IDS[8:10])
    assert summary["display_payload"]["heldout_curves"]["h400_available"] is False
    assert summary["display_payload"]["frontier_status"] == "unavailable_frozen_tier"
    assert b"Unavailable" in comparison_table_bytes(rows)
    render_periodic_figure(summary, tmp_path / "failure.pdf", tmp_path / "failure.png")


def test_compact_validator_rejects_cherry_picking_and_promoted_failure() -> None:
    summary, rows = build_compact(_bundle(full=False))
    with pytest.raises(ValueError, match="row order"):
        validate_compact(summary, rows[:-1])
    tampered = deepcopy(rows)
    tampered[2]["relative_reduction"] = 0.5
    with pytest.raises(ValueError, match="Unavailable row"):
        validate_compact(summary, tampered)
    changed_panels = deepcopy(summary)
    changed_panels["fixed_display_contract"]["visualization_plan"] = ["best result"]
    with pytest.raises(ValueError, match="Panel plan"):
        validate_compact(changed_panels, rows)
    promoted = deepcopy(rows)
    promoted[1]["inference_role"] = "confirmatory_selection_aware_primary"
    with pytest.raises(ValueError, match="descriptive"):
        validate_compact(summary, promoted)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":2}\n')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_json(path)
