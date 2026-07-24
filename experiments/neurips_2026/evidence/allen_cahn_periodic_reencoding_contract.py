"""Authentication and outcome-invariant reduction for periodic evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.paths import REPO_ROOT


PACKET_ID = "allen_cahn_periodic_reencoding_confirmation"
CARD = REPO_ROOT / (
    "experiments/neurips_2026/allen_cahn_periodic_reencoding_v5/"
    "prediction_card.json"
)
SOURCE_MANIFEST = REPO_ROOT / (
    "experiments/neurips_2026/allen_cahn_periodic_reencoding_v5/"
    "source_manifest.sha256"
)
EXPECTED_CARD_SHA256 = (
    "f90861bc060a9f13461fb470fd7a12f428fe49c7e92533cae90c46cc4c0c5bf4"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "3e702edab74521c55ad1c78fa90170e49ff61a4653f441566b6760294c1dc278"
)
CADENCE_GRID: tuple[str | int, ...] = (
    "direct", 1, 2, 5, 10, 20, 25, 50, 100,
)
VISUALIZATION_PLAN = (
    "Validation cadence-risk curve for each arm with frozen selected markers and direct",
    "Held-out cumulative MSE versus physical time for direct and selected policies, with a vertical line at T=20",
    "Paired-seed sparse/dense ratios at H200, H400, and H201--H400 with the equality line",
    "Accuracy-versus-refresh-count or latency frontier so improved accuracy is not presented as free, including optional p=200 at H400",
)
TABLE_ROW_IDS = (
    "selected_sparse_vs_dense_h200",
    "direct_sparse_vs_dense_h200",
    "selected_sparse_vs_dense_h400",
    "direct_sparse_vs_dense_h400",
    "selected_sparse_vs_dense_h201_h400",
    "direct_sparse_vs_dense_h201_h400",
    "dense_selected_vs_direct_h200",
    "sparse_selected_vs_direct_h200",
    "dense_selected_vs_direct_h201_h400",
    "sparse_selected_vs_direct_h201_h400",
)
WINDOW_NAMES = ("selection_validity", "evaluation_validity")
HARDWARE_GATES = {
    "telemetry_interval_seconds": 60,
    "boundary_samples_excluded_per_side": 0,
    "minimum_all_window_samples_before_boundary_exclusion": 3,
    "minimum_retained_all_window_samples": 3,
    "minimum_mean_retained_all_window_gpu_utilization_percent": 90,
    "maximum_peak_memory_fraction": 0.8,
    "no_padding": True,
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(), object_pairs_hook=_pairs)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hash(path: Path, expected: str, label: str) -> None:
    _require(path.is_file(), f"Missing {label}: {path}")
    actual = sha256_path(path)
    _require(actual == expected, f"{label} SHA-256 mismatch: {actual}")


def verify_frozen_sources() -> tuple[dict[str, Any], dict[str, str]]:
    """Authenticate the V5 card and its immutable execution sources.

    Regression tests remain listed with their executed hashes for provenance,
    but are allowed to evolve after a terminal packet. They are not cluster
    payloads and cannot change the recorded scientific result.
    """

    _hash(CARD, EXPECTED_CARD_SHA256, "V5 prediction card")
    _hash(
        SOURCE_MANIFEST,
        EXPECTED_SOURCE_MANIFEST_SHA256,
        "V5 source manifest",
    )
    card = read_json(CARD)
    _require(
        card.get("execution_attempt_id")
        == "allen_cahn_periodic_reencoding_clean_rerun_v5",
        "V5 execution-attempt identity drifted",
    )
    _require(
        tuple(card.get("visualization_plan", ())) == VISUALIZATION_PLAN,
        "Frozen four-panel design drifted",
    )
    _require(
        tuple(card["cadence_selection"]["cadence_grid"]) == CADENCE_GRID,
        "Frozen cadence grid drifted",
    )
    _require(
        card["roster"]["model_seeds"] == list(range(64, 74)),
        "Paired model-seed roster drifted",
    )
    manifest: dict[str, str] = {}
    for line_number, line in enumerate(SOURCE_MANIFEST.read_text().splitlines(), 1):
        parts = line.split("  ", 1)
        _require(len(parts) == 2, f"Malformed source-manifest line {line_number}")
        expected, relative = parts
        _require(relative not in manifest, f"Duplicate manifest path: {relative}")
        path = REPO_ROOT / relative
        if not relative.startswith("tests/"):
            _hash(path, expected, f"frozen source {relative}")
        manifest[relative] = expected
    required = set(card["source_and_outcome_guard"]["required_manifest_paths"])
    _require(set(manifest) == required, "V5 source-manifest roster drifted")
    return card, manifest


def _bound_path(root: Path, value: object, relative: str, label: str) -> Path:
    expected = (root / relative).resolve()
    observed = Path(str(value)).resolve()
    _require(observed == expected, f"{label} escaped the frozen source root")
    return expected


def _telemetry_checks(window: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "exact_unconditional_boundary_exclusion": (
            window["boundary_samples_excluded_per_side"]
            == plan["boundary_samples_excluded_per_side"] == 0
        ),
        "no_utilization_filter": window["utilization_filter_applied"] is False,
        "minimum_all_window_samples": int(window["all_window_samples"]) >= int(plan["minimum_all_window_samples_before_boundary_exclusion"]),
        "minimum_retained_all_window_samples": int(window["retained_all_window_samples"]) >= int(plan["minimum_retained_all_window_samples"]),
        "minimum_mean_retained_utilization": float(window["mean_retained_all_window_gpu_utilization_percent"]) >= float(plan["minimum_mean_retained_all_window_gpu_utilization_percent"]),
        "strict_peak_memory_fraction": float(window["peak_memory_fraction"]) < float(plan["maximum_peak_memory_fraction"]),
    }


def _validate_telemetry(
    root: Path,
    card: Mapping[str, Any],
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    audit_path = _bound_path(
        root, guard.get("telemetry_audit_path"), "telemetry_audit.json", "telemetry audit"
    )
    _hash(audit_path, str(guard.get("telemetry_audit_sha256")), "telemetry audit")
    audit = read_json(audit_path)
    fixed = {
        "status": "passed",
        "card_sha256": EXPECTED_CARD_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "scientific_payload_opened": False,
        "forecast_outcomes_accessed": False,
        "both_disjoint_windows_required": True,
        "disjoint_window_ordering_verified": True,
        "evaluation_end_epoch_identity_verified": True,
        "every_retained_sample_including_zero_used": True,
    }
    _require(
        all(audit.get(key) == value for key, value in fixed.items()),
        "GPU telemetry audit failed a frozen provenance gate",
    )
    for key in ("slurm_job_id", "gpu_uuid", "validity_windows", "validity_checks"):
        _require(audit.get(key) == guard.get(key), f"Guard/audit {key} drifted")
    _require(
        set(audit["validity_windows"]) == set(WINDOW_NAMES),
        "Telemetry window roster drifted",
    )
    plan = card["hardware_plan"]
    _require(
        all(plan.get(key) == value for key, value in HARDWARE_GATES.items()),
        "Frozen hardware gate drifted",
    )
    compact_windows: dict[str, Any] = {}
    for name in WINDOW_NAMES:
        window = audit["validity_windows"][name]
        checks = _telemetry_checks(window, plan)
        _require(checks == audit["validity_checks"][name], "Telemetry checks do not recompute")
        _require(all(checks.values()), f"GPU window {name} did not pass")
        compact_windows[name] = {
            "mean_gpu_utilization_percent": float(
                window["mean_retained_all_window_gpu_utilization_percent"]
            ),
            "p10_gpu_utilization_percent": float(
                window["p10_retained_all_window_gpu_utilization_percent"]
            ),
            "median_sample_cadence_seconds": float(
                window["median_sample_cadence_seconds"]
            ),
            "maximum_sample_gap_seconds": float(window["maximum_sample_gap_seconds"]),
            "leading_marker_edge_gap_seconds": float(
                window["leading_marker_edge_gap_seconds"]
            ),
            "trailing_marker_edge_gap_seconds": float(
                window["trailing_marker_edge_gap_seconds"]
            ),
            "retained_samples": int(window["retained_all_window_samples"]),
            "zero_utilization_samples_retained": int(
                window["zero_utilization_retained_samples"]
            ),
            "peak_memory_fraction": float(window["peak_memory_fraction"]),
        }
    raw_path = _bound_path(
        root, guard.get("raw_telemetry_path"), "raw_gpu_telemetry.csv", "raw telemetry"
    )
    _hash(raw_path, str(guard.get("raw_telemetry_sha256")), "raw telemetry")
    runtime_path = _bound_path(
        root, guard.get("runtime_lineage_path"), "runtime_lineage.json", "runtime lineage"
    )
    _hash(runtime_path, str(guard.get("runtime_lineage_sha256")), "runtime lineage")
    return {
        "status": "passed",
        "slurm_job_id": str(guard["slurm_job_id"]),
        "gpu_name": str(audit["gpu_name"]),
        "gpu_uuid": str(audit["gpu_uuid"]),
        "telemetry_interval_seconds": int(card["hardware_plan"]["telemetry_interval_seconds"]),
        "gated_metrics": [
            "natural_in_window_sample_count", "all_sample_mean_gpu_utilization",
            "peak_memory_fraction",
        ],
        "descriptive_only_metrics": [
            "p10_gpu_utilization", "sample_cadence", "maximum_sample_gap",
            "leading_marker_edge_gap", "trailing_marker_edge_gap",
        ],
        "windows": compact_windows,
        "both_disjoint_windows_passed": True,
        "zeros_retained": True,
        "no_padding": bool(card["hardware_plan"]["no_padding"]),
    }


def authenticate_summary(
    source_root: Path,
    expected_summary_receipt_sha256: str,
) -> dict[str, Any]:
    """Open only the already-authorized summary and its authenticated companions."""

    card, source_files = verify_frozen_sources()
    root = source_root.resolve()
    frozen_root = Path(card["prospective_datasets"]["output_root"]).resolve()
    _require(root == frozen_root, "Source root differs from the frozen V5 card")
    _require(
        len(expected_summary_receipt_sha256) == 64,
        "An exact summary-receipt SHA-256 is required",
    )
    receipt_path = root / "summary/summary_receipt.json"
    _hash(receipt_path, expected_summary_receipt_sha256, "summary receipt")
    receipt = read_json(receipt_path)
    receipt_keys = {
        "schema_version", "card_sha256", "source_manifest_sha256",
        "outcome_guard_receipt_sha256", "scientific_payload_sha256",
        "summary_path", "summary_sha256", "decision_path", "decision_sha256",
        "seed_rows_path", "seed_rows_sha256", "forecast_curve_path",
        "forecast_curve_sha256",
    }
    _require(set(receipt) == receipt_keys, "Summary-receipt schema drifted")
    _require(receipt.get("schema_version") == 1, "Summary-receipt version drifted")
    _require(
        receipt.get("card_sha256") == EXPECTED_CARD_SHA256
        and receipt.get("source_manifest_sha256") == EXPECTED_SOURCE_MANIFEST_SHA256,
        "Summary receipt is not bound to frozen V5 sources",
    )
    guard_path = root / "outcome_guard_receipt.json"
    _hash(guard_path, receipt["outcome_guard_receipt_sha256"], "outcome guard")
    guard = read_json(guard_path)
    guard_fixed = {
        "status": "authorized_for_dependent_cpu_summary",
        "card_sha256": EXPECTED_CARD_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "scientific_payload_sha256": receipt["scientific_payload_sha256"],
        "scientific_payload_opened": False,
        "forecast_outcomes_accessed": False,
        "both_disjoint_windows_required": True,
        "disjoint_window_ordering_verified": True,
        "evaluation_end_epoch_identity_verified": True,
    }
    _require(
        all(guard.get(key) == value for key, value in guard_fixed.items()),
        "Outcome guard failed authentication",
    )
    for stem, relative in (
        ("selection_decision", "selection_decision.json"),
        ("validation_data_manifest", "validation_data_manifest.json"),
        ("test_data_manifest", "test_data_manifest.json"),
        ("scientific_payload", "scientific_payload.json"),
    ):
        path = _bound_path(root, guard.get(f"{stem}_path"), relative, stem)
        _hash(path, str(guard.get(f"{stem}_sha256")), stem)
    roster = guard.get("checkpoint_roster", [])
    _require(
        len(roster) == 20
        and {(row.get("arm"), row.get("seed")) for row in roster}
        == {(arm, seed) for arm in ("dense", "sparse") for seed in range(64, 74)},
        "Authenticated checkpoint roster drifted",
    )
    smoke_path = Path(str(guard.get("smoke_receipt_path"))).resolve()
    expected_smoke = (
        Path(card["outcome_free_smoke"]["output_root"]) / "smoke_receipt.json"
    ).resolve()
    _require(smoke_path == expected_smoke, "Smoke receipt escaped its frozen root")
    _hash(smoke_path, str(guard.get("smoke_receipt_sha256")), "smoke receipt")
    smoke = read_json(smoke_path)
    _require(
        smoke.get("status") == "passed_outcome_free_gpu_smoke"
        and smoke.get("card_sha256") == EXPECTED_CARD_SHA256
        and smoke.get("source_manifest_sha256")
        == EXPECTED_SOURCE_MANIFEST_SHA256
        and smoke.get("scientific_outcomes_accessed") is False,
        "Outcome-free smoke provenance drifted",
    )
    telemetry = _validate_telemetry(root, card, guard)
    artifacts = {
        "summary": ("summary/summary.json", "summary_path", "summary_sha256"),
        "decision": ("summary/decision.json", "decision_path", "decision_sha256"),
        "seed_rows": ("summary/seed_rows.csv", "seed_rows_path", "seed_rows_sha256"),
        "curves": (
            "summary/forecast_curves_and_frontier.json",
            "forecast_curve_path", "forecast_curve_sha256",
        ),
    }
    paths: dict[str, Path] = {}
    for label, (relative, path_key, hash_key) in artifacts.items():
        path = _bound_path(root, receipt[path_key], relative, label)
        _hash(path, receipt[hash_key], label)
        paths[label] = path
    summary, decision, curves = (
        read_json(paths[name]) for name in ("summary", "decision", "curves")
    )
    linked = {
        "card_sha256": EXPECTED_CARD_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "outcome_guard_receipt_sha256": receipt["outcome_guard_receipt_sha256"],
    }
    _require(
        all(summary.get(key) == value for key, value in linked.items()),
        "Summary provenance linkage drifted",
    )
    _require(
        all(curves.get(key) == value for key, value in linked.items()),
        "Curve provenance linkage drifted",
    )
    _require(
        curves.get("scientific_payload_sha256")
        == receipt["scientific_payload_sha256"],
        "Curve/scientific payload linkage drifted",
    )
    curve_link = summary.get("forecast_curve_artifact", {})
    _require(
        set(curve_link) == {"path", "sha256"}
        and Path(str(curve_link["path"])).resolve() == paths["curves"]
        and curve_link["sha256"] == receipt["forecast_curve_sha256"],
        "Summary/curve linkage drifted",
    )
    _require(decision.get("decision") == summary.get("decision"), "Decision duplication drifted")
    _require(
        decision.get("confirmatory_h200_endpoint")
        == summary.get("confirmatory_h200_endpoint"),
        "Decision/summary H200 endpoint drifted",
    )
    selected = summary["selection_lineage"]["selected_cadences"]
    _require(decision.get("selected_cadences") == selected, "Selected cadence linkage drifted")
    _validate_seed_csv(paths["seed_rows"], summary)
    _validate_curves(curves, card, selected)
    return {
        "card": card,
        "source_manifest_entries": source_files,
        "receipt": receipt,
        "receipt_sha256": expected_summary_receipt_sha256,
        "guard": guard,
        "telemetry": telemetry,
        "summary": summary,
        "decision": decision,
        "curves": curves,
        "source_paths": {key: str(value) for key, value in paths.items()},
    }


def _validate_seed_csv(path: Path, summary: Mapping[str, Any]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    _require(
        fieldnames
        == ["model_seed", "endpoint", "dense_mse", "sparse_mse", "reduction"],
        "Source seed-row header drifted",
    )
    endpoints = {"h200_selection_aware_primary": summary["confirmatory_h200_endpoint"]}
    stress = summary.get("conditional_h400_stress_summary")
    if stress is not None:
        endpoints.update(stress["selected_recipe_comparison"]["endpoints"])
    _require(len(rows) == 10 * len(endpoints), "Source seed-row roster is incomplete")
    for endpoint_name, endpoint in endpoints.items():
        subset = [row for row in rows if row["endpoint"] == endpoint_name]
        _require([int(row["model_seed"]) for row in subset] == list(range(64, 74)), "Seed order drifted")
        dense = np.asarray([float(row["dense_mse"]) for row in subset])
        sparse = np.asarray([float(row["sparse_mse"]) for row in subset])
        np.testing.assert_allclose(dense, endpoint["dense_paired_seed_values"], rtol=0, atol=0)
        np.testing.assert_allclose(sparse, endpoint["sparse_paired_seed_values"], rtol=0, atol=0)


def _validate_curves(
    curves: Mapping[str, Any], card: Mapping[str, Any], selected: Mapping[str, Any]
) -> None:
    _require(curves.get("schema_version") == 1, "Curve schema drifted")
    for tier_name, expected_horizon in (("h200", 200), ("h400", 400)):
        tier = curves.get(tier_name)
        if tier is None:
            _require(tier_name == "h400", "The frozen H200 curve tier is mandatory")
            continue
        _require(tier.get("horizon_steps") == expected_horizon, "Curve horizon drifted")
        _require(float(tier.get("time_step")) == 0.1, "Curve time step drifted")
        records = tier.get("records", [])
        keys = {(row.get("arm"), row.get("cadence")) for row in records}
        required = {
            (arm, cadence)
            for arm in ("dense", "sparse")
            for cadence in ("direct", selected[arm])
        }
        _require(required <= keys, f"{tier_name} lacks direct/selected curves")
        for row in records:
            for name in tier["complete_curve_names"]:
                values = np.asarray(row[f"mean_{name}"], dtype=np.float64)
                _require(
                    values.shape == (expected_horizon,)
                    and np.isfinite(values).all()
                    and np.all(values >= 0.0),
                    f"Invalid {tier_name} {name} curve",
                )
    full_grid = curves.get("accuracy_refresh_frontier_h400")
    if full_grid is not None:
        rows = full_grid["endpoints"]["h400_cumulative_field_mse"]
        cadences = [row["cadence"] for row in rows]
        _require(
            cadences in (list(CADENCE_GRID), [*CADENCE_GRID, 200]),
            "Accuracy/refresh frontier dropped or reordered a cadence",
        )
        for row in rows:
            values = [row["dense_arm_mean_mse"], row["sparse_arm_mean_mse"]]
            _require(all(math.isfinite(float(value)) and float(value) >= 0 for value in values), "Invalid frontier MSE")


__all__ = [
    "CARD", "CADENCE_GRID", "EXPECTED_CARD_SHA256",
    "EXPECTED_SOURCE_MANIFEST_SHA256", "PACKET_ID", "SOURCE_MANIFEST",
    "TABLE_ROW_IDS", "VISUALIZATION_PLAN", "authenticate_summary", "read_json",
    "sha256_path", "verify_frozen_sources",
]
