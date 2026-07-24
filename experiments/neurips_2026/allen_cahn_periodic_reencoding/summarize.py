"""Open guarded outcomes and adjudicate the frozen periodic-reencoding test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.allen_cahn_periodic_reencoding.adjudication import (
    adjudicate,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.curve_summary import (
    combined_accuracy_refresh_frontier,
    summarize_curve_panel,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.h200_sensitivities import (
    summarize_same_cadence_h200,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.h400_full_grid import (
    summarize_full_grid_h400_pipeline,
    validate_full_grid_h400_rows,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.forecast_skill import (
    cadence_cost_table,
    summarize_selected_absolute_skill,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    CARD_PATH,
    MANIFEST_PATH,
    duplicate_safe_json,
    load_card,
    sha256_path,
    verify_file,
    verify_source_manifest,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.pipeline_inference import (
    summarize_pipeline_inference,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.p200_one_refresh import (
    P200,
    summarize_optional_p200_one_refresh,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.policy_pipeline import (
    summarize_policy_pipeline_bootstrap,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.policy_statistics import (
    summarize_selected_vs_direct,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    summarize_test_rows,
    validate_primary_test_rows,
    validate_test_rows,
    validate_validation_rows,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.stress import (
    validate_stress_prefix,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.summary_integrity import (
    validate_materialized_fields,
    verify_h400_failure_lineage,
    verify_selection_lineage,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.telemetry import (
    WINDOW_MARKERS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    return parser.parse_args()


def _guarded_payload(
    args: argparse.Namespace,
    card: dict[str, Any],
    *,
    card_hash: str,
    source_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if args.root != Path(card["prospective_datasets"]["output_root"]):
        raise RuntimeError("Summary root differs from the prediction card")
    guard_path = args.root / "outcome_guard_receipt.json"
    verify_file(guard_path, args.expected_guard_sha256)
    guard = duplicate_safe_json(guard_path)
    expected_guard = {
        "status": "authorized_for_dependent_cpu_summary",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "scientific_payload_opened": False,
        "forecast_outcomes_accessed": False,
        "both_disjoint_windows_required": True,
        "disjoint_window_ordering_verified": True,
        "evaluation_end_epoch_identity_verified": True,
    }
    if any(guard.get(key) != value for key, value in expected_guard.items()):
        raise RuntimeError("Outcome guard is invalid or already outcome-aware")
    expected_windows = set(WINDOW_MARKERS)
    checks = guard.get("validity_checks", {})
    if set(checks) != expected_windows or not all(
        all(bool(value) for value in row.values()) for row in checks.values()
    ):
        raise RuntimeError("Both GPU-compute telemetry windows must pass")
    expected_markers = {
        "selection_start",
        "selection_end",
        "evaluation_start",
        "evaluation_compute_end",
        "evaluation_end",
    }
    if set(guard.get("marker_bindings", {})) != expected_markers:
        raise RuntimeError("Guard lacks the exact marker-binding roster")
    runtime_path = Path(str(guard["runtime_lineage_path"]))
    if runtime_path != args.root / "runtime_lineage.json":
        raise RuntimeError("Guard runtime path escaped the frozen root")
    verify_file(runtime_path, str(guard["runtime_lineage_sha256"]))
    runtime = duplicate_safe_json(runtime_path)
    if (
        runtime.get("card_sha256") != card_hash
        or runtime.get("source_manifest_sha256") != source_hash
        or runtime.get("slurm_job_id") != guard.get("slurm_job_id")
        or runtime.get("scientific_metrics_printed") is not False
    ):
        raise RuntimeError("Runtime lineage differs from the guard")
    scientific_path = Path(str(guard["scientific_payload_path"]))
    if scientific_path != args.root / "scientific_payload.json":
        raise RuntimeError("Scientific payload path escaped the frozen root")
    verify_file(scientific_path, str(guard["scientific_payload_sha256"]))
    if runtime.get("scientific_payload_sha256") != guard["scientific_payload_sha256"]:
        raise RuntimeError("Runtime and guard scientific hashes differ")

    # This is the first scientific deserialization; all preceding checks are
    # metric-free hashes, schemas, telemetry gates, and paths.
    scientific = duplicate_safe_json(scientific_path)
    if (
        scientific.get("card_sha256") != card_hash
        or scientific.get("source_manifest_sha256") != source_hash
    ):
        raise RuntimeError("Scientific payload lineage drifted")
    selection_path = Path(str(scientific["selection_decision_path"]))
    if selection_path != args.root / "selection_decision.json":
        raise RuntimeError("Selection decision path escaped the frozen root")
    selection_hash = str(scientific["selection_decision_sha256"])
    if guard.get("selection_decision_sha256") != selection_hash:
        raise RuntimeError("Guard and scientific selection hashes differ")
    verify_file(selection_path, selection_hash)
    return guard, duplicate_safe_json(selection_path), scientific


def _primary_endpoint(pipeline: dict[str, Any]) -> dict[str, Any]:
    point = dict(pipeline["primary_point_test"])
    selection_aware = pipeline["selection_aware_pipeline_inference"]
    point["exact_one_sided_studentized_sign_flip"] = selection_aware[
        "exact_one_sided_studentized_sign_flip"
    ]
    point["paired_ratio_bootstrap"] = selection_aware["paired_seed_bootstrap"]
    point["inference_role"] = "confirmatory_selection_aware_primary"
    return point


def _h400_pipeline_endpoint(record: dict[str, Any]) -> dict[str, Any]:
    point = dict(record["point_test"])
    inference = record["selection_aware_pipeline_inference"]
    point["exact_one_sided_studentized_sign_flip"] = inference[
        "exact_one_sided_studentized_arm_swap"
    ]
    point["paired_ratio_bootstrap"] = inference["paired_seed_bootstrap"]
    point["inference_role"] = "selection_aware_h400_durability_diagnostic"
    return point


def _write_seed_rows(path: Path, endpoints: dict[str, dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_seed", "endpoint", "dense_mse", "sparse_mse", "reduction"],
        )
        writer.writeheader()
        for endpoint_name, endpoint in endpoints.items():
            for offset, (dense, sparse) in enumerate(
                zip(
                    endpoint["dense_paired_seed_values"],
                    endpoint["sparse_paired_seed_values"],
                    strict=True,
                )
            ):
                writer.writerow(
                    {
                        "model_seed": 64 + offset,
                        "endpoint": endpoint_name,
                        "dense_mse": f"{float(dense):.17g}",
                        "sparse_mse": f"{float(sparse):.17g}",
                        "reduction": f"{1.0-float(sparse)/float(dense):.17g}",
                    }
                )


def main() -> None:
    args = parse_args()
    card, card_hash = load_card(args.card, expected_sha256=args.expected_card_sha256)
    source_hash = verify_source_manifest(
        card,
        path=args.source_manifest,
        expected_sha256=args.expected_source_manifest_sha256,
    )
    guard, selection, scientific = _guarded_payload(
        args, card, card_hash=card_hash, source_hash=source_hash
    )
    validation_rows = scientific["validation_rows"]
    primary_rows = scientific["primary_test_rows"]
    stress_rows = scientific["stress_rows"]
    stress_failures = scientific["stress_failures"]
    grid_stress_failures = scientific["grid_stress_failures"]
    p200_failures = scientific["p200_failures"]
    required_stress_failures = scientific["required_stress_failures"]
    validate_validation_rows(validation_rows, card)
    validate_primary_test_rows(primary_rows, card)
    lineage = verify_selection_lineage(
        selection,
        scientific,
        validation_rows,
        card,
        card_hash=card_hash,
        source_hash=source_hash,
    )
    fields = validate_materialized_fields(guard, scientific, card, args.root)
    pipeline = summarize_pipeline_inference(validation_rows, primary_rows, card)
    if pipeline["selected_cadences_from_validation"] != lineage["selected_cadences"]:
        raise RuntimeError("Pipeline selector differs from the sealed decision")
    primary = _primary_endpoint(pipeline)
    sensitivities = summarize_same_cadence_h200(primary_rows, card)
    policy_pipeline = summarize_policy_pipeline_bootstrap(
        validation_rows, primary_rows, card
    )
    selected = lineage["selected_cadences"]
    conditional_policy = pipeline["within_arm_selected_vs_direct_h200"]
    policy_h200 = {
        "conditional": conditional_policy,
        "selection_aware": policy_pipeline["arms"],
    }
    failure_lineage = verify_h400_failure_lineage(scientific, card, selected)
    absolute_skill_h200 = summarize_selected_absolute_skill(
        primary_rows, card, selected, horizon=200
    )
    stress_summary = None
    stress_policy = None
    full_grid_h400 = None
    absolute_skill_h400 = None
    required_cadences = [
        value
        for value in card["cadence_selection"]["cadence_grid"]
        if value in {"direct", selected["dense"], selected["sparse"]}
    ]
    if scientific["required_stress_cadences"] != required_cadences:
        raise RuntimeError("Required H400 cadence union drifted")
    required_rows = [row for row in stress_rows if row["cadence"] in required_cadences]
    if not required_stress_failures:
        validate_test_rows(required_rows, card, selected)
        validate_stress_prefix(primary_rows, required_rows)
        stress_summary = summarize_test_rows(required_rows, card, selected)
        stress_policy = summarize_selected_vs_direct(required_rows, card, selected)
        absolute_skill_h400 = summarize_selected_absolute_skill(
            required_rows, card, selected, horizon=400
        )
    grid_rows = [
        row
        for row in stress_rows
        if row["cadence"] in card["cadence_selection"]["cadence_grid"]
    ]
    if not grid_stress_failures:
        validate_full_grid_h400_rows(grid_rows, card)
        validate_stress_prefix(primary_rows, grid_rows)
        full_grid_h400 = summarize_full_grid_h400_pipeline(
            validation_rows, grid_rows, card
        )
        if full_grid_h400["selected_cadences_from_h200_validation"] != selected:
            raise RuntimeError("H400 pipeline selector differs from the sealed decision")
    h400_tail = None
    h400_policy = None
    if full_grid_h400 is not None:
        tail_record = full_grid_h400["endpoints"]["h201_h400_tail_field_mse"]
        h400_tail = _h400_pipeline_endpoint(tail_record)
        h400_policy = tail_record[
            "selection_aware_within_arm_selected_vs_direct"
        ]
    h200_direct_rows = [row for row in primary_rows if row["cadence"] == "direct"]
    p200_rows = [row for row in stress_rows if row["cadence"] == P200]
    direct_h400_rows = [row for row in stress_rows if row["cadence"] == "direct"]
    p200_diagnostic = summarize_optional_p200_one_refresh(
        None if p200_failures else p200_rows,
        None if any(row.get("cadence") == "direct" for row in stress_failures)
        else direct_h400_rows,
        h200_direct_rows,
        card,
    )
    h200_curves = summarize_curve_panel(
        primary_rows,
        card,
        cadences=card["cadence_selection"]["cadence_grid"],
        horizon=200,
        tier="independent_full_grid_h200",
    )
    h400_curves = None
    if full_grid_h400 is not None:
        h400_curves = summarize_curve_panel(
            grid_rows,
            card,
            cadences=card["cadence_selection"]["cadence_grid"],
            horizon=400,
            tier="complete_full_grid_h400",
        )
    elif stress_summary is not None:
        h400_curves = summarize_curve_panel(
            required_rows,
            card,
            cadences=required_cadences,
            horizon=400,
            tier="fixed_selected_direct_h400_fallback",
        )
    p200_curves = None
    if p200_diagnostic["status"] == "complete":
        p200_curves = summarize_curve_panel(
            [*direct_h400_rows, *p200_rows],
            card,
            cadences=["direct", P200],
            horizon=400,
            tier="optional_p200_one_refresh_h400",
        )
    curve_evidence = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "outcome_guard_receipt_sha256": args.expected_guard_sha256,
        "scientific_payload_sha256": guard["scientific_payload_sha256"],
        "h200": h200_curves,
        "h400": h400_curves,
        "p200": p200_curves,
        "accuracy_refresh_frontier_h400": combined_accuracy_refresh_frontier(
            full_grid_h400, p200_diagnostic
        ),
    }
    decision = adjudicate(
        primary=primary,
        direct_h200=sensitivities["direct"],
        policy_h200=policy_h200,
        absolute_skill_h200=absolute_skill_h200,
        absolute_skill_h400=absolute_skill_h400,
        stress_summary=stress_summary,
        stress_policy=stress_policy,
        selection_aware_h400_tail=h400_tail,
        stress_failures=grid_stress_failures,
        truth_difficulty=scientific["truth_difficulty"],
        truth_threshold=float(
            card["test_evaluation"]["late_truth_language_gate"][
                "near_stationary_threshold"
            ]
        ),
        selection_aware_h400_policy=h400_policy,
    )
    summary_dir = args.root / "summary"
    curve_path = summary_dir / "forecast_curves_and_frontier.json"
    write_json_once(curve_path, curve_evidence)
    summary = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "outcome_guard_receipt_sha256": args.expected_guard_sha256,
        "selection_lineage": lineage,
        "field_artifact_validation": fields,
        "h400_failure_lineage": failure_lineage,
        "pipeline_inference_h200": pipeline,
        "confirmatory_h200_endpoint": primary,
        "same_cadence_h200_sensitivities": sensitivities,
        "periodic_policy_h200": {
            "conditional_fixed_selection": conditional_policy,
            "selection_aware_pipeline_bootstrap": policy_pipeline,
        },
        "absolute_forecast_skill_h200": absolute_skill_h200,
        "absolute_forecast_skill_h400": absolute_skill_h400,
        "cadence_compute_cost_h200": cadence_cost_table(
            card["cadence_selection"]["cadence_grid"], 200
        ),
        "cadence_compute_cost_h400": cadence_cost_table(
            card["cadence_selection"]["cadence_grid"], 400
        ),
        "conditional_h400_stress_summary": stress_summary,
        "conditional_h400_selected_vs_direct": stress_policy,
        "selection_aware_full_grid_h400": full_grid_h400,
        "optional_p200_one_refresh_at_t20": p200_diagnostic,
        "forecast_curve_artifact": {
            "path": str(curve_path),
            "sha256": sha256_path(curve_path),
        },
        "h400_stress_failures": stress_failures,
        "h400_selectable_grid_failures": grid_stress_failures,
        "p200_failures": p200_failures,
        "truth_difficulty": scientific["truth_difficulty"],
        "decision": decision,
    }
    summary_path = summary_dir / "summary.json"
    decision_path = summary_dir / "decision.json"
    seed_rows_path = summary_dir / "seed_rows.csv"
    write_json_once(summary_path, summary)
    write_json_once(
        decision_path,
        {
            "schema_version": 1,
            "protocol_id": card["protocol_id"],
            "selected_cadences": selected,
            "confirmatory_h200_endpoint": primary,
            "decision": decision,
            "truth_difficulty": scientific["truth_difficulty"],
        },
    )
    seed_endpoints = {"h200_selection_aware_primary": primary}
    if stress_summary is not None:
        seed_endpoints.update(
            stress_summary["selected_recipe_comparison"]["endpoints"]
        )
    _write_seed_rows(seed_rows_path, seed_endpoints)
    receipt_path = summary_dir / "summary_receipt.json"
    write_json_once(
        receipt_path,
        {
            "schema_version": 1,
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "outcome_guard_receipt_sha256": args.expected_guard_sha256,
            "scientific_payload_sha256": guard["scientific_payload_sha256"],
            "summary_path": str(summary_path),
            "summary_sha256": sha256_path(summary_path),
            "decision_path": str(decision_path),
            "decision_sha256": sha256_path(decision_path),
            "seed_rows_path": str(seed_rows_path),
            "seed_rows_sha256": sha256_path(seed_rows_path),
            "forecast_curve_path": str(curve_path),
            "forecast_curve_sha256": sha256_path(curve_path),
        },
    )
    tail_reduction = None
    if stress_summary is not None:
        tail_reduction = stress_summary["selected_recipe_comparison"]["endpoints"][
            "h201_h400_tail_field_mse"
        ]["relative_reduction_of_arm_means"]
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_cadences": selected,
                "decision_branch": decision["branch"],
                "h200_relative_reduction": primary[
                    "relative_reduction_of_arm_means"
                ],
                "h201_h400_relative_reduction": tail_reduction,
                "summary_receipt_sha256": sha256_path(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
