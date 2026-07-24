"""Deterministically extract compact rows from authenticated V2 artifacts."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Mapping

from experiments.neurips_2026.evidence.global_k_distinct_laws_v2_contract import (
    PACKET_ID,
    PROTOCOL_ID,
    SOURCE_FILES,
    json_bytes,
    load_sources,
    validate_source_agreement,
)


def _decision_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    packet = source["packet"]
    decision = source["decision"]
    audit = source["supplemental_audit"]
    card = source["card"]
    gpu = source["gpu_assessment"]
    selected = audit["selected_checkpoint_validation"]
    return {
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "invalid_negative",
        "positive_claim_permitted": False,
        "benchmark": {
            "system": card["benchmark"]["system"],
            "state_dimension": card["benchmark"]["state_dimension"],
            "known_evaluation_basin_count": card["benchmark"][
                "known_evaluation_basin_count"
            ],
            "dt": card["benchmark"]["dt"],
            "scientific_model_seeds": card["new_seed_contract"]["scientific_seeds"],
            "model_seed_count_per_arm": card["new_seed_contract"][
                "seed_count_per_arm"
            ],
            "label_policy": {
                "training_uses_basin_labels_or_count": card["benchmark"][
                    "training_uses_basin_labels_or_count"
                ],
                "evaluation_only_use_of_labels_and_count": card["benchmark"][
                    "evaluation_only_use_of_labels_and_count"
                ],
            },
            "family_corpus": card["label_free_family_discovery"],
            "evaluation_seeds": {
                "calibration_seed": card["evaluation_only_family_matching"][
                    "calibration_seed"
                ],
                "verification_seed": card["evaluation_only_family_matching"][
                    "verification_seed"
                ],
                "coordinate_null_seed": card["closure_and_coordinate_null"][
                    "coordinate_null_seed"
                ],
                "finite_radius_direction_seed": card[
                    "finite_radius_robustness_not_selection"
                ]["direction_seed"],
            },
            "geometry_expected_point_counts": card["benchmark"][
                "geometry_authentication"
            ]["expected_point_counts"],
        },
        "claim_boundary": packet["claim_boundary"],
        "mechanism_tier": decision["mechanism_tier"],
        "mechanism_text": decision["mechanism_text"],
        "relative_specificity_tier": decision["relative_specificity_tier"],
        "relative_specificity_text": decision["relative_specificity_text"],
        "mandatory_caveat": decision["mandatory_caveat"],
        "validity": decision["validity"],
        "sparse_gates": decision["sparse_gates"],
        "sparse_distributions": decision["sparse_distributions"],
        "dense_specificity": decision["specificity"],
        "audited_parameter_counts_by_arm": decision[
            "audited_parameter_counts_by_arm"
        ],
        "bootstrap": decision["bootstrap"],
        "per_basin_replication": audit["per_basin_replication"],
        "finite_radius_integrity": {
            key: audit["finite_radius_integrity"][key]
            for key in (
                "passed",
                "eligible_model_rows",
                "checked_H_G_basin_pairs",
                "required_H_G_basin_pairs",
                "expected_ordered_radii",
                "failures",
            )
        },
        "selected_checkpoint_validation": {
            key: selected[key]
            for key in (
                "passed",
                "passed_checkpoint_count",
                "required_checkpoint_count",
                "required_joint_finite_trajectories_each",
            )
        },
        "supplemental_integrity": {
            "status": audit["status"],
            "integrity_passed": audit["integrity_passed"],
            "independent_adjudication_reproduction": audit[
                "independent_adjudication_reproduction"
            ],
            "relative_specificity_adverse_completion": audit[
                "relative_specificity_adverse_completion"
            ],
            "limitations": audit["limitations"],
        },
        "gpu_assessment": {
            key: gpu[key]
            for key in (
                "assessment_complete",
                "flagged_low_utilization",
                "outcomes_inspected",
                "active_sample_count",
                "active_telemetry_expected_interval_coverage_fraction",
                "complete_rolling_ten_minute_window_count",
                "minimum_rolling_ten_minute_gpu_utilization_percent",
                "pack_wall_seconds",
                "tail_included_through_final_training_end",
                "utilization_alert_threshold_percent",
            )
        },
        "rebuttal_use": {
            "permitted": (
                "Report the preregistered failure and the descriptive G/closure values "
                "only as motivation for a separately frozen follow-up."
            ),
            "forbidden": [
                "Do not claim that one unchanged global K recovered three local laws.",
                "Do not claim invariant support-coordinate subspaces.",
                "Do not claim sparse-over-dense recipe specificity.",
                "Do not claim high-dimensional, realistic, or long-horizon forecasting evidence from this controlled two-dimensional diagnostic.",
            ],
        },
    }


SEED_FIELDS = (
    "arm", "seed", "task_id", "checkpoint_sha256", "checkpoint_authenticated",
    "checkpoint_validation_passed", "selector_horizon_steps", "selector_rollout",
    "selector_joint_finite_count", "selector_trajectory_count", "selector_final_error",
    "status", "law_valid", "kink_complete_seed_pass", "joint_h_g_seed_pass",
    "h_only_seed_pass", "g_only_seed_pass", "h_global_positive_control_pass",
    "direct_closure_valid", "dense_secondary_closure_valid", "family_valid",
    "discovered_family_count", "matched_family_count", "matched_support_cardinalities",
    "verification_point_counts_by_basin", "h_identity_unique",
    "h_max_own_relative_error", "h_max_own_over_nearest_wrong",
    "h_identity_over_best_nonidentity", "h_observed_over_null_median",
    "g_identity_unique", "g_max_own_relative_error", "g_max_own_over_nearest_wrong",
    "g_identity_over_best_nonidentity", "g_observed_over_null_median",
    "closure_per_seed_pass", "closure_counts_toward_aggregate",
    "closure_observed_max", "closure_median_null_max",
    "closure_observed_over_null_median", "failure_reasons",
)


def _seed_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    decision_rows = source["decision"]["seed_rows"]
    checked_rows = source["supplemental_audit"]["selected_checkpoint_validation"][
        "rows"
    ]
    validation = {(row["arm"], row["seed"]): row for row in checked_rows}
    result = []
    for row in decision_rows:
        checked = validation[(row["arm"], row["seed"])]
        routing = row["routing"]
        sparse_routing = routing.get("paired_sparse_routing", routing)
        closure = row["active_code_cloud_closure"]
        finite = checked["finite_trajectories"]
        result.append({
            "arm": row["arm"], "seed": row["seed"], "task_id": checked["task_id"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_authenticated": checked["checkpoint_authenticated_across_all_sources"],
            "checkpoint_validation_passed": checked["passed"],
            "selector_horizon_steps": checked["selector_horizon_steps"],
            "selector_rollout": checked["selector_rollout"],
            "selector_joint_finite_count": finite["joint_finite_count"],
            "selector_trajectory_count": finite["trajectory_count"],
            "selector_final_error": checked["recomputed_final_error"],
            "status": row["status"], "law_valid": row["law_valid"],
            "kink_complete_seed_pass": row["kink_complete_seed_pass"],
            "joint_h_g_seed_pass": row["joint_h_g_seed_pass"],
            "h_only_seed_pass": row["h_only_seed_pass"],
            "g_only_seed_pass": row["g_only_seed_pass"],
            "h_global_positive_control_pass": row["H_global_positive_control_pass"],
            "direct_closure_valid": row["direct_closure_valid"],
            "dense_secondary_closure_valid": row["dense_secondary_closure_valid"],
            "family_valid": routing["family_valid"],
            "discovered_family_count": len(sparse_routing["calibration_count_matrix"][0]),
            "matched_family_count": len(sparse_routing["matched_family_by_basin"]),
            "matched_support_cardinalities": json.dumps(
                sparse_routing["matched_support_cardinality_by_basin"], separators=(",", ":")
            ),
            "verification_point_counts_by_basin": json.dumps(
                sparse_routing["matched_verification_point_count_by_basin"], separators=(",", ":")
            ),
            "h_identity_unique": row["H_block"]["identity_is_unique_optimum"],
            "h_max_own_relative_error": row["H_block"]["max_own_relative_error"],
            "h_max_own_over_nearest_wrong": row["H_block"]["max_own_over_nearest_wrong"],
            "h_identity_over_best_nonidentity": row["H_block"]["identity_over_best_nonidentity"],
            "h_observed_over_null_median": row["H_observed_over_null_median"],
            "g_identity_unique": row["G_block"]["identity_is_unique_optimum"],
            "g_max_own_relative_error": row["G_block"]["max_own_relative_error"],
            "g_max_own_over_nearest_wrong": row["G_block"]["max_own_over_nearest_wrong"],
            "g_identity_over_best_nonidentity": row["G_block"]["identity_over_best_nonidentity"],
            "g_observed_over_null_median": row["G_observed_over_null_median"],
            "closure_per_seed_pass": closure["per_seed_pass"],
            "closure_counts_toward_aggregate": (
                closure["per_seed_pass"] and row["kink_complete_seed_pass"]
            ),
            "closure_observed_max": closure["observed_max_change_normalized_leakage"],
            "closure_median_null_max": closure["median_null_max"],
            "closure_observed_over_null_median": closure["observed_over_median_null"],
            "failure_reasons": ";".join(row["failure_reasons"]),
        })
    return result


BASIN_FIELDS = (
    "basin_index", "planned_model_seeds", "kink_valid", "g_own_nearest",
    "g_ratio_at_most_0p8", "h_own_nearest", "h_ratio_at_most_0p8",
    "closure_evaluable", "closure_at_most_0p50", "affine_both_guards",
    "finite_every_gate",
)


def _basin_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    replication = source["supplemental_audit"]["per_basin_replication"]
    return [{
        "basin_index": row["basin_index"],
        "planned_model_seeds": replication["planned_model_seeds_per_basin"],
        "kink_valid": row["kink_valid"], "g_own_nearest": row["G_own_nearest"],
        "g_ratio_at_most_0p8": row["G_ratio_at_most_0.8"],
        "h_own_nearest": row["H_own_nearest"],
        "h_ratio_at_most_0p8": row["H_ratio_at_most_0.8"],
        "closure_evaluable": row["closure_evaluable"],
        "closure_at_most_0p50": row["closure_at_most_0.50"],
        "affine_both_guards": row["affine_both_guards"],
        "finite_every_gate": row["finite_every_gate"],
    } for row in replication["counts_by_basin"]]


def _csv_bytes(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _provenance_payload(source: Mapping[str, Any]) -> dict[str, Any]:
    packet = source["packet"]
    audit = source["supplemental_audit"]
    gpu = source["gpu_assessment"]
    return {
        "schema_version": 1, "packet_id": PACKET_ID, "protocol_id": PROTOCOL_ID,
        "status": "authenticated_negative_evidence",
        "external_source_roots": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in SOURCE_FILES.items()
        },
        "source_packet_provenance": packet["provenance"],
        "supplemental_audit_provenance": audit["provenance"],
        "gpu_assessment_provenance": gpu["provenance"],
        "execution": {
            "scientific_training_job_id": 10164075,
            "supplemental_cpu_audit_job_id": 10164630,
            "scientific_pack_wall_seconds": gpu["pack_wall_seconds"],
            "scientific_minimum_rolling_ten_minute_gpu_utilization_percent": gpu[
                "minimum_rolling_ten_minute_gpu_utilization_percent"
            ],
        },
        "authentication": {
            "independent_decision_byte_and_value_equality": True,
            "all_twenty_checkpoints_identified": True,
            "all_sixteen_selector_trajectories_per_checkpoint_finite": True,
            "all_sixty_h_g_radius_arrays_order_authenticated": True,
            "raw_external_artifacts_modified": False,
        },
        "portable_scope": (
            "The tracked packet authenticates its compact bytes and records all raw "
            "roots. --check-sources additionally rehashes the mounted external archive "
            "and regenerates the compact bytes."
        ),
    }


def build_payloads() -> dict[str, bytes]:
    source = load_sources()
    validate_source_agreement(source)
    return {
        "decision.json": json_bytes(_decision_payload(source)),
        "seed_rows.csv": _csv_bytes(_seed_rows(source), SEED_FIELDS),
        "basin_rows.csv": _csv_bytes(_basin_rows(source), BASIN_FIELDS),
        "provenance.json": json_bytes(_provenance_payload(source)),
    }
