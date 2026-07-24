"""Frozen, branch-invariant reduction for periodic-reencoding displays."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.neurips_2026.evidence.allen_cahn_periodic_reencoding_contract import (
    CADENCE_GRID,
    EXPECTED_CARD_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    PACKET_ID,
    TABLE_ROW_IDS,
    VISUALIZATION_PLAN,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unavailable(row_id: str, family: str, endpoint: str) -> dict[str, Any]:
    return {
        "row_id": row_id, "family": family, "endpoint": endpoint,
        "contrast": "frozen comparison unavailable after the preregistered failure policy",
        "baseline_policy": "unavailable", "comparison_policy": "unavailable",
        "baseline_mean_mse": None, "comparison_mean_mse": None,
        "relative_reduction": None, "ci95_lower": None, "ci95_upper": None,
        "one_sided_p": None, "wins_out_of_10": None,
        "inference_role": "unavailable_no_finite_authorized_tier",
        "status": "unavailable_frozen_tier",
    }


def _cross_row(
    row_id: str,
    endpoint_label: str,
    endpoint: Mapping[str, Any] | None,
    *,
    dense_cadence: str | int,
    sparse_cadence: str | int,
    inference_role: str,
    status: str,
    inference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if endpoint is None:
        return _unavailable(row_id, "sparse_vs_dense", endpoint_label)
    infer = inference or endpoint
    bootstrap = infer.get("paired_seed_bootstrap", infer.get("paired_ratio_bootstrap"))
    exact = infer.get(
        "exact_one_sided_studentized_arm_swap",
        infer.get("exact_one_sided_studentized_sign_flip"),
    )
    return {
        "row_id": row_id, "family": "sparse_vs_dense",
        "endpoint": endpoint_label,
        "contrast": "one minus sparse-over-dense recipe MSE",
        "baseline_policy": f"dense@{dense_cadence}",
        "comparison_policy": f"sparse@{sparse_cadence}",
        "baseline_mean_mse": float(endpoint["dense_mean"]),
        "comparison_mean_mse": float(endpoint["sparse_mean"]),
        "relative_reduction": float(endpoint["relative_reduction_of_arm_means"]),
        "ci95_lower": float(bootstrap["ci95_lower"]),
        "ci95_upper": float(bootstrap["ci95_upper"]),
        "one_sided_p": float(exact["one_sided_exact_p"]) if exact else None,
        "wins_out_of_10": int(endpoint["sparse_seed_wins"]),
        "inference_role": inference_role, "status": status,
    }


def _within_row(
    row_id: str,
    arm: str,
    endpoint_label: str,
    conditional: Mapping[str, Any] | None,
    selection_aware: Mapping[str, Any] | None,
    selected_cadence: str | int,
) -> dict[str, Any]:
    if conditional is None:
        return _unavailable(row_id, "selected_vs_direct", endpoint_label)
    if selection_aware is None:
        bootstrap = conditional["paired_ratio_bootstrap"]
        lower, upper = float(bootstrap["ci95_lower"]), float(bootstrap["ci95_upper"])
        effect = float(conditional["relative_reduction_of_arm_means"])
        wins = int(conditional["selected_seed_wins"])
        role, status = "conditional_fixed_validation_selection", "available_conditional_only"
    else:
        lower = float(selection_aware["selection_aware_bootstrap_ci95_lower"])
        upper = float(selection_aware["selection_aware_bootstrap_ci95_upper"])
        effect = float(selection_aware["heldout_point_relative_reduction"])
        wins = int(selection_aware["heldout_point_selected_seed_wins"])
        role, status = "selection_aware_selector_rerun_bootstrap", "available_selection_aware"
    return {
        "row_id": row_id, "family": "selected_vs_direct", "endpoint": endpoint_label,
        "contrast": f"one minus validation-selected-over-direct {arm} MSE",
        "baseline_policy": f"{arm}@direct",
        "comparison_policy": f"{arm}@{selected_cadence}",
        "baseline_mean_mse": float(conditional["direct_mean"]),
        "comparison_mean_mse": float(conditional["selected_mean"]),
        "relative_reduction": effect, "ci95_lower": lower, "ci95_upper": upper,
        "one_sided_p": None, "wins_out_of_10": wins,
        "inference_role": role, "status": status,
    }


def _direct_stress(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    stress = summary.get("conditional_h400_stress_summary")
    if stress is None:
        return None
    sensitivities = stress["same_cadence_sensitivity"]
    if "direct" in sensitivities:
        return sensitivities["direct"]
    selected = stress["selected_recipe_comparison"]
    if selected["dense_cadence"] == selected["sparse_cadence"] == "direct":
        return selected
    raise ValueError("Finite H400 stress tier lacks its mandatory direct sensitivity")


def comparison_rows(
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[float] | None]]:
    """Materialize every frozen comparison row, never an outcome-selected subset."""

    selected = summary["selection_lineage"]["selected_cadences"]
    primary = summary["confirmatory_h200_endpoint"]
    direct_h200 = summary["same_cadence_h200_sensitivities"]["direct"]
    full = summary.get("selection_aware_full_grid_h400")
    stress = summary.get("conditional_h400_stress_summary")
    direct_stress = _direct_stress(summary)
    selected_stress = None if stress is None else stress["selected_recipe_comparison"]
    rows = [
        _cross_row(
            TABLE_ROW_IDS[0], "H200 cumulative field MSE", primary,
            dense_cadence=selected["dense"], sparse_cadence=selected["sparse"],
            inference_role="confirmatory_selection_aware_primary",
            status="available_selection_aware",
        ),
        _cross_row(
            TABLE_ROW_IDS[1], "H200 cumulative field MSE", direct_h200,
            dense_cadence="direct", sparse_cadence="direct",
            inference_role="mandatory_unadjusted_descriptive_sensitivity",
            status="available_descriptive",
        ),
    ]
    paired: dict[str, list[float] | None] = {
        "h200": (
            np.asarray(primary["sparse_paired_seed_values"], dtype=float)
            / np.asarray(primary["dense_paired_seed_values"], dtype=float)
        ).tolist(),
        "h400": None, "h201_h400": None,
    }
    specs = (
        ("h400_cumulative_field_mse", "H400 cumulative field MSE", 2, 3, "h400"),
        ("h201_h400_tail_field_mse", "H201--H400 tail field MSE", 4, 5, "h201_h400"),
    )
    for name, label, selected_index, direct_index, paired_key in specs:
        if full is not None:
            full_record = full["endpoints"][name]
            selected_endpoint = full_record["point_test"]
            selected_inference = full_record["selection_aware_pipeline_inference"]
            role, status = "selection_aware_h400_durability", "available_selection_aware"
        elif selected_stress is not None:
            selected_endpoint = selected_stress["endpoints"][name]
            selected_inference = None
            role, status = "conditional_fixed_validation_selection", "available_conditional_only"
        else:
            selected_endpoint = selected_inference = None
            role, status = "unavailable", "unavailable_frozen_tier"
        rows.append(_cross_row(
            TABLE_ROW_IDS[selected_index], label, selected_endpoint,
            dense_cadence=selected["dense"], sparse_cadence=selected["sparse"],
            inference_role=role, status=status, inference=selected_inference,
        ))
        direct_endpoint = None if direct_stress is None else direct_stress["endpoints"][name]
        rows.append(_cross_row(
            TABLE_ROW_IDS[direct_index], label, direct_endpoint,
            dense_cadence="direct", sparse_cadence="direct",
            inference_role="mandatory_unadjusted_descriptive_sensitivity",
            status="available_descriptive" if direct_endpoint else "unavailable_frozen_tier",
        ))
        if selected_endpoint is not None:
            paired[paired_key] = (
                np.asarray(selected_endpoint["sparse_paired_seed_values"], dtype=float)
                / np.asarray(selected_endpoint["dense_paired_seed_values"], dtype=float)
            ).tolist()
    conditional_h200 = summary["pipeline_inference_h200"]["within_arm_selected_vs_direct_h200"]
    aware_h200 = summary["periodic_policy_h200"]["selection_aware_pipeline_bootstrap"]["arms"]
    conditional_h400 = summary.get("conditional_h400_selected_vs_direct")
    for offset, arm in enumerate(("dense", "sparse")):
        rows.append(_within_row(
            TABLE_ROW_IDS[6 + offset], arm, "H200 cumulative field MSE",
            conditional_h200[arm], aware_h200[arm], selected[arm],
        ))
        tail_conditional = (
            None if conditional_h400 is None
            else conditional_h400["arms"][arm]["endpoints"]["h201_h400_tail_field_mse"]
        )
        tail_aware = (
            None if full is None else full["endpoints"]["h201_h400_tail_field_mse"]
            ["selection_aware_within_arm_selected_vs_direct"][arm]
        )
        rows.append(_within_row(
            TABLE_ROW_IDS[8 + offset], arm, "H201--H400 tail field MSE",
            tail_conditional, tail_aware, selected[arm],
        ))
    by_id = {row["row_id"]: row for row in rows}
    _require(len(by_id) == len(rows) == len(TABLE_ROW_IDS), "Comparison row roster drifted")
    return [by_id[row_id] for row_id in TABLE_ROW_IDS], paired


def _validation_scores(
    summary: Mapping[str, Any], selected: Mapping[str, Any]
) -> dict[str, Any]:
    scores = summary["pipeline_inference_h200"]["validation_candidate_scores"]
    output: dict[str, Any] = {}
    for arm in ("dense", "sparse"):
        rows = scores[arm]
        _require([row["cadence"] for row in rows] == list(CADENCE_GRID), "Validation cadence order drifted")
        values = np.asarray([row["h200_cumulative_field_mse"] for row in rows], dtype=float)
        _require(values.shape == (9,) and np.isfinite(values).all(), "Invalid validation risks")
        minimum = float(values.min())
        tied = [row["cadence"] for row in rows if float(row["h200_cumulative_field_mse"]) == minimum]
        tie_order = lambda cadence: (0, 0) if cadence == "direct" else (1, -int(cadence))
        _require(min(tied, key=tie_order) == selected[arm], "Selected cadence is not the frozen exact minimum")
        output[arm] = [{"cadence": row["cadence"], "mse": float(row["h200_cumulative_field_mse"])} for row in rows]
    return output


def _curve_display(
    curves: Mapping[str, Any], selected: Mapping[str, Any]
) -> dict[str, Any]:
    tier_name = "h400" if curves.get("h400") is not None else "h200"
    tier = curves[tier_name]
    lookup = {(row["arm"], row["cadence"]): row for row in tier["records"]}
    series = []
    for arm in ("dense", "sparse"):
        for policy, cadence in (("direct", "direct"), ("selected", selected[arm])):
            series.append({
                "arm": arm, "policy": policy, "cadence": cadence,
                "mean_cumulative_field_mse": lookup[(arm, cadence)]["mean_cumulative_field_mse"],
            })
    return {
        "tier": tier["tier"], "horizon_steps": int(tier["horizon_steps"]),
        "time_step": float(tier["time_step"]), "series": series,
        "h400_available": curves.get("h400") is not None,
    }


def build_compact(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = bundle["summary"]
    selected = source["selection_lineage"]["selected_cadences"]
    rows, paired = comparison_rows(source)
    frontier = bundle["curves"].get("accuracy_refresh_frontier_h400")
    frontier_rows = [] if frontier is None else frontier["endpoints"]["h400_cumulative_field_mse"]
    compact = {
        "schema_version": 1, "packet_id": PACKET_ID,
        "fixed_display_contract": {
            "visualization_plan": list(VISUALIZATION_PLAN),
            "panel_ids": ["validation_risk", "heldout_curves", "paired_ratios", "accuracy_refresh_frontier"],
            "table_row_ids": list(TABLE_ROW_IDS),
            "outcome_dependent_row_or_panel_selection": False,
        },
        "protocol": {
            "system": "512-dimensional spatially extended four-well reaction--diffusion benchmark",
            "state_dimension": 512, "latent_dimension": 2048,
            "stored_time_step": 0.1, "trained_physical_horizon": 20.0,
            "stress_physical_horizon": 40.0, "paired_model_seeds": list(range(64, 74)),
            "validation_datasets": 3, "sealed_test_datasets": 3,
            "trajectories_per_dataset": 256,
        },
        "selected_cadences": dict(selected),
        "decision": source["decision"], "truth_difficulty": source["truth_difficulty"],
        "gpu_evaluation": bundle["telemetry"],
        "display_payload": {
            "validation_scores": _validation_scores(source, selected),
            "heldout_curves": _curve_display(bundle["curves"], selected),
            "paired_sparse_over_dense_ratios": paired,
            "accuracy_refresh_frontier_h400": frontier_rows,
            "frontier_status": "complete" if frontier is not None else "unavailable_frozen_tier",
        },
        "source_authentication": {
            "card_sha256": EXPECTED_CARD_SHA256,
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "summary_receipt_sha256": bundle["receipt_sha256"],
            "outcome_guard_receipt_sha256": bundle["receipt"]["outcome_guard_receipt_sha256"],
            "scientific_payload_sha256": bundle["receipt"]["scientific_payload_sha256"],
            "summary_sha256": bundle["receipt"]["summary_sha256"],
            "decision_sha256": bundle["receipt"]["decision_sha256"],
            "seed_rows_sha256": bundle["receipt"]["seed_rows_sha256"],
            "forecast_curve_sha256": bundle["receipt"]["forecast_curve_sha256"],
            "telemetry_audit_sha256": bundle["guard"]["telemetry_audit_sha256"],
            "runtime_lineage_sha256": bundle["guard"]["runtime_lineage_sha256"],
            "raw_telemetry_sha256": bundle["guard"]["raw_telemetry_sha256"],
            "source_root": str(Path(bundle["source_paths"]["summary"]).parents[1]),
        },
        "claim_boundary": source["decision"]["claim_boundary"],
    }
    validate_compact(compact, rows)
    return compact, rows


def validate_compact(
    summary: Mapping[str, Any], rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]
) -> None:
    _require(summary.get("packet_id") == PACKET_ID, "Compact packet ID drifted")
    contract = summary["fixed_display_contract"]
    _require(tuple(contract["visualization_plan"]) == VISUALIZATION_PLAN, "Panel plan drifted")
    _require(tuple(contract["table_row_ids"]) == TABLE_ROW_IDS, "Table plan drifted")
    _require(contract["outcome_dependent_row_or_panel_selection"] is False, "Cherry-picking guard drifted")
    _require([row["row_id"] for row in rows] == list(TABLE_ROW_IDS), "Comparison row order drifted")
    _require(len({row["row_id"] for row in rows}) == len(TABLE_ROW_IDS), "Duplicate comparison row")
    _require(
        [row["family"] for row in rows]
        == ["sparse_vs_dense"] * 6 + ["selected_vs_direct"] * 4,
        "Comparison-family semantics drifted",
    )
    allowed = {"available_selection_aware", "available_conditional_only", "available_descriptive", "unavailable_frozen_tier"}
    for index, row in enumerate(rows):
        _require(row["status"] in allowed, "Unknown row status")
        values = [row[key] for key in ("relative_reduction", "ci95_lower", "ci95_upper")]
        if row["status"] == "unavailable_frozen_tier":
            unavailable = values + [
                row["baseline_mean_mse"], row["comparison_mean_mse"],
                row["one_sided_p"], row["wins_out_of_10"],
            ]
            _require(all(value is None for value in unavailable), "Unavailable row contains an effect")
        else:
            _require(all(math.isfinite(float(value)) for value in values), "Available row has nonfinite evidence")
            baseline, comparison = (
                float(row["baseline_mean_mse"]), float(row["comparison_mean_mse"])
            )
            _require(baseline > 0 and comparison >= 0, "Available row has invalid MSE")
            _require(
                math.isclose(
                    float(row["relative_reduction"]),
                    1.0 - comparison / baseline,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ),
                "Available row effect differs from its displayed means",
            )
            _require(float(row["ci95_lower"]) <= float(row["ci95_upper"]), "Reversed confidence interval")
            _require(0 <= int(row["wins_out_of_10"]) <= 10, "Invalid paired-seed wins")
            if row["one_sided_p"] is not None:
                _require(0 <= float(row["one_sided_p"]) <= 1, "Invalid exact p-value")
        if index >= 6:
            _require(row["one_sided_p"] is None, "Within-arm row must not claim an exact p-value")
    _require(
        rows[0]["inference_role"] == "confirmatory_selection_aware_primary"
        and rows[0]["status"] == "available_selection_aware",
        "The sole confirmatory row drifted",
    )
    _require(
        all(
            row["inference_role"] == "mandatory_unadjusted_descriptive_sensitivity"
            for row in (rows[1], rows[3], rows[5])
            if row["status"] != "unavailable_frozen_tier"
        ),
        "Direct same-cadence rows must remain descriptive",
    )
    payload = summary["display_payload"]
    _require(set(payload["validation_scores"]) == {"dense", "sparse"}, "Validation panel arm roster drifted")
    for arm in ("dense", "sparse"):
        scores = payload["validation_scores"][arm]
        _require([row["cadence"] for row in scores] == list(CADENCE_GRID), "Compact validation cadence order drifted")
        _require(all(math.isfinite(float(row["mse"])) and float(row["mse"]) >= 0 for row in scores), "Compact validation score is invalid")
    curves = payload["heldout_curves"]
    series = curves["series"]
    expected_series = [
        ("dense", "direct"), ("dense", "selected"),
        ("sparse", "direct"), ("sparse", "selected"),
    ]
    _require([(row["arm"], row["policy"]) for row in series] == expected_series, "Held-out panel must contain four fixed series")
    horizon = int(curves["horizon_steps"])
    for record in series:
        values = np.asarray(record["mean_cumulative_field_mse"], dtype=float)
        _require(values.shape == (horizon,) and np.isfinite(values).all() and np.all(values >= 0), "Compact held-out curve is invalid")
    _require(set(payload["paired_sparse_over_dense_ratios"]) == {"h200", "h400", "h201_h400"}, "Ratio endpoint roster drifted")
    for key, values in payload["paired_sparse_over_dense_ratios"].items():
        if key == "h200" or values is not None:
            array = np.asarray(values, dtype=float)
            _require(array.shape == (10,) and np.isfinite(array).all() and np.all(array >= 0), "Compact paired-ratio roster is invalid")
    frontier = payload["accuracy_refresh_frontier_h400"]
    if frontier:
        cadences = [row["cadence"] for row in frontier]
        _require(cadences in (list(CADENCE_GRID), [*CADENCE_GRID, 200]), "Compact frontier cadence roster drifted")
        _require(all(
            int(row["refresh_count"]) >= 0
            and math.isfinite(float(row["dense_arm_mean_mse"]))
            and math.isfinite(float(row["sparse_arm_mean_mse"]))
            for row in frontier
        ), "Compact frontier contains invalid values")
    _require(summary["gpu_evaluation"]["both_disjoint_windows_passed"] is True, "GPU windows did not pass")
    auth = summary["source_authentication"]
    _require(auth["card_sha256"] == EXPECTED_CARD_SHA256, "Compact card hash drifted")
    _require(auth["source_manifest_sha256"] == EXPECTED_SOURCE_MANIFEST_SHA256, "Compact source hash drifted")


__all__ = ["build_compact", "comparison_rows", "validate_compact"]
