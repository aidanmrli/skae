"""Outcome-blind publication guards layered on frozen V2 adjudication."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


EXPECTED_ORDERED_RADII = (0.01, 0.03, 0.06, 0.12, 0.18)


def _ordered_radius_records(records: Any) -> dict[str, Any]:
    """Validate one H/G radius sweep without coercing malformed values."""
    failures: list[str] = []
    observed: list[float | None] = []
    if not isinstance(records, list):
        return {
            "passed": False,
            "expected_ordered_radii": list(EXPECTED_ORDERED_RADII),
            "observed_ordered_radii": [],
            "failure_reasons": ["records_not_a_list"],
        }
    if len(records) != len(EXPECTED_ORDERED_RADII):
        failures.append(
            f"record_count_{len(records)}_expected_{len(EXPECTED_ORDERED_RADII)}"
        )
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            observed.append(None)
            failures.append(f"record_{index}_not_an_object")
            continue
        value = record.get("radius")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            observed.append(None)
            failures.append(f"record_{index}_radius_not_finite_numeric")
            continue
        observed.append(float(value))
    if len(observed) == len(EXPECTED_ORDERED_RADII) and any(
        value != expected
        for value, expected in zip(observed, EXPECTED_ORDERED_RADII)
    ):
        failures.append("ordered_radius_sequence_mismatch")
    return {
        "passed": not failures,
        "expected_ordered_radii": list(EXPECTED_ORDERED_RADII),
        "observed_ordered_radii": observed,
        "failure_reasons": failures,
    }


def finite_radius_integrity(
    rows: list[dict[str, Any]], card: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on parent-card, basin, and paired H/G radius alignment."""
    card_radii = card.get("finite_radius_robustness_not_selection", {}).get(
        "radii"
    )
    card_check = _ordered_radius_records(
        [{"radius": value} for value in card_radii]
        if isinstance(card_radii, list)
        else card_radii
    )
    checked_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    eligible_model_rows = 0
    checked_h_g_pairs = 0
    for row in rows:
        result = row.get("result", {})
        if result.get("status") != "eligible":
            continue
        eligible_model_rows += 1
        identity = {
            "task_id": row.get("task_id"),
            "arm": row.get("arm"),
            "seed": row.get("seed"),
        }
        by_basin = result.get("finite_radius_robustness", {}).get("by_basin")
        if not isinstance(by_basin, list) or len(by_basin) != 3:
            failures.append({
                **identity,
                "failure": "finite_radius_by_basin_must_have_exactly_three_rows",
            })
            continue
        for basin_index, basin in enumerate(by_basin):
            if not isinstance(basin, dict) or basin.get("basin") != basin_index:
                failures.append({
                    **identity,
                    "basin_index": basin_index,
                    "failure": "basin_row_index_mismatch",
                })
                continue
            h_check = _ordered_radius_records(basin.get("H"))
            g_check = _ordered_radius_records(basin.get("G"))
            paired = bool(
                h_check["passed"]
                and g_check["passed"]
                and h_check["observed_ordered_radii"]
                == g_check["observed_ordered_radii"]
            )
            checked_h_g_pairs += 1
            checked = {
                **identity,
                "basin_index": basin_index,
                "H": h_check,
                "G": g_check,
                "H_G_radius_index_alignment": paired,
                "passed": bool(h_check["passed"] and g_check["passed"] and paired),
            }
            checked_rows.append(checked)
            if not checked["passed"]:
                failures.append({
                    **identity,
                    "basin_index": basin_index,
                    "failure": "H_G_ordered_radius_integrity_failed",
                    "H_failure_reasons": h_check["failure_reasons"],
                    "G_failure_reasons": g_check["failure_reasons"],
                    "H_G_radius_index_alignment": paired,
                })
    expected_pairs = 3 * eligible_model_rows
    return {
        "passed": bool(
            card_check["passed"]
            and checked_h_g_pairs == expected_pairs
            and not failures
        ),
        "expected_ordered_radii": list(EXPECTED_ORDERED_RADII),
        "parent_card_radius_check": card_check,
        "eligible_model_rows": eligible_model_rows,
        "required_H_G_basin_pairs": expected_pairs,
        "checked_H_G_basin_pairs": checked_h_g_pairs,
        "rows": checked_rows,
        "failures": failures,
    }


def per_basin_counts(
    rows: list[dict[str, Any]], card: dict[str, Any], mechanism_tier: str,
    minimum: int,
) -> dict[str, Any]:
    residual_limit = float(
        card["finite_radius_robustness_not_selection"][
            "maximum_normalized_linear_fit_residual_each_radius"
        ]
    )
    keys = (
        "kink_valid", "H_own_nearest", "G_own_nearest", "H_ratio_at_most_0.8",
        "G_ratio_at_most_0.8", "closure_evaluable", "closure_at_most_0.50",
        "affine_both_guards", "finite_every_gate",
    )
    counts = [{key: 0 for key in keys} for _ in range(3)]
    passing_seeds = [{key: [] for key in keys} for _ in range(3)]
    for row in rows:
        result = row["result"]
        if row["arm"] != "sparse" or result["status"] != "eligible":
            continue
        h_law = result["H_block"]["law_identification"]
        g_law = result["G_block"]["law_identification"]
        for basin_index in range(3):
            kink = result["kink_guard"]["rows"][basin_index][
                "passed_both_estimands_both_epsilons"
            ]
            if not kink:
                continue
            basin_counts = counts[basin_index]
            basin_seeds = passing_seeds[basin_index]

            def record(name: str, passed: bool = True) -> None:
                if passed:
                    basin_counts[name] += 1
                    basin_seeds[name].append(int(row["seed"]))

            record("kink_valid")
            h_costs = np.asarray(h_law["cost_matrix"], dtype=np.float64)
            g_costs = np.asarray(g_law["cost_matrix"], dtype=np.float64)
            record(
                "H_own_nearest",
                int(np.argmin(h_costs[basin_index])) == basin_index
            )
            record(
                "G_own_nearest",
                int(np.argmin(g_costs[basin_index])) == basin_index
            )
            record(
                "H_ratio_at_most_0.8",
                h_law["own_over_nearest_wrong_by_basin"][basin_index] <= 0.8
            )
            record(
                "G_ratio_at_most_0.8",
                g_law["own_over_nearest_wrong_by_basin"][basin_index] <= 0.8
            )
            if result.get("direct_closure_valid", False):
                record("closure_evaluable")
                closure = result["direct_active_code_cloud_closure"]["rows"][
                    basin_index
                ]["change_normalized_leakage"]
                record("closure_at_most_0.50", closure <= 0.50)
            guard = result["center_forecast_guards"][basin_index]
            affine = bool(
                guard["restricted_forecast"] <= 0.25
                and guard["k_induced_update"] <= 0.25
            )
            record("affine_both_guards", affine)
            finite = result["finite_radius_robustness"]["by_basin"][basin_index]
            h_records, g_records = finite["H"], finite["G"]
            h_radius = _ordered_radius_records(h_records)
            g_radius = _ordered_radius_records(g_records)
            radius_pair_aligned = bool(
                h_radius["passed"]
                and g_radius["passed"]
                and h_radius["observed_ordered_radii"]
                == g_radius["observed_ordered_radii"]
            )
            finite_checks = [affine, h_radius["passed"], g_radius["passed"], radius_pair_aligned]
            if h_radius["passed"] and g_radius["passed"] and radius_pair_aligned:
                finite_checks.extend((
                    all(item["autograd_agreement"] <= 0.25 for item in h_records[:2]),
                    all(item["autograd_agreement"] <= 0.25 for item in g_records[:2]),
                    all(item["own_law_is_nearest"] for item in h_records),
                    all(item["own_law_is_nearest"] for item in g_records),
                    all(
                        math.isfinite(item["normalized_linear_fit_residual"])
                        and item["normalized_linear_fit_residual"] <= residual_limit
                        for item in h_records
                    ),
                    all(
                        math.isfinite(item["normalized_linear_fit_residual"])
                        and item["normalized_linear_fit_residual"] <= residual_limit
                        for item in g_records
                    ),
                ))
            record("finite_every_gate", all(finite_checks))
    tier_requirements = {
        "finite_neighborhood_local_laws": (
            "H_own_nearest", "G_own_nearest", "H_ratio_at_most_0.8",
            "G_ratio_at_most_0.8", "closure_at_most_0.50",
            "affine_both_guards", "finite_every_gate",
        ),
        "affine_fixed_point_local_laws": (
            "H_own_nearest", "G_own_nearest", "H_ratio_at_most_0.8",
            "G_ratio_at_most_0.8", "closure_at_most_0.50",
            "affine_both_guards",
        ),
        "distinct_predictor_and_k_induced_jacobians_only": (
            "H_own_nearest", "G_own_nearest", "H_ratio_at_most_0.8",
            "G_ratio_at_most_0.8", "closure_at_most_0.50",
        ),
        "distinct_k_induced_update_jacobians_only": (
            "G_own_nearest", "G_ratio_at_most_0.8", "closure_at_most_0.50",
        ),
        "restricted_predictor_jacobians_mechanism_unresolved": (
            "H_own_nearest", "H_ratio_at_most_0.8",
        ),
    }
    required = tier_requirements.get(mechanism_tier, ())
    permitted = (
        None if not required else all(
            counts[basin][name] >= minimum
            for basin in range(3) for name in required
        )
    )
    return {
        "planned_model_seeds_per_basin": 10,
        "minimum_adequate_passes_per_basin": minimum,
        "counts_by_basin": [
            {
                "basin_index": index,
                **basin_counts,
                "metric_details": {
                    name: {
                        "numerator": basin_counts[name],
                        "eligible_denominator": (
                            basin_counts["closure_evaluable"]
                            if name == "closure_at_most_0.50"
                            else 10 if name == "kink_valid"
                            else basin_counts["kink_valid"]
                        ),
                        "planned_denominator": 10,
                        "passing_seed_ids": passing_seeds[index][name],
                    }
                    for name in keys
                },
            }
            for index, basin_counts in enumerate(counts)
        ],
        "tier_relevant_requirements": list(required),
        "blanket_three_law_wording_applicable": bool(required),
        "blanket_three_law_wording_permitted": permitted,
    }


def adverse_specificity_guard(
    decision: dict[str, Any], card: dict[str, Any],
) -> dict[str, Any]:
    specificity = decision["specificity"]
    expected = [int(seed) for seed in card["new_seed_contract"]["scientific_seeds"]]
    paired = [int(seed) for seed in specificity["paired_seeds"]]
    if len(expected) != 10 or len(set(paired)) != len(paired):
        raise RuntimeError("Malformed fixed-ten specificity roster")
    if not set(paired).issubset(expected):
        raise RuntimeError("Specificity roster contains an unexpected seed")
    missing = [seed for seed in expected if seed not in paired]
    ratios = specificity["ratios_by_seed"]
    if any(len(ratios[name]) != len(paired) for name in ("H_row", "H_assignment")):
        raise RuntimeError("Specificity ratio/seed cardinality mismatch")
    if not specificity.get("missing_or_ineligible_pairs_counted_as_sign_failures"):
        raise RuntimeError("Frozen reducer did not count missing pairs as sign failures")
    threshold = card["dense_recipe_specificity_gate"]

    def complete(name: str, limit_name: str) -> dict[str, Any]:
        observed = [float(value) for value in ratios[name]]
        if not all(math.isfinite(value) and value >= 0.0 for value in observed):
            raise RuntimeError(f"Invalid observed specificity ratio: {name}")
        completed = observed + [math.inf] * len(missing)
        median = float(np.median(completed))
        return {
            "observed_count": len(observed),
            "imputed_positive_infinity_count": len(missing),
            "adverse_fixed_ten_median": median if math.isfinite(median) else None,
            "passed": math.isfinite(median) and median <= float(threshold[limit_name]),
        }

    h_row = complete("H_row", "maximum_median_h_row_ratio")
    h_assignment = complete(
        "H_assignment", "maximum_median_h_assignment_ratio"
    )
    wins = int(specificity["H_sparse_better_both_count"])
    if not 0 <= wins <= len(paired):
        raise RuntimeError("Malformed fixed-ten specificity win count")
    win_pass = wins >= int(
        threshold["minimum_paired_seeds_sparse_better_on_both_h_metrics"]
    )
    adverse_pass = bool(
        specificity["passed"] and h_row["passed"]
        and h_assignment["passed"] and win_pass
    )
    return {
        "all_ten_pairs_branch_evaluable": len(paired) == len(expected),
        "paired_seeds": paired,
        "globally_excluded_or_incomplete_pair_seeds": missing,
        "missing_pairs_counted_as_sign_failures": True,
        "adverse_completion_method": "append_positive_infinity_to_fixed_ten",
        "H_row": h_row,
        "H_assignment": h_assignment,
        "H_sparse_better_both_count_out_of_fixed_ten": wins,
        "fixed_ten_win_gate_passed": win_pass,
        "original_frozen_specificity_passed": bool(specificity["passed"]),
        "adverse_completion_passed": adverse_pass,
        "positive_relative_specificity_claim_permitted": adverse_pass,
    }
