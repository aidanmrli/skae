"""Recompute decision-driving diagnostics from persisted primitive values."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


EXPECTED_PARAMETER_COUNTS = {
    "sparse_trainable_parameter_counts": {
        "total": 95104,
        "encoder": 29056,
        "decoder": 512,
        "koopman": 65536,
        "other": 0,
    },
    "dense_trainable_parameter_counts": {
        "total": 87040,
        "encoder": 20992,
        "decoder": 512,
        "koopman": 65536,
        "other": 0,
    },
}


def _finite(value: Any, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RuntimeError(f"{role} is not finite")
    return float(value)


def _same(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-12, atol=1e-12))


def recompute_route_diagnostics(
    shard: dict[str, Any], card: dict[str, Any]
) -> dict[str, Any]:
    """Recompute every route/null validity decision used by the summary."""

    fit = shard["label_free_family_fit"]
    audit = shard["held_out_route_audit"]
    null = shard["matched_coordinate_null"]
    route = card["label_free_support_routing"]
    retained = int(fit["retained_family_count"])
    cardinalities = fit["support_cardinalities"]
    expected_fit_valid = bool(
        fit["fallback_used"] is False
        and fit["maximum_family_truncation_used"] is False
        and int(route["minimum_retained_families"])
        <= retained
        <= int(route["maximum_retained_families"])
        and all(isinstance(value, int) and 0 < value < 256 for value in cardinalities)
        and fit["sign_pair_exclusivity"] is True
        and _finite(fit["retained_fit_coverage"], "fit coverage")
        >= float(route["minimum_retained_fit_coverage"])
    )
    if fit["fit_valid"] is not expected_fit_valid:
        raise RuntimeError("Persisted fit_valid disagrees with recomputed thresholds")

    counts = audit["assignment_count_by_family"]
    total = sum(counts)
    active = sum(
        count >= float(route["minimum_audit_family_fraction"]) * total
        for count in counts
    )
    if audit["active_family_count_at_minimum_fraction"] != active:
        raise RuntimeError("Persisted active-family count disagrees with raw counts")
    expected_audit_valid = bool(
        _finite(audit["mean_nearest_jaccard"], "mean audit Jaccard")
        >= float(route["minimum_mean_audit_jaccard"])
        and _finite(
            audit["confident_assignment_fraction"], "confident audit fraction"
        )
        >= float(route["minimum_confident_audit_fraction"])
        and active >= int(route["minimum_retained_families"])
    )
    if audit["label_free_route_audit_valid"] is not expected_audit_valid:
        raise RuntimeError(
            "Persisted label_free_route_audit_valid disagrees with recomputed thresholds"
        )

    alignment = audit["evaluation_only_alignment"]
    contingency = alignment["contingency_family_by_label"]
    row_sums = [sum(row) for row in contingency]
    if row_sums != counts:
        raise RuntimeError("Contingency row sums disagree with assignment counts")
    contingency_total = sum(row_sums)
    if contingency_total != total or total <= 0:
        raise RuntimeError("Contingency total disagrees with route-audit counts")
    purity = sum(max(row) for row in contingency) / total
    persisted_purity = _finite(
        alignment["family_conditional_basin_purity"], "persisted route purity"
    )
    if not _same(persisted_purity, purity):
        raise RuntimeError("Persisted route purity disagrees with contingency")

    null_spec = card["matched_sign_pair_permutation_null"]
    lower = float(null_spec["minimum_scale_ratio"])
    upper = float(null_spec["maximum_scale_ratio"])
    selected_eligible = []
    for row in null["selected_scale_rows"]:
        source_ratio = _finite(row["source_rms_ratio"], "null source ratio")
        update_ratio = _finite(row["update_rms_ratio"], "null update ratio")
        expected_score = abs(math.log(source_ratio)) + abs(math.log(update_ratio))
        if not _same(_finite(row["score"], "null scale score"), expected_score):
            raise RuntimeError("Persisted null score disagrees with scale ratios")
        eligible = lower <= source_ratio <= upper and lower <= update_ratio <= upper
        if row["eligible"] is not eligible:
            raise RuntimeError("Persisted null eligibility disagrees with scale ratios")
        selected_eligible.append(eligible)
    expected_scale_valid = bool(
        int(null["eligible_candidate_count"]) >= int(null_spec["selected_count"])
    )
    if expected_scale_valid and not all(selected_eligible):
        raise RuntimeError("A valid scale match selected an ineligible null")
    if null["scale_match_valid"] is not expected_scale_valid:
        raise RuntimeError("Persisted scale_match_valid disagrees with candidate count")

    return {
        "fit_valid": expected_fit_valid,
        "label_free_route_audit_valid": expected_audit_valid,
        "scale_match_valid": expected_scale_valid,
        "basin_purity": purity,
        "basin_purity_passed": purity
        >= float(card["mechanism_gate"]["minimum_route_audit_basin_purity"]),
    }


def summarize_route_validity(
    shards: list[dict[str, Any]], card: dict[str, Any]
) -> dict[str, Any]:
    """Build the route gate exclusively from recomputed diagnostic decisions."""

    rows = []
    for shard in shards:
        recomputed = recompute_route_diagnostics(shard, card)
        checks = {
            "label_free_fit_valid": recomputed["fit_valid"],
            "held_out_label_free_route_audit_valid": recomputed[
                "label_free_route_audit_valid"
            ],
            "evaluation_only_basin_purity": recomputed["basin_purity_passed"],
            "matched_null_scale_valid": recomputed["scale_match_valid"],
        }
        rows.append(
            {
                "model_seed": int(shard["model_seed"]),
                "basin_purity": recomputed["basin_purity"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return {
        "rows": rows,
        "passed": len(rows) == 10 and all(row["passed"] for row in rows),
    }


def validate_count_weighted_basin_h200(
    methods: dict[str, dict[str, Any]],
    basin_rows: dict[str, dict[str, Any]],
    trajectory_count: int,
) -> None:
    """Authenticate global H200 MSE against count-weighted basin aggregates."""

    counts = [int(basin_rows[label]["trajectory_count"]) for label in sorted(basin_rows)]
    if sum(counts) != trajectory_count:
        raise RuntimeError("Basin counts do not cover the full trajectory panel")
    for name, method in methods.items():
        global_value = method["through_h200_mse"]
        basin_values = [
            basin_rows[label]["through_h200_mse_by_method"][name]
            for label in sorted(basin_rows)
        ]
        if global_value is None:
            if all(value is not None for value in basin_values):
                raise RuntimeError(
                    f"Method {name} suppresses global H200 despite complete basin values"
                )
            continue
        if any(value is None for value in basin_values):
            raise RuntimeError(f"Method {name} has incomplete basin H200 values")
        weighted = sum(
            count * _finite(value, f"basin H200 for {name}")
            for count, value in zip(counts, basin_values)
        ) / trajectory_count
        if not np.isclose(
            _finite(global_value, f"global H200 for {name}"),
            weighted,
            rtol=1e-5,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"Method {name} global H200 disagrees with count-weighted basins"
            )
