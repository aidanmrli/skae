"""Validation contracts for reviewer-facing Allen--Cahn curve evidence."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


PACKET_ID = "allen_cahn_new_ic_replication"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_curve_compact(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
) -> None:
    _require(summary.get("schema_version") == 2, "Curve schema version drifted")
    _require(summary.get("packet_id") == PACKET_ID, "Curve packet ID drifted")
    _require(
        summary.get("status")
        == "descriptive_full_horizon_with_separate_primary_h200_inference",
        "Curve status drifted",
    )
    _require(len(rows) == 200, "Compact curve does not have 200 rows")
    steps = np.asarray([int(row["horizon_step"]) for row in rows])
    time = np.asarray([float(row["physical_time"]) for row in rows])
    _require(np.array_equal(steps, np.arange(1, 201)), "Curve horizon order drifted")
    np.testing.assert_allclose(time, 0.1 * steps, rtol=0, atol=1e-15)

    schema = summary["metric_schema"]["through_horizon_mean_field_mse"]
    _require(
        "divided by h" in schema["definition"],
        "Through-horizon mean definition is missing",
    )
    _require(
        schema["not_an_unnormalized_cumulative_sum"] is True,
        "Cumulative-sum caveat is missing",
    )
    dense = np.asarray(
        [float(row["dense_mean_through_horizon_mean_field_mse"]) for row in rows]
    )
    sparse = np.asarray(
        [float(row["sparse_mean_through_horizon_mean_field_mse"]) for row in rows]
    )
    dense_instantaneous = np.asarray(
        [float(row["dense_mean_instantaneous_field_mse"]) for row in rows]
    )
    sparse_instantaneous = np.asarray(
        [float(row["sparse_mean_instantaneous_field_mse"]) for row in rows]
    )
    reduction = np.asarray(
        [float(row["relative_reduction_of_arm_means"]) for row in rows]
    )
    _require(
        np.isfinite(dense).all() and np.isfinite(sparse).all(),
        "Compact curve is nonfinite",
    )
    _require(np.all(dense > 0) and np.all(sparse > 0), "Compact MSE is nonpositive")
    np.testing.assert_allclose(reduction, 1.0 - sparse / dense, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        dense, np.cumsum(dense_instantaneous) / steps, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        sparse, np.cumsum(sparse_instantaneous) / steps, rtol=0, atol=1e-15
    )
    _require(np.all(reduction > 0), "A descriptive arm-mean curve point reverses")

    for arm in ("dense", "sparse"):
        mean = np.asarray(
            [float(row[f"{arm}_mean_through_horizon_mean_field_mse"]) for row in rows]
        )
        lower = np.asarray(
            [float(row[f"{arm}_pointwise_ci95_lower"]) for row in rows]
        )
        upper = np.asarray(
            [float(row[f"{arm}_pointwise_ci95_upper"]) for row in rows]
        )
        _require(
            np.all(lower <= mean) and np.all(mean <= upper),
            f"{arm} band ordering drifted",
        )
    reduction_lower = np.asarray(
        [float(row["relative_reduction_pointwise_ci95_lower"]) for row in rows]
    )
    reduction_upper = np.asarray(
        [float(row["relative_reduction_pointwise_ci95_upper"]) for row in rows]
    )
    _require(
        np.all(reduction_lower <= reduction) and np.all(reduction <= reduction_upper),
        "Reduction band ordering drifted",
    )

    curve = summary["curve"]
    _require(
        "not_simultaneous" in curve["pointwise_interval_role"],
        "Band caveat is missing",
    )
    primary = summary["primary_h200"]
    _require("outcome_aware" in primary["role"], "Outcome-aware disclosure is missing")
    _require(
        "separate_from_curve_bands" in primary["role"],
        "Primary/band separation is missing",
    )
    np.testing.assert_allclose(primary["dense_mean"], dense[-1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(primary["sparse_mean"], sparse[-1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        primary["relative_reduction_of_arm_means"], reduction[-1], rtol=0, atol=1e-15
    )
    _require(primary["bootstrap_replicates"] == 100_000, "Bootstrap count drifted")

    trace = summary["full_trace_descriptive"]
    signed_gap = dense_instantaneous - sparse_instantaneous
    expected_count = int(np.count_nonzero(signed_gap > 0.0))
    expected_total = float(signed_gap.sum())
    expected_after_t2 = float(signed_gap[time > 2.0].sum())
    _require(
        trace["sparse_arm_mean_lower_count"] == expected_count == 200,
        "Instantaneous win count drifted",
    )
    _require(
        trace["sparse_arm_mean_lower_at_all_evaluated_times"] is True,
        "Instantaneous trace reverses",
    )
    np.testing.assert_allclose(
        trace["signed_gap_sum_steps_1_through_200"], expected_total, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        trace["signed_gap_sum_after_physical_time_2"], expected_after_t2, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        trace["signed_gap_share_after_physical_time_2"],
        expected_after_t2 / expected_total,
        rtol=0,
        atol=1e-15,
    )
    _require(
        "four-cell gate failed" in summary["figure_disclosure"],
        "Figure gate disclosure is missing",
    )
    title = summary["figure_title"].lower()
    _require("same-checkpoint" in title, "Figure title omits same-checkpoint scope")
    _require("robustness check" in title, "Figure title omits robustness-check scope")
    _require("replication" not in title, "Figure title overstates independent replication")
    _require(
        not np.isclose(
            primary["ci95_lower"],
            curve["h200_pointwise_ci95_lower"],
            rtol=0,
            atol=1e-8,
        ),
        "Primary and descriptive H200 intervals were conflated",
    )
