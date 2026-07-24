"""Fail-closed telemetry-gate and forecast-method validation."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from experiments.neurips_2026.global_k_residual_forecast.protocol import load_json


SPARSE_METHODS = (
    "sparse_routed_residual",
    "sparse_routed_nonresidual",
    "sparse_global_residual",
    "sparse_global_standard_reencode",
    *(f"support_permutation_null_{index:02d}" for index in range(32)),
    "sparse_global_pure_k",
    "persistence_identity",
)
DENSE_METHODS = (
    "dense_global_standard_reencode",
    "dense_global_residual",
    "dense_global_pure_k",
)
ALL_METHODS = SPARSE_METHODS + DENSE_METHODS
H500_METHODS = {
    "sparse_routed_residual",
    "sparse_routed_nonresidual",
    "sparse_global_residual",
    "sparse_global_standard_reencode",
    "sparse_global_pure_k",
    "persistence_identity",
    *DENSE_METHODS,
}
PREDICTOR_ASSERTIONS = {
    "one_unchanged_global_k_per_checkpoint",
    "no_latent_dynamics_or_local_operator_fit",
    "support_family_fit_and_assignment_use_no_labels_or_basin_count",
    "every_predicted_physical_state_is_reencoded_at_every_step",
    "no_teacher_forcing_truth_reset_or_periodic_refresh",
    "support_routed_predictor_is_autonomous_nonlinear_not_pure_k_power",
}
TELEMETRY_CHECKS = {
    "minimum_compute_window_samples",
    "minimum_compute_window_duration",
    "trace_brackets_compute_start",
    "trace_brackets_compute_end",
    "allocation_sampling_cadence",
    "compute_sampling_cadence",
    "allocation_maximum_sampling_gap",
    "compute_maximum_sampling_gap",
    "minimum_compute_window_mean_utilization",
    "minimum_compute_window_p10_utilization",
    "minimum_rolling_utilization",
    "minimum_allocation_wide_mean_utilization",
}
SMOKE_CHECKS = {
    "exact_schema",
    "schema_version",
    "protocol_id",
    "artifact_role",
    "task_id",
    "freeze",
    "all_required_predictions_finite",
    "exact_method_count",
    "route_fit_completed",
    "route_audit_completed",
    "null_scale_matching_completed",
    "no_metrics_labels_or_alignment_persisted",
    "outcomes_not_inspected",
    "finite_positive_elapsed_seconds",
}


def exact_keys(value: Any, expected: set[str], role: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = set(value) if isinstance(value, dict) else type(value)
        raise RuntimeError(f"{role} keys drifted: {observed}")
    return value


def finite_number(
    value: Any, role: str, *, minimum: float | None = None
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RuntimeError(f"{role} is not a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise RuntimeError(f"{role} is below {minimum}")
    return result


def require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{role} is not a SHA-256 digest")
    return value


def validate_gate(
    path: Path,
    *,
    mode: str,
    task_ids: list[int],
    protocol_id: str,
    freeze: dict[str, Any],
    output_root: Path,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    from experiments.neurips_2026.global_k_residual_forecast.telemetry import (
        _assess_one,
        _validate_smoke_shard,
    )

    gate = exact_keys(
        load_json(path),
        {
            "schema_version", "protocol_id", "artifact_role", "mode", "freeze",
            "rows", "smoke_checks", "passed", "forecast_outcomes_read",
        },
        f"{mode} telemetry gate",
    )
    if (
        gate["schema_version"] != 1
        or gate["protocol_id"] != protocol_id
        or gate["artifact_role"] != f"outcome_blind_{mode}_gpu_assessment"
        or gate["mode"] != mode
        or gate["freeze"] != freeze
        or gate["passed"] is not True
        or gate["forecast_outcomes_read"] is not False
    ):
        raise RuntimeError(f"{mode} telemetry gate identity failed")
    rows = gate["rows"]
    if not isinstance(rows, list) or [row.get("task_id") for row in rows] != task_ids:
        raise RuntimeError(f"{mode} telemetry task roster drifted")
    row_keys = {
        "schema_version", "protocol_id", "artifact_role", "mode", "task_id", "freeze",
        "trace_path", "compute_window_path", "allocation_sample_count",
        "compute_window_sample_count", "gpu_uuid", "gpu_name",
        "allocation_duration_seconds", "compute_window_duration_seconds",
        "trace_to_compute_start_gap_seconds", "trace_after_compute_end_gap_seconds",
        "allocation_median_sampling_interval_seconds",
        "compute_median_sampling_interval_seconds",
        "allocation_maximum_sampling_gap_seconds",
        "compute_maximum_sampling_gap_seconds", "allocation_mean_utilization_percent",
        "compute_window_mean_utilization_percent",
        "compute_window_p10_utilization_percent", "minimum_rolling_utilization_percent",
        "peak_memory_mib", "memory_total_mib", "checks", "passed",
    }
    nonnumeric = {
        "schema_version", "protocol_id", "artifact_role", "mode", "task_id", "freeze",
        "trace_path", "compute_window_path", "gpu_uuid", "gpu_name", "checks", "passed",
    }
    for task_id, row in zip(task_ids, rows):
        exact_keys(row, row_keys, f"{mode} telemetry row {task_id}")
        expected_dir = output_root / mode
        if (
            row["schema_version"] != 1
            or row["protocol_id"] != protocol_id
            or row["artifact_role"] != "outcome_blind_gpu_task_assessment"
            or row["mode"] != mode
            or row["task_id"] != task_id
            or row["freeze"] != freeze
            or Path(row["trace_path"])
            != expected_dir / "telemetry" / f"task_{task_id:02d}.csv"
            or Path(row["compute_window_path"])
            != expected_dir / "compute_windows" / f"task_{task_id:02d}.json"
            or row["passed"] is not True
        ):
            raise RuntimeError(f"{mode} telemetry row identity failed for task {task_id}")
        if not isinstance(row["gpu_uuid"], str) or not row["gpu_uuid"]:
            raise RuntimeError("Telemetry GPU UUID is missing")
        if "A100" not in str(row["gpu_name"]):
            raise RuntimeError("Telemetry GPU is not A100 class")
        for key in row_keys - nonnumeric:
            finite_number(row[key], f"telemetry {key}", minimum=0.0)
        if row["memory_total_mib"] < 75 * 1024:
            raise RuntimeError("Telemetry GPU capacity is below 75 GiB")
        checks = exact_keys(row["checks"], TELEMETRY_CHECKS, "telemetry checks")
        if any(value is not True for value in checks.values()):
            raise RuntimeError("A telemetry check is not exactly true")
        recomputed = _assess_one(
            expected_dir / "telemetry" / f"task_{task_id:02d}.csv",
            expected_dir / "compute_windows" / f"task_{task_id:02d}.json",
            thresholds,
            protocol_id=protocol_id,
            mode=mode,
            task_id=task_id,
            freeze=freeze,
        )
        if row != recomputed:
            raise RuntimeError(
                f"{mode} telemetry row {task_id} disagrees with raw trace/marker"
            )
    if mode == "smoke":
        checks = exact_keys(gate["smoke_checks"], SMOKE_CHECKS, "smoke checks")
        if any(value is not True for value in checks.values()):
            raise RuntimeError("Smoke shard validation did not pass every field")
        recomputed_smoke = _validate_smoke_shard(
            output_root / "smoke" / "shards" / "task_00.json",
            protocol_id=protocol_id,
            freeze=freeze,
        )
        if checks != recomputed_smoke:
            raise RuntimeError("Smoke checks disagree with the raw smoke shard")
    elif gate["smoke_checks"] is not None:
        raise RuntimeError("Scientific telemetry unexpectedly contains smoke checks")
    return gate


def validate_method(name: str, row: Any) -> None:
    base_keys = {
        "mean_mse_curve", "finite_through_h200_for_every_trajectory",
        "through_h200_mse", "terminal_h200_mse", "late_h101_h200_mse",
        "nonfinite_policy",
    }
    h500_keys = {
        "finite_through_h500_for_every_trajectory", "through_h500_mse",
        "terminal_h500_mse",
    }
    expected = base_keys | (h500_keys if name in H500_METHODS else set())
    exact_keys(row, expected, f"method {name}")
    expected_length = 500 if name in H500_METHODS else 200
    curve = row["mean_mse_curve"]
    if not isinstance(curve, list) or len(curve) != expected_length:
        raise RuntimeError(f"Method {name} curve length drifted")
    for value in curve:
        if value is not None:
            finite_number(value, f"method {name} curve value", minimum=0.0)
    h200_finite = row["finite_through_h200_for_every_trajectory"]
    if not isinstance(h200_finite, bool):
        raise RuntimeError(f"Method {name} H200 finiteness is not boolean")
    h200 = curve[:200]
    h200_endpoints = (
        row["through_h200_mse"], row["terminal_h200_mse"],
        row["late_h101_h200_mse"],
    )
    if h200_finite:
        if any(value is None for value in h200):
            raise RuntimeError(f"Method {name} loses a valid H200 curve prefix")
        observed = np.asarray(h200, dtype=np.float64)
        expected_values = (observed.mean(), observed[-1], observed[100:].mean())
        if any(value is None for value in h200_endpoints) or not np.allclose(
            np.asarray(h200_endpoints, dtype=np.float64),
            expected_values,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(f"Method {name} H200 endpoint identity failed")
    elif (
        any(value is not None for value in h200_endpoints)
        or all(value is not None for value in h200)
    ):
        raise RuntimeError(f"Method {name} invalid H200 endpoint was not suppressed")
    if name not in H500_METHODS:
        return
    h500_finite = row["finite_through_h500_for_every_trajectory"]
    if not isinstance(h500_finite, bool):
        raise RuntimeError(f"Method {name} H500 finiteness is not boolean")
    endpoints = (row["through_h500_mse"], row["terminal_h500_mse"])
    if h500_finite:
        if any(value is None for value in curve):
            raise RuntimeError(f"Method {name} finite H500 curve contains missing values")
        observed = np.asarray(curve, dtype=np.float64)
        if any(value is None for value in endpoints) or not np.allclose(
            np.asarray(endpoints, dtype=np.float64),
            (observed.mean(), observed[-1]),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(f"Method {name} H500 endpoint identity failed")
    elif (
        any(value is not None for value in endpoints)
        or all(value is not None for value in curve)
    ):
        raise RuntimeError(f"Method {name} invalid H500 endpoint was not suppressed")
