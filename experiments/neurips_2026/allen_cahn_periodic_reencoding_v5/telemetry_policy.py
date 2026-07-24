"""Sparse natural-sample GPU monitoring policy for execution V5."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


REQUIRED_HARDWARE_PLAN: dict[str, int | float | str | bool] = {
    "device_name": "NVIDIA A100L 80GB",
    "boundary_samples_excluded_per_side": 0,
    "minimum_all_window_samples_before_boundary_exclusion": 3,
    "minimum_retained_all_window_samples": 3,
    "minimum_mean_retained_all_window_gpu_utilization_percent": 90.0,
    "maximum_peak_memory_fraction": 0.8,
    "no_padding": True,
}


def window_statistics(
    samples: list[dict[str, Any]], *, start: float, end: float
) -> dict[str, Any]:
    """Summarize every natural in-window sample without dropping or padding."""

    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("Evaluation marker interval is invalid")
    retained = [
        row for row in samples if start <= float(row["epoch_seconds"]) <= end
    ]
    if not retained:
        raise ValueError("No telemetry samples fall inside the evaluation interval")
    epochs = np.asarray(
        [row["epoch_seconds"] for row in retained], dtype=np.float64
    )
    utilization = np.asarray(
        [row["utilization_percent"] for row in retained], dtype=np.float64
    )
    memory_fraction = np.asarray(
        [row["memory_used_mib"] / row["memory_total_mib"] for row in retained],
        dtype=np.float64,
    )
    gaps = np.diff(epochs)
    return {
        "start_epoch_seconds": float(start),
        "end_epoch_seconds": float(end),
        "duration_seconds": float(end - start),
        "all_window_samples": int(len(retained)),
        "boundary_samples_excluded_per_side": 0,
        "retained_all_window_samples": int(len(retained)),
        "zero_utilization_retained_samples": int(np.sum(utilization == 0.0)),
        "mean_retained_all_window_gpu_utilization_percent": float(
            utilization.mean()
        ),
        "p10_retained_all_window_gpu_utilization_percent": float(
            np.quantile(utilization, 0.10)
        ),
        "median_sample_cadence_seconds": (
            float(np.median(gaps)) if gaps.size else float("inf")
        ),
        "maximum_sample_gap_seconds": (
            float(gaps.max()) if gaps.size else float("inf")
        ),
        "leading_marker_edge_gap_seconds": float(epochs[0] - start),
        "trailing_marker_edge_gap_seconds": float(end - epochs[-1]),
        "peak_memory_fraction": float(memory_fraction.max()),
        "utilization_filter_applied": False,
    }


def gate_checks(window: dict[str, Any], plan: dict[str, Any]) -> dict[str, bool]:
    """Gate sample count, all-sample mean utilization, and memory only."""

    checks = {
        "exact_unconditional_boundary_exclusion": (
            window["boundary_samples_excluded_per_side"]
            == plan["boundary_samples_excluded_per_side"]
            == 0
        ),
        "no_utilization_filter": window["utilization_filter_applied"] is False,
        "minimum_all_window_samples": window["all_window_samples"]
        >= int(plan["minimum_all_window_samples_before_boundary_exclusion"]),
        "minimum_retained_all_window_samples": window["retained_all_window_samples"]
        >= int(plan["minimum_retained_all_window_samples"]),
        "minimum_mean_retained_utilization": window[
            "mean_retained_all_window_gpu_utilization_percent"
        ]
        >= float(plan["minimum_mean_retained_all_window_gpu_utilization_percent"]),
        "strict_peak_memory_fraction": window["peak_memory_fraction"]
        < float(plan["maximum_peak_memory_fraction"]),
    }
    return {name: bool(passed) for name, passed in checks.items()}


def install() -> None:
    """Install only the V5 execution-monitoring policy into frozen auditors."""

    from experiments.neurips_2026.allen_cahn_periodic_reencoding import (
        smoke_audit,
        telemetry,
    )

    telemetry.REQUIRED_HARDWARE_PLAN = REQUIRED_HARDWARE_PLAN
    telemetry.gate_checks = gate_checks
    telemetry.window_statistics = window_statistics
    smoke_audit.gate_checks = gate_checks
    smoke_audit.window_statistics = window_statistics
