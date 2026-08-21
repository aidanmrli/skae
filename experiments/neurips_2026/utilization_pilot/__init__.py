"""Exact-shape GPU utilization pilot and measurement helpers.

The pilot is deliberately diagnostic-only.  It records enough provenance to
compare a base and candidate implementation, but it never grants production
eligibility from GPU-Util or from a missing SM occupancy measurement.
"""

from .pilot import (
    NCU_SMO_METRIC,
    TASK_IDENTITY,
    MetricUnavailable,
    atomic_write_json,
    build_source_manifest,
    exact_task_identity,
    parse_ncu_smo_csv,
    parse_ncu_metrics,
    require_ncu_smo_metrics,
    validate_task_identity,
    with_measurement_window,
)

__all__ = [
    "NCU_SMO_METRIC",
    "TASK_IDENTITY",
    "MetricUnavailable",
    "atomic_write_json",
    "build_source_manifest",
    "exact_task_identity",
    "parse_ncu_smo_csv",
    "parse_ncu_metrics",
    "require_ncu_smo_metrics",
    "validate_task_identity",
    "with_measurement_window",
]
