from __future__ import annotations

from pathlib import Path

import pytest

from experiments.neurips_2026.allen_cahn_forecast_replication.io import (
    CARD_PATH,
    load_card,
)
from experiments.neurips_2026.allen_cahn_forecast_replication.telemetry import (
    _bind_slurm_job,
    evaluation_gate_checks,
    parse_samples,
    window_statistics,
)


CARD, _CARD_HASH = load_card(CARD_PATH)


def _telemetry_sample(epoch: float, utilization: float) -> dict[str, object]:
    return {
        "epoch_seconds": float(epoch),
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "utilization_percent": float(utilization),
        "memory_used_mib": 8_000.0,
        "memory_total_mib": 80_000.0,
    }


def test_telemetry_gate_includes_zero_utilization_in_all_retained_samples() -> None:
    samples = [_telemetry_sample(0, 100)]
    samples.extend(_telemetry_sample(epoch, 0) for epoch in range(1, 101))
    samples.extend(_telemetry_sample(epoch, 100) for epoch in range(101, 111))
    samples.append(_telemetry_sample(111, 100))
    statistics = window_statistics(
        samples,
        start=0.0,
        end=111.0,
        boundary_exclusion_per_side=1,
    )
    checks = evaluation_gate_checks(statistics, CARD["hardware_plan"])
    assert statistics["all_window_samples"] == 112
    assert statistics["retained_all_window_samples"] == 110
    assert statistics["zero_utilization_retained_samples_descriptive"] == 100
    assert statistics["utilization_filter_applied"] is False
    assert statistics["mean_retained_all_window_gpu_utilization_percent"] == pytest.approx(
        1000.0 / 110.0
    )
    assert checks["mean_retained_all_window_gpu_utilization"] is False
    assert checks["strict_p10_retained_all_window_gpu_utilization"] is False
    assert all(
        passed
        for name, passed in checks.items()
        if name
        not in {
            "mean_retained_all_window_gpu_utilization",
            "strict_p10_retained_all_window_gpu_utilization",
        }
    )


def test_telemetry_boundary_cadence_gap_and_marker_coverage_are_fail_closed() -> None:
    clean = [_telemetry_sample(epoch, 100) for epoch in range(12)]
    clean_statistics = window_statistics(
        clean,
        start=0.0,
        end=11.0,
        boundary_exclusion_per_side=1,
    )
    clean_checks = evaluation_gate_checks(clean_statistics, CARD["hardware_plan"])
    assert clean_statistics["boundary_samples_excluded_per_side"] == 1
    assert clean_statistics["retained_all_window_samples"] == 10
    assert all(clean_checks.values())

    gapped_times = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
    gapped = [_telemetry_sample(epoch, 100) for epoch in gapped_times]
    gapped_statistics = window_statistics(
        gapped,
        start=0.0,
        end=12.0,
        boundary_exclusion_per_side=1,
    )
    gapped_checks = evaluation_gate_checks(gapped_statistics, CARD["hardware_plan"])
    assert gapped_statistics["maximum_sample_gap_seconds"] == 2.0
    assert gapped_checks["maximum_sample_gap"] is False

    uncovered = [_telemetry_sample(epoch, 100) for epoch in range(2, 14)]
    uncovered_statistics = window_statistics(
        uncovered,
        start=0.0,
        end=13.0,
        boundary_exclusion_per_side=1,
    )
    uncovered_checks = evaluation_gate_checks(
        uncovered_statistics, CARD["hardware_plan"]
    )
    assert uncovered_statistics["leading_marker_edge_gap_seconds"] == 2.0
    assert uncovered_checks["leading_marker_edge_coverage"] is False


def test_raw_telemetry_requires_one_gpu_uuid_and_strict_timestamps(
    tmp_path: Path,
) -> None:
    header = (
        "timestamp, uuid, name, utilization.gpu [%], memory.used [MiB], "
        "memory.total [MiB]\n"
    )
    two_gpu = tmp_path / "two_gpu.csv"
    two_gpu.write_text(
        header
        + "2026/07/20 12:00:00.000, GPU-a, NVIDIA A100, 95 %, 100 MiB, 80000 MiB\n"
        + "2026/07/20 12:00:01.000, GPU-b, NVIDIA A100, 95 %, 100 MiB, 80000 MiB\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one valid GPU UUID"):
        parse_samples(two_gpu)

    duplicate_time = tmp_path / "duplicate_time.csv"
    duplicate_time.write_text(
        header
        + "2026/07/20 12:00:00.000, GPU-a, NVIDIA A100, 95 %, 100 MiB, 80000 MiB\n"
        + "2026/07/20 12:00:00.000, GPU-a, NVIDIA A100, 95 %, 100 MiB, 80000 MiB\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_samples(duplicate_time)


def test_telemetry_binds_available_slurm_job_without_breaking_portability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = {"environment": {"slurm_job_id": "12345"}}
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    assert _bind_slurm_job(runtime) == "12345"
    monkeypatch.setenv("SLURM_JOB_ID", "54321")
    with pytest.raises(RuntimeError, match="lineage differ"):
        _bind_slurm_job(runtime)
    monkeypatch.delenv("SLURM_JOB_ID")
    assert _bind_slurm_job(
        {"environment": {"slurm_job_id": "not_recorded"}}
    ) == "not_recorded"
