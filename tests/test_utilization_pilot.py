"""Contract tests for the diagnostic utilization pilot."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from experiments.neurips_2026.utilization_pilot.pilot import (
    NCU_SMO_METRIC,
    MetricUnavailable,
    atomic_write_json,
    exact_task_identity,
    parse_ncu_smo_csv,
    validate_task_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_SCRIPT = REPO_ROOT / "scripts/neurips_2026/utilization_pilot/run.sh"
CPU_SCRIPT = REPO_ROOT / "scripts/neurips_2026/utilization_pilot/test_allocation.sh"
RUNNER = REPO_ROOT / "skae/training/runner.py"


def test_ncu_parser_requires_actual_smo_rows(tmp_path: Path) -> None:
    output = tmp_path / "ncu.csv"
    output.write_text(
        "==PROF== connected\n"
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        f'"1","kernel_a","{NCU_SMO_METRIC}","%", "27.5%"\n'
        f'"2","kernel_b","{NCU_SMO_METRIC}","%", "31.5"\n',
        encoding="utf-8",
    )
    parsed = parse_ncu_smo_csv(output)
    assert parsed["sample_count"] == 2
    assert parsed["values_percent"] == [27.5, 31.5]
    assert parsed["mean_percent"] == pytest.approx(29.5)
    assert parsed["gpu_utilization_as_smo"] is False


def test_ncu_parser_fails_closed_without_metric_rows(tmp_path: Path) -> None:
    output = tmp_path / "ncu-empty.csv"
    output.write_text(
        '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"1","kernel_a","sm__cycles_active.avg.pct_of_peak_sustained_active","%","99%"\n',
        encoding="utf-8",
    )
    with pytest.raises(MetricUnavailable, match="GPU-Util is not SMO"):
        parse_ncu_smo_csv(output)


def test_ncu_parser_rejects_wrong_units_and_out_of_bounds_values(tmp_path: Path) -> None:
    output = tmp_path / "ncu-invalid.csv"
    output.write_text(
        '"Metric Name","Metric Unit","Metric Value"\n'
        f'"{NCU_SMO_METRIC}","ratio","0.5"\n'
        f'"{NCU_SMO_METRIC}","%","101"\n',
        encoding="utf-8",
    )
    with pytest.raises(MetricUnavailable):
        parse_ncu_smo_csv(output)


def test_task_identity_freezes_scientific_shape() -> None:
    identity = exact_task_identity(label="candidate", variant="fused")
    validate_task_identity(identity)
    assert identity["comparison"] == {"label": "candidate", "variant": "fused"}
    assert identity["batch_size"] == 256
    assert identity["target_size"] == 256
    assert identity["sequence_length"] == 8
    assert identity["seed"] == 4
    mutated = dict(identity)
    mutated["batch_size"] = 512
    with pytest.raises(ValueError, match="batch_size"):
        validate_task_identity(mutated)


def test_restart_progress_marker_is_atomic_and_json(tmp_path: Path) -> None:
    marker = tmp_path / "progress.json"
    atomic_write_json(marker, {"status": "running", "completed_phases": ["warmup"]})
    assert json.loads(marker.read_text(encoding="utf-8"))["status"] == "running"
    assert not list(tmp_path.glob(".progress.json.*.tmp"))


def test_gpu_script_has_bounded_mila_allocation_and_requeue_contract() -> None:
    text = GPU_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --partition=long" in text
    assert "#SBATCH --gres=gpu:rtx8000:1" in text
    assert "#SBATCH --requeue" in text
    assert "#SBATCH --signal=B:TERM@60" in text
    assert "uv run python -m experiments.neurips_2026.utilization_pilot.run_pilot" in text
    assert re.search(r"#SBATCH --cpus-per-task=[4-8]", text)
    assert "#SBATCH --time=00:10:00" in text
    assert "production_eligible" not in text
    runtime_text = (REPO_ROOT / "experiments/neurips_2026/utilization_pilot/runtime.py").read_text(encoding="utf-8")
    assert "--profile-from-start" in runtime_text and '"off"' in runtime_text
    assert "nvidia_smi_unprofiled_1s.csv" in (REPO_ROOT / "experiments/neurips_2026/utilization_pilot/run_pilot.py").read_text(encoding="utf-8")


def test_cpu_test_script_has_restart_progress_receipt() -> None:
    text = CPU_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --partition=long" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=8G" in text
    assert "#SBATCH --time=00:10:00" in text
    assert "#SBATCH --requeue" in text
    assert "#SBATCH --signal=B:TERM@60" in text
    assert "uv run pytest tests/test_utilization_pilot.py" in text
    assert "--phase hold" in text
    assert "--phase validate-hold" in text
    assert "--phase resume" in text
    assert "--phase complete" in text
    assert "srun --exact" in text


def test_runner_pilot_instrumentation_is_opt_in_and_exact_range_is_supported() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "pilot_measure_steps: int = 0" in text
    assert "pilot_warmup_steps: int = 0" in text
    assert "cudaProfilerStart" in text
    assert "cudaProfilerStop" in text
    assert "cuda_synchronized_before_and_after" in text
    assert "not pilot_mode" in text
