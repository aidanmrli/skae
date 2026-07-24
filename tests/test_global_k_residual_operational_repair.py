from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.neurips_2026.global_k_residual_forecast.evaluate import (
    _load_data_manifest,
)
from experiments.neurips_2026.global_k_residual_forecast.protocol import (
    atomic_json,
    sha256_path,
)
from experiments.neurips_2026.global_k_residual_forecast.telemetry import _assess_one


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "neurips_2026" / "global_k_residual_forecast"


def _card() -> dict:
    return json.loads((PACKAGE / "prediction_card.json").read_text())


def test_atomic_json_does_not_broadly_coerce_invalid_values(tmp_path: Path) -> None:
    class Deceptive:
        def __bool__(self) -> bool:
            return True

        def __float__(self) -> float:
            return 1.0

    for invalid in (
        Deceptive(), np.bool_(False), np.int64(5), np.float32(82.2),
        np.asarray([True]), {1, 2},
    ):
        with pytest.raises(TypeError, match="not JSON serializable"):
            atomic_json(tmp_path / "invalid.json", {"value": invalid})
    assert not (tmp_path / "invalid.json").exists()


def test_atomic_json_still_rejects_nonfinite_numpy_scalars_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stable.json"
    original = '{"stable": true}\n'
    path.write_text(original)
    with pytest.raises(ValueError, match="Out of range float"):
        atomic_json(path, {"value": np.float64(np.nan)})
    assert path.read_text() == original
    assert not path.with_suffix(".json.tmp").exists()


def test_short_window_assessment_is_strict_json_and_fails_closed(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.csv"
    utilization = [0, 33, 98, 96, 99, 85, 0]
    trace_path.write_text(
        "epoch_seconds,gpu_uuid,gpu_name,utilization_gpu,memory_used_mib,"
        "memory_total_mib\n"
        + "\n".join(
            f"{epoch},GPU-a,NVIDIA A100-SXM4-80GB,{util},4433,81920"
            for epoch, util in zip(range(99, 106), utilization)
        )
        + "\n"
    )
    window_path = tmp_path / "window.json"
    window_path.write_text(json.dumps({
        "schema_version": 1,
        "protocol_id": _card()["protocol_id"],
        "artifact_role": "forecast_compute_window",
        "mode": "smoke",
        "task_id": 0,
        "start_epoch_seconds": 100.0,
        "end_epoch_seconds": 104.0,
        "elapsed_seconds": 4.0,
    }))
    row = _assess_one(
        trace_path,
        window_path,
        _card()["gpu_utilization_gate"],
        protocol_id=_card()["protocol_id"],
        mode="smoke",
        task_id=0,
        freeze={"card_sha256": "a"},
    )
    assert row["minimum_rolling_utilization_percent"] is None
    assert row["checks"]["minimum_rolling_utilization"] is False
    assert all(type(value) is bool for value in row["checks"].values())
    assert row["passed"] is False
    output = tmp_path / "assessment.json"
    atomic_json(output, {"row": row, "passed": row["passed"]})
    decoded = json.loads(output.read_text())
    assert decoded["row"]["minimum_rolling_utilization_percent"] is None
    assert decoded["passed"] is False


def test_repaired_workload_and_failed_attempt_are_frozen() -> None:
    card = _card()
    corpora = card["outcome_free_trajectory_corpora"]
    assert corpora["evaluation"] == {
        "seeds": [801336907, 901337023, 1101337049],
        "trajectory_count_each": 131072,
        "horizon_steps": 500,
    }
    assert corpora["smoke_evaluation"] == {
        "seed": 314713093,
        "dataset_index": 0,
        "trajectory_count": 131072,
        "horizon_steps": 500,
    }
    assert corpora["route_fit"] == {
        "seed": 310947201, "trajectory_count": 1024, "horizon_steps": 128,
    }
    assert corpora["route_audit"] == {
        "seed": 517204603, "trajectory_count": 1024, "horizon_steps": 128,
    }
    assert [
        corpora["evaluation"]["trajectory_count_each"],
        corpora["evaluation"]["horizon_steps"] + 1,
        card["benchmark"]["state_dimension"],
    ] == [131072, 501, 2]
    assert [
        corpora["smoke_evaluation"]["trajectory_count"],
        corpora["smoke_evaluation"]["horizon_steps"] + 1,
        card["benchmark"]["state_dimension"],
    ] == [131072, 501, 2]
    history = card["computational_repair_history"]
    assert history["failed_v1_freeze"]["card_sha256"] == (
        "d60d833d84961da0c5931e6ee6cf3dbf763c12ccf3607764c5700f7cba2808dc"
    )
    assert history["failed_v1_jobs"]["science_array"]["state"] == "CANCELLED"
    assert history["failed_v1_outcome_blind_gate_observations"][
        "compute_window_sample_count"
    ] == 5
    assert history["failed_v1_outcome_blind_gate_observations"][
        "minimum_rolling_utilization_percent"
    ] is None
    assert history["failed_v1_artifact_sha256"]["smoke_shard"] == (
        "b8688a9a83527b8fe981757a121a06cc166df4c80f8b87aef6511a3fd51574d6"
    )
    assert history["failed_v1_artifact_sha256"]["telemetry_trace"] == (
        "6cf85b75c8945abfc9a432d09015376be12f3404bb4b160c5c82ab63b231e4bb"
    )
    assert history["failed_v2_freeze"] == {
        "card_sha256": (
            "a89f06ea60804e9c04359ca42b057adec476ecf5e8b66f75bec3f2cb23ee2bd6"
        ),
        "task_manifest_sha256": (
            "86a3dce2ce8fd6ca569aebcccb6812ac6c3ee206ec21ba8e2ccf2642305fb024"
        ),
        "source_manifest_sha256": (
            "a4bfe0715a5f08226b616805789354e276fa6319e2f702e33c7b75ea43492d15"
        ),
        "queue_script_sha256": (
            "fdf9105bb41522dfd0f8c2632923d136c55ed5e4eeda088079267c254a515e87"
        ),
        "output_root": (
            "/network/scratch/l/lia/skae/global_k_residual_forecast_v2_20260721"
        ),
    }
    assert history["failed_v2_jobs"]["smoke_gate"] == {
        "job_id": 10165742, "state": "FAILED", "exit_code": "2:0",
    }
    assert history["failed_v2_jobs"]["science_array"]["state"] == "CANCELLED"
    assert history["failed_v2_artifact_sha256"]["gpu_assessment"] == (
        "7ea2bcccb8a6fdafff486a588071fc6d8117636f1ede36d5eb1b28f68432d320"
    )
    assert history["failed_v2_artifact_sha256"]["smoke_shard"] == (
        "d33b7edb2886b163dd6fd9e8b88fd6c1bf5ce227c8019491f068b846103afa4a"
    )
    v2 = history["failed_v2_outcome_blind_gate_observations"]
    assert v2["compute_window_sample_count"] == 46
    assert v2["compute_window_mean_utilization_percent"] == 96.84782608695652
    assert v2["outside_window_sample_count"] == 18
    assert v2["outside_window_utilization_sum"] == 18.0
    assert v2["allocation_wide_mean_utilization_percent"] == 69.890625
    assert v2["failed_checks"] == ["minimum_allocation_wide_mean_utilization"]
    assert v2["forecast_outcomes_read"] is False
    assert card["statistics"]["primary_unit"] == "ten paired model initialization seeds"
    assert card["matched_sign_pair_permutation_null"]["scale_match_points"] == 8192
    gate = card["gpu_utilization_gate"]
    assert gate["minimum_compute_window_samples"] == 31
    assert gate["minimum_compute_window_duration_seconds"] == 30.0
    assert gate["minimum_compute_window_mean_utilization_percent"] == 85.0
    assert gate["minimum_compute_window_p10_utilization_percent"] == 80.0
    assert gate["minimum_rolling_utilization_percent"] == 80.0
    assert gate["minimum_allocation_wide_mean_utilization_percent"] == 70.0
    assert "no padding" in card["gpu_utilization_gate"]["meaningful_workload"]
    rationale = card["gpu_utilization_gate"]["workload_scaling_rationale"]
    assert "64.2495" in rationale
    assert "54266 MiB" in rationale
    assert "70.38 percent" in rationale
    assert "no forecast outcome was inspected" in rationale
    assert card["freeze"]["output_root"].endswith(
        "global_k_residual_forecast_v3_20260721"
    )
    # Authorization is a historical lifecycle fact: V3 was launched and later
    # classified invalid after a nonfinite payload blocked the exact packet.
    assert card["freeze"]["launch_authorized"] is True


def test_data_manifest_hash_and_shape_are_fail_closed(tmp_path: Path) -> None:
    card = _card()
    card["outcome_free_trajectory_corpora"] = {
        "route_fit": {"seed": 11, "trajectory_count": 2, "horizon_steps": 1},
        "route_audit": {"seed": 12, "trajectory_count": 2, "horizon_steps": 1},
        "evaluation": {
            "seeds": [13, 14, 15], "trajectory_count_each": 2, "horizon_steps": 1,
        },
        "smoke_evaluation": {
            "seed": 16, "dataset_index": 0, "trajectory_count": 2,
            "horizon_steps": 1,
        },
    }
    freeze = {"card_sha256": "a", "source_manifest_sha256": "b"}
    data_dir = tmp_path / "outcome_free_data"
    data_dir.mkdir()
    specs = [
        ("route_fit", None, 11, "route_fit.pt"),
        ("route_audit", None, 12, "route_audit.pt"),
        ("evaluation", 0, 13, "evaluation_0.pt"),
        ("evaluation", 1, 14, "evaluation_1.pt"),
        ("evaluation", 2, 15, "evaluation_2.pt"),
        ("smoke_evaluation", 0, 16, "smoke_evaluation_0.pt"),
    ]
    rows = []
    for role, index, seed, filename in specs:
        path = data_dir / filename
        metadata = {
            "protocol_id": card["protocol_id"],
            "role": role,
            "seed": seed,
            "shape": [2, 2, 2],
            "contains_forecast_or_representation_outcomes": False,
        }
        row = {"role": role, "seed": seed, "path": str(path)}
        if index is not None:
            metadata["dataset_index"] = index
            row["dataset_index"] = index
        torch.save(
            {"trajectories": torch.zeros(2, 2, 2), "metadata": metadata}, path
        )
        row["sha256"] = sha256_path(path)
        rows.append(row)
    (data_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": "outcome_free_physical_trajectory_manifest",
        "freeze": freeze,
        "rows": rows,
    }))
    assert len(_load_data_manifest(tmp_path, card, freeze)["rows"]) == 6
    tampered = data_dir / "evaluation_0.pt"
    tampered.write_bytes(tampered.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _load_data_manifest(tmp_path, card, freeze)
