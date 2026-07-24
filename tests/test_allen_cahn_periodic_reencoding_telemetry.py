from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.neurips_2026.allen_cahn_periodic_reencoding.telemetry import (
    REQUIRED_HARDWARE_PLAN,
    audit_and_issue_guard,
    gate_checks,
    read_samples,
    sha256_path,
    window_statistics,
)


def _sample(
    epoch: float,
    utilization: float = 100.0,
    memory_fraction: float = 0.25,
) -> dict[str, object]:
    total = 81_920.0
    return {
        "epoch_seconds": float(epoch),
        "gpu_uuid": "GPU-periodic-test",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "utilization_percent": float(utilization),
        "memory_used_mib": total * float(memory_fraction),
        "memory_total_mib": total,
    }


def test_marker_window_drops_exact_edges_and_retains_all_zeros() -> None:
    samples = [_sample(0.0)]
    samples.extend(_sample(float(index), 0.0) for index in range(1, 11))
    samples.append(_sample(11.0))
    window = window_statistics(samples, start=0.0, end=11.0)
    checks = gate_checks(window, dict(REQUIRED_HARDWARE_PLAN))

    assert window["all_window_samples"] == 12
    assert window["boundary_samples_excluded_per_side"] == 1
    assert window["retained_all_window_samples"] == 10
    assert window["zero_utilization_retained_samples"] == 10
    assert window["mean_retained_all_window_gpu_utilization_percent"] == 0.0
    assert window["utilization_filter_applied"] is False
    assert checks["minimum_mean_retained_utilization"] is False
    assert checks["strict_p10_retained_utilization"] is False
    assert all(
        passed
        for name, passed in checks.items()
        if name
        not in {
            "minimum_mean_retained_utilization",
            "strict_p10_retained_utilization",
        }
    )


def test_strict_p10_and_peak_memory_thresholds_are_fail_closed() -> None:
    utilization = [100.0, 80.0, 80.0] + [100.0] * 9
    p10_samples = [
        _sample(float(index), value) for index, value in enumerate(utilization)
    ]
    p10_window = window_statistics(p10_samples, start=0.0, end=11.0)
    p10_checks = gate_checks(p10_window, dict(REQUIRED_HARDWARE_PLAN))
    assert p10_window["mean_retained_all_window_gpu_utilization_percent"] == 96.0
    assert p10_window["p10_retained_all_window_gpu_utilization_percent"] == 80.0
    assert p10_checks["minimum_mean_retained_utilization"] is True
    assert p10_checks["strict_p10_retained_utilization"] is False

    memory_samples = [
        _sample(float(index), memory_fraction=0.8) for index in range(12)
    ]
    memory_window = window_statistics(memory_samples, start=0.0, end=11.0)
    memory_checks = gate_checks(memory_window, dict(REQUIRED_HARDWARE_PLAN))
    assert memory_window["peak_memory_fraction"] == pytest.approx(0.8)
    assert memory_checks["strict_peak_memory_fraction"] is False


def test_cadence_gap_and_marker_edge_gates() -> None:
    clean = [_sample(float(index)) for index in range(12)]
    assert all(
        gate_checks(
            window_statistics(clean, start=0.0, end=11.0),
            dict(REQUIRED_HARDWARE_PLAN),
        ).values()
    )

    gapped_times = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]
    gapped = [_sample(float(epoch)) for epoch in gapped_times]
    gapped_checks = gate_checks(
        window_statistics(gapped, start=0.0, end=13.0),
        dict(REQUIRED_HARDWARE_PLAN),
    )
    assert gapped_checks["maximum_sample_gap"] is False

    uncovered = [_sample(float(epoch)) for epoch in range(3, 15)]
    uncovered_checks = gate_checks(
        window_statistics(uncovered, start=0.0, end=14.0),
        dict(REQUIRED_HARDWARE_PLAN),
    )
    assert uncovered_checks["leading_marker_edge_coverage"] is False


def test_raw_trace_requires_one_gpu_and_strict_timestamps(tmp_path: Path) -> None:
    first = "2026/07/21 12:00:00.000"
    second = "2026/07/21 12:00:01.000"
    two_gpu = tmp_path / "two_gpu.csv"
    two_gpu.write_text(
        f"{first}, GPU-a, NVIDIA A100-SXM4-80GB, 95, 100, 81920\n"
        f"{second}, GPU-b, NVIDIA A100-SXM4-80GB, 95, 100, 81920\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one valid GPU UUID"):
        read_samples(two_gpu)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        f"{first}, GPU-a, NVIDIA A100-SXM4-80GB, 95, 100, 81920\n"
        f"{first}, GPU-a, NVIDIA A100-SXM4-80GB, 95, 100, 81920\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        read_samples(duplicate)


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _guard_fixture(tmp_path: Path, job_id: str = "424242") -> dict[str, Path | str]:
    root = tmp_path / "scientific"
    smoke_root = tmp_path / "smoke"
    root.mkdir(parents=True)
    smoke_root.mkdir()
    card = tmp_path / "prediction_card.json"
    parent_card = tmp_path / "parent_prediction_card.json"
    source = tmp_path / "source_manifest.sha256"
    source.write_text("frozen-source-manifest\n", encoding="utf-8")
    checkpoint_roster = [
        {
            "arm": arm,
            "seed": seed,
            "checkpoint_step": 20_000,
            "path": f"/frozen/{arm}/seed_{seed}/checkpoint.pt",
            "sha256": f"{arm[0]}{seed:02d}".ljust(64, "0"),
        }
        for arm in ("dense", "sparse")
        for seed in range(64, 74)
    ]
    _json(parent_card, {"checkpoint_roster": {"runs": checkpoint_roster}})
    _json(
        card,
        {
            "protocol_id": "allen_cahn_periodic_reencoding_test",
            "output_root": str(root),
            "hardware_plan": dict(REQUIRED_HARDWARE_PLAN),
            "outcome_free_smoke": {"output_root": str(smoke_root)},
            "prospective_datasets": {
                "validation": [
                    {"index": index, "seed": seed}
                    for index, seed in enumerate((101, 102, 103))
                ],
                "test": [
                    {"index": index, "seed": seed}
                    for index, seed in enumerate((201, 202, 203))
                ],
            },
            "system": {
                "trajectories_per_dataset": 256,
                "validation_horizon_steps": 200,
                "test_horizon_steps": 400,
                "grid_size": 16,
                "channels": 2,
            },
            "frozen_parent": {
                "checkpoint_card": str(parent_card),
                "checkpoint_card_sha256": sha256_path(parent_card),
            },
        },
    )
    card_hash = sha256_path(card)
    source_hash = sha256_path(source)
    smoke_receipt = smoke_root / "smoke_receipt.json"
    _json(
        smoke_receipt,
        {
            "status": "passed_outcome_free_gpu_smoke",
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "slurm_job_id": "313131",
            "scientific_outcomes_accessed": False,
        },
    )
    artifacts = {
        "selection_decision": root / "selection_decision.json",
        "validation_data_manifest": root / "validation_data_manifest.json",
        "test_data_manifest": root / "test_data_manifest.json",
        "scientific_payload": root / "scientific_payload.json",
    }
    for name in ("selection_decision", "scientific_payload"):
        # Invalid JSON proves telemetry only hashes these outcome-bearing artifacts.
        artifacts[name].write_bytes(f"opaque-{name}".encode())
    field_paths: list[Path] = []
    for role, seeds, horizon in (
        ("validation", (101, 102, 103), 200),
        ("test", (201, 202, 203), 400),
    ):
        rows = []
        shape = [256, horizon + 1, 16, 16, 2]
        for index, seed in enumerate(seeds):
            field_path = root / "data" / f"{role}_seed{seed}_fields.pt"
            field_path.parent.mkdir(exist_ok=True)
            field_path.write_bytes(f"opaque-{role}-{index}".encode())
            field_paths.append(field_path)
            rows.append(
                {
                    "role": role,
                    "dataset_index": index,
                    "dataset_seed": seed,
                    "path": str(field_path),
                    "sha256": sha256_path(field_path),
                    "shape": shape,
                    "storage_bytes": 4 * math.prod(shape),
                }
            )
        _json(
            artifacts[f"{role}_data_manifest"],
            {
                "schema_version": 1,
                "protocol_id": "allen_cahn_periodic_reencoding_test",
                "role": role,
                "datasets": rows,
            },
        )
    artifact_hashes = {name: sha256_path(path) for name, path in artifacts.items()}
    roster_encoded = json.dumps(
        checkpoint_roster, sort_keys=True, separators=(",", ":")
    ).encode()
    roster_hash = hashlib.sha256(roster_encoded).hexdigest()
    _json(
        root / "runtime_lineage.json",
        {
            "status": "scientific_payload_written_but_not_authorized_for_summary",
            "card_sha256": card_hash,
            "source_manifest_sha256": source_hash,
            "slurm_job_id": job_id,
            "scientific_metrics_printed": False,
            "smoke_receipt_path": str(smoke_receipt),
            "smoke_receipt_sha256": sha256_path(smoke_receipt),
            **{
                f"{name}_path": str(path) for name, path in artifacts.items()
            },
            **{
                f"{name}_sha256": artifact_hashes[name] for name in artifacts
            },
            "scientific_hash": artifact_hashes["scientific_payload"],
            "checkpoint_roster": checkpoint_roster,
            "checkpoint_roster_sha256": roster_hash,
            "environment": {
                "slurm_job_id": job_id,
                "gpu_name": "NVIDIA A100-SXM4-80GB",
                "gpu_uuid": "GPU-runtime-format",
            },
        },
    )
    base = datetime.strptime("2026/07/21 12:00:00", "%Y/%m/%d %H:%M:%S").timestamp()
    common = {
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "slurm_job_id": job_id,
    }
    _json(
        root / "markers/selection_start.json",
        {**common, "stage": "selection_start", "epoch_seconds": base},
    )
    _json(
        root / "markers/selection_end.json",
        {**common, "stage": "selection_end", "epoch_seconds": base + 11.0},
    )
    _json(
        root / "markers/evaluation_start.json",
        {**common, "stage": "evaluation_start", "epoch_seconds": base + 20.0},
    )
    _json(
        root / "markers/evaluation_compute_end.json",
        {**common, "stage": "evaluation_compute_end", "epoch_seconds": base + 31.0},
    )
    _json(
        root / "markers/evaluation_end.json",
        {
            **common,
            "stage": "evaluation_end",
            "epoch_seconds": base + 31.0,
            "scientific_payload_sha256": artifact_hashes["scientific_payload"],
        },
    )
    raw = tmp_path / "raw.csv"
    rows = []
    for index in range(32):
        stamp = datetime.fromtimestamp(base + index).strftime("%Y/%m/%d %H:%M:%S.%f")
        rows.append(
            f"{stamp}, GPU-periodic, NVIDIA A100-SXM4-80GB, 100, 20000, 81920"
        )
    raw.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "root": root,
        "card": card,
        "card_hash": card_hash,
        "source": source,
        "source_hash": source_hash,
        "scientific": artifacts["scientific_payload"],
        "scientific_hash": artifact_hashes["scientific_payload"],
        "validation_manifest": artifacts["validation_data_manifest"],
        "field_file": field_paths[0],
        "smoke_receipt": smoke_receipt,
        "raw": raw,
    }


def _audit(fixture: dict[str, Path | str]) -> dict[str, object]:
    return audit_and_issue_guard(
        root=fixture["root"], card_path=fixture["card"],
        expected_card_sha256=fixture["card_hash"], source_manifest=fixture["source"],
        expected_source_manifest_sha256=fixture["source_hash"],
        raw_telemetry=fixture["raw"],
    )


def test_guard_binds_job_and_opaque_scientific_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _guard_fixture(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    result = _audit(fixture)
    guard = json.loads(Path(result["outcome_guard_receipt_path"]).read_text())
    assert guard["status"] == "authorized_for_dependent_cpu_summary"
    assert guard["slurm_job_id"] == "424242"
    assert guard["scientific_payload_sha256"] == fixture["scientific_hash"]
    assert guard["validation_data_manifest_sha256"] == sha256_path(
        fixture["validation_manifest"]
    )
    assert len(guard["checkpoint_roster"]) == 20
    assert len(guard["checkpoint_roster_sha256"]) == 64
    assert set(guard["marker_bindings"]) == {
        "selection_start", "selection_end", "evaluation_start", "evaluation_end",
        "evaluation_compute_end",
    }
    assert set(guard["validity_checks"]) == {
        "selection_validity", "evaluation_validity"
    }
    assert all(
        all(checks.values()) for checks in guard["validity_checks"].values()
    )
    assert guard["smoke_receipt_status"] == "passed_outcome_free_gpu_smoke"
    assert guard["field_manifest_bindings"]["validation"]["row_count"] == 3
    assert guard["field_manifest_bindings"]["test"]["row_count"] == 3
    assert guard["gpu_identity_binding"]["uuid_equality_required"] is False
    assert guard["disjoint_window_ordering_verified"] is True
    assert guard["evaluation_end_epoch_identity_verified"] is True
    assert guard["whole_allocation_descriptive"][
        "mean_gpu_utilization_percent_descriptive"
    ] == 100.0
    assert guard["scientific_payload_opened"] is False
    assert guard["forecast_outcomes_accessed"] is False


def test_guard_rejects_slurm_or_scientific_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_job = _guard_fixture(tmp_path / "job")
    monkeypatch.setenv("SLURM_JOB_ID", "999999")
    with pytest.raises(RuntimeError, match="runtime lineage"):
        _audit(wrong_job)

    hash_drift = _guard_fixture(tmp_path / "hash")
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    Path(hash_drift["scientific"]).write_bytes(b"changed-after-runtime-lineage")
    with pytest.raises(RuntimeError, match="scientific_payload binding failed"):
        _audit(hash_drift)

    marker_drift = _guard_fixture(tmp_path / "marker")
    marker_path = Path(marker_drift["root"]) / "markers/evaluation_compute_end.json"
    marker = json.loads(marker_path.read_text())
    marker["epoch_seconds"] += 0.001
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="compute-end and sealed end epochs differ"):
        _audit(marker_drift)


def test_guard_rejects_manifest_and_checkpoint_roster_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_drift = _guard_fixture(tmp_path / "manifest")
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    Path(manifest_drift["validation_manifest"]).write_bytes(b"changed-manifest")
    with pytest.raises(RuntimeError, match="validation_data_manifest binding failed"):
        audit_and_issue_guard(
            root=manifest_drift["root"],
            card_path=manifest_drift["card"],
            expected_card_sha256=manifest_drift["card_hash"],
            source_manifest=manifest_drift["source"],
            expected_source_manifest_sha256=manifest_drift["source_hash"],
            raw_telemetry=manifest_drift["raw"],
        )

    roster_drift = _guard_fixture(tmp_path / "roster")
    runtime_path = Path(roster_drift["root"]) / "runtime_lineage.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["checkpoint_roster_sha256"] = "0" * 64
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Checkpoint roster"):
        audit_and_issue_guard(
            root=roster_drift["root"],
            card_path=roster_drift["card"],
            expected_card_sha256=roster_drift["card_hash"],
            source_manifest=roster_drift["source"],
            expected_source_manifest_sha256=roster_drift["source_hash"],
            raw_telemetry=roster_drift["raw"],
        )


def test_both_disjoint_windows_must_independently_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _guard_fixture(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    raw_path = Path(fixture["raw"])
    lines = raw_path.read_text().splitlines()
    for index in range(1, 11):
        fields = [value.strip() for value in lines[index].split(",")]
        fields[3] = "0"
        lines[index] = ", ".join(fields)
    raw_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="disjoint GPU-utilization window failed"):
        audit_and_issue_guard(
            root=fixture["root"],
            card_path=fixture["card"],
            expected_card_sha256=fixture["card_hash"],
            source_manifest=fixture["source"],
            expected_source_manifest_sha256=fixture["source_hash"],
            raw_telemetry=raw_path,
        )
    audit = json.loads((Path(fixture["root"]) / "telemetry_audit.json").read_text())
    assert not all(audit["validity_checks"]["selection_validity"].values())
    assert all(audit["validity_checks"]["evaluation_validity"].values())
    assert not (Path(fixture["root"]) / "outcome_guard_receipt.json").exists()


def test_guard_rejects_smoke_status_and_field_file_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "424242")
    smoke_drift = _guard_fixture(tmp_path / "smoke")
    smoke_path = Path(smoke_drift["smoke_receipt"])
    smoke = json.loads(smoke_path.read_text())
    smoke["status"] = "failed"
    smoke_path.write_text(json.dumps(smoke))
    runtime_path = Path(smoke_drift["root"]) / "runtime_lineage.json"
    runtime = json.loads(runtime_path.read_text())
    runtime["smoke_receipt_sha256"] = sha256_path(smoke_path)
    runtime_path.write_text(json.dumps(runtime))
    with pytest.raises(RuntimeError, match="Smoke receipt status"):
        audit_and_issue_guard(
            root=smoke_drift["root"],
            card_path=smoke_drift["card"],
            expected_card_sha256=smoke_drift["card_hash"],
            source_manifest=smoke_drift["source"],
            expected_source_manifest_sha256=smoke_drift["source_hash"],
            raw_telemetry=smoke_drift["raw"],
        )

    field_drift = _guard_fixture(tmp_path / "field")
    Path(field_drift["field_file"]).write_bytes(b"changed-field-file")
    with pytest.raises(RuntimeError, match="field file 0 hash binding failed"):
        audit_and_issue_guard(
            root=field_drift["root"],
            card_path=field_drift["card"],
            expected_card_sha256=field_drift["card_hash"],
            source_manifest=field_drift["source"],
            expected_source_manifest_sha256=field_drift["source_hash"],
            raw_telemetry=field_drift["raw"],
        )
