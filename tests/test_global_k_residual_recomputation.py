from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.neurips_2026.global_k_residual_forecast import evaluate, protocol
from experiments.neurips_2026.global_k_residual_forecast.diagnostic_recompute import (
    recompute_route_diagnostics,
    summarize_route_validity,
    validate_count_weighted_basin_h200,
)
from experiments.neurips_2026.global_k_residual_forecast.telemetry import _assess_one
from experiments.neurips_2026.global_k_residual_forecast.validation import validate_gate


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "neurips_2026" / "global_k_residual_forecast"


def _card() -> dict:
    return json.loads((PACKAGE / "prediction_card.json").read_text())


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def _fake_v2_bundle(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    paths = {
        name: tmp_path / name
        for name in (
            "representative.json", "dense.json", "task.tsv", "audit.json",
            "packet.json", "full_manifest.json", "smoke_manifest.json",
            "smoke_tasks.tsv",
        )
    }
    hashes = {
        name: _write(path, b"{}" if path.suffix == ".json" else b"row\n")
        for name, path in paths.items()
    }
    v2_card_path = tmp_path / "v2_card.json"
    v2_card = {
        "protocol_id": "test-v2",
        "training_arms": {
            "sparse": {
                "representative_frozen_config": str(paths["representative.json"]),
                "representative_frozen_config_sha256": hashes[
                    "representative.json"
                ],
            },
            "dense": {
                "source_recipe_card": str(paths["dense.json"]),
                "source_recipe_card_sha256": hashes["dense.json"],
            },
        },
    }
    v2_card_bytes = json.dumps(v2_card, sort_keys=True).encode()
    v2_card_hash = _write(v2_card_path, v2_card_bytes)
    source_lock_path = tmp_path / "source_lock.json"
    source_lock = {
        "schema_version": 1,
        "protocol_id": "test-v2",
        "card_sha256": v2_card_hash,
        "external_inputs": {
            "full_manifest": {
                "path": str(paths["full_manifest.json"]),
                "sha256": hashes["full_manifest.json"],
            },
            "full_task_tsv": {
                "path": str(paths["task.tsv"]),
                "sha256": hashes["task.tsv"],
            },
            "smoke_manifest": {
                "path": str(paths["smoke_manifest.json"]),
                "sha256": hashes["smoke_manifest.json"],
            },
            "smoke_task_tsv": {
                "path": str(paths["smoke_tasks.tsv"]),
                "sha256": hashes["smoke_tasks.tsv"],
            },
        },
    }
    source_lock_hash = _write(
        source_lock_path, json.dumps(source_lock, sort_keys=True).encode()
    )
    residual = {
        "authenticated_v2_inputs": {
            "card_path": str(v2_card_path),
            "card_sha256": v2_card_hash,
            "task_path": str(paths["task.tsv"]),
            "task_sha256": hashes["task.tsv"],
            "source_lock_path": str(source_lock_path),
            "source_lock_sha256": source_lock_hash,
            "audit_summary_path": str(paths["audit.json"]),
            "audit_summary_sha256": hashes["audit.json"],
            "packet_path": str(paths["packet.json"]),
            "packet_sha256": hashes["packet.json"],
        }
    }
    paths.update({"card": v2_card_path, "source_lock": source_lock_path})
    return residual, paths


def test_frozen_v2_bundle_roots_authenticate_without_parsing_outcomes() -> None:
    bundle = protocol.authenticate_v2_inputs(_card())
    assert bundle["card"]["protocol_id"] == (
        "global_k_distinct_laws_gated_local_linear_v2_new_seeds"
    )
    assert protocol.sha256_bytes(bundle["representative_config_bytes"]) == (
        bundle["card"]["training_arms"]["sparse"]
        ["representative_frozen_config_sha256"]
    )


def test_v2_bundle_authenticates_every_runtime_input_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    residual, _ = _fake_v2_bundle(tmp_path)
    bundle = protocol.authenticate_v2_inputs(residual)
    assert bundle["card"]["protocol_id"] == "test-v2"
    assert bundle["representative_config_bytes"] == b"{}"

    events = []

    def stop_before_gpu(_card: dict) -> dict:
        events.append("v2_authenticated")
        raise RuntimeError("stop before GPU")

    monkeypatch.setattr(evaluate, "authenticate_v2_inputs", stop_before_gpu)
    with pytest.raises(RuntimeError, match="stop before GPU"):
        evaluate._evaluate(
            mode="smoke", task_index=0, card={}, tasks={}, freeze={},
            output_root=tmp_path, compute_window_path=tmp_path / "window.json",
        )
    assert events == ["v2_authenticated"]


@pytest.mark.parametrize(
    "name",
    [
        "card", "task.tsv", "source_lock", "audit.json", "packet.json",
        "representative.json", "dense.json", "full_manifest.json",
        "smoke_manifest.json", "smoke_tasks.tsv",
    ],
)
def test_v2_bundle_rejects_each_tampered_external_input(
    tmp_path: Path, name: str,
) -> None:
    residual, paths = _fake_v2_bundle(tmp_path)
    paths[name].write_bytes(paths[name].read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        protocol.authenticate_v2_inputs(residual)


def test_v2_source_lock_must_bind_declared_task_pair(tmp_path: Path) -> None:
    residual, paths = _fake_v2_bundle(tmp_path)
    replacement = tmp_path / "replacement.tsv"
    residual["authenticated_v2_inputs"]["task_path"] = str(replacement)
    residual["authenticated_v2_inputs"]["task_sha256"] = _write(
        replacement, b"different but authenticated\n"
    )
    with pytest.raises(RuntimeError, match="full task TSV"):
        protocol.authenticate_v2_inputs(residual)


def _valid_diagnostics() -> dict:
    counts = [44000, 44000, 43072]
    return {
        "model_seed": 100,
        "label_free_family_fit": {
            "retained_family_count": 3,
            "retained_fit_coverage": 0.95,
            "support_cardinalities": [4, 5, 6],
            "sign_pair_exclusivity": True,
            "fallback_used": False,
            "maximum_family_truncation_used": False,
            "fit_valid": True,
        },
        "held_out_route_audit": {
            "assignment_count_by_family": counts,
            "mean_nearest_jaccard": 0.9,
            "confident_assignment_fraction": 0.99,
            "active_family_count_at_minimum_fraction": 3,
            "label_free_route_audit_valid": True,
            "evaluation_only_alignment": {
                "contingency_family_by_label": [
                    [44000, 0, 0], [0, 44000, 0], [0, 0, 43072]
                ],
                "family_conditional_basin_purity": 1.0,
            },
        },
        "matched_coordinate_null": {
            "eligible_candidate_count": 32,
            "selected_scale_rows": [
                {
                    "candidate_index": index,
                    "score": 0.0,
                    "source_rms_ratio": 1.0,
                    "update_rms_ratio": 1.0,
                    "eligible": True,
                }
                for index in range(32)
            ],
            "scale_match_valid": True,
        },
    }


def test_route_decisions_are_recomputed_not_trusted() -> None:
    card = _card()
    valid = _valid_diagnostics()
    assert all(recompute_route_diagnostics(valid, card).values())
    cases = []
    fit = copy.deepcopy(valid)
    fit["label_free_family_fit"]["retained_fit_coverage"] = 0.1
    cases.append((fit, "fit_valid"))
    audit = copy.deepcopy(valid)
    audit["held_out_route_audit"]["mean_nearest_jaccard"] = 0.1
    cases.append((audit, "route_audit_valid"))
    active = copy.deepcopy(valid)
    active["held_out_route_audit"]["active_family_count_at_minimum_fraction"] = 2
    cases.append((active, "active-family"))
    null = copy.deepcopy(valid)
    null["matched_coordinate_null"]["eligible_candidate_count"] = 31
    cases.append((null, "scale_match_valid"))
    null_ratio = copy.deepcopy(valid)
    null_ratio["matched_coordinate_null"]["selected_scale_rows"][0][
        "source_rms_ratio"
    ] = 3.0
    cases.append((null_ratio, "null score"))
    for forged, message in cases:
        with pytest.raises(RuntimeError, match=message):
            recompute_route_diagnostics(forged, card)


def test_contingency_rows_and_route_purity_are_recomputed() -> None:
    card = _card()
    valid = _valid_diagnostics()
    wrong_rows = copy.deepcopy(valid)
    wrong_rows["held_out_route_audit"]["evaluation_only_alignment"][
        "contingency_family_by_label"
    ][0][0] -= 1
    with pytest.raises(RuntimeError, match="row sums"):
        recompute_route_diagnostics(wrong_rows, card)
    wrong_purity = copy.deepcopy(valid)
    wrong_purity["held_out_route_audit"]["evaluation_only_alignment"][
        "family_conditional_basin_purity"
    ] = 0.99
    with pytest.raises(RuntimeError, match="route purity"):
        recompute_route_diagnostics(wrong_purity, card)

    low = copy.deepcopy(valid)
    counts = low["held_out_route_audit"]["assignment_count_by_family"]
    contingency = [[count // 2, count - count // 2, 0] for count in counts]
    purity = sum(max(row) for row in contingency) / sum(counts)
    alignment = low["held_out_route_audit"]["evaluation_only_alignment"]
    alignment["contingency_family_by_label"] = contingency
    alignment["family_conditional_basin_purity"] = purity
    summary = summarize_route_validity([low] * 10, card)
    assert summary["passed"] is False
    assert summary["rows"][0]["basin_purity"] == purity


def test_count_weighted_basin_h200_must_match_global_metric() -> None:
    methods = {"method": {"through_h200_mse": 2.0}}
    basins = {
        "0": {"trajectory_count": 2, "through_h200_mse_by_method": {"method": 1.0}},
        "1": {"trajectory_count": 3, "through_h200_mse_by_method": {"method": 2.0}},
        "2": {"trajectory_count": 5, "through_h200_mse_by_method": {"method": 2.4}},
    }
    validate_count_weighted_basin_h200(methods, basins, 10)
    forged = copy.deepcopy(basins)
    forged["2"]["through_h200_mse_by_method"]["method"] = 2.5
    with pytest.raises(RuntimeError, match="count-weighted"):
        validate_count_weighted_basin_h200(methods, forged, 10)
    with pytest.raises(RuntimeError, match="suppresses global H200"):
        validate_count_weighted_basin_h200(
            {"method": {"through_h200_mse": None}}, basins, 10
        )


def _valid_telemetry_gate(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    output_root = tmp_path / "output"
    directory = output_root / "scientific"
    trace = directory / "telemetry" / "task_03.csv"
    window = directory / "compute_windows" / "task_03.json"
    trace.parent.mkdir(parents=True)
    window.parent.mkdir(parents=True)
    rows = [
        "epoch_seconds,gpu_uuid,gpu_name,utilization_gpu,memory_used_mib,memory_total_mib"
    ] + [
        f"{epoch},GPU-a,NVIDIA A100-SXM4-80GB,90,1000,81920"
        for epoch in range(99, 142)
    ]
    trace.write_text("\n".join(rows) + "\n")
    card = _card()
    marker = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": "forecast_compute_window",
        "mode": "scientific",
        "task_id": 3,
        "start_epoch_seconds": 100.0,
        "end_epoch_seconds": 140.0,
        "elapsed_seconds": 40.0,
    }
    window.write_text(json.dumps(marker))
    freeze = {"card_sha256": "a"}
    assessed = _assess_one(
        trace, window, card["gpu_utilization_gate"],
        protocol_id=card["protocol_id"], mode="scientific", task_id=3,
        freeze=freeze,
    )
    gate = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": "outcome_blind_scientific_gpu_assessment",
        "mode": "scientific",
        "freeze": freeze,
        "rows": [assessed],
        "smoke_checks": None,
        "passed": True,
        "forecast_outcomes_read": False,
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate))
    return gate_path, output_root, gate, card


@pytest.mark.parametrize(
    ("field", "forged"),
    [
        ("compute_window_mean_utilization_percent", 84.0),
        ("compute_window_p10_utilization_percent", 79.0),
        ("minimum_rolling_utilization_percent", 79.0),
        ("allocation_median_sampling_interval_seconds", 2.0),
        ("compute_median_sampling_interval_seconds", 2.0),
        ("allocation_maximum_sampling_gap_seconds", 3.0),
        ("compute_maximum_sampling_gap_seconds", 3.0),
        ("trace_to_compute_start_gap_seconds", 3.0),
        ("trace_after_compute_end_gap_seconds", 3.0),
        ("memory_total_mib", 70 * 1024.0),
        ("peak_memory_mib", 90000.0),
    ],
)
def test_summary_recomputes_telemetry_instead_of_trusting_true_checks(
    tmp_path: Path, field: str, forged: float,
) -> None:
    gate_path, output_root, gate, card = _valid_telemetry_gate(tmp_path)
    validate_gate(
        gate_path, mode="scientific", task_ids=[3],
        protocol_id=card["protocol_id"], freeze=gate["freeze"],
        output_root=output_root, thresholds=card["gpu_utilization_gate"],
    )
    gate["rows"][0][field] = forged
    gate_path.write_text(json.dumps(gate))
    with pytest.raises(RuntimeError):
        validate_gate(
            gate_path, mode="scientific", task_ids=[3],
            protocol_id=card["protocol_id"], freeze=gate["freeze"],
            output_root=output_root, thresholds=card["gpu_utilization_gate"],
        )
