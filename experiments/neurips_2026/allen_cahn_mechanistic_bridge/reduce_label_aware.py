"""Open labels only after field artifacts freeze; score and reduce the bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.aggregation import (
    aggregate,
    decide,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.conditional_guard import (
    load_and_validate,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.families import (
    alignment_metrics,
    modal_accuracy,
    modal_well_fates,
    truth_difficulty,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.integrity import (
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import (
    CARD_PATH,
    finite_tree,
    load_card,
    load_training_fates,
    sha256_path,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.probes import (
    fit_nested_ridge,
    score_fitted,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field_manifest", type=Path, required=True)
    parser.add_argument("--expected_field_manifest_sha256", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected_decision_sha256", required=True)
    parser.add_argument("--expected_dataset_manifest_sha256", required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--expected_profile_decision_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    return parser.parse_args()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_artifacts(
    manifest_path: Path,
    *,
    expected_hash: str,
    card_hash: str,
    source_hash: str,
    dataset_manifest_hash: str,
    decision_hash: str,
    profile_hash: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    if sha256_path(manifest_path) != expected_hash:
        raise RuntimeError("Field-only manifest differs from launcher root")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "field_only_roster_frozen_before_label_key_access"
        or manifest.get("dataset_payloads_deserialized") is not True
        or manifest.get("label_tensors_may_have_been_deserialized") is not True
        or manifest.get("label_keys_accessed") is not False
        or manifest.get("label_values_used") is not False
        or manifest.get("card_sha256") != card_hash
        or manifest.get("source_manifest_sha256") != source_hash
        or manifest.get("dataset_manifest_sha256") != dataset_manifest_hash
        or manifest.get("mechanism_decision_sha256") != decision_hash
        or manifest.get("profile_decision_sha256") != profile_hash
    ):
        raise RuntimeError("Field-only manifest root checks failed")
    result = {}
    for record in manifest["artifacts"]:
        path = Path(record["artifact"])
        if sha256_path(path) != record["artifact_sha256"]:
            raise RuntimeError(f"Field artifact hash mismatch: {path}")
        artifact = _torch_load(path)
        if not finite_tree(artifact):
            raise FloatingPointError(f"Field artifact contains nonfinite values: {path}")
        key = (int(artifact["model_seed"]), int(artifact["dataset_seed"]))
        if key in result:
            raise RuntimeError("Duplicate crossed field artifact")
        telemetry_scope = artifact.get("gpu_telemetry_scope", {})
        checks = (
            artifact["status"] == "field_only_complete",
            artifact["route_lock"]["locked"],
            not artifact["route_lock"]["future_states_used"],
            artifact["route_lock"]["future_field_tensors_deserialized_before_route_lock"]
            is True,
            artifact["route_lock"]["future_values_used_for_routing"] is False,
            artifact["route_lock"][
                "x0_probe_features_materialized_before_future_encoding"
            ]
            is True,
            artifact["route_lock"]["label_tensors_may_have_been_deserialized"] is True,
            artifact["route_lock"]["label_keys_accessed"] is False,
            artifact["route_lock"]["label_values_used"] is False,
            artifact["dataset_payloads_deserialized"] is True,
            artifact["label_tensors_may_have_been_deserialized"] is True,
            artifact["label_keys_accessed"] is False,
            artifact["label_values_used"] is False,
            artifact["requested_dataset_keys"] == ["fields", "split_indices"],
            artifact["card_sha256"] == card_hash,
            artifact["source_manifest_sha256"] == source_hash,
            artifact["dataset_manifest_sha256"] == dataset_manifest_hash,
            artifact["mechanism_decision_sha256"] == decision_hash,
            telemetry_scope.get("evaluator_owned_start_marker")
            == record.get("gpu_start_marker"),
            telemetry_scope.get("evaluator_owned_done_marker")
            == record.get("gpu_done_marker"),
            telemetry_scope.get("start_marker_sha256")
            == record.get("gpu_start_marker_sha256"),
            telemetry_scope.get("done_marker_sha256")
            == record.get("gpu_done_marker_sha256"),
            telemetry_scope.get("preload_and_serialization_excluded") is True,
        )
        if not all(checks):
            raise RuntimeError(f"Field artifact firewall checks failed: {path}")
        result[key] = artifact
    return result


def _mean(record: dict[str, Any], metric: str) -> float:
    values = record[metric]
    if not isinstance(values, torch.Tensor) or values.numel() == 0:
        raise ValueError(f"Missing per-trajectory metric {metric}")
    return float(values.double().mean())


def _forecast_record(record: dict[str, Any], truth: torch.Tensor) -> dict[str, float]:
    return {
        "through_mse": _mean(record, "through_mse"),
        "terminal_mse": _mean(record, "terminal_mse"),
        "modal_fate_accuracy": modal_accuracy(record["final_prediction"], truth),
        "finite_fraction": float(record["finite"].double().mean()),
    }


def _cell_row(
    artifact: dict[str, Any], probe_scores: dict[str, Any]
) -> dict[str, Any]:
    truth = artifact["truth_fields"]
    fates = {
        horizon: modal_well_fates(truth[str(horizon)]).numpy()
        for horizon in (200, 400)
    }
    alignment = {
        arm: {
            str(horizon): alignment_metrics(
                artifact["codebooks"][arm]["new_assignments"].numpy(), fates[horizon]
            )
            for horizon in (200, 400)
        }
        for arm in ("sparse", "dense")
    }
    ordinary = {}
    for arm in ("sparse", "dense"):
        ordinary[arm] = {}
        for horizon in (160, 200, 400):
            record = artifact["ordinary_forecast"][arm][str(horizon)]
            ordinary[arm][str(horizon)] = _forecast_record(
                record, truth[str(horizon)]
            )
    routed = artifact["routed_forecast"]
    covered_indices = routed["trajectory_indices"].long()
    route_summary: dict[str, Any] = {
        "covered_count": int(covered_indices.numel()),
        "coverage": float(covered_indices.numel() / truth["0"].shape[0]),
        "wrong_control_count": int(
            routed["wrong_control"]["trajectory_indices"].numel()
        ),
        "same_subset_for_all_modes": routed["wrong_control"][
            "same_subset_for_all_modes"
        ] is True,
        "paired_cardinality_exact": routed["wrong_control"][
            "paired_cardinality_exact"
        ] is True,
    }
    route_summary["all_covered_horizons"] = {}
    if route_summary["covered_count"] > 0:
        for horizon in (160, 200, 400):
            modes = routed["correct"]
            full_reference = artifact["ordinary_forecast"]["sparse"][str(horizon)][
                "through_mse"
            ][covered_indices]
            routed_full = modes["full"][str(horizon)]["through_mse"]
            if not torch.allclose(
                routed_full.double(), full_reference.double(), rtol=1e-5, atol=1e-8
            ):
                raise RuntimeError("Routed full rollout failed the ordinary-rollout cross-check")
            route_summary["all_covered_horizons"][str(horizon)] = {
                mode: {
                    "through_mse": _mean(modes[mode][str(horizon)], "through_mse"),
                    "terminal_mse": _mean(modes[mode][str(horizon)], "terminal_mse"),
                    "modal_fate_accuracy": modal_accuracy(
                        modes[mode][str(horizon)]["final_prediction"],
                        truth[str(horizon)][covered_indices],
                    ),
                }
                for mode in ("full", "mask_once", "restricted")
            }
    if route_summary["wrong_control_count"] > 0:
        control = routed["wrong_control"]
        indices = control["trajectory_indices"].long()
        route_summary["wrong_control_coverage"] = float(
            indices.numel() / truth["0"].shape[0]
        )
        initial = control["initial_projection"]
        cardinality_equal = torch.equal(
            initial["correct_cardinality"], initial["wrong_cardinality"]
        )
        distinct = bool(torch.all(initial["jaccard"] < 1.0))
        if not cardinality_equal or not distinct:
            raise RuntimeError("Wrong-support cardinality/distinctness control failed")
        route_summary["initial_projection"] = {
            "correct_capture_fraction": float(
                initial["correct_capture_fraction"].double().mean()
            ),
            "wrong_capture_fraction": float(
                initial["wrong_capture_fraction"].double().mean()
            ),
            "correct_reconstruction_mse": float(
                initial["correct_reconstruction_mse"].double().mean()
            ),
            "wrong_reconstruction_mse": float(
                initial["wrong_reconstruction_mse"].double().mean()
            ),
            "correct_cardinality_mean": float(
                initial["correct_cardinality"].double().mean()
            ),
            "wrong_cardinality_mean": float(
                initial["wrong_cardinality"].double().mean()
            ),
            "support_jaccard_mean": float(initial["jaccard"].double().mean()),
            "paired_cardinality_exact": True,
            "all_supports_distinct": True,
            "mask_once_and_restricted_share_identical_t0_state": True,
        }
        route_summary["horizons"] = {}
        for horizon in (160, 200, 400):
            sparse_full = artifact["ordinary_forecast"]["sparse"][str(horizon)]
            full_subset = {
                key: value[indices] if isinstance(value, torch.Tensor) else value
                for key, value in sparse_full.items()
            }
            horizon_record: dict[str, Any] = {
                "full": _forecast_record(full_subset, truth[str(horizon)][indices])
            }
            for route in ("correct", "wrong"):
                horizon_record[route] = {
                    mode: _forecast_record(
                        control[route][mode][str(horizon)],
                        truth[str(horizon)][indices],
                    )
                    for mode in ("mask_once", "restricted")
                }
            route_summary["horizons"][str(horizon)] = horizon_record
    else:
        route_summary["wrong_control_coverage"] = 0.0
        route_summary["initial_projection"] = {}
        route_summary["horizons"] = {}
    stability = {
        arm: {
            str(time): float(
                artifact["support_stability"][str(time)][arm]["jaccard_to_x0"]
                .double()
                .mean()
            )
            for time in (0, 40, 80, 120, 160, 200, 400)
        }
        for arm in ("sparse", "dense")
    }
    return {
        "model_seed": int(artifact["model_seed"]),
        "dataset_seed": int(artifact["dataset_seed"]),
        "ordinary": ordinary,
        "alignment": alignment,
        "probes": probe_scores,
        "routing": route_summary,
        "support_stability_mean_jaccard": stability,
        "truth_difficulty": truth_difficulty(truth["0"], truth["200"], truth["400"]),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Bridge card differs from launcher root")
    source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Bridge source manifest differs from launcher root")
    _, decision_hash, _ = load_and_validate(
        args.decision, expected_sha256=args.expected_decision_sha256, card=card
    )
    artifacts = _load_artifacts(
        args.field_manifest,
        expected_hash=args.expected_field_manifest_sha256,
        card_hash=card_hash,
        source_hash=source_hash,
        dataset_manifest_hash=args.expected_dataset_manifest_sha256,
        decision_hash=decision_hash,
        profile_hash=args.expected_profile_decision_sha256,
    )
    expected = {
        (int(model), int(dataset))
        for model in card["roster"]["model_seeds"]
        for dataset in card["new_datasets"]["seeds"]
    }
    if set(artifacts) != expected:
        raise RuntimeError("Field artifact crossed roster is incomplete")
    train_labels = load_training_fates(card).numpy()
    probe_cfg = card["probes"]
    fitted, audits = {}, {}
    first_dataset = int(card["new_datasets"]["seeds"][0])
    for model_seed in card["roster"]["model_seeds"]:
        reference = artifacts[(int(model_seed), first_dataset)]["features"]["train"]
        for dataset_seed in card["new_datasets"]["seeds"][1:]:
            candidate = artifacts[(int(model_seed), int(dataset_seed))]["features"]["train"]
            if any(not torch.equal(reference[name], candidate[name]) for name in probe_cfg["feature_sets"]):
                raise RuntimeError("Training features drift across dataset tasks")
        for feature_name in probe_cfg["feature_sets"]:
            model, audit = fit_nested_ridge(
                reference[feature_name].numpy(),
                train_labels,
                alphas=probe_cfg["alpha_grid"],
                outer_folds=int(probe_cfg["outer_folds"]),
                inner_folds=int(probe_cfg["inner_folds"]),
                final_folds=int(probe_cfg["final_selection_folds"]),
                seed=int(probe_cfg["split_seed"]),
            )
            fitted[(int(model_seed), feature_name)] = model
            audits[(int(model_seed), feature_name)] = audit
    rows = []
    for key in sorted(artifacts):
        artifact = artifacts[key]
        new_fates = modal_well_fates(artifact["truth_fields"]["200"]).numpy()
        scores = {}
        for feature_name in probe_cfg["feature_sets"]:
            scores[feature_name] = {
                **audits[(key[0], feature_name)],
                **score_fitted(
                    fitted[(key[0], feature_name)],
                    artifact["features"]["new"][feature_name].numpy(),
                    new_fates,
                ),
            }
        rows.append(_cell_row(artifact, scores))
    aggregate_result = aggregate(rows, card)
    decision = decide(aggregate_result, card)
    valid = finite_tree(rows) and finite_tree(aggregate_result)
    payload = {
        "schema_version": 1,
        "status": "complete" if valid else "invalid",
        "decision": decision,
        "aggregate": aggregate_result,
        "cells": rows,
        "h400_is_secondary_only": True,
        "original_four_cell_confirmation_gate_failure_preserved": True,
        "label_access_audit": {
            "field_stage_label_tensors_may_have_been_deserialized": True,
            "field_stage_label_keys_accessed": False,
            "field_stage_label_values_used": False,
            "reduction_stage_training_label_key_accessed": True,
            "new_dataset_fates_derived_post_hoc_from_frozen_truth_fields": True,
            "labels_used_for_kae_training_or_routing": False,
        },
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "mechanism_decision_sha256": decision_hash,
        "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
        "field_manifest_sha256": args.expected_field_manifest_sha256,
        "profile_decision_sha256": args.expected_profile_decision_sha256,
    }
    if payload["status"] == "invalid":
        payload["decision"] = {
            "branch": "invalid",
            "interpretation": card["decision_branches"]["invalid"],
        }
    args.output_dir.mkdir(parents=True)
    write_json_once(args.output_dir / "decision.json", payload)
    write_json_once(args.output_dir / "provenance.json", {
        "decision_sha256": sha256_path(args.output_dir / "decision.json"),
        "field_manifest_sha256": args.expected_field_manifest_sha256,
        "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "profile_decision_sha256": args.expected_profile_decision_sha256,
    })
    print(json.dumps({"status": payload["status"], "branch": payload["decision"]["branch"]}))


if __name__ == "__main__":
    main()
