"""Freeze the complete 10-by-3 field-only artifact roster before labels open."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.conditional_guard import (
    load_and_validate,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.integrity import (
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import (
    CARD_PATH,
    load_card,
    sha256_path,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.telemetry import (
    EXTRACTION_TELEMETRY_CHECK_KEYS,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--expected_decision_sha256", required=True)
    parser.add_argument("--expected_dataset_manifest_sha256", required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--expected_profile_decision_sha256", required=True)
    parser.add_argument("--telemetry_root", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    return parser.parse_args()


def build_manifest(args: argparse.Namespace, card: dict, card_hash: str, source_hash: str) -> dict:
    _, decision_hash, _ = load_and_validate(
        args.decision,
        expected_sha256=args.expected_decision_sha256,
        card=card,
    )
    records = []
    for model_seed in card["roster"]["model_seeds"]:
        for dataset_seed in card["new_datasets"]["seeds"]:
            stem = f"model_{model_seed}_data_{dataset_seed}"
            artifact = args.artifact_root / "field_artifacts" / f"{stem}.pt"
            sidecar = args.artifact_root / "field_artifacts" / f"{stem}.json"
            if not artifact.is_file() or not sidecar.is_file():
                raise FileNotFoundError(artifact)
            record = json.loads(sidecar.read_text())
            expected = {
                "status": "field_only_complete",
                "model_seed": int(model_seed),
                "dataset_seed": int(dataset_seed),
                "artifact": str(artifact),
                "artifact_sha256": sha256_path(artifact),
                "card_sha256": card_hash,
                "source_manifest_sha256": source_hash,
                "profile_decision_sha256": args.expected_profile_decision_sha256,
                "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
                "mechanism_decision_sha256": decision_hash,
                "requested_dataset_keys": ["fields", "split_indices"],
                "future_encoding_after_route_lock": True,
                "x0_probe_features_materialized_before_future_encoding": True,
                "dataset_payloads_deserialized": True,
                "label_tensors_may_have_been_deserialized": True,
                "label_keys_accessed": False,
                "label_values_used": False,
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"Field-only sidecar failed: {sidecar}")
            scope = record.get("gpu_telemetry_scope", {})
            start_path = args.telemetry_root / f"{stem}.start"
            done_path = args.telemetry_root / f"{stem}.done"
            if (
                scope.get("evaluator_owned_start_marker") != str(start_path)
                or scope.get("evaluator_owned_done_marker") != str(done_path)
                or scope.get("preload_and_serialization_excluded") is not True
                or not start_path.is_file()
                or not done_path.is_file()
                or scope.get("start_marker_sha256") != sha256_path(start_path)
                or scope.get("done_marker_sha256") != sha256_path(done_path)
            ):
                raise RuntimeError(f"Evaluator telemetry scope failed: {sidecar}")
            telemetry_path = args.telemetry_root / f"{stem}.json"
            if not telemetry_path.is_file():
                raise FileNotFoundError(telemetry_path)
            telemetry = json.loads(telemetry_path.read_text())
            if (
                telemetry.get("status") != "passed"
                or int(telemetry.get("model_seed", -1)) != int(model_seed)
                or int(telemetry.get("dataset_seed", -1)) != int(dataset_seed)
                or telemetry.get("card_sha256") != card_hash
                or telemetry.get("source_manifest_sha256") != source_hash
                or telemetry.get("profile_decision_sha256")
                != args.expected_profile_decision_sha256
                or str(telemetry.get("slurm_job_id"))
                != str(record.get("slurm_job_id"))
                or not isinstance(telemetry.get("checks"), dict)
                or set(telemetry["checks"]) != EXTRACTION_TELEMETRY_CHECK_KEYS
                or not telemetry["checks"]
                or not all(value is True for value in telemetry["checks"].values())
                or telemetry.get("scope")
                != "evaluator-owned GPU start through final CUDA synchronization"
                or telemetry.get("gpu_start_marker_filename") != start_path.name
                or telemetry.get("gpu_done_marker_filename") != done_path.name
                or telemetry.get("gpu_start_marker_sha256")
                != scope["start_marker_sha256"]
                or telemetry.get("gpu_done_marker_sha256")
                != scope["done_marker_sha256"]
            ):
                raise RuntimeError(f"Extraction telemetry failed: {telemetry_path}")
            raw_name = str(telemetry.get("raw_telemetry_filename", ""))
            raw_path = args.telemetry_root / raw_name
            if (
                Path(raw_name).name != raw_name
                or not raw_path.is_file()
                or sha256_path(raw_path) != telemetry["raw_telemetry_sha256"]
            ):
                raise RuntimeError(f"Raw telemetry hash mismatch: {raw_path}")
            records.append(
                {
                    "model_seed": int(model_seed),
                    "dataset_seed": int(dataset_seed),
                    "artifact": str(artifact),
                    "artifact_sha256": expected["artifact_sha256"],
                    "sidecar": str(sidecar),
                    "sidecar_sha256": sha256_path(sidecar),
                    "telemetry": str(telemetry_path),
                    "telemetry_sha256": sha256_path(telemetry_path),
                    "raw_telemetry": telemetry["raw_telemetry_filename"],
                    "raw_telemetry_sha256": telemetry["raw_telemetry_sha256"],
                    "gpu_start_marker": str(start_path),
                    "gpu_start_marker_sha256": scope["start_marker_sha256"],
                    "gpu_done_marker": str(done_path),
                    "gpu_done_marker_sha256": scope["done_marker_sha256"],
                    "slurm_job_id": record.get("slurm_job_id", "not_recorded"),
                }
            )
    return {
        "schema_version": 1,
        "status": "field_only_roster_frozen_before_label_key_access",
        "dataset_payloads_deserialized": True,
        "label_tensors_may_have_been_deserialized": True,
        "label_keys_accessed": False,
        "label_values_used": False,
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "profile_decision_sha256": args.expected_profile_decision_sha256,
        "dataset_manifest_sha256": args.expected_dataset_manifest_sha256,
        "mechanism_decision_sha256": decision_hash,
        "artifacts": records,
    }


def main() -> None:
    args = parse_args()
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Bridge card differs from launcher root")
    source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Bridge source manifest differs from launcher root")
    payload = build_manifest(args, card, card_hash, source_hash)
    write_json_once(args.output, payload)
    print(json.dumps({"status": payload["status"], "sha256": sha256_path(args.output)}))


if __name__ == "__main__":
    main()
