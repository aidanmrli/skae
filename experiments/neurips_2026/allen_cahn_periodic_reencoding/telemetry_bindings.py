"""Metric-free artifact authentication for periodic-reencoding telemetry."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    duplicate_safe_json,
    sha256_path,
)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _bound_artifact(
    runtime: dict[str, Any], root: Path, stem: str, filename: str
) -> tuple[Path, str]:
    path = Path(str(runtime.get(f"{stem}_path", "")))
    digest = str(runtime.get(f"{stem}_sha256", ""))
    if path != root / filename or len(digest) != 64 or sha256_path(path) != digest:
        raise RuntimeError(f"Hash-only {stem} binding failed")
    return path, digest


def _checkpoint_roster_binding(
    card: dict[str, Any], card_path: Path, runtime: dict[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    parent_record = card.get("frozen_parent", {})
    parent_path = Path(str(parent_record.get("checkpoint_card", "")))
    if not parent_path.is_absolute():
        parent_path = card_path.resolve().parents[3] / parent_path
    if sha256_path(parent_path) != parent_record.get("checkpoint_card_sha256"):
        raise RuntimeError("Frozen parent checkpoint card binding failed")
    parent = duplicate_safe_json(parent_path)
    expected = [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "checkpoint_step": int(row["checkpoint_step"]),
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in parent["checkpoint_roster"]["runs"]
    ]
    roster = runtime.get("checkpoint_roster")
    digest = str(runtime.get("checkpoint_roster_sha256", ""))
    if roster != expected or len(expected) != 20 or _canonical_digest(roster) != digest:
        raise RuntimeError("Checkpoint roster or canonical digest binding failed")
    return expected, digest


def _smoke_binding(
    card: dict[str, Any],
    runtime: dict[str, Any],
    *,
    card_hash: str,
    source_hash: str,
) -> dict[str, Any]:
    path = Path(str(runtime.get("smoke_receipt_path", "")))
    expected_path = Path(card["outcome_free_smoke"]["output_root"]) / "smoke_receipt.json"
    digest = str(runtime.get("smoke_receipt_sha256", ""))
    if path != expected_path or len(digest) != 64 or sha256_path(path) != digest:
        raise RuntimeError("Smoke-receipt path or hash binding failed")
    receipt = duplicate_safe_json(path)
    expected = {
        "status": "passed_outcome_free_gpu_smoke",
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "scientific_outcomes_accessed": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Smoke receipt status or scientific lineage failed")
    return {
        "smoke_receipt_path": str(path),
        "smoke_receipt_sha256": digest,
        "smoke_receipt_status": receipt["status"],
        "smoke_slurm_job_id": str(receipt.get("slurm_job_id", "not_recorded")),
    }


def _manifest_binding(
    path: Path,
    card: dict[str, Any],
    root: Path,
    *,
    role: str,
) -> dict[str, Any]:
    manifest = duplicate_safe_json(path)
    expected_top = {"schema_version", "protocol_id", "role", "datasets"}
    if (
        set(manifest) != expected_top
        or manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != card["protocol_id"]
        or manifest.get("role") != role
    ):
        raise RuntimeError(f"{role} field manifest schema or lineage failed")
    rows = manifest.get("datasets")
    expected_seeds = [int(row["seed"]) for row in card["prospective_datasets"][role]]
    horizon = int(card["system"][f"{role}_horizon_steps"])
    expected_shape = [
        int(card["system"]["trajectories_per_dataset"]),
        horizon + 1,
        int(card["system"]["grid_size"]),
        int(card["system"]["grid_size"]),
        int(card["system"]["channels"]),
    ]
    expected_keys = {
        "role", "dataset_index", "dataset_seed", "path", "sha256", "shape",
        "storage_bytes",
    }
    if not isinstance(rows, list) or len(rows) != 3:
        raise RuntimeError(f"{role} field manifest must contain exactly three rows")
    authenticated = []
    for index, (row, seed) in enumerate(zip(rows, expected_seeds, strict=True)):
        expected_path = root / "data" / f"{role}_seed{seed}_fields.pt"
        if (
            not isinstance(row, dict)
            or set(row) != expected_keys
            or row.get("role") != role
            or row.get("dataset_index") != index
            or row.get("dataset_seed") != seed
            or Path(str(row.get("path", ""))) != expected_path
            or row.get("shape") != expected_shape
            or row.get("storage_bytes") != 4 * int(math.prod(expected_shape))
        ):
            raise RuntimeError(f"{role} field-manifest row {index} drifted")
        digest = str(row.get("sha256", ""))
        if len(digest) != 64 or sha256_path(expected_path) != digest:
            raise RuntimeError(f"{role} field file {index} hash binding failed")
        authenticated.append({"path": str(expected_path), "sha256": digest})
    return {"role": role, "row_count": 3, "field_files": authenticated}


def authenticate_runtime_bindings(
    card: dict[str, Any],
    card_path: Path,
    runtime: dict[str, Any],
    root: Path,
    *,
    card_hash: str,
    source_hash: str,
) -> dict[str, Any]:
    selection_path, selection_hash = _bound_artifact(
        runtime, root, "selection_decision", "selection_decision.json"
    )
    validation_path, validation_hash = _bound_artifact(
        runtime, root, "validation_data_manifest", "validation_data_manifest.json"
    )
    test_path, test_hash = _bound_artifact(
        runtime, root, "test_data_manifest", "test_data_manifest.json"
    )
    scientific_path, scientific_hash = _bound_artifact(
        runtime, root, "scientific_payload", "scientific_payload.json"
    )
    if runtime.get("scientific_hash") != scientific_hash:
        raise RuntimeError("Scientific payload hash binding failed")
    roster, roster_hash = _checkpoint_roster_binding(card, card_path, runtime)
    smoke = _smoke_binding(
        card, runtime, card_hash=card_hash, source_hash=source_hash
    )
    return {
        "selection_decision_path": str(selection_path),
        "selection_decision_sha256": selection_hash,
        "validation_data_manifest_path": str(validation_path),
        "validation_data_manifest_sha256": validation_hash,
        "test_data_manifest_path": str(test_path),
        "test_data_manifest_sha256": test_hash,
        "scientific_payload_path": str(scientific_path),
        "scientific_payload_sha256": scientific_hash,
        "checkpoint_roster": roster,
        "checkpoint_roster_sha256": roster_hash,
        "field_manifest_bindings": {
            "validation": _manifest_binding(
                validation_path, card, root, role="validation"
            ),
            "test": _manifest_binding(test_path, card, root, role="test"),
        },
        **smoke,
    }
