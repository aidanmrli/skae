"""Authenticate generated field manifests and, after outcome release, payloads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    duplicate_safe_json,
    torch_load,
    validate_field_payload,
    verify_file,
)


def _expected_shape(card: dict[str, Any], role: str) -> list[int]:
    system = card["system"]
    return [
        int(system["trajectories_per_dataset"]),
        int(system[f"{role}_horizon_steps"]) + 1,
        int(system["grid_size"]),
        int(system["grid_size"]),
        int(system["channels"]),
    ]


def verify_field_artifact_manifest(
    path: Path,
    card: dict[str, Any],
    *,
    expected_sha256: str,
    role: str,
    root: Path,
    validate_payloads: bool = True,
) -> dict[str, Any]:
    """Verify the exact three-file field roster, hashes, and optional contents."""
    if role not in {"validation", "test"}:
        raise ValueError(f"Unknown field-artifact role {role}")
    root = root.resolve()
    expected_manifest = root / f"{role}_data_manifest.json"
    if path.is_symlink() or path.resolve() != expected_manifest:
        raise RuntimeError(f"{role} field-manifest path escaped the frozen root")
    manifest_hash = verify_file(path, expected_sha256)
    manifest = duplicate_safe_json(path)
    if (
        set(manifest) != {"schema_version", "protocol_id", "role", "datasets"}
        or manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != card["protocol_id"]
        or manifest.get("role") != role
    ):
        raise RuntimeError(f"{role} field-manifest schema or lineage drifted")
    frozen = card["prospective_datasets"][role]
    if (
        not isinstance(frozen, list)
        or len(frozen) != 3
        or [(row.get("index"), row.get("seed")) for row in frozen]
        != [(index, row.get("seed")) for index, row in enumerate(frozen)]
    ):
        raise RuntimeError(f"Frozen {role} dataset roster drifted")
    rows = manifest.get("datasets")
    if not isinstance(rows, list) or len(rows) != 3:
        raise RuntimeError(f"{role} field manifest must contain exactly three rows")
    shape = _expected_shape(card, role)
    storage_bytes = 4 * int(math.prod(shape))
    row_keys = {
        "role",
        "dataset_index",
        "dataset_seed",
        "path",
        "sha256",
        "shape",
        "storage_bytes",
    }
    authenticated: list[dict[str, Any]] = []
    for index, (row, frozen_row) in enumerate(zip(rows, frozen, strict=True)):
        seed = int(frozen_row["seed"])
        field_path = root / "data" / f"{role}_seed{seed}_fields.pt"
        if (
            not isinstance(row, dict)
            or set(row) != row_keys
            or row.get("role") != role
            or row.get("dataset_index") != index
            or row.get("dataset_seed") != seed
            or Path(str(row.get("path", ""))) != field_path
            or row.get("shape") != shape
            or row.get("storage_bytes") != storage_bytes
            or field_path.is_symlink()
        ):
            raise RuntimeError(f"{role} field-manifest row {index} drifted")
        digest = str(row.get("sha256", ""))
        verify_file(field_path, digest)
        if validate_payloads:
            payload = torch_load(field_path)
            validate_field_payload(
                payload,
                card,
                role=role,
                dataset_index=index,
                seed=seed,
            )
        authenticated.append(
            {
                "dataset_index": index,
                "dataset_seed": seed,
                "path": str(field_path),
                "sha256": digest,
                "payload_validated": bool(validate_payloads),
            }
        )
    return {
        "role": role,
        "manifest_path": str(path),
        "manifest_sha256": manifest_hash,
        "field_files": authenticated,
    }
