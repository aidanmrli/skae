"""Outcome-aware integrity checks run before periodic-result adjudication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.neurips_2026.allen_cahn_periodic_reencoding.field_integrity import (
    verify_field_artifact_manifest,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.lineage import (
    canonical_digest,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    select_recipe_cadences,
    validation_candidate_scores,
)


def verify_selection_lineage(
    selection: dict[str, Any],
    scientific: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    card: dict[str, Any],
    *,
    card_hash: str,
    source_hash: str,
) -> dict[str, Any]:
    """Recompute the validation-only selector and its complete provenance."""

    expected_keys = {
        "schema_version",
        "protocol_id",
        "card_sha256",
        "source_manifest_sha256",
        "selection_endpoint",
        "selection_scope",
        "selected_cadences",
        "candidate_scores",
        "validation_rows_sha256",
    }
    if set(selection) != expected_keys:
        raise RuntimeError("Selection decision schema drifted")
    expected_fixed = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "source_manifest_sha256": source_hash,
        "selection_endpoint": "H200 cumulative field MSE",
        "selection_scope": "one recipe-level cadence per arm",
    }
    if any(selection.get(key) != value for key, value in expected_fixed.items()):
        raise RuntimeError("Selection decision fixed metadata drifted")
    digest = canonical_digest(validation_rows)
    scores = validation_candidate_scores(validation_rows, card)
    selected = select_recipe_cadences(validation_rows, card)
    if selection["validation_rows_sha256"] != digest:
        raise RuntimeError("Validation-row digest differs from the decision")
    if selection["candidate_scores"] != scores:
        raise RuntimeError("Stored validation candidate scores do not recompute")
    if selection["selected_cadences"] != selected:
        raise RuntimeError("Stored validation cadence choice does not recompute")
    if scientific.get("selected_cadences") != selected:
        raise RuntimeError("Scientific payload cadence choice does not recompute")
    return {
        "selected_cadences": selected,
        "validation_rows_sha256": digest,
        "candidate_scores_recomputed_exactly": True,
    }


def validate_materialized_fields(
    guard: dict[str, Any],
    scientific: dict[str, Any],
    card: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Hash, deserialize, and schema-check all six prospective field files."""

    result: dict[str, Any] = {}
    for role in ("validation", "test"):
        manifest_path = root / f"{role}_data_manifest.json"
        expected_manifest_hash = str(scientific[f"{role}_data_manifest_sha256"])
        if Path(str(scientific[f"{role}_data_manifest_path"])) != manifest_path:
            raise RuntimeError(f"Scientific {role} manifest path drifted")
        if guard.get(f"{role}_data_manifest_sha256") != expected_manifest_hash:
            raise RuntimeError(f"Guard and scientific {role} manifest hashes differ")
        result[role] = verify_field_artifact_manifest(
            manifest_path,
            card,
            expected_sha256=expected_manifest_hash,
            role=role,
            root=root,
            validate_payloads=True,
        )
    return {
        "all_six_field_artifacts_hash_and_schema_validated": True,
        "roles": result,
    }


def verify_h400_failure_lineage(
    scientific: dict[str, Any],
    card: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, int]:
    """Authenticate nested H400 failure tiers without opening partial prefixes."""

    all_failures = scientific["stress_failures"]
    required = scientific["required_stress_failures"]
    grid = scientific["grid_stress_failures"]
    p200 = scientific["p200_failures"]
    if not all(isinstance(rows, list) for rows in (all_failures, required, grid, p200)):
        raise RuntimeError("H400 failure tiers must be lists")
    cadence_grid = set(card["cadence_selection"]["cadence_grid"])
    required_grid = {"direct", selected["dense"], selected["sparse"]}
    model_seeds = set(card["roster"]["model_seeds"])
    for row in all_failures:
        if not isinstance(row, dict):
            raise RuntimeError("An H400 failure record is not an object")
        if row.get("status") == "whole_h400_policy_nonfinite":
            if (
                set(row)
                != {
                    "arm", "model_seed", "cadence", "status", "error_type",
                    "finite_prefix_scored",
                }
                or row["arm"] not in {"dense", "sparse"}
                or row["model_seed"] not in model_seeds
                or row["cadence"] not in cadence_grid | {200}
                or row["finite_prefix_scored"] is not False
            ):
                raise RuntimeError("A model-level H400 failure record drifted")
        elif row.get("status") == "strict_h400_integrity_failure":
            if (
                set(row)
                != {"tier", "status", "error_type", "error", "finite_prefix_scored"}
                or row["tier"] not in {"required", "full_grid", "p200"}
                or row["finite_prefix_scored"] is not False
            ):
                raise RuntimeError("An H400 integrity-failure record drifted")
        else:
            raise RuntimeError("Unknown H400 failure status")
    digest_set = {canonical_digest(row) for row in all_failures}
    if len(digest_set) != len(all_failures):
        raise RuntimeError("H400 failure roster contains duplicates")
    for rows in (required, grid, p200):
        if any(canonical_digest(row) not in digest_set for row in rows):
            raise RuntimeError("An H400 tier failure is absent from the all-failure roster")
    expected_required = [
        row for row in all_failures
        if row.get("cadence") in required_grid or row.get("tier") == "required"
    ]
    expected_grid = [
        row for row in all_failures
        if row.get("cadence") in cadence_grid or row.get("tier") == "full_grid"
    ]
    expected_p200 = [
        row for row in all_failures
        if row.get("cadence") == 200 or row.get("tier") == "p200"
    ]
    for observed, expected, tier in (
        (required, expected_required, "required"),
        (grid, expected_grid, "full_grid"),
        (p200, expected_p200, "p200"),
    ):
        if {canonical_digest(row) for row in observed} != {
            canonical_digest(row) for row in expected
        }:
            raise RuntimeError(f"{tier} H400 failure classification drifted")
    return {
        "all": len(all_failures),
        "required": len(required),
        "full_grid": len(grid),
        "p200": len(p200),
    }
