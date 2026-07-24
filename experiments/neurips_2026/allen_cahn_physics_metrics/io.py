"""Fail-closed I/O and lineage checks for physics-aware Allen--Cahn scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
CARD_PATH = PACKAGE_DIR / "prediction_card.json"
MANIFEST_PATH = PACKAGE_DIR / "source_manifest.sha256"
FORBIDDEN_PATH_TOKENS = ("early_fate", "20260725")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_safe_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def verify_file(path: Path, expected_sha256: str) -> str:
    assert_paths_sealed([path])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}")
    return observed


def assert_paths_sealed(values: Iterable[object]) -> None:
    for value in values:
        lowered = str(value).lower()
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            raise AssertionError("Permanently sealed Allen--Cahn artifact token at runtime")


def load_card(
    path: Path = CARD_PATH,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    assert_paths_sealed([path])
    observed = sha256_path(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"Prediction-card hash mismatch: {observed} != {expected_sha256}")
    card = duplicate_safe_json(path)
    if card.get("protocol_id") != "allen_cahn_existing_new_ic_physics_metrics_v1":
        raise RuntimeError("Unexpected physics-metric protocol")
    assert_paths_sealed(_walk_strings(card["authenticated_inputs"]))
    return card, observed


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _walk_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_strings(nested)


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    assert_paths_sealed([path])
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def _resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def verify_source_manifest(path: Path, *, expected_sha256: str) -> str:
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"Source-manifest hash mismatch: {observed} != {expected_sha256}")
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, separator, source = line.partition("  ")
        if not separator or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Malformed source-manifest line {line_number}")
        if source in entries:
            raise ValueError(f"Duplicate source path {source}")
        entries[source] = digest
    if not entries:
        raise ValueError("Source manifest is empty")
    assert_paths_sealed(entries)
    for source, digest in entries.items():
        verify_file(_resolve_manifest_path(source), digest)
    return observed


def authenticated_prior(card: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata lineage without opening any field or prior metric payload."""

    inputs = card["authenticated_inputs"]
    source_card_path = Path(inputs["source_card"]["path"])
    source_manifest_path = Path(inputs["source_manifest"]["path"])
    dataset_manifest_path = Path(inputs["dataset_manifest"]["path"])
    receipt_path = Path(inputs["outcome_guard_receipt"]["path"])
    verify_file(source_card_path, inputs["source_card"]["sha256"])
    verify_file(source_manifest_path, inputs["source_manifest"]["sha256"])
    verify_file(dataset_manifest_path, inputs["dataset_manifest"]["sha256"])
    verify_file(receipt_path, inputs["outcome_guard_receipt"]["sha256"])
    source_card = duplicate_safe_json(source_card_path)
    dataset_manifest = duplicate_safe_json(dataset_manifest_path)
    receipt = duplicate_safe_json(receipt_path)
    if source_card.get("protocol_id") != "allen_cahn_global_k_new_ic_replication_v1":
        raise RuntimeError("Authenticated source card has the wrong protocol")
    if dataset_manifest.get("card_sha256") != inputs["source_card"]["sha256"]:
        raise RuntimeError("Dataset manifest does not bind the authenticated source card")
    if dataset_manifest.get("source_manifest_sha256") != inputs["source_manifest"]["sha256"]:
        raise RuntimeError("Dataset manifest does not bind the authenticated source manifest")
    if receipt.get("card_sha256") != inputs["source_card"]["sha256"]:
        raise RuntimeError("Prior receipt does not bind the authenticated source card")
    if receipt.get("dataset_manifest_sha256") != inputs["dataset_manifest"]["sha256"]:
        raise RuntimeError("Prior receipt does not bind the authenticated dataset manifest")
    if receipt.get("checkpoint_roster_sha256") != inputs["checkpoint_roster_sha256"]:
        raise RuntimeError("Prior receipt does not bind the frozen checkpoint roster")
    if receipt.get("status") != "authorized_for_dependent_cpu_summary":
        raise RuntimeError("Prior scientific packet was not authorized")
    datasets = dataset_manifest.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 3:
        raise RuntimeError("Dataset manifest is not the exact three-dataset panel")
    exact_datasets = inputs["datasets"]
    observed = [
        {
            "dataset_index": int(row["dataset_index"]),
            "dataset_seed": int(row["dataset_seed"]),
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "field_shape": list(row["field_shape"]),
        }
        for row in datasets
    ]
    if observed != exact_datasets:
        raise RuntimeError("Authenticated dataset roster drifted from the frozen card")
    expected_roster = [
        {
            "arm": str(row["arm"]),
            "seed": int(row["seed"]),
            "checkpoint_step": int(row["checkpoint_step"]),
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
        }
        for row in source_card["checkpoint_roster"]["runs"]
    ]
    if receipt.get("checkpoint_roster") != expected_roster or len(expected_roster) != 20:
        raise RuntimeError("Authenticated checkpoint roster is incomplete or reordered")
    assert_paths_sealed(
        [row["path"] for row in exact_datasets]
        + [row["path"] for row in expected_roster]
    )
    return {
        "source_card": source_card,
        "datasets": exact_datasets,
        "checkpoint_roster": expected_roster,
        "checkpoint_roster_sha256": inputs["checkpoint_roster_sha256"],
        "prior_receipt_sha256": inputs["outcome_guard_receipt"]["sha256"],
    }
