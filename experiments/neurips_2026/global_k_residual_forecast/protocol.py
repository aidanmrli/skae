"""Frozen protocol helpers for the residualized one-global-K diagnostic."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import copy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CARD = PACKAGE_ROOT / "prediction_card.json"
DEFAULT_TASKS = PACKAGE_ROOT / "task_manifest.json"
DEFAULT_SOURCES = PACKAGE_ROOT / "source_manifest.sha256"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _decode_json_object(value: bytes, role: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, item in pairs:
            if key in decoded:
                raise ValueError(f"Duplicate JSON key {key!r} in {role}")
            decoded[key] = item
        return decoded

    decoded = json.loads(value, object_pairs_hook=reject_duplicates)
    if not isinstance(decoded, dict):
        raise TypeError(f"Expected a JSON object in {role}")
    return decoded


def load_json(path: Path) -> dict[str, Any]:
    return _decode_json_object(path.read_bytes(), str(path))


def read_verified_bytes(path: Path, expected: str, role: str) -> bytes:
    value = path.read_bytes()
    observed = sha256_bytes(value)
    if observed != expected:
        raise RuntimeError(
            f"{role} SHA-256 mismatch for {path}: {observed} != {expected}"
        )
    return value


def load_verified_json(path: Path, expected: str, role: str) -> dict[str, Any]:
    value = read_verified_bytes(path, expected, role)
    return _decode_json_object(value, role)


def load_torch_payload(value: bytes, map_location: str = "cpu") -> dict[str, Any]:
    payload = torch.load(io.BytesIO(value), map_location=map_location)
    if not isinstance(payload, dict):
        raise RuntimeError("Authenticated torch payload is not a dictionary")
    return payload


def _external_path(value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{role} path is missing")
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def authenticate_v2_inputs(card: dict[str, Any]) -> dict[str, Any]:
    """Read-once authenticate the complete V2 bundle used by this protocol."""

    v2 = card["authenticated_v2_inputs"]
    pairs = ("card", "task", "source_lock", "audit_summary", "packet")
    payloads: dict[str, bytes] = {}
    paths: dict[str, Path] = {}
    for name in pairs:
        path = _external_path(v2.get(f"{name}_path"), f"V2 {name}")
        expected = v2.get(f"{name}_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError(f"V2 {name} SHA-256 is missing")
        paths[name] = path
        payloads[name] = read_verified_bytes(path, expected, f"V2 {name}")

    v2_card = _decode_json_object(payloads["card"], "authenticated V2 card")
    source_lock = _decode_json_object(
        payloads["source_lock"], "authenticated V2 source lock"
    )
    if (
        source_lock.get("schema_version") != 1
        or source_lock.get("protocol_id") != v2_card.get("protocol_id")
        or source_lock.get("card_sha256") != v2["card_sha256"]
    ):
        raise RuntimeError("V2 source lock does not bind the authenticated V2 card")
    external = source_lock.get("external_inputs")
    if not isinstance(external, dict) or set(external) != {
        "full_manifest", "full_task_tsv", "smoke_manifest", "smoke_task_tsv"
    }:
        raise RuntimeError("V2 source-lock external-input roster drifted")
    full_task = external["full_task_tsv"]
    if not isinstance(full_task, dict) or (
        full_task.get("path") != v2["task_path"]
        or full_task.get("sha256") != v2["task_sha256"]
    ):
        raise RuntimeError("V2 source lock does not bind the declared full task TSV")
    for name, row in external.items():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise RuntimeError(f"Malformed V2 source-lock external input: {name}")
        if name == "full_task_tsv":
            continue
        read_verified_bytes(
            _external_path(row["path"], f"V2 source-lock {name}"),
            row["sha256"],
            f"V2 source-lock {name}",
        )

    sparse = v2_card["training_arms"]["sparse"]
    representative_path = _external_path(
        sparse["representative_frozen_config"], "V2 sparse representative config"
    )
    representative = read_verified_bytes(
        representative_path,
        sparse["representative_frozen_config_sha256"],
        "V2 sparse representative config",
    )
    representative_config = _decode_json_object(
        representative, "authenticated V2 representative config"
    )
    dense = v2_card["training_arms"]["dense"]
    read_verified_bytes(
        _external_path(dense["source_recipe_card"], "V2 dense recipe card"),
        dense["source_recipe_card_sha256"],
        "V2 dense recipe card",
    )
    return {
        "card": v2_card,
        "representative_config_bytes": representative,
        "representative_config": representative_config,
        "authenticated_paths": {name: str(path) for name, path in paths.items()},
    }


def verify_sha(path: Path, expected: str, role: str) -> str:
    observed = sha256_path(path)
    if observed != expected:
        raise RuntimeError(
            f"{role} SHA-256 mismatch for {path}: {observed} != {expected}"
        )
    return observed


def verify_source_manifest_bytes(
    value: bytes, role: str = "source manifest"
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(value.decode().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pieces = line.split(maxsplit=1)
        if (
            len(pieces) != 2
            or len(pieces[0]) != 64
            or any(character not in "0123456789abcdef" for character in pieces[0])
        ):
            raise RuntimeError(
                f"Malformed {role} line {line_number}: {raw}"
            )
        expected, relative = pieces
        if relative in seen:
            raise RuntimeError(f"Duplicate source manifest path: {relative}")
        seen.add(relative)
        source = (REPOSITORY_ROOT / relative).resolve()
        if not source.is_relative_to(REPOSITORY_ROOT.resolve()):
            raise RuntimeError(f"Source manifest path escapes repository: {relative}")
        verify_sha(source, expected, f"source manifest line {line_number}")
        rows.append({"path": relative, "sha256": expected})
    if not rows:
        raise RuntimeError("Source manifest is empty")
    return rows


def verify_source_manifest(path: Path = DEFAULT_SOURCES) -> list[dict[str, str]]:
    return verify_source_manifest_bytes(path.read_bytes(), str(path))


def validate_launch_authorization_transition(
    preauthorization_card: dict[str, Any], authorized_card: dict[str, Any]
) -> None:
    """Permit exactly the audited false-to-true lifecycle transition."""

    before = copy.deepcopy(preauthorization_card)
    after = copy.deepcopy(authorized_card)
    if before.get("freeze", {}).get("launch_authorized") is not False:
        raise RuntimeError("Preauthorization card must have launch_authorized=false")
    if after.get("freeze", {}).get("launch_authorized") is not True:
        raise RuntimeError("Authorized card must have launch_authorized=true")
    after["freeze"]["launch_authorized"] = False
    if after != before:
        raise RuntimeError(
            "Launch authorization may change only freeze.launch_authorized false-to-true"
        )


def load_frozen_protocol(
    *,
    card_path: Path,
    task_path: Path,
    source_manifest_path: Path,
    expected_card_sha256: str,
    expected_task_sha256: str,
    expected_source_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    card_bytes = read_verified_bytes(card_path, expected_card_sha256, "prediction card")
    task_bytes = read_verified_bytes(task_path, expected_task_sha256, "task manifest")
    source_bytes = read_verified_bytes(
        source_manifest_path, expected_source_manifest_sha256, "source manifest"
    )
    card = _decode_json_object(card_bytes, "authenticated prediction card")
    tasks = _decode_json_object(task_bytes, "authenticated task manifest")
    if card["protocol_id"] != tasks["protocol_id"]:
        raise RuntimeError("Card/task protocol IDs differ")
    if card["freeze"].get("launch_authorized") is not True:
        raise RuntimeError("Prediction card is still frozen with launch_authorized=false")
    if card["freeze"]["task_manifest_sha256"] != expected_task_sha256:
        raise RuntimeError("Card does not bind the authorized task manifest")
    if card["freeze"]["source_manifest_sha256"] != expected_source_manifest_sha256:
        raise RuntimeError("Card does not bind the authorized source manifest")
    sources = verify_source_manifest_bytes(source_bytes, "authenticated source manifest")
    return card, tasks, {
        "card_sha256": expected_card_sha256,
        "task_manifest_sha256": expected_task_sha256,
        "source_manifest_sha256": expected_source_manifest_sha256,
        "source_rows": sources,
    }


def task_by_index(tasks: dict[str, Any], task_index: int) -> dict[str, Any]:
    rows = tasks["tasks"]
    expected = list(range(len(rows)))
    observed = [int(row["task_id"]) for row in rows]
    if observed != expected or len(rows) != 10:
        raise RuntimeError(f"Frozen task roster is not exactly 0..9: {observed}")
    if task_index not in expected:
        raise IndexError(task_index)
    row = rows[task_index]
    if int(row["model_seed"]) != 100 + task_index:
        raise RuntimeError("Task/model-seed ordering drifted")
    return row


def authenticate_checkpoint_roster(tasks: dict[str, Any]) -> int:
    """Hash every frozen checkpoint without deserializing or exposing outcomes."""

    from experiments.neurips_2026.global_k_residual_forecast.diagnostic_recompute import (
        EXPECTED_PARAMETER_COUNTS,
    )

    if (
        set(tasks) != {
            "schema_version", "protocol_id", "roster_rule", "provenance_contract",
            "tasks",
        }
        or tasks["schema_version"] != 1
        or tasks["provenance_contract"] != EXPECTED_PARAMETER_COUNTS
    ):
        raise RuntimeError("Frozen task-manifest schema or parameter contract drifted")
    count = 0
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for task_index in range(10):
        task = task_by_index(tasks, task_index)
        if set(task) != {
            "task_id", "model_seed", "sparse_checkpoint", "dense_checkpoint"
        }:
            raise RuntimeError(f"Task {task_index} schema drifted")
        for arm in ("sparse", "dense"):
            item = task[f"{arm}_checkpoint"]
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                raise RuntimeError(f"Task {task_index} {arm} binding drifted")
            path_text = item["path"]
            digest = item["sha256"]
            seed_marker = f"/seed_{task['model_seed']}/"
            if (
                not isinstance(path_text, str)
                or not Path(path_text).is_absolute()
                or seed_marker not in path_text
                or not path_text.endswith("/checkpoint.pt")
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or path_text in seen_paths
                or digest in seen_hashes
            ):
                raise RuntimeError(f"Task {task_index} {arm} provenance is invalid")
            seen_paths.add(path_text)
            seen_hashes.add(digest)
            read_verified_bytes(
                Path(path_text),
                digest,
                f"task {task_index} {arm} checkpoint",
            )
            count += 1
    return count


def stable_sign_pair_permutations(
    latent_dim: int, count: int, seed: int,
) -> np.ndarray:
    if latent_dim <= 0 or latent_dim % 2:
        raise ValueError("A positive even sign-split latent dimension is required")
    base_dim = latent_dim // 2
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    identity = np.arange(latent_dim)
    seen: set[bytes] = set()
    while len(rows) < count:
        base = rng.permutation(base_dim)
        permutation = np.concatenate([base, base + base_dim]).astype(np.int64)
        key = permutation.tobytes()
        if np.array_equal(permutation, identity) or key in seen:
            continue
        seen.add(key)
        rows.append(permutation)
    return np.stack(rows)


def nearest_family(
    latent: torch.Tensor,
    representatives: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign every row to its maximum-Jaccard retained family."""
    if latent.ndim != 2 or representatives.ndim != 2:
        raise ValueError((latent.shape, representatives.shape))
    if latent.shape[1] != representatives.shape[1] or representatives.shape[0] < 1:
        raise ValueError((latent.shape, representatives.shape))
    support = latent.abs() > float(threshold)
    support_float = support.to(dtype=torch.float32)
    reps_float = representatives.to(device=latent.device, dtype=torch.float32)
    intersection = support_float @ reps_float.T
    union = (
        support_float.sum(dim=1, keepdim=True)
        + reps_float.sum(dim=1).unsqueeze(0)
        - intersection
    )
    similarities = torch.where(
        union > 0,
        intersection / union.clamp_min(1.0),
        torch.ones_like(union),
    )
    best_similarity, assignment = similarities.max(dim=1)
    return assignment, best_similarity


def select_projectors(
    assignments: torch.Tensor,
    representatives: torch.Tensor,
) -> torch.Tensor:
    if assignments.dtype != torch.long:
        assignments = assignments.long()
    return representatives.to(assignments.device).index_select(0, assignments)


def exact_sign_flip_pvalue(differences: Iterable[float]) -> float:
    values = np.asarray(list(differences), dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError(values)
    observed = float(values.mean())
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=values.size)))
    null = (signs * values[None, :]).mean(axis=1)
    return float(np.mean(null >= observed - 1e-15))


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(pvalues[key])))
        adjusted[key] = running
    return adjusted


def paired_bootstrap_reduction_interval(
    treatment: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    treatment = np.asarray(treatment, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if treatment.shape != baseline.shape or treatment.ndim != 1:
        raise ValueError((treatment.shape, baseline.shape))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, treatment.size, size=(replicates, treatment.size))
    treatment_mean = treatment[indices].mean(axis=1)
    baseline_mean = baseline[indices].mean(axis=1)
    reduction = (baseline_mean - treatment_mean) / np.maximum(baseline_mean, 1e-30)
    low, high = np.quantile(reduction, [0.025, 0.975])
    return float(low), float(high)


def publish_h500_extension(mechanism_supported: bool, raw_h500_gate: bool) -> bool:
    """The secondary H500 tier can never substitute for failed H200 evidence."""

    return bool(mechanism_supported and raw_h500_gate)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.chmod(0o600)
    temporary.replace(path)
