"""Fail-closed input, seed, and artifact I/O for the bridge packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


CARD_PATH = Path(__file__).with_name("prediction_card.json")
RESERVED_SEED = "20260725"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_safe_json(path: Path) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(), object_pairs_hook=hook)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def load_card(path: Path = CARD_PATH) -> tuple[dict[str, Any], str]:
    card = _duplicate_safe_json(path)
    if RESERVED_SEED in json.dumps(card["new_datasets"]["paths"]):
        raise AssertionError("Reserved conditional holdout appears in new dataset paths")
    seeds = [int(value) for value in card["new_datasets"]["seeds"]]
    if len(seeds) != len(set(seeds)) or RESERVED_SEED in {str(value) for value in seeds}:
        raise AssertionError("New dataset seed roster is invalid")
    return card, sha256_path(path)


def verify_file(path: Path, expected_sha256: str) -> None:
    if RESERVED_SEED in str(path):
        raise AssertionError("Reserved conditional holdout path is forbidden")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}")


def verify_frozen_inputs(card: dict[str, Any]) -> None:
    for record in card["inputs"].values():
        if isinstance(record, dict) and "path" in record and "sha256" in record:
            verify_file(Path(record["path"]), str(record["sha256"]))


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def assert_field_only_keys(keys: list[str], card: dict[str, Any]) -> None:
    fragments = tuple(card["field_only_stage"]["forbidden_key_fragments"])
    bad = [key for key in keys if any(fragment in key.lower() for fragment in fragments)]
    if bad:
        raise AssertionError(f"Field-only stage requested forbidden keys: {bad}")
    allowed = set(card["field_only_stage"]["allowed_dataset_keys"])
    if not set(keys).issubset(allowed):
        raise AssertionError(f"Field-only stage requested non-whitelisted keys: {keys}")


def load_fields_only(
    path: Path,
    *,
    split: str,
    card: dict[str, Any],
    expected_sha256: str | None = None,
    expected_count: int | None = None,
    expected_horizon: int | None = None,
) -> torch.Tensor:
    if expected_sha256 is not None:
        verify_file(path, expected_sha256)
    elif RESERVED_SEED in str(path):
        raise AssertionError("Reserved conditional holdout path is forbidden")
    requested = ["fields", "split_indices"]
    assert_field_only_keys(requested, card)
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError("Dataset must be a mapping")
    fields = payload["fields"]
    indices = payload["split_indices"][split]
    selected = fields[indices].detach().cpu().to(dtype=torch.float32)
    if selected.ndim != 5 or selected.shape[-1] != 2:
        raise ValueError(f"Unexpected field tensor shape {tuple(selected.shape)}")
    if expected_count is not None and selected.shape[0] != int(expected_count):
        raise AssertionError("Dataset trajectory count differs from frozen card")
    if expected_horizon is not None and selected.shape[1] != int(expected_horizon) + 1:
        raise AssertionError("Dataset trajectory horizon differs from frozen card")
    return selected.reshape(selected.shape[0], selected.shape[1], -1).contiguous()


def load_training_fates(card: dict[str, Any]) -> torch.Tensor:
    """Label-aware stage only: load the original train split's T20 fates."""

    record = card["inputs"]["training_dataset"]
    path = Path(record["path"])
    verify_file(path, str(record["sha256"]))
    payload = _torch_load(path)
    labels = payload["global_basin_labels"]
    indices = payload["split_indices"][record["field_only_split"]]
    selected = labels[indices].detach().cpu().to(dtype=torch.int64)
    if selected.shape != (int(record["expected_trajectories"]),):
        raise AssertionError("Training fate vector differs from frozen roster")
    return selected


def load_dataset_manifest(path: Path, card: dict[str, Any]) -> dict[str, Any]:
    manifest = _duplicate_safe_json(path)
    expected_seeds = [int(value) for value in card["new_datasets"]["seeds"]]
    records = manifest.get("datasets", [])
    if [int(item["seed"]) for item in records] != expected_seeds:
        raise RuntimeError("Dataset manifest seed roster mismatch")
    for item, relative in zip(records, card["new_datasets"]["paths"]):
        candidate = Path(card["new_datasets"]["output_root"]) / relative
        if Path(item["path"]) != candidate or RESERVED_SEED in str(candidate):
            raise RuntimeError("Dataset manifest path mismatch")
        verify_file(candidate, str(item["sha256"]))
        verify_file(Path(str(candidate) + ".summary.json"), str(item["summary_sha256"]))
    return manifest


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite_tree(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return bool(torch.isfinite(torch.tensor(value)))
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    return True
