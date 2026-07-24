"""Fail-closed provenance, source loading, and field-only serialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable

import torch


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
CARD_PATH = PACKAGE_DIR / "prediction_card.json"
MANIFEST_PATH = PACKAGE_DIR / "source_manifest.sha256"
RESERVED_TOKEN = "20260725"


@dataclass(frozen=True)
class CheckpointSpec:
    arm: str
    seed: int
    checkpoint_step: int
    path: Path
    sha256: str


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


def load_card(
    path: Path = CARD_PATH,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    observed = sha256_path(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"Prediction-card hash mismatch: {observed} != {expected_sha256}")
    card = duplicate_safe_json(path)
    if card.get("protocol_id") != "allen_cahn_global_k_new_ic_replication_v1":
        raise RuntimeError("Unexpected replication protocol")
    return card, observed


def assert_runtime_values_safe(values: Iterable[object]) -> None:
    for value in values:
        if RESERVED_TOKEN in str(value):
            raise AssertionError("Reserved Allen--Cahn holdout token is forbidden at runtime")


def verify_file(path: Path, expected_sha256: str) -> str:
    assert_runtime_values_safe([path])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}")
    return observed


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def torch_save_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)


def torch_load(path: Path, *, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_source_manifest(path: Path = MANIFEST_PATH) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed source manifest line {line_number}")
        digest, source = parts[0], parts[1].strip()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Malformed SHA-256 on source manifest line {line_number}")
        if source in entries:
            raise ValueError(f"Duplicate source-manifest path {source}")
        entries[source] = digest
    return entries


def verify_source_manifest(
    card: dict[str, Any],
    *,
    path: Path = MANIFEST_PATH,
    expected_sha256: str | None = None,
) -> str:
    observed_manifest = sha256_path(path)
    if expected_sha256 is not None and observed_manifest != expected_sha256:
        raise RuntimeError(
            f"Source-manifest hash mismatch: {observed_manifest} != {expected_sha256}"
        )
    entries = parse_source_manifest(path)
    expected_paths = set(card["source_and_outcome_lock"]["required_manifest_paths"])
    if set(entries) != expected_paths:
        raise RuntimeError(
            "Source-manifest roster mismatch: "
            f"missing={sorted(expected_paths - set(entries))}, "
            f"extra={sorted(set(entries) - expected_paths)}"
        )
    for source, digest in entries.items():
        verify_file(_resolve_manifest_path(source), digest)
    return observed_manifest


def load_pinned_module(record: dict[str, Any]) -> ModuleType:
    path = Path(str(record["path"])).resolve()
    digest = verify_file(path, str(record["sha256"]))
    module_name = f"_allen_replication_{record['role']}_{digest[:16]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        if Path(str(existing.__file__)).resolve() != path:
            raise RuntimeError("Pinned module cache points to the wrong source root")
        return existing
    before_path = tuple(sys.path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    if tuple(sys.path) != before_path:
        raise RuntimeError("Pinned source mutated sys.path")
    if Path(str(module.__file__)).resolve() != path:
        raise RuntimeError("Pinned module resolved outside its exact source path")
    return module


def pinned_source(card: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [record for record in card["frozen_inputs"]["pinned_sources"] if record["role"] == role]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one pinned source for role {role}, found {len(matches)}")
    return matches[0]


def checkpoint_specs(card: dict[str, Any]) -> list[CheckpointSpec]:
    specs = [
        CheckpointSpec(
            arm=str(row["arm"]),
            seed=int(row["seed"]),
            checkpoint_step=int(row["checkpoint_step"]),
            path=Path(str(row["path"])),
            sha256=str(row["sha256"]),
        )
        for row in card["checkpoint_roster"]["runs"]
    ]
    expected = {
        (arm, int(seed))
        for arm in card["checkpoint_roster"]["arms"]
        for seed in card["checkpoint_roster"]["model_seeds"]
    }
    actual = {(spec.arm, spec.seed) for spec in specs}
    if len(specs) != 20 or len(actual) != 20 or actual != expected:
        raise RuntimeError("Checkpoint roster is not the exact 20-run crossed arm/seed panel")
    return specs


def load_checkpoint_model(
    spec: CheckpointSpec,
    card: dict[str, Any],
    *,
    device: torch.device,
) -> torch.nn.Module:
    verify_file(spec.path, spec.sha256)
    checkpoint = torch_load(spec.path)
    if checkpoint.get("model_family") != "spatial_conv_koopman":
        raise AssertionError(f"Unexpected model family for {spec.path}")
    module = load_pinned_module(pinned_source(card, "checkpoint_model"))
    config = module.SpatialConvKoopmanConfig.from_mapping(checkpoint["model_config"])
    expected_common = {
        "grid_size": 16,
        "channels": 2,
        "z_dim": 2048,
        "hidden_channels": 32,
        "num_blocks": 2,
        "lista_loops": 1,
        "decoder_kind": "upsample",
        "conv_activation": "tanh",
        "dense_activation": "tanh",
        "padding_mode": "circular",
    }
    observed = {key: getattr(config, key) for key in expected_common}
    if observed != expected_common:
        raise AssertionError(f"Checkpoint model configuration drifted: {observed}")
    if spec.arm == "dense":
        treatment_ok = config.encoder_kind == "dense" and float(config.lista_alpha) == 0.0
    elif spec.arm == "sparse":
        treatment_ok = config.encoder_kind == "lista" and float(config.lista_alpha) == 0.15
    else:
        treatment_ok = False
    if not treatment_ok:
        raise AssertionError(f"Checkpoint treatment drifted for {spec.arm}")
    model = module.SpatialConvKoopman(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.float().to(device).eval()
    if tuple(model.kmat.shape) != (2048, 2048) or model.kmat.dtype != torch.float32:
        raise AssertionError("Checkpoint K is not the frozen float32 full 2048x2048 matrix")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise AssertionError("Checkpoint model contains a non-float32 parameter")
    return model


def _metadata_expected(card: dict[str, Any], *, dataset_index: int, seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": "prospective_field_only",
        "dataset_index": int(dataset_index),
        "dataset_seed": int(seed),
        "field_shape": list(card["field_only_firewall"]["saved_field_shape"]),
        "stored_dt": 0.1,
        "physical_horizon": 20.0,
        "integration_dtype": "float32",
        "rng_rule": "dataset_seed_plus_10000_times_trajectory_index",
    }


def field_payload(
    fields: torch.Tensor,
    card: dict[str, Any],
    *,
    dataset_index: int,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "fields": fields.detach().cpu().contiguous(),
        "split_indices": {"val": torch.arange(fields.shape[0], dtype=torch.int64)},
        "metadata": _metadata_expected(card, dataset_index=dataset_index, seed=seed),
    }
    validate_field_payload(payload, card, dataset_index=dataset_index, seed=seed)
    return payload


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_keys(nested)


def validate_field_payload(
    payload: Any,
    card: dict[str, Any],
    *,
    dataset_index: int,
    seed: int,
) -> None:
    firewall = card["field_only_firewall"]
    if not isinstance(payload, dict) or set(payload) != set(firewall["exact_dataset_top_level_keys"]):
        raise AssertionError("Field artifact top-level keys violate the exact whitelist")
    forbidden = tuple(str(value).lower() for value in firewall["forbidden_key_fragments"])
    bad_keys = [key for key in _walk_keys(payload) if any(fragment in key.lower() for fragment in forbidden)]
    if bad_keys:
        raise AssertionError(f"Field artifact contains forbidden key fragments: {bad_keys}")
    fields = payload["fields"]
    expected_shape = tuple(int(value) for value in firewall["saved_field_shape"])
    if not isinstance(fields, torch.Tensor) or tuple(fields.shape) != expected_shape:
        raise AssertionError(f"Unexpected field tensor shape: {getattr(fields, 'shape', None)}")
    if fields.dtype != torch.float32 or not bool(torch.isfinite(fields).all()):
        raise FloatingPointError("Field tensor is not entirely finite float32")
    if bool((fields.abs() > 8.0).any()):
        raise FloatingPointError("Field tensor exceeds the frozen magnitude bound")
    split = payload["split_indices"]
    if not isinstance(split, dict) or set(split) != set(firewall["exact_split_index_keys"]):
        raise AssertionError("Split-index keys violate the exact whitelist")
    expected_indices = torch.arange(expected_shape[0], dtype=torch.int64)
    if not isinstance(split["val"], torch.Tensor) or not torch.equal(split["val"].cpu(), expected_indices):
        raise AssertionError("Validation indices must be the exact ordered 0..255 roster")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != set(firewall["exact_metadata_keys"]):
        raise AssertionError("Metadata keys violate the exact recursive whitelist")
    if metadata != _metadata_expected(card, dataset_index=dataset_index, seed=seed):
        raise AssertionError("Field-only metadata differs from the exact frozen schema")


def load_fields_only(
    path: Path,
    card: dict[str, Any],
    *,
    expected_sha256: str,
    dataset_index: int,
    seed: int,
) -> torch.Tensor:
    verify_file(path, expected_sha256)
    payload = torch_load(path)
    validate_field_payload(payload, card, dataset_index=dataset_index, seed=seed)
    fields = payload["fields"]
    indices = payload["split_indices"]["val"]
    return fields.index_select(0, indices).reshape(fields.shape[0], fields.shape[1], -1).contiguous()
