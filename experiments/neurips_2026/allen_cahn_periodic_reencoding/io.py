"""Fail-closed I/O and provenance for the periodic-reencoding confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from experiments.neurips_2026.allen_cahn_forecast_replication import io as parent_io


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
CARD_PATH = PACKAGE_DIR / "prediction_card.json"
MANIFEST_PATH = PACKAGE_DIR / "source_manifest.sha256"
PARENT_CARD_PATH = parent_io.CARD_PATH
ARCHITECTURE_AUDIT_RELATIVE = Path(
    "docs/figures/neurips_paper_2026/_data/"
    "allen_cahn_global_k_forecast_optimized_architecture_audit.json"
)
ARCHITECTURE_ARTIFACT_MANIFEST_RELATIVE = Path(
    "docs/figures/neurips_paper_2026/_data/"
    "allen_cahn_global_k_forecast_optimized_artifact_manifest.json"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_safe_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            payload[key] = value
        return payload

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
    if card.get("protocol_id") != "allen_cahn_periodic_reencoding_confirmation_v1":
        raise RuntimeError("Unexpected periodic-reencoding protocol")
    return card, observed


def _expected_audit_roster(parent: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["arm"]), int(row["seed"])): {
            "checkpoint_step": int(row["checkpoint_step"]),
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
        }
        for row in parent["checkpoint_roster"]["runs"]
    }


def validate_architecture_audit(
    audit: dict[str, Any],
    artifact_manifest: dict[str, Any],
    parent: dict[str, Any],
) -> None:
    """Require the audit to prove the exact dense control and parent roster."""
    if (
        audit.get("audit_id")
        != "allen_cahn_global_k_forecast_optimized_architecture_audit"
        or audit.get("status")
        != "passed_capacity_and_forward_path_parity_with_joint_sparse_treatment"
        or audit.get("packet_id") != "allen_cahn_global_k_forecast_optimized"
    ):
        raise RuntimeError("Parent architecture audit identity or status drifted")
    model_source = parent_io.pinned_source(parent, "checkpoint_model")
    launch = audit["launch_sources"]
    if (
        launch.get("relevant_sources_identical_at_both_commits") is not True
        or launch.get("sha256", {}).get("skae/benchmarks/spatialized_conv_koopman.py")
        != model_source["sha256"]
    ):
        raise RuntimeError("Parent architecture audit is not bound to the model source")
    config = audit["configuration_audit"]
    expected_model = {
        "channels": 2,
        "conv_activation": "tanh",
        "decoder_kind": "upsample",
        "dense_activation": "tanh",
        "grid_size": 16,
        "hidden_channels": 32,
        "k_init_scale": 0.0,
        "lista_loops": 1,
        "num_blocks": 2,
        "padding_mode": "circular",
        "z_dim": 2048,
    }
    expected_loss = {
        "gradient": 0.05,
        "k_stability": 0.0,
        "latent": 0.1,
        "prediction": 1.0,
        "reconstruction": 0.25,
        "temporal_group_sparsity": 0.0,
    }
    expected_differences = [
        {"field": "model_config.encoder_kind", "dense": "dense", "sparse": "lista"},
        {"field": "model_config.lista_alpha", "dense": 0.0, "sparse": 0.15},
        {"field": "loss_weights.sparsity", "dense": 0.0, "sparse": 0.01},
        {
            "field": "training_args.model_variant",
            "dense": "conv_dense",
            "sparse": "conv_lista",
        },
        {"field": "training_args.lista_alpha", "dense": 0.0, "sparse": 0.15},
        {"field": "training_args.sparsity_weight", "dense": 0.0, "sparse": 0.01},
    ]
    if (
        config.get("common_model_config") != expected_model
        or config.get("common_loss_weights") != expected_loss
        or config.get("common_training_settings", {}).get("weight_decay") != 0.0
        or config.get("paired_scientific_differences") != expected_differences
        or config.get("no_other_paired_scientific_configuration_differences") is not True
    ):
        raise RuntimeError("Parent dense-control configuration audit failed")
    forward = audit["forward_path_audit"]
    structure = audit["common_checkpoint_structure"]
    shape_digest = "f4d258eee7bebf7c3970f569048912de8aad1fd90c74aa98ac0d12a762393615"
    if (
        forward.get("dense_encode") != "code"
        or forward.get("decoder_normalization_difference") is not False
        or forward.get("capacity_difference") is not False
        or structure.get("trainable_parameter_count_from_source_and_state_elements")
        != 12_698_690
        or structure.get("model_state_tensor_count") != 40
        or structure.get("shape_digest") != shape_digest
        or structure.get("effective_forward_parameter_count_excluding_inert_lista_s")
        != 8_504_386
        or structure.get("lista_s_shape") != [2048, 2048]
        or structure.get("lista_s_parameter_count") != 4_194_304
    ):
        raise RuntimeError("Parent forward-path capacity audit failed")
    expected = _expected_audit_roster(parent)
    audit_runs = audit.get("runs")
    if not isinstance(audit_runs, list) or len(audit_runs) != len(expected):
        raise RuntimeError("Parent architecture-audit roster is incomplete")
    observed_audit: set[tuple[str, int]] = set()
    for row in audit_runs:
        key = (str(row.get("arm")), int(row.get("seed", -1)))
        if (
            key not in expected
            or key in observed_audit
            or row.get("checkpoint_sha256") != expected[key]["sha256"]
            or row.get("parameter_count") != 12_698_690
            or row.get("tensor_count") != 40
            or row.get("shape_digest") != shape_digest
            or row.get("lista_s_nonzero_count") != 0
            or row.get("lista_s_max_abs") != 0.0
        ):
            raise RuntimeError("Parent architecture-audit run roster drifted")
        observed_audit.add(key)
    manifest_runs = artifact_manifest.get("runs")
    if (
        set(artifact_manifest) != {"schema_version", "description", "packet_id", "runs"}
        or artifact_manifest.get("schema_version") != 1
        or artifact_manifest.get("packet_id") != "allen_cahn_global_k_forecast_optimized"
        or not isinstance(manifest_runs, list)
        or len(manifest_runs) != len(expected)
    ):
        raise RuntimeError("Parent artifact-manifest schema drifted")
    observed_manifest: set[tuple[str, int]] = set()
    for row in manifest_runs:
        key = (str(row.get("arm")), int(row.get("seed", -1)))
        checkpoint = row.get("checkpoint", {})
        if (
            key not in expected
            or key in observed_manifest
            or row.get("checkpoint_step") != expected[key]["checkpoint_step"]
            or checkpoint
            != {"path": expected[key]["path"], "sha256": expected[key]["sha256"]}
        ):
            raise RuntimeError("Parent artifact-manifest checkpoint roster drifted")
        observed_manifest.add(key)


def load_parent_architecture_audit(
    card: dict[str, Any], parent: dict[str, Any]
) -> dict[str, Any]:
    record = card["frozen_parent"]
    if Path(str(record["architecture_audit"])) != ARCHITECTURE_AUDIT_RELATIVE:
        raise RuntimeError("Parent architecture-audit path drifted")
    path = REPO_ROOT / ARCHITECTURE_AUDIT_RELATIVE
    verify_file(path, str(record["architecture_audit_sha256"]))
    audit = duplicate_safe_json(path)
    artifact = audit.get("artifact_manifest", {})
    if Path(str(artifact.get("path", ""))) != ARCHITECTURE_ARTIFACT_MANIFEST_RELATIVE:
        raise RuntimeError("Parent architecture artifact-manifest path drifted")
    artifact_path = REPO_ROOT / ARCHITECTURE_ARTIFACT_MANIFEST_RELATIVE
    verify_file(artifact_path, str(artifact.get("sha256_at_audit", "")))
    if artifact.get("checkpoint_roster_crosscheck") != "20_of_20_exact_matches":
        raise RuntimeError("Parent audit did not cross-check all checkpoints")
    validate_architecture_audit(audit, duplicate_safe_json(artifact_path), parent)
    return audit


def load_parent_card(card: dict[str, Any]) -> dict[str, Any]:
    record = card["frozen_parent"]
    expected_path = REPO_ROOT / str(record["checkpoint_card"])
    if expected_path.resolve() != PARENT_CARD_PATH.resolve():
        raise RuntimeError("Parent-card path drifted")
    parent, observed = parent_io.load_card(
        PARENT_CARD_PATH,
        expected_sha256=str(record["checkpoint_card_sha256"]),
    )
    if observed != str(record["checkpoint_card_sha256"]):
        raise RuntimeError("Parent-card digest drifted")
    for child_key, parent_role in (
        ("pinned_model_source", "checkpoint_model"),
        ("pinned_generator_source", "physics_and_initial_conditions"),
    ):
        if record.get(child_key) != parent_io.pinned_source(parent, parent_role):
            raise RuntimeError(f"Child {child_key} is not the exact parent source record")
    parent_manifest = REPO_ROOT / str(record["checkpoint_source_manifest"])
    if parent_manifest.resolve() != parent_io.MANIFEST_PATH.resolve():
        raise RuntimeError("Parent source-manifest path drifted")
    parent_io.verify_source_manifest(
        parent,
        path=parent_manifest,
        expected_sha256=str(record["checkpoint_source_manifest_sha256"]),
    )
    load_parent_architecture_audit(card, parent)
    return parent


def checkpoint_specs(card: dict[str, Any]) -> list[parent_io.CheckpointSpec]:
    return parent_io.checkpoint_specs(load_parent_card(card))


def load_checkpoint_model(
    spec: parent_io.CheckpointSpec,
    card: dict[str, Any],
    *,
    device: torch.device,
) -> torch.nn.Module:
    return parent_io.load_checkpoint_model(spec, load_parent_card(card), device=device)


def verify_file(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}")
    return observed


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
    observed = sha256_path(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"Source-manifest hash mismatch: {observed} != {expected_sha256}")
    entries = parse_source_manifest(path)
    required = card["source_and_outcome_guard"]["required_manifest_paths"]
    if (
        not isinstance(required, list)
        or any(not isinstance(value, str) or not value for value in required)
        or len(required) != len(set(required))
    ):
        raise RuntimeError("Prediction-card source roster is malformed or duplicated")
    expected = set(required)
    if set(entries) != expected:
        raise RuntimeError(
            f"Source-manifest roster mismatch: missing={sorted(expected-set(entries))}, "
            f"extra={sorted(set(entries)-expected)}"
        )
    for source, digest in entries.items():
        verify_file(_resolve_manifest_path(source), digest)
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


def torch_load(path: Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _all_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _all_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _all_keys(nested)


def _exact_owned_storage(fields: torch.Tensor) -> bool:
    return (
        fields.is_contiguous()
        and fields.storage_offset() == 0
        and fields.untyped_storage().nbytes() == fields.numel() * fields.element_size()
    )


def field_payload(
    fields: torch.Tensor,
    card: dict[str, Any],
    *,
    role: str,
    dataset_index: int,
    seed: int,
) -> dict[str, Any]:
    horizon = 200 if role == "validation" else 400
    owned = fields.detach().cpu().clone().contiguous()
    if not _exact_owned_storage(owned):
        raise RuntimeError("Field slice retained backing storage")
    payload = {
        "fields": owned,
        "metadata": {
            "schema_version": 1,
            "protocol_id": card["protocol_id"],
            "artifact_role": role,
            "dataset_index": int(dataset_index),
            "dataset_seed": int(seed),
            "field_shape": [256, horizon + 1, 16, 16, 2],
            "stored_dt": 0.1,
            "physical_horizon": float(horizon) * 0.1,
            "integration_dtype": "float32",
            "rng_rule": "base_seed_plus_10000_times_trajectory_index",
        },
    }
    validate_field_payload(
        payload,
        card,
        role=role,
        dataset_index=dataset_index,
        seed=seed,
    )
    return payload


def validate_field_payload(
    payload: Any,
    card: dict[str, Any],
    *,
    role: str,
    dataset_index: int,
    seed: int,
) -> None:
    if role not in {"validation", "test"}:
        raise ValueError(f"Unknown dataset role {role}")
    if not isinstance(payload, dict) or set(payload) != {"fields", "metadata"}:
        raise AssertionError("Field artifact violates the exact top-level whitelist")
    forbidden = ("label", "basin", "fate", "well", "center", "count", "region")
    bad = [key for key in _all_keys(payload) if any(token in key.lower() for token in forbidden)]
    if bad:
        raise AssertionError(f"Field artifact contains forbidden metadata keys: {bad}")
    horizon = 200 if role == "validation" else 400
    fields = payload["fields"]
    expected_shape = (256, horizon + 1, 16, 16, 2)
    if not isinstance(fields, torch.Tensor) or tuple(fields.shape) != expected_shape:
        raise AssertionError(f"Unexpected field shape {getattr(fields, 'shape', None)}")
    if fields.dtype != torch.float32 or not bool(torch.isfinite(fields).all()):
        raise FloatingPointError("Field tensor is not entirely finite float32")
    if bool((fields.abs() > 8.0).any()):
        raise FloatingPointError("Field tensor exceeds the magnitude guard")
    if not _exact_owned_storage(fields):
        raise RuntimeError("Serialized field tensor does not own exact storage")
    expected_metadata = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": role,
        "dataset_index": int(dataset_index),
        "dataset_seed": int(seed),
        "field_shape": list(expected_shape),
        "stored_dt": 0.1,
        "physical_horizon": float(horizon) * 0.1,
        "integration_dtype": "float32",
        "rng_rule": "base_seed_plus_10000_times_trajectory_index",
    }
    if payload["metadata"] != expected_metadata:
        raise AssertionError("Field metadata drifted from the exact whitelist")


def load_fields_only(
    path: Path,
    card: dict[str, Any],
    *,
    expected_sha256: str,
    role: str,
    dataset_index: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    verify_file(path, expected_sha256)
    payload = torch_load(path)
    validate_field_payload(
        payload,
        card,
        role=role,
        dataset_index=dataset_index,
        seed=seed,
    )
    fields = payload["fields"].reshape(payload["fields"].shape[0], payload["fields"].shape[1], -1)
    return fields.to(device=device, dtype=torch.float32).contiguous()
