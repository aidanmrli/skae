"""Fail-closed I/O and pinned-model loading for the early-fate probe."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable

import numpy as np
import torch


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
CARD_PATH = PACKAGE_DIR / "prediction_card.json"
TASK_PATH = PACKAGE_DIR / "task_manifest.json"
SOURCE_MANIFEST_PATH = PACKAGE_DIR / "source_manifest.sha256"
RESERVED_TOKEN = "20260725"


@dataclass(frozen=True)
class CheckpointSpec:
    arm: str
    seed: int
    checkpoint_step: int
    path: Path
    sha256: str
    git_commit: str


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

    payload = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def load_card(
    path: Path = CARD_PATH, *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    observed = sha256_path(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"Prediction-card SHA mismatch: {observed} != {expected_sha256}")
    card = duplicate_safe_json(path)
    if card.get("protocol_id") != "allen_cahn_early_fate_probe_v1":
        raise RuntimeError("Unexpected early-fate protocol")
    return card, observed


def assert_runtime_values_safe(values: Iterable[object]) -> None:
    if any(RESERVED_TOKEN in str(value) for value in values):
        raise AssertionError("Reserved Allen--Cahn seed token is forbidden at runtime")


def verify_file(path: Path, expected_sha256: str) -> str:
    assert_runtime_values_safe([path])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_path(path)
    if observed != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected_sha256}")
    return observed


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_task_manifest(
    card: dict[str, Any], *, expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    record = card["inputs"]["task_manifest"]
    path = resolve_path(str(record["path"]))
    card_expected = str(record["sha256"])
    if expected_sha256 is not None and expected_sha256 != card_expected:
        raise RuntimeError("Caller task-manifest root differs from the card")
    observed = verify_file(path, card_expected)
    payload = duplicate_safe_json(path)
    if (
        payload.get("protocol_id") != card["protocol_id"]
        or payload.get("status") != "outcome_blind_task_roster_frozen"
        or payload.get("output_root")
        != "/network/scratch/l/lia/skae/allen_cahn_early_fate_probe_20260721_v1"
        or len(payload.get("gpu_tasks", [])) != 1
        or len(payload.get("cpu_tasks", [])) != 1
    ):
        raise RuntimeError("Task manifest contract failed")
    return payload, observed


def parse_source_manifest(path: Path = SOURCE_MANIFEST_PATH) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            raise ValueError(f"Blank source-manifest line {line_number}")
        digest, source = raw.split("  ", 1)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid digest on source-manifest line {line_number}")
        if source in result:
            raise ValueError(f"Duplicate source path {source}")
        result[source] = digest
    return result


def verify_source_manifest(
    card: dict[str, Any],
    *,
    path: Path = SOURCE_MANIFEST_PATH,
    expected_sha256: str | None = None,
) -> str:
    observed = sha256_path(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(f"Source-manifest SHA mismatch: {observed} != {expected_sha256}")
    records = parse_source_manifest(path)
    expected = set(card["source_lock"]["required_manifest_paths"])
    if set(records) != expected:
        raise RuntimeError(
            f"Source roster mismatch; missing={sorted(expected-set(records))}, "
            f"extra={sorted(set(records)-expected)}"
        )
    for source, digest in records.items():
        verify_file(resolve_path(source), digest)
    return observed


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def finite_tree(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return True


def checkpoint_specs(card: dict[str, Any]) -> dict[tuple[str, int], CheckpointSpec]:
    record = card["inputs"]["checkpoint_roster"]
    path = resolve_path(str(record["path"]))
    verify_file(path, str(record["sha256"]))
    specs: dict[tuple[str, int], CheckpointSpec] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["arm"]), int(row["seed"]))
            if key in specs:
                raise RuntimeError(f"Duplicate checkpoint {key}")
            specs[key] = CheckpointSpec(
                arm=key[0],
                seed=key[1],
                checkpoint_step=int(row["checkpoint_step"]),
                path=Path(row["checkpoint_path"]),
                sha256=str(row["checkpoint_sha256"]),
                git_commit=str(row["git_commit"]),
            )
    expected = {
        (str(arm), int(seed))
        for arm in card["roster"]["arms"]
        for seed in card["roster"]["model_seeds"]
    }
    if len(specs) != int(record["rows"]) or set(specs) != expected:
        raise RuntimeError("Checkpoint roster is not the exact paired 20-run panel")
    return specs


def load_architecture_audit(card: dict[str, Any]) -> dict[str, Any]:
    record = card["inputs"]["architecture_audit"]
    path = resolve_path(str(record["path"]))
    verify_file(path, str(record["sha256"]))
    audit = duplicate_safe_json(path)
    if (
        audit.get("status")
        != "passed_capacity_and_forward_path_parity_with_joint_sparse_treatment"
        or audit["configuration_audit"][
            "no_other_paired_scientific_configuration_differences"
        ]
        is not True
    ):
        raise RuntimeError("Pinned architecture audit did not pass")
    return audit


def _load_pinned_model_module(card: dict[str, Any]) -> ModuleType:
    record = card["inputs"]["pinned_model_source"]
    path = Path(record["path"]).resolve()
    digest = verify_file(path, str(record["sha256"]))
    name = f"_early_fate_pinned_model_{digest[:16]}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(str(module.__file__)).resolve() != path:
        raise RuntimeError("Pinned model resolved outside the frozen path")
    return module


def load_model(
    spec: CheckpointSpec,
    card: dict[str, Any],
    audit: dict[str, Any],
    *,
    device: torch.device,
) -> torch.nn.Module:
    verify_file(spec.path, spec.sha256)
    checkpoint = torch_load(spec.path)
    if checkpoint.get("model_family") != "spatial_conv_koopman":
        raise AssertionError("Unexpected checkpoint model family")
    module = _load_pinned_model_module(card)
    config = module.SpatialConvKoopmanConfig.from_mapping(checkpoint["model_config"])
    common = card["model_assertions"]["common"]
    expected = {key: value for key, value in common.items() if key not in {"k_shape", "dropout_modules", "relu_modules"}}
    observed = {key: getattr(config, key) for key in expected}
    if observed != expected:
        raise AssertionError(f"Model configuration drifted: {observed}")
    treatment = card["model_assertions"][spec.arm]
    if (
        str(config.encoder_kind) != str(treatment["encoder_kind"])
        or float(config.lista_alpha) != float(treatment["lista_alpha"])
        or float(checkpoint["loss_weights"]["sparsity"])
        != float(treatment["sparsity_loss_weight"])
    ):
        raise AssertionError(f"Treatment configuration drifted for {spec.arm}")
    model = module.SpatialConvKoopman(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.float().to(device).eval()
    if tuple(model.kmat.shape) != tuple(common["k_shape"]):
        raise AssertionError("Koopman matrix shape drifted")
    if sum(isinstance(item, torch.nn.ReLU) for item in model.modules()) != int(common["relu_modules"]):
        raise AssertionError("ReLU appeared in the dense/sparse matched model")
    if sum(isinstance(item, torch.nn.modules.dropout._DropoutNd) for item in model.modules()) != int(common["dropout_modules"]):
        raise AssertionError("Dropout appeared in the matched model")
    if torch.count_nonzero(model.lista_s.weight).item() != 0:
        raise AssertionError("Inert LISTA recurrence is not exactly zero")
    matching = [
        row for row in audit["runs"]
        if row["arm"] == spec.arm and int(row["seed"]) == spec.seed
    ]
    if len(matching) != 1 or matching[0]["checkpoint_sha256"] != spec.sha256:
        raise AssertionError("Checkpoint is not covered by the architecture audit")
    del checkpoint
    return model


def _select_split_fields(
    path: Path,
    *,
    sha256: str,
    split: str,
    expected_count: int,
    expected_states: int,
    card: dict[str, Any],
) -> torch.Tensor:
    verify_file(path, sha256)
    payload = torch_load(path)
    required = ["fields", "split_indices"]
    forbidden = tuple(card["information_firewall"]["gpu_forbidden_key_fragments"])
    if any(fragment in key.lower() for key in required for fragment in forbidden):
        raise AssertionError("GPU stage requested a forbidden semantic key")
    fields = payload["fields"]
    indices = payload["split_indices"][split]
    selected = fields[indices].detach().cpu().float()
    if tuple(selected.shape) != (expected_count, expected_states, 16, 16, 2):
        raise ValueError(f"Unexpected field shape {tuple(selected.shape)}")
    if not torch.isfinite(selected).all():
        raise FloatingPointError("Dataset fields contain nonfinite values")
    return selected.reshape(expected_count, expected_states, 512).contiguous()


def load_field_roster(card: dict[str, Any]) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, Any]]:
    training = card["inputs"]["training_dataset"]
    train = _select_split_fields(
        Path(training["path"]),
        sha256=str(training["sha256"]),
        split=str(training["split"]),
        expected_count=int(training["trajectories"]),
        expected_states=int(training["stored_states"]),
        card=card,
    )
    record = card["inputs"]["new_ic_dataset_manifest"]
    manifest_path = Path(record["path"])
    verify_file(manifest_path, str(record["sha256"]))
    manifest = duplicate_safe_json(manifest_path)
    observed = manifest.get("datasets", [])
    expected = record["datasets"]
    if len(observed) != len(expected):
        raise RuntimeError("New-IC dataset manifest length drifted")
    test_sets = []
    for item, frozen in zip(observed, expected):
        if (
            int(item["dataset_seed"]) != int(frozen["seed"])
            or str(item["sha256"]) != str(frozen["sha256"])
            or int(item["dataset_seed"]) not in card["roster"]["dataset_seeds"]
        ):
            raise RuntimeError("New-IC dataset record drifted")
        test_sets.append(
            _select_split_fields(
                Path(item["path"]),
                sha256=str(item["sha256"]),
                split=str(record["split"]),
                expected_count=int(record["trajectories_each"]),
                expected_states=int(record["stored_states"]),
                card=card,
            )
        )
    return train, test_sets, manifest


def load_training_labels(card: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Label-aware CPU stage only: return stored labels and final fields."""

    record = card["inputs"]["training_dataset"]
    path = Path(record["path"])
    verify_file(path, str(record["sha256"]))
    payload = torch_load(path)
    indices = payload["split_indices"][record["split"]]
    labels = payload["global_basin_labels"][indices].detach().cpu().long()
    final_fields = payload["fields"][indices, int(card["system"]["final_target_index"])].detach().cpu().float()
    return labels, final_fields


def verify_opened_context(card: dict[str, Any]) -> None:
    for name in ("opened_forecast_context", "architecture_audit"):
        record = card["inputs"][name]
        verify_file(resolve_path(str(record["path"])), str(record["sha256"]))
