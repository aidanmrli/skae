"""Pinned input and checkpoint loading for the Allen--Cahn subspace audit."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


CARD_PATH = Path(__file__).with_name("prediction_card.json")


@dataclass(frozen=True)
class CheckpointSpec:
    arm: str
    seed: int
    path: Path
    sha256: str
    git_commit: str


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_card(path: Path = CARD_PATH) -> tuple[dict[str, Any], str]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in prediction card: {key}")
            result[key] = value
        return result

    card = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    return card, sha256_path(path)


def verify_path(path: Path, expected: str) -> None:
    observed = sha256_path(path)
    if observed != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def checkpoint_roster(card: dict[str, Any]) -> dict[tuple[str, int], CheckpointSpec]:
    record = card["inputs"]["artifact_roster"]
    path = Path(record["path"])
    verify_path(path, str(record["sha256"]))
    roster: dict[tuple[str, int], CheckpointSpec] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row["arm"]), int(row["seed"]))
            if key in roster:
                raise RuntimeError(f"Duplicate checkpoint roster key: {key}")
            roster[key] = CheckpointSpec(
                arm=key[0],
                seed=key[1],
                path=Path(row["checkpoint_path"]),
                sha256=str(row["checkpoint_sha256"]),
                git_commit=str(row["git_commit"]),
            )
    if len(roster) != int(card["validity_gates"]["checkpoint_count"]):
        raise RuntimeError(f"Expected 20 checkpoints, found {len(roster)}")
    expected = {
        (arm, int(seed))
        for arm in card["roster"]["arms"]
        for seed in card["roster"]["model_seeds"]
    }
    if set(roster) != expected:
        raise RuntimeError(f"Checkpoint roster mismatch: {sorted(expected ^ set(roster))}")
    return roster


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_pinned_module(card: dict[str, Any]) -> ModuleType:
    record = card["inputs"]["pinned_model_source"]
    path = Path(record["path"])
    verify_path(path, str(record["sha256"]))
    module_name = "_skae_pinned_spatialized_conv_koopman"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned model source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model(
    spec: CheckpointSpec,
    card: dict[str, Any],
    device: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    verify_path(spec.path, spec.sha256)
    checkpoint = _torch_load(spec.path)
    if checkpoint.get("model_family") != "spatial_conv_koopman":
        raise AssertionError(f"Unexpected model family in {spec.path}")
    module = load_pinned_module(card)
    config = module.SpatialConvKoopmanConfig.from_mapping(checkpoint["model_config"])
    model = module.SpatialConvKoopman(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    audit_model(model, checkpoint, spec, card)
    return model, checkpoint


def audit_model(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    spec: CheckpointSpec,
    card: dict[str, Any],
) -> None:
    arm = spec.arm
    expected = card["model_assertions"]
    cfg = model.cfg
    loss = checkpoint.get("loss_weights", {})
    exact_config = {
        "grid_size": int(expected["grid_size"]),
        "channels": int(expected["channels"]),
        "z_dim": int(card["roster"]["latent_dim"]),
        "hidden_channels": int(expected["hidden_channels"]),
        "num_blocks": int(expected["num_blocks"]),
        "lista_loops": int(expected["lista_loops"]),
        "decoder_kind": str(expected["decoder_kind"]),
        "conv_activation": str(expected["conv_activation"]),
        "dense_activation": str(expected["dense_activation"]),
        "padding_mode": str(expected["padding_mode"]),
    }
    observed_config = {key: getattr(cfg, key) for key in exact_config}
    if observed_config != exact_config:
        raise AssertionError(f"Common model configuration mismatch: {observed_config}")
    if tuple(model.kmat.shape) != tuple(expected["full_K_shape"]):
        raise AssertionError("The Koopman operator is not the frozen full 2048-by-2048 matrix")
    if not isinstance(model.kmat, torch.nn.Parameter) or not model.kmat.requires_grad:
        raise AssertionError("The Koopman matrix is not a full trainable parameter")
    dropout_count = sum(isinstance(module, torch.nn.modules.dropout._DropoutNd)
                        for module in model.modules())
    if dropout_count != int(expected["dropout_module_count"]):
        raise AssertionError(f"Unexpected dropout module count: {dropout_count}")
    relu_count = sum(isinstance(module, torch.nn.ReLU) for module in model.modules())
    if relu_count != int(expected["relu_module_count"]):
        raise AssertionError(f"Unexpected ReLU module count: {relu_count}")
    if torch.count_nonzero(model.lista_s.weight).item() != 0:
        raise AssertionError("lista_s is not exactly inert")
    for name, value in expected["common_loss_weights"].items():
        if float(loss.get(name, float("nan"))) != float(value):
            raise AssertionError(f"Common loss mismatch for {name}")
    if arm == "sparse":
        checks = (
            str(cfg.encoder_kind) == str(expected["sparse_encoder_kind"]),
            float(cfg.lista_alpha) == float(expected["sparse_softshrink_threshold"]),
            float(loss.get("sparsity", float("nan")))
            == float(expected["sparse_elementwise_l1_weight"]),
            str(cfg.encoder_kind) != "lista_signsplit",
        )
    elif arm == "dense":
        checks = (
            str(cfg.encoder_kind) == str(expected["dense_encoder_kind"]),
            float(cfg.lista_alpha) == float(expected["dense_softshrink_threshold"]),
            float(loss.get("sparsity", float("nan")))
            == float(expected["dense_elementwise_l1_weight"]),
        )
    else:
        raise AssertionError(f"Unknown arm {arm}")
    if not all(checks):
        raise AssertionError(f"Model-treatment audit failed for {arm}: {checks}")

    training_args_path = spec.path.parent / "training_args.json"
    training_args = json.loads(training_args_path.read_text())
    training_expected = expected["common_training_args"]
    for name, value in training_expected.items():
        if training_args.get(name) != value:
            raise AssertionError(f"Training argument mismatch for {name}: {training_args.get(name)}")
    treatment_args = expected[f"{arm}_training_args"]
    for name, value in treatment_args.items():
        if training_args.get(name) != value:
            raise AssertionError(f"{arm} treatment argument mismatch for {name}")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        groups = optimizer_state.get("param_groups", [])
        if not groups or any(float(group.get("weight_decay", float("nan")))
                             != float(expected["weight_decay"]) for group in groups):
            raise AssertionError("Checkpoint optimizer weight decay is not exactly zero")

    architecture = load_architecture_audit(card)
    matching = [
        run for run in architecture["runs"]
        if run["arm"] == arm and int(run["seed"]) == int(spec.seed)
    ]
    if len(matching) != 1 or matching[0]["checkpoint_sha256"] != spec.sha256:
        raise AssertionError("Checkpoint is not the one covered by the pinned architecture audit")
    if sha256_path(training_args_path) != matching[0]["training_args_sha256"]:
        raise AssertionError("Training arguments drifted from the pinned architecture audit")
    if int(matching[0]["lista_s_nonzero_count"]) != 0:
        raise AssertionError("Pinned architecture audit reports nonzero LISTA recurrence")


def load_architecture_audit(card: dict[str, Any]) -> dict[str, Any]:
    record = card["inputs"]["architecture_audit"]
    path = Path(record["path"])
    verify_path(path, str(record["sha256"]))
    payload = json.loads(path.read_text())
    if payload.get("status") != "passed_capacity_and_forward_path_parity_with_joint_sparse_treatment":
        raise AssertionError("Pinned architecture audit did not pass")
    if not payload["configuration_audit"]["no_other_paired_scientific_configuration_differences"]:
        raise AssertionError("Architecture audit found an uncontrolled paired difference")
    if payload["configuration_audit"]["common_training_settings"]["optimizer"] != "Adam":
        raise AssertionError("Pinned optimizer is not Adam")
    return payload


def load_reference_forecasts(card: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, float]]:
    record = card["inputs"]["ordinary_forecast_seed_rows"]
    path = Path(record["path"])
    verify_path(path, str(record["sha256"]))
    requested_horizons = set(int(value) for value in card["roster"]["horizons"])
    result: dict[tuple[str, int, int], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            horizon = int(row["horizon"])
            arm = str(row["arm"])
            seed = int(row["seed"])
            if arm in card["roster"]["arms"] and horizon in requested_horizons:
                key = (arm, seed, horizon)
                if key in result:
                    raise RuntimeError(f"Duplicate ordinary forecast row: {key}")
                result[key] = {
                    "field_mse": float(row["field_mse"]),
                    "terminal_field_mse": float(row["final_field_mse"]),
                }
    expected = {
        (arm, int(seed), int(horizon))
        for arm in card["roster"]["arms"]
        for seed in card["roster"]["model_seeds"]
        for horizon in card["roster"]["horizons"]
    }
    if set(result) != expected:
        raise RuntimeError(f"Ordinary forecast reference roster mismatch: {expected ^ set(result)}")
    return result


def load_fields(card: dict[str, Any], input_name: str) -> torch.Tensor:
    record = card["inputs"][input_name]
    path = Path(record["path"])
    verify_path(path, str(record["sha256"]))
    payload = _torch_load(path)
    if not isinstance(payload, dict):
        raise TypeError("Expected a mapping dataset")
    allowed = set(card["information_firewall"]["allowed_dataset_keys"])
    required = {"fields", "split_indices"}
    assert_no_forbidden_mapping_access(sorted(required), card)
    if not required.issubset(allowed):
        raise AssertionError("Prediction card does not whitelist required field keys")
    fields = payload["fields"]
    indices = payload["split_indices"][record["allowed_split"]]
    selected = fields[indices].detach().cpu().to(dtype=torch.float32)
    expected_count = record.get("expected_selected_trajectories")
    if expected_count is not None and int(selected.shape[0]) != int(expected_count):
        raise AssertionError(
            f"Expected {expected_count} selected trajectories, found {selected.shape[0]}"
        )
    if selected.ndim != 5 or selected.shape[-1] != 2:
        raise ValueError(f"Unexpected field shape {tuple(selected.shape)}")
    return selected.reshape(selected.shape[0], selected.shape[1], -1).contiguous()


def assert_no_forbidden_mapping_access(mapping_keys: list[str], card: dict[str, Any]) -> None:
    """Fail tests or callers that try to request a label-valued dataset key."""

    fragments = tuple(card["information_firewall"]["forbidden_key_fragments"])
    bad = [key for key in mapping_keys if any(part in key.lower() for part in fragments)]
    if bad:
        raise AssertionError(f"Forbidden label-valued key request: {bad}")
