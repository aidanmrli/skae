"""Frozen generation, direct Koopman rollout, and full-curve kernels."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from experiments.neurips_2026.allen_cahn_forecast_replication.io import (
    CheckpointSpec,
    load_pinned_module,
    pinned_source,
)


HORIZON = 200


def realized_rng_streams(card: dict[str, Any]) -> dict[str, Any]:
    proof = card["prospective_datasets"]["rng_stream_proof"]
    stride = 10_000
    new_streams = {
        int(seed) + stride * trajectory_index
        for seed in card["prospective_datasets"]["seeds"]
        for trajectory_index in range(256)
    }
    if len(new_streams) != 3 * 256:
        raise AssertionError("Prospective trajectory RNG streams are not unique")
    excluded_streams: set[int] = set()
    excluded_residues: set[int] = set()
    for record in proof["excluded_streams"]:
        seed = int(record["base_seed"])
        start, stop = (int(value) for value in record["indices"])
        excluded_residues.add(seed % stride)
        excluded_streams.update(seed + stride * index for index in range(start, stop + 1))
    new_residues = {int(seed) % stride for seed in card["prospective_datasets"]["seeds"]}
    if len(new_residues) != 3 or new_residues & excluded_residues:
        raise AssertionError("Modular RNG-stream disjointness proof failed")
    if new_streams & excluded_streams:
        raise AssertionError("Prospective and excluded trajectory RNG streams overlap")
    if min(new_streams) < 0 or max(new_streams) >= 2**31:
        raise AssertionError("Prospective RNG stream lies outside the signed-31-bit bound")
    digest_payload = json.dumps(sorted(new_streams), separators=(",", ":")).encode("utf-8")
    return {
        "new_stream_cardinality": len(new_streams),
        "new_stream_minimum": min(new_streams),
        "new_stream_maximum": max(new_streams),
        "new_stream_sha256": hashlib.sha256(digest_payload).hexdigest(),
        "excluded_intersection_empty": True,
        "modular_residue_proof_passed": True,
    }


def _reaction_diffusion_rhs(
    field: torch.Tensor,
    centers: torch.Tensor,
    *,
    beta: float,
    reaction_strength: float,
    diffusion: float,
    laplacian_scale: float,
) -> torch.Tensor:
    center_view = centers.view(1, 1, 1, centers.shape[0], 2)
    differences = field.unsqueeze(-2) - center_view
    squared_distances = differences.square().sum(dim=-1)
    weights = torch.softmax(-float(beta) * squared_distances, dim=-1)
    reaction = -float(reaction_strength) * (
        weights.unsqueeze(-1) * differences
    ).sum(dim=-2)
    laplacian = (
        torch.roll(field, shifts=1, dims=1)
        + torch.roll(field, shifts=-1, dims=1)
        + torch.roll(field, shifts=1, dims=2)
        + torch.roll(field, shifts=-1, dims=2)
        - 4.0 * field
    )
    return reaction + float(diffusion) * float(laplacian_scale) * laplacian


def _rk4_step(
    field: torch.Tensor,
    centers: torch.Tensor,
    card: dict[str, Any],
) -> torch.Tensor:
    config = card["system_and_generator"]
    kwargs = {
        "beta": float(config["allen_cahn_beta"]),
        "reaction_strength": float(config["allen_cahn_reaction_strength"]),
        "diffusion": float(config["diffusion"]),
        "laplacian_scale": float(int(config["grid_size"]) ** 2),
    }
    dt = float(config["rk4_dt"])
    k1 = _reaction_diffusion_rhs(field, centers, **kwargs)
    k2 = _reaction_diffusion_rhs(field + 0.5 * dt * k1, centers, **kwargs)
    k3 = _reaction_diffusion_rhs(field + 0.5 * dt * k2, centers, **kwargs)
    k4 = _reaction_diffusion_rhs(field + dt * k3, centers, **kwargs)
    return field + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@torch.inference_mode()
def generate_all_fields(card: dict[str, Any], *, device: torch.device) -> torch.Tensor:
    """Generate all three datasets together as ``[3,256,201,16,16,2]``."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The frozen prospective generator requires one CUDA GPU")
    realized_rng_streams(card)
    source = load_pinned_module(pinned_source(card, "physics_and_initial_conditions"))
    config = card["system_and_generator"]
    source_config = source.SpatialReactionDiffusionConfig(
        source_system="allen_cahn_4",
        grid_size=int(config["grid_size"]),
        diffusion=float(config["diffusion"]),
        rk4_dt=float(config["rk4_dt"]),
        substeps_per_observation=int(config["substeps_per_observation"]),
        trajectory_length=int(config["trajectory_length"]),
        label_extra_observations=0,
        train_trajectories=0,
        val_trajectories=256,
        test_trajectories=0,
        min_regions=int(config["min_regions"]),
        max_regions=int(config["max_regions"]),
        mask_temperature=float(config["mask_temperature"]),
        low_frequency_cutoff=int(config["low_frequency_cutoff"]),
        noise_scale=float(config["noise_scale"]),
        require_min_area_fraction=float(config["require_min_area_fraction"]),
        allen_cahn_beta=float(config["allen_cahn_beta"]),
        allen_cahn_reaction_strength=float(config["allen_cahn_reaction_strength"]),
        allen_cahn_center_radius=float(config["allen_cahn_center_radius"]),
    )
    centers = source.extract_attractor_centers(
        source.get_source_system("allen_cahn_4", source_config)
    ).to(dtype=torch.float32)
    initial_fields: list[torch.Tensor] = []
    for dataset_seed in card["prospective_datasets"]["seeds"]:
        dataset_fields: list[torch.Tensor] = []
        for trajectory_index in range(256):
            generator = torch.Generator().manual_seed(
                int(dataset_seed) + 10_000 * trajectory_index
            )
            field, internal_selected_identities, internal_region_areas = source._sample_initial_condition(
                centers.cpu(), source_config, generator
            )
            del internal_selected_identities, internal_region_areas
            dataset_fields.append(field.to(dtype=torch.float32))
        initial_fields.append(torch.stack(dataset_fields))
    initial = torch.stack(initial_fields)
    expected_initial = (3, 256, 16, 16, 2)
    if tuple(initial.shape) != expected_initial or initial.dtype != torch.float32:
        raise AssertionError(f"Unexpected packed initial fields: {tuple(initial.shape)}")
    current = initial.reshape(3 * 256, 16, 16, 2).to(device)
    device_centers = centers.to(device)
    frames = torch.empty(
        (3 * 256, HORIZON + 1, 16, 16, 2),
        dtype=torch.float32,
        device=device,
    )
    frames[:, 0] = current
    with torch.autocast(device_type="cuda", enabled=False):
        for observation in range(1, HORIZON + 1):
            for _ in range(int(config["substeps_per_observation"])):
                current = _rk4_step(current, device_centers, card)
            frames[:, observation] = current
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(frames).all()):
        raise FloatingPointError("Prospective generator produced a nonfinite field")
    if bool((frames.abs() > 8.0).any()):
        raise FloatingPointError("Prospective generator exceeded the frozen magnitude bound")
    return frames.reshape(3, 256, HORIZON + 1, 16, 16, 2).cpu().contiguous()


@torch.inference_mode()
def direct_rollout(model: torch.nn.Module, x0: torch.Tensor, *, horizon: int) -> torch.Tensor:
    """Exactly one encode followed by ``F.linear(z, kmat)`` and one batched decode."""

    if x0.dtype != torch.float32 or any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise AssertionError("Direct rollout requires float32 fields and model parameters")
    device_type = x0.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        latent = model.encode(x0)
        latent_steps: list[torch.Tensor] = []
        for _ in range(int(horizon)):
            latent = F.linear(latent, model.kmat)
            latent_steps.append(latent)
        stacked = torch.stack(latent_steps, dim=1)
        predictions = model.decode(stacked.reshape(-1, stacked.shape[-1])).reshape(
            x0.shape[0], int(horizon), -1
        )
    if not bool(torch.isfinite(predictions).all()):
        raise FloatingPointError("Direct repeated-K rollout contains a nonfinite value")
    return predictions


def curve_metrics(
    predictions: torch.Tensor,
    truth: torch.Tensor,
    x0: torch.Tensor,
) -> dict[str, list[float]]:
    if predictions.shape != truth.shape or predictions.ndim != 3:
        raise ValueError("Prediction and truth must be equal [trajectory,horizon,state] tensors")
    if predictions.shape[1] != HORIZON or x0.shape != predictions[:, 0].shape:
        raise ValueError("Curve kernel requires the exact H200 truth and x0 shapes")
    tensors = (predictions, truth, x0)
    if any(value.dtype != torch.float32 for value in tensors):
        raise AssertionError("Curve kernel requires float32 tensors")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("Curve input contains a nonfinite value")
    instantaneous = (predictions - truth).square().mean(dim=(0, 2)).to(torch.float64)
    persistence = (x0[:, None, :] - truth).square().mean(dim=(0, 2)).to(torch.float64)
    denominator = torch.arange(1, HORIZON + 1, dtype=torch.float64, device=truth.device)
    cumulative = instantaneous.cumsum(dim=0) / denominator
    persistence_cumulative = persistence.cumsum(dim=0) / denominator
    instantaneous_ratio = instantaneous / persistence
    cumulative_ratio = cumulative / persistence_cumulative
    values = {
        "instantaneous_field_mse": instantaneous,
        "cumulative_field_mse": cumulative,
        "instantaneous_persistence_mse": persistence,
        "cumulative_persistence_mse": persistence_cumulative,
        "instantaneous_model_over_persistence": instantaneous_ratio,
        "cumulative_model_over_persistence": cumulative_ratio,
    }
    if any(not bool(torch.isfinite(value).all()) for value in values.values()):
        raise FloatingPointError("Curve metric or unclipped ratio is nonfinite")
    return {name: value.detach().cpu().tolist() for name, value in values.items()}


def validate_curve_record(record: dict[str, Any]) -> None:
    names = (
        "instantaneous_field_mse",
        "cumulative_field_mse",
        "instantaneous_persistence_mse",
        "cumulative_persistence_mse",
        "instantaneous_model_over_persistence",
        "cumulative_model_over_persistence",
    )
    if any(name not in record for name in names):
        raise KeyError("Curve record is missing a frozen curve")
    tensors = {name: torch.as_tensor(record[name], dtype=torch.float64) for name in names}
    if any(tuple(value.shape) != (HORIZON,) for value in tensors.values()):
        raise AssertionError("Every frozen curve must contain all h=1..200 points")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise FloatingPointError("Stored curve contains a nonfinite value")
    horizon = torch.arange(1, HORIZON + 1, dtype=torch.float64)
    torch.testing.assert_close(
        tensors["cumulative_field_mse"],
        tensors["instantaneous_field_mse"].cumsum(0) / horizon,
        rtol=1e-12,
        atol=1e-14,
    )
    torch.testing.assert_close(
        tensors["cumulative_persistence_mse"],
        tensors["instantaneous_persistence_mse"].cumsum(0) / horizon,
        rtol=1e-12,
        atol=1e-14,
    )
    torch.testing.assert_close(
        tensors["instantaneous_model_over_persistence"],
        tensors["instantaneous_field_mse"] / tensors["instantaneous_persistence_mse"],
        rtol=1e-12,
        atol=1e-14,
    )
    torch.testing.assert_close(
        tensors["cumulative_model_over_persistence"],
        tensors["cumulative_field_mse"] / tensors["cumulative_persistence_mse"],
        rtol=1e-12,
        atol=1e-14,
    )


@torch.inference_mode()
def evaluate_model_packed(
    model: torch.nn.Module,
    fields: torch.Tensor,
    *,
    batch_size: int = 256,
) -> list[dict[str, list[float]]]:
    """Score packed ``[dataset,trajectory,201,state]`` fields without reencoding."""

    if fields.ndim != 4 or tuple(fields.shape[:3]) != (3, 256, HORIZON + 1):
        raise ValueError(f"Unexpected packed scoring fields {tuple(fields.shape)}")
    if fields.dtype != torch.float32 or fields.device != next(model.parameters()).device:
        raise AssertionError("Packed fields must be float32 on the model device")
    flat = fields.reshape(3 * 256, HORIZON + 1, fields.shape[-1])
    prediction_chunks = []
    for start in range(0, flat.shape[0], int(batch_size)):
        stop = min(flat.shape[0], start + int(batch_size))
        prediction_chunks.append(direct_rollout(model, flat[start:stop, 0], horizon=HORIZON))
    predictions = torch.cat(prediction_chunks).reshape(3, 256, HORIZON, fields.shape[-1])
    results = []
    for dataset_index in range(3):
        results.append(
            curve_metrics(
                predictions[dataset_index],
                fields[dataset_index, :, 1:],
                fields[dataset_index, :, 0],
            )
        )
    return results


@torch.inference_mode()
def evaluate_model_sequential(
    model: torch.nn.Module,
    fields: torch.Tensor,
) -> list[dict[str, list[float]]]:
    """Synthetic reference: score each dataset separately with the same kernels."""

    results = []
    for dataset_index in range(fields.shape[0]):
        predictions = direct_rollout(model, fields[dataset_index, :, 0], horizon=HORIZON)
        results.append(
            curve_metrics(
                predictions,
                fields[dataset_index, :, 1:],
                fields[dataset_index, :, 0],
            )
        )
    return results


def crossed_rows(
    specs_and_models: Sequence[tuple[CheckpointSpec, torch.nn.Module]],
    fields: torch.Tensor,
    card: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_seeds = [int(value) for value in card["prospective_datasets"]["seeds"]]
    for spec, model in specs_and_models:
        per_dataset = evaluate_model_packed(model, fields, batch_size=256)
        for dataset_index, curves in enumerate(per_dataset):
            row: dict[str, Any] = {
                "arm": spec.arm,
                "model_seed": int(spec.seed),
                "dataset_index": dataset_index,
                "dataset_seed": dataset_seeds[dataset_index],
                "checkpoint_step": int(spec.checkpoint_step),
                "checkpoint_sha256": spec.sha256,
                **curves,
            }
            validate_curve_record(row)
            rows.append(row)
    validate_crossed_rows(rows, card)
    return rows


def validate_crossed_rows(rows: Iterable[dict[str, Any]], card: dict[str, Any]) -> None:
    materialized = list(rows)
    expected = {
        (arm, int(model_seed), int(dataset_seed))
        for arm in card["checkpoint_roster"]["arms"]
        for model_seed in card["checkpoint_roster"]["model_seeds"]
        for dataset_seed in card["prospective_datasets"]["seeds"]
    }
    actual = {
        (str(row["arm"]), int(row["model_seed"]), int(row["dataset_seed"]))
        for row in materialized
    }
    if len(materialized) != 60 or len(actual) != 60 or actual != expected:
        raise AssertionError("Scientific payload is not the exact 20x3 crossed roster")
    for row in materialized:
        validate_curve_record(row)
