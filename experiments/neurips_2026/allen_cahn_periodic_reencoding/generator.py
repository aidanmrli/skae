"""Frozen, label-free Allen--Cahn field generation for periodic evaluation."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from experiments.neurips_2026.allen_cahn_forecast_replication import io as parent_io
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import load_parent_card


def realized_rng_streams(card: dict[str, Any]) -> dict[str, Any]:
    stride = 10_000
    role_seeds = {
        role: [int(record["seed"]) for record in card["prospective_datasets"][role]]
        for role in ("validation", "test")
    }
    all_seeds = role_seeds["validation"] + role_seeds["test"]
    streams = {
        seed + stride * trajectory_index
        for seed in all_seeds
        for trajectory_index in range(256)
    }
    parent = load_parent_card(card)
    parent_exclusions = parent["prospective_datasets"]["rng_stream_proof"][
        "excluded_streams"
    ]
    excluded: set[int] = set()
    for record in parent_exclusions:
        start, stop = (int(value) for value in record["indices"])
        excluded.update(
            int(record["base_seed"]) + stride * index
            for index in range(start, stop + 1)
        )
    parent_bases = {int(record["base_seed"]) for record in parent_exclusions}
    for seed in card["prospective_datasets"]["excluded"]:
        if int(seed) not in parent_bases:
            excluded.update(int(seed) + stride * index for index in range(256))
    residues = {seed % stride for seed in all_seeds}
    excluded_residues = {
        int(seed) % stride for seed in card["prospective_datasets"]["excluded"]
    }
    if len(all_seeds) != 6 or len(set(all_seeds)) != 6:
        raise AssertionError("Expected six distinct prospective base seeds")
    if len(streams) != 6 * 256 or streams & excluded:
        raise AssertionError("Prospective RNG streams collide")
    if len(residues) != 6 or residues & excluded_residues:
        raise AssertionError("Modular RNG-stream disjointness failed")
    if min(streams) < 0 or max(streams) >= 2**31:
        raise AssertionError("Prospective RNG stream exceeds signed-31-bit range")
    return {
        "stream_count": len(streams),
        "minimum": min(streams),
        "maximum": max(streams),
        "residues_mod_10000": sorted(residues),
        "excluded_intersection_empty": True,
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
    system = card["system"]
    parameters = system["generator_parameters"]
    kwargs = {
        "beta": float(parameters["allen_cahn_beta"]),
        "reaction_strength": float(parameters["allen_cahn_reaction_strength"]),
        "diffusion": float(parameters["diffusion"]),
        "laplacian_scale": float(int(system["grid_size"]) ** 2),
    }
    dt = float(system["rk4_dt"])
    k1 = _reaction_diffusion_rhs(field, centers, **kwargs)
    k2 = _reaction_diffusion_rhs(field + 0.5 * dt * k1, centers, **kwargs)
    k3 = _reaction_diffusion_rhs(field + 0.5 * dt * k2, centers, **kwargs)
    k4 = _reaction_diffusion_rhs(field + dt * k3, centers, **kwargs)
    return field + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def sample_initial_fields(
    card: dict[str, Any],
    *,
    seeds: Sequence[int],
    horizon: int,
) -> torch.Tensor:
    """Sample the complete initial-condition panel on CPU without future fields."""

    if int(horizon) not in {200, 400} or len(seeds) != 3:
        raise ValueError("Expected exactly three H200 or H400 datasets")
    realized_rng_streams(card)
    parent = load_parent_card(card)
    source_record = parent_io.pinned_source(parent, "physics_and_initial_conditions")
    source = parent_io.load_pinned_module(source_record)
    system = card["system"]
    parameters = system["generator_parameters"]
    source_config = source.SpatialReactionDiffusionConfig(
        source_system=str(parameters["source_system"]),
        grid_size=int(system["grid_size"]),
        diffusion=float(parameters["diffusion"]),
        rk4_dt=float(system["rk4_dt"]),
        substeps_per_observation=int(system["substeps_per_observation"]),
        trajectory_length=int(horizon),
        label_extra_observations=0,
        train_trajectories=0,
        val_trajectories=256,
        test_trajectories=0,
        min_regions=int(parameters["min_regions"]),
        max_regions=int(parameters["max_regions"]),
        mask_temperature=float(parameters["mask_temperature"]),
        low_frequency_cutoff=int(parameters["low_frequency_cutoff"]),
        noise_scale=float(parameters["noise_scale"]),
        require_min_area_fraction=float(parameters["require_min_area_fraction"]),
        allen_cahn_beta=float(parameters["allen_cahn_beta"]),
        allen_cahn_reaction_strength=float(parameters["allen_cahn_reaction_strength"]),
        allen_cahn_center_radius=float(parameters["allen_cahn_center_radius"]),
    )
    centers = source.extract_attractor_centers(
        source.get_source_system(str(parameters["source_system"]), source_config)
    ).to(dtype=torch.float32)
    initial_datasets: list[torch.Tensor] = []
    for dataset_seed in seeds:
        trajectories: list[torch.Tensor] = []
        for trajectory_index in range(256):
            generator = torch.Generator().manual_seed(
                int(dataset_seed) + 10_000 * trajectory_index
            )
            field, internal_identities, internal_areas = source._sample_initial_condition(
                centers.cpu(), source_config, generator
            )
            del internal_identities, internal_areas
            trajectories.append(field.to(dtype=torch.float32))
        initial_datasets.append(torch.stack(trajectories))
    initial = torch.stack(initial_datasets)
    if tuple(initial.shape) != (3, 256, 16, 16, 2):
        raise AssertionError(f"Unexpected initial field shape {tuple(initial.shape)}")
    if initial.device.type != "cpu" or initial.dtype != torch.float32:
        raise AssertionError("Initial fields must be CPU float32")
    if not bool(torch.isfinite(initial).all()):
        raise FloatingPointError("Initial-condition sampler produced a nonfinite field")
    return initial.contiguous()


@torch.inference_mode()
def integrate_initial_fields(
    card: dict[str, Any],
    initial: torch.Tensor,
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    """Integrate an already sampled three-dataset panel entirely on CUDA."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Prospective field integration requires CUDA")
    if int(horizon) not in {200, 400}:
        raise ValueError("Field integration supports only frozen H200/H400 horizons")
    if tuple(initial.shape) != (3, 256, 16, 16, 2):
        raise ValueError("Initial panel shape drifted")
    parent = load_parent_card(card)
    source = parent_io.load_pinned_module(
        parent_io.pinned_source(parent, "physics_and_initial_conditions")
    )
    system = card["system"]
    parameters = system["generator_parameters"]
    source_config = source.SpatialReactionDiffusionConfig(
        source_system=str(parameters["source_system"]),
        grid_size=int(system["grid_size"]),
        diffusion=float(parameters["diffusion"]),
        rk4_dt=float(system["rk4_dt"]),
        substeps_per_observation=int(system["substeps_per_observation"]),
        trajectory_length=int(horizon),
        label_extra_observations=0,
        train_trajectories=0,
        val_trajectories=256,
        test_trajectories=0,
        min_regions=int(parameters["min_regions"]),
        max_regions=int(parameters["max_regions"]),
        mask_temperature=float(parameters["mask_temperature"]),
        low_frequency_cutoff=int(parameters["low_frequency_cutoff"]),
        noise_scale=float(parameters["noise_scale"]),
        require_min_area_fraction=float(parameters["require_min_area_fraction"]),
        allen_cahn_beta=float(parameters["allen_cahn_beta"]),
        allen_cahn_reaction_strength=float(parameters["allen_cahn_reaction_strength"]),
        allen_cahn_center_radius=float(parameters["allen_cahn_center_radius"]),
    )
    centers = source.extract_attractor_centers(
        source.get_source_system(str(parameters["source_system"]), source_config)
    ).to(device=device, dtype=torch.float32)
    current = initial.reshape(3 * 256, 16, 16, 2).to(device)
    frames = torch.empty(
        (3 * 256, int(horizon) + 1, 16, 16, 2),
        device=device,
        dtype=torch.float32,
    )
    frames[:, 0] = current
    with torch.autocast(device_type="cuda", enabled=False):
        for observation in range(1, int(horizon) + 1):
            for _ in range(int(system["substeps_per_observation"])):
                current = _rk4_step(current, centers, card)
            frames[:, observation] = current
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(frames).all()):
        raise FloatingPointError("Prospective generator produced a nonfinite field")
    if bool((frames.abs() > 8.0).any()):
        raise FloatingPointError("Prospective generator exceeded the magnitude bound")
    return frames.reshape(3, 256, int(horizon) + 1, 16, 16, 2).contiguous()


def generate_fields(
    card: dict[str, Any],
    *,
    seeds: Sequence[int],
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample on CPU and integrate on CUDA; callers may split these stages."""

    initial = sample_initial_fields(card, seeds=seeds, horizon=horizon)
    return integrate_initial_fields(card, initial, horizon=horizon, device=device)
