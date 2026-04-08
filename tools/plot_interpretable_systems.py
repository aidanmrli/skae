#!/usr/bin/env python3
"""Plot the three interpretable transition-rich toy systems."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_SYSTEM_KEYS = (
    "multiwell_strong_transition",
    "gated_local_linear",
    "gated_transfer_linear",
)
DEFAULT_FORMATS = ("png", "svg", "pdf")
DEFAULT_LAYOUT = "overview"


def integrate_rk4(
    x: torch.Tensor,
    dt: float,
    dynamics_fn,
) -> torch.Tensor:
    """Vectorized RK4 step."""

    k1 = dynamics_fn(x)
    k2 = dynamics_fn(x + 0.5 * dt * k1)
    k3 = dynamics_fn(x + 0.5 * dt * k2)
    k4 = dynamics_fn(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def parse_systems_arg(value: str | None) -> list[str]:
    if value is None or value.strip() == "" or value.strip().lower() == "all":
        return list(DEFAULT_SYSTEM_KEYS)
    systems = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(systems) - set(DEFAULT_SYSTEM_KEYS))
    if unknown:
        raise ValueError(f"Unknown systems: {unknown}. Expected subset of {DEFAULT_SYSTEM_KEYS}.")
    return systems


def parse_formats_arg(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return DEFAULT_FORMATS
    formats = []
    for item in value.split(","):
        cleaned = item.strip().lower().lstrip(".")
        if cleaned:
            formats.append(cleaned)
    allowed = {"png", "svg", "pdf"}
    unknown = sorted(set(formats) - allowed)
    if unknown:
        raise ValueError(f"Unknown formats: {unknown}. Expected subset of {sorted(allowed)}.")
    return tuple(formats)


def validate_layout(value: str) -> str:
    cleaned = value.strip().lower()
    allowed = {"overview", "catalog"}
    if cleaned not in allowed:
        raise ValueError(f"Unknown layout: {value!r}. Expected one of {sorted(allowed)}.")
    return cleaned


def _as_batch(state: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if state.ndim == 1:
        return state.unsqueeze(0), True
    return state, False


def _deterministic_grid(limit: float, points_per_axis: int) -> torch.Tensor:
    coords = torch.linspace(-float(limit), float(limit), steps=int(points_per_axis))
    mesh_x, mesh_y = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=1)


class PlotSystem(Protocol):
    key: str
    title: str
    init_range: float

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        ...

    def step(self, state: torch.Tensor) -> torch.Tensor:
        ...

    def region_label(self, state: torch.Tensor) -> torch.Tensor:
        ...

    def basin_label(self, state: torch.Tensor) -> torch.Tensor:
        ...

    def trajectory_starts(self, points_per_axis: int) -> torch.Tensor:
        ...

    def plot_guides(self, ax) -> None:
        ...


@dataclass
class MultiwellStrongTransitionSystem:
    key: str = "multiwell_strong_transition"
    title: str = "Multiwell Strong Transition"
    dt: float = 0.02
    sigma: float = 0.7
    init_range: float = 2.5
    core_radius: float = 0.7
    center_core_radius: float = 0.7
    transition_radius: float = 1.35
    strong_alpha: float = 0.8
    strong_beta: float = 1.0

    def __post_init__(self) -> None:
        self.points_2d = torch.tensor(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        self.num_basins = int(self.points_2d.shape[0])
        self.center_basin_index = int(torch.norm(self.points_2d, dim=1).argmin().item())
        self.core_radii = torch.full((self.num_basins,), float(self.core_radius))
        self.core_radii[self.center_basin_index] = float(self.center_core_radius)
        self._sigma2 = float(self.sigma) * float(self.sigma)

    @staticmethod
    def _rot90(vec: torch.Tensor) -> torch.Tensor:
        rot = torch.zeros_like(vec)
        rot[..., 0] = -vec[..., 1]
        rot[..., 1] = vec[..., 0]
        return rot

    def _potential_gradient(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch(state)
        diff = batch.unsqueeze(1) - self.points_2d.unsqueeze(0)
        r2 = (diff * diff).sum(dim=-1)
        weights = torch.exp(-r2 / self._sigma2)
        grad = 2.0 * (diff * weights.unsqueeze(-1)).sum(dim=1)
        return grad[0] if squeezed else grad

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch(state)
        grad = self._potential_gradient(batch)
        radius = torch.norm(batch, dim=-1, keepdim=True)
        velocity = (
            -grad
            + self.strong_alpha * self._rot90(grad)
            + self.strong_beta * torch.cos(2.0 * radius) * batch
        )
        return velocity[0] if squeezed else velocity

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return integrate_rk4(state, self.dt, self.dynamics)

    def basin_label(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch(state)
        diff = batch.unsqueeze(1) - self.points_2d.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=-1)
        labels = dist_sq.argmin(dim=-1).to(dtype=torch.long)
        return labels[0] if squeezed else labels

    def region_label(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch(state)
        diff = batch.unsqueeze(1) - self.points_2d.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=-1)
        nearest = dist_sq.argmin(dim=-1).to(dtype=torch.long)
        min_dist = dist_sq.min(dim=-1).values.sqrt()
        nearest_core_radius = self.core_radii[nearest]
        in_core = min_dist <= nearest_core_radius
        in_transition_corridor = torch.norm(batch, dim=-1) <= float(self.transition_radius)
        gate_label = torch.full_like(nearest, int(self.num_basins))
        labels = torch.where(in_core, nearest, torch.where(in_transition_corridor, gate_label, nearest))
        return labels[0] if squeezed else labels

    def trajectory_starts(self, points_per_axis: int) -> torch.Tensor:
        return _deterministic_grid(limit=float(self.init_range) * 0.9, points_per_axis=points_per_axis)

    def plot_guides(self, ax) -> None:
        points = self.points_2d.detach().cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], color="black", marker="x", s=70, linewidths=1.8, zorder=4)
        for index, point in enumerate(points):
            radius = float(self.core_radii[index].item())
            ax.add_patch(
                Circle((float(point[0]), float(point[1])), radius, fill=False, linestyle=":", linewidth=1.0, edgecolor="#0f172a", alpha=0.8)
            )
        ax.add_patch(
            Circle((0.0, 0.0), float(self.transition_radius), fill=False, linestyle="--", linewidth=1.0, edgecolor="#475569", alpha=0.6)
        )


@dataclass
class GatedLocalLinearStandaloneSystem:
    key: str = "gated_local_linear"
    title: str = "Gated Local Linear"
    dt: float = 0.04
    num_basins: int = 3
    center_radius: float = 1.75
    basin_radius: float = 1.05
    init_range: float = 2.6
    gate_contraction: float = 1.35
    gate_swirl: float = 0.9

    def __post_init__(self) -> None:
        self.center_angles = torch.linspace(0.0, 2.0 * torch.pi, steps=self.num_basins + 1, dtype=torch.float32)[:-1]
        self.points_2d = self._build_centers(radius=self.center_radius)
        self.basin_matrices = self._build_basin_matrices()
        self.gate_matrices = self._build_gate_matrices()

    def _build_centers(self, radius: float) -> torch.Tensor:
        return torch.stack([radius * torch.cos(self.center_angles), radius * torch.sin(self.center_angles)], dim=1)

    @staticmethod
    def _rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
        c = torch.cos(angle)
        s = torch.sin(angle)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])], dim=0)

    def _build_basin_matrices(self) -> torch.Tensor:
        templates = torch.stack(
            [
                torch.tensor([[-0.9, -1.2], [1.2, -0.9]], dtype=torch.float32),
                torch.tensor([[-1.35, 0.2], [-0.3, -0.7]], dtype=torch.float32),
                torch.tensor([[-0.7, -0.1], [0.5, -1.2]], dtype=torch.float32),
            ],
            dim=0,
        )
        matrices = []
        for basin_index in range(self.num_basins):
            base = templates[basin_index % templates.shape[0]]
            rot = self._rotation_matrix(self.center_angles[basin_index])
            matrices.append(rot @ base @ rot.transpose(0, 1))
        return torch.stack(matrices, dim=0)

    def _build_gate_matrices(self) -> torch.Tensor:
        base = torch.tensor(
            [
                [-self.gate_contraction, -self.gate_swirl],
                [self.gate_swirl, -self.gate_contraction],
            ],
            dtype=torch.float32,
        )
        return torch.stack([base.clone() for _ in range(self.num_basins)], dim=0)

    def _sector_index(self, state_2d: torch.Tensor) -> torch.Tensor:
        angles = torch.atan2(state_2d[..., 1], state_2d[..., 0])
        deltas = torch.remainder(angles.unsqueeze(-1) - self.center_angles + torch.pi, 2.0 * torch.pi) - torch.pi
        return deltas.abs().argmin(dim=-1).to(dtype=torch.long)

    def region_label(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch(state)
        diff = batch.unsqueeze(1) - self.points_2d.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        nearest_basin = dist.argmin(dim=-1).to(dtype=torch.long)
        in_basin = dist.min(dim=-1).values <= self.basin_radius
        gate_sector = self._sector_index(batch)
        labels = torch.where(in_basin, nearest_basin, self.num_basins + gate_sector)
        return labels[0] if squeezed else labels

    def basin_label(self, state: torch.Tensor) -> torch.Tensor:
        region = self.region_label(state)
        if isinstance(region, torch.Tensor) and region.ndim == 0:
            return region if int(region.item()) < self.num_basins else region - self.num_basins
        return torch.where(region < self.num_basins, region, region - self.num_basins)

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch(state)
        basin = self.basin_label(batch)
        region = self.region_label(batch)
        centers = self.points_2d[basin]
        delta = batch - centers
        basin_mats = self.basin_matrices[basin]
        gate_mats = self.gate_matrices[basin]
        matrices = torch.where((region < self.num_basins)[..., None, None], basin_mats, gate_mats)
        velocity = torch.einsum("...ij,...j->...i", matrices, delta)
        return velocity[0] if squeezed else velocity

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return integrate_rk4(state, self.dt, self.dynamics)

    def trajectory_starts(self, points_per_axis: int) -> torch.Tensor:
        return _deterministic_grid(limit=float(self.init_range) * 0.9, points_per_axis=points_per_axis)

    def plot_guides(self, ax) -> None:
        points = self.points_2d.detach().cpu().numpy()
        ax.scatter(points[:, 0], points[:, 1], color="black", marker="x", s=70, linewidths=1.8, zorder=4)
        for point in points:
            ax.add_patch(
                Circle((float(point[0]), float(point[1])), float(self.basin_radius), fill=False, linestyle="--", linewidth=1.0, edgecolor="#475569", alpha=0.6)
            )


@dataclass
class GatedTransferLinearStandaloneSystem:
    key: str = "gated_transfer_linear"
    title: str = "Gated Transfer Linear"
    dt: float = 0.04
    num_basins: int = 3
    center_radius: float = 1.85
    center_phase: float = 0.0
    core_radius: float = 0.30
    source_radius: float = 0.80
    handoff_radius: float = 0.45
    exit_min_radius: float = 0.60
    channel_half_width: float = 0.22
    channel_lane_offset: float = 0.28
    exit_half_angle: float = 0.72
    init_range: float = 2.8
    return_rate_scale: float = 0.65
    background_rate_scale: float = 0.50
    exit_forward_rate: float = 1.0
    exit_transverse_rate: float = 1.8
    exit_handoff_offset: float = 0.40
    channel_speed: float = 1.55
    channel_transverse_contraction: float = 2.8

    def __post_init__(self) -> None:
        self.center_angles = torch.linspace(0.0, 2.0 * torch.pi, steps=self.num_basins + 1, dtype=torch.float32)[:-1] + self.center_phase
        self.points_2d = self._build_centers(radius=self.center_radius)
        self.outward_directions = self._normalize(self.points_2d)
        self.ordered_pairs = [(source, dest) for source in range(self.num_basins) for dest in range(self.num_basins) if dest != source]
        self.pair_to_index = {pair: index for index, pair in enumerate(self.ordered_pairs)}
        self.source_pair_indices_by_basin = [
            torch.tensor([self.pair_to_index[(source, dest)] for dest in range(self.num_basins) if dest != source], dtype=torch.long)
            for source in range(self.num_basins)
        ]
        self.exit_directions_2d = torch.stack(
            [self._normalize((self.points_2d[dest] - self.points_2d[source]).unsqueeze(0))[0] for source, dest in self.ordered_pairs],
            dim=0,
        )
        self.exit_normals_2d = self._rot90(self.exit_directions_2d)
        (
            self.channel_directions_2d,
            self.channel_normals_2d,
            self.channel_entry_points_2d,
            self.channel_handoff_points_2d,
            self.channel_handoff_targets_2d,
            self.channel_lengths,
            self.channel_destination_basins,
        ) = self._build_channels()
        self.basin_matrices = self._build_basin_matrices()
        self.return_matrices = self.return_rate_scale * self.basin_matrices
        self.background_matrices = self.background_rate_scale * self.basin_matrices
        self.core_offset = 0
        self.return_offset = self.num_basins
        self.exit_offset = self.return_offset + self.num_basins
        self.channel_offset = self.exit_offset + len(self.ordered_pairs)
        self.num_regions = self.channel_offset + len(self.ordered_pairs)

    def _build_centers(self, radius: float) -> torch.Tensor:
        return torch.stack([radius * torch.cos(self.center_angles), radius * torch.sin(self.center_angles)], dim=1)

    @staticmethod
    def _normalize(vectors: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)
        return vectors / norms

    @staticmethod
    def _rot90(vec: torch.Tensor) -> torch.Tensor:
        rot = torch.zeros_like(vec)
        rot[..., 0] = -vec[..., 1]
        rot[..., 1] = vec[..., 0]
        return rot

    @staticmethod
    def _rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
        c = torch.cos(angle)
        s = torch.sin(angle)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])], dim=0)

    def _build_basin_matrices(self) -> torch.Tensor:
        templates = torch.stack(
            [
                torch.tensor([[-1.0, -1.1], [1.1, -1.0]], dtype=torch.float32),
                torch.tensor([[-1.4, 0.2], [-0.2, -0.7]], dtype=torch.float32),
                torch.tensor([[-0.8, -0.3], [0.5, -1.3]], dtype=torch.float32),
            ],
            dim=0,
        )
        matrices = []
        for basin_index in range(self.num_basins):
            base = templates[basin_index % templates.shape[0]]
            rot = self._rotation_matrix(self.center_angles[basin_index])
            matrices.append(rot @ base @ rot.transpose(0, 1))
        return torch.stack(matrices, dim=0)

    def _build_channels(self):
        directions = []
        normals = []
        entry_points = []
        handoff_points = []
        handoff_targets = []
        lengths = []
        destination_basins = []
        for pair_index, (source, dest) in enumerate(self.ordered_pairs):
            source_to_dest = self.exit_directions_2d[pair_index]
            source_exit_normal = self.exit_normals_2d[pair_index]
            destination_outward = self.outward_directions[dest]
            destination_tangent = self._rot90(destination_outward)
            incoming_delta = float(torch.sin(self.center_angles[source] - self.center_angles[dest]))
            tangent_sign = 1.0 if incoming_delta >= 0.0 else -1.0

            entry = (
                self.points_2d[source]
                + self.source_radius * source_to_dest
                + (tangent_sign * self.channel_lane_offset) * source_exit_normal
            )
            handoff = (
                self.points_2d[dest]
                + self.handoff_radius * destination_outward
                + (tangent_sign * self.channel_lane_offset) * destination_tangent
            )
            direction = self._normalize((handoff - entry).unsqueeze(0))[0]
            normal = self._rot90(direction)
            handoff_target = entry + self.exit_handoff_offset * direction
            directions.append(direction)
            normals.append(normal)
            entry_points.append(entry)
            handoff_points.append(handoff)
            handoff_targets.append(handoff_target)
            lengths.append(torch.dot(direction, handoff - entry))
            destination_basins.append(dest)
        return (
            torch.stack(directions, dim=0),
            torch.stack(normals, dim=0),
            torch.stack(entry_points, dim=0),
            torch.stack(handoff_points, dim=0),
            torch.stack(handoff_targets, dim=0),
            torch.stack(lengths, dim=0),
            torch.tensor(destination_basins, dtype=torch.long),
        )

    def _nearest_basin(self, batch: torch.Tensor):
        diff = batch.unsqueeze(1) - self.points_2d.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        nearest_dist, nearest_basin = dist.min(dim=1)
        return nearest_basin.to(dtype=torch.long), nearest_dist

    def _channel_membership(self, batch: torch.Tensor):
        rel = batch.unsqueeze(1) - self.channel_entry_points_2d.unsqueeze(0)
        longitudinal = (rel * self.channel_directions_2d.unsqueeze(0)).sum(dim=-1)
        transverse = (rel * self.channel_normals_2d.unsqueeze(0)).sum(dim=-1)
        mask = (
            (longitudinal >= 0.0)
            & (longitudinal <= self.channel_lengths.unsqueeze(0))
            & (torch.abs(transverse) <= self.channel_half_width)
        )
        has_channel = mask.any(dim=1)
        masked_distance = torch.where(mask, torch.abs(transverse), torch.full_like(transverse, float("inf")))
        closest_match = masked_distance.argmin(dim=1)
        channel_index = torch.where(has_channel, closest_match.to(dtype=torch.long), torch.full_like(closest_match, -1, dtype=torch.long))
        return channel_index, longitudinal, transverse

    def _source_neighborhood_label_flat(self, batch: torch.Tensor, nearest_basin: torch.Tensor, nearest_dist: torch.Tensor, channel_index: torch.Tensor | None = None):
        if channel_index is None:
            channel_index, _, _ = self._channel_membership(batch)
        return torch.where((nearest_dist <= self.source_radius) & (channel_index < 0), nearest_basin, torch.full_like(nearest_basin, -1))

    def _core_basin_label_flat(self, nearest_basin: torch.Tensor, nearest_dist: torch.Tensor):
        return torch.where(nearest_dist <= self.core_radius, nearest_basin, torch.full_like(nearest_basin, -1))

    def _exit_pair_index_flat(self, batch: torch.Tensor, source_basin: torch.Tensor):
        exit_pair = torch.full((batch.shape[0],), -1, dtype=torch.long, device=batch.device)
        min_cosine = float(torch.cos(torch.tensor(self.exit_half_angle)))
        for basin in range(self.num_basins):
            basin_mask = source_basin == basin
            if not bool(basin_mask.any()):
                continue
            local_states = batch[basin_mask]
            rel = local_states - self.points_2d[basin]
            rel_radius = torch.norm(rel, dim=-1)
            rel_unit = self._normalize(rel)
            pair_indices = self.source_pair_indices_by_basin[basin].to(device=batch.device)
            pair_dirs = self.exit_directions_2d[pair_indices]
            cosine = rel_unit @ pair_dirs.transpose(0, 1)
            best_cosine, best_local = cosine.max(dim=1)
            selected_pairs = pair_indices[best_local]
            exit_pair[basin_mask] = torch.where(
                (best_cosine >= min_cosine) & (rel_radius >= self.exit_min_radius),
                selected_pairs,
                torch.full_like(selected_pairs, -1),
            )
        return exit_pair

    def source_neighborhood_label(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch(state)
        nearest_basin, nearest_dist = self._nearest_basin(batch)
        channel_index, _, _ = self._channel_membership(batch)
        labels = self._source_neighborhood_label_flat(batch, nearest_basin, nearest_dist, channel_index)
        return labels[0] if squeezed else labels

    def region_label(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch(state)
        nearest_basin, nearest_dist = self._nearest_basin(batch)
        core_basin = self._core_basin_label_flat(nearest_basin, nearest_dist)
        channel_index, _, _ = self._channel_membership(batch)
        source_basin = self._source_neighborhood_label_flat(batch, nearest_basin, nearest_dist, channel_index)
        exit_pair = self._exit_pair_index_flat(batch, source_basin)
        labels = self.return_offset + nearest_basin
        channel_mask = channel_index >= 0
        labels = torch.where(channel_mask, self.channel_offset + channel_index, labels)
        exit_mask = (source_basin >= 0) & (core_basin < 0) & (channel_index < 0) & (exit_pair >= 0)
        labels = torch.where(exit_mask, self.exit_offset + exit_pair, labels)
        core_mask = core_basin >= 0
        labels = torch.where(core_mask, core_basin, labels)
        return labels[0] if squeezed else labels

    def basin_label(self, state: torch.Tensor) -> torch.Tensor:
        region = self.region_label(state)
        region_batch, squeezed = _as_batch(region if isinstance(region, torch.Tensor) else torch.tensor(region))
        basin = torch.full_like(region_batch, -1)
        core_mask = region_batch < self.num_basins
        basin = torch.where(core_mask, region_batch, basin)
        return_mask = (region_batch >= self.return_offset) & (region_batch < self.exit_offset)
        basin = torch.where(return_mask, region_batch - self.return_offset, basin)
        exit_mask = (region_batch >= self.exit_offset) & (region_batch < self.channel_offset)
        if bool(exit_mask.any()):
            exit_pair = region_batch[exit_mask] - self.exit_offset
            basin[exit_mask] = self.channel_destination_basins[exit_pair]
        channel_mask = region_batch >= self.channel_offset
        if bool(channel_mask.any()):
            channel_pair = region_batch[channel_mask] - self.channel_offset
            basin[channel_mask] = self.channel_destination_basins[channel_pair]
        return basin[0] if squeezed else basin

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        batch, squeezed = _as_batch(state)
        nearest_basin, nearest_dist = self._nearest_basin(batch)
        core_basin = self._core_basin_label_flat(nearest_basin, nearest_dist)
        channel_index, _, _ = self._channel_membership(batch)
        source_basin = self._source_neighborhood_label_flat(batch, nearest_basin, nearest_dist, channel_index)
        exit_pair = self._exit_pair_index_flat(batch, source_basin)
        derivatives = torch.zeros_like(batch)

        channel_mask = channel_index >= 0
        if bool(channel_mask.any()):
            idx = channel_index[channel_mask]
            rel = batch[channel_mask] - self.channel_entry_points_2d[idx]
            transverse = (rel * self.channel_normals_2d[idx]).sum(dim=-1, keepdim=True)
            derivatives[channel_mask] = (
                self.channel_speed * self.channel_directions_2d[idx]
                - self.channel_transverse_contraction * transverse * self.channel_normals_2d[idx]
            )

        non_channel_mask = ~channel_mask
        if bool(non_channel_mask.any()):
            active_indices = non_channel_mask.nonzero(as_tuple=False).reshape(-1)
            active_states = batch[active_indices]
            active_nearest = nearest_basin[active_indices]
            active_source = source_basin[active_indices]
            active_core = core_basin[active_indices]
            active_exit = exit_pair[active_indices]

            matrices = self.background_matrices[active_nearest].clone()
            centers = self.points_2d[active_nearest].clone()
            in_source_mask = active_source >= 0
            if bool(in_source_mask.any()):
                source_basins = active_source[in_source_mask]
                matrices[in_source_mask] = self.return_matrices[source_basins]
                centers[in_source_mask] = self.points_2d[source_basins]
            core_mask = active_core >= 0
            if bool(core_mask.any()):
                core_basins = active_core[core_mask]
                matrices[core_mask] = self.basin_matrices[core_basins]
                centers[core_mask] = self.points_2d[core_basins]

            exit_mask = (active_source >= 0) & (active_core < 0) & (active_exit >= 0)
            if bool(exit_mask.any()):
                exit_indices = active_exit[exit_mask]
                source_centers = self.points_2d[active_source[exit_mask]]
                exit_directions = self.exit_directions_2d[exit_indices]
                exit_normals = self.exit_normals_2d[exit_indices]
                rel = active_states[exit_mask] - source_centers
                transverse = (rel * exit_normals).sum(dim=-1, keepdim=True)
                exit_derivatives = (
                    self.exit_forward_rate * exit_directions
                    - self.exit_transverse_rate * transverse * exit_normals
                )
                derivatives[active_indices[exit_mask]] = exit_derivatives
                non_exit_keep_mask = ~exit_mask
                if bool(non_exit_keep_mask.any()):
                    kept_indices = active_indices[non_exit_keep_mask]
                    derivatives[kept_indices] = torch.einsum(
                        "...ij,...j->...i",
                        matrices[non_exit_keep_mask],
                        active_states[non_exit_keep_mask] - centers[non_exit_keep_mask],
                    )
            else:
                derivatives[active_indices] = torch.einsum("...ij,...j->...i", matrices, active_states - centers)

        return derivatives[0] if squeezed else derivatives

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return integrate_rk4(state, self.dt, self.dynamics)

    def trajectory_starts(self, points_per_axis: int) -> torch.Tensor:
        per_basin_points = []
        for center in self.points_2d:
            coords_x = torch.linspace(float(center[0] - self.source_radius), float(center[0] + self.source_radius), steps=int(points_per_axis))
            coords_y = torch.linspace(float(center[1] - self.source_radius), float(center[1] + self.source_radius), steps=int(points_per_axis))
            mesh_x, mesh_y = torch.meshgrid(coords_x, coords_y, indexing="ij")
            points = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=1)
            inside = torch.norm(points - center.unsqueeze(0), dim=1) <= float(self.source_radius) + 1e-8
            per_basin_points.append(points[inside])
        return torch.cat(per_basin_points, dim=0)

    def plot_guides(self, ax) -> None:
        centers = self.points_2d.detach().cpu().numpy()
        ax.scatter(centers[:, 0], centers[:, 1], color="black", marker="x", s=70, linewidths=1.8, zorder=4)
        for center in centers:
            ax.add_patch(Circle((float(center[0]), float(center[1])), float(self.source_radius), fill=False, linestyle="--", linewidth=0.9, edgecolor="#475569", alpha=0.55))
            ax.add_patch(Circle((float(center[0]), float(center[1])), float(self.core_radius), fill=False, linestyle=":", linewidth=1.0, edgecolor="#0f172a", alpha=0.9))
        for pair_index in range(len(self.ordered_pairs)):
            entry = self.channel_entry_points_2d[pair_index].detach().cpu().numpy()
            handoff = self.channel_handoff_points_2d[pair_index].detach().cpu().numpy()
            ax.plot([entry[0], handoff[0]], [entry[1], handoff[1]], linestyle="-.", linewidth=1.0, color="#334155", alpha=0.45, zorder=2)


def build_system(system_key: str) -> PlotSystem:
    if system_key == "multiwell_strong_transition":
        return MultiwellStrongTransitionSystem()
    if system_key == "gated_local_linear":
        return GatedLocalLinearStandaloneSystem()
    if system_key == "gated_transfer_linear":
        return GatedTransferLinearStandaloneSystem()
    raise ValueError(f"Unsupported system_key: {system_key}")


def _compute_field(system: PlotSystem, grid_points: int):
    coords = torch.linspace(-float(system.init_range), float(system.init_range), steps=int(grid_points))
    mesh_x, mesh_y = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=1)
    velocity = system.dynamics(points)
    regions = system.region_label(points)
    return (
        mesh_x.detach().cpu().numpy(),
        mesh_y.detach().cpu().numpy(),
        velocity[:, 0].reshape(grid_points, grid_points).detach().cpu().numpy(),
        velocity[:, 1].reshape(grid_points, grid_points).detach().cpu().numpy(),
        regions.reshape(grid_points, grid_points).detach().cpu().numpy(),
    )


def _generate_trajectories(system: PlotSystem, points_per_axis: int, trajectory_length: int) -> list[np.ndarray]:
    starts = system.trajectory_starts(points_per_axis)
    trajectories = []
    for start in starts:
        traj = [start]
        current = start
        for _ in range(int(trajectory_length)):
            current = system.step(current)
            traj.append(current)
        trajectories.append(torch.stack(traj, dim=0).detach().cpu().numpy())
    return trajectories


def _save_formats(fig, stem: Path, formats: Iterable[str]) -> list[Path]:
    output_paths: list[Path] = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        output_paths.append(path)
    return output_paths


def _system_centers(system: PlotSystem) -> np.ndarray:
    centers = getattr(system, "points_2d", None)
    if centers is None:
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(centers, torch.Tensor):
        return centers.detach().cpu().numpy()
    return np.asarray(centers, dtype=np.float32)


def plot_system_catalog_portrait(
    system: PlotSystem,
    output_dir: Path,
    *,
    grid_points: int,
    trajectory_length: int,
    start_points_per_axis: int,
    formats: tuple[str, ...],
) -> list[Path]:
    x, y, u, v, _ = _compute_field(system, grid_points=grid_points)
    trajectories = _generate_trajectories(
        system,
        points_per_axis=start_points_per_axis,
        trajectory_length=trajectory_length,
    )
    centers = _system_centers(system)
    if centers.size:
        n_basins = int(centers.shape[0])
    else:
        sample_point = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        n_basins = int(system.basin_label(sample_point).max().item()) + 1
    cmap = plt.get_cmap("tab10", max(n_basins, 1))
    colors = [cmap(i) for i in range(n_basins)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    traj_ax, field_ax, basin_ax = axes

    n_plot = min(100, len(trajectories))
    if n_plot > 0:
        indices = np.random.RandomState(42).choice(len(trajectories), n_plot, replace=False)
    else:
        indices = np.array([], dtype=int)
    for index in indices:
        trajectory = trajectories[int(index)]
        endpoint_label = int(
            system.basin_label(torch.tensor(trajectory[-1], dtype=torch.float32)).item()
        )
        color = colors[endpoint_label] if endpoint_label >= 0 else "gray"
        alpha = 0.3 if endpoint_label >= 0 else 0.12
        traj_ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            alpha=alpha,
            linewidth=0.5,
        )
    if centers.size:
        for basin_index, center in enumerate(centers):
            traj_ax.plot(
                center[0],
                center[1],
                "X",
                color=colors[basin_index],
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=10,
            )
    system.plot_guides(traj_ax)
    traj_ax.set_title(f"Trajectories (N={n_plot})", fontsize=11)
    traj_ax.set_xlabel("$x_1$")
    traj_ax.set_ylabel("$x_2$")
    traj_ax.set_xlim(-float(system.init_range), float(system.init_range))
    traj_ax.set_ylim(-float(system.init_range), float(system.init_range))
    traj_ax.set_aspect("equal", adjustable="box")
    traj_ax.grid(True, alpha=0.3)

    speed = np.sqrt(u ** 2 + v ** 2)
    field_ax.streamplot(
        x[:, 0],
        y[0, :],
        u.T,
        v.T,
        color=np.log1p(speed).T,
        cmap="viridis",
        linewidth=0.8,
        density=1.5,
        arrowsize=0.8,
    )
    if centers.size:
        for center in centers:
            field_ax.plot(
                center[0],
                center[1],
                "X",
                color="red",
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=10,
            )
    system.plot_guides(field_ax)
    field_ax.set_title("Vector field", fontsize=11)
    field_ax.set_xlabel("$x_1$")
    field_ax.set_ylabel("$x_2$")
    field_ax.set_xlim(-float(system.init_range), float(system.init_range))
    field_ax.set_ylim(-float(system.init_range), float(system.init_range))
    field_ax.set_aspect("equal", adjustable="box")
    field_ax.grid(True, alpha=0.3)

    x_fine = np.linspace(-float(system.init_range), float(system.init_range), 80)
    y_fine = np.linspace(-float(system.init_range), float(system.init_range), 80)
    mesh_x, mesh_y = np.meshgrid(x_fine, y_fine)
    grid_points_xy = torch.tensor(
        np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1),
        dtype=torch.float32,
    )
    basin_map = (
        system.basin_label(grid_points_xy)
        .reshape(mesh_x.shape)
        .detach()
        .cpu()
        .numpy()
    )
    region_cmap = ListedColormap(colors[:n_basins])
    basin_ax.pcolormesh(
        mesh_x,
        mesh_y,
        basin_map,
        cmap=region_cmap,
        alpha=0.3,
        shading="auto",
    )
    for index in indices[:50]:
        trajectory = trajectories[int(index)]
        endpoint_label = int(
            system.basin_label(torch.tensor(trajectory[-1], dtype=torch.float32)).item()
        )
        if endpoint_label < 0:
            continue
        basin_ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=colors[endpoint_label],
            alpha=0.4,
            linewidth=0.3,
        )
    if centers.size:
        for basin_index, center in enumerate(centers):
            basin_ax.plot(
                center[0],
                center[1],
                "X",
                color=colors[basin_index],
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=10,
            )
    system.plot_guides(basin_ax)
    basin_ax.set_title("Basin map + trajectories", fontsize=11)
    basin_ax.set_xlabel("$x_1$")
    basin_ax.set_ylabel("$x_2$")
    basin_ax.set_xlim(-float(system.init_range), float(system.init_range))
    basin_ax.set_ylim(-float(system.init_range), float(system.init_range))
    basin_ax.set_aspect("equal", adjustable="box")
    basin_ax.grid(True, alpha=0.3)

    fig.suptitle(system.title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / system.key
    output_paths = _save_formats(fig, stem, formats=formats)
    plt.close(fig)
    return output_paths


def plot_system_overview(
    system: PlotSystem,
    output_dir: Path,
    *,
    grid_points: int,
    trajectory_length: int,
    start_points_per_axis: int,
    formats: tuple[str, ...],
) -> list[Path]:
    x, y, u, v, regions = _compute_field(system, grid_points=grid_points)
    trajectories = _generate_trajectories(system, points_per_axis=start_points_per_axis, trajectory_length=trajectory_length)
    region_count = int(np.max(regions)) + 1
    region_cmap = ListedColormap(plt.get_cmap("tab20").colors[: max(region_count, 3)])

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), constrained_layout=True)
    phase_ax, region_ax = axes

    speed = np.sqrt(u ** 2 + v ** 2)
    phase_ax.streamplot(
        x[:, 0],
        y[0, :],
        u.T,
        v.T,
        color=speed.T,
        cmap="Greys",
        density=1.25,
        linewidth=1.0,
        arrowsize=1.0,
        minlength=0.15,
        zorder=1,
    )
    palette = plt.get_cmap("tab10").colors
    for trajectory in trajectories:
        endpoint = int(system.basin_label(torch.tensor(trajectory[-1], dtype=torch.float32)).item())
        phase_ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=palette[endpoint % len(palette)],
            alpha=0.55,
            linewidth=1.2,
            zorder=2,
        )
    system.plot_guides(phase_ax)
    phase_ax.set_title(f"{system.title}: Phase Portrait")
    phase_ax.set_xlim(-float(system.init_range), float(system.init_range))
    phase_ax.set_ylim(-float(system.init_range), float(system.init_range))
    phase_ax.set_aspect("equal", adjustable="box")
    phase_ax.set_xlabel("x1")
    phase_ax.set_ylabel("x2")
    phase_ax.grid(alpha=0.2)

    region_ax.imshow(
        regions.T,
        origin="lower",
        extent=[-float(system.init_range), float(system.init_range), -float(system.init_range), float(system.init_range)],
        cmap=region_cmap,
        interpolation="nearest",
        aspect="equal",
        alpha=0.82,
    )
    system.plot_guides(region_ax)
    region_ax.set_title(f"{system.title}: Region Map")
    region_ax.set_xlim(-float(system.init_range), float(system.init_range))
    region_ax.set_ylim(-float(system.init_range), float(system.init_range))
    region_ax.set_aspect("equal", adjustable="box")
    region_ax.set_xlabel("x1")
    region_ax.set_ylabel("x2")
    region_ax.grid(alpha=0.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"{system.key}_interpretable_overview"
    output_paths = _save_formats(fig, stem, formats=formats)
    plt.close(fig)
    return output_paths


def plot_selected_systems(
    *,
    systems: list[str],
    output_dir: Path,
    grid_points: int,
    trajectory_length: int,
    start_points_per_axis: int,
    formats: tuple[str, ...],
    layout: str = DEFAULT_LAYOUT,
) -> list[Path]:
    layout = validate_layout(layout)
    output_paths: list[Path] = []
    for system_key in systems:
        system = build_system(system_key)
        if layout == "overview":
            output_paths.extend(
                plot_system_overview(
                    system,
                    output_dir,
                    grid_points=grid_points,
                    trajectory_length=trajectory_length,
                    start_points_per_axis=start_points_per_axis,
                    formats=formats,
                )
            )
            continue
        output_paths.extend(
            plot_system_catalog_portrait(
                system,
                output_dir,
                grid_points=grid_points,
                trajectory_length=trajectory_length,
                start_points_per_axis=start_points_per_axis,
                formats=formats,
            )
        )
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        type=str,
        default="all",
        help="Comma-separated subset of systems to plot, or 'all'.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("docs/figures/interpretable_systems_20260406"),
    )
    parser.add_argument("--grid_points", type=int, default=61)
    parser.add_argument("--trajectory_length", type=int, default=120)
    parser.add_argument("--start_points_per_axis", type=int, default=6)
    parser.add_argument(
        "--layout",
        type=str,
        default=DEFAULT_LAYOUT,
        help="Figure layout: overview or catalog.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,svg,pdf",
        help="Comma-separated output formats drawn from png, svg, pdf.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    systems = parse_systems_arg(args.systems)
    formats = parse_formats_arg(args.formats)
    plot_selected_systems(
        systems=systems,
        output_dir=args.output_dir.resolve(),
        grid_points=int(args.grid_points),
        trajectory_length=int(args.trajectory_length),
        start_points_per_axis=int(args.start_points_per_axis),
        formats=formats,
        layout=args.layout,
    )


if __name__ == "__main__":
    main()
