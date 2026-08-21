"""Independent origin/main behavior checks for GatedTransferLinear."""

import math

import pytest
import torch

from skae.config import Config
from skae.data import GatedTransferLinear, integrate_rk4


def _config() -> Config:
    cfg = Config()
    cfg.ENV.ENV_NAME = "gated_transfer_linear"
    cfg.ENV.GATED_TRANSFER_LINEAR.DT = 0.04
    return cfg


class _OriginMainGatedTransferReference:
    """Independent equations transcribed from origin/main before PR1.

    This reference intentionally does not instantiate or call
    ``GatedTransferLinear``. It records pre-PR behavior at origin/main commit
    ``6a4c24a`` so tests cannot compare two paths through the refactored class.
    """

    ORIGIN_MAIN_COMMIT = "6a4c24a"

    def __init__(self, dt: float = 0.04):
        self.dt = float(dt)
        self.num_basins = 3
        self.center_radius = 1.85
        self.core_radius = 0.30
        self.source_radius = 0.80
        self.handoff_radius = 0.45
        self.exit_min_radius = 0.60
        self.channel_half_width = 0.22
        self.channel_lane_offset = 0.28
        self.exit_half_angle = 0.72
        self.return_rate_scale = 0.65
        self.background_rate_scale = 0.50
        self.exit_forward_rate = 1.0
        self.exit_transverse_rate = 1.8
        self.exit_handoff_offset = 0.40
        self.channel_speed = 1.55
        self.channel_transverse_contraction = 2.8

        self.center_angles = torch.linspace(
            0.0, 2.0 * torch.pi, steps=self.num_basins + 1, dtype=torch.float32
        )[:-1]
        self.points_2d = torch.stack(
            [
                self.center_radius * torch.cos(self.center_angles),
                self.center_radius * torch.sin(self.center_angles),
            ],
            dim=1,
        )
        self.outward_directions = self._normalize(self.points_2d)
        self.ordered_pairs = [
            (source, dest)
            for source in range(self.num_basins)
            for dest in range(self.num_basins)
            if dest != source
        ]
        self.pair_to_index = {
            pair: index for index, pair in enumerate(self.ordered_pairs)
        }
        self.source_pair_indices_by_basin = [
            torch.tensor(
                [
                    self.pair_to_index[(source, dest)]
                    for dest in range(self.num_basins)
                    if dest != source
                ],
                dtype=torch.long,
            )
            for source in range(self.num_basins)
        ]
        self.exit_directions_2d = torch.stack(
            [
                self._normalize(
                    (self.points_2d[dest] - self.points_2d[source]).unsqueeze(0)
                )[0]
                for source, dest in self.ordered_pairs
            ],
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
        self.return_offset = self.num_basins
        self.exit_offset = self.return_offset + self.num_basins
        self.channel_offset = self.exit_offset + len(self.ordered_pairs)

    @staticmethod
    def _normalize(vectors: torch.Tensor) -> torch.Tensor:
        return vectors / torch.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)

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
        directions, normals = [], []
        entry_points, handoff_points, handoff_targets = [], [], []
        lengths, destination_basins = [], []
        for pair_index, (source, dest) in enumerate(self.ordered_pairs):
            source_to_dest = self.exit_directions_2d[pair_index]
            source_exit_normal = self.exit_normals_2d[pair_index]
            destination_outward = self.outward_directions[dest]
            destination_tangent = self._rot90(destination_outward)
            incoming_delta = float(
                torch.sin(self.center_angles[source] - self.center_angles[dest])
            )
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
            directions.append(direction)
            normals.append(normal)
            entry_points.append(entry)
            handoff_points.append(handoff)
            handoff_targets.append(entry + self.exit_handoff_offset * direction)
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
        closest_match = torch.where(
            mask,
            torch.abs(transverse),
            torch.full_like(transverse, float("inf")),
        ).argmin(dim=1)
        return torch.where(
            has_channel,
            closest_match.to(dtype=torch.long),
            torch.full_like(closest_match, -1, dtype=torch.long),
        )

    def _exit_pair_index(self, batch: torch.Tensor, source_basin: torch.Tensor):
        exit_pair = torch.full((batch.shape[0],), -1, dtype=torch.long)
        min_cosine = math.cos(self.exit_half_angle)
        for basin in range(self.num_basins):
            basin_mask = source_basin == basin
            if not bool(basin_mask.any()):
                continue
            local_states = batch[basin_mask]
            rel = local_states - self.points_2d[basin]
            rel_radius = torch.norm(rel, dim=-1)
            rel_unit = self._normalize(rel)
            pair_indices = self.source_pair_indices_by_basin[basin]
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

    def _labels_and_masks(self, state: torch.Tensor):
        batch = state.unsqueeze(0) if state.ndim == 1 else state
        squeezed = state.ndim == 1
        nearest_basin, nearest_dist = self._nearest_basin(batch)
        core_basin = torch.where(
            nearest_dist <= self.core_radius,
            nearest_basin,
            torch.full_like(nearest_basin, -1),
        )
        channel_index = self._channel_membership(batch)
        source_basin = torch.where(
            (nearest_dist <= self.source_radius) & (channel_index < 0),
            nearest_basin,
            torch.full_like(nearest_basin, -1),
        )
        exit_pair = self._exit_pair_index(batch, source_basin)
        labels = self.return_offset + nearest_basin
        labels = torch.where(channel_index >= 0, self.channel_offset + channel_index, labels)
        exit_mask = (source_basin >= 0) & (core_basin < 0) & (channel_index < 0) & (exit_pair >= 0)
        labels = torch.where(exit_mask, self.exit_offset + exit_pair, labels)
        labels = torch.where(core_basin >= 0, core_basin, labels)
        return batch, squeezed, nearest_basin, core_basin, channel_index, source_basin, exit_pair, labels

    def region_label(self, state: torch.Tensor) -> torch.Tensor:
        *_, labels = self._labels_and_masks(state)
        return labels[0] if state.ndim == 1 else labels

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        (
            batch,
            squeezed,
            nearest_basin,
            core_basin,
            channel_index,
            source_basin,
            exit_pair,
            _,
        ) = self._labels_and_masks(state)
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
                derivatives[active_indices] = torch.einsum(
                    "...ij,...j->...i", matrices, active_states - centers
                )
        return derivatives[0] if squeezed else derivatives

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return integrate_rk4(state, None, self.dt, lambda current, _a=None: self.dynamics(current))


def _coverage_states(reference: _OriginMainGatedTransferReference) -> torch.Tensor:
    pair_index = reference.pair_to_index[(0, 1)]
    return torch.stack(
        [
            reference.points_2d[0],
            reference.points_2d[0] + torch.tensor([0.40, 0.0]),
            reference.points_2d[0] + 0.70 * reference.exit_directions_2d[pair_index],
            reference.channel_entry_points_2d[0] + 0.10 * reference.channel_directions_2d[0],
            torch.tensor([0.0, 2.8]),
        ],
        dim=0,
    )


def test_origin_main_reference_covers_regions_and_empty_masks_exactly_on_cpu():
    target = GatedTransferLinear(_config())
    reference = _OriginMainGatedTransferReference()
    states = _coverage_states(reference)
    labels = target.region_label(states)

    assert int(labels[0]) == target.core_offset
    assert target.return_offset <= int(labels[1]) < target.exit_offset
    assert target.exit_offset <= int(labels[2]) < target.channel_offset
    assert int(labels[3]) >= target.channel_offset
    assert torch.equal(target.step(states), reference.step(states))
    assert torch.equal(target.dynamics(states[[4, 4]]), reference.dynamics(states[[4, 4]]))
    empty = torch.empty((0, 2), dtype=torch.float32)
    assert torch.equal(target.step(empty), reference.step(empty))
    assert torch.equal(target.dynamics(states[[3, 3]]), reference.dynamics(states[[3, 3]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_origin_main_reference_matches_native_cuda_with_tight_tolerance():
    target = GatedTransferLinear(_config())
    reference = _OriginMainGatedTransferReference()
    states = _coverage_states(reference)
    assert torch.allclose(target.step(states.cuda()).cpu(), reference.step(states), atol=5e-6, rtol=5e-6)
