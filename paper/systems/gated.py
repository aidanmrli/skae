"""The two gated local-linear multibasin systems.

Both are piecewise-linear vector fields around three basin centers, defined in
float32 and advanced by one RK4 step per stored observation. ``dynamics``
accepts one state ``[2]`` or a batch ``[B, 2]``.
"""

from __future__ import annotations

import math

import torch

from paper.systems.analytic import rk4_step


def _rotation(angle: torch.Tensor) -> torch.Tensor:
    c = torch.cos(angle)
    s = torch.sin(angle)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])], dim=0)


def _normalize(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / torch.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-8)


def _rot90(vectors: torch.Tensor) -> torch.Tensor:
    rotated = torch.zeros_like(vectors)
    rotated[..., 0] = -vectors[..., 1]
    rotated[..., 1] = vectors[..., 0]
    return rotated


def _as_batch(state: torch.Tensor):
    """``(batch [B, 2], was_single)``: a single state becomes a batch of one."""
    if state.ndim == 1:
        return state.unsqueeze(0), True
    return state, False


class _GatedSystem:
    """Three basin centers at angles 0, 2pi/3, 4pi/3 and radius ``center_radius``.

    Basin ``k`` contracts with ``templates[k]`` rotated to the basin's angle;
    initial conditions are uniform in the square ``[-init_range, init_range]^2``.
    """

    num_basins = 3

    def __init__(self, dt: float, center_radius: float, init_range: float, templates):
        self.dt = dt
        self.init_range = init_range
        self.ic_box = [(-init_range, init_range), (-init_range, init_range)]
        self.center_angles = torch.linspace(0.0, 2.0 * torch.pi, steps=self.num_basins + 1, dtype=torch.float32)[:-1]
        self.centers = torch.stack(
            [center_radius * torch.cos(self.center_angles), center_radius * torch.sin(self.center_angles)], dim=1
        )
        # Basin k contracts with template k rotated to the basin's angle.
        templates = torch.tensor(templates, dtype=torch.float32)
        self.basin_matrices = torch.stack(
            [_rotation(angle) @ base @ _rotation(angle).transpose(0, 1)
             for base, angle in zip(templates, self.center_angles)], dim=0
        )

    def reset(self, generator: torch.Generator) -> torch.Tensor:
        return torch.empty(2, dtype=torch.float32).uniform_(-self.init_range, self.init_range, generator=generator)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        return rk4_step(self.dynamics, state, self.dt)


class GatedLocalLinear(_GatedSystem):
    """Stable spirals inside discs around the centers; outside, angular gate sectors.

    A state within ``basin_radius`` of its nearest center contracts toward it
    with that basin's matrix; any other state contracts with a shared spiral
    toward the center of its angular sector.
    """

    def __init__(self, dt: float):
        super().__init__(dt, center_radius=1.75, init_range=2.6, templates=[
            [[-0.9, -1.2], [1.2, -0.9]],
            [[-1.35, 0.2], [-0.3, -0.7]],
            [[-0.7, -0.1], [0.5, -1.2]],
        ])
        self.basin_radius = 1.05
        self.gate_matrix = torch.tensor([[-1.35, -0.9], [0.9, -1.35]], dtype=torch.float32)

    def _sector(self, batch: torch.Tensor) -> torch.Tensor:
        angles = torch.atan2(batch[..., 1], batch[..., 0])
        deltas = torch.remainder(angles.unsqueeze(-1) - self.center_angles + torch.pi, 2.0 * torch.pi) - torch.pi
        return deltas.abs().argmin(dim=-1).to(dtype=torch.long)

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        batch, single = _as_batch(state)
        distance = torch.norm(batch.unsqueeze(1) - self.centers.unsqueeze(0), dim=-1)
        in_basin = distance.min(dim=-1).values <= self.basin_radius
        basin = torch.where(in_basin, distance.argmin(dim=-1), self._sector(batch))
        matrices = torch.where(in_basin[..., None, None], self.basin_matrices[basin], self.gate_matrix)
        velocity = torch.einsum("...ij,...j->...i", matrices, batch - self.centers[basin])
        return velocity[0] if single else velocity


class GatedTransferLinear(_GatedSystem):
    """Basins joined by straight transfer channels.

    Regions, by priority: a straight channel from each basin toward each other
    basin (constant flow along it, contraction across it; it takes precedence
    over every other region), a core disc around each center (basin matrix), an
    exit cone that launches states from a source neighborhood into a channel,
    the source neighborhood (a weaker return toward the center) and the
    background (weaker still, toward the nearest center).
    """

    def __init__(self, dt: float):
        super().__init__(dt, center_radius=1.85, init_range=2.8, templates=[
            [[-1.0, -1.1], [1.1, -1.0]],
            [[-1.4, 0.2], [-0.2, -0.7]],
            [[-0.8, -0.3], [0.5, -1.3]],
        ])
        self.core_radius = 0.30
        self.source_radius = 0.80
        self.handoff_radius = 0.45
        self.exit_min_radius = 0.60
        self.channel_half_width = 0.22
        self.channel_lane_offset = 0.28
        self.exit_half_angle = 0.72
        self.exit_forward_rate = 1.0
        self.exit_transverse_rate = 1.8
        self.channel_speed = 1.55
        self.channel_transverse_contraction = 2.8
        self.return_matrices = 0.65 * self.basin_matrices
        self.background_matrices = 0.50 * self.basin_matrices

        # One exit direction and one channel per ordered pair (source, destination).
        pairs = [(s, d) for s in range(self.num_basins) for d in range(self.num_basins) if d != s]
        self.source_pairs = [
            torch.tensor([index for index, (s, _) in enumerate(pairs) if s == source], dtype=torch.long)
            for source in range(self.num_basins)
        ]
        self.exit_directions = torch.stack(
            [_normalize((self.centers[d] - self.centers[s]).unsqueeze(0))[0] for s, d in pairs], dim=0
        )
        self.exit_normals = _rot90(self.exit_directions)
        outward = _normalize(self.centers)
        directions, entries, lengths = [], [], []
        for index, (source, dest) in enumerate(pairs):
            lane = 1.0 if float(torch.sin(self.center_angles[source] - self.center_angles[dest])) >= 0.0 else -1.0
            entry = (self.centers[source] + self.source_radius * self.exit_directions[index]
                     + (lane * self.channel_lane_offset) * self.exit_normals[index])
            handoff = (self.centers[dest] + self.handoff_radius * outward[dest]
                       + (lane * self.channel_lane_offset) * _rot90(outward[dest]))
            direction = _normalize((handoff - entry).unsqueeze(0))[0]
            directions.append(direction)
            entries.append(entry)
            lengths.append(torch.dot(direction, handoff - entry))
        self.channel_directions = torch.stack(directions, dim=0)
        self.channel_normals = _rot90(self.channel_directions)
        self.channel_entries = torch.stack(entries, dim=0)
        self.channel_lengths = torch.stack(lengths, dim=0)

    def _channel(self, batch: torch.Tensor) -> torch.Tensor:
        """Index of the channel containing each state (the closest to its axis), or -1."""
        rel = batch.unsqueeze(1) - self.channel_entries.unsqueeze(0)
        longitudinal = (rel * self.channel_directions.unsqueeze(0)).sum(dim=-1)
        transverse = (rel * self.channel_normals.unsqueeze(0)).sum(dim=-1)
        inside = (
            (longitudinal >= 0.0)
            & (longitudinal <= self.channel_lengths.unsqueeze(0))
            & (torch.abs(transverse) <= self.channel_half_width)
        )
        closest = torch.where(inside, torch.abs(transverse), torch.full_like(transverse, float("inf"))).argmin(dim=1)
        return torch.where(inside.any(dim=1), closest.to(dtype=torch.long), torch.full_like(closest, -1))

    def _exit_pair(self, batch: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Index of the exit cone containing each source-neighborhood state, or -1."""
        exit_pair = torch.full((batch.shape[0],), -1, dtype=torch.long, device=batch.device)
        min_cosine = math.cos(self.exit_half_angle)
        for basin in range(self.num_basins):
            mask = source == basin
            if not bool(mask.any()):
                continue
            rel = batch[mask] - self.centers[basin]
            pair_indices = self.source_pairs[basin].to(device=batch.device)
            cosine = _normalize(rel) @ self.exit_directions[pair_indices].transpose(0, 1)
            best_cosine, best = cosine.max(dim=1)
            selected = pair_indices[best]
            exit_pair[mask] = torch.where(
                (best_cosine >= min_cosine) & (torch.norm(rel, dim=-1) >= self.exit_min_radius),
                selected,
                torch.full_like(selected, -1),
            )
        return exit_pair

    def dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """Velocity ``[2]`` or ``[B, 2]`` from each state's region, in the priority of the class docstring.

        A state in two channels follows the one whose axis is closer; a state in
        two exit cones follows the better-aligned exit direction.
        """
        batch, single = _as_batch(state)
        distance = torch.norm(batch.unsqueeze(1) - self.centers.unsqueeze(0), dim=-1)
        nearest_dist, nearest = distance.min(dim=1)
        nearest = nearest.to(dtype=torch.long)
        core = torch.where(nearest_dist <= self.core_radius, nearest, torch.full_like(nearest, -1))
        channel = self._channel(batch)
        source = torch.where((nearest_dist <= self.source_radius) & (channel < 0), nearest, torch.full_like(nearest, -1))
        exit_pair = self._exit_pair(batch, source)
        derivatives = torch.zeros_like(batch)

        in_channel = channel >= 0
        if bool(in_channel.any()):
            idx = channel[in_channel]
            rel = batch[in_channel] - self.channel_entries[idx]
            transverse = (rel * self.channel_normals[idx]).sum(dim=-1, keepdim=True)
            derivatives[in_channel] = (
                self.channel_speed * self.channel_directions[idx]
                - self.channel_transverse_contraction * transverse * self.channel_normals[idx]
            )

        outside = ~in_channel
        if bool(outside.any()):
            active = outside.nonzero(as_tuple=False).reshape(-1)
            states = batch[active]
            active_nearest = nearest[active]
            active_source = source[active]
            active_core = core[active]
            active_exit = exit_pair[active]

            # Background matrix, overridden by the return matrix in a source
            # neighborhood and by the basin matrix in a core.
            matrices = self.background_matrices[active_nearest].clone()
            centers = self.centers[active_nearest].clone()
            in_source = active_source >= 0
            if bool(in_source.any()):
                matrices[in_source] = self.return_matrices[active_source[in_source]]
                centers[in_source] = self.centers[active_source[in_source]]
            in_core = active_core >= 0
            if bool(in_core.any()):
                matrices[in_core] = self.basin_matrices[active_core[in_core]]
                centers[in_core] = self.centers[active_core[in_core]]

            in_exit = (active_source >= 0) & (active_core < 0) & (active_exit >= 0)
            if bool(in_exit.any()):
                exits = active_exit[in_exit]
                rel = states[in_exit] - self.centers[active_source[in_exit]]
                transverse = (rel * self.exit_normals[exits]).sum(dim=-1, keepdim=True)
                derivatives[active[in_exit]] = (
                    self.exit_forward_rate * self.exit_directions[exits]
                    - self.exit_transverse_rate * transverse * self.exit_normals[exits]
                )
                kept = ~in_exit
                if bool(kept.any()):
                    derivatives[active[kept]] = torch.einsum(
                        "...ij,...j->...i", matrices[kept], states[kept] - centers[kept]
                    )
            else:
                derivatives[active] = torch.einsum("...ij,...j->...i", matrices, states - centers)

        return derivatives[0] if single else derivatives
