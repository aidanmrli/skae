"""Frozen feature definitions for Allen--Cahn fate readout."""

from __future__ import annotations

import numpy as np
import torch


def well_centers(*, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0], [0.0, -1.5]],
        dtype=dtype,
    )


def _grid_fields(flat_fields: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(flat_fields)
    if values.ndim != 2 or values.shape[1] != 512:
        raise ValueError("Expected flattened [trajectory,512] Allen--Cahn fields")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Physical fields contain nonfinite values")
    return values.reshape(-1, 16, 16, 2)


def nearest_well_maps(flat_fields: torch.Tensor) -> torch.Tensor:
    fields = _grid_fields(flat_fields).double()
    centers = well_centers(dtype=torch.float64).to(fields)
    distances = (fields.unsqueeze(-2) - centers.view(1, 1, 1, 4, 2)).square().sum(-1)
    return distances.argmin(-1).long()


def well_area_fractions(flat_fields: torch.Tensor) -> torch.Tensor:
    maps = nearest_well_maps(flat_fields)
    return torch.stack([(maps == index).double().mean((1, 2)) for index in range(4)], dim=1)


def modal_well_labels(flat_fields: torch.Tensor) -> torch.Tensor:
    """Lowest-index tie-broken modal nearest-well label."""

    return well_area_fractions(flat_fields).argmax(1).long()


def interface_fraction(flat_fields: torch.Tensor) -> torch.Tensor:
    maps = nearest_well_maps(flat_fields)
    right = maps != torch.roll(maps, shifts=-1, dims=2)
    down = maps != torch.roll(maps, shifts=-1, dims=1)
    return torch.logical_or(right, down).double().mean((1, 2))[:, None]


def field_summary(flat_fields: torch.Tensor) -> torch.Tensor:
    fields = _grid_fields(flat_fields).double()
    means = fields.mean((1, 2))
    stds = fields.std((1, 2), correction=0)
    dx = torch.roll(fields, shifts=-1, dims=2) - fields
    dy = torch.roll(fields, shifts=-1, dims=1) - fields
    gradient_energy = (dx.square() + dy.square()).mean((1, 2))
    fractions = well_area_fractions(flat_fields)
    interface = interface_fraction(flat_fields)
    result = torch.cat((means, stds, gradient_energy, fractions, interface), dim=1)
    if result.shape[1] != 11 or not torch.isfinite(result).all():
        raise AssertionError("Frozen 11-value physical summary contract failed")
    return result


def matched_topk_masks(dense_values: np.ndarray, sparse_masks: np.ndarray) -> np.ndarray:
    dense = np.asarray(dense_values, dtype=np.float64)
    sparse = np.asarray(sparse_masks, dtype=bool)
    if dense.shape != sparse.shape or dense.ndim != 2:
        raise ValueError("Dense values and sparse masks must share [row,coordinate] shape")
    if not np.isfinite(dense).all():
        raise FloatingPointError("Dense latent values contain nonfinite entries")
    order = np.argsort(-np.abs(dense), axis=1, kind="stable")
    cardinalities = sparse.sum(1).astype(np.int64)
    result = np.zeros_like(sparse)
    for row, cardinality in enumerate(cardinalities):
        if cardinality:
            result[row, order[row, : int(cardinality)]] = True
    if not np.array_equal(result.sum(1), cardinalities):
        raise AssertionError("Dense top-k cardinality matching failed")
    return result


def observation_tensor(
    train_fields: torch.Tensor,
    test_fields: list[torch.Tensor],
    observation_indices: list[int],
) -> tuple[torch.Tensor, dict[str, object]]:
    if len(test_fields) != 3:
        raise ValueError("Frozen probe requires exactly three test datasets")
    combined = torch.cat([train_fields, *test_fields], dim=0)
    expected_rows = 512 + 3 * 256
    if tuple(combined.shape) != (expected_rows, 201, 512):
        raise ValueError(f"Unexpected combined field shape {tuple(combined.shape)}")
    selected = combined[:, observation_indices].permute(1, 0, 2).contiguous()
    layout = {
        "time_major": True,
        "observation_indices": list(observation_indices),
        "rows_per_time": expected_rows,
        "train_slice": [0, 512],
        "test_slices": [[512, 768], [768, 1024], [1024, 1280]],
    }
    return selected.reshape(-1, 512).contiguous(), layout


@torch.no_grad()
def encode_in_batches(
    model: torch.nn.Module, observations: torch.Tensor, *, batch_size: int
) -> torch.Tensor:
    if observations.device != next(model.parameters()).device:
        raise AssertionError("Observations and model must already share the GPU")
    chunks = []
    for start in range(0, observations.shape[0], int(batch_size)):
        chunks.append(model.encode(observations[start : start + int(batch_size)]))
    result = torch.cat(chunks)
    if result.shape != (observations.shape[0], 2048) or not torch.isfinite(result).all():
        raise AssertionError("Encoded latent shape/finiteness contract failed")
    return result
