"""Support, closure, and fixed-projector forecast metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


EPS = 1e-20


@dataclass(frozen=True)
class FamilyCodebook:
    representatives: np.ndarray
    fit_counts: np.ndarray


def jaccard_scores(masks: np.ndarray, representatives: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(masks[:, None, :], representatives[None, :, :]).sum(axis=2)
    union = np.logical_or(masks[:, None, :], representatives[None, :, :]).sum(axis=2)
    return np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union != 0,
    )


def fit_codebook(
    masks: np.ndarray,
    *,
    min_jaccard: float,
    max_representatives: int,
    min_fit_count: int,
) -> FamilyCodebook:
    if masks.ndim != 2 or masks.shape[0] == 0:
        raise ValueError("masks must have shape [states, latent] and be nonempty")
    unique, counts = np.unique(masks.astype(bool), axis=0, return_counts=True)
    packed = np.packbits(unique.astype(np.uint8), axis=1)
    order = sorted(
        range(unique.shape[0]),
        key=lambda index: (-int(counts[index]), packed[index].tobytes()),
    )
    representatives: list[np.ndarray] = []
    for index in order:
        candidate = unique[index]
        similarity = max(
            (float(jaccard_scores(candidate[None], np.stack(representatives))[0].max())
             for _ in (0,)),
            default=-1.0,
        ) if representatives else -1.0
        if similarity < float(min_jaccard):
            representatives.append(candidate.copy())
        if len(representatives) >= int(max_representatives):
            break
    reps = np.stack(representatives)
    labels, similarities = assign_codebook(masks, reps, min_jaccard=min_jaccard)
    del similarities
    fit_counts = np.bincount(labels[labels >= 0], minlength=reps.shape[0])
    retained = fit_counts >= int(min_fit_count)
    if not np.any(retained):
        return FamilyCodebook(
            representatives=np.empty((0, masks.shape[1]), dtype=bool),
            fit_counts=np.empty(0, dtype=np.int64),
        )
    reps = reps[retained]
    labels, _ = assign_codebook(masks, reps, min_jaccard=min_jaccard)
    fit_counts = np.bincount(labels[labels >= 0], minlength=reps.shape[0]).astype(np.int64)
    retained = fit_counts >= int(min_fit_count)
    return FamilyCodebook(reps[retained], fit_counts[retained])


def assign_codebook(
    masks: np.ndarray,
    representatives: np.ndarray,
    *,
    min_jaccard: float,
) -> tuple[np.ndarray, np.ndarray]:
    if representatives.shape[0] == 0:
        return (
            np.full(masks.shape[0], -1, dtype=np.int64),
            np.zeros(masks.shape[0], dtype=np.float64),
        )
    scores = jaccard_scores(masks.astype(bool), representatives.astype(bool))
    labels = scores.argmax(axis=1).astype(np.int64)
    similarities = scores[np.arange(scores.shape[0]), labels]
    labels[similarities < float(min_jaccard)] = -1
    return labels, similarities


def matched_topk_masks(dense: np.ndarray, sparse_masks: np.ndarray) -> np.ndarray:
    if dense.shape != sparse_masks.shape or dense.ndim != 2:
        raise ValueError("dense values and sparse masks must have the same [state, latent] shape")
    cardinality = sparse_masks.sum(axis=1).astype(np.int64)
    order = np.argsort(-np.abs(dense), axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])
    return ranks < cardinality[:, None]


def ordinary_permutations(dimension: int, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    return [rng.permutation(int(dimension)) for _ in range(int(count))]


def _safe_root_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= EPS else math.sqrt(numerator / denominator)


@torch.no_grad()
def closure_metrics(
    latents: torch.Tensor,
    masks: torch.Tensor,
    matrix: torch.Tensor,
    *,
    horizon: int,
    state_batch_size: int,
) -> dict[str, float | None]:
    """Score one fixed initial projector per trajectory in row-vector convention."""

    if latents.ndim != 3 or masks.shape != (latents.shape[0], latents.shape[2]):
        raise ValueError("Expected latents [trajectory,time,latent] and masks [trajectory,latent]")
    if horizon >= latents.shape[1]:
        raise ValueError("Horizon exceeds latent trajectory")
    source_states = latents[:, :horizon].reshape(-1, latents.shape[-1])
    target_states = latents[:, 1 : horizon + 1].reshape(-1, latents.shape[-1])
    expanded_masks = masks[:, None, :].expand(-1, horizon, -1).reshape(-1, masks.shape[-1])
    sums = {key: 0.0 for key in (
        "outside", "gated", "change", "source", "state", "target_outside", "target",
        "global_residual", "identity_residual", "restricted_inside_residual", "target_inside",
    )}
    for start in range(0, source_states.shape[0], int(state_batch_size)):
        stop = min(source_states.shape[0], start + int(state_batch_size))
        z = source_states[start:stop]
        z_next = target_states[start:stop]
        mask = expanded_masks[start:stop].to(dtype=z.dtype)
        source = z * mask
        gated = source @ matrix
        outside = gated * (1.0 - mask)
        restricted = gated * mask
        target_inside = z_next * mask
        global_prediction = z @ matrix
        tensors = {
            "outside": outside,
            "gated": gated,
            "change": gated - source,
            "source": source,
            "state": z,
            "target_outside": z_next * (1.0 - mask),
            "target": z_next,
            "global_residual": global_prediction - z_next,
            "identity_residual": z - z_next,
            "restricted_inside_residual": restricted - target_inside,
            "target_inside": target_inside,
        }
        for key, value in tensors.items():
            sums[key] += float(value.double().square().sum().item())
    k_leakage = _safe_root_ratio(sums["outside"], sums["gated"])
    change_leakage = _safe_root_ratio(sums["outside"], sums["change"])
    identity = _safe_root_ratio(sums["identity_residual"], sums["target"])
    global_residual = _safe_root_ratio(sums["global_residual"], sums["target"])
    return {
        "activity_k_leakage_rms": k_leakage,
        "activity_kminusI_leakage_rms": change_leakage,
        "source_capture_rms": _safe_root_ratio(sums["source"], sums["state"]),
        "encoded_future_outside_rms": _safe_root_ratio(sums["target_outside"], sums["target"]),
        "restricted_inside_residual_rms": _safe_root_ratio(
            sums["restricted_inside_residual"], sums["target_inside"]
        ),
        "global_latent_residual_rms": global_residual,
        "identity_latent_residual_rms": identity,
        "global_k_over_identity_residual": (
            None if identity is None or identity <= EPS or global_residual is None
            else global_residual / identity
        ),
    }


@torch.no_grad()
def matrix_leakage_metrics(masks: torch.Tensor, matrix: torch.Tensor) -> dict[str, float | None]:
    mask = masks.to(dtype=matrix.dtype)
    matrix_sq = matrix.square()
    change_sq = (matrix - torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)).square()
    selected_outputs = mask @ matrix_sq
    outside = float((selected_outputs * (1.0 - mask)).double().sum().item())
    total = float(selected_outputs.double().sum().item())
    change_total = float((mask @ change_sq).double().sum().item())
    return {
        "matrix_k_leakage_fro": _safe_root_ratio(outside, total),
        "matrix_kminusI_leakage_fro": _safe_root_ratio(outside, change_total),
    }


def summarize_null(records: list[dict[str, float | None]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in records[0] if records else ():
        values = [item[key] for item in records]
        clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        result[key] = float(np.median(clean)) if clean else None
    return result


@torch.no_grad()
def operator_distance(representatives: torch.Tensor, matrix: torch.Tensor) -> float | None:
    if representatives.shape[0] < 2:
        return None
    masks = representatives.to(dtype=matrix.dtype)
    change_sq = (matrix - torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)).square()
    norm_sq = ((masks @ change_sq) * masks).sum(dim=1).double()
    distances: list[float] = []
    for left in range(masks.shape[0]):
        for right in range(left + 1, masks.shape[0]):
            overlap = masks[left] * masks[right]
            inner = float(((overlap @ change_sq) * overlap).sum().double().item())
            n_left, n_right = math.sqrt(float(norm_sq[left])), math.sqrt(float(norm_sq[right]))
            if n_left > 0 and n_right > 0:
                squared = max(0.0, n_left * n_left + n_right * n_right - 2.0 * inner)
                distances.append(math.sqrt(squared) / (0.5 * (n_left + n_right)))
    return float(np.mean(distances)) if distances else None


@torch.no_grad()
def restricted_operator_signature(mask: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Low-order orthogonal-similarity invariants of one active (A-I) block."""

    indices = torch.nonzero(mask.to(dtype=torch.bool), as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError("An operator signature requires a nonempty support")
    dimension = int(indices.numel())
    block = matrix.index_select(0, indices).index_select(1, indices).double()
    block = block - torch.eye(dimension, device=matrix.device, dtype=torch.float64)
    mu = torch.trace(block) / dimension
    symmetric_centered = 0.5 * (block + block.T) - mu * torch.eye(
        dimension, device=matrix.device, dtype=torch.float64
    )
    skew = 0.5 * (block - block.T)
    signature = torch.stack((
        mu,
        torch.linalg.vector_norm(symmetric_centered) / math.sqrt(dimension),
        torch.linalg.vector_norm(skew) / math.sqrt(dimension),
    ))
    if not bool(torch.isfinite(signature).all()):
        raise FloatingPointError("Nonfinite restricted-operator signature")
    return signature


@torch.no_grad()
def operator_signature_distance(
    representatives: torch.Tensor, matrix: torch.Tensor
) -> float | None:
    """Mean normalized signature distance; used with exactly two frozen families."""

    if representatives.shape[0] < 2:
        return None
    signatures = [restricted_operator_signature(mask, matrix) for mask in representatives]
    distances = []
    for left in range(len(signatures)):
        for right in range(left + 1, len(signatures)):
            denominator = torch.linalg.vector_norm(signatures[left]) + torch.linalg.vector_norm(
                signatures[right]
            )
            if float(denominator) <= EPS:
                return None
            value = 2.0 * torch.linalg.vector_norm(
                signatures[left] - signatures[right]
            ) / denominator
            distances.append(float(value.item()))
    return float(np.mean(distances)) if distances else None


@torch.no_grad()
def decoded_rollout_metrics(
    model: torch.nn.Module,
    fields: torch.Tensor,
    masks: torch.Tensor,
    *,
    horizons: list[int],
    batch_size: int,
    family_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    max_horizon = max(horizons)
    device = next(model.parameters()).device
    mode_names = ("full", "mask_once", "restricted")
    totals = {h: {name: [0.0, 0, 0.0, 0] for name in mode_names} for h in horizons}
    family_totals: dict[int, dict[int, dict[str, list[float | int]]]] = {}
    for start in range(0, fields.shape[0], int(batch_size)):
        stop = min(fields.shape[0], start + int(batch_size))
        batch = fields[start:stop].to(device)
        mask = masks[start:stop].to(device=device, dtype=batch.dtype)
        z0 = model.encode(batch[:, 0])
        if not torch.equal(z0 * mask, z0 * mask * mask):
            raise AssertionError("Projector is not Boolean")
        states = torch.cat((z0, z0 * mask, z0 * mask), dim=0)
        cumulative = torch.zeros(3, stop - start, device=device, dtype=torch.float64)
        local_labels = None if family_labels is None else family_labels[start:stop]
        for step in range(1, max_horizon + 1):
            states = model.step_latent(states)
            states[2 * (stop - start) :] *= mask
            prediction = model.decode(states)
            truth = batch[:, step]
            for mode_index, name in enumerate(mode_names):
                pred = prediction[mode_index * (stop - start) : (mode_index + 1) * (stop - start)]
                per_trajectory_sse = (pred - truth).double().square().sum(dim=1)
                cumulative[mode_index] += per_trajectory_sse
                if step in horizons:
                    totals[step][name][0] += float(cumulative[mode_index].sum().item())
                    totals[step][name][1] += int((stop - start) * step * truth.shape[1])
                    totals[step][name][2] += float(per_trajectory_sse.sum().item())
                    totals[step][name][3] += int((stop - start) * truth.shape[1])
                    if local_labels is not None:
                        for family in sorted(set(int(value) for value in local_labels if value >= 0)):
                            keep = torch.as_tensor(local_labels == family, device=device)
                            record = family_totals.setdefault(family, {}).setdefault(
                                step, {key: [0.0, 0, 0.0, 0] for key in mode_names}
                            )[name]
                            record[0] += float(cumulative[mode_index][keep].sum().item())
                            record[1] += int(keep.sum().item() * step * truth.shape[1])
                            record[2] += float(per_trajectory_sse[keep].sum().item())
                            record[3] += int(keep.sum().item() * truth.shape[1])
    def finish(record: list[float | int]) -> dict[str, float]:
        return {
            "field_mse": float(record[0]) / max(1, int(record[1])),
            "terminal_field_mse": float(record[2]) / max(1, int(record[3])),
        }
    result: dict[str, Any] = {
        str(h): {name: finish(totals[h][name]) for name in mode_names} for h in horizons
    }
    if family_labels is not None:
        result["families"] = {
            str(family): {
                "trajectory_count": int(np.sum(family_labels == family)),
                **{
                    str(h): {name: finish(records[name]) for name in mode_names}
                    for h, records in by_horizon.items()
                },
            }
            for family, by_horizon in family_totals.items()
        }
    return result
