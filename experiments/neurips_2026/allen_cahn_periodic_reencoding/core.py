"""Strict rollout and streaming curve kernels for periodic reencoding.

Periodic refreshes use only the model's decoded prediction at a completed
segment boundary.  Ground-truth future states are accepted only by the scoring
kernel, after rollout, and can therefore never enter the forecasting state.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any

import torch
import torch.nn.functional as F


DIRECT_MODE = "direct"


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_period_candidates(
    candidates: Sequence[int],
    *,
    horizon: int,
) -> tuple[int, ...]:
    """Validate a periodic-cadence grid; direct rollout is a separate arm.

    A period equal to the horizon is allowed as an explicit no-refresh
    boundary case.  Larger periods are rejected because they duplicate that
    case without adding a distinct policy.
    """

    checked_horizon = _positive_integer(horizon, name="horizon")
    if isinstance(candidates, (str, bytes)):
        raise ValueError("period candidates must be a non-string sequence")
    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("period candidates must not be empty")
    checked = tuple(
        _positive_integer(value, name="period candidate") for value in materialized
    )
    if len(checked) != len(set(checked)):
        raise ValueError("period candidates must be unique")
    if any(period > checked_horizon for period in checked):
        raise ValueError("period candidates must not exceed the rollout horizon")
    return checked


def _model_device(model: torch.nn.Module) -> torch.device:
    if not hasattr(model, "kmat") or not isinstance(model.kmat, torch.Tensor):
        raise AttributeError("model must expose a tensor-valued kmat")
    return model.kmat.device


def _validate_float32_model(model: torch.nn.Module) -> torch.device:
    device = _model_device(model)
    tensors = [*model.parameters(), *model.buffers()]
    if not any(value is model.kmat for value in tensors):
        tensors.append(model.kmat)
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in tensors
    ):
        raise AssertionError("Periodic-reencoding evaluation requires a float32 model")
    if any(value.device != device for value in tensors):
        raise AssertionError("All model tensors must reside on one device")
    if model.kmat.ndim != 2 or model.kmat.shape[0] != model.kmat.shape[1]:
        raise ValueError("model.kmat must be a square matrix")
    if not bool(torch.isfinite(model.kmat).all()):
        raise FloatingPointError("model.kmat contains a nonfinite value")
    return device


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains a nonfinite value")


@torch.inference_mode()
def segmented_rollout(
    model: torch.nn.Module,
    x0: torch.Tensor,
    *,
    horizon: int,
    period: int | None = None,
    max_decode_segment: int = 100,
) -> torch.Tensor:
    """Forecast from ``x0`` with direct or decoded-boundary reencoding.

    ``period=None`` is direct repeated-``K`` rollout.  Otherwise the current
    latent is advanced for one segment with ``F.linear(z, model.kmat)``.  All
    states in a bounded decode chunk are decoded together.  Decode chunks never
    alter the latent state: the last decoded state is encoded only at a true
    cadence boundary when another forecast step remains.  Thus direct mode has
    exactly one encoder call, and a final unused boundary is never encoded.
    """

    checked_horizon = _positive_integer(horizon, name="horizon")
    checked_period = (
        None if period is None else _positive_integer(period, name="period")
    )
    checked_max_decode_segment = _positive_integer(
        max_decode_segment, name="max_decode_segment"
    )
    if x0.ndim != 2 or x0.shape[0] <= 0 or x0.shape[1] <= 0:
        raise ValueError("x0 must have shape [batch,state] with nonzero dimensions")
    device = _validate_float32_model(model)
    if x0.dtype != torch.float32 or x0.device != device:
        raise AssertionError("x0 must be float32 on the model device")
    _require_finite(x0, name="x0")

    model.eval()
    batch_size, state_size = x0.shape
    prediction_chunks: list[torch.Tensor] = []
    finite_checks: list[torch.Tensor] = []
    remaining = checked_horizon
    steps_since_reencode = 0

    with torch.autocast(device_type=device.type, enabled=False):
        latent = model.encode(x0)
        if latent.ndim != 2 or latent.shape[0] != batch_size:
            raise ValueError("model.encode must return [batch,latent]")
        if latent.dtype != torch.float32 or latent.device != device:
            raise AssertionError("model.encode must return float32 on the model device")
        if latent.shape[1] != model.kmat.shape[1]:
            raise ValueError("encoded width does not match model.kmat")
        finite_checks.append(torch.isfinite(latent).all())

        while remaining > 0:
            steps_to_boundary = (
                remaining
                if checked_period is None
                else checked_period - steps_since_reencode
            )
            segment_length = min(
                checked_max_decode_segment,
                steps_to_boundary,
                remaining,
            )
            latent_steps: list[torch.Tensor] = []
            for _ in range(segment_length):
                latent = F.linear(latent, model.kmat)
                latent_steps.append(latent)
            segment_latents = torch.stack(latent_steps, dim=1)
            finite_checks.append(torch.isfinite(segment_latents).all())

            decoded_flat = model.decode(
                segment_latents.reshape(batch_size * segment_length, -1)
            )
            if (
                decoded_flat.ndim != 2
                or decoded_flat.shape[0] != batch_size * segment_length
                or decoded_flat.shape[1] != state_size
            ):
                raise ValueError("model.decode must return [batch*segment,state]")
            if decoded_flat.dtype != torch.float32 or decoded_flat.device != device:
                raise AssertionError("model.decode must return float32 on the model device")
            decoded = decoded_flat.reshape(batch_size, segment_length, state_size)
            finite_checks.append(torch.isfinite(decoded).all())
            prediction_chunks.append(decoded)

            remaining -= segment_length
            if checked_period is not None:
                steps_since_reencode += segment_length
            reached_reencode_boundary = (
                checked_period is not None
                and steps_since_reencode == checked_period
            )
            if reached_reencode_boundary and remaining > 0:
                # This is the sole refresh path: the model's own boundary
                # prediction, never a future state from the scoring tensor.
                latent = model.encode(decoded[:, -1, :])
                if (
                    latent.ndim != 2
                    or tuple(latent.shape) != (batch_size, model.kmat.shape[1])
                ):
                    raise ValueError("boundary encode must return [batch,latent]")
                if latent.dtype != torch.float32 or latent.device != device:
                    raise AssertionError(
                        "boundary encode must return float32 on the model device"
                    )
                finite_checks.append(torch.isfinite(latent).all())
                steps_since_reencode = 0

    predictions = torch.cat(prediction_chunks, dim=1)
    if tuple(predictions.shape) != (batch_size, checked_horizon, state_size):
        raise AssertionError("segmented rollout did not produce the full horizon")
    if not bool(torch.stack(finite_checks).all()):
        raise FloatingPointError(
            "full rollout or an intermediate contains a nonfinite value"
        )
    return predictions


def _curves_from_sse(
    field_sse: torch.Tensor,
    persistence_sse: torch.Tensor,
    *,
    trajectory_count: int,
    state_size: int,
) -> dict[str, list[float]]:
    if field_sse.dtype != torch.float64 or persistence_sse.dtype != torch.float64:
        raise AssertionError("streaming SSE accumulators must be float64")
    if field_sse.ndim != 1 or field_sse.shape != persistence_sse.shape:
        raise ValueError("SSE curves must be equal one-dimensional tensors")
    if not bool(
        torch.stack(
            (torch.isfinite(field_sse).all(), torch.isfinite(persistence_sse).all())
        ).all()
    ):
        raise FloatingPointError("a streaming SSE curve contains a nonfinite value")

    sample_denominator = float(trajectory_count * state_size)
    horizons = torch.arange(
        1,
        field_sse.numel() + 1,
        dtype=torch.float64,
        device=field_sse.device,
    )
    instantaneous = field_sse / sample_denominator
    persistence = persistence_sse / sample_denominator
    cumulative = instantaneous.cumsum(0) / horizons
    persistence_cumulative = persistence.cumsum(0) / horizons
    curves = {
        "instantaneous_field_sse": field_sse,
        "instantaneous_persistence_sse": persistence_sse,
        "instantaneous_field_mse": instantaneous,
        "cumulative_field_mse": cumulative,
        "instantaneous_persistence_mse": persistence,
        "cumulative_persistence_mse": persistence_cumulative,
        "instantaneous_model_over_persistence": instantaneous / persistence,
        "cumulative_model_over_persistence": cumulative / persistence_cumulative,
    }
    if any(value.dtype != torch.float64 for value in curves.values()):
        raise AssertionError("all accumulated curves must remain float64")
    if not bool(
        torch.stack([torch.isfinite(value).all() for value in curves.values()]).all()
    ):
        raise FloatingPointError("a derived full-horizon curve contains a nonfinite value")
    return {name: value.detach().cpu().tolist() for name, value in curves.items()}


@torch.inference_mode()
def evaluate_model_packed(
    model: torch.nn.Module,
    fields: torch.Tensor,
    *,
    horizon: int | None = None,
    period: int | None = None,
    batch_size: int = 64,
    max_decode_segment: int = 100,
) -> list[dict[str, Any]]:
    """Stream per-horizon SSE and derived curves for packed field datasets.

    ``fields`` has configurable shape ``[dataset, trajectory, time, *state]``.
    The first two axes are packed into a single rollout-batching axis, so one
    sufficiently large batch evaluates every dataset together.  SSE is still
    reduced into distinct float64 per-dataset, per-horizon accumulators.  Every
    selected trajectory must complete the entire requested horizon finitely.
    """

    checked_batch_size = _positive_integer(batch_size, name="batch_size")
    checked_max_decode_segment = _positive_integer(
        max_decode_segment, name="max_decode_segment"
    )
    if fields.ndim < 4:
        raise ValueError("fields must have shape [dataset,trajectory,time,*state]")
    dataset_count, trajectory_count, stored_states = fields.shape[:3]
    if dataset_count <= 0 or trajectory_count <= 0 or stored_states <= 1:
        raise ValueError(
            "packed dataset, trajectory, and forecast dimensions must be nonzero"
        )
    checked_horizon = (
        stored_states - 1
        if horizon is None
        else _positive_integer(horizon, name="horizon")
    )
    if checked_horizon >= stored_states:
        raise ValueError("horizon requires more stored truth states than fields provides")
    if period is not None:
        _positive_integer(period, name="period")
    device = _validate_float32_model(model)
    if fields.dtype != torch.float32 or fields.device != device:
        raise AssertionError("fields must be float32 on the model device")
    _require_finite(fields, name="packed fields")

    state_size = prod(fields.shape[3:])
    if state_size <= 0:
        raise ValueError("flattened field state must be nonempty")
    packed_trajectory_count = dataset_count * trajectory_count
    flat_fields = fields.reshape(packed_trajectory_count, stored_states, state_size)
    field_sse = torch.zeros(
        (dataset_count, checked_horizon), dtype=torch.float64, device=device
    )
    persistence_sse = torch.zeros_like(field_sse)
    error_finite_checks: list[torch.Tensor] = []
    with torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, packed_trajectory_count, checked_batch_size):
            stop = min(packed_trajectory_count, start + checked_batch_size)
            initial = flat_fields[start:stop, 0]
            predictions = segmented_rollout(
                model,
                initial,
                horizon=checked_horizon,
                period=period,
                max_decode_segment=checked_max_decode_segment,
            )
            # Future truth is first materialized after the autonomous rollout.
            truth = flat_fields[start:stop, 1 : checked_horizon + 1]
            error = predictions - truth
            persistence_error = initial[:, None, :] - truth
            error_finite_checks.extend(
                (
                    torch.isfinite(error).all(),
                    torch.isfinite(persistence_error).all(),
                )
            )

            # Dataset-major packing makes each overlap a contiguous batch
            # slice.  Reductions remain on device, while no model call is split
            # merely because a batch crosses a dataset boundary.
            first_dataset = start // trajectory_count
            last_dataset = (stop - 1) // trajectory_count
            for dataset_index in range(first_dataset, last_dataset + 1):
                dataset_start = dataset_index * trajectory_count
                overlap_start = max(start, dataset_start) - start
                overlap_stop = min(stop, dataset_start + trajectory_count) - start
                field_sse[dataset_index].add_(
                    error[overlap_start:overlap_stop]
                    .square()
                    .sum(dim=(0, 2), dtype=torch.float64)
                )
                persistence_sse[dataset_index].add_(
                    persistence_error[overlap_start:overlap_stop]
                    .square()
                    .sum(dim=(0, 2), dtype=torch.float64)
                )
    if not bool(torch.stack(error_finite_checks).all()):
        raise FloatingPointError("a full-horizon scoring error is nonfinite")

    results: list[dict[str, Any]] = []
    for dataset_index in range(dataset_count):
        curves = _curves_from_sse(
            field_sse[dataset_index],
            persistence_sse[dataset_index],
            trajectory_count=trajectory_count,
            state_size=state_size,
        )
        results.append(
            {
                "dataset_index": dataset_index,
                "trajectory_count": trajectory_count,
                "state_size": state_size,
                "horizon": checked_horizon,
                "rollout_mode": DIRECT_MODE if period is None else "periodic_reencode",
                "period": period,
                **curves,
            }
        )
    return results
