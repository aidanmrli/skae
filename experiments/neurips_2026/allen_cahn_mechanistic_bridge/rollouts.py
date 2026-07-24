"""Per-trajectory direct and fixed-support rollout artifacts."""

from __future__ import annotations

from typing import Any

import torch


def _empty_result(horizons: list[int], modes: tuple[str, ...]) -> dict[str, Any]:
    return {
        mode: {
            str(horizon): {
                "through_mse": [],
                "terminal_mse": [],
                "final_prediction": [],
                "finite": [],
            }
            for horizon in horizons
        }
        for mode in modes
    }


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    for by_horizon in result.values():
        for record in by_horizon.values():
            for key in ("through_mse", "terminal_mse", "final_prediction", "finite"):
                record[key] = torch.cat(record[key], dim=0)
    return result


@torch.no_grad()
def rollout_full(
    model: torch.nn.Module,
    fields: torch.Tensor,
    *,
    horizons: list[int],
    batch_size: int,
) -> dict[str, Any]:
    max_horizon = max(horizons)
    if fields.shape[1] <= max_horizon:
        raise ValueError("Ground truth is shorter than requested direct rollout")
    device = next(model.parameters()).device
    result = _empty_result(horizons, ("full",))
    for start in range(0, fields.shape[0], int(batch_size)):
        batch = fields[start : start + int(batch_size)].to(device)
        state = model.encode(batch[:, 0])
        cumulative = torch.zeros(batch.shape[0], device=device, dtype=torch.float64)
        for step in range(1, max_horizon + 1):
            state = model.step_latent(state)
            prediction = model.decode(state)
            error = (prediction - batch[:, step]).double().square()
            cumulative += error.sum(1)
            if step in horizons:
                record = result["full"][str(step)]
                record["through_mse"].append((cumulative / (step * error.shape[1])).cpu())
                record["terminal_mse"].append(error.mean(1).cpu())
                record["final_prediction"].append(prediction.float().cpu())
                record["finite"].append(torch.isfinite(prediction).all(1).cpu())
    return _finish(result)["full"]


@torch.no_grad()
def rollout_projected_modes(
    model: torch.nn.Module,
    fields: torch.Tensor,
    masks: torch.Tensor,
    *,
    horizons: list[int],
    batch_size: int,
) -> dict[str, Any]:
    if masks.shape[0] != fields.shape[0]:
        raise ValueError("Projected masks and trajectories are unpaired")
    max_horizon = max(horizons)
    device = next(model.parameters()).device
    result = _empty_result(horizons, ("full", "mask_once", "restricted"))
    for start in range(0, fields.shape[0], int(batch_size)):
        batch = fields[start : start + int(batch_size)].to(device)
        mask = masks[start : start + int(batch_size)].to(device=device, dtype=batch.dtype)
        z0 = model.encode(batch[:, 0])
        states = torch.cat((z0, z0 * mask, z0 * mask), dim=0)
        cumulative = torch.zeros(3, batch.shape[0], device=device, dtype=torch.float64)
        for step in range(1, max_horizon + 1):
            states = model.step_latent(states)
            states[2 * batch.shape[0] :] *= mask
            predictions = model.decode(states)
            for mode_index, mode in enumerate(("full", "mask_once", "restricted")):
                prediction = predictions[
                    mode_index * batch.shape[0] : (mode_index + 1) * batch.shape[0]
                ]
                error = (prediction - batch[:, step]).double().square()
                cumulative[mode_index] += error.sum(1)
                if step in horizons:
                    record = result[mode][str(step)]
                    record["through_mse"].append(
                        (cumulative[mode_index] / (step * error.shape[1])).cpu()
                    )
                    record["terminal_mse"].append(error.mean(1).cpu())
                    record["final_prediction"].append(prediction.float().cpu())
                    record["finite"].append(torch.isfinite(prediction).all(1).cpu())
    return _finish(result)


@torch.no_grad()
def rollout_restricted(
    model: torch.nn.Module,
    fields: torch.Tensor,
    masks: torch.Tensor,
    *,
    horizons: list[int],
    batch_size: int,
) -> dict[str, Any]:
    if masks.shape[0] != fields.shape[0]:
        raise ValueError("Restricted masks and trajectories are unpaired")
    max_horizon = max(horizons)
    device = next(model.parameters()).device
    result = _empty_result(horizons, ("restricted",))
    for start in range(0, fields.shape[0], int(batch_size)):
        batch = fields[start : start + int(batch_size)].to(device)
        mask = masks[start : start + int(batch_size)].to(device=device, dtype=batch.dtype)
        state = model.encode(batch[:, 0]) * mask
        cumulative = torch.zeros(batch.shape[0], device=device, dtype=torch.float64)
        for step in range(1, max_horizon + 1):
            state = model.step_latent(state) * mask
            prediction = model.decode(state)
            error = (prediction - batch[:, step]).double().square()
            cumulative += error.sum(1)
            if step in horizons:
                record = result["restricted"][str(step)]
                record["through_mse"].append((cumulative / (step * error.shape[1])).cpu())
                record["terminal_mse"].append(error.mean(1).cpu())
                record["final_prediction"].append(prediction.float().cpu())
                record["finite"].append(torch.isfinite(prediction).all(1).cpu())
    return _finish(result)["restricted"]


@torch.no_grad()
def initial_projection_controls(
    model: torch.nn.Module,
    fields: torch.Tensor,
    correct_masks: torch.Tensor,
    wrong_masks: torch.Tensor,
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Measure t0 capture/reconstruction before either repeated restriction."""

    if correct_masks.shape != wrong_masks.shape or correct_masks.shape[0] != fields.shape[0]:
        raise ValueError("Correct and wrong masks must be paired to the same trajectories")
    output: dict[str, list[torch.Tensor]] = {
        "correct_capture_fraction": [], "wrong_capture_fraction": [],
        "correct_reconstruction_mse": [], "wrong_reconstruction_mse": [],
        "correct_cardinality": [], "wrong_cardinality": [], "jaccard": [],
    }
    device = next(model.parameters()).device
    for start in range(0, fields.shape[0], int(batch_size)):
        batch = fields[start : start + int(batch_size)].to(device)
        correct = correct_masks[start : start + int(batch_size)].to(device=device)
        wrong = wrong_masks[start : start + int(batch_size)].to(device=device)
        z0 = model.encode(batch[:, 0])
        correct_z, wrong_z = z0 * correct, z0 * wrong
        decoded = model.decode(torch.cat((correct_z, wrong_z), dim=0))
        denominator = z0.double().square().sum(1).clamp_min(1e-20)
        count = batch.shape[0]
        output["correct_capture_fraction"].append(
            (correct_z.double().square().sum(1) / denominator).cpu()
        )
        output["wrong_capture_fraction"].append(
            (wrong_z.double().square().sum(1) / denominator).cpu()
        )
        output["correct_reconstruction_mse"].append(
            (decoded[:count] - batch[:, 0]).double().square().mean(1).cpu()
        )
        output["wrong_reconstruction_mse"].append(
            (decoded[count:] - batch[:, 0]).double().square().mean(1).cpu()
        )
        output["correct_cardinality"].append(correct.sum(1).long().cpu())
        output["wrong_cardinality"].append(wrong.sum(1).long().cpu())
        union = torch.logical_or(correct, wrong).sum(1)
        intersection = torch.logical_and(correct, wrong).sum(1)
        output["jaccard"].append(
            torch.where(union > 0, intersection.double() / union, torch.ones_like(union).double()).cpu()
        )
    return {key: torch.cat(values) for key, values in output.items()}


@torch.no_grad()
def rollout_support_contrast(
    model: torch.nn.Module,
    fields: torch.Tensor,
    correct_masks: torch.Tensor,
    wrong_masks: torch.Tensor,
    *,
    horizons: list[int],
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate correct/wrong mask-once and restriction on one paired subset."""

    if correct_masks.shape != wrong_masks.shape or correct_masks.shape[0] != fields.shape[0]:
        raise ValueError("Correct and wrong masks must be paired to the same trajectories")
    modes = (
        "correct_mask_once", "correct_restricted", "wrong_mask_once",
        "wrong_restricted",
    )
    result = _empty_result(horizons, modes)
    max_horizon = max(horizons)
    device = next(model.parameters()).device
    for start in range(0, fields.shape[0], int(batch_size)):
        batch = fields[start : start + int(batch_size)].to(device)
        correct = correct_masks[start : start + int(batch_size)].to(device=device)
        wrong = wrong_masks[start : start + int(batch_size)].to(device=device)
        z0 = model.encode(batch[:, 0])
        states = torch.cat((z0 * correct, z0 * correct, z0 * wrong, z0 * wrong), dim=0)
        cumulative = torch.zeros(4, batch.shape[0], device=device, dtype=torch.float64)
        for step in range(1, max_horizon + 1):
            states = model.step_latent(states)
            states[batch.shape[0] : 2 * batch.shape[0]] *= correct
            states[3 * batch.shape[0] :] *= wrong
            predictions = model.decode(states)
            for mode_index, mode in enumerate(modes):
                prediction = predictions[
                    mode_index * batch.shape[0] : (mode_index + 1) * batch.shape[0]
                ]
                error = (prediction - batch[:, step]).double().square()
                cumulative[mode_index] += error.sum(1)
                if step in horizons:
                    record = result[mode][str(step)]
                    record["through_mse"].append(
                        (cumulative[mode_index] / (step * error.shape[1])).cpu()
                    )
                    record["terminal_mse"].append(error.mean(1).cpu())
                    record["final_prediction"].append(prediction.float().cpu())
                    record["finite"].append(torch.isfinite(prediction).all(1).cpu())
    return _finish(result)
