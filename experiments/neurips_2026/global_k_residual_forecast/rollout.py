"""Direct multistep forecast maps and endpoint-safe metric accumulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.neurips_2026.global_k_residual_forecast.protocol import (
    nearest_family,
    select_projectors,
)


def projected_decode_components(
    model, latent: torch.Tensor, projector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return D((zP)KP) and D(zP) with all leading dimensions preserved."""
    if latent.shape != projector.shape:
        raise ValueError((latent.shape, projector.shape))
    source = latent * projector.to(latent.dtype)
    stepped = (source @ model.kmatrix()) * projector.to(latent.dtype)
    decoded_step = model.decode(stepped.reshape(-1, stepped.shape[-1])).reshape(
        *stepped.shape[:-1], -1
    )
    decoded_source = model.decode(source.reshape(-1, source.shape[-1])).reshape_as(
        decoded_step
    )
    return decoded_step, decoded_source


def _method_payload(
    curve: torch.Tensor,
    per_trajectory_h200: torch.Tensor,
    per_trajectory_h500: torch.Tensor | None,
    finite_h200: bool,
    finite_h500: bool | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    curve_np = curve.detach().cpu().double().numpy()
    curve_serialized = [float(value) if np.isfinite(value) else None for value in curve_np]
    payload: dict[str, Any] = {
        "mean_mse_curve": curve_serialized,
        "finite_through_h200_for_every_trajectory": finite_h200,
        "through_h200_mse": float(np.mean(curve_np[:200])) if finite_h200 else None,
        "terminal_h200_mse": float(curve_np[199]) if finite_h200 else None,
        "late_h101_h200_mse": float(np.mean(curve_np[100:200])) if finite_h200 else None,
        "nonfinite_policy": (
            "suppress an entire predeclared endpoint if any trajectory is nonfinite "
            "within it; H200 is not retroactively rescued from a shorter prefix"
        ),
    }
    private = {"h200": per_trajectory_h200.detach().cpu().double().numpy() / 200.0}
    if per_trajectory_h500 is not None:
        payload.update(
            {
                "finite_through_h500_for_every_trajectory": bool(finite_h500),
                "through_h500_mse": (
                    float(np.mean(curve_np[:500])) if finite_h500 else None
                ),
                "terminal_h500_mse": float(curve_np[499]) if finite_h500 else None,
            }
        )
        private["h500"] = per_trajectory_h500.detach().cpu().double().numpy() / 500.0
    return payload, private


@torch.no_grad()
def sparse_forecasts(
    model,
    representatives: torch.Tensor,
    null_banks: torch.Tensor,
    truth_cpu: torch.Tensor,
    card: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    forecast = card["forecast_protocol"]
    horizon = int(forecast["stress_horizon_steps"])
    null_horizon = int(forecast["primary_horizon_steps"])
    device = next(model.parameters()).device
    truth = truth_cpu.to(device)
    x0 = truth[:, 0]
    names = [
        "sparse_routed_residual",
        "sparse_routed_nonresidual",
        "sparse_global_residual",
        "sparse_global_standard_reencode",
    ] + [f"support_permutation_null_{index:02d}" for index in range(null_banks.shape[0])]
    current = x0.unsqueeze(0).expand(len(names), -1, -1).clone()
    finite = torch.ones(len(names), dtype=torch.bool, device=device)
    finite_h200: torch.Tensor | None = None
    sums_h200 = torch.zeros((len(names), x0.shape[0]), device=device)
    sums_h500 = torch.zeros((4, x0.shape[0]), device=device)
    curves: list[list[torch.Tensor]] = [[] for _ in names]
    k_matrix = model.kmatrix()
    threshold = float(card["label_free_support_routing"]["support_threshold"])
    previous_route: torch.Tensor | None = None
    route_switches = torch.zeros((), device=device)
    route_transitions = torch.zeros((), device=device)
    route_similarity_sum = torch.zeros((), device=device)
    route_confident = torch.zeros((), device=device)
    route_count = torch.zeros((), device=device)
    route_usage = torch.zeros(representatives.shape[0], device=device)

    pure_z = model.encode(x0)
    pure_curve: list[torch.Tensor] = []
    pure_finite = torch.ones((), dtype=torch.bool, device=device)
    pure_finite_h200: torch.Tensor | None = None
    pure_h200 = torch.zeros(x0.shape[0], device=device)
    pure_h500 = torch.zeros(x0.shape[0], device=device)
    persistence_curve: list[torch.Tensor] = []
    persistence_h200 = torch.zeros(x0.shape[0], device=device)
    persistence_h500 = torch.zeros(x0.shape[0], device=device)

    for step in range(1, horizon + 1):
        active_names = names if step <= null_horizon else names[:4]
        if step == null_horizon + 1:
            current = current[:4]
        z = model.encode(current.reshape(-1, current.shape[-1])).reshape(
            len(active_names), x0.shape[0], -1
        )
        route_indices = [0, 1] + list(range(4, len(active_names)))
        route_z = z[route_indices]
        assignment, similarity = nearest_family(
            route_z.reshape(-1, route_z.shape[-1]), representatives, threshold
        )
        assignment = assignment.reshape(len(route_indices), x0.shape[0])
        similarity = similarity.reshape(len(route_indices), x0.shape[0])
        projectors = []
        for local_index in range(len(route_indices)):
            bank = representatives if local_index < 2 else null_banks[local_index - 2]
            projectors.append(select_projectors(assignment[local_index], bank))
        p = torch.stack(projectors).to(route_z.dtype)
        decoded_step, decoded_source = projected_decode_components(model, route_z, p)
        next_state = torch.empty_like(current)
        for local_index, method_index in enumerate(route_indices):
            if method_index == 1:
                next_state[method_index] = decoded_step[local_index]
            else:
                next_state[method_index] = (
                    current[method_index]
                    + decoded_step[local_index]
                    - decoded_source[local_index]
                )
        global_z = z[2:4]
        global_step = global_z @ k_matrix
        global_decoded_step = model.decode(
            global_step.reshape(-1, global_step.shape[-1])
        ).reshape(2, x0.shape[0], -1)
        global_decoded_source = model.decode(
            global_z.reshape(-1, global_z.shape[-1])
        ).reshape_as(global_decoded_step)
        next_state[2] = current[2] + global_decoded_step[0] - global_decoded_source[0]
        next_state[3] = global_decoded_step[1]

        finite[: len(active_names)] &= torch.isfinite(next_state).all(dim=(1, 2))
        error = (next_state - truth[:, step].unsqueeze(0)).square().mean(dim=-1)
        for index in range(len(active_names)):
            curves[index].append(error[index].mean())
        if step <= null_horizon:
            sums_h200[: len(active_names)] += error
        sums_h500 += error[:4]
        current = next_state

        observed_assignment = assignment[0]
        observed_similarity = similarity[0]
        if previous_route is not None:
            route_switches += (observed_assignment != previous_route).sum()
            route_transitions += observed_assignment.numel()
        previous_route = observed_assignment
        route_similarity_sum += observed_similarity.sum()
        route_confident += (
            observed_similarity >= float(
                card["label_free_support_routing"]["minimum_confident_jaccard"]
            )
        ).sum()
        route_count += observed_similarity.numel()
        route_usage += torch.bincount(
            observed_assignment, minlength=representatives.shape[0]
        )

        pure_z = pure_z @ k_matrix
        pure_x = model.decode(pure_z)
        pure_finite &= torch.isfinite(pure_x).all()
        pure_error = (pure_x - truth[:, step]).square().mean(dim=-1)
        pure_curve.append(pure_error.mean())
        if step <= null_horizon:
            pure_h200 += pure_error
        pure_h500 += pure_error

        persistence_error = (x0 - truth[:, step]).square().mean(dim=-1)
        persistence_curve.append(persistence_error.mean())
        if step <= null_horizon:
            persistence_h200 += persistence_error
        persistence_h500 += persistence_error
        if step == null_horizon:
            finite_h200 = finite.clone()
            pure_finite_h200 = pure_finite.clone()

    if finite_h200 is None or pure_finite_h200 is None:
        raise RuntimeError("H200 finiteness snapshot was not created")
    method_rows: dict[str, Any] = {}
    private: dict[str, dict[str, np.ndarray]] = {}
    for index, name in enumerate(names):
        curve = torch.stack(curves[index])
        h500 = sums_h500[index] if index < 4 else None
        row, hidden = _method_payload(
            curve, sums_h200[index], h500, bool(finite_h200[index].item()),
            bool(finite[index].item()) if index < 4 else None,
        )
        method_rows[name], private[name] = row, hidden
    row, hidden = _method_payload(
        torch.stack(pure_curve), pure_h200, pure_h500,
        bool(pure_finite_h200.item()), bool(pure_finite.item()),
    )
    method_rows["sparse_global_pure_k"], private["sparse_global_pure_k"] = row, hidden
    row, hidden = _method_payload(
        torch.stack(persistence_curve), persistence_h200, persistence_h500, True, True
    )
    method_rows["persistence_identity"], private["persistence_identity"] = row, hidden
    return {
        "methods": method_rows,
        "routing_during_sparse_routed_residual": {
            "mean_nearest_jaccard": float((route_similarity_sum / route_count).item()),
            "confident_assignment_fraction": float((route_confident / route_count).item()),
            "family_switch_fraction": float(
                (route_switches / route_transitions.clamp_min(1)).item()
            ),
            "assignment_count_by_family": route_usage.long().cpu().tolist(),
        },
    }, private


@torch.no_grad()
def dense_forecasts(
    model, truth_cpu: torch.Tensor, card: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    horizon = int(card["forecast_protocol"]["stress_horizon_steps"])
    device = next(model.parameters()).device
    truth = truth_cpu.to(device)
    x0 = truth[:, 0]
    names = ["dense_global_standard_reencode", "dense_global_residual"]
    current = x0.unsqueeze(0).expand(2, -1, -1).clone()
    finite = torch.ones(2, dtype=torch.bool, device=device)
    finite_h200: torch.Tensor | None = None
    sums_h200 = torch.zeros((2, x0.shape[0]), device=device)
    sums_h500 = torch.zeros((2, x0.shape[0]), device=device)
    curves: list[list[torch.Tensor]] = [[], []]
    k_matrix = model.kmatrix()
    pure_z = model.encode(x0)
    pure_curve: list[torch.Tensor] = []
    pure_finite = torch.ones((), dtype=torch.bool, device=device)
    pure_finite_h200: torch.Tensor | None = None
    pure_h200 = torch.zeros(x0.shape[0], device=device)
    pure_h500 = torch.zeros(x0.shape[0], device=device)
    for step in range(1, horizon + 1):
        z = model.encode(current.reshape(-1, 2)).reshape(2, x0.shape[0], -1)
        stepped = z @ k_matrix
        decoded_step = model.decode(stepped.reshape(-1, stepped.shape[-1])).reshape(
            2, x0.shape[0], 2
        )
        decoded_source = model.decode(z.reshape(-1, z.shape[-1])).reshape_as(decoded_step)
        next_state = torch.stack(
            [decoded_step[0], current[1] + decoded_step[1] - decoded_source[1]]
        )
        finite &= torch.isfinite(next_state).all(dim=(1, 2))
        error = (next_state - truth[:, step].unsqueeze(0)).square().mean(dim=-1)
        for index in range(2):
            curves[index].append(error[index].mean())
        if step <= 200:
            sums_h200 += error
        sums_h500 += error
        current = next_state

        pure_z = pure_z @ k_matrix
        pure_x = model.decode(pure_z)
        pure_finite &= torch.isfinite(pure_x).all()
        pure_error = (pure_x - truth[:, step]).square().mean(dim=-1)
        pure_curve.append(pure_error.mean())
        if step <= 200:
            pure_h200 += pure_error
        pure_h500 += pure_error
        if step == 200:
            finite_h200 = finite.clone()
            pure_finite_h200 = pure_finite.clone()

    if finite_h200 is None or pure_finite_h200 is None:
        raise RuntimeError("Dense H200 finiteness snapshot was not created")
    methods, private = {}, {}
    for index, name in enumerate(names):
        row, hidden = _method_payload(
            torch.stack(curves[index]), sums_h200[index], sums_h500[index],
            bool(finite_h200[index].item()), bool(finite[index].item()),
        )
        methods[name], private[name] = row, hidden
    row, hidden = _method_payload(
        torch.stack(pure_curve), pure_h200, pure_h500,
        bool(pure_finite_h200.item()), bool(pure_finite.item()),
    )
    methods["dense_global_pure_k"], private["dense_global_pure_k"] = row, hidden
    return {"methods": methods}, private


def stratify_after_forecasting(
    env,
    truth: torch.Tensor,
    hidden: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    labels = env.unwrapped.basin_label(truth[:, 0]).cpu().numpy().astype(np.int64)
    rows = {}
    for label in np.unique(labels):
        mask = labels == label
        rows[str(int(label))] = {
            "trajectory_count": int(mask.sum()),
            "through_h200_mse_by_method": {
                method: (
                    float(values["h200"][mask].mean())
                    if np.isfinite(values["h200"][mask]).all()
                    else None
                )
                for method, values in hidden.items()
            },
        }
    return {
        "labels_are_evaluation_only_and_were_not_passed_to_any_predictor": True,
        "label_source": "ground-truth initial-state basin_label",
        "rows": rows,
    }
