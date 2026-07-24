"""Outcome-blind routing, rollout, and provenance helpers for the Allen--Cahn audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.neurips_2026.allen_cahn_support_subspaces.io import sha256_path
from experiments.neurips_2026.allen_cahn_support_subspaces.metrics import (
    assign_codebook,
    closure_metrics,
    decoded_rollout_metrics,
    fit_codebook,
    matrix_leakage_metrics,
    summarize_null,
)


REQUIRED_SOURCE_PATHS = frozenset({
    "experiments/neurips_2026/allen_cahn_support_subspaces/build_evidence.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/evaluate.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/evaluation_helpers.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/family_reduction.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/io.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/metrics.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/prediction_card.json",
    "experiments/neurips_2026/allen_cahn_support_subspaces/profile.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/reduction_statistics.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/reporting.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/select_profile.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/summarize.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/summarize_gpu_telemetry.py",
    "experiments/neurips_2026/allen_cahn_support_subspaces/validate_canary.py",
    "scripts/neurips_2026/allen_cahn_support_subspaces/run_canary_validation.sh",
    "scripts/neurips_2026/allen_cahn_support_subspaces/run_array.sh",
    "scripts/neurips_2026/allen_cahn_support_subspaces/run_profile.sh",
    "scripts/neurips_2026/allen_cahn_support_subspaces/run_summary.sh",
    "scripts/neurips_2026/allen_cahn_support_subspaces/run_v4_validation.sh",
    "tests/test_allen_cahn_support_subspaces.py",
    "tests/test_allen_cahn_support_subspaces_canary.py",
})


def verify_source_manifest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    failures, records = [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            raise RuntimeError("Blank line in frozen source manifest")
        digest, relative = line.split("  ", 1)
        if relative in records:
            raise RuntimeError(f"Duplicate frozen source path: {relative}")
        candidate_relative = Path(relative)
        if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
            raise RuntimeError(f"Unsafe frozen source path: {relative}")
        records[relative] = digest
        candidate = Path(relative)
        if sha256_path(candidate) != digest:
            failures.append(relative)
    if set(records) != REQUIRED_SOURCE_PATHS:
        missing = sorted(REQUIRED_SOURCE_PATHS - set(records))
        extra = sorted(set(records) - REQUIRED_SOURCE_PATHS)
        raise RuntimeError(f"Frozen source roster mismatch; missing={missing}, extra={extra}")
    if failures:
        raise RuntimeError(f"Frozen source hash mismatch: {failures}")
    return sha256_path(path)


def load_profile_decision(
    path: Path,
    requested_batch: int,
    *,
    card: dict[str, Any],
    card_hash: str,
    source_manifest_hash: str,
) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    contract = card["hardware_profile"]
    expected_batches = [int(value) for value in contract["candidate_batch_sizes"]]
    candidates = payload.get("candidates", [])
    observed_batches = [int(item["batch_size"]) for item in candidates]
    integrity = bool(
        payload.get("status") == "passed"
        and payload.get("synthetic_inputs_only") is True
        and payload.get("outcomes_quarantined") is True
        and payload.get("card_sha256") == card_hash
        and payload.get("source_manifest_sha256") == source_manifest_hash
        and payload.get("candidate_batch_sizes") == expected_batches
        and payload.get("selection_rule") == "smallest passing frozen candidate"
        and observed_batches == expected_batches
        and len(set(observed_batches)) == len(expected_batches)
        and int(payload.get("telemetry_interval_seconds", -1))
        == int(contract["telemetry_interval_seconds"])
    )
    recomputed_passing = []
    for item in candidates:
        profile, telemetry = item["profile"], item["telemetry"]
        batch = int(item["batch_size"])
        candidate_checks = {
            "integrity": bool(
                profile.get("status") == "completed"
                and profile.get("synthetic_inputs_only") is True
                and profile.get("outcomes_accessed") is False
                and profile.get("datasets_opened") is False
                and int(profile.get("batch_size", -1)) == batch
                and profile.get("card_sha256") == card_hash
                and profile.get("source_manifest_sha256") == source_manifest_hash
                and str(profile.get("slurm_job_id", "not_recorded")) != "not_recorded"
                and str(profile.get("slurm_job_gpus", "not_recorded")) != "not_recorded"
                and int(profile.get("visible_cuda_device_count", 0))
                == int(contract["required_visible_cuda_device_count"])
                and str(contract["required_device_name_fragment"])
                in str(profile.get("device_name", ""))
                and str(profile.get("device_uuid", "")) == str(telemetry.get("gpu_uuid", ""))
                and int(profile.get("resident_model_count", 0))
                == int(contract["resident_model_count"])
                and int(profile.get("closure_state_batch_size", 0))
                == int(contract["closure_state_batch_size"])
                and profile.get("historical_provenance_kernel_profiled") is True
                and int(profile.get("historical_reproduction_batch_size", -1))
                == int(card["inputs"]["ordinary_forecast_seed_rows"][
                    "historical_reproduction_batch_size"
                ])
                and profile.get("historical_reproduction_horizons")
                == [int(value) for value in card["inputs"][
                    "ordinary_forecast_seed_rows"
                ]["historical_evaluator_horizon_sequence"]]
                and int(telemetry.get("total_samples", 0))
                >= int(telemetry.get("active_samples", 0)) > 0
                and 0.95 * float(telemetry.get("reported_total_memory_mib", 0.0))
                <= float(profile.get("device_total_memory_bytes", 0.0)) / (1024 * 1024)
                <= 1.05 * float(telemetry.get("reported_total_memory_mib", 0.0))
            ),
            "duration": float(profile.get("profile_seconds", 0.0))
            >= float(contract["minimum_profile_seconds_each"]),
            "active_samples": int(telemetry.get("active_samples", 0))
            >= int(contract["minimum_active_samples"]),
            "mean_active_utilization": float(
                telemetry.get("mean_active_gpu_utilization_percent", 0.0)
            ) >= float(contract["minimum_mean_active_gpu_utilization_percent"]),
            "mean_all_utilization": float(
                telemetry.get("mean_all_gpu_utilization_percent", 0.0)
            ) >= float(contract["minimum_mean_all_gpu_utilization_percent"]),
            "peak_memory": float(telemetry.get("peak_memory_fraction", 1.0))
            <= float(contract["maximum_peak_memory_fraction"]),
            "profile_peak_memory": float(profile.get("peak_reserved_bytes", float("inf")))
            / max(1.0, float(profile.get("device_total_memory_bytes", 0.0)))
            <= float(contract["maximum_peak_memory_fraction"]),
        }
        candidate_passed = all(candidate_checks.values())
        profile_filename = str(item.get("profile_filename", ""))
        telemetry_filename = str(item.get("telemetry_filename", ""))
        raw_hashes_valid = bool(
            Path(profile_filename).name == profile_filename
            and Path(telemetry_filename).name == telemetry_filename
            and sha256_path(path.parent / profile_filename) == item.get("profile_sha256")
            and sha256_path(path.parent / telemetry_filename) == item.get("telemetry_sha256")
        )
        integrity = integrity and item.get("gates") == candidate_checks \
            and bool(item.get("passed")) == candidate_passed and raw_hashes_valid
        if candidate_passed:
            recomputed_passing.append(batch)
    selected = min(recomputed_passing) if recomputed_passing else None
    integrity = integrity and payload.get("selected_batch_size") == selected
    if not integrity or selected is None:
        raise RuntimeError("GPU profile decision failed independent contract validation")
    if int(selected) != int(requested_batch):
        raise RuntimeError("Requested batch is not the smallest passing frozen candidate")
    return payload, sha256_path(path)


@torch.no_grad()
def encode_states(
    model: torch.nn.Module,
    states: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    shape = states.shape
    flat = states.reshape(-1, shape[-1])
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, flat.shape[0], int(batch_size)):
        chunks.append(model.encode(flat[start : start + int(batch_size)].to(device)).cpu())
    return torch.cat(chunks).reshape(*shape[:-1], -1).contiguous()


def matrix_for_row_vectors(model: torch.nn.Module, card: dict[str, Any] | None = None) -> torch.Tensor:
    matrix = model.kmat.detach().T.contiguous()
    probe = torch.linspace(-1.0, 1.0, matrix.shape[0], device=matrix.device)[None]
    assertions = {} if card is None else card["model_assertions"]
    rtol = float(assertions.get("row_operator_probe_relative_tolerance", 1e-6))
    atol = float(assertions.get("row_operator_probe_absolute_tolerance", 1e-7))
    if not torch.allclose(model.step_latent(probe), probe @ matrix, atol=atol, rtol=rtol):
        raise AssertionError("Koopman matrix orientation mismatch")
    return matrix


def verify_forecast_reproduction(
    forecasts: dict[str, Any],
    references: dict[tuple[str, int, int], dict[str, float]],
    *,
    seed: int,
    card: dict[str, Any],
) -> dict[str, Any]:
    record = card["inputs"]["ordinary_forecast_seed_rows"]
    checks: dict[str, Any] = {}
    for arm in card["roster"]["arms"]:
        for horizon in card["roster"]["horizons"]:
            key = (str(arm), int(seed), int(horizon))
            expected = references[key]
            observed = forecasts[str(arm)][str(horizon)]["full"]
            field_pass = bool(np.isclose(
                float(observed["field_mse"]), expected["field_mse"],
                rtol=float(record["field_mse_relative_tolerance"]),
                atol=float(record["field_mse_absolute_tolerance"]),
            ))
            terminal_pass = bool(np.isclose(
                float(observed["terminal_field_mse"]), expected["terminal_field_mse"],
                rtol=float(record["terminal_field_mse_relative_tolerance"]),
                atol=float(record["terminal_field_mse_absolute_tolerance"]),
            ))
            checks[f"{arm}_H{horizon}"] = {
                "passed": field_pass and terminal_pass,
                "observed_field_mse": float(observed["field_mse"]),
                "expected_field_mse": expected["field_mse"],
                "observed_terminal_field_mse": float(observed["terminal_field_mse"]),
                "expected_terminal_field_mse": expected["terminal_field_mse"],
            }
    if not all(item["passed"] for item in checks.values()):
        raise AssertionError(f"Full-K rollout failed ordinary forecast cross-check: {checks}")
    return {"passed": True, "cells": checks}


@torch.no_grad()
def historical_forecast_reproduction_metrics(
    model: torch.nn.Module,
    fields: torch.Tensor,
    *,
    horizons: list[int],
    batch_size: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Exactly reproduce the historical full-K evaluation control flow.

    This provenance-only kernel intentionally starts a separate rollout for each
    horizon, decodes the complete ``[batch * horizon]`` latent stack, and sums
    squared residuals in float32 before transferring each batch sum to Python.
    It is not the scientific three-mode estimator used by this experiment.
    """

    if fields.dtype != torch.float32 or fields.ndim != 3:
        raise ValueError("Historical reproduction requires float32 [N,T,state] fields")
    device = next(model.parameters()).device
    result: dict[str, dict[str, dict[str, float]]] = {}
    for horizon in horizons:
        total_sse = 0.0
        total_count = 0
        terminal_sse = 0.0
        terminal_count = 0
        for start in range(0, fields.shape[0], int(batch_size)):
            batch = fields[start : start + int(batch_size)].to(device)
            truth = batch[:, 1 : int(horizon) + 1]
            _, prediction = model.rollout_observation_discrete(
                batch[:, 0], horizon=int(horizon)
            )
            if prediction.dtype != torch.float32 or prediction.shape != truth.shape:
                raise AssertionError("Historical rollout dtype or shape drifted")
            difference = prediction - truth
            total_sse += float(difference.square().sum().item())
            total_count += int(difference.numel())
            terminal = difference[:, -1]
            terminal_sse += float(terminal.square().sum().item())
            terminal_count += int(terminal.numel())
        result[str(int(horizon))] = {
            "full": {
                "field_mse": total_sse / max(1, total_count),
                "terminal_field_mse": terminal_sse / max(1, terminal_count),
            }
        }
    return result


def forecast_kernel_discrepancy(
    scientific: dict[str, Any], historical: dict[str, Any]
) -> dict[str, Any]:
    """Describe, but never gate on, floating-point differences between kernels."""

    cells: dict[str, dict[str, float]] = {}
    for horizon, historical_modes in historical.items():
        if horizon not in scientific:
            continue
        for metric in ("field_mse", "terminal_field_mse"):
            reference = float(historical_modes["full"][metric])
            observed = float(scientific[horizon]["full"][metric])
            absolute = abs(observed - reference)
            cells[f"H{horizon}_{metric}"] = {
                "scientific_three_mode": observed,
                "historical_provenance": reference,
                "absolute_difference": absolute,
                "relative_difference": absolute / max(abs(reference), 1e-20),
            }
    return {
        "descriptive_only_not_a_scientific_gate": True,
        "cells": cells,
        "maximum_relative_difference": max(
            value["relative_difference"] for value in cells.values()
        ),
    }


@torch.no_grad()
def initial_projection_diagnostics(
    model: torch.nn.Module,
    fields: torch.Tensor,
    latents: torch.Tensor,
    masks: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, float]:
    device = next(model.parameters()).device
    source_sq = projected_sq = full_reconstruction_sq = projected_reconstruction_sq = 0.0
    field_count = 0
    for start in range(0, fields.shape[0], int(batch_size)):
        stop = min(fields.shape[0], start + int(batch_size))
        x0 = fields[start:stop, 0].to(device)
        z0 = latents[start:stop].to(device)
        mask = torch.from_numpy(masks[start:stop]).to(device=device, dtype=z0.dtype)
        projected = z0 * mask
        source_sq += float(z0.double().square().sum().item())
        projected_sq += float(projected.double().square().sum().item())
        full_reconstruction_sq += float((model.decode(z0) - x0).double().square().sum().item())
        projected_reconstruction_sq += float(
            (model.decode(projected) - x0).double().square().sum().item()
        )
        field_count += int(x0.numel())
    return {
        "source_capture_rms": float(np.sqrt(projected_sq / max(source_sq, 1e-20))),
        "full_initial_reconstruction_mse": full_reconstruction_sq / max(1, field_count),
        "projected_initial_reconstruction_mse": projected_reconstruction_sq / max(1, field_count),
    }


def family_summary(
    train_masks: np.ndarray,
    score_masks: np.ndarray,
    card: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    support = card["support"]
    codebook = fit_codebook(
        train_masks,
        min_jaccard=float(support["family_jaccard"]),
        max_representatives=int(support["max_representatives"]),
        min_fit_count=int(support["min_fit_trajectories"]),
    )
    labels, similarities = assign_codebook(
        score_masks,
        codebook.representatives,
        min_jaccard=float(support["family_jaccard"]),
    )
    counts = np.bincount(labels[labels >= 0], minlength=codebook.representatives.shape[0])
    qualified = counts >= int(support["min_score_trajectories_for_per_family_estimates"])
    representatives = codebook.representatives
    fit_counts = codebook.fit_counts
    covered = labels >= 0
    qualified_covered = covered & qualified[np.maximum(labels, 0)] if qualified.size else covered & False
    score_counts = counts
    top_two_fit = sorted(
        range(representatives.shape[0]),
        key=lambda value: (-int(fit_counts[value]), int(value)),
    )[:2]
    top_two_score_qualified = len(top_two_fit) == 2 and all(
        bool(qualified[value]) for value in top_two_fit
    )
    coverage = float(covered.mean())
    top = float(score_counts.max() / max(1, score_counts.sum())) if score_counts.size else 1.0
    mean_jaccard = float(similarities[covered].mean()) if np.any(covered) else 0.0
    gates = card["family_gates"]
    eligible = bool(
        int(qualified.sum()) >= int(gates["minimum_qualified_families"])
        and top_two_score_qualified
        and coverage >= float(gates["minimum_score_coverage"])
        and float(qualified_covered.mean()) >= float(gates["minimum_qualified_family_coverage"])
        and top <= float(gates["maximum_top_family_fraction"])
        and mean_jaccard >= float(gates["minimum_mean_jaccard"])
    )
    return {
        "eligible": eligible,
        "train_fit_family_count": int(representatives.shape[0]),
        "qualified_family_count": int(qualified.sum()),
        "qualified_family_indices": np.flatnonzero(qualified).tolist(),
        "fit_frozen_top_two_family_indices": top_two_fit,
        "fit_frozen_top_two_score_qualified": top_two_score_qualified,
        "fit_counts": fit_counts.tolist(),
        "score_counts": score_counts.tolist(),
        "all_routed_score_coverage": coverage,
        "qualified_family_score_coverage": float(qualified_covered.mean()),
        "score_top_family_fraction": top,
        "score_mean_jaccard": mean_jaccard,
        "unknown_count": int((~covered).sum()),
    }, labels, representatives, qualified


@torch.no_grad()
def closure_bundle(
    latents_cpu: torch.Tensor,
    masks_numpy: np.ndarray,
    matrix: torch.Tensor,
    permutations: list[np.ndarray],
    horizons: list[int],
    state_batch_size: int,
) -> dict[str, Any]:
    device = matrix.device
    latents = latents_cpu.to(device)
    masks = torch.from_numpy(masks_numpy).to(device)
    by_horizon: dict[str, Any] = {}
    for horizon in horizons:
        true = closure_metrics(
            latents, masks, matrix, horizon=horizon, state_batch_size=state_batch_size
        )
        null_records = []
        for permutation in permutations:
            indices = torch.as_tensor(permutation, device=device)
            null_records.append(closure_metrics(
                latents.index_select(-1, indices),
                masks.index_select(-1, indices),
                matrix,
                horizon=horizon,
                state_batch_size=state_batch_size,
            ))
        by_horizon[str(horizon)] = {
            "true": true,
            "null_median": summarize_null(null_records),
            "null_replicates": null_records,
        }
    matrix_true = matrix_leakage_metrics(masks, matrix)
    matrix_null = []
    for permutation in permutations:
        indices = torch.as_tensor(permutation, device=device)
        matrix_null.append(matrix_leakage_metrics(masks.index_select(-1, indices), matrix))
    return {
        "horizons": by_horizon,
        "matrix_true": matrix_true,
        "matrix_null_median": summarize_null(matrix_null),
    }


def masks_for_labels(labels: np.ndarray, representatives: np.ndarray) -> np.ndarray:
    covered = labels >= 0
    return representatives[labels[covered]], covered


def forecast_ratios(payload: dict[str, Any]) -> dict[str, Any]:
    ratios = {}
    for horizon, record in payload.items():
        if horizon == "families":
            continue
        ratios[horizon] = {
            "mean_restricted_over_mask_once": (
                record["restricted"]["field_mse"] / record["mask_once"]["field_mse"]
            ),
            "terminal_restricted_over_mask_once": (
                record["restricted"]["terminal_field_mse"]
                / record["mask_once"]["terminal_field_mse"]
            ),
        }
    return ratios
