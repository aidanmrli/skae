"""Separately gated H400 stress execution that cannot erase complete H200 rows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


def truth_difficulty(fields: torch.Tensor) -> list[dict[str, float | int]]:
    """Quantify whether the H201--H400 truth continues to evolve."""

    if tuple(fields.shape[:3]) != (3, 256, 401):
        raise ValueError("Truth-difficulty diagnostic requires the exact H400 panel")
    flat = fields.reshape(3, 256, 401, -1).to(torch.float64)
    early = (flat[:, :, 200] - flat[:, :, 0]).square().mean(dim=(1, 2))
    continued = (flat[:, :, 400] - flat[:, :, 200]).square().mean(dim=(1, 2))
    early_step = (flat[:, :, 1:201] - flat[:, :, :200]).square().mean(
        dim=(1, 2, 3)
    )
    late_step = (flat[:, :, 201:] - flat[:, :, 200:-1]).square().mean(dim=(1, 2, 3))
    if not bool(
        torch.isfinite(torch.stack((early, continued, early_step, late_step))).all()
    ):
        raise FloatingPointError("Truth-difficulty metric is nonfinite")
    result: list[dict[str, float | int]] = []
    for index in range(3):
        early_value = float(early[index].item())
        if early_value <= 0.0:
            raise FloatingPointError("Truth-difficulty denominator is nonpositive")
        early_step_value = float(early_step[index].item())
        if early_step_value <= 0.0:
            raise FloatingPointError("Early one-step truth change is nonpositive")
        continued_value = float(continued[index].item())
        result.append(
            {
                "dataset_index": index,
                "x200_minus_x0_mse": early_value,
                "x400_minus_x200_mse": continued_value,
                "continued_change_ratio": continued_value / early_value,
                "mean_h1_h200_one_step_truth_change_mse": early_step_value,
                "mean_h201_h400_one_step_truth_change_mse": float(
                    late_step[index].item()
                ),
                "late_over_early_one_step_truth_change_ratio": float(
                    late_step[index].item()
                ) / early_step_value,
            }
        )
    return result


def validate_stress_prefix(
    primary_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
) -> None:
    """Require independently computed H400 curves to reproduce H200 exactly."""

    primary = {
        (row["arm"], row["model_seed"], row["dataset_seed"], row["cadence"]): row
        for row in primary_rows
    }
    if len(primary) != len(primary_rows):
        raise RuntimeError("Primary H200 rows contain a duplicate")
    for stress in stress_rows:
        key = (
            stress["arm"],
            stress["model_seed"],
            stress["dataset_seed"],
            stress["cadence"],
        )
        if key not in primary:
            raise RuntimeError("H400 stress row lacks its independent H200 counterpart")
        for curve in ("instantaneous_field_mse", "cumulative_field_mse"):
            short = torch.as_tensor(primary[key][curve], dtype=torch.float64)
            prefix = torch.as_tensor(stress[curve][:200], dtype=torch.float64)
            torch.testing.assert_close(short, prefix, rtol=1e-12, atol=1e-14)


def evaluate_stress_cross(
    evaluate_cross: Callable[..., list[dict[str, Any]]],
    specs_and_models: Iterable[tuple[Any, torch.nn.Module]],
    fields: torch.Tensor,
    *,
    dataset_seeds: list[int],
    cadences: list[str | int],
    batch_size: int,
    max_decode_segment: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate complete policies; record nonfinite policy failures, never prefixes."""

    materialized = list(specs_and_models)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec, model in materialized:
        for cadence in cadences:
            try:
                cell_rows = evaluate_cross(
                    [(spec, model)],
                    fields,
                    dataset_seeds=dataset_seeds,
                    cadences=[cadence],
                    horizon=400,
                    batch_size=batch_size,
                    max_decode_segment=max_decode_segment,
                )
            except FloatingPointError as error:
                failures.append(
                    {
                        "arm": str(spec.arm),
                        "model_seed": int(spec.seed),
                        "cadence": cadence,
                        "status": "whole_h400_policy_nonfinite",
                        "error_type": type(error).__name__,
                        "finite_prefix_scored": False,
                    }
                )
                continue
            if len(cell_rows) != len(dataset_seeds):
                raise RuntimeError("A finite H400 policy lacks the full dataset cross")
            rows.extend(cell_rows)
    if failures:
        # Suppress the entire stress tier. Partial H400 rows remain provenance
        # only and must never be summarized as an effect.
        return rows, failures
    expected = len(materialized) * len(cadences) * len(dataset_seeds)
    if len(rows) != expected:
        raise RuntimeError("The H400 stress cross is incomplete")
    return rows, failures
