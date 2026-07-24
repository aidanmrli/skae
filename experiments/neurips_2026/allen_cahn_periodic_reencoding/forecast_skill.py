"""Absolute persistence skill and inference-cost descriptors."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    ARMS,
    DIRECT,
    TEST_HORIZON,
    VALIDATION_HORIZON,
    _frozen_card,
    _selected_cadences,
)


def deployment_cost(cadence: str | int, horizon: int) -> dict[str, int]:
    """Return exact model-call counts implied by boundary-only reencoding."""

    if cadence == DIRECT:
        refreshes = 0
    else:
        period = int(cadence)
        if period <= 0:
            raise ValueError("Periodic cadence must be positive")
        refreshes = (int(horizon) - 1) // period
    return {
        "horizon_steps": int(horizon),
        "refresh_count": refreshes,
        "encoder_calls": 1 + refreshes,
        "decoder_calls": int(horizon),
        "latent_k_steps": int(horizon),
    }


def cadence_cost_table(cadences: Sequence[str | int], horizon: int) -> list[dict[str, Any]]:
    return [
        {"cadence": cadence, **deployment_cost(cadence, horizon)}
        for cadence in cadences
    ]


def _curves(row: Mapping[str, Any], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    model = np.asarray(row["instantaneous_field_mse"], dtype=np.float64)
    persistence = np.asarray(row["instantaneous_persistence_mse"], dtype=np.float64)
    if (
        model.shape != (horizon,)
        or persistence.shape != (horizon,)
        or not np.isfinite(model).all()
        or not np.isfinite(persistence).all()
        or np.any(model < 0.0)
        or np.any(persistence < 0.0)
    ):
        raise FloatingPointError("Absolute-skill curves are incomplete or invalid")
    return model, persistence


def _endpoints(horizon: int) -> tuple[tuple[str, slice], ...]:
    if horizon == VALIDATION_HORIZON:
        return (("h200_cumulative_field_mse", slice(0, 200)),)
    if horizon == TEST_HORIZON:
        return (
            ("h200_cumulative_field_mse", slice(0, 200)),
            ("h400_cumulative_field_mse", slice(0, 400)),
            ("h201_h400_tail_field_mse", slice(200, 400)),
            ("h400_terminal_field_mse", slice(399, 400)),
        )
    raise ValueError("Absolute skill supports only H200 or H400 rows")


def summarize_selected_absolute_skill(
    rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    selected_cadences: Mapping[str, Any],
    *,
    horizon: int,
) -> dict[str, Any]:
    """Compare each validation-selected arm policy with deployable x0 persistence."""

    frozen = _frozen_card(card)
    selected = _selected_cadences(selected_cadences, frozen)
    lookup: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        arm = row.get("arm")
        if arm not in ARMS or row.get("cadence") != selected[arm]:
            continue
        if int(row.get("horizon_steps", -1)) != horizon:
            raise ValueError("Selected absolute-skill row has the wrong horizon")
        key = (arm, int(row["model_seed"]), int(row["dataset_seed"]))
        if key in lookup:
            raise ValueError("Duplicate selected absolute-skill row")
        lookup[key] = _curves(row, horizon)
    expected = {
        (arm, model_seed, dataset_seed)
        for arm in ARMS
        for model_seed in frozen["model_seeds"]
        for dataset_seed in frozen["test_seeds"]
    }
    if set(lookup) != expected:
        raise ValueError("Selected absolute-skill rows do not form the exact cross")
    endpoint_report: dict[str, Any] = {}
    for endpoint_name, window in _endpoints(horizon):
        arm_report = {}
        for arm in ARMS:
            model_seed_values = []
            persistence_seed_values = []
            for model_seed in frozen["model_seeds"]:
                pairs = [
                    lookup[(arm, model_seed, dataset_seed)]
                    for dataset_seed in frozen["test_seeds"]
                ]
                model_seed_values.append(np.mean([pair[0][window].mean() for pair in pairs]))
                persistence_seed_values.append(
                    np.mean([pair[1][window].mean() for pair in pairs])
                )
            model_values = np.asarray(model_seed_values)
            persistence_values = np.asarray(persistence_seed_values)
            persistence_mean = float(persistence_values.mean())
            if persistence_mean <= 0.0:
                raise ValueError("Persistence endpoint mean is nonpositive")
            per_dataset = []
            for dataset_seed in frozen["test_seeds"]:
                pairs = [
                    lookup[(arm, model_seed, dataset_seed)]
                    for model_seed in frozen["model_seeds"]
                ]
                model_mean = float(np.mean([pair[0][window].mean() for pair in pairs]))
                baseline = float(np.mean([pair[1][window].mean() for pair in pairs]))
                if baseline <= 0.0:
                    raise ValueError("A per-dataset persistence mean is nonpositive")
                per_dataset.append(
                    {
                        "dataset_seed": int(dataset_seed),
                        "model_mse": model_mean,
                        "x0_persistence_mse": baseline,
                        "model_over_x0_persistence": model_mean / baseline,
                    }
                )
            arm_report[arm] = {
                "model_mse": float(model_values.mean()),
                "x0_persistence_mse": persistence_mean,
                "model_over_x0_persistence": float(model_values.mean()) / persistence_mean,
                "model_seed_wins_over_persistence": int(
                    np.sum(model_values < persistence_values)
                ),
                "all_three_dataset_ratios_below_one": all(
                    row["model_over_x0_persistence"] < 1.0 for row in per_dataset
                ),
                "per_dataset": per_dataset,
            }
        endpoint_report[endpoint_name] = arm_report
    return {
        "selected_cadences": selected,
        "endpoints": endpoint_report,
        "baseline": "deployable_x0_persistence",
        "cost_by_arm": {
            arm: deployment_cost(selected[arm], horizon) for arm in ARMS
        },
    }
