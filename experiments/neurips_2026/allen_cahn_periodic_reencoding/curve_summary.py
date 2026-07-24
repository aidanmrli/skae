"""Frozen aggregate forecast-curve and accuracy--refresh evidence reducers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_periodic_reencoding.forecast_skill import (
    deployment_cost,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.statistics import (
    ARMS,
    _frozen_card,
)


CURVES = (
    "instantaneous_field_mse",
    "cumulative_field_mse",
    "instantaneous_persistence_mse",
    "cumulative_persistence_mse",
    "instantaneous_model_over_persistence",
    "cumulative_model_over_persistence",
)


def summarize_curve_panel(
    rows: Sequence[Mapping[str, Any]],
    card: Mapping[str, Any],
    *,
    cadences: Sequence[str | int],
    horizon: int,
    tier: str,
) -> dict[str, Any]:
    """Aggregate an exact arm x seed x dataset x cadence curve panel."""

    frozen = _frozen_card(card)
    cadence_tuple = tuple(cadences)
    if not cadence_tuple or len(set(cadence_tuple)) != len(cadence_tuple):
        raise ValueError("Curve-panel cadences must be unique and nonempty")
    expected = {
        (arm, model_seed, dataset_seed, cadence)
        for arm in ARMS
        for model_seed in frozen["model_seeds"]
        for dataset_seed in frozen["test_seeds"]
        for cadence in cadence_tuple
    }
    lookup: dict[tuple[str, int, int, str | int], dict[str, np.ndarray]] = {}
    for index, row in enumerate(rows):
        key = (
            row.get("arm"),
            int(row.get("model_seed", -1)),
            int(row.get("dataset_seed", -1)),
            row.get("cadence"),
        )
        if key not in expected:
            raise ValueError(f"Curve row {index} lies outside the requested panel")
        if key in lookup or int(row.get("horizon_steps", -1)) != horizon:
            raise ValueError("Curve panel has a duplicate or wrong horizon")
        curves = {}
        for name in CURVES:
            values = np.asarray(row.get(name), dtype=np.float64)
            if values.shape != (horizon,) or not np.isfinite(values).all():
                raise FloatingPointError(f"Curve panel has invalid {name}")
            if np.any(values < 0.0):
                raise ValueError(f"Curve panel has negative {name}")
            curves[name] = values
        lookup[key] = curves
    if set(lookup) != expected:
        raise ValueError("Curve rows do not form the exact requested cross")
    records = []
    for arm in ARMS:
        for cadence in cadence_tuple:
            per_seed: dict[str, list[np.ndarray]] = {name: [] for name in CURVES}
            for model_seed in frozen["model_seeds"]:
                cells = [
                    lookup[(arm, model_seed, dataset_seed, cadence)]
                    for dataset_seed in frozen["test_seeds"]
                ]
                for name in CURVES:
                    per_seed[name].append(
                        np.mean([cell[name] for cell in cells], axis=0)
                    )
            seed_arrays = {
                name: np.stack(values, axis=0) for name, values in per_seed.items()
            }
            record: dict[str, Any] = {
                "arm": arm,
                "cadence": cadence,
                **deployment_cost(cadence, horizon),
            }
            for name, values in seed_arrays.items():
                record[f"mean_{name}"] = values.mean(axis=0).tolist()
            record["per_seed_cumulative_field_mse"] = seed_arrays[
                "cumulative_field_mse"
            ].tolist()
            records.append(record)
    return {
        "tier": tier,
        "horizon_steps": horizon,
        "time_step": float(card["system"]["stored_dt"]),
        "cadences": list(cadence_tuple),
        "records": records,
        "aggregation": "three_datasets_within_each_seed_then_ten_seed_mean",
        "complete_curve_names": list(CURVES),
    }


def combined_accuracy_refresh_frontier(
    full_grid_h400: Mapping[str, Any] | None,
    p200: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Append the optional one-refresh point to each complete H400 frontier."""

    if full_grid_h400 is None:
        return None
    p200_complete = p200.get("status") == "complete"
    result = {}
    for endpoint_name, endpoint in full_grid_h400["endpoints"].items():
        rows = [dict(row) for row in endpoint["descriptive_accuracy_compute_frontier"]]
        if p200_complete and endpoint_name in p200["fixed_p200_sparse_vs_dense"][
            "endpoints"
        ]:
            fixed = p200["fixed_p200_sparse_vs_dense"]["endpoints"][endpoint_name]
            rows.append(
                {
                    "cadence": 200,
                    "dense_arm_mean_mse": fixed["dense_mean"],
                    "sparse_arm_mean_mse": fixed["sparse_mean"],
                    "fixed_same_cadence_sparse_over_dense_ratio_of_arm_means": fixed[
                        "sparse_over_dense_ratio_of_arm_means"
                    ],
                    "fixed_same_cadence_relative_reduction_of_arm_means": fixed[
                        "relative_reduction_of_arm_means"
                    ],
                    "refresh_count": 1,
                    "encoder_calls": 2,
                    "rollout_horizon_steps": 400,
                    "aggregation": (
                        "balanced_mean_over_ten_models_and_three_fixed_test_panels"
                    ),
                    "inference_role": "optional_descriptive_p200_not_selected",
                }
            )
        result[endpoint_name] = rows
    return {
        "endpoints": result,
        "selection_use": "none_descriptive_frontier_only",
        "p200_included": p200_complete,
    }
