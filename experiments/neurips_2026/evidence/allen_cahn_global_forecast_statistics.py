"""Statistics and decision gate for the Allen--Cahn global-K packet."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SEEDS = tuple(range(64, 74))
HORIZONS = (80, 120, 160, 200)
DECISION_HORIZONS = (160, 200)
METRICS = ("field_mse", "final_field_mse")
TEST_CELLS = (
    (160, "field_mse", "H160 through-horizon mean"),
    (160, "final_field_mse", "H160 terminal"),
    (200, "field_mse", "H200 through-horizon mean"),
    (200, "final_field_mse", "H200 terminal"),
)


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    raise ValueError(f"Expected a boolean, got {value!r}")


def by_seed(rows: pd.DataFrame, arm: str, horizon: int, metric: str) -> np.ndarray:
    selected = rows.loc[
        (rows["arm"] == arm) & (rows["horizon"] == horizon), ["seed", metric]
    ].sort_values("seed")
    if selected["seed"].tolist() != list(SEEDS):
        raise ValueError(f"Missing paired seeds for {arm}, H{horizon}, {metric}")
    return selected[metric].to_numpy(dtype=np.float64)


def paired_bootstrap_interval(
    sparse: Sequence[float],
    dense: Sequence[float],
    *,
    replicates: int = 50_000,
    seed: int = 20_260_719,
) -> dict[str, object]:
    """Paired percentile interval for the relative reduction of arm means."""

    sparse_values = np.asarray(sparse, dtype=np.float64)
    dense_values = np.asarray(dense, dtype=np.float64)
    if sparse_values.shape != dense_values.shape or sparse_values.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be equal vectors")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, sparse_values.size, size=(replicates, sparse_values.size)
    )
    samples = 1.0 - sparse_values[indices].mean(axis=1) / np.maximum(
        dense_values[indices].mean(axis=1), 1e-12
    )
    low, high = np.quantile(samples, (0.025, 0.975))
    point = 1.0 - sparse_values.mean() / max(float(dense_values.mean()), 1e-12)
    return {
        "relative_reduction_of_means": float(point),
        "ci95_lower": float(low),
        "ci95_upper": float(high),
        "replicates": int(replicates),
        "seed": int(seed),
        "estimand": "1 - mean(sparse_seed_metric) / mean(dense_seed_metric)",
        "coverage": (
            "marginal paired-seed percentile interval; the four-cell decision "
            "is an intersection-union gate"
        ),
    }


def exact_max_t_sensitivity(rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    """One-sided exact paired sign-flip max-t sensitivity over four cells."""

    differences = np.stack(
        [
            by_seed(rows, "dense", horizon, metric)
            - by_seed(rows, "sparse", horizon, metric)
            for horizon, metric, _label in TEST_CELLS
        ]
    )

    def studentized(values: np.ndarray) -> np.ndarray:
        means = values.mean(axis=-1)
        scales = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
        statistics = np.zeros_like(means)
        np.divide(means, scales, out=statistics, where=scales > 0)
        statistics[(scales == 0) & (means > 0)] = np.inf
        statistics[(scales == 0) & (means < 0)] = -np.inf
        return statistics

    observed = studentized(differences)
    integers = np.arange(2 ** len(SEEDS), dtype=np.uint16)[:, None]
    bits = (integers >> np.arange(len(SEEDS), dtype=np.uint16)) & 1
    signs = (2.0 * bits.astype(np.float64) - 1.0)[:, None, :]
    permuted = studentized(signs * differences[None, :, :])
    maximum = permuted.max(axis=1)
    result: dict[str, dict[str, float]] = {}
    for index, (_horizon, _metric, label) in enumerate(TEST_CELLS):
        result[label] = {
            "observed_studentized_statistic": float(observed[index]),
            "one_sided_exact_sign_flip_p": float(
                np.mean(permuted[:, index] >= observed[index] - 1e-12)
            ),
            "one_sided_max_t_fwer_adjusted_p": float(
                np.mean(maximum >= observed[index] - 1e-12)
            ),
            "swaps": float(2 ** len(SEEDS)),
        }
    return result


def summarize(
    rows: pd.DataFrame,
    protocol: Mapping[str, object],
    *,
    packet_id: str,
) -> dict[str, object]:
    """Recompute qualification guards, four-cell effects, and the stop gate."""

    cells: dict[str, object] = {}
    for horizon in DECISION_HORIZONS:
        for metric in METRICS:
            dense = by_seed(rows, "dense", horizon, metric)
            sparse = by_seed(rows, "sparse", horizon, metric)
            interval = paired_bootstrap_interval(sparse, dense)
            cells[f"h{horizon}_{metric}"] = {
                "dense_mean": float(dense.mean()),
                "sparse_mean": float(sparse.mean()),
                "sparse_seed_wins": int(np.sum(sparse < dense)),
                **interval,
            }

    qualification = protocol["qualification"]
    sparse_runs = rows.loc[(rows["arm"] == "sparse") & (rows["horizon"] == 200)]
    dense_runs = rows.loc[(rows["arm"] == "dense") & (rows["horizon"] == 200)]
    all_runs = rows.loc[rows["horizon"] == 200]
    dense_audit = protocol["dense_no_sparsity_audit"]
    audit = dense_audit["audit"]
    dense_audit_passed = bool(
        audit["passed"]
        and audit["applicable"]
        and all(_as_bool(value) for value in audit["checks"].values())
        and not audit["forbidden_modules"]
        and audit["latent_output"] == "linear"
        and audit["operator_parameterization"] == "full_dense_matrix"
        and audit["operator_regularization"] == "none"
        and audit["optimizer"] == "Adam"
        and int(dense_audit["audited_seed_count"])
        == int(qualification["dense_audits_required"])
    )
    finite_passed = bool(
        np.isfinite(
            rows[
                [
                    "field_mse",
                    "final_field_mse",
                    "persistence_field_mse",
                    "persistence_final_field_mse",
                ]
            ].to_numpy(dtype=np.float64)
        ).all()
    )
    sparse_minimum = float(sparse_runs["near_zero_fraction_at_1e_minus_3"].min())
    sparsity_passed = sparse_minimum >= float(
        qualification["sparse_minimum_near_zero_fraction_at_1e_minus_3"]
    )
    utilization_minimum = float(all_runs["mean_active_gpu_utilization_percent"].min())
    utilization_passed = utilization_minimum >= float(
        qualification["minimum_mean_active_gpu_utilization_percent"]
    )
    all_qualified = bool(
        finite_passed and sparsity_passed and utilization_passed and dense_audit_passed
    )

    all_means_lower = all(
        float(cell["sparse_mean"]) < float(cell["dense_mean"])
        for cell in cells.values()
    )
    all_lower_positive = all(
        float(cell["ci95_lower"]) > 0.0 for cell in cells.values()
    )
    h200_effects = all(
        float(cells[f"h200_{metric}"]["relative_reduction_of_means"]) >= 0.05
        for metric in METRICS
    )
    h200_wins = all(
        int(cells[f"h200_{metric}"]["sparse_seed_wins"]) >= 8
        for metric in METRICS
    )
    gate_passed = bool(
        all_qualified
        and all_means_lower
        and all_lower_positive
        and h200_effects
        and h200_wins
    )
    return {
        "schema_version": 1,
        "packet_id": packet_id,
        "status": (
            "passed" if gate_passed else "confirmation_gate_failed_secondary_evidence"
        ),
        "decision": (
            "evaluate_one_sealed_holdout" if gate_passed else "terminate_allen_cahn_tuning"
        ),
        "sealed_holdout": protocol["holdout"]["status"],
        "scope": protocol["scope"],
        "comparison": {
            "passed": gate_passed,
            "all_four_means_lower": all_means_lower,
            "all_four_ci95_lower_bounds_above_zero": all_lower_positive,
            "h200_minimum_effects_passed": h200_effects,
            "h200_minimum_win_counts_passed": h200_wins,
            "cells": cells,
        },
        "max_t_sensitivity": exact_max_t_sensitivity(rows),
        "qualification": {
            "all_runs_qualified": all_qualified,
            "finite_rollouts_passed": finite_passed,
            "dense_no_sparsity_audit_passed": dense_audit_passed,
            "dense_no_sparsity_audited_seeds": int(dense_audit["audited_seed_count"]),
            "sparse_near_zero_fraction_minimum": sparse_minimum,
            "sparse_near_zero_fraction_mean": float(
                sparse_runs["near_zero_fraction_at_1e_minus_3"].mean()
            ),
            "sparse_near_zero_fraction_maximum": float(
                sparse_runs["near_zero_fraction_at_1e_minus_3"].max()
            ),
            "dense_near_zero_fraction_minimum": float(
                dense_runs["near_zero_fraction_at_1e_minus_3"].min()
            ),
            "dense_near_zero_fraction_mean": float(
                dense_runs["near_zero_fraction_at_1e_minus_3"].mean()
            ),
            "dense_near_zero_fraction_maximum": float(
                dense_runs["near_zero_fraction_at_1e_minus_3"].max()
            ),
            "sparsity_guard_passed": sparsity_passed,
            "gpu_utilization_minimum_percent": utilization_minimum,
            "gpu_utilization_mean_percent": float(
                all_runs["mean_active_gpu_utilization_percent"].mean()
            ),
            "gpu_utilization_maximum_percent": float(
                all_runs["mean_active_gpu_utilization_percent"].max()
            ),
            "gpu_utilization_guard_passed": utilization_passed,
        },
        "completeness": {
            "arms": sorted(rows["arm"].unique().tolist()),
            "seeds": sorted(int(value) for value in rows["seed"].unique()),
            "horizons": sorted(int(value) for value in rows["horizon"].unique()),
            "rows": int(len(rows)),
            "unique_evaluation_artifacts": int(rows["evaluation_sha256"].nunique()),
        },
        "protocol": {
            "state_dim": int(protocol["system"]["state_dim"]),
            "latent_dim": int(protocol["system"]["latent_dim"]),
            "overcomplete_factor": int(protocol["system"]["overcomplete_factor"]),
            "fresh_validation_trajectories": int(protocol["evaluation"]["trajectories"]),
            "rollout_mode": protocol["evaluation"]["rollout_mode"],
            "labels_used": False,
            "bootstrap_replicates": int(protocol["decision_gate"]["bootstrap_replicates"]),
            "bootstrap_seed": int(protocol["decision_gate"]["bootstrap_seed"]),
            "max_t_swaps": int(protocol["decision_gate"]["max_t_swaps"]),
        },
    }
