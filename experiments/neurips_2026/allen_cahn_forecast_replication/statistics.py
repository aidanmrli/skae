"""Frozen paired-model-seed inference and descriptive curve reduction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_forecast_replication.core import (
    HORIZON,
    validate_crossed_rows,
)


def _studentized(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=-1)
    scales = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
    result = np.zeros_like(means, dtype=np.float64)
    np.divide(means, scales, out=result, where=scales > 0)
    result[(scales == 0) & (means > 0)] = np.inf
    result[(scales == 0) & (means < 0)] = -np.inf
    return result


def exact_paired_sign_flip(differences: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Exact paired sign-flip test requires ten finite differences")
    observed = float(_studentized(values[None, :])[0])
    integers = np.arange(2**10, dtype=np.uint16)[:, None]
    bits = (integers >> np.arange(10, dtype=np.uint16)) & 1
    signs = 2.0 * bits.astype(np.float64) - 1.0
    permuted = _studentized(signs * values[None, :])
    p_value = float(np.mean(permuted >= observed))
    return {
        "observed_studentized_statistic": observed,
        "one_sided_exact_p": p_value,
        "enumerated_sign_vectors": 1024,
        "alternative": "mean_dense_minus_sparse_is_positive",
        "comparison": "T_perm >= T_observed_literal_no_tolerance",
    }


def paired_seed_bootstrap(
    sparse: Sequence[float],
    dense: Sequence[float],
    *,
    replicates: int = 100_000,
    seed: int = 20_260_720,
) -> dict[str, Any]:
    sparse_values = np.asarray(sparse, dtype=np.float64)
    dense_values = np.asarray(dense, dtype=np.float64)
    if sparse_values.shape != (10,) or dense_values.shape != (10,):
        raise ValueError("Paired bootstrap requires ten sparse and ten dense seed rows")
    if not np.isfinite(sparse_values).all() or not np.isfinite(dense_values).all():
        raise ValueError("Paired bootstrap received a nonfinite value")
    if np.any(dense_values <= 0):
        raise ValueError("Dense MSE must be strictly positive for an unclipped ratio")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, 10, size=(int(replicates), 10))
    dense_means = dense_values[indices].mean(axis=1)
    if np.any(dense_means <= 0):
        raise ValueError("Bootstrap produced a nonpositive dense mean")
    samples = 1.0 - sparse_values[indices].mean(axis=1) / dense_means
    lower, upper = np.quantile(samples, (0.025, 0.975))
    point = 1.0 - sparse_values.mean() / dense_values.mean()
    return {
        "relative_reduction_of_arm_means": float(point),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "replicates": int(replicates),
        "seed": int(seed),
        "resampling_unit": "paired_model_seed_after_three_dataset_average",
    }


def _seed_values(
    rows: list[dict[str, Any]],
    card: dict[str, Any],
    *,
    curve: str,
    horizon_index: int,
) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = {"dense": [], "sparse": []}
    dataset_seeds = [int(value) for value in card["prospective_datasets"]["seeds"]]
    for arm in ("dense", "sparse"):
        for model_seed in card["checkpoint_roster"]["model_seeds"]:
            selected = [
                row
                for row in rows
                if row["arm"] == arm and int(row["model_seed"]) == int(model_seed)
            ]
            selected.sort(key=lambda row: dataset_seeds.index(int(row["dataset_seed"])))
            if [int(row["dataset_seed"]) for row in selected] != dataset_seeds:
                raise AssertionError("Model-seed reduction lacks the exact three-dataset roster")
            result[arm].append(
                float(np.mean([float(row[curve][horizon_index]) for row in selected]))
            )
    arrays = {arm: np.asarray(values, dtype=np.float64) for arm, values in result.items()}
    if any(value.shape != (10,) or not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Paired model-seed reduction is incomplete or nonfinite")
    return arrays


def _descriptive_endpoint(
    rows: list[dict[str, Any]],
    card: dict[str, Any],
    *,
    curve: str,
    horizon: int,
) -> dict[str, Any]:
    values = _seed_values(rows, card, curve=curve, horizon_index=int(horizon) - 1)
    dense = values["dense"]
    sparse = values["sparse"]
    if float(dense.mean()) <= 0:
        raise ValueError("Dense endpoint mean is not positive")
    return {
        "horizon": int(horizon),
        "curve": curve,
        "dense_mean": float(dense.mean()),
        "sparse_mean": float(sparse.mean()),
        "relative_reduction_of_arm_means": float(1.0 - sparse.mean() / dense.mean()),
        "paired_model_seed_wins": int(np.sum(sparse < dense)),
        "inference_policy": "secondary_descriptive_no_test_no_interval_no_rescue",
    }


def _paired_seed_curve_arrays(
    rows: list[dict[str, Any]],
    card: dict[str, Any],
    *,
    curve: str,
) -> dict[str, np.ndarray]:
    dataset_seeds = [int(value) for value in card["prospective_datasets"]["seeds"]]
    result: dict[str, list[np.ndarray]] = {"dense": [], "sparse": []}
    for arm in ("dense", "sparse"):
        for model_seed in card["checkpoint_roster"]["model_seeds"]:
            selected = [
                row
                for row in rows
                if row["arm"] == arm and int(row["model_seed"]) == int(model_seed)
            ]
            selected.sort(key=lambda row: dataset_seeds.index(int(row["dataset_seed"])))
            if [int(row["dataset_seed"]) for row in selected] != dataset_seeds:
                raise AssertionError("Curve visualization lacks a complete three-dataset seed row")
            values = np.asarray([row[curve] for row in selected], dtype=np.float64)
            if values.shape != (3, HORIZON) or not np.isfinite(values).all():
                raise FloatingPointError("Curve visualization input is incomplete or nonfinite")
            result[arm].append(values.mean(axis=0))
    arrays = {arm: np.asarray(values, dtype=np.float64) for arm, values in result.items()}
    if any(value.shape != (10, HORIZON) for value in arrays.values()):
        raise AssertionError("Curve visualization is not ten paired model-seed curves")
    return arrays


def pointwise_paired_curve_bands(
    dense: np.ndarray,
    sparse: np.ndarray,
    *,
    replicates: int = 50_000,
    seed: int = 20_260_721,
    chunk_size: int = 1_000,
) -> dict[str, Any]:
    if dense.shape != (10, HORIZON) or sparse.shape != (10, HORIZON):
        raise ValueError("Pointwise bands require paired 10x200 model-seed curves")
    if not np.isfinite(dense).all() or not np.isfinite(sparse).all() or np.any(dense <= 0):
        raise ValueError("Pointwise bands received nonfinite values or a nonpositive denominator")
    dense_samples = np.empty((int(replicates), HORIZON), dtype=np.float64)
    sparse_samples = np.empty_like(dense_samples)
    reduction_samples = np.empty_like(dense_samples)
    generator = np.random.default_rng(seed)
    for start in range(0, int(replicates), int(chunk_size)):
        stop = min(int(replicates), start + int(chunk_size))
        indices = generator.integers(0, 10, size=(stop - start, 10))
        dense_chunk = dense[indices].mean(axis=1)
        sparse_chunk = sparse[indices].mean(axis=1)
        if np.any(dense_chunk <= 0):
            raise ValueError("Pointwise bootstrap produced a nonpositive dense arm mean")
        dense_samples[start:stop] = dense_chunk
        sparse_samples[start:stop] = sparse_chunk
        reduction_samples[start:stop] = 1.0 - sparse_chunk / dense_chunk

    def interval(values: np.ndarray) -> dict[str, list[float]]:
        quantiles = np.quantile(values, (0.025, 0.975), axis=0)
        return {"lower": quantiles[0].tolist(), "upper": quantiles[1].tolist()}

    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "chunk_size": int(chunk_size),
        "resampling_unit": "paired_model_seed_after_three_dataset_curve_average",
        "dense_cumulative_field_mse": interval(dense_samples),
        "sparse_cumulative_field_mse": interval(sparse_samples),
        "cumulative_relative_reduction": {
            "point": (1.0 - sparse.mean(axis=0) / dense.mean(axis=0)).tolist(),
            **interval(reduction_samples),
        },
        "coverage_policy": "pointwise_descriptive_not_simultaneous_no_test_no_rescue",
    }


def descriptive_arm_curves(
    rows: list[dict[str, Any]],
    card: dict[str, Any],
    *,
    bootstrap_replicates: int = 50_000,
    bootstrap_seed: int = 20_260_721,
    bootstrap_chunk_size: int = 1_000,
) -> dict[str, Any]:
    source_names = (
        "instantaneous_field_mse",
        "cumulative_field_mse",
        "instantaneous_persistence_mse",
        "cumulative_persistence_mse",
        "instantaneous_model_over_persistence",
        "cumulative_model_over_persistence",
    )
    paired_seed_curves: dict[str, dict[str, list[list[float]]]] = {
        "dense": {},
        "sparse": {},
    }
    arm_means: dict[str, dict[str, list[float]]] = {"dense": {}, "sparse": {}}
    cumulative_arrays: dict[str, np.ndarray] | None = None
    for name in source_names:
        arrays = _paired_seed_curve_arrays(rows, card, curve=name)
        for arm in ("dense", "sparse"):
            paired_seed_curves[arm][name] = arrays[arm].tolist()
            arm_means[arm][name] = arrays[arm].mean(axis=0).tolist()
        if name == "cumulative_field_mse":
            cumulative_arrays = arrays
    if cumulative_arrays is None:
        raise AssertionError("Cumulative field-MSE curves were not reduced")
    dense_instant = np.asarray(arm_means["dense"]["instantaneous_field_mse"])
    sparse_instant = np.asarray(arm_means["sparse"]["instantaneous_field_mse"])
    dense_cumulative = np.asarray(arm_means["dense"]["cumulative_field_mse"])
    sparse_cumulative = np.asarray(arm_means["sparse"]["cumulative_field_mse"])
    if np.any(dense_instant <= 0) or np.any(dense_cumulative <= 0):
        raise ValueError("Dense arm curve has a nonpositive denominator")
    ratios = {
        "sparse_over_dense_instantaneous_field_mse": (sparse_instant / dense_instant).tolist(),
        "sparse_over_dense_cumulative_field_mse": (sparse_cumulative / dense_cumulative).tolist(),
    }
    if not np.isfinite(np.asarray(list(ratios.values()))).all():
        raise FloatingPointError("Descriptive sparse/dense curve ratio is nonfinite")
    return {
        "horizons": list(range(1, HORIZON + 1)),
        "paired_model_seed_curves_after_three_dataset_average": paired_seed_curves,
        "arm_means": arm_means,
        "sparse_over_dense_ratios": ratios,
        "pointwise_paired_seed_bootstrap": pointwise_paired_curve_bands(
            cumulative_arrays["dense"],
            cumulative_arrays["sparse"],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            chunk_size=bootstrap_chunk_size,
        ),
        "aggregation_order": "average_three_datasets_within_each_model_seed_then_average_ten_model_seeds",
        "inference_policy": "pointwise_descriptive_bands_not_simultaneous_no_tests_no_horizon_selection_no_rescue",
    }


def summarize_rows(rows: list[dict[str, Any]], card: dict[str, Any]) -> dict[str, Any]:
    validate_crossed_rows(rows, card)
    primary_values = _seed_values(
        rows,
        card,
        curve="cumulative_field_mse",
        horizon_index=HORIZON - 1,
    )
    dense = primary_values["dense"]
    sparse = primary_values["sparse"]
    interval = paired_seed_bootstrap(sparse, dense)
    exact = exact_paired_sign_flip(dense - sparse)
    dataset_effects: list[dict[str, Any]] = []
    for dataset_seed in card["prospective_datasets"]["seeds"]:
        arm_means = {}
        for arm in ("dense", "sparse"):
            values = [
                float(row["cumulative_field_mse"][-1])
                for row in rows
                if row["arm"] == arm and int(row["dataset_seed"]) == int(dataset_seed)
            ]
            if len(values) != 10 or not np.isfinite(values).all():
                raise AssertionError("Dataset-specific primary reduction is incomplete")
            arm_means[arm] = float(np.mean(values))
        if arm_means["dense"] <= 0:
            raise ValueError("Dataset-specific dense mean is not positive")
        dataset_effects.append(
            {
                "dataset_seed": int(dataset_seed),
                "dense_mean": arm_means["dense"],
                "sparse_mean": arm_means["sparse"],
                "relative_reduction": 1.0 - arm_means["sparse"] / arm_means["dense"],
            }
        )
    effect = float(interval["relative_reduction_of_arm_means"])
    wins = int(np.sum(sparse < dense))
    strong = bool(
        effect >= 0.05
        and float(interval["ci95_lower"]) > 0.0
        and float(exact["one_sided_exact_p"]) <= 0.05
        and wins >= 8
        and all(item["sparse_mean"] < item["dense_mean"] for item in dataset_effects)
    )
    if strong:
        branch = "strong_replication"
    elif effect > 0.0:
        branch = "directional_but_below_strong_gate"
    else:
        branch = "null_or_reversal"
    secondary = {
        "h160_cumulative": _descriptive_endpoint(
            rows, card, curve="cumulative_field_mse", horizon=160
        ),
        "h160_terminal": _descriptive_endpoint(
            rows, card, curve="instantaneous_field_mse", horizon=160
        ),
        "h200_terminal": _descriptive_endpoint(
            rows, card, curve="instantaneous_field_mse", horizon=200
        ),
    }
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "status": branch,
        "claim_boundary": card["claim_boundary"],
        "mandatory_branch_disclosure": card["decision_branches"][branch],
        "primary": {
            "endpoint": "H200 cumulative through-horizon field MSE",
            "dense_mean": float(dense.mean()),
            "sparse_mean": float(sparse.mean()),
            "paired_model_seed_wins": wins,
            "paired_model_seed_values": {
                "dense": dense.tolist(),
                "sparse": sparse.tolist(),
            },
            "dataset_effects": dataset_effects,
            "bootstrap": interval,
            "exact_sign_flip": exact,
            "strong_gate_passed": strong,
        },
        "secondary": secondary,
        "descriptive_curves": descriptive_arm_curves(rows, card),
    }
