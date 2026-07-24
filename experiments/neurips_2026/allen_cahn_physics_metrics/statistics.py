"""Frozen paired-seed reductions for secondary Allen--Cahn physics metrics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from experiments.neurips_2026.allen_cahn_physics_metrics.core import (
    HORIZON,
    METRIC_NAMES,
    METRIC_SPECS,
)


def _studentized(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=-1)
    scales = values.std(axis=-1, ddof=1) / np.sqrt(values.shape[-1])
    result = np.zeros_like(means, dtype=np.float64)
    np.divide(means, scales, out=result, where=scales > 0)
    result[(scales == 0) & (means > 0)] = np.inf
    result[(scales == 0) & (means < 0)] = -np.inf
    return result


def exact_one_sided_sign_flip(improvements: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(improvements, dtype=np.float64)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Exact paired sign flip requires ten finite model-seed improvements")
    observed = float(_studentized(values[None, :])[0])
    integers = np.arange(2**10, dtype=np.uint16)[:, None]
    bits = (integers >> np.arange(10, dtype=np.uint16)) & 1
    signs = 2.0 * bits.astype(np.float64) - 1.0
    permuted = _studentized(signs * values[None, :])
    return {
        "observed_studentized_statistic": observed,
        "one_sided_exact_p": float(np.mean(permuted >= observed)),
        "enumerated_sign_vectors": 1024,
        "alternative": "oriented_sparse_improvement_is_positive",
        "literal_comparison": "T_perm_greater_than_or_equal_to_T_observed",
    }


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    if set(raw) != set(METRIC_NAMES):
        raise ValueError("Holm family must contain every frozen physics metric")
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (count - rank) * float(raw[name]))
        adjusted[name] = min(1.0, running)
    return {name: adjusted[name] for name in METRIC_NAMES}


def paired_bootstrap(
    improvements_dense: np.ndarray,
    improvements_sparse: np.ndarray,
    *,
    direction: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    dense = np.asarray(improvements_dense, dtype=np.float64)
    sparse = np.asarray(improvements_sparse, dtype=np.float64)
    if dense.shape != (10,) or sparse.shape != (10,) or not np.isfinite([dense, sparse]).all():
        raise ValueError("Bootstrap requires two finite paired ten-seed arrays")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, 10, size=(int(replicates), 10))
    dense_samples = dense[indices].mean(axis=1)
    sparse_samples = sparse[indices].mean(axis=1)
    if direction == "lower":
        samples = dense_samples - sparse_samples
        point = float(dense.mean() - sparse.mean())
    elif direction == "higher":
        samples = sparse_samples - dense_samples
        point = float(sparse.mean() - dense.mean())
    else:
        raise ValueError(f"Unknown metric direction {direction}")
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {
        "oriented_absolute_improvement": point,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "replicates": int(replicates),
        "seed": int(seed),
        "resampling_unit": "paired_model_seed_after_three_dataset_average",
    }


def _paired_curves(
    rows: list[dict[str, Any]], metric: str, kind: str
) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {"dense": [], "sparse": []}
    for arm in ("dense", "sparse"):
        for seed in range(64, 74):
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["arm"] == arm and int(row["model_seed"]) == seed
                ],
                key=lambda row: int(row["dataset_index"]),
            )
            if [int(row["dataset_index"]) for row in selected] != [0, 1, 2]:
                raise AssertionError("Paired-seed curve lacks the exact three-dataset roster")
            values = np.asarray([row["curves"][metric][kind] for row in selected], dtype=np.float64)
            if values.shape != (3, HORIZON) or not np.isfinite(values).all():
                raise FloatingPointError("Paired-seed physics curve is incomplete")
            result[arm].append(values.mean(axis=0))
    arrays = {arm: np.asarray(values, dtype=np.float64) for arm, values in result.items()}
    if any(value.shape != (10, HORIZON) for value in arrays.values()):
        raise AssertionError("Physics reduction does not have ten paired model-seed curves")
    return arrays


def _control_curve(rows: list[dict[str, Any]], metric: str, kind: str) -> np.ndarray:
    selected = sorted(
        [row for row in rows if row["arm"] == "persistence"],
        key=lambda row: int(row["dataset_index"]),
    )
    if [int(row["dataset_index"]) for row in selected] != [0, 1, 2]:
        raise AssertionError("Persistence control lacks the exact three-dataset roster")
    values = np.asarray([row["curves"][metric][kind] for row in selected], dtype=np.float64)
    if values.shape != (3, HORIZON) or not np.isfinite(values).all():
        raise FloatingPointError("Persistence physics curve is incomplete")
    return values.mean(axis=0)


def _oriented(dense: np.ndarray, sparse: np.ndarray, direction: str) -> np.ndarray:
    return dense - sparse if direction == "lower" else sparse - dense


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def arm_mean_effect_summary(
    dense_mean: float, sparse_mean: float, direction: str
) -> dict[str, float | str]:
    """Report a direction-correct effect, including the zero-baseline case."""

    if not np.isfinite([dense_mean, sparse_mean]).all():
        raise ValueError("Arm-mean effect requires finite inputs")
    if direction == "lower":
        if dense_mean > 0:
            return {
                "label": "relative_error_reduction_of_arm_means",
                "value": float(1.0 - sparse_mean / dense_mean),
            }
        return {
            "label": "oriented_absolute_error_improvement_dense_minus_sparse",
            "value": float(dense_mean - sparse_mean),
        }
    if direction == "higher":
        return {
            "label": "accuracy_percentage_point_difference",
            "value": float(100.0 * (sparse_mean - dense_mean)),
        }
    raise ValueError(f"Unknown metric direction {direction}")


def _tie_diagnostic_curves(rows: list[dict[str, Any]]) -> dict[str, Any]:
    controls = sorted(
        [row for row in rows if row["arm"] == "persistence"],
        key=lambda row: int(row["dataset_index"]),
    )
    if [int(row["dataset_index"]) for row in controls] != [0, 1, 2]:
        raise AssertionError("Tie diagnostics lack the three persistence cells")
    truth = np.asarray(
        [row["diagnostics"]["truth_modal_tie_rate"] for row in controls],
        dtype=np.float64,
    ).mean(axis=0)
    persistence = np.asarray(
        [row["diagnostics"]["candidate_modal_tie_rate"] for row in controls],
        dtype=np.float64,
    ).mean(axis=0)
    result: dict[str, Any] = {
        "truth_modal_tie_rate": truth.tolist(),
        "persistence_modal_tie_rate": persistence.tolist(),
    }
    for arm in ("dense", "sparse"):
        seed_curves = []
        for seed in range(64, 74):
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["arm"] == arm and int(row["model_seed"]) == seed
                ],
                key=lambda row: int(row["dataset_index"]),
            )
            values = np.asarray(
                [row["diagnostics"]["candidate_modal_tie_rate"] for row in selected],
                dtype=np.float64,
            )
            if values.shape != (3, HORIZON) or not np.isfinite(values).all():
                raise FloatingPointError("Candidate tie diagnostic is incomplete")
            seed_curves.append(values.mean(axis=0))
        result[f"{arm}_modal_tie_rate"] = np.asarray(seed_curves).mean(axis=0).tolist()
    if any(
        np.asarray(values).shape != (HORIZON,) or not np.isfinite(values).all()
        for values in result.values()
    ):
        raise FloatingPointError("A reduced tie diagnostic is incomplete")
    result["policy"] = "ties_reported_without_exclusion_or_relabeling"
    return result


def classify_secondary_pattern(
    directional_count: int, directional_families: set[str]
) -> str:
    """Apply the three mutually exclusive frozen descriptive branches."""

    required_families = {spec.family for spec in METRIC_SPECS}
    if not 0 <= int(directional_count) <= len(METRIC_SPECS):
        raise ValueError("Directional metric count lies outside the frozen roster")
    if not directional_families <= required_families:
        raise ValueError("Directional family lies outside the frozen roster")
    if directional_count <= 2:
        return "little_or_reversed_secondary_translation"
    if directional_count >= 5 and directional_families == required_families:
        return "broad_secondary_concordance"
    return "mixed_secondary_translation"


def validate_bootstrap_seed_schedule(
    schedule: Mapping[str, int],
) -> dict[str, int]:
    if tuple(schedule) != METRIC_NAMES:
        raise ValueError("Bootstrap seed schedule must follow the exact seven-metric roster")
    seeds = [schedule[name] for name in METRIC_NAMES]
    if (
        any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(METRIC_NAMES)
    ):
        raise ValueError("Bootstrap seed schedule must contain seven unique nonnegative integers")
    return {name: int(schedule[name]) for name in METRIC_NAMES}


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seeds_by_metric: Mapping[str, int],
) -> dict[str, Any]:
    bootstrap_seeds = validate_bootstrap_seed_schedule(bootstrap_seeds_by_metric)
    raw_p: dict[str, float] = {}
    metrics: dict[str, Any] = {}
    directional_count = 0
    directional_families: set[str] = set()
    for spec in METRIC_SPECS:
        instantaneous = _paired_curves(rows, spec.name, "instantaneous")
        cumulative = _paired_curves(rows, spec.name, "cumulative")
        persistence_instantaneous = _control_curve(rows, spec.name, "instantaneous")
        persistence_cumulative = _control_curve(rows, spec.name, "cumulative")
        dense_primary = cumulative["dense"][:, -1]
        sparse_primary = cumulative["sparse"][:, -1]
        improvements = _oriented(dense_primary, sparse_primary, spec.direction)
        test = exact_one_sided_sign_flip(improvements)
        raw_p[spec.name] = float(test["one_sided_exact_p"])
        if float(improvements.mean()) > 0:
            directional_count += 1
            directional_families.add(spec.family)
        endpoints = {
            "h160_cumulative": {
                "dense": float(cumulative["dense"][:, 159].mean()),
                "sparse": float(cumulative["sparse"][:, 159].mean()),
                "persistence": float(persistence_cumulative[159]),
            },
            "h160_terminal": {
                "dense": float(instantaneous["dense"][:, 159].mean()),
                "sparse": float(instantaneous["sparse"][:, 159].mean()),
                "persistence": float(persistence_instantaneous[159]),
            },
            "h200_cumulative": {
                "dense": float(dense_primary.mean()),
                "sparse": float(sparse_primary.mean()),
                "persistence": float(persistence_cumulative[-1]),
            },
            "h200_terminal": {
                "dense": float(instantaneous["dense"][:, -1].mean()),
                "sparse": float(instantaneous["sparse"][:, -1].mean()),
                "persistence": float(persistence_instantaneous[-1]),
            },
            "late_t10p1_to_t20": {
                "dense": float(instantaneous["dense"][:, 100:].mean()),
                "sparse": float(instantaneous["sparse"][:, 100:].mean()),
                "persistence": float(persistence_instantaneous[100:].mean()),
            },
        }
        for endpoint in endpoints.values():
            endpoint["sparse_over_dense"] = _safe_ratio(endpoint["sparse"], endpoint["dense"])
            endpoint["dense_over_persistence"] = _safe_ratio(
                endpoint["dense"], endpoint["persistence"]
            )
            endpoint["sparse_over_persistence"] = _safe_ratio(
                endpoint["sparse"], endpoint["persistence"]
            )
        dense_mean = float(dense_primary.mean())
        sparse_mean = float(sparse_primary.mean())
        arm_mean_effect = arm_mean_effect_summary(
            dense_mean, sparse_mean, spec.direction
        )
        metrics[spec.name] = {
            "direction": spec.direction,
            "family": spec.family,
            "h200_cumulative_paired_seed_dense": dense_primary.tolist(),
            "h200_cumulative_paired_seed_sparse": sparse_primary.tolist(),
            "h200_cumulative_paired_seed_improvement": improvements.tolist(),
            "h200_cumulative_seed_wins": int(np.sum(improvements > 0)),
            "arm_mean_effect": arm_mean_effect,
            "exact_secondary_test": test,
            "paired_bootstrap": paired_bootstrap(
                dense_primary,
                sparse_primary,
                direction=spec.direction,
                replicates=bootstrap_replicates,
                seed=bootstrap_seeds[spec.name],
            ),
            "mandatory_endpoints": endpoints,
            "full_curves": {
                "dense_instantaneous_mean": instantaneous["dense"].mean(axis=0).tolist(),
                "sparse_instantaneous_mean": instantaneous["sparse"].mean(axis=0).tolist(),
                "persistence_instantaneous": persistence_instantaneous.tolist(),
                "dense_cumulative_mean": cumulative["dense"].mean(axis=0).tolist(),
                "sparse_cumulative_mean": cumulative["sparse"].mean(axis=0).tolist(),
                "persistence_cumulative": persistence_cumulative.tolist(),
            },
        }
    adjusted = holm_adjust(raw_p)
    for name in METRIC_NAMES:
        metrics[name]["exact_secondary_test"]["holm_adjusted_p_across_seven_metrics"] = adjusted[name]
    families = {spec.family for spec in METRIC_SPECS}
    pattern = classify_secondary_pattern(directional_count, directional_families)
    return {
        "metrics": metrics,
        "modal_tie_diagnostics": _tie_diagnostic_curves(rows),
        "secondary_pattern": {
            "classification": pattern,
            "directionally_sparse_better_metrics": directional_count,
            "directionally_positive_families": sorted(directional_families),
            "required_families": sorted(families),
            "policy": "classification_is_descriptive_and_cannot_reclassify_prior_field_MSE_inference",
        },
        "multiplicity": {
            "family": list(METRIC_NAMES),
            "method": "Holm across all seven H200 cumulative secondary metrics",
            "raw_one_sided_p": raw_p,
            "adjusted_p": adjusted,
        },
    }
