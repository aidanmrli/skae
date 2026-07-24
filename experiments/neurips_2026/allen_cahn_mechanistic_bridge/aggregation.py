"""Crossed reductions and preregistered bridge decisions."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from experiments.neurips_2026.allen_cahn_mechanistic_bridge.io import finite_tree
from experiments.neurips_2026.allen_cahn_mechanistic_bridge.statistics import (
    difference_summary,
    ratio_summary,
)


def _matrix(
    rows: list[dict[str, Any]],
    models: list[int],
    datasets: list[int],
    getter: Callable[[dict[str, Any]], float],
) -> np.ndarray:
    selected = [
        row for row in rows
        if row["model_seed"] in models and row["dataset_seed"] in datasets
    ]
    by_key = {(row["model_seed"], row["dataset_seed"]): row for row in selected}
    expected = {(model, dataset) for model in models for dataset in datasets}
    if len(selected) != len(expected) or set(by_key) != expected:
        raise RuntimeError("Crossed bridge rows are incomplete or duplicated")
    return np.asarray(
        [[getter(by_key[(model, dataset)]) for dataset in datasets] for model in models],
        dtype=np.float64,
    )


def _metric_summary(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if metric.endswith("mse"):
        return ratio_summary(candidate, control, replicates=replicates, seed=seed)
    return difference_summary(candidate, control, replicates=replicates, seed=seed)


def _forecast_aggregate(
    rows: list[dict[str, Any]],
    models: list[int],
    datasets: list[int],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon_index, horizon in enumerate((160, 200, 400)):
        result[str(horizon)] = {}
        for metric_index, metric in enumerate(
            ("through_mse", "terminal_mse", "modal_fate_accuracy")
        ):
            sparse = _matrix(
                rows, models, datasets,
                lambda row, h=str(horizon), m=metric: row["ordinary"]["sparse"][h][m],
            )
            dense = _matrix(
                rows, models, datasets,
                lambda row, h=str(horizon), m=metric: row["ordinary"]["dense"][h][m],
            )
            result[str(horizon)][metric] = _metric_summary(
                sparse, dense, metric=metric, replicates=replicates,
                seed=seed + 10 * horizon_index + metric_index,
            )
    return result


def _alignment_aggregate(
    rows: list[dict[str, Any]],
    models: list[int],
    datasets: list[int],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"by_horizon": {}}
    for offset, horizon in enumerate((200, 400)):
        sparse_ari = _matrix(
            rows, models, datasets,
            lambda row, h=str(horizon): row["alignment"]["sparse"][h]["ari"],
        )
        dense_ari = _matrix(
            rows, models, datasets,
            lambda row, h=str(horizon): row["alignment"]["dense"][h]["ari"],
        )
        sparse_coverage = _matrix(
            rows, models, datasets,
            lambda row, h=str(horizon): row["alignment"]["sparse"][h]["coverage"],
        )
        dense_coverage = _matrix(
            rows, models, datasets,
            lambda row, h=str(horizon): row["alignment"]["dense"][h]["coverage"],
        )
        summary = difference_summary(
            sparse_ari, dense_ari, replicates=replicates, seed=seed + offset
        )
        summary.update({
            "sparse_ari_mean": float(sparse_ari.mean()),
            "dense_ari_mean": float(dense_ari.mean()),
            "sparse_coverage_mean": float(sparse_coverage.mean()),
            "dense_coverage_mean": float(dense_coverage.mean()),
            "per_dataset_sparse_coverage": sparse_coverage.mean(0).tolist(),
            "per_dataset_dense_coverage": dense_coverage.mean(0).tolist(),
            "minimum_dataset_sparse_coverage": float(sparse_coverage.mean(0).min()),
        })
        result["by_horizon"][str(horizon)] = summary
    result["primary_h200"] = result["by_horizon"]["200"]
    return result


def _probe_aggregate(
    rows: list[dict[str, Any]],
    models: list[int],
    datasets: list[int],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    sparse = _matrix(
        rows, models, datasets,
        lambda row: row["probes"]["sparse_support"]["test"]["balanced_accuracy"],
    )
    dense = _matrix(
        rows, models, datasets,
        lambda row: row["probes"]["dense_topk"]["test"]["balanced_accuracy"],
    )
    result = difference_summary(sparse, dense, replicates=replicates, seed=seed)
    result.update({
        "sparse_support_balanced_accuracy_mean": float(sparse.mean()),
        "dense_topk_balanced_accuracy_mean": float(dense.mean()),
        "per_dataset_sparse_support_balanced_accuracy": sparse.mean(0).tolist(),
        "per_dataset_dense_topk_balanced_accuracy": dense.mean(0).tolist(),
    })
    return result


def _routing_metric(
    rows: list[dict[str, Any]],
    models: list[int],
    datasets: list[int],
    horizon: str,
    arm: str,
    mode: str,
    metric: str,
) -> np.ndarray:
    return _matrix(
        rows, models, datasets,
        lambda row: row["routing"]["horizons"][horizon][arm][mode][metric],
    )


def _routing_aggregate(
    rows: list[dict[str, Any]],
    models: list[int],
    datasets: list[int],
    *,
    replicates: int,
    seed: int,
    minimum_models: int,
) -> dict[str, Any]:
    evaluable_models = [
        model for model in models
        if all(
            row["routing"]["wrong_control_count"] > 0
            and row["routing"]["paired_cardinality_exact"]
            and row["routing"]["same_subset_for_all_modes"]
            for row in rows
            if row["model_seed"] == model
        )
        and sum(row["model_seed"] == model for row in rows) == len(datasets)
    ]
    minimum = int(minimum_models)
    available = len(evaluable_models) >= minimum
    result: dict[str, Any] = {
        "all_cells_available": available,
        "minimum_evaluable_model_seeds": minimum,
        "evaluable_model_seeds": evaluable_models,
        "evaluable_model_seed_count": len(evaluable_models),
        "excluded_model_seeds": sorted(set(models) - set(evaluable_models)),
        "selection_used_forecast_errors_or_fates": False,
    }
    if not available:
        return result
    coverage = _matrix(
        rows, evaluable_models, datasets,
        lambda row: row["routing"]["wrong_control_coverage"],
    )
    result.update({
        "coverage_mean": float(coverage.mean()),
        "per_dataset_coverage": coverage.mean(0).tolist(),
        "minimum_dataset_coverage": float(coverage.mean(0).min()),
        "paired_cardinality_exact": True,
        "same_subset_for_all_modes": True,
        "by_horizon": {},
    })
    for horizon_index, horizon in enumerate(("160", "200", "400")):
        cell: dict[str, Any] = {}
        for metric_index, metric in enumerate(("through_mse", "terminal_mse")):
            correct_restricted = _routing_metric(
                rows, evaluable_models, datasets, horizon, "correct", "restricted", metric
            )
            correct_once = _routing_metric(
                rows, evaluable_models, datasets, horizon, "correct", "mask_once", metric
            )
            wrong_restricted = _routing_metric(
                rows, evaluable_models, datasets, horizon, "wrong", "restricted", metric
            )
            wrong_once = _routing_metric(
                rows, evaluable_models, datasets, horizon, "wrong", "mask_once", metric
            )
            full = _matrix(
                rows, evaluable_models, datasets,
                lambda row, h=horizon, m=metric: row["routing"]["horizons"][h]["full"][m],
            )
            base_seed = seed + 20 * horizon_index + 4 * metric_index
            cell[metric] = {
                "correct_over_wrong_restricted": ratio_summary(
                    correct_restricted, wrong_restricted,
                    replicates=replicates, seed=base_seed,
                ),
                "correct_restricted_over_full": ratio_summary(
                    correct_restricted, full, replicates=replicates, seed=base_seed + 1,
                ),
                "restriction_interaction": ratio_summary(
                    correct_restricted / correct_once,
                    wrong_restricted / wrong_once,
                    replicates=replicates, seed=base_seed + 2,
                ),
                "correct_restriction_factor_mean": float(
                    np.mean(correct_restricted / correct_once)
                ),
                "wrong_restriction_factor_mean": float(
                    np.mean(wrong_restricted / wrong_once)
                ),
            }
        correct_modal = _routing_metric(
            rows, evaluable_models, datasets, horizon, "correct", "restricted",
            "modal_fate_accuracy",
        )
        wrong_modal = _routing_metric(
            rows, evaluable_models, datasets, horizon, "wrong", "restricted",
            "modal_fate_accuracy",
        )
        cell["modal_fate_accuracy"] = difference_summary(
            correct_modal, wrong_modal, replicates=replicates,
            seed=seed + 20 * horizon_index + 10,
        )
        result["by_horizon"][horizon] = cell

    correct_capture = _matrix(
        rows, evaluable_models, datasets,
        lambda row: row["routing"]["initial_projection"]["correct_capture_fraction"],
    )
    wrong_capture = _matrix(
        rows, evaluable_models, datasets,
        lambda row: row["routing"]["initial_projection"]["wrong_capture_fraction"],
    )
    correct_reconstruction = _matrix(
        rows, evaluable_models, datasets,
        lambda row: row["routing"]["initial_projection"]["correct_reconstruction_mse"],
    )
    wrong_reconstruction = _matrix(
        rows, evaluable_models, datasets,
        lambda row: row["routing"]["initial_projection"]["wrong_reconstruction_mse"],
    )
    result["initial_projection"] = {
        "correct_minus_wrong_capture": difference_summary(
            correct_capture, wrong_capture, replicates=replicates, seed=seed + 100
        ),
        "correct_over_wrong_reconstruction": ratio_summary(
            correct_reconstruction, wrong_reconstruction,
            replicates=replicates, seed=seed + 101,
        ),
        "correct_capture_mean": float(correct_capture.mean()),
        "wrong_capture_mean": float(wrong_capture.mean()),
        "correct_reconstruction_mse_mean": float(correct_reconstruction.mean()),
        "wrong_reconstruction_mse_mean": float(wrong_reconstruction.mean()),
    }
    return result


def _h400_classification(
    rows: list[dict[str, Any]], datasets: list[int]
) -> dict[str, Any]:
    per_dataset: list[dict[str, Any]] = []
    for dataset in datasets:
        records = [row["truth_difficulty"] for row in rows if row["dataset_seed"] == dataset]
        if len(records) != 10:
            raise RuntimeError("H400 difficulty did not have ten crossed repeats")
        reference = records[0]
        for candidate in records[1:]:
            for key in (
                "continued_change_mse", "h200_change_from_initial_mse",
                "continued_change_ratio", "modal_fate_change_fraction",
            ):
                if not np.isclose(candidate[key], reference[key], rtol=0.0, atol=1e-12):
                    raise RuntimeError("H400 truth difficulty drifted across model repeats")
        per_dataset.append({"dataset_seed": dataset, **reference})
    numerator = np.asarray([item["continued_change_mse"] for item in per_dataset])
    denominator = np.asarray([
        item["h200_change_from_initial_mse"] for item in per_dataset
    ])
    changes = np.asarray([item["modal_fate_change_fraction"] for item in per_dataset])
    ratio = float(numerator.mean() / max(denominator.mean(), 1e-20))
    change = float(changes.mean())
    dynamic = bool(ratio >= 0.05 or change >= 0.05)
    return {
        "unique_dataset_count": len(per_dataset),
        "crossed_model_repeats_excluded": True,
        "per_dataset": per_dataset,
        "continued_change_ratio_of_unique_dataset_means": ratio,
        "modal_fate_change_fraction_unique_dataset_mean": change,
        "classification": (
            "dynamic_temporal_extrapolation" if dynamic
            else "asymptotic_stability_extrapolation"
        ),
    }


def aggregate(rows: list[dict[str, Any]], card: dict[str, Any]) -> dict[str, Any]:
    models = [int(value) for value in card["roster"]["model_seeds"]]
    datasets = [int(value) for value in card["new_datasets"]["seeds"]]
    reps = int(card["aggregation"]["bootstrap_replicates"])
    seed = int(card["aggregation"]["bootstrap_seed"])
    result = {
        "forecast": _forecast_aggregate(
            rows, models, datasets, replicates=reps, seed=seed
        ),
        "alignment": _alignment_aggregate(
            rows, models, datasets, replicates=reps, seed=seed + 100
        ),
        "probe": _probe_aggregate(
            rows, models, datasets, replicates=reps, seed=seed + 200
        ),
        "routing": _routing_aggregate(
            rows, models, datasets, replicates=reps, seed=seed + 300,
            minimum_models=int(
                card["primary_gates"]["routing"]["minimum_evaluable_model_seeds"]
            ),
        ),
        "h400_truth_classification": _h400_classification(rows, datasets),
    }
    if not finite_tree(result):
        raise FloatingPointError("Bridge aggregate contains None or a nonfinite value")
    return result


def decide(aggregate: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    if not finite_tree(aggregate):
        return {
            "branch": "invalid",
            "interpretation": card["decision_branches"]["invalid"],
            "checks": {},
        }
    gates = card["primary_gates"]
    forecast_checks = []
    for horizon in ("160", "200"):
        cell = aggregate["forecast"][horizon]["through_mse"]
        forecast_checks.append(
            cell["ratio_of_cell_means"]
            < gates["forecast"]["h160_h200_sparse_full_over_dense_full_ratio_below"]
            and cell["bootstrap_interval"][1]
            < gates["forecast"]["two_way_bootstrap_upper_below"]
            and cell["model_seed_candidate_wins"]
            >= gates["forecast"]["minimum_model_seed_wins_after_averaging_datasets"]
            and cell["dataset_seed_candidate_wins"]
            >= gates["forecast"]["minimum_dataset_seed_wins"]
        )
    forecast_passed = all(forecast_checks)
    alignment = aggregate["alignment"]["primary_h200"]
    alignment_passed = (
        alignment["sparse_coverage_mean"]
        >= gates["family_alignment"]["minimum_sparse_coverage"]
        and alignment["minimum_dataset_sparse_coverage"]
        >= gates["family_alignment"]["minimum_each_dataset_sparse_coverage"]
        and alignment["sparse_ari_mean"]
        >= gates["family_alignment"]["minimum_sparse_ari"]
        and alignment["difference_mean"]
        >= gates["family_alignment"]["minimum_sparse_minus_dense_topk_ari"]
        and alignment["model_seed_candidate_wins"]
        >= gates["family_alignment"]["minimum_model_seed_wins"]
        and alignment["dataset_seed_candidate_wins"]
        >= gates["family_alignment"]["minimum_dataset_seed_wins"]
        and alignment["bootstrap_interval"][0]
        > gates["family_alignment"]["two_way_bootstrap_lower_above"]
    )
    probe = aggregate["probe"]
    probe_passed = (
        probe["sparse_support_balanced_accuracy_mean"]
        >= gates["probe"]["minimum_sparse_support_balanced_accuracy"]
        and probe["difference_mean"]
        >= gates["probe"]["minimum_sparse_support_minus_dense_topk_balanced_accuracy"]
        and probe["model_seed_candidate_wins"]
        >= gates["probe"]["minimum_model_seed_wins"]
        and probe["dataset_seed_candidate_wins"]
        >= gates["probe"]["minimum_dataset_seed_wins"]
        and probe["bootstrap_interval"][0]
        > gates["probe"]["two_way_bootstrap_lower_above"]
    )
    routing = aggregate["routing"]
    routing_passed = False
    if routing["all_cells_available"]:
        h200 = routing["by_horizon"]["200"]["through_mse"]
        correct_wrong = h200["correct_over_wrong_restricted"]
        correct_full = h200["correct_restricted_over_full"]
        interaction = h200["restriction_interaction"]
        routing_passed = (
            routing["minimum_dataset_coverage"]
            >= gates["routing"]["minimum_each_dataset_route_coverage"]
            and correct_wrong["ratio_of_cell_means"]
            <= gates["routing"]["maximum_h200_correct_over_wrong_restricted_mse"]
            and correct_full["ratio_of_cell_means"]
            <= gates["routing"]["maximum_h200_correct_restricted_over_full_mse"]
            and interaction["ratio_of_cell_means"]
            < gates["routing"]["maximum_h200_restriction_interaction"]
            and min(
                correct_wrong["model_seed_candidate_wins"],
                interaction["model_seed_candidate_wins"],
            ) >= gates["routing"]["minimum_model_seed_wins"]
            and min(
                correct_wrong["dataset_seed_candidate_wins"],
                interaction["dataset_seed_candidate_wins"],
            ) >= gates["routing"]["minimum_dataset_seed_wins"]
            and max(
                correct_wrong["bootstrap_interval"][1],
                interaction["bootstrap_interval"][1],
            ) < gates["routing"]["two_way_bootstrap_upper_below"]
        )
    representation = alignment_passed and probe_passed
    if forecast_passed and representation and routing_passed:
        branch = "same_checkpoint_mechanistic_bridge"
    elif forecast_passed and representation:
        branch = "forecast_and_alignment_without_routing"
    elif forecast_passed:
        branch = "forecast_only"
    elif representation:
        branch = "representation_only"
    else:
        branch = "failed"
    return {
        "branch": branch,
        "interpretation": card["decision_branches"][branch],
        "checks": {
            "forecast": forecast_passed,
            "family_alignment": alignment_passed,
            "support_probe": probe_passed,
            "routing": routing_passed,
        },
    }
