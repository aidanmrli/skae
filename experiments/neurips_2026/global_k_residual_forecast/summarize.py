"""Fail-closed seed-level adjudication of residualized one-K forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.neurips_2026.global_k_residual_forecast.diagnostic_recompute import (
    summarize_route_validity,
)
from experiments.neurips_2026.global_k_residual_forecast.protocol import (
    DEFAULT_CARD,
    DEFAULT_SOURCES,
    DEFAULT_TASKS,
    authenticate_checkpoint_roster,
    authenticate_v2_inputs,
    atomic_json,
    exact_sign_flip_pvalue,
    holm_adjust,
    load_frozen_protocol,
    load_json,
    paired_bootstrap_reduction_interval,
    publish_h500_extension,
    sha256_path,
)
from experiments.neurips_2026.global_k_residual_forecast.shard_validation import (
    validate_scientific_shards,
)
from experiments.neurips_2026.global_k_residual_forecast.validation import validate_gate

CORE_COMPARATORS = {
    "global_standard": "sparse_global_standard_reencode",
    "global_residual": "sparse_global_residual",
    "routed_nonresidual": "sparse_routed_nonresidual",
    "permutation_null": "support_permutation_null_median",
    "persistence": "persistence_identity",
}
DENSE_COMPARATORS = {
    "dense_standard": "dense_global_standard_reencode",
    "dense_residual": "dense_global_residual",
}
CURVE_METHODS = [
    "sparse_routed_residual",
    "sparse_routed_nonresidual",
    "sparse_global_residual",
    "sparse_global_standard_reencode",
    "support_permutation_null_median",
    "persistence_identity",
    "dense_global_standard_reencode",
    "dense_global_residual",
    "sparse_global_pure_k",
    "dense_global_pure_k",
]


def _seed_values(
    shards: list[dict[str, Any]], method: str, metric: str,
) -> np.ndarray:
    rows = []
    for shard in shards:
        values = []
        for dataset in shard["dataset_rows"]:
            methods = {**dataset["sparse"]["methods"], **dataset["dense"]["methods"]}
            if method == "support_permutation_null_median":
                nulls = [
                    row[metric]
                    for name, row in methods.items()
                    if name.startswith("support_permutation_null_")
                ]
                if len(nulls) != 32 or any(value is None for value in nulls):
                    values.append(None)
                else:
                    values.append(float(np.median(nulls)))
            else:
                values.append(methods[method][metric])
        if len(values) != 3 or any(value is None for value in values):
            rows.append(np.nan)
        else:
            rows.append(float(np.mean(values)))
    return np.asarray(rows, dtype=np.float64)


def _dataset_reductions(
    shards: list[dict[str, Any]], treatment: str, baseline: str, metric: str,
) -> list[float | None]:
    reductions = []
    for dataset_index in range(3):
        treatment_values, baseline_values = [], []
        for shard in shards:
            dataset = shard["dataset_rows"][dataset_index]
            methods = {**dataset["sparse"]["methods"], **dataset["dense"]["methods"]}
            treatment_values.append(methods[treatment][metric])
            if baseline == "support_permutation_null_median":
                nulls = [
                    row[metric]
                    for name, row in methods.items()
                    if name.startswith("support_permutation_null_")
                ]
                baseline_values.append(
                    None if len(nulls) != 32 or any(value is None for value in nulls)
                    else float(np.median(nulls))
                )
            else:
                baseline_values.append(methods[baseline][metric])
        if any(value is None for value in treatment_values + baseline_values):
            reductions.append(None)
        else:
            treatment_mean = float(np.mean(treatment_values))
            baseline_mean = float(np.mean(baseline_values))
            reductions.append((baseline_mean - treatment_mean) / max(baseline_mean, 1e-30))
    return reductions


def _basin_reductions(
    shards: list[dict[str, Any]], treatment: str, baseline: str,
) -> dict[str, float | None]:
    labels = {"0", "1", "2"}
    output = {}
    for label in sorted(labels):
        treatment_values, baseline_values = [], []
        for shard in shards:
            for dataset in shard["dataset_rows"]:
                row = dataset["evaluation_only_basin_stratification"]["rows"].get(label)
                if row is None:
                    return {key: None for key in labels}
                methods = row["through_h200_mse_by_method"]
                treatment_values.append(methods[treatment])
                if baseline == "support_permutation_null_median":
                    nulls = [
                        value for name, value in methods.items()
                        if name.startswith("support_permutation_null_")
                    ]
                    baseline_values.append(
                        None if len(nulls) != 32 or any(value is None for value in nulls)
                        else float(np.median(nulls))
                    )
                else:
                    baseline_values.append(methods[baseline])
        if any(value is None for value in treatment_values + baseline_values):
            output[label] = None
        else:
            treatment_mean = float(np.mean(treatment_values))
            baseline_mean = float(np.mean(baseline_values))
            output[label] = (baseline_mean - treatment_mean) / max(baseline_mean, 1e-30)
    return output


def _contrast(
    shards: list[dict[str, Any]],
    *,
    name: str,
    baseline: str,
    metric: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    treatment = _seed_values(shards, "sparse_routed_residual", metric)
    control = _seed_values(shards, baseline, metric)
    finite = bool(np.isfinite(treatment).all() and np.isfinite(control).all())
    if not finite:
        return {
            "contrast": name,
            "baseline_method": baseline,
            "valid": False,
            "reason": "one or more complete predeclared endpoint metrics is unavailable",
        }
    differences = control - treatment
    reduction = float((control.mean() - treatment.mean()) / max(control.mean(), 1e-30))
    interval = paired_bootstrap_reduction_interval(
        treatment,
        control,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "contrast": name,
        "baseline_method": baseline,
        "valid": True,
        "treatment_seed_values": treatment.tolist(),
        "baseline_seed_values": control.tolist(),
        "mean_treatment_mse": float(treatment.mean()),
        "mean_baseline_mse": float(control.mean()),
        "relative_reduction": reduction,
        "paired_seed_win_count": int(np.sum(differences > 0)),
        "paired_seed_tie_count": int(np.sum(differences == 0)),
        "exact_one_sided_sign_flip_pvalue": exact_sign_flip_pvalue(differences),
        "paired_model_seed_bootstrap_reduction_interval_95": list(interval),
        "dataset_reductions": _dataset_reductions(
            shards, "sparse_routed_residual", baseline, metric
        ),
        "evaluation_only_initial_basin_reductions": _basin_reductions(
            shards, "sparse_routed_residual", baseline
        ),
    }


def _curve_summary(shards: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for method in CURVE_METHODS:
        seed_curves = []
        for shard in shards:
            dataset_curves = []
            for dataset in shard["dataset_rows"]:
                methods = {**dataset["sparse"]["methods"], **dataset["dense"]["methods"]}
                if method == "support_permutation_null_median":
                    curves = np.asarray([
                        [np.nan if value is None else value for value in row["mean_mse_curve"]]
                        for name, row in methods.items()
                        if name.startswith("support_permutation_null_")
                    ], dtype=np.float64)
                    dataset_curves.append(np.median(curves, axis=0))
                else:
                    curve = methods[method]["mean_mse_curve"]
                    dataset_curves.append(np.asarray(
                        [np.nan if value is None else value for value in curve],
                        dtype=np.float64,
                    ))
            seed_curves.append(np.mean(np.stack(dataset_curves), axis=0))
        stacked = np.stack(seed_curves)
        mean_curve = np.mean(stacked, axis=0)
        serialized_mean = [
            float(value) if np.isfinite(value) else None for value in mean_curve
        ]
        serialized_seeds = [
            [float(value) if np.isfinite(value) else None for value in curve]
            for curve in seed_curves
        ]
        output[method] = {
            "curve_length": int(stacked.shape[1]),
            "finite_h200_model_seed_count": int(
                np.sum(np.isfinite(stacked[:, :200]).all(axis=1))
            ),
            "finite_full_curve_model_seed_count": int(
                np.sum(np.isfinite(stacked).all(axis=1))
            ),
            "mean_over_model_seeds_and_datasets": serialized_mean,
            "model_seed_curves_after_dataset_averaging": serialized_seeds,
            "descriptive_only_not_used_for_selection": True,
        }
    return output


def _apply_gates(
    contrasts: dict[str, dict[str, Any]],
    adjusted: dict[str, float],
    gates: dict[str, Any],
    *,
    require_basin_consistency: set[str],
) -> dict[str, Any]:
    rows = {}
    for name, row in contrasts.items():
        threshold = gates["minimum_relative_reduction"][name]
        checks = {
            "valid": bool(row.get("valid")),
            "minimum_relative_reduction": bool(
                row.get("valid") and row["relative_reduction"] >= float(threshold)
            ),
            "minimum_paired_seed_wins": bool(
                row.get("valid")
                and row["paired_seed_win_count"] >= int(gates["minimum_paired_seed_wins"])
            ),
            "holm_adjusted_pvalue": bool(
                row.get("valid")
                and adjusted[name] <= float(gates["maximum_holm_adjusted_pvalue"])
            ),
            "positive_bootstrap_lower_bound": bool(
                row.get("valid")
                and row["paired_model_seed_bootstrap_reduction_interval_95"][0] > 0
            ),
            "positive_on_every_dataset": bool(
                row.get("valid")
                and all(value is not None and value > 0 for value in row["dataset_reductions"])
            ),
        }
        if name in require_basin_consistency:
            checks["positive_in_every_evaluation_basin"] = bool(
                row.get("valid")
                and set(row["evaluation_only_initial_basin_reductions"]) == {"0", "1", "2"}
                and all(
                    value is not None and value > 0
                    for value in row["evaluation_only_initial_basin_reductions"].values()
                )
            )
        rows[name] = {"checks": checks, "passed": all(checks.values())}
    return {"rows": rows, "passed": all(row["passed"] for row in rows.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--expected-card-sha256", required=True)
    parser.add_argument("--expected-task-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--smoke-assessment", type=Path, required=True)
    parser.add_argument("--scientific-telemetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    card, tasks, freeze = load_frozen_protocol(
        card_path=args.card,
        task_path=args.tasks,
        source_manifest_path=args.sources,
        expected_card_sha256=args.expected_card_sha256,
        expected_task_sha256=args.expected_task_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
    )
    authenticate_v2_inputs(card)
    authenticate_checkpoint_roster(tasks)
    task_ids = [int(task["task_id"]) for task in tasks["tasks"]]
    output_root = args.input_dir.parents[1]
    smoke = validate_gate(
        args.smoke_assessment,
        mode="smoke",
        task_ids=[0],
        protocol_id=card["protocol_id"],
        freeze=freeze,
        output_root=output_root,
        thresholds=card["gpu_utilization_gate"],
    )
    telemetry = validate_gate(
        args.scientific_telemetry,
        mode="scientific",
        task_ids=task_ids,
        protocol_id=card["protocol_id"],
        freeze=freeze,
        output_root=output_root,
        thresholds=card["gpu_utilization_gate"],
    )
    shards, shard_rows = [], []
    for task in tasks["tasks"]:
        path = args.input_dir / f"task_{int(task['task_id']):02d}.json"
        shard = load_json(path)
        shards.append(shard)
        shard_rows.append({"path": str(path), "sha256": sha256_path(path)})
    validate_scientific_shards(
        shards,
        tasks=tasks,
        card=card,
        freeze=freeze,
        output_root=output_root,
        evaluator_path=Path(__file__).with_name("evaluate.py"),
    )

    statistics = card["statistics"]
    core = {
        name: _contrast(
            shards,
            name=name,
            baseline=baseline,
            metric="through_h200_mse",
            bootstrap_replicates=int(statistics["bootstrap_replicates"]),
            bootstrap_seed=int(statistics["bootstrap_seed"]) + index,
        )
        for index, (name, baseline) in enumerate(CORE_COMPARATORS.items())
    }
    core_raw = {
        name: row.get("exact_one_sided_sign_flip_pvalue", 1.0)
        for name, row in core.items()
    }
    core_adjusted = holm_adjust(core_raw)
    core_gate = _apply_gates(
        core,
        core_adjusted,
        card["mechanism_gate"],
        require_basin_consistency={"global_standard", "global_residual"},
    )

    dense = {
        name: _contrast(
            shards,
            name=name,
            baseline=baseline,
            metric="through_h200_mse",
            bootstrap_replicates=int(statistics["bootstrap_replicates"]),
            bootstrap_seed=int(statistics["bootstrap_seed"]) + 100 + index,
        )
        for index, (name, baseline) in enumerate(DENSE_COMPARATORS.items())
    }
    dense_raw = {
        name: row.get("exact_one_sided_sign_flip_pvalue", 1.0)
        for name, row in dense.items()
    }
    dense_adjusted = holm_adjust(dense_raw)
    dense_gate = _apply_gates(
        dense,
        dense_adjusted,
        card["dense_recipe_superiority_gate"],
        require_basin_consistency=set(),
    )
    route = summarize_route_validity(shards, card)

    stress = {}
    for index, name in enumerate(("global_standard", "global_residual")):
        stress[name] = _contrast(
            shards,
            name=name,
            baseline=CORE_COMPARATORS[name],
            metric="through_h500_mse",
            bootstrap_replicates=int(statistics["bootstrap_replicates"]),
            bootstrap_seed=int(statistics["bootstrap_seed"]) + 200 + index,
        )
    raw_stress_gate = bool(
        all(row.get("valid") for row in stress.values())
        and all(row["relative_reduction"] > 0 for row in stress.values())
        and all(row["paired_seed_win_count"] >= 8 for row in stress.values())
        and all(
            all(value is not None and value > 0 for value in row["dataset_reductions"])
            for row in stress.values()
        )
    )

    mechanism_supported = bool(route["passed"] and core_gate["passed"])
    dense_supported = bool(mechanism_supported and dense_gate["passed"])
    published_stress_extension = publish_h500_extension(mechanism_supported, raw_stress_gate)
    payload = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "artifact_role": "authenticated_residual_one_k_forecast_decision",
        "freeze": freeze,
        "input_authentication": {
            "smoke_assessment_path": str(args.smoke_assessment),
            "smoke_assessment_sha256": sha256_path(args.smoke_assessment),
            "scientific_telemetry_path": str(args.scientific_telemetry),
            "scientific_telemetry_sha256": sha256_path(args.scientific_telemetry),
            "scientific_shards": shard_rows,
        },
        "validity": {
            "exact_ten_seed_roster": True,
            "outcome_blind_smoke_passed": bool(smoke["passed"]),
            "scientific_gpu_gate_passed": bool(telemetry["passed"]),
            "label_free_routing_and_matched_null_gate": route,
            "primary_h200_valid": bool(route["passed"] and all(row.get("valid") for row in core.values())),
        },
        "primary_h200_physical_time": 8.0,
        "primary_contrasts": core,
        "primary_holm_adjusted_pvalues": core_adjusted,
        "mechanism_gate": core_gate,
        "dense_contextual_contrasts": dense,
        "dense_holm_adjusted_pvalues": dense_adjusted,
        "dense_recipe_superiority_gate": dense_gate,
        "stress_h500_physical_time": 20.0,
        "stress_h500_contrasts": stress,
        "stress_h500_raw_gate_passed": raw_stress_gate,
        "stress_h500_extension_supported": published_stress_extension,
        "stress_h500_publication_rule": (
            "publish only when the H200 mechanism gate and raw H500 stress gate both pass"
        ),
        "forecast_curve_summary": _curve_summary(shards),
        "decision": {
            "residualized_support_routed_one_k_forecasting_supported": mechanism_supported,
            "sparse_recipe_superiority_over_paired_exact_dense_tanh_supported": dense_supported,
            "rebuttal_promotion_permitted": bool(mechanism_supported and dense_supported),
            "claim_boundary": (
                "A positive result supports an autonomous per-step-reencoded support-routed "
                "predictor built from one unchanged K on this controlled three-basin system. "
                "It does not establish pure K^h invariance, sparsity-component causality, or "
                "generalization beyond this frozen recipe/system."
            ),
        },
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": "complete",
                "valid": payload["validity"]["primary_h200_valid"],
                "mechanism_supported": mechanism_supported,
                "dense_recipe_superiority_supported": dense_supported,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
