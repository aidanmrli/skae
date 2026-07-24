"""Reduce paired-seed Allen--Cahn support-subspace shards and apply frozen gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.neurips_2026.allen_cahn_support_subspaces.family_reduction import (
    family_decision,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.evaluation_helpers import (
    load_profile_decision,
    verify_source_manifest,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.io import (
    CARD_PATH,
    checkpoint_roster,
    load_card,
    sha256_path,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.reduction_statistics import (
    bootstrap_mean_interval,
    bootstrap_ratio_interval,
    exact_max_t_adjusted_p,
    finite_tree,
    median,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.reporting import write_rows
from experiments.neurips_2026.allen_cahn_support_subspaces.summarize_gpu_telemetry import (
    telemetry_receipt_checks,
)
from experiments.neurips_2026.allen_cahn_support_subspaces.validate_canary import (
    validate_release_receipt,
)


SOURCE_MANIFEST = Path(__file__).with_name("source_manifest.sha256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=CARD_PATH)
    parser.add_argument("--profile_decision", type=Path, required=True)
    parser.add_argument("--telemetry_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--expected_card_sha256", required=True)
    parser.add_argument("--expected_source_manifest_sha256", required=True)
    parser.add_argument("--expected_profile_decision_sha256", required=True)
    return parser.parse_args()


def closure_cell(shard: dict[str, Any], arm: str, regime: str, horizon: int) -> dict[str, Any]:
    if regime == "exact":
        return shard["closure"][arm]["horizons"][str(horizon)]
    return shard["sparse_family"]["closure"]["horizons"][str(horizon)]


def closure_decision(
    shards: list[dict[str, Any]],
    card: dict[str, Any],
    *,
    regime: str,
) -> dict[str, Any]:
    gate = card["closure_gates"]
    eligible = [item for item in shards if regime == "exact" or item["family"]["sparse"]["eligible"]]
    metrics = {
        "activity_k_leakage_rms": float(gate["maximum_activity_K_leakage"]),
        "activity_kminusI_leakage_rms": float(gate["maximum_activity_KminusI_leakage"]),
    }
    cells, differences, specificity = {}, {}, {}
    for horizon in card["roster"]["horizons"]:
        for metric, absolute_limit in metrics.items():
            true = [float(closure_cell(item, "sparse", regime, horizon)["true"][metric]) for item in eligible]
            null = [float(closure_cell(item, "sparse", regime, horizon)["null_median"][metric]) for item in eligible]
            ratios = [left / right for left, right in zip(true, null)]
            name = f"{metric}_H{horizon}"
            differences[name] = [right - left for left, right in zip(true, null)]
            cells[name] = {
                "eligible_seed_count": len(true),
                "true_median": median(true),
                "null_median": median(null),
                "ratio_median": median(ratios),
                "seed_wins": int(sum(left < right for left, right in zip(true, null))),
                "absolute_pass": median(true) <= absolute_limit,
                "null_ratio_pass": median(ratios) <= float(gate["maximum_observed_over_null_ratio"]),
                "wins_pass": sum(left < right for left, right in zip(true, null))
                >= int(gate["minimum_seed_wins"]),
            }
            if regime == "exact":
                dense_true = [
                    float(closure_cell(item, "dense", regime, horizon)["true"][metric])
                    for item in eligible
                ]
                dense_null = [
                    float(closure_cell(item, "dense", regime, horizon)["null_median"][metric])
                    for item in eligible
                ]
                dense_null_ratios = [left / right for left, right in zip(dense_true, dense_null)]
                ratio_ci = bootstrap_ratio_interval(
                    true, dense_true,
                    replicates=int(card["aggregation"]["bootstrap_replicates"]),
                    seed=int(card["aggregation"]["bootstrap_seed"]) + 2 * int(horizon)
                    + (0 if metric == "activity_k_leakage_rms" else 1),
                )
                specificity_gate = card["closure_specificity_gates"]
                ratio_of_means = float(np.mean(true) / np.mean(dense_true))
                wins = int(sum(left < right for left, right in zip(true, dense_true)))
                specificity[name] = {
                    "sparse_true_median": median(true),
                    "dense_true_median": median(dense_true),
                    "dense_null_median": median(dense_null),
                    "dense_observed_over_null_median": median(dense_null_ratios),
                    "sparse_over_dense_ratio_of_seed_means": ratio_of_means,
                    "sparse_seed_wins": wins,
                    "paired_ratio_bootstrap": ratio_ci,
                    "passed": ratio_of_means <= float(
                        specificity_gate["maximum_sparse_over_dense_ratio_of_seed_means"]
                    ) and wins >= int(specificity_gate["minimum_seed_wins"])
                    and ratio_ci[1] < float(
                        specificity_gate["paired_ratio_bootstrap_upper_below"]
                    ),
                }
    matrix_metrics = {
        "matrix_k_leakage_fro": float(gate["maximum_matrix_K_leakage"]),
        "matrix_kminusI_leakage_fro": float(gate["maximum_matrix_KminusI_leakage"]),
    }
    for metric, absolute_limit in matrix_metrics.items():
        true = [
            float((item["closure"]["sparse"] if regime == "exact"
                   else item["sparse_family"]["closure"])["matrix_true"][metric])
            for item in eligible
        ]
        null = [
            float((item["closure"]["sparse"] if regime == "exact"
                   else item["sparse_family"]["closure"])["matrix_null_median"][metric])
            for item in eligible
        ]
        ratios = [left / right for left, right in zip(true, null)]
        name = metric
        differences[name] = [right - left for left, right in zip(true, null)]
        cells[name] = {
            "eligible_seed_count": len(true),
            "true_median": median(true),
            "null_median": median(null),
            "ratio_median": median(ratios),
            "seed_wins": int(sum(left < right for left, right in zip(true, null))),
            "absolute_pass": median(true) <= absolute_limit,
            "null_ratio_pass": median(ratios) <= float(gate["maximum_observed_over_null_ratio"]),
            "wins_pass": sum(left < right for left, right in zip(true, null))
            >= int(gate["minimum_seed_wins"]),
        }
        if regime == "exact":
            dense_true = [float(item["closure"]["dense"]["matrix_true"][metric]) for item in eligible]
            dense_null = [
                float(item["closure"]["dense"]["matrix_null_median"][metric])
                for item in eligible
            ]
            ratio_ci = bootstrap_ratio_interval(
                true, dense_true,
                replicates=int(card["aggregation"]["bootstrap_replicates"]),
                seed=int(card["aggregation"]["bootstrap_seed"]) + 2600
                + (0 if metric == "matrix_k_leakage_fro" else 1),
            )
            ratio_of_means = float(np.mean(true) / np.mean(dense_true))
            wins = int(sum(left < right for left, right in zip(true, dense_true)))
            specificity_gate = card["closure_specificity_gates"]
            specificity[name] = {
                "sparse_true_median": median(true),
                "dense_true_median": median(dense_true),
                "dense_null_median": median(dense_null),
                "dense_observed_over_null_median": median([
                    left / right for left, right in zip(dense_true, dense_null)
                ]),
                "sparse_over_dense_ratio_of_seed_means": ratio_of_means,
                "sparse_seed_wins": wins,
                "paired_ratio_bootstrap": ratio_ci,
                "passed": ratio_of_means <= float(
                    specificity_gate["maximum_sparse_over_dense_ratio_of_seed_means"]
                ) and wins >= int(specificity_gate["minimum_seed_wins"])
                and ratio_ci[1] < float(specificity_gate["paired_ratio_bootstrap_upper_below"]),
            }
    adjusted = exact_max_t_adjusted_p(differences) if differences and eligible else {}
    for name, value in adjusted.items():
        cells[name]["max_t_adjusted_p"] = value
        cells[name]["max_t_pass"] = value <= float(gate["maximum_max_t_adjusted_p"])
    future = [
        float(closure_cell(item, "sparse", regime, 200)["true"]["encoded_future_outside_rms"])
        for item in eligible
    ]
    global_identity = [
        float(closure_cell(item, "sparse", regime, 200)["true"]["global_k_over_identity_residual"])
        for item in eligible
    ]
    minimum_seed_count = (
        len(eligible) >= int(card["family_gates"]["minimum_qualifying_seeds"])
        if regime == "family"
        else len(eligible) == len(card["roster"]["model_seeds"])
    )
    activity_names = [name for name in cells if name.startswith("activity_")]
    matrix_names = [name for name in cells if name.startswith("matrix_")]
    activity_cells_pass = all(
        cells[name]["absolute_pass"] and cells[name]["null_ratio_pass"]
        and cells[name]["wins_pass"]
        and cells[name].get("max_t_pass", False)
        for name in activity_names
    )
    matrix_cells_pass = all(
        cells[name]["absolute_pass"] and cells[name]["null_ratio_pass"]
        and cells[name]["wins_pass"]
        and cells[name].get("max_t_pass", False)
        for name in matrix_names
    )
    future_pass = median(future) <= float(gate["maximum_encoded_future_outside_ratio"])
    identity_pass = median(global_identity) <= float(gate["maximum_global_K_over_identity_residual"])
    activity_specificity = all(
        specificity[name]["passed"] for name in specificity if name.startswith("activity_")
    )
    matrix_specificity = all(
        specificity[name]["passed"] for name in specificity if name.startswith("matrix_")
    )
    activity_passed = minimum_seed_count and activity_cells_pass and future_pass and identity_pass \
        and (regime != "exact" or activity_specificity)
    matrix_passed = minimum_seed_count and matrix_cells_pass \
        and (regime != "exact" or matrix_specificity)
    checks = {
        "minimum_seed_count": minimum_seed_count,
        "activity_cells": activity_cells_pass,
        "matrix_cells": matrix_cells_pass,
        "encoded_future_outside": future_pass,
        "global_over_identity": identity_pass,
    }
    if regime == "exact":
        checks["dense_activity_specificity"] = activity_specificity
        checks["dense_matrix_specificity"] = matrix_specificity
    return {
        "passed": activity_passed and matrix_passed,
        "activity_weighted_passed": activity_passed,
        "operator_subspace_passed": matrix_passed,
        "checks": checks,
        "cells": cells,
        "encoded_future_outside_median": median(future),
        "global_over_identity_median": median(global_identity),
        "dense_matched_cardinality_specificity": specificity,
    }


def forecast_decision(shards: list[dict[str, Any]], card: dict[str, Any]) -> dict[str, Any]:
    gate = card["forecast_gates"]
    reps, seed = int(card["aggregation"]["bootstrap_replicates"]), int(card["aggregation"]["bootstrap_seed"])
    cells, projected_cells = {}, {}
    for horizon in card["roster"]["horizons"]:
        key = str(horizon)
        sparse_mean = [float(item["forecast"]["sparse"]["ratios"][key]["mean_restricted_over_mask_once"]) for item in shards]
        sparse_terminal = [float(item["forecast"]["sparse"]["ratios"][key]["terminal_restricted_over_mask_once"]) for item in shards]
        dense_mean = [float(item["forecast"]["dense"]["ratios"][key]["mean_restricted_over_mask_once"]) for item in shards]
        factor_ratios = [left / right for left, right in zip(sparse_mean, dense_mean)]
        factor_ratio_of_means = float(np.mean(sparse_mean) / np.mean(dense_mean))
        ratio_ci = bootstrap_ratio_interval(sparse_mean, dense_mean, replicates=reps, seed=seed + horizon)
        cells[key] = {
            "sparse_mean_rho_median": median(sparse_mean),
            "sparse_terminal_rho_median": median(sparse_terminal),
            "sparse_over_dense_restriction_factor_median_guard": median(factor_ratios),
            "sparse_over_dense_restriction_factor_ratio_of_seed_means": factor_ratio_of_means,
            "sparse_restriction_factor_seed_wins": int(sum(left < right for left, right in zip(sparse_mean, dense_mean))),
            "sparse_over_dense_restriction_factor_ratio_bootstrap": ratio_ci,
            "retention_pass": median(sparse_mean)
            <= float(gate["maximum_sparse_mean_restricted_over_mask_once"])
            and median(sparse_terminal)
            <= float(gate["maximum_sparse_terminal_restricted_over_mask_once"]),
            "specificity_pass": median(factor_ratios)
            <= float(gate["maximum_sparse_over_dense_restriction_factor_median_guard"])
            and factor_ratio_of_means <= float(
                gate["maximum_sparse_over_dense_restriction_factor_ratio_of_seed_means"]
            )
            and sum(left < right for left, right in zip(sparse_mean, dense_mean))
            >= int(gate["minimum_seed_wins"])
            and ratio_ci[1] < float(gate["paired_ratio_bootstrap_upper_below"]),
        }
        sparse_projected = [
            float(item["forecast"]["sparse"][key]["restricted"]["field_mse"])
            for item in shards
        ]
        dense_full = [
            float(item["forecast"]["dense"][key]["full"]["field_mse"])
            for item in shards
        ]
        ratio_interval = bootstrap_ratio_interval(
            sparse_projected, dense_full, replicates=reps, seed=seed + 1000 + horizon
        )
        reduction = 1.0 - float(np.mean(sparse_projected) / np.mean(dense_full))
        reduction_interval = [1.0 - ratio_interval[1], 1.0 - ratio_interval[0]]
        wins = int(sum(left < right for left, right in zip(sparse_projected, dense_full)))
        projected_cells[key] = {
            "ratio_of_seed_means_reduction": reduction,
            "seed_wins": wins,
            "bootstrap_interval": reduction_interval,
            "direction_and_uncertainty_pass": wins >= int(gate["minimum_seed_wins"])
            and reduction_interval[0]
            > float(gate["strong_sparse_projected_vs_dense_full_lower_bound_above"]),
            "h200_effect_pass": horizon != 200 or reduction
            >= float(gate["strong_minimum_H200_sparse_projected_vs_dense_full_reduction"]),
        }
    return {
        "passed": all(cell["retention_pass"] and cell["specificity_pass"] for cell in cells.values()),
        "projected_vs_dense_full_passed": all(
            cell["direction_and_uncertainty_pass"] and cell["h200_effect_pass"]
            for cell in projected_cells.values()
        ),
        "cells": cells,
        "projected_vs_dense_full": projected_cells,
    }


def select_decision(
    *,
    validity: bool,
    exact_closure: dict[str, Any],
    forecast: dict[str, Any],
    family: dict[str, Any],
) -> str:
    if not validity:
        return "invalid"
    if exact_closure["passed"] and forecast["passed"] and family["family_passed"]:
        if not family["signature_differentiation_passed"]:
            return "multiple_closed_support_charts"
        if not family["routing_specificity_passed"]:
            return "distinct_signatures_without_routing_specificity"
        if not forecast["projected_vs_dense_full_passed"]:
            return "distinct_signatures_without_dense_forecast_win"
        return "strong_routed_low_leakage_charts_with_distinct_signatures"
    if exact_closure["passed"] and forecast["passed"]:
        return "state_specific_support_utility"
    if exact_closure["passed"]:
        return "latent_closure_only"
    if exact_closure.get("activity_weighted_passed", False):
        return "activity_weighted_closure_only"
    return "failed"


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    card, card_hash = load_card(args.card)
    if card_hash != args.expected_card_sha256:
        raise RuntimeError("Card differs from launcher root of trust")
    current_source_hash = verify_source_manifest(SOURCE_MANIFEST)
    if current_source_hash != args.expected_source_manifest_sha256:
        raise RuntimeError("Source manifest differs from launcher root of trust")
    _, current_profile_hash = load_profile_decision(
        args.profile_decision, args.batch_size, card=card, card_hash=card_hash,
        source_manifest_hash=current_source_hash,
    )
    if current_profile_hash != args.expected_profile_decision_sha256:
        raise RuntimeError("Profile decision differs from launcher root of trust")
    canary_path = args.input_root / "canary" / "validation.json"
    validate_release_receipt(
        canary_path, card_hash=card_hash, source_hash=current_source_hash,
        profile_hash=current_profile_hash,
    )
    shards = [json.loads(path.read_text()) for path in sorted((args.input_root / "shards").glob("seed_*.json"))]
    expected_seeds = [int(value) for value in card["roster"]["model_seeds"]]
    telemetry = {
        seed: json.loads((args.telemetry_dir / f"seed_{seed}.json").read_text())
        for seed in expected_seeds
    }
    expected_roster = checkpoint_roster(card)
    source_paths = {item.get("source_manifest_path") for item in shards}
    profile_paths = {item.get("profile_decision_path") for item in shards}
    source_current = len(source_paths) == 1 and None not in source_paths and sha256_path(
        Path(next(iter(source_paths)))
    ) == next(iter({item["source_manifest_sha256"] for item in shards}), "")
    profile_current = len(profile_paths) == 1 and None not in profile_paths and sha256_path(
        Path(next(iter(profile_paths)))
    ) == next(iter({item["profile_decision_sha256"] for item in shards}), "")
    expected_keys = ["fields", "split_indices"]
    provenance_valid = bool(
        len(shards) == len(expected_seeds)
        and sorted(int(item["seed"]) for item in shards) == expected_seeds
        and all(item["status"] == "completed" and item["card_sha256"] == card_hash for item in shards)
        and len({item["source_manifest_sha256"] for item in shards}) == 1
        and len({item["profile_decision_sha256"] for item in shards}) == 1
        and all(item["card_sha256"] == args.expected_card_sha256
                and item["source_manifest_sha256"] == args.expected_source_manifest_sha256
                and item["profile_decision_sha256"] == args.expected_profile_decision_sha256
                for item in shards)
        and all(all(value is True for value in telemetry_receipt_checks(
            telemetry[seed], args.telemetry_dir, card_hash=card_hash,
            source_hash=current_source_hash, seed=seed,
            slurm_job_id=str(next(item for item in shards
                                  if int(item["seed"]) == seed)["slurm_job_id"]),
            evaluator_scope=next(item for item in shards
                                 if int(item["seed"]) == seed)["gpu_telemetry_scope"],
        ).values()) for seed in expected_seeds)
        and source_current and profile_current
        and all(item["information_firewall"]["requested_dataset_keys"] == expected_keys
                and item["information_firewall"]["requested_dataset_key_name_firewall_passed"]
                and item["information_firewall"]["future_encoding_began_only_after_support_lock"]
                and not item["information_firewall"]["future_states_used_for_routing"]
                and not item["information_firewall"]["periodic_reencoding"] for item in shards)
        and all(item["mask_summary"]["paired_cardinality_exact"] for item in shards)
        and all(item["validity_audits"]["score_trajectory_count"]
                == int(card["validity_gates"]["score_trajectory_count"])
                and item["validity_audits"]["architecture_and_treatment_audit_passed"]
                and item["validity_audits"]["row_operator_orientation_audit_passed"]
                and item["validity_audits"]["full_K_ordinary_forecast_reproduction"]["passed"]
                for item in shards)
        and all(
            item["provenance"][arm]["checkpoint_path"]
            == str(expected_roster[(arm, int(item["seed"]))].path)
            and item["provenance"][arm]["checkpoint_sha256"]
            == expected_roster[(arm, int(item["seed"]))].sha256
            for item in shards for arm in ("sparse", "dense")
        )
        and all(finite_tree(item) for item in shards)
        and all(finite_tree(item) for item in telemetry.values())
    )
    exact_capture = [float(item["initial_projection"]["sparse"]["source_capture_rms"]) for item in shards]
    source_capture_valid = bool(exact_capture) and min(exact_capture) >= float(
        card["validity_gates"]["minimum_sparse_exact_initial_source_capture_rms"]
    )
    exact_closure = closure_decision(shards, card, regime="exact") if provenance_valid else {"passed": False}
    forecast = forecast_decision(shards, card) if provenance_valid else {"passed": False}
    family = family_decision(shards, card, closure_reducer=closure_decision) if provenance_valid else {
        "family_passed": False, "signature_differentiation_passed": False,
        "routing_specificity_passed": False,
    }
    validity = provenance_valid and source_capture_valid
    decision = select_decision(
        validity=validity, exact_closure=exact_closure, forecast=forecast, family=family
    )
    result = {
        "schema_version": 1,
        "decision": decision,
        "interpretation": card["decision_branches"][decision],
        "validity": {
            "passed": validity,
            "provenance_and_firewall": provenance_valid,
            "outcome_blind_canary_release": True,
            "exact_initial_source_capture": source_capture_valid,
            "minimum_exact_initial_source_capture": min(exact_capture) if exact_capture else None,
        },
        "exact_fixed_P0_closure": exact_closure,
        "decoded_forecast": forecast,
        "family": family,
        "card_sha256": card_hash,
        "source_manifest_sha256": shards[0]["source_manifest_sha256"] if shards else None,
        "profile_decision_sha256": shards[0]["profile_decision_sha256"] if shards else None,
        "inference_scope": card["aggregation"]["scope"],
        "gpu_utilization": {
            "minimum_seed_mean_all_percent": min(
                float(telemetry[seed]["telemetry"]["mean_all_gpu_utilization_percent"])
                for seed in expected_seeds
            ),
            "mean_seed_mean_all_percent": float(np.mean([
                telemetry[seed]["telemetry"]["mean_all_gpu_utilization_percent"]
                for seed in expected_seeds
            ])),
        },
    }
    args.output_dir.mkdir(parents=True)
    write_rows(args.output_dir / "seed_rows.csv", shards)
    (args.output_dir / "decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "card_sha256": card_hash,
        "source_manifest_sha256": result["source_manifest_sha256"],
        "profile_decision_sha256": result["profile_decision_sha256"],
        "canary_release_sha256": sha256_path(canary_path),
        "shards": {
            f"seed_{item['seed']}.json": sha256_path(args.input_root / "shards" / f"seed_{item['seed']}.json")
            for item in shards
        },
        "telemetry": {
            f"seed_{seed}.json": sha256_path(args.telemetry_dir / f"seed_{seed}.json")
            for seed in expected_seeds
        },
        "raw_telemetry": {
            telemetry[seed]["raw_telemetry_filename"]: telemetry[seed]["raw_telemetry_sha256"]
            for seed in expected_seeds
        },
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "output": str(args.output_dir)}), flush=True)


if __name__ == "__main__":
    main()
