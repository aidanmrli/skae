"""Compute outcomes only after the direct packet passes authentication."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from experiments.neurips_2026.allen_cahn_direct_baseline.core import write_json_once
from experiments.neurips_2026.allen_cahn_direct_baseline.summarize import (
    NEW_IC_IDS,
    authenticate_seed,
    descriptive_effect,
    holm,
    paired_effect,
    parse_args,
    seed_directories,
)
from experiments.neurips_2026.allen_cahn_direct_baseline.train import verify_lock


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    lock = verify_lock(args.task_lock, args.expected_task_lock_sha256)
    seeds = [int(value) for value in lock["scientific_protocol"]["model_seeds"]]
    if seed_directories(args.training_root) != set(seeds):
        raise RuntimeError("Training seed-directory roster is incomplete or has extras")
    if seed_directories(args.evaluation_root) != set(seeds):
        raise RuntimeError("Evaluation seed-directory roster is incomplete or has extras")

    evaluations: dict[int, dict[str, Any]] = {}
    authentication: dict[str, Any] = {}
    for seed in seeds:
        evaluation, seed_auth = authenticate_seed(
            seed=seed,
            training_root=args.training_root,
            evaluation_root=args.evaluation_root,
            lock=lock,
            task_lock_sha256=args.expected_task_lock_sha256,
        )
        evaluations[seed] = evaluation
        authentication[str(seed)] = seed_auth

    comparator_by_seed = {
        int(row["model_seed"]): row for row in lock["frozen_comparator_seed_rows"]
    }
    if set(comparator_by_seed) != set(seeds):
        raise RuntimeError("Comparator seed roster mismatch")

    direct_primary = []
    direct_terminal = []
    seed_rows = []
    for seed in seeds:
        evaluation = evaluations[seed]
        cumulative = [
            float(
                evaluation["datasets"][dataset_id]["metrics"]["endpoints"]["200"]
                ["through_horizon_field_mse"]
            )
            for dataset_id in NEW_IC_IDS
        ]
        terminal = [
            float(
                evaluation["datasets"][dataset_id]["metrics"]["endpoints"]["200"]
                ["terminal_field_mse"]
            )
            for dataset_id in NEW_IC_IDS
        ]
        direct_primary.append(float(np.mean(cumulative)))
        direct_terminal.append(float(np.mean(terminal)))
        seed_rows.append(
            {
                "model_seed": seed,
                "direct_new_ic_h200_through_horizon_field_mse": direct_primary[-1],
                "direct_new_ic_h200_terminal_field_mse": direct_terminal[-1],
                "new_ic_h200_through_horizon_by_dataset": dict(
                    zip(NEW_IC_IDS, cumulative)
                ),
                "new_ic_h200_terminal_by_dataset": dict(zip(NEW_IC_IDS, terminal)),
                "development_endpoints": evaluation["datasets"][
                    "development_20260724"
                ]["metrics"]["endpoints"],
            }
        )

    direct = np.asarray(direct_primary)
    sparse = np.asarray(
        [
            comparator_by_seed[seed]["sparse_h200_through_horizon_field_mse"]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    dense = np.asarray(
        [
            comparator_by_seed[seed]["dense_h200_through_horizon_field_mse"]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    stats = lock["statistics"]
    effects = {
        "sparse": paired_effect(
            direct,
            sparse,
            bootstrap_seed=stats["bootstrap_seed"],
            bootstrap_replicates=stats["bootstrap_replicates"],
        ),
        "dense": paired_effect(
            direct,
            dense,
            bootstrap_seed=stats["bootstrap_seed"],
            bootstrap_replicates=stats["bootstrap_replicates"],
        ),
    }
    for p_key, output_key in (
        ("direct_one_sided_exact_sign_flip_p", "direct_one_sided_holm_p"),
        ("comparator_one_sided_exact_sign_flip_p", "comparator_one_sided_holm_p"),
    ):
        adjusted = holm({name: effect[p_key] for name, effect in effects.items()})
        for name in effects:
            effects[name][output_key] = adjusted[name]

    direct_terminal_array = np.asarray(direct_terminal)
    terminal_context = {
        name: descriptive_effect(
            direct_terminal_array,
            np.asarray(
                [
                    comparator_by_seed[seed][f"{name}_h200_terminal_field_mse"]
                    for seed in seeds
                ]
            ),
        )
        for name in ("sparse", "dense")
    }
    development_context: dict[str, Any] = {}
    for metric in ("through_horizon_field_mse", "terminal_field_mse"):
        direct_development = np.asarray(
            [
                evaluations[seed]["datasets"]["development_20260724"]["metrics"]
                ["endpoints"]["200"][metric]
                for seed in seeds
            ]
        )
        development_context[metric] = {
            name: descriptive_effect(
                direct_development,
                np.asarray(
                    [
                        comparator_by_seed[seed][f"development_{name}_h200_{metric}"]
                        for seed in seeds
                    ]
                ),
            )
            for name in ("sparse", "dense")
        }
    direct_development_horizons = {
        str(horizon): {
            metric: float(
                np.mean(
                    [
                        evaluations[seed]["datasets"]["development_20260724"]
                        ["metrics"]["endpoints"][str(horizon)][metric]
                        for seed in seeds
                    ]
                )
            )
            for metric in ("through_horizon_field_mse", "terminal_field_mse")
        }
        for horizon in (80, 120, 160, 200)
    }

    sparse_effect = effects["sparse"]
    margin = float(stats["meaningful_margin"])
    minimum_wins = int(stats["minimum_seed_wins"])
    gate_checks = {
        "direct_meaningfully_better_than_sparse": {
            "relative_reduction_at_least_5_percent": sparse_effect[
                "direct_advantage"
            ]
            >= margin,
            "paired_bootstrap_lower_above_zero": sparse_effect[
                "direct_advantage_ci95"
            ][0]
            > 0.0,
            "at_least_8_of_10_seed_wins": sparse_effect["direct_seed_wins"]
            >= minimum_wins,
        },
        "sparse_meaningfully_better_than_direct": {
            "relative_reduction_at_least_5_percent": sparse_effect[
                "comparator_advantage"
            ]
            >= margin,
            "paired_bootstrap_lower_above_zero": sparse_effect[
                "comparator_advantage_ci95"
            ][0]
            > 0.0,
            "at_least_8_of_10_seed_wins": sparse_effect[
                "comparator_seed_wins"
            ]
            >= minimum_wins,
        },
    }
    direct_strong = all(gate_checks["direct_meaningfully_better_than_sparse"].values())
    sparse_strong = all(gate_checks["sparse_meaningfully_better_than_direct"].values())
    if direct_strong and sparse_strong:
        raise RuntimeError("Mutually exclusive branches both passed")
    branch = (
        "direct_meaningfully_better_than_sparse"
        if direct_strong
        else "sparse_meaningfully_better_than_direct"
        if sparse_strong
        else "practically_comparable_or_uncertain"
    )

    payload = {
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "task_lock_sha256": args.expected_task_lock_sha256,
        "status": "valid_complete_report_always_packet",
        "decision_branch": branch,
        "primary_endpoint": (
            "three-new-IC-dataset average H200 through-horizon field MSE "
            "within each of ten model seeds"
        ),
        "capacity_and_budget": lock["model_and_compute_match"],
        "gate_checks": gate_checks,
        "failed_gate_checks": {
            name: [key for key, value in checks.items() if not value]
            for name, checks in gate_checks.items()
        },
        "primary_effects": effects,
        "h200_terminal_context": terminal_context,
        "development_h200_context": development_context,
        "direct_development_horizon_means": direct_development_horizons,
        "direct_seed_rows": seed_rows,
        "authentication_by_seed": authentication,
        "mandatory_interpretation": lock["mandatory_interpretation"],
        "claim_boundary": lock["claim_boundary"],
    }
    write_json_once(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
