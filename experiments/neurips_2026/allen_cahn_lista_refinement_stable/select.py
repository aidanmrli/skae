"""Select refinement depth using direct rollout and fail-closed gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


HISTORICAL_SPARSE_DIRECT_RISK = 0.04743836535156204


def _forecast_checkpoint_improved(model_dir: Path) -> tuple[bool, float, float]:
    rows = [
        json.loads(line)
        for line in (model_dir / "metrics_history.jsonl").read_text().splitlines()
        if line.strip()
    ]
    pretrain = [row for row in rows if int(row.get("step", -2)) == -1]
    forecast = [row for row in rows if int(row.get("step", -1)) >= 0]
    if len(pretrain) != 1 or not forecast:
        return False, math.inf, math.inf
    pretrain_score = float(pretrain[0]["checkpoint_score"])
    best_forecast = min(float(row["checkpoint_score"]) for row in forecast)
    return best_forecast < pretrain_score, pretrain_score, best_forecast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    depth_policy_values: dict[int, dict[str, list[float]]] = {
        2: defaultdict(list),
        3: defaultdict(list),
    }
    seed_rows = []
    checkpoint_gates: dict[str, dict[str, object]] = {}
    finite = True
    for refinements in (2, 3):
        for seed in (64, 65):
            run_root = args.root / f"refinements_{refinements}" / f"seed_{seed}"
            payload = json.loads((run_root / "validation.json").read_text())
            per_seed: dict[str, list[float]] = defaultdict(list)
            for dataset in payload["datasets"]:
                for policy, metrics in dataset["policies"].items():
                    for key in ("cumulative_field_mse", "terminal_field_mse"):
                        finite &= math.isfinite(float(metrics[key]))
                    value = float(metrics["cumulative_field_mse"])
                    per_seed[policy].append(value)
                    depth_policy_values[refinements][policy].append(value)
            improved, pretrain_score, forecast_score = _forecast_checkpoint_improved(
                run_root / "model"
            )
            checkpoint_gates[f"depth_{refinements}_seed_{seed}"] = {
                "forecast_checkpoint_improved": improved,
                "pretrain_score": pretrain_score,
                "best_forecast_score": forecast_score,
            }
            seed_rows.append({
                "refinements": refinements,
                "seed": seed,
                "policy_means": {
                    key: sum(values) / len(values)
                    for key, values in sorted(per_seed.items())
                },
            })

    depth_summaries = {}
    for refinements in (2, 3):
        means = {
            policy: sum(values) / len(values)
            for policy, values in sorted(depth_policy_values[refinements].items())
        }
        periodic = {key: value for key, value in means.items() if key != "direct"}
        best_periodic = min(periodic, key=periodic.get)
        depth_summaries[str(refinements)] = {
            "direct_risk": means["direct"],
            "periodic_policy_means_secondary": periodic,
            "best_periodic_policy_secondary": best_periodic,
            "best_periodic_risk_secondary": periodic[best_periodic],
        }

    all_forecast_checkpoints = all(
        bool(row["forecast_checkpoint_improved"])
        for row in checkpoint_gates.values()
    )
    competitive = min(
        depth_summaries["2"]["direct_risk"],
        depth_summaries["3"]["direct_risk"],
    ) < HISTORICAL_SPARSE_DIRECT_RISK
    eligible = finite and all_forecast_checkpoints and competitive
    risk2 = float(depth_summaries["2"]["direct_risk"])
    risk3 = float(depth_summaries["3"]["direct_risk"])
    relative_difference = abs(risk2 - risk3) / min(risk2, risk3)
    selected = None
    if eligible:
        selected = 2 if relative_difference <= 0.01 or risk2 < risk3 else 3

    result = {
        "schema_version": 1,
        "status": "selected_on_open_validation_only" if eligible else "rejected_both_depths",
        "selected_refinements": selected,
        "selector_metric": "direct H200 cumulative field MSE",
        "tie_rule_applied": eligible and relative_difference <= 0.01,
        "relative_direct_risk_difference": relative_difference,
        "gates": {
            "all_external_metrics_finite": finite,
            "all_four_forecast_checkpoints_improved_over_pretrain": all_forecast_checkpoints,
            "best_depth_beats_historical_sparse_direct_risk": competitive,
            "historical_sparse_direct_risk": HISTORICAL_SPARSE_DIRECT_RISK,
        },
        "checkpoint_gates": checkpoint_gates,
        "depth_summaries": depth_summaries,
        "seed_rows": seed_rows,
        "next_step": (
            "Promote the selected depth to new outcome-quarantined panels and more paired seeds."
            if eligible
            else "Do not promote; revise optimization or architecture using development data only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
