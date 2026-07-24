"""Select Allen--Cahn refinement depth from complete validation outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for refinements in (2, 3):
        for seed in (64, 65):
            path = args.root / f"refinements_{refinements}" / f"seed_{seed}" / "validation.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            runs.append((refinements, seed, json.loads(path.read_text())))

    policy_values: dict[int, dict[str, list[float]]] = {
        2: defaultdict(list), 3: defaultdict(list)
    }
    seed_rows = []
    for refinements, seed, payload in runs:
        per_seed: dict[str, list[float]] = defaultdict(list)
        for dataset in payload["datasets"]:
            for policy, metrics in dataset["policies"].items():
                value = float(metrics["cumulative_field_mse"])
                policy_values[refinements][policy].append(value)
                per_seed[policy].append(value)
        seed_rows.append({
            "refinements": refinements,
            "seed": seed,
            "policy_means": {key: sum(values) / len(values) for key, values in sorted(per_seed.items())},
        })

    depth_summaries = {}
    for refinements in (2, 3):
        means = {
            policy: sum(values) / len(values)
            for policy, values in sorted(policy_values[refinements].items())
        }
        selected_policy = min(means, key=means.get)
        depth_summaries[str(refinements)] = {
            "policy_means": means,
            "selected_policy": selected_policy,
            "selected_risk": means[selected_policy],
            "direct_risk": means["direct"],
        }
    risk2 = depth_summaries["2"]["selected_risk"]
    risk3 = depth_summaries["3"]["selected_risk"]
    relative_difference = abs(risk2 - risk3) / min(risk2, risk3)
    selected = 2 if relative_difference <= 0.01 or risk2 < risk3 else 3
    result = {
        "schema_version": 1,
        "status": "selected_on_open_validation_only",
        "selected_refinements": selected,
        "tie_rule_applied": relative_difference <= 0.01,
        "relative_risk_difference": relative_difference,
        "depth_summaries": depth_summaries,
        "seed_rows": seed_rows,
        "next_step": "Promote the selected depth to additional paired seeds and newly sealed field panels before a confirmatory paper claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
