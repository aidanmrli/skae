#!/usr/bin/env python3
"""Adjudicate the frozen GatedLocalLinear distinct-law shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.global_k_distinct_laws import load_protocol_card


def _load_shards(output_dir: Path, card: dict[str, Any], card_hash: str) -> dict[str, list[dict[str, Any]]]:
    by_arm: dict[str, list[dict[str, Any]]] = {"sparse": [], "dense": []}
    for arm in by_arm:
        expected = {int(seed) for seed in card["scope"][f"{arm}_seeds"]}
        observed: dict[int, dict[str, Any]] = {}
        for path in sorted((output_dir / "shards").glob(f"{arm}_seed_*.json")):
            payload = json.loads(path.read_text())
            if payload.get("protocol_id") != card["protocol_id"]:
                raise RuntimeError(f"Protocol mismatch in {path}")
            if payload.get("card_sha256") != card_hash:
                raise RuntimeError(f"Card hash mismatch in {path}")
            seed = int(payload["seed"])
            if seed in observed:
                raise RuntimeError(f"Duplicate {arm} seed {seed}")
            observed[seed] = payload
        if set(observed) != expected:
            raise RuntimeError(
                f"{arm} shard mismatch: missing={sorted(expected - set(observed))}, "
                f"extra={sorted(set(observed) - expected)}"
            )
        by_arm[arm] = [observed[seed] for seed in sorted(expected)]
    return by_arm


def _metric(payload: dict[str, Any], *keys: str) -> float:
    value: Any = payload["result"]
    for key in keys:
        value = value[key]
    return float(value)


def adjudicate(by_arm: dict[str, list[dict[str, Any]]], card: dict[str, Any]) -> dict[str, Any]:
    sparse = by_arm["sparse"]
    dense = by_arm["dense"]
    sparse_valid = all(row["result"]["status"] == "eligible" for row in sparse)
    sparse_strong = sparse_valid and all(
        bool(row["result"]["strong_distinct_law_pass"]) for row in sparse
    )
    dense_valid = all(row["result"]["status"] == "eligible" for row in dense)
    sparse_row = [
        _metric(row, "block", "law_identification", "max_own_over_nearest_wrong")
        for row in sparse
    ] if sparse_valid else []
    dense_row = [
        _metric(row, "block", "law_identification", "max_own_over_nearest_wrong")
        for row in dense
    ] if dense_valid else []
    sparse_assignment = [
        _metric(row, "block", "law_identification", "identity_over_best_nonidentity")
        for row in sparse
    ] if sparse_valid else []
    dense_assignment = [
        _metric(row, "block", "law_identification", "identity_over_best_nonidentity")
        for row in dense
    ] if dense_valid else []
    specificity = card["sparse_specificity_gate"]
    row_ratios = [
        left / max(right, 1e-12) for left, right in zip(sparse_row, dense_row)
    ]
    assignment_ratios = [
        left / max(right, 1e-12)
        for left, right in zip(sparse_assignment, dense_assignment)
    ]
    sparse_specific = bool(
        sparse_strong
        and dense_valid
        and all(
            value <= float(specificity["max_sparse_over_dense_row_identification_ratio"])
            for value in row_ratios
        )
        and all(
            value <= float(specificity["max_sparse_over_dense_assignment_ratio"])
            for value in assignment_ratios
        )
    )
    global_threshold = float(
        card["strong_distinct_law_gate"]["max_global_change_matrix_relative_error"]
    )
    global_positive = sparse_valid and all(
        _metric(row, "global", "law_identification", "max_own_relative_error")
        <= global_threshold
        for row in sparse
    )
    if not sparse_valid:
        decision = "invalid"
    elif sparse_specific:
        decision = "strong_distinct_laws_sparse_specific"
    elif sparse_strong:
        decision = "strong_distinct_laws_not_sparse_specific"
    elif global_positive:
        decision = "global_map_only"
    else:
        decision = "failed"
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "decision": decision,
        "decision_text": card["decision_branches"][decision],
        "gates": {
            "sparse_valid_3_of_3": sparse_valid,
            "sparse_strong_3_of_3": sparse_strong,
            "dense_valid_3_of_3": dense_valid,
            "global_positive_3_of_3": global_positive,
            "sparse_specific": sparse_specific,
        },
        "paired_specificity": {
            "sparse_over_dense_row_identification_ratio_by_seed": row_ratios,
            "sparse_over_dense_assignment_ratio_by_seed": assignment_ratios,
            "numeric_gate_evaluable": dense_valid,
        },
        "seed_metrics": {
            "sparse_row_identification": sparse_row,
            "dense_row_identification": dense_row,
            "sparse_assignment": sparse_assignment,
            "dense_assignment": dense_assignment,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=Path(__file__).with_name("global_k_distinct_laws_card.json"))
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    card, card_hash = load_protocol_card(args.card)
    by_arm = _load_shards(args.output_dir, card, card_hash)
    decision = adjudicate(by_arm, card)
    summary = args.output_dir / "summary" / "decision.json"
    if summary.exists():
        raise FileExistsError(f"Refusing to overwrite {summary}")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(summary), "decision": decision["decision"]}))


if __name__ == "__main__":
    main()
