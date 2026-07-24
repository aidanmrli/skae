#!/usr/bin/env python3
"""Summarize the preregistered dense top-k specificity control."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.neurips_2026.global_k_dense_zero_wd_tasks import load_card, sha256_path


SUMMARIZER_PATH = Path(__file__)
TASK_MODULE_PATH = Path(__file__).with_name("global_k_dense_zero_wd_tasks.py")


METRICS = {
    "activity_leakage": "activity_k_leakage_rms",
    "restricted_residual": "posthoc_pkp_inside_residual_rms",
}


def _median(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    return None if numerator is None or denominator is None or denominator <= 0 else numerator / denominator


def _run_row(payload: dict[str, Any]) -> dict[str, Any]:
    true, null = payload["regime"]["true"], payload["regime"]["null_summary"]
    row = {
        **{key: payload["provenance"][key] for key in ("system_key", "system_name", "seed", "dense_run_dir")},
        "status": payload["status"],
        **payload["routing"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    for display, metric in METRICS.items():
        true_value = true["aggregate"].get(metric)
        null_value = null["aggregate"].get(metric, {}).get("median")
        row[f"{display}_true"] = true_value
        row[f"{display}_null"] = null_value
        row[f"{display}_true_over_null"] = _ratio(true_value, null_value)
    return row


def _system_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[row["system_key"]].append(row)
    result = []
    for system, rows in sorted(groups.items()):
        eligible = [row for row in rows if row["status"] == "eligible"]
        summary: dict[str, Any] = {
            "system_key": system,
            "system_name": rows[0]["system_name"],
            "run_count": len(rows),
            "eligible_seed_count": len(eligible),
            "system_eligible": len(eligible) >= 2,
        }
        for display in METRICS:
            for suffix in ("true", "null", "true_over_null"):
                summary[f"{display}_{suffix}"] = _median(
                    row[f"{display}_{suffix}"] for row in eligible
                )
        result.append(summary)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decide(
    run_rows: list[dict[str, Any]], system_rows: list[dict[str, Any]], card: dict[str, Any],
) -> dict[str, Any]:
    validity = card["matched_evaluation"]["validity"]
    eligible_runs = sum(row["status"] == "eligible" for row in run_rows)
    eligible_systems = [row for row in system_rows if row["system_eligible"]]
    dense = {
        f"{display}_{suffix}": _median(row[f"{display}_{suffix}"] for row in eligible_systems)
        for display in METRICS for suffix in ("true", "null", "true_over_null")
    }
    sparse_decision_path = Path(card["frozen_sparse_reference"]["decision_json"])
    if sha256_path(sparse_decision_path) != card["frozen_sparse_reference"]["decision_sha256"]:
        raise RuntimeError("Frozen sparse decision hash mismatch")
    sparse = json.loads(sparse_decision_path.read_text())["system_medians"]
    sparse_activity = sparse["activity_leakage_true_over_null"]
    sparse_residual = sparse["restricted_inside_residual_true_over_null"]
    activity_specificity = _ratio(sparse_activity, dense["activity_leakage_true_over_null"])
    residual_specificity = _ratio(sparse_residual, dense["restricted_residual_true_over_null"])
    gate = card["matched_evaluation"]["specificity_gate"]
    checks = {
        "roster_complete": len(run_rows) == int(validity["required_run_count"]),
        "eligible_runs": eligible_runs >= int(validity["min_eligible_runs"]),
        "eligible_systems": len(eligible_systems) >= int(validity["min_eligible_systems"]),
        "activity_specificity": activity_specificity is not None
        and activity_specificity <= float(gate["max_sparse_over_dense_activity_leakage_null_ratio"]),
        "restricted_residual_specificity": residual_specificity is not None
        and residual_specificity <= float(gate["max_sparse_over_dense_restricted_residual_null_ratio"]),
    }
    valid = all(checks[key] for key in ("roster_complete", "eligible_runs", "eligible_systems"))
    if not valid:
        decision = "invalid_dense_control"
    elif checks["activity_specificity"] and checks["restricted_residual_specificity"]:
        decision = "sparse_support_specific"
    else:
        decision = "not_sparse_specific"
    return {
        "schema_version": 1,
        "decision": decision,
        "eligible_run_count": eligible_runs,
        "eligible_system_count": len(eligible_systems),
        "checks": checks,
        "dense_system_medians": dense,
        "sparse_system_medians": {
            "activity_leakage_true_over_null": sparse_activity,
            "restricted_residual_true_over_null": sparse_residual,
        },
        "sparse_over_dense": {
            "activity_leakage_null_ratio": activity_specificity,
            "restricted_residual_null_ratio": residual_specificity,
        },
        "interpretation": {
            "sparse_support_specific": "Both frozen 0.90 ratios pass: the sparse recipe's support closure is specific relative to sparse-cardinality-matched dense tanh coordinate masks.",
            "not_sparse_specific": "At least one frozen 0.90 ratio fails; do not claim that support closure is sparse-specific.",
            "invalid_dense_control": card["matched_evaluation"]["invalid_branch"],
        }[decision],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--source_lock", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    card, card_hash = load_card(args.card)
    payloads = [
        json.loads(path.read_text())
        for path in sorted((args.input_dir / "shards").glob("task_*.json"))
    ]
    if any(payload["card_sha256"] != card_hash for payload in payloads):
        raise RuntimeError("At least one dense shard used a different card")
    unique = {(item["provenance"]["system_key"], item["provenance"]["seed"]) for item in payloads}
    expected = {
        (system, int(seed))
        for system in card["training"]["systems"]
        for seed in card["training"]["seeds"]
    }
    if unique != expected or len(payloads) != len(expected):
        raise RuntimeError(
            "Dense shard roster mismatch; "
            f"missing={sorted(expected - unique)}, extra={sorted(unique - expected)}, "
            f"payload_count={len(payloads)}"
        )
    task_tsv_hashes = {payload.get("task_tsv_sha256") for payload in payloads}
    if len(task_tsv_hashes) != 1 or None in task_tsv_hashes:
        raise RuntimeError("Dense shards do not share one authenticated task table")
    source_records = {
        json.dumps(payload.get("authenticated_sources"), sort_keys=True)
        for payload in payloads
    }
    if len(source_records) != 1 or "null" in source_records:
        raise RuntimeError("Dense shards do not share one authenticated evaluator source set")
    evaluator_sources = json.loads(next(iter(source_records)))
    source_lock_hashes = {
        payload.get("source_lock", {}).get("sha256") for payload in payloads
    }
    actual_source_lock_hash = sha256_path(args.source_lock)
    if source_lock_hashes != {actual_source_lock_hash}:
        raise RuntimeError("Dense shards do not share the reducer's source lock")
    source_lock = json.loads(args.source_lock.read_text())
    if source_lock.get("protocol_id") != card.get("protocol_id"):
        raise RuntimeError("Source lock/card protocol mismatch")
    task_module_hash = sha256_path(TASK_MODULE_PATH)
    if evaluator_sources["dense_task_module"]["sha256"] != task_module_hash:
        raise RuntimeError("Dense task module drifted between evaluation and reduction")
    run_rows = [_run_row(payload) for payload in payloads]
    system_rows = _system_rows(run_rows)
    decision = decide(run_rows, system_rows, card)
    decision["authenticated_sources"] = {
        **evaluator_sources,
        "dense_specificity_summarizer": {
            "path": "experiments/neurips_2026/summarize_global_k_dense_specificity.py",
            "sha256": sha256_path(SUMMARIZER_PATH),
        },
        "frozen_card": {"path": str(args.card), "sha256": card_hash},
        "task_tsv": {"sha256": next(iter(task_tsv_hashes))},
        "source_lock": {"path": str(args.source_lock), "sha256": actual_source_lock_hash},
    }
    decision["aggregation_contract"] = (
        "run metric divided by its within-run 16-permutation null median; "
        "median across eligible seeds within system; median across eligible systems"
    )
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "run_rows.csv", run_rows)
    _write_csv(args.output_dir / "system_rows.csv", system_rows)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
