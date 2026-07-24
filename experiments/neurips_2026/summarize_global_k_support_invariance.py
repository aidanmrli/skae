#!/usr/bin/env python3
"""Aggregate the frozen global-K support-invariance run without pseudoreplication."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.neurips_2026.global_k_support_invariance import DEFAULT_CARD, load_card


METRICS = {
    "activity_leakage": ("aggregate", "activity_k_leakage_rms"),
    "matrix_leakage": ("aggregate", "matrix_k_leakage_fro_activity_weighted_mean"),
    "activity_change_leakage": ("aggregate", "activity_k_change_leakage_rms"),
    "matrix_change_leakage": ("aggregate", "matrix_k_change_leakage_fro_activity_weighted_mean"),
    "restricted_inside_residual": ("aggregate", "posthoc_pkp_inside_residual_rms"),
    "encoded_next_outside": ("aggregate", "encoded_next_outside_rms"),
    "global_over_identity": ("aggregate", "global_k_over_identity_residual"),
    "operator_distance": ("operator", "mean_symmetric_normalized_frobenius_distance"),
}


def _median(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    return None if numerator is None or denominator is None or denominator <= 0 else numerator / denominator


def _at_most(value: float | None, limit: float) -> bool:
    return value is not None and math.isfinite(value) and value <= limit


def _at_least(value: float | None, limit: float) -> bool:
    return value is not None and math.isfinite(value) and value >= limit


def _extract_run_row(payload: dict[str, Any]) -> dict[str, Any]:
    regime = payload["regimes"]["persistent_family"]
    true, null = regime["true"], regime["null_summary"]
    row = {
        **{key: payload["provenance"][key] for key in ("system_key", "system_name", "seed", "run_dir")},
        "status": payload["status"],
        **payload["routing"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    for display, (section, metric) in METRICS.items():
        true_value = true[section].get(metric)
        null_record = null[section].get(metric, {})
        null_value = null_record.get("median")
        row[f"{display}_true"] = true_value
        row[f"{display}_null"] = null_value
        row[f"{display}_true_over_null"] = _ratio(true_value, null_value)
    return row


def _system_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[row["system_key"]].append(row)
    rows = []
    for system, group in sorted(grouped.items()):
        eligible = [row for row in group if row["status"] == "eligible"]
        row: dict[str, Any] = {
            "system_key": system,
            "system_name": group[0]["system_name"],
            "run_count": len(group),
            "eligible_seed_count": len(eligible),
            "system_eligible": len(eligible) >= 2,
        }
        for display in METRICS:
            for suffix in ("true", "null", "true_over_null"):
                row[f"{display}_{suffix}"] = _median(
                    item[f"{display}_{suffix}"] for item in eligible
                )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decide(run_rows: list[dict[str, Any]], system_rows: list[dict[str, Any]], card: dict[str, Any]) -> dict[str, Any]:
    gate = card["strong_gate"]
    eligible_runs = sum(row["status"] == "eligible" for row in run_rows)
    eligible_systems = [row for row in system_rows if row["system_eligible"]]
    system_aggregate = {}
    for display in METRICS:
        for suffix in ("true", "null", "true_over_null"):
            system_aggregate[f"{display}_{suffix}"] = _median(
                row[f"{display}_{suffix}"] for row in eligible_systems
            )
    activity_wins = sum(row["activity_leakage_true_over_null"] is not None
                        and row["activity_leakage_true_over_null"] < 1.0
                        for row in eligible_systems)
    activity_change_wins = sum(row["activity_change_leakage_true_over_null"] is not None
                               and row["activity_change_leakage_true_over_null"] < 1.0
                               for row in eligible_systems)
    residual_wins = sum(row["restricted_inside_residual_true_over_null"] is not None
                        and row["restricted_inside_residual_true_over_null"] < 1.0
                        for row in eligible_systems)
    checks = {
        "roster_complete": len(run_rows) == int(card["primary_sparse"]["expected_run_count"]),
        "eligible_runs": eligible_runs >= int(gate["min_eligible_runs"]),
        "eligible_systems": len(eligible_systems) >= int(gate["min_eligible_systems"]),
        "activity_leakage_absolute": _at_most(
            system_aggregate["activity_leakage_true"], float(gate["max_activity_leakage"])),
        "activity_leakage_null_ratio": _at_most(
            system_aggregate["activity_leakage_true_over_null"],
            float(gate["max_activity_leakage_pair_null_ratio"])),
        "activity_leakage_system_wins": activity_wins
            >= int(gate["min_systems_activity_leakage_better_than_null"]),
        "activity_change_leakage_absolute": _at_most(
            system_aggregate["activity_change_leakage_true"],
            float(gate["max_activity_change_leakage"])),
        "activity_change_leakage_null_ratio": _at_most(
            system_aggregate["activity_change_leakage_true_over_null"],
            float(gate["max_activity_change_leakage_pair_null_ratio"])),
        "activity_change_leakage_system_wins": activity_change_wins
            >= int(gate["min_systems_activity_change_leakage_better_than_null"]),
        "restricted_residual_null_ratio": _at_most(
            system_aggregate["restricted_inside_residual_true_over_null"],
            float(gate["max_restricted_inside_residual_pair_null_ratio"])),
        "restricted_residual_system_wins": residual_wins
            >= int(gate["min_systems_residual_better_than_null"]),
        "encoded_next_outside": _at_most(
            system_aggregate["encoded_next_outside_true"], float(gate["max_encoded_next_outside_ratio"])),
        "global_k_over_identity": _at_most(
            system_aggregate["global_over_identity_true"], float(gate["max_global_K_over_identity_residual"])),
        "operator_differentiation_guard": _at_least(
            system_aggregate["operator_distance_true_over_null"],
            float(gate["min_operator_distance_pair_null_ratio"])),
    }
    coverage_valid = all(checks[key] for key in ("roster_complete", "eligible_runs", "eligible_systems"))
    closure_core = coverage_valid and all(checks[key] for key in (
        "activity_leakage_absolute", "activity_leakage_null_ratio", "activity_leakage_system_wins",
        "activity_change_leakage_absolute", "activity_change_leakage_null_ratio",
        "activity_change_leakage_system_wins",
        "encoded_next_outside", "global_k_over_identity",
    ))
    if not coverage_valid:
        decision = "invalid"
    elif all(checks.values()):
        decision = "strong_direct_sum"
    elif closure_core:
        decision = "partial_closure"
    else:
        decision = "failed"
    return {
        "schema_version": 1,
        "decision": decision,
        "eligible_run_count": eligible_runs,
        "eligible_system_count": len(eligible_systems),
        "activity_leakage_system_wins": activity_wins,
        "activity_change_leakage_system_wins": activity_change_wins,
        "restricted_residual_system_wins": residual_wins,
        "system_medians": system_aggregate,
        "checks": checks,
        "interpretation": card["decision_branches"][decision],
        "conditional_dense_tanh_triggered": decision in {"strong_direct_sum", "partial_closure"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    card, card_hash = load_card(args.card)
    payloads = [json.loads(path.read_text()) for path in sorted((args.input_dir / "shards").glob("task_*.json"))]
    if any(payload["card_sha256"] != card_hash for payload in payloads):
        raise RuntimeError("At least one shard used a different prediction card")
    unique = {(item["provenance"]["system_key"], item["provenance"]["seed"]) for item in payloads}
    if len(unique) != len(payloads):
        raise RuntimeError("Duplicate system/seed shards")
    run_rows = [_extract_run_row(payload) for payload in payloads]
    system_rows = _system_rows(run_rows)
    decision = decide(run_rows, system_rows, card)
    args.output_dir.mkdir(parents=True)
    _write_csv(args.output_dir / "run_rows.csv", run_rows)
    _write_csv(args.output_dir / "system_rows.csv", system_rows)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
