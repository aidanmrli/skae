#!/usr/bin/env python3
"""Extract the evaluator-emitted all-current global-K guard from raw shards.

This is a deterministic post-hoc reduction of a regime that the authenticated
evaluator emitted before execution.  It does not replace the internally
frozen persistent-family primary reduction or its decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.neurips_2026.global_k_support_invariance import (
    DEFAULT_CARD,
    load_card,
)


RUN_NAME = "global_k_support_closure_all_current_run_rows.csv"
SYSTEM_NAME = "global_k_support_closure_all_current_system_rows.csv"
ROSTER_NAME = "global_k_support_closure_all_current_source_roster.json"

METRICS = {
    "activity_leakage": ("aggregate", "activity_k_leakage_rms"),
    "matrix_leakage": (
        "aggregate",
        "matrix_k_leakage_fro_activity_weighted_mean",
    ),
    "activity_change_leakage": (
        "aggregate",
        "activity_k_change_leakage_rms",
    ),
    "matrix_change_leakage": (
        "aggregate",
        "matrix_k_change_leakage_fro_activity_weighted_mean",
    ),
    "restricted_inside_residual": (
        "aggregate",
        "posthoc_pkp_inside_residual_rms",
    ),
    "encoded_next_outside": ("aggregate", "encoded_next_outside_rms"),
    "global_over_identity": ("aggregate", "global_k_over_identity_residual"),
    "operator_distance": (
        "operator",
        "mean_symmetric_normalized_frobenius_distance",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: Iterable[float | None]) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(clean)) if clean else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _run_row(payload: dict[str, Any]) -> dict[str, Any]:
    regime = payload["regimes"]["all_current"]
    true, null = regime["true"], regime["null_summary"]
    row = {
        **{
            key: payload["provenance"][key]
            for key in ("system_key", "system_name", "seed", "run_dir")
        },
        "status": payload["status"],
        **payload["routing"],
        "transition_count": true["aggregate"]["transition_count"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }
    for display, (section, metric) in METRICS.items():
        true_value = true[section].get(metric)
        null_value = null[section].get(metric, {}).get("median")
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
        record: dict[str, Any] = {
            "system_key": system,
            "system_name": group[0]["system_name"],
            "run_count": len(group),
            "eligible_seed_count": len(eligible),
            "system_eligible": len(eligible) >= 2,
            "current_coverage": _median(
                row["current_coverage"] for row in eligible
            ),
            "transition_count": _median(
                row["transition_count"] for row in eligible
            ),
        }
        for metric in METRICS:
            for suffix in ("true", "null", "true_over_null"):
                column = f"{metric}_{suffix}"
                record[column] = _median(row[column] for row in eligible)
        rows.append(record)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extract(input_dir: Path, output_dir: Path, card_path: Path) -> None:
    output_paths = [output_dir / name for name in (RUN_NAME, SYSTEM_NAME, ROSTER_NAME)]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("Refusing to overwrite an all-current guard output")
    card, card_hash = load_card(card_path)
    shard_paths = sorted((input_dir / "shards").glob("task_*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shard_paths]
    expected = {
        (system, int(seed))
        for system in card["primary_sparse"]["systems"]
        for seed in card["primary_sparse"]["seeds"]
    }
    observed = {
        (payload["provenance"]["system_key"], int(payload["provenance"]["seed"]))
        for payload in payloads
    }
    if observed != expected or len(payloads) != len(expected):
        raise ValueError("Raw shard roster does not match the internally frozen card")
    if sorted(payload["task_index"] for payload in payloads) != list(range(len(payloads))):
        raise ValueError("Raw shard task indices are incomplete")
    if any(payload["card_sha256"] != card_hash for payload in payloads):
        raise ValueError("Raw shards do not share the internally frozen card")
    if any("all_current" not in payload["regimes"] for payload in payloads):
        raise ValueError("At least one raw shard lacks the evaluator-emitted guard")
    if any(len(payload["regimes"]["all_current"]["null_replicates"]) != 16
           for payload in payloads):
        raise ValueError("All-current guard does not contain exactly 16 nulls per run")

    run_rows = [_run_row(payload) for payload in payloads]
    system_rows = _system_rows(run_rows)
    hash_lines = [f"{sha256(path)}  {path.name}\n" for path in shard_paths]
    portable_digest = hashlib.sha256("".join(hash_lines).encode("utf-8")).hexdigest()
    roster = {
        "schema_version": 1,
        "status": (
            "post-hoc reduction of a pre-execution evaluator-emitted guard; "
            "not the internally frozen primary and not a public preregistration"
        ),
        "source_root": str(input_dir / "shards"),
        "card_sha256": card_hash,
        "shard_count": len(shard_paths),
        "portable_digest": portable_digest,
        "portable_digest_specification": (
            "SHA-256 of sorted lines '<shard_sha256>  <basename>\\n'"
        ),
        "shards": [
            {
                "task_index": payload["task_index"],
                "basename": path.name,
                "sha256": hash_lines[index].split()[0],
                "system_key": payload["provenance"]["system_key"],
                "seed": payload["provenance"]["seed"],
                "checkpoint_sha256": payload["provenance"]["checkpoint_sha256"],
            }
            for index, (path, payload) in enumerate(zip(shard_paths, payloads))
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_paths[0], run_rows)
    _write_csv(output_paths[1], system_rows)
    output_paths[2].write_text(
        json.dumps(roster, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"runs": len(run_rows), "systems": len(system_rows),
                      "portable_digest": portable_digest}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    args = parser.parse_args()
    extract(args.input_dir, args.output_dir, args.card)


if __name__ == "__main__":
    main()
