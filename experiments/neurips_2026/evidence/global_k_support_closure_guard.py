"""Validation for the compact all-current global-K closure guard."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


METRICS = (
    "activity_leakage",
    "matrix_leakage",
    "activity_change_leakage",
    "matrix_change_leakage",
    "restricted_inside_residual",
    "encoded_next_outside",
    "global_over_identity",
    "operator_distance",
)


def _median(values: Iterable[float | None]) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return float(np.median(clean)) if clean else None


def recompute_guard_system_rows(run_rows: pd.DataFrame) -> pd.DataFrame:
    """Reduce run rows by seed within each system."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows.to_dict("records"):
        grouped[str(row["system_key"])].append(row)
    records = []
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
        records.append(record)
    return pd.DataFrame.from_records(records)


def _compare_frames(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
        raise ValueError("All-current system-row schema or shape drifted")
    for column in actual.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(
                pd.to_numeric(actual[column]).to_numpy(dtype=np.float64),
                pd.to_numeric(expected[column]).to_numpy(dtype=np.float64),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )
        elif actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
            raise ValueError(f"All-current system-row values differ in {column}")


def summarize_guard(
    run_rows: pd.DataFrame,
    system_rows: pd.DataFrame,
    primary_run_rows: pd.DataFrame,
    card: dict[str, Any],
) -> dict[str, Any]:
    """Return the post-hoc guard summary without assigning a primary decision."""
    eligible = system_rows.loc[system_rows["system_eligible"]]
    medians: dict[str, float | None] = {
        "current_coverage": _median(eligible["current_coverage"]),
        "transition_count": _median(eligible["transition_count"]),
    }
    for metric in METRICS:
        for suffix in ("true", "null", "true_over_null"):
            column = f"{metric}_{suffix}"
            medians[column] = _median(eligible[column])
    wins = {
        metric: int((eligible[f"{metric}_true_over_null"] < 1.0).sum())
        for metric in (
            "activity_leakage",
            "matrix_leakage",
            "activity_change_leakage",
            "matrix_change_leakage",
            "restricted_inside_residual",
            "global_over_identity",
        )
    }
    wins["operator_distance_above_null"] = int(
        (eligible["operator_distance_true_over_null"] > 1.0).sum()
    )
    score_transitions = (
        int(card["corpus"]["score_trajectories"])
        * int(card["corpus"]["trajectory_length"])
    )
    persistent_counts = np.rint(
        primary_run_rows["persistent_coverage"].to_numpy(dtype=np.float64)
        * score_transitions
    ).astype(np.int64)
    added = int(run_rows["transition_count"].sum() - persistent_counts.sum())
    gate = card["strong_gate"]
    reference_checks = {
        "activity_leakage_absolute": medians["activity_leakage_true"]
        <= gate["max_activity_leakage"],
        "activity_leakage_null_ratio": medians["activity_leakage_true_over_null"]
        <= gate["max_activity_leakage_pair_null_ratio"],
        "activity_change_leakage_absolute": medians["activity_change_leakage_true"]
        <= gate["max_activity_change_leakage"],
        "activity_change_leakage_null_ratio": medians[
            "activity_change_leakage_true_over_null"
        ] <= gate["max_activity_change_leakage_pair_null_ratio"],
        "restricted_residual_null_ratio": medians[
            "restricted_inside_residual_true_over_null"
        ] <= gate["max_restricted_inside_residual_pair_null_ratio"],
        "operator_differentiation_guard": medians[
            "operator_distance_true_over_null"
        ] >= gate["min_operator_distance_pair_null_ratio"],
    }
    return {
        "status": (
            "post-hoc reduction of a pre-execution evaluator-emitted guard; "
            "not the internally frozen primary"
        ),
        "eligible_run_count": int((run_rows["status"] == "eligible").sum()),
        "eligible_system_count": int(system_rows["system_eligible"].sum()),
        "system_medians": medians,
        "system_wins": wins,
        "added_transition_count": added,
        "added_transition_percentage_points": (
            100.0 * added / (len(run_rows) * score_transitions)
        ),
        "reference_checks_not_a_second_frozen_decision": reference_checks,
    }


def verify_guard(
    run_path: Path,
    system_path: Path,
    roster_path: Path,
    primary_run_rows: pd.DataFrame,
    card: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Authenticate roster structure and reproduce the compact guard reduction."""
    run_rows = pd.read_csv(run_path, float_precision="round_trip")
    frozen_system_rows = pd.read_csv(system_path, float_precision="round_trip")
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    expected = {
        (system, int(seed))
        for system in card["primary_sparse"]["systems"]
        for seed in card["primary_sparse"]["seeds"]
    }
    observed = set(zip(run_rows["system_key"], run_rows["seed"].astype(int)))
    roster_keys = {
        (row["system_key"], int(row["seed"])) for row in roster["shards"]
    }
    if observed != expected or roster_keys != expected or len(run_rows) != 45:
        raise ValueError("All-current compact roster differs from the frozen roster")
    if roster["card_sha256"] != card.get("_authenticated_sha256"):
        raise ValueError("All-current source roster card hash drifted")
    if roster["shard_count"] != 45 or roster["portable_digest"] != (
        "cd3c708ba7314f91074bf5b4c84a93e5736c8a6930f2a3869d1970d49b8f9836"
    ):
        raise ValueError("All-current source-shard roster drifted")
    if sorted(row["task_index"] for row in roster["shards"]) != list(range(45)):
        raise ValueError("All-current source tasks are incomplete")
    if set(run_rows["status"]) != {"eligible"}:
        raise ValueError("All-current compact rows must retain all eligible runs")
    if float(run_rows["current_coverage"].min()) < card["eligibility"]["min_current_coverage"]:
        raise ValueError("All-current guard current-state coverage failed")
    shared = ("system_key", "seed", "run_dir", "status", "current_coverage",
              "persistent_coverage")
    left = run_rows.sort_values(["system_key", "seed"]).reset_index(drop=True)
    right = primary_run_rows.sort_values(["system_key", "seed"]).reset_index(drop=True)
    for column in shared:
        if left[column].astype(str).tolist() != right[column].astype(str).tolist():
            raise ValueError(f"All-current and primary rosters differ in {column}")
    recomputed = recompute_guard_system_rows(run_rows)
    _compare_frames(recomputed, frozen_system_rows)
    return frozen_system_rows, summarize_guard(run_rows, recomputed, right, card), roster
