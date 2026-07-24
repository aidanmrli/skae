"""Build and verify the frozen global-K support-closure evidence packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import numpy as np
import pandas as pd

from experiments.neurips_2026.evidence.global_k_support_closure_rendering import (
    render_support_closure,
)
from experiments.neurips_2026.evidence.global_k_support_closure_guard import (
    verify_guard,
)
from experiments.neurips_2026.paths import (
    PAPER_DATA_DIR,
    PAPER_EVIDENCE_DIR,
    PAPER_TABLE_DIR,
)


RUN_ROWS = PAPER_DATA_DIR / "global_k_support_closure_run_rows.csv"
SYSTEM_ROWS = PAPER_DATA_DIR / "global_k_support_closure_system_rows.csv"
DECISION = PAPER_DATA_DIR / "global_k_support_closure_decision.json"
CARD = PAPER_DATA_DIR / "global_k_support_closure_prediction_card.json"
PROVENANCE = PAPER_DATA_DIR / "global_k_support_closure_protocol_provenance.json"
GUARD_RUN_ROWS = PAPER_DATA_DIR / "global_k_support_closure_all_current_run_rows.csv"
GUARD_SYSTEM_ROWS = PAPER_DATA_DIR / "global_k_support_closure_all_current_system_rows.csv"
GUARD_SOURCE_ROSTER = (
    PAPER_DATA_DIR / "global_k_support_closure_all_current_source_roster.json"
)
TABLE_NAME = "global_k_support_closure_summary.tex"
FIGURE_PDF_NAME = "global_k_support_closure_paired_systems.pdf"
FIGURE_PNG_NAME = "global_k_support_closure_paired_systems.png"

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


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch for {path}: expected {expected}, got {actual}")


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


def _at_most(value: float | None, limit: float) -> bool:
    return value is not None and math.isfinite(value) and value <= limit


def _at_least(value: float | None, limit: float) -> bool:
    return value is not None and math.isfinite(value) and value >= limit


def recompute_system_rows(run_rows: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the internally frozen pre-execution seed-median reduction."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows.to_dict("records"):
        grouped[str(row["system_key"])].append(row)
    records: list[dict[str, Any]] = []
    for system, group in sorted(grouped.items()):
        eligible = [row for row in group if row["status"] == "eligible"]
        record: dict[str, Any] = {
            "system_key": system,
            "system_name": group[0]["system_name"],
            "run_count": len(group),
            "eligible_seed_count": len(eligible),
            "system_eligible": len(eligible) >= 2,
        }
        for metric in METRICS:
            for suffix in ("true", "null", "true_over_null"):
                column = f"{metric}_{suffix}"
                record[column] = _median(row[column] for row in eligible)
        records.append(record)
    return pd.DataFrame.from_records(records)


def recompute_decision(
    run_rows: pd.DataFrame,
    system_rows: pd.DataFrame,
    card: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce the internally frozen decision without importing scratch code."""
    gate = card["strong_gate"]
    eligible_runs = int((run_rows["status"] == "eligible").sum())
    eligible = system_rows.loc[system_rows["system_eligible"]].to_dict("records")
    aggregate: dict[str, float | None] = {}
    for metric in METRICS:
        for suffix in ("true", "null", "true_over_null"):
            column = f"{metric}_{suffix}"
            aggregate[column] = _median(row[column] for row in eligible)
    activity_wins = sum(
        row["activity_leakage_true_over_null"] < 1.0 for row in eligible
    )
    change_wins = sum(
        row["activity_change_leakage_true_over_null"] < 1.0 for row in eligible
    )
    residual_wins = sum(
        row["restricted_inside_residual_true_over_null"] < 1.0
        for row in eligible
    )
    checks = {
        "roster_complete": len(run_rows)
        == int(card["primary_sparse"]["expected_run_count"]),
        "eligible_runs": eligible_runs >= int(gate["min_eligible_runs"]),
        "eligible_systems": len(eligible) >= int(gate["min_eligible_systems"]),
        "activity_leakage_absolute": _at_most(
            aggregate["activity_leakage_true"], gate["max_activity_leakage"]
        ),
        "activity_leakage_null_ratio": _at_most(
            aggregate["activity_leakage_true_over_null"],
            gate["max_activity_leakage_pair_null_ratio"],
        ),
        "activity_leakage_system_wins": activity_wins
        >= int(gate["min_systems_activity_leakage_better_than_null"]),
        "activity_change_leakage_absolute": _at_most(
            aggregate["activity_change_leakage_true"],
            gate["max_activity_change_leakage"],
        ),
        "activity_change_leakage_null_ratio": _at_most(
            aggregate["activity_change_leakage_true_over_null"],
            gate["max_activity_change_leakage_pair_null_ratio"],
        ),
        "activity_change_leakage_system_wins": change_wins
        >= int(gate["min_systems_activity_change_leakage_better_than_null"]),
        "restricted_residual_null_ratio": _at_most(
            aggregate["restricted_inside_residual_true_over_null"],
            gate["max_restricted_inside_residual_pair_null_ratio"],
        ),
        "restricted_residual_system_wins": residual_wins
        >= int(gate["min_systems_residual_better_than_null"]),
        "encoded_next_outside": _at_most(
            aggregate["encoded_next_outside_true"],
            gate["max_encoded_next_outside_ratio"],
        ),
        "global_k_over_identity": _at_most(
            aggregate["global_over_identity_true"],
            gate["max_global_K_over_identity_residual"],
        ),
        "operator_differentiation_guard": _at_least(
            aggregate["operator_distance_true_over_null"],
            gate["min_operator_distance_pair_null_ratio"],
        ),
    }
    coverage_valid = all(
        checks[name]
        for name in ("roster_complete", "eligible_runs", "eligible_systems")
    )
    closure_core = coverage_valid and all(
        checks[name]
        for name in (
            "activity_leakage_absolute",
            "activity_leakage_null_ratio",
            "activity_leakage_system_wins",
            "activity_change_leakage_absolute",
            "activity_change_leakage_null_ratio",
            "activity_change_leakage_system_wins",
            "encoded_next_outside",
            "global_k_over_identity",
        )
    )
    if not coverage_valid:
        branch = "invalid"
    elif all(checks.values()):
        branch = "strong_direct_sum"
    elif closure_core:
        branch = "partial_closure"
    else:
        branch = "failed"
    return {
        "schema_version": 1,
        "decision": branch,
        "eligible_run_count": eligible_runs,
        "eligible_system_count": len(eligible),
        "activity_leakage_system_wins": activity_wins,
        "activity_change_leakage_system_wins": change_wins,
        "restricted_residual_system_wins": residual_wins,
        "system_medians": aggregate,
        "checks": checks,
        "interpretation": card["decision_branches"][branch],
        "conditional_dense_tanh_triggered": branch
        in {"strong_direct_sum", "partial_closure"},
    }


def _compare_frames(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    if list(actual.columns) != list(expected.columns):
        raise ValueError("Recomputed system-row columns differ from frozen rows")
    if actual.shape != expected.shape:
        raise ValueError("Recomputed system-row shape differs from frozen rows")
    for column in actual.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            np.testing.assert_allclose(
                pd.to_numeric(actual[column]).to_numpy(dtype=np.float64),
                pd.to_numeric(expected[column]).to_numpy(dtype=np.float64),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )
        else:
            if actual[column].astype(str).tolist() != expected[column].astype(str).tolist():
                raise ValueError(f"Recomputed values differ in {column}")


def verify_full_packet() -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    dict[str, Any],
]:
    """Authenticate compact inputs, roster, reduction, decision, and scope."""
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    for name, specification in provenance["frozen_inputs"].items():
        require_hash(PAPER_DATA_DIR / name, specification["frozen_sha256"])
    for name, specification in provenance["guard_inputs"].items():
        require_hash(PAPER_DATA_DIR / name, specification["sha256"])
    card = json.loads(CARD.read_text(encoding="utf-8"))
    card["_authenticated_sha256"] = provenance["authenticated_sources"][
        "prediction_card"
    ]["sha256"]
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    # The reducer serialized full Python-float representations.  Pandas' fast
    # default parser may round the final bit, so require its round-trip parser
    # before reproducing the exact JSON decision.
    run_rows = pd.read_csv(RUN_ROWS, float_precision="round_trip")
    frozen_system_rows = pd.read_csv(SYSTEM_ROWS, float_precision="round_trip")

    expected_keys = {
        (system, int(seed))
        for system in card["primary_sparse"]["systems"]
        for seed in card["primary_sparse"]["seeds"]
    }
    observed_keys = set(zip(run_rows["system_key"], run_rows["seed"].astype(int)))
    if observed_keys != expected_keys or len(run_rows) != len(expected_keys):
        raise ValueError("Frozen run rows do not contain the exact pre-execution roster")
    if set(run_rows["status"]) != {"eligible"}:
        raise ValueError("All 45 frozen runs must be eligible")

    recomputed_system_rows = recompute_system_rows(run_rows)
    _compare_frames(recomputed_system_rows, frozen_system_rows)
    recomputed = recompute_decision(run_rows, recomputed_system_rows, card)
    if recomputed != decision:
        raise ValueError("Frozen decision does not match a clean compact-row reduction")

    protocol = provenance["protocol"]
    result = provenance["result"]
    boundary = provenance["claim_boundary"]
    if protocol["scope"].split()[0] != "one-step":
        raise ValueError("Protocol must remain one-step and support-projected")
    if "current and next supports" not in protocol["scope"]:
        raise ValueError("Future-conditioned persistent-family scope is undisclosed")
    if protocol["row_vector_convention"] != (
        "z_next is predicted as z @ K; P is diagonal, so cross-support leakage "
        "is z @ P @ K @ (I-P)."
    ):
        raise ValueError("Row-vector convention drifted")
    if result["decision"] != "partial_closure":
        raise ValueError("Only the internally frozen partial-closure decision is valid")
    failed = [name for name, passed in decision["checks"].items() if not passed]
    if failed != ["operator_differentiation_guard"]:
        raise ValueError(f"Unexpected gate pattern: {failed}")
    if result["dense_specificity_control"] != "pending":
        raise ValueError("Dense specificity status changed without a new packet")
    correction = result.get("executed_decision_string_correction", "")
    if "overstrong" not in correction or "not an invariant chart" not in correction:
        raise ValueError("The executed decision's overstrong wording is not superseded")
    prohibited = " ".join(boundary["not_supported"]).lower()
    if "invariant subspaces" not in prohibited or "distinct local laws" not in prohibited:
        raise ValueError("Mandatory claim exclusions are missing")
    guard_rows, guard_summary, roster = verify_guard(
        GUARD_RUN_ROWS,
        GUARD_SYSTEM_ROWS,
        GUARD_SOURCE_ROSTER,
        run_rows,
        card,
    )
    if guard_summary != provenance["all_current_guard"]:
        raise ValueError("All-current guard summary differs from authenticated provenance")
    if "not a public preregistration" not in roster["status"]:
        raise ValueError("All-current source roster overstates its registration status")
    return frozen_system_rows, decision, provenance, guard_rows, guard_summary


def verify_packet() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Backward-compatible primary-packet view; the guard is still verified."""
    primary, decision, provenance, _guard, _summary = verify_full_packet()
    return primary, decision, provenance


def render_table(decision: dict[str, Any], guard: dict[str, Any]) -> bytes:
    """Return the compact paper table as deterministic UTF-8 bytes."""
    medians = decision["system_medians"]
    primary_rows = (
        ("Raw-$K$ activity leakage", "activity_leakage", "15/15", "Pass"),
        (
            "Raw-$K$ matrix Frobenius leakage",
            "matrix_leakage",
            "15/15",
            "Diagnostic",
        ),
        ("$(K-I)$-normalized leakage", "activity_change_leakage", "15/15", "Pass"),
        (
            "$(K-I)$ matrix Frobenius leakage",
            "matrix_change_leakage",
            "15/15",
            "Diagnostic",
        ),
        ("Post-hoc $PKP$ inside residual", "restricted_inside_residual", "15/15", "Pass"),
        ("Restricted-operator distance", "operator_distance", "--", r"\textbf{Fail}"),
    )
    guard_medians = guard["system_medians"]
    guard_wins = guard["system_wins"]
    guard_rows = (
        (
            "Raw-$K$ activity leakage",
            "activity_leakage",
            f"{guard_wins['activity_leakage']}/15",
            "Ref. pass",
        ),
        (
            "Raw-$K$ matrix Frobenius leakage",
            "matrix_leakage",
            f"{guard_wins['matrix_leakage']}/15",
            "Diagnostic",
        ),
        (
            "$(K-I)$-normalized leakage",
            "activity_change_leakage",
            f"{guard_wins['activity_change_leakage']}/15",
            "Ref. pass",
        ),
        (
            "$(K-I)$ matrix Frobenius leakage",
            "matrix_change_leakage",
            f"{guard_wins['matrix_change_leakage']}/15",
            "Diagnostic",
        ),
        (
            "Post-hoc $PKP$ inside residual",
            "restricted_inside_residual",
            f"{guard_wins['restricted_inside_residual']}/15",
            "Ref. pass",
        ),
        (
            "Restricted-operator distance",
            "operator_distance",
            "0/15",
            r"\textbf{Ref. fail}",
        ),
    )
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Slice & Diagnostic & Observed & Matched null & Obs./null & Wins & Gate \\",
        r"\midrule",
    ]
    for index, (label, metric, wins, gate) in enumerate(guard_rows):
        slice_label = r"All-current guard$^{\dagger}$" if index == 0 else ""
        lines.append(
            f"{slice_label} & {label} & {guard_medians[f'{metric}_true']:.4f} & "
            f"{guard_medians[f'{metric}_null']:.4f} & "
            f"{guard_medians[f'{metric}_true_over_null']:.3f} & {wins} & {gate} \\\\"
        )
    lines.append(r"\midrule")
    for index, (label, metric, wins, gate) in enumerate(primary_rows):
        lines.append(
            f"{'Persistent primary' if index == 0 else ''} & {label} & "
            f"{medians[f'{metric}_true']:.4f} & {medians[f'{metric}_null']:.4f} & "
            f"{medians[f'{metric}_true_over_null']:.3f} & {wins} & {gate} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{7}{l}{\footnotesize $^\dagger$Post-hoc reduction of a pre-execution evaluator-emitted no-next-state guard; not a second frozen decision.} \\",
            r"\multicolumn{7}{l}{\footnotesize The guard adds only 3,527 transitions (0.96 percentage points); medians are over system-level seed medians.} \\",
            r"\multicolumn{7}{l}{\footnotesize Frozen primary decision: partial closure; the differentiation guard fails; $PKP$ is a post-hoc restriction.} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def build(figure_dir: Path = PAPER_EVIDENCE_DIR, table_dir: Path = PAPER_TABLE_DIR) -> None:
    primary_rows, decision, _provenance, guard_rows, guard_summary = verify_full_packet()
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / TABLE_NAME).write_bytes(render_table(decision, guard_summary))
    render_support_closure(
        guard_rows,
        primary_rows,
        figure_dir / FIGURE_PDF_NAME,
        figure_dir / FIGURE_PNG_NAME,
    )


def check() -> None:
    _rows, _decision, provenance, _guard, _summary = verify_full_packet()
    outputs = provenance["generated_outputs"]
    expected_names = {TABLE_NAME, FIGURE_PDF_NAME, FIGURE_PNG_NAME}
    if set(outputs) != expected_names:
        raise ValueError("Provenance does not enumerate exactly the three outputs")
    active = {
        TABLE_NAME: PAPER_TABLE_DIR / TABLE_NAME,
        FIGURE_PDF_NAME: PAPER_EVIDENCE_DIR / FIGURE_PDF_NAME,
        FIGURE_PNG_NAME: PAPER_EVIDENCE_DIR / FIGURE_PNG_NAME,
    }
    for name, path in active.items():
        require_hash(path, outputs[name]["sha256"])
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        build(root / "figures", root / "tables")
        for name, active_path in active.items():
            rebuilt = (
                root / "tables" / name
                if name == TABLE_NAME
                else root / "figures" / name
            )
            if rebuilt.read_bytes() != active_path.read_bytes():
                raise ValueError(f"Evidence artifact is not byte-reproducible: {name}")
    print("Global-K support-closure packet is authenticated and byte-reproducible.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        check()
    else:
        build()
        print("Built global-K support-closure table and paired-system figure.")


if __name__ == "__main__":
    main()
