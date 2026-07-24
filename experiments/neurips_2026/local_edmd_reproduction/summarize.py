"""Check exact historical reproduction tolerances and summarize valid evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.neurips_2026.local_edmd_reproduction.contract import (
    BENCHMARKS,
    CARD_PATH,
    EXPECTED_AGGREGATES,
    METHOD_ID,
    SEEDS,
    expected_keys,
)
from experiments.neurips_2026.local_edmd_reproduction.freeze import OUTPUT_NAMES
from experiments.neurips_2026.local_edmd_reproduction.source_lock import (
    sha256_file,
    verify_source_lock,
)


EXACT_FIELDS = (
    "status",
    "method",
    "num_components_grid",
    "selection_horizons",
    "selected_num_components",
    "fitted_component_count",
    "component_counts",
    "feature_method",
    "route_space",
    "feature_dim",
    "train_transitions",
    "state_dim",
    "train_trajectories",
    "validation_trajectories",
    "test_trajectories",
    "num_trajectories",
    "trajectory_length",
    "edmd_degree",
    "min_component_transitions",
)
CONTINUOUS_FIELDS = (
    "endpoint_mse_mean",
    "endpoint_mse_median",
    "endpoint_mse_per_dim_mean",
    "cumulative_mse_mean",
    "cumulative_mse_median",
    "cumulative_mse_per_dim_mean",
    "validation_score",
    "env_dt",
    "train_fraction",
    "validation_fraction",
    "ridge_lambda",
    "max_abs_state_for_fit",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _benchmark_for_system(system: str) -> str:
    matches = [name for name, spec in BENCHMARKS.items() if system in spec.systems]
    if len(matches) != 1:
        raise ValueError(f"Cannot map historical system {system!r}")
    return matches[0]


def _raw_map(root: Path) -> dict[tuple[str, str, int, int], dict[str, str]]:
    mapped: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for path in sorted((root / "runs").glob("**/rows.csv")):
        for row in _read_csv(path):
            if row.get("method") != METHOD_ID:
                continue
            key = (
                _benchmark_for_system(row["system"]),
                row["system"],
                int(row["seed"]),
                int(float(row["horizon"])),
            )
            if key in mapped:
                raise ValueError(f"Duplicate raw key {key}")
            mapped[key] = row
    return mapped


def _load_compact(
    evidence_dir: Path,
) -> tuple[dict[tuple[str, str, int, int], dict[str, str]], dict[str, Any]]:
    provenance_path = evidence_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    mapped: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for benchmark, name in OUTPUT_NAMES.items():
        path = evidence_dir / name
        expected_hash = provenance["outputs"][name]["sha256"]
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Compact evidence hash mismatch for {name}")
        for row in _read_csv(path):
            key = (
                benchmark,
                row["system"],
                int(row["seed"]),
                int(row["horizon"]),
            )
            if key in mapped:
                raise ValueError(f"Duplicate compact key {key}")
            source = Path(row["source_file"])
            if source.is_absolute() or "/network/" in row["source_file"]:
                raise ValueError("Compact evidence contains an absolute source path")
            mapped[key] = row
    if set(mapped) != expected_keys():
        raise ValueError("Compact evidence grid does not match the frozen contract")
    return mapped, provenance


def _candidate_mismatches(
    current: str,
    historical: str,
    *,
    rtol: float,
    atol: float,
) -> list[str]:
    new_rows, old_rows = json.loads(current), json.loads(historical)
    if len(new_rows) != len(old_rows):
        return ["candidate_count"]
    failures: list[str] = []
    for index, (new, old) in enumerate(zip(new_rows, old_rows)):
        for field in (
            "num_components",
            "status",
            "fitted_component_count",
            "component_counts",
        ):
            if new.get(field) != old.get(field):
                failures.append(f"candidate[{index}].{field}")
        if not math.isclose(
            float(new.get("score", float("inf"))),
            float(old.get("score", float("inf"))),
            rel_tol=rtol,
            abs_tol=atol,
        ):
            failures.append(f"candidate[{index}].score")
    return failures


def compare_reproduction(
    current: dict[tuple[str, str, int, int], dict[str, str]],
    historical: dict[tuple[str, str, int, int], dict[str, str]],
    *,
    rtol: float,
    atol: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Apply the pre-frozen exact/discrete and numeric tolerances."""

    mismatches: list[dict[str, Any]] = []
    maxima = {"absolute": 0.0, "relative": 0.0}
    if set(current) != set(historical):
        mismatches.append(
            {
                "kind": "key_grid",
                "missing": len(set(historical) - set(current)),
                "unexpected": len(set(current) - set(historical)),
            }
        )
    for key in sorted(set(current) & set(historical)):
        new, old = current[key], historical[key]
        fields: list[str] = []
        for field in EXACT_FIELDS:
            if str(new.get(field, "")) != str(old.get(field, "")):
                fields.append(field)
        if str(new.get("finite_fraction", "")) != str(old.get("finite_fraction", "")):
            fields.append("finite_fraction")
        for field in CONTINUOUS_FIELDS:
            new_value, old_value = float(new[field]), float(old[field])
            absolute = abs(new_value - old_value)
            relative = absolute / max(abs(old_value), atol)
            maxima["absolute"] = max(maxima["absolute"], absolute)
            maxima["relative"] = max(maxima["relative"], relative)
            if not math.isclose(new_value, old_value, rel_tol=rtol, abs_tol=atol):
                fields.append(field)
        fields.extend(
            _candidate_mismatches(
                new["candidate_scores_json"],
                old["candidate_scores_json"],
                rtol=rtol,
                atol=atol,
            )
        )
        if fields:
            mismatches.append({"kind": "row", "key": list(key), "fields": fields})
    return mismatches, maxima


def _summaries(
    rows: dict[tuple[str, str, int, int], dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (benchmark, system, seed, horizon), row in rows.items():
        del seed
        grouped[(benchmark, system, horizon)].append(
            float(row["cumulative_mse_mean"])
        )
    per_system: list[dict[str, object]] = []
    cross: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (benchmark, system, horizon), values in sorted(grouped.items()):
        if len(values) != len(SEEDS):
            raise ValueError("A system/horizon cell does not contain three seeds")
        seed_mean = float(np.mean(np.asarray(values, dtype=float)))
        per_system.append(
            {
                "benchmark": benchmark,
                "system": system,
                "horizon": horizon,
                "seed_mean_mse": seed_mean,
                "num_seeds": len(values),
            }
        )
        cross[(benchmark, horizon)].append(seed_mean)
    aggregate: list[dict[str, object]] = []
    for (benchmark, horizon), values in sorted(cross.items()):
        array = np.asarray(values, dtype=float)
        aggregate.append(
            {
                "benchmark": benchmark,
                "method": METHOD_ID,
                "horizon": horizon,
                "cross_system_seed_mean": float(np.mean(array)),
                "cross_system_seed_median": float(np.median(array)),
                "num_systems": int(array.size),
            }
        )
    return per_system, aggregate


def _csv_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def run_check(
    result_root: Path, evidence_dir: Path, summary_dir: Path
) -> dict[str, Any]:
    lock = verify_source_lock()
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    tolerance = card["exact_reproduction_tolerances"]["continuous_fields"]
    aggregate_tolerance = card["exact_reproduction_tolerances"]["aggregate_fields"]
    rtol, atol = float(tolerance["relative"]), float(tolerance["absolute"])
    compact, provenance = _load_compact(evidence_dir)
    current = _raw_map(result_root)
    if set(current) != set(compact):
        raise ValueError("Raw reproduction and compact evidence grids differ")
    for key, compact_row in compact.items():
        raw = current[key]
        source = result_root / compact_row["source_file"]
        if sha256_file(source) != compact_row["source_sha256"]:
            raise ValueError(f"Raw reproduction source hash mismatch for {key}")
        if raw["cumulative_mse_mean"] != compact_row["cumulative_mse_mean"]:
            raise ValueError(f"Compact metric drift for {key}")
    historical: dict[tuple[str, str, int, int], dict[str, str]] = {}
    for item in lock["external_trees"].values():
        historical.update(_raw_map(Path(item["root"])))
    mismatches, maxima = compare_reproduction(
        current, historical, rtol=rtol, atol=atol
    )
    per_system, aggregate = _summaries(current)
    aggregate_mismatches = []
    for row in aggregate:
        benchmark, horizon = str(row["benchmark"]), int(row["horizon"])
        expected = EXPECTED_AGGREGATES[benchmark][horizon]
        actual = float(row["cross_system_seed_mean"])
        if not math.isclose(
            actual,
            expected,
            rel_tol=float(aggregate_tolerance["relative"]),
            abs_tol=float(aggregate_tolerance["absolute"]),
        ):
            aggregate_mismatches.append(
                {
                    "benchmark": benchmark,
                    "horizon": horizon,
                    "expected": expected,
                    "actual": actual,
                }
            )
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_payload, aggregate_payload = _csv_bytes(per_system), _csv_bytes(aggregate)
    (summary_dir / "per_system.csv").write_bytes(per_payload)
    (summary_dir / "aggregate.csv").write_bytes(aggregate_payload)
    decision = {
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "valid": not mismatches and not aggregate_mismatches,
        "adjudication": (
            "exact_historical_reproduction_within_frozen_tolerances"
            if not mismatches and not aggregate_mismatches
            else "historical_reproduction_failed_frozen_tolerances"
        ),
        "known_outcome_reproduction": True,
        "row_mismatch_count": len(mismatches),
        "aggregate_mismatch_count": len(aggregate_mismatches),
        "mismatches": mismatches,
        "aggregate_mismatches": aggregate_mismatches,
        "maximum_continuous_deviation": maxima,
        "tolerances": card["exact_reproduction_tolerances"],
        "aggregate_rows": aggregate,
        "input_provenance_sha256": hashlib.sha256(
            (evidence_dir / "provenance.json").read_bytes()
        ).hexdigest(),
        "output_sha256": {
            "per_system.csv": hashlib.sha256(per_payload).hexdigest(),
            "aggregate.csv": hashlib.sha256(aggregate_payload).hexdigest(),
        },
        "claim_limits": provenance["claim_limits"],
        "formal_paired_significance": "not_inferred",
    }
    decision_path = summary_dir / "reproduction_check.json"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {summary_dir / 'per_system.csv'}")
    print(f"Wrote {summary_dir / 'aggregate.csv'}")
    print(f"Wrote {decision_path}")
    if not decision["valid"]:
        raise RuntimeError("Historical reproduction failed frozen tolerances")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_check(args.result_root, args.evidence_dir, args.summary_dir)


if __name__ == "__main__":
    main()
