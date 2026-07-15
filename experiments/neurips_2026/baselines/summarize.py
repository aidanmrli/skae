"""Build standalone-control sidecars from frozen row-level paper evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from experiments.neurips_2026.evidence.statistics import (
    IQM_CONVENTION,
    interquartile_mean,
)
from experiments.neurips_2026.baselines.freeze import (
    CSV_COLUMNS,
    HORIZONS,
    METHOD_ORDER,
    METRIC_PROTOCOLS,
    OUTPUT_NAMES,
    SYSTEMS,
)
from experiments.neurips_2026.paths import PAPER_DATA_DIR, PAPER_TABLE_DIR


DATA_DIR = PAPER_DATA_DIR
TABLE_DIR = PAPER_TABLE_DIR
DEFAULT_PROVENANCE = DATA_DIR / "paper_baseline_evidence_provenance.json"
DEFAULT_INPUTS = {
    benchmark: DATA_DIR / name for benchmark, name in OUTPUT_NAMES.items()
}
OUTPUT_FILES = {
    "multibasin_per_system": "paper_baseline_multibasin_per_system.csv",
    "multibasin_summary": "paper_baseline_multibasin_summary.csv",
    "dysts_per_system": "paper_baseline_dysts_per_system.csv",
    "dysts_summary": "paper_baseline_dysts_summary.csv",
}
METADATA_NAME = "paper_baseline_suite_summary_metadata.json"
METHOD_DISPLAY = {
    "dmd": "DMD",
    "edmd_poly": "Polynomial EDMD",
    "rbf_dictionary_edmd": "RBF-dictionary EDMD",
    "kmeans_hard": r"$k$-means local linear",
    "gmm_hard": "GMM local linear",
    "gmm_soft": "Soft-gated GMM local linear",
}


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _iqm(values: Iterable[float]) -> float:
    return interquartile_mean(values)


def _summarize(
    rows: Sequence[Dict[str, str]], horizons: Sequence[int]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    values: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok" or row.get("method") not in METHOD_ORDER:
            continue
        try:
            horizon = int(float(row.get("horizon", "")))
        except ValueError:
            continue
        metric = _safe_float(row.get("mse"))
        if horizon not in horizons or metric is None:
            continue
        key = (row["method"], horizon, row.get("system", ""))
        values[key].append(metric)

    per_system: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for (method, horizon, system), seed_values in sorted(values.items()):
        center = _iqm(seed_values)
        per_system.append(
            {
                "method": method,
                "horizon": horizon,
                "system": system,
                "seed_iqm_mse": center,
                "num_seeds": len(seed_values),
            }
        )
        grouped[(method, horizon)].append(center)

    summary: List[Dict[str, object]] = []
    for (method, horizon), system_values in sorted(grouped.items()):
        array = np.asarray(system_values, dtype=float)
        summary.append(
            {
                "method": method,
                "display": METHOD_DISPLAY[method],
                "horizon": horizon,
                "cross_system_seed_iqm_mean": float(np.mean(array)),
                "cross_system_seed_iqm_median": float(np.median(array)),
                "num_systems": int(array.size),
            }
        )
    return per_system, summary


def _csv_bytes(rows: Sequence[Dict[str, object]]) -> bytes:
    if not rows:
        return b""
    fieldnames = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_inputs(
    input_paths: dict[str, Path], provenance_path: Path
) -> dict[str, object]:
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid baseline provenance: {provenance_path}") from error
    for benchmark, path in input_paths.items():
        try:
            specification = provenance["outputs"][path.name]
            expected = specification["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Missing provenance entry for {path.name}") from error
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Frozen baseline hash mismatch for {benchmark}: "
                f"expected {expected}, got {actual}"
            )
    return provenance


def _validate_rows(rows: Sequence[Dict[str, str]], benchmark: str) -> None:
    if not rows or set(rows[0]) != set(CSV_COLUMNS):
        raise ValueError(f"Unexpected frozen baseline schema for {benchmark}")
    expected = {
        (method, system, seed, horizon)
        for method in METHOD_ORDER
        for system in SYSTEMS[benchmark]
        for seed in (0, 1, 2)
        for horizon in HORIZONS[benchmark]
    }
    observed = set()
    for row in rows:
        if row["benchmark"] != benchmark or row["status"] != "ok":
            raise ValueError(f"Unexpected benchmark/status in {benchmark} rows")
        key = (
            row["method"],
            row["system"],
            int(row["seed"]),
            int(float(row["horizon"])),
        )
        if key in observed:
            raise ValueError(f"Duplicate frozen baseline key: {key}")
        observed.add(key)
        expected_protocol = METRIC_PROTOCOLS[row["source_family"]]
        if row["metric_protocol"] != expected_protocol:
            raise ValueError(f"Metric-protocol mismatch for {key}")
        source = Path(row["source_file"])
        if source.is_absolute() or "/network/" in row["source_file"]:
            raise ValueError(f"Machine-specific source path for {key}")
        fraction = _safe_float(row["finite_start_fraction"])
        if fraction is None or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Invalid finite-start fraction for {key}")
        metric = _safe_float(row["mse"])
        if metric is not None and metric < 0.0:
            raise ValueError(f"Negative MSE for {key}")
    if observed != expected:
        raise ValueError(
            f"Frozen {benchmark} grid mismatch: "
            f"missing={len(expected - observed)}, unexpected={len(observed - expected)}"
        )


def _coverage(rows: Sequence[Dict[str, str]], benchmark: str) -> dict[str, object]:
    finite_counts: Counter[tuple[str, int, str]] = Counter()
    finite_metric_rows = 0
    for row in rows:
        cell = (row["method"], int(float(row["horizon"])), row["system"])
        if _safe_float(row["mse"]) is not None:
            finite_counts[cell] += 1
            finite_metric_rows += 1
    all_cells = [
        (method, horizon, system)
        for method in METHOD_ORDER
        for horizon in HORIZONS[benchmark]
        for system in SYSTEMS[benchmark]
    ]
    distribution = Counter(finite_counts[cell] for cell in all_cells)
    incomplete = [
        {
            "method": method,
            "horizon": horizon,
            "system": system,
            "finite_seeds": finite_counts[(method, horizon, system)],
        }
        for method, horizon, system in all_cells
        if finite_counts[(method, horizon, system)] < 3
    ]
    return {
        "configured_rows": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "finite_metric_rows": finite_metric_rows,
        "expected_system_cells": len(all_cells),
        "finite_system_cells": sum(
            cell_count
            for seed_count, cell_count in distribution.items()
            if seed_count > 0
        ),
        "complete_three_seed_cells": distribution[3],
        "partial_finite_seed_cells": distribution[1] + distribution[2],
        "zero_finite_seed_cells": distribution[0],
        "finite_seed_count_distribution": {
            str(count): distribution[count] for count in range(4)
        },
        "incomplete_cells": incomplete,
    }


def build_outputs(
    input_paths: dict[str, Path] = DEFAULT_INPUTS,
    provenance_path: Path = DEFAULT_PROVENANCE,
    *,
    out_dir: Path = TABLE_DIR,
) -> dict[Path, bytes]:
    provenance = verify_inputs(input_paths, provenance_path)
    rows_by_benchmark = {
        benchmark: _read_csv(path) for benchmark, path in input_paths.items()
    }
    for benchmark, rows in rows_by_benchmark.items():
        _validate_rows(rows, benchmark)

    generated: dict[str, bytes] = {}
    for benchmark, rows in rows_by_benchmark.items():
        per_system, summary = _summarize(rows, HORIZONS[benchmark])
        generated[OUTPUT_FILES[f"{benchmark}_per_system"]] = _csv_bytes(per_system)
        generated[OUTPUT_FILES[f"{benchmark}_summary"]] = _csv_bytes(summary)

    output_specs = {
        name: {
            "rows": payload.count(b"\n") - 1,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in generated.items()
    }
    metadata = {
        "schema_version": 2,
        "description": (
            "Aggregate inputs for the six unmatched standalone controls "
            "rendered in the forecasting table."
        ),
        "generated_by": "experiments.neurips_2026.baselines.summarize",
        "evidence_level": "sanitized row-level method/system/seed/horizon evidence is frozen",
        "comparison_scope": (
            "three seeds with independently generated trajectories; "
            "descriptive and unmatched"
        ),
        "included_methods": METHOD_ORDER,
        "configured_seeds": [0, 1, 2],
        "seed_summary_convention": IQM_CONVENTION,
        "horizons": {key: list(value) for key, value in HORIZONS.items()},
        "metric_protocols": provenance["metric_protocols"],
        "inputs": {
            path.name: provenance["outputs"][path.name]
            for path in input_paths.values()
        },
        "finite_coverage": {
            benchmark: _coverage(rows, benchmark)
            for benchmark, rows in rows_by_benchmark.items()
        },
        "outputs": output_specs,
        "provenance": "_data/paper_baseline_evidence_provenance.json",
    }
    generated[METADATA_NAME] = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode()
    return {out_dir / name: payload for name, payload in generated.items()}


def write_or_check(outputs: dict[Path, bytes], *, check: bool) -> None:
    stale = [
        path
        for path, payload in outputs.items()
        if not path.is_file() or path.read_bytes() != payload
    ]
    if check:
        if stale:
            raise RuntimeError("Stale baseline sidecars: " + ", ".join(map(str, stale)))
        print(f"Verified {len(outputs)} standalone-control artifacts")
        return
    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multibasin-rows", type=Path, default=DEFAULT_INPUTS["multibasin"])
    parser.add_argument("--dysts-rows", type=Path, default=DEFAULT_INPUTS["dysts"])
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--out-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {"multibasin": args.multibasin_rows, "dysts": args.dysts_rows}
    outputs = build_outputs(inputs, args.provenance, out_dir=args.out_dir)
    write_or_check(outputs, check=args.check)


if __name__ == "__main__":
    main()
