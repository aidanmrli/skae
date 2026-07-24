"""Freeze compact row-level evidence from the authenticated reproduction runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Sequence

from experiments.neurips_2026.local_edmd_reproduction.contract import (
    BENCHMARKS,
    CARD_PATH,
    LOCK_PATH,
    METHOD_ID,
    NUM_COMPONENTS_GRID,
    SEEDS,
    expected_keys,
    expected_task_count,
)
from experiments.neurips_2026.local_edmd_reproduction.source_lock import (
    sha256_file,
    verify_source_lock,
)


CSV_FIELDS = (
    "benchmark",
    "system",
    "seed",
    "method",
    "status",
    "horizon",
    "cumulative_mse_mean",
    "finite_fraction",
    "selected_num_components",
    "validation_score",
    "source_file",
    "source_sha256",
)
OUTPUT_NAMES = {
    "controlled": "local_edmd_poly_controlled_rows.csv",
    "dysts": "local_edmd_poly_dysts_rows.csv",
}


def _benchmark_for_system(system: str) -> str:
    matches = [
        benchmark
        for benchmark, spec in BENCHMARKS.items()
        if system in spec.systems
    ]
    if len(matches) != 1:
        raise ValueError(f"System {system!r} maps to {len(matches)} benchmarks")
    return matches[0]


def _read_raw_rows(result_root: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    paths = sorted((result_root / "runs" / "local_edmd_koopman").glob("**/rows.csv"))
    if len(paths) != expected_task_count():
        raise ValueError(
            f"Expected {expected_task_count()} raw row files, found {len(paths)}"
        )
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(result_root).as_posix()
        source_hash = sha256_file(path)
        hashes[relative] = source_hash
        metadata_path = path.with_name("task_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("row_file_sha256") != source_hash:
            raise ValueError(f"Task metadata hash mismatch for {relative}")
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                system = str(raw.get("system", ""))
                benchmark = _benchmark_for_system(system)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "system": system,
                        "seed": int(raw["seed"]),
                        "method": raw.get("method", ""),
                        "status": raw.get("status", ""),
                        "horizon": int(float(raw["horizon"])),
                        "cumulative_mse_mean": raw.get("cumulative_mse_mean", ""),
                        "finite_fraction": raw.get("finite_fraction", ""),
                        "selected_num_components": int(
                            raw["selected_num_components"]
                        ),
                        "validation_score": raw.get("validation_score", ""),
                        "source_file": relative,
                        "source_sha256": source_hash,
                    }
                )
    return rows, hashes


def _validate_rows(rows: Sequence[dict[str, object]]) -> None:
    observed = {
        (
            str(row["benchmark"]),
            str(row["system"]),
            int(row["seed"]),
            int(row["horizon"]),
        )
        for row in rows
    }
    expected = expected_keys()
    if len(observed) != len(rows) or observed != expected:
        raise ValueError(
            f"Reproduction grid mismatch: missing={len(expected - observed)}, "
            f"unexpected={len(observed - expected)}, duplicates={len(rows)-len(observed)}"
        )
    for row in rows:
        benchmark = str(row["benchmark"])
        if row["status"] != "ok" or row["method"] != METHOD_ID:
            raise ValueError(f"Non-ok or wrong-method reproduction row: {row}")
        if int(row["seed"]) not in SEEDS:
            raise ValueError("Unexpected reproduction seed")
        if int(row["horizon"]) not in BENCHMARKS[benchmark].horizons:
            raise ValueError("Unexpected reproduction horizon")
        if int(row["selected_num_components"]) not in NUM_COMPONENTS_GRID:
            raise ValueError("Selected route count is outside the frozen grid")
        metric = float(str(row["cumulative_mse_mean"]))
        fraction = float(str(row["finite_fraction"]))
        if metric < 0.0 or not 0.0 <= fraction <= 1.0:
            raise ValueError("Invalid metric or finite fraction")


def _csv_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def build_outputs(result_root: Path, evidence_dir: Path) -> dict[Path, bytes]:
    """Build portable compact evidence and provenance in memory."""

    lock = verify_source_lock()
    rows, source_hashes = _read_raw_rows(result_root)
    _validate_rows(rows)
    outputs: dict[Path, bytes] = {}
    output_specs = {}
    method_rank = {METHOD_ID: 0}
    for benchmark, output_name in OUTPUT_NAMES.items():
        subset = [row for row in rows if row["benchmark"] == benchmark]
        subset.sort(
            key=lambda row: (
                method_rank[str(row["method"])],
                int(row["horizon"]),
                str(row["system"]),
                int(row["seed"]),
            )
        )
        payload = _csv_bytes(subset)
        outputs[evidence_dir / output_name] = payload
        output_specs[output_name] = {
            "rows": len(subset),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "columns": list(CSV_FIELDS),
        }
    provenance = {
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "description": (
            "Known-outcome reproduction of the June label-free local polynomial "
            "EDMD switching-style baseline."
        ),
        "historical_status": "post_hoc_provenance_reproduction_not_prospective",
        "method": METHOD_ID,
        "label_policy": (
            "No labels or basin counts enter fitting, selection, refitting, routing, "
            "or forecasting; raw label columns are evaluation diagnostics only."
        ),
        "rollout_update": "reroute_each_predicted_state",
        "card_sha256": sha256_file(CARD_PATH),
        "source_lock_sha256": sha256_file(LOCK_PATH),
        "source_files": source_hashes,
        "outputs": output_specs,
        "claim_limits": [
            "unmatched descriptive reference",
            "three configured seeds",
            "no formal paired significance",
            "no support-alignment readout",
            "no sparsity-only causal conclusion",
        ],
    }
    provenance_payload = (
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    ).encode()
    if b"/network/" in provenance_payload or b"/home/" in provenance_payload:
        raise ValueError("Compact provenance contains a machine-specific path")
    outputs[evidence_dir / "provenance.json"] = provenance_payload
    return outputs


def write_outputs(outputs: dict[Path, bytes]) -> None:
    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_outputs(build_outputs(args.result_root, args.evidence_dir))


if __name__ == "__main__":
    main()

