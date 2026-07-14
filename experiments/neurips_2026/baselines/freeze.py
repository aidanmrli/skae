"""Freeze sanitized row-level evidence for the standalone paper controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Sequence

from experiments.neurips_2026.protocol import (
    CONTROLLED_PAPER_PROTOCOL,
    DYSTS_PAPER_PROTOCOL,
)
from experiments.neurips_2026.paths import PAPER_DATA_DIR


DEFAULT_OUT_DIR = PAPER_DATA_DIR
METHOD_ORDER = [
    "dmd",
    "edmd_poly",
    "rbf_dictionary_edmd",
    "kmeans_hard",
    "gmm_hard",
    "gmm_soft",
]
HORIZONS = {
    "multibasin": (100, 500, 1000),
    "dysts": (100, 2000, 4000),
}
SYSTEMS = {
    "multibasin": CONTROLLED_PAPER_PROTOCOL.system_keys,
    "dysts": DYSTS_PAPER_PROTOCOL.system_keys,
}
OUTPUT_NAMES = {
    "multibasin": "paper_baseline_multibasin_rows.csv",
    "dysts": "paper_baseline_dysts_rows.csv",
}
FAMILY_METHODS = {
    "classical_koopman": {"dmd", "edmd_poly", "rbf_dictionary_edmd"},
    "mixture_local_linear": {"kmeans_hard", "gmm_hard", "gmm_soft"},
}
METRIC_FIELDS = {
    "classical_koopman": ("cumulative_mse_mean", "finite_fraction"),
    "mixture_local_linear": ("rollout_mse", "rollout_finite_fraction"),
}
METRIC_PROTOCOLS = {
    "classical_koopman": "ordinary_through_h_mean_finite_starts",
    "mixture_local_linear": "finite_step_prefix_mean",
}
CSV_COLUMNS = (
    "benchmark",
    "source_family",
    "system",
    "seed",
    "method",
    "status",
    "horizon",
    "mse",
    "finite_start_fraction",
    "metric_protocol",
    "source_file",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _finite_text(value: object) -> str:
    text = str(value or "").strip()
    try:
        return text if math.isfinite(float(text)) else ""
    except ValueError:
        return ""


def _csv_bytes(rows: Sequence[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=CSV_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _source_rows(
    root: Path, benchmark: str
) -> tuple[list[dict[str, object]], dict[str, str]]:
    run_root = root / "runs"
    paths = sorted(run_root.glob("**/rows.csv"))
    if not paths:
        raise FileNotFoundError(f"No rows.csv files under {run_root}")
    rows: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        if len(parts) < 3 or parts[0] != "runs" or parts[1] not in FAMILY_METHODS:
            raise ValueError(f"Unexpected standalone-control source path: {relative}")
        family = parts[1]
        payload = path.read_bytes()
        hashes[relative] = _sha256(payload)
        reader = csv.DictReader(io.StringIO(payload.decode()))
        metric_field, fraction_field = METRIC_FIELDS[family]
        for raw in reader:
            method = str(raw.get("method", ""))
            if method not in METHOD_ORDER:
                continue
            if method not in FAMILY_METHODS[family]:
                raise ValueError(f"Method {method} appears under wrong family {family}")
            rows.append(
                {
                    "benchmark": benchmark,
                    "source_family": family,
                    "system": str(raw.get("system", "")),
                    "seed": str(raw.get("seed", "")),
                    "method": method,
                    "status": str(raw.get("status", "")),
                    "horizon": str(raw.get("horizon", "")),
                    "mse": _finite_text(raw.get(metric_field)),
                    "finite_start_fraction": _finite_text(raw.get(fraction_field)),
                    "metric_protocol": METRIC_PROTOCOLS[family],
                    "source_file": relative,
                }
            )
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    rows.sort(
        key=lambda row: (
            method_rank[str(row["method"])],
            int(float(str(row["horizon"]))),
            str(row["system"]),
            int(str(row["seed"])),
        )
    )
    return rows, hashes


def _validate_grid(rows: Sequence[dict[str, object]], benchmark: str) -> None:
    expected = {
        (method, system, seed, horizon)
        for method in METHOD_ORDER
        for system in SYSTEMS[benchmark]
        for seed in (0, 1, 2)
        for horizon in HORIZONS[benchmark]
    }
    observed = {
        (
            str(row["method"]),
            str(row["system"]),
            int(str(row["seed"])),
            int(float(str(row["horizon"]))),
        )
        for row in rows
    }
    if len(observed) != len(rows):
        raise ValueError(f"Duplicate {benchmark} method/system/seed/horizon rows")
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        unexpected = sorted(observed - expected)[:5]
        raise ValueError(
            f"{benchmark} grid mismatch; missing={missing}, unexpected={unexpected}"
        )
    if any(row["status"] != "ok" for row in rows):
        raise ValueError(f"{benchmark} frozen packet contains non-ok rows")
    for row in rows:
        fraction = float(str(row["finite_start_fraction"]))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Invalid finite-start fraction in {benchmark}")


def build_outputs(
    multibasin_root: Path,
    dysts_root: Path,
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[Path, bytes]:
    source_roots = {"multibasin": multibasin_root, "dysts": dysts_root}
    csv_outputs: dict[Path, bytes] = {}
    source_campaigns = {}
    output_specs = {}
    for benchmark, source_root in source_roots.items():
        rows, hashes = _source_rows(source_root, benchmark)
        _validate_grid(rows, benchmark)
        payload = _csv_bytes(rows)
        name = OUTPUT_NAMES[benchmark]
        csv_outputs[out_dir / name] = payload
        output_specs[name] = {
            "bytes": len(payload),
            "columns": list(CSV_COLUMNS),
            "rows": len(rows),
            "sha256": _sha256(payload),
        }
        source_campaigns[benchmark] = {
            "campaign_id": source_root.name,
            "file_count": len(hashes),
            "files": hashes,
        }
    provenance = {
        "schema_version": 1,
        "description": "Sanitized row-level evidence for unmatched standalone controls.",
        "configured_seeds": [0, 1, 2],
        "included_methods": METHOD_ORDER,
        "horizons": {key: list(value) for key, value in HORIZONS.items()},
        "metric_protocols": {
            "ordinary_through_h_mean_finite_starts": (
                "Classical DMD/EDMD: ordinary through-H mean per start; "
                "only starts with a finite mean contribute."
            ),
            "finite_step_prefix_mean": (
                "Mixture-local controls: mean finite steps per start, so a "
                "finite prefix may contribute."
            ),
        },
        "outputs": output_specs,
        "source_campaigns": source_campaigns,
    }
    provenance_payload = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    if b"/network/" in provenance_payload or b"/home/" in provenance_payload:
        raise ValueError("Frozen provenance contains a machine-specific path")
    csv_outputs[out_dir / "paper_baseline_evidence_provenance.json"] = provenance_payload
    return csv_outputs


def write_or_check(outputs: dict[Path, bytes], *, check: bool) -> None:
    stale = [
        path
        for path, payload in outputs.items()
        if not path.is_file() or path.read_bytes() != payload
    ]
    if check:
        if stale:
            raise RuntimeError("Stale baseline evidence: " + ", ".join(map(str, stale)))
        print(f"Verified {len(outputs)} frozen baseline-evidence artifacts")
        return
    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multibasin-root", type=Path, required=True)
    parser.add_argument("--dysts-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_outputs(
        args.multibasin_root, args.dysts_root, out_dir=args.out_dir
    )
    write_or_check(outputs, check=args.check)


if __name__ == "__main__":
    main()
