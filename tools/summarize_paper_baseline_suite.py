#!/usr/bin/env python3
"""Summarize standalone paper baseline suites into paper-table rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


METHOD_ORDER = [
    "dmd",
    "edmd_poly",
    "rbf_dictionary_edmd",
    "kmeans_hard",
    "gmm_hard",
    "gmm_soft",
    "local_edmd_poly_kmeans",
    "local_rbf_edmd_kmeans",
]

METHOD_DISPLAY = {
    "dmd": "DMD",
    "edmd_poly": "Polynomial EDMD",
    "rbf_dictionary_edmd": "RBF-dictionary EDMD",
    "kmeans_hard": r"$k$-means local linear",
    "gmm_hard": "GMM local linear",
    "gmm_soft": "Soft-gated GMM local linear",
    "local_edmd_poly_kmeans": r"Local polynomial EDMD",
    "local_rbf_edmd_kmeans": r"Local RBF-EDMD",
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
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan")
    if arr.size < 4:
        # With the current three-seed baseline protocol, the seed IQM is the
        # ordinary mean over the available finite seed values.
        return float(np.mean(arr))
    lo, hi = np.percentile(arr, [25, 75])
    keep = arr[(arr >= lo) & (arr <= hi)]
    return float(np.mean(keep if keep.size else arr))


def _metric_from_row(row: Dict[str, str]) -> Optional[float]:
    for key in ("cumulative_mse_mean", "rollout_mse"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _read_rows(root: Path) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    rows: List[Dict[str, str]] = []
    status_counts: Dict[str, int] = defaultdict(int)
    for path in sorted((root / "runs").glob("**/rows.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_source"] = str(path)
                rows.append(row)
                status_counts[row.get("status", "")] += 1
    return rows, dict(status_counts)


def _read_rows_many(roots: Sequence[Path]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    rows: List[Dict[str, str]] = []
    status_counts: Dict[str, int] = defaultdict(int)
    for root in roots:
        root_rows, root_status_counts = _read_rows(root)
        rows.extend(root_rows)
        for status, count in root_status_counts.items():
            status_counts[status] += count
    return rows, dict(status_counts)


def _summarize(
    rows: Sequence[Dict[str, str]],
    horizons: Sequence[int],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    values: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    seed_counts: Dict[Tuple[str, int, str], int] = defaultdict(int)
    for row in rows:
        if row.get("status") != "ok":
            continue
        method = row.get("method", "")
        horizon_raw = row.get("horizon", "")
        try:
            horizon = int(float(horizon_raw))
        except ValueError:
            continue
        if horizon not in horizons:
            continue
        metric = _metric_from_row(row)
        if metric is None:
            continue
        system = row.get("system", "")
        values[(method, horizon, system)].append(metric)
        seed_counts[(method, horizon, system)] += 1

    per_system: List[Dict[str, object]] = []
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for (method, horizon, system), vals in sorted(values.items()):
        center = _iqm(vals)
        per_system.append(
            {
                "method": method,
                "horizon": horizon,
                "system": system,
                "seed_iqm_mse": center,
                "num_seeds": seed_counts[(method, horizon, system)],
            }
        )
        if math.isfinite(center):
            grouped[(method, horizon)].append(center)

    summary: List[Dict[str, object]] = []
    for (method, horizon), vals in sorted(grouped.items()):
        arr = np.asarray(vals, dtype=float)
        summary.append(
            {
                "method": method,
                "display": METHOD_DISPLAY.get(method, method.replace("_", " ")),
                "horizon": horizon,
                # Mean over system-level seed-IQMs, matching the paper tables.
                "cross_system_seed_iqm_mean": float(np.mean(arr)),
                "cross_system_seed_iqm_median": float(np.median(arr)),
                "num_systems": int(arr.size),
            }
        )
    return per_system, summary


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tex_num(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return r"\textemdash"
    value = float(value)
    if value == 0.0:
        return "$0$"
    abs_value = abs(value)
    if 1e-3 <= abs_value < 1e4:
        return f"${value:.3g}$"
    exponent = int(math.floor(math.log10(abs_value)))
    mantissa = value / (10.0**exponent)
    return rf"${mantissa:.2g}{{\times}}10^{{{exponent}}}$"


def _summary_lookup(summary: Sequence[Dict[str, object]]) -> Dict[Tuple[str, int], float]:
    return {
        (str(row["method"]), int(row["horizon"])): float(row["cross_system_seed_iqm_mean"])
        for row in summary
    }


def _write_combined_tex(
    path: Path,
    *,
    multibasin_summary: Sequence[Dict[str, object]],
    dysts_summary: Sequence[Dict[str, object]],
    multibasin_horizons: Sequence[int],
    dysts_horizons: Sequence[int],
) -> None:
    mb = _summary_lookup(multibasin_summary)
    dy = _summary_lookup(dysts_summary)
    lines = [
        r"\begin{tabular}{@{}l rrr rrr@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Multibasin, 15 systems} & \multicolumn{3}{c}{Dysts \(dt{\times}30\), 10 systems} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        "Baseline & "
        + " & ".join(f"H{h}" for h in multibasin_horizons)
        + " & "
        + " & ".join(f"H{h}" for h in dysts_horizons)
        + r" \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        if not any((method, h) in mb for h in multibasin_horizons) and not any(
            (method, h) in dy for h in dysts_horizons
        ):
            continue
        cells = [_tex_num(mb.get((method, h))) for h in multibasin_horizons]
        cells.extend(_tex_num(dy.get((method, h))) for h in dysts_horizons)
        lines.append(f"{METHOD_DISPLAY.get(method, method)} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_horizons(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_dirs(raw: str) -> List[Path]:
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multibasin_dir", type=Path, required=True)
    parser.add_argument("--dysts_dir", type=Path, required=True)
    parser.add_argument(
        "--extra_multibasin_dirs",
        default="",
        help="Comma-separated additional multibasin result roots to append.",
    )
    parser.add_argument(
        "--extra_dysts_dirs",
        default="",
        help="Comma-separated additional Dysts result roots to append.",
    )
    parser.add_argument("--multibasin_horizons", default="100,500,1000")
    parser.add_argument("--dysts_horizons", default="100,2000,4000")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("docs/figures/neurips_paper_2026/_tables"),
    )
    parser.add_argument(
        "--tex_name",
        default="table_standalone_state_space_baselines.tex",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    multibasin_horizons = _parse_horizons(args.multibasin_horizons)
    dysts_horizons = _parse_horizons(args.dysts_horizons)
    multibasin_dirs = [args.multibasin_dir, *_parse_dirs(args.extra_multibasin_dirs)]
    dysts_dirs = [args.dysts_dir, *_parse_dirs(args.extra_dysts_dirs)]
    mb_rows, mb_status = _read_rows_many(multibasin_dirs)
    dy_rows, dy_status = _read_rows_many(dysts_dirs)
    mb_per_system, mb_summary = _summarize(mb_rows, multibasin_horizons)
    dy_per_system, dy_summary = _summarize(dy_rows, dysts_horizons)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "paper_baseline_multibasin_per_system.csv", mb_per_system)
    _write_csv(args.out_dir / "paper_baseline_multibasin_summary.csv", mb_summary)
    _write_csv(args.out_dir / "paper_baseline_dysts_per_system.csv", dy_per_system)
    _write_csv(args.out_dir / "paper_baseline_dysts_summary.csv", dy_summary)
    _write_combined_tex(
        args.out_dir / args.tex_name,
        multibasin_summary=mb_summary,
        dysts_summary=dy_summary,
        multibasin_horizons=multibasin_horizons,
        dysts_horizons=dysts_horizons,
    )
    metadata = {
        "multibasin_dir": str(args.multibasin_dir),
        "dysts_dir": str(args.dysts_dir),
        "multibasin_dirs": [str(path) for path in multibasin_dirs],
        "dysts_dirs": [str(path) for path in dysts_dirs],
        "multibasin_status_counts": mb_status,
        "dysts_status_counts": dy_status,
        "multibasin_horizons": multibasin_horizons,
        "dysts_horizons": dysts_horizons,
    }
    (args.out_dir / "paper_baseline_suite_summary_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_dir / args.tex_name}", flush=True)
    if any(status and status != "ok" for status in dy_status) or any(
        status and status != "ok" for status in mb_status
    ):
        print("Warning: non-ok rows are present in one or more baseline suites.", flush=True)


if __name__ == "__main__":
    main()
