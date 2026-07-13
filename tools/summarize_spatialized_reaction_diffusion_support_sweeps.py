#!/usr/bin/env python3
"""Summarize spatialized PDE support-threshold sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence


def _safe_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out


def _safe_mean(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(mean(clean)) if clean else float("nan")


def _path_metadata(path: Path) -> Dict[str, str]:
    parts = path.parts
    if "runs" not in parts:
        return {"model_variant": "", "source_system": "", "grid": "", "seed": "", "setting_slug": ""}
    start = parts.index("runs")
    tail = list(parts[start + 1 :])
    out = {"model_variant": "", "source_system": "", "grid": "", "seed": "", "setting_slug": ""}
    if len(tail) >= 4:
        out["model_variant"] = tail[0]
        out["source_system"] = tail[1]
        out["grid"] = tail[2]
        out["seed"] = tail[3].replace("seed_", "")
    if len(tail) >= 5:
        out["setting_slug"] = "/".join(tail[4:-1])
    return out


def _rows_from_file(path: Path) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text())
    meta = _path_metadata(path)
    model_cfg = payload.get("model_config", {})
    if isinstance(model_cfg, dict):
        meta["model_variant"] = str(model_cfg.get("model_variant") or meta["model_variant"])
    rows: List[Dict[str, object]] = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        all_test = row.get("all_test", {})
        deep_test = row.get("deep_test", {})
        if not isinstance(all_test, dict) or not isinstance(deep_test, dict):
            continue
        rows.append(
            {
                "source_system": meta["source_system"],
                "seed": meta["seed"],
                "model_variant": meta["model_variant"],
                "grid": meta["grid"],
                "setting_slug": meta["setting_slug"],
                "support_threshold": row.get("support_threshold"),
                "family_jaccard": row.get("family_jaccard"),
                "validation_representative_count": row.get("validation_representative_count"),
                "validation_support_size_mean": row.get("validation_support_size_mean"),
                "test_support_size_mean": row.get("test_support_size_mean"),
                "all_test_num_families": all_test.get("num_test_families"),
                "all_test_h_basin_given_family": all_test.get("h_basin_given_family"),
                "all_test_h_family_given_basin": all_test.get("h_family_given_basin"),
                "all_test_purity": all_test.get("purity"),
                "all_test_nmi": all_test.get("nmi"),
                "all_test_ari": all_test.get("ari"),
                "deep_test_num_families": deep_test.get("num_test_families"),
                "deep_test_h_basin_given_family": deep_test.get("h_basin_given_family"),
                "deep_test_h_family_given_basin": deep_test.get("h_family_given_basin"),
                "deep_test_purity": deep_test.get("purity"),
                "deep_test_nmi": deep_test.get("nmi"),
                "deep_test_ari": deep_test.get("ari"),
                "sweep_path": str(path),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: Sequence[Dict[str, object]], nominal_basin_count: int) -> List[Dict[str, object]]:
    groups: Dict[tuple[object, object, object], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_variant"], row["support_threshold"], row["family_jaccard"])].append(row)
    summary: List[Dict[str, object]] = []
    for (model, threshold, jaccard), group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        all_families = [_safe_float(row["all_test_num_families"]) for row in group]
        deep_families = [_safe_float(row["deep_test_num_families"]) for row in group]
        summary.append(
            {
                "model_variant": model,
                "support_threshold": threshold,
                "family_jaccard": jaccard,
                "num_runs": len(group),
                "all_test_mean_num_families": _safe_mean(all_families),
                "all_test_mean_abs_family_minus_basins": _safe_mean(
                    abs(value - nominal_basin_count) for value in all_families
                ),
                "all_test_mean_h_basin_given_family": _safe_mean(
                    _safe_float(row["all_test_h_basin_given_family"]) for row in group
                ),
                "all_test_mean_h_family_given_basin": _safe_mean(
                    _safe_float(row["all_test_h_family_given_basin"]) for row in group
                ),
                "all_test_mean_purity": _safe_mean(_safe_float(row["all_test_purity"]) for row in group),
                "all_test_mean_nmi": _safe_mean(_safe_float(row["all_test_nmi"]) for row in group),
                "deep_test_mean_num_families": _safe_mean(deep_families),
                "deep_test_mean_abs_family_minus_basins": _safe_mean(
                    abs(value - nominal_basin_count) for value in deep_families
                ),
                "deep_test_mean_h_basin_given_family": _safe_mean(
                    _safe_float(row["deep_test_h_basin_given_family"]) for row in group
                ),
                "deep_test_mean_h_family_given_basin": _safe_mean(
                    _safe_float(row["deep_test_h_family_given_basin"]) for row in group
                ),
                "validation_support_size_mean": _safe_mean(
                    _safe_float(row["validation_support_size_mean"]) for row in group
                ),
                "test_support_size_mean": _safe_mean(_safe_float(row["test_support_size_mean"]) for row in group),
            }
        )
    summary.sort(
        key=lambda row: (
            _safe_float(row["all_test_mean_abs_family_minus_basins"]),
            _safe_float(row["all_test_mean_h_basin_given_family"]),
            _safe_float(row["all_test_mean_h_family_given_basin"]),
        )
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--nominal_basin_count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    rows: List[Dict[str, object]] = []
    for path in sorted(input_root.glob("runs/**/support_threshold_sweep.json")):
        rows.extend(_rows_from_file(path))
    summary = _summary_rows(rows, nominal_basin_count=int(args.nominal_basin_count))
    _write_csv(Path(args.rows_csv), rows)
    _write_csv(Path(args.summary_csv), summary)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(
        json.dumps(
            {
                "input_root": str(input_root),
                "num_rows": len(rows),
                "num_summary_rows": len(summary),
                "nominal_basin_count": int(args.nominal_basin_count),
                "top_20": summary[:20],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {len(rows)} rows and {len(summary)} summary rows.")


if __name__ == "__main__":
    main()
