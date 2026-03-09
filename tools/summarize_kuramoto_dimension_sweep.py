#!/usr/bin/env python3
"""Summarize Kuramoto dimension-scaling results from collected forecasting rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            out = float(raw)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def _safe_int(value: object) -> Optional[int]:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _parse_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _seed_name(row: Dict[str, object]) -> str:
    seed_name = str(row.get("seed_name", "")).strip()
    if seed_name and seed_name != "None":
        return seed_name
    seed = _safe_int(row.get("seed"))
    return f"seed_{seed or 0}"


def _latest_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("root_label", "")).strip(),
            str(row.get("system_key", "")).strip(),
            _seed_name(row),
        )
        prev = latest.get(key)
        run_key = (str(row.get("run_id", "")), str(row.get("run_dir", "")))
        prev_key = (str(prev.get("run_id", "")), str(prev.get("run_dir", ""))) if prev else None
        if prev is None or run_key > prev_key:
            latest[key] = row
    return sorted(latest.values(), key=lambda row: (str(row.get("root_label", "")), _seed_name(row)))


def _infer_dimension(row: Dict[str, object]) -> Optional[int]:
    value = _safe_int(row.get("kuramoto_num_oscillators"))
    if value is not None:
        return value
    root_label = str(row.get("root_label", "")).strip()
    match = re.search(r"_n(\d+)$", root_label)
    if match:
        return int(match.group(1))
    run_dir = str(row.get("run_dir", ""))
    match = re.search(r"/n_(\d+)(?:/|$)", run_dir)
    if match:
        return int(match.group(1))
    return None


def _median(values: List[float]) -> Optional[float]:
    return median(values) if values else None


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def _root_prefix(root_label: str) -> str:
    match = re.match(r"(.+)_n\d+$", root_label)
    return match.group(1) if match else root_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Kuramoto dimension-scaling rows.")
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--good_threshold", type=float, default=10.0)
    parser.add_argument("--horizons", nargs="+", type=int, default=[100, 500, 1000])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latest_rows = _latest_rows(_parse_rows(Path(args.rows_csv)))
    grouped: Dict[int, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in latest_rows:
        if str(row.get("system_key", "")).strip() != "kuramoto":
            continue
        dimension = _infer_dimension(row)
        if dimension is None:
            continue
        grouped[dimension][str(row.get("root_label", "")).strip()].append(row)

    horizons = list(args.horizons)
    root_summary: Dict[str, Dict[str, object]] = {}
    dimension_summary: List[Dict[str, object]] = []

    for dimension in sorted(grouped):
        by_root = grouped[dimension]
        roots = sorted(by_root)
        row_payload: Dict[str, object] = {"dimension": dimension}
        for root in roots:
            prefix = _root_prefix(root)
            rows = by_root[root]
            root_payload: Dict[str, object] = {
                "dimension": dimension,
                "root_label": root,
                "root_prefix": prefix,
                "n_seeds": len(rows),
            }
            for horizon in horizons:
                best_vals = [
                    value
                    for value in (_safe_float(row.get(f"h{horizon}_best_periodic_mean")) for row in rows)
                    if value is not None
                ]
                every_vals = [
                    value
                    for value in (_safe_float(row.get(f"h{horizon}_every_step_mean")) for row in rows)
                    if value is not None
                ]
                root_payload[f"h{horizon}_best_periodic_min"] = min(best_vals) if best_vals else None
                root_payload[f"h{horizon}_best_periodic_median"] = _median(best_vals)
                root_payload[f"h{horizon}_best_periodic_max"] = max(best_vals) if best_vals else None
                root_payload[f"h{horizon}_every_step_median"] = _median(every_vals)
            root_payload["good_at_h1000"] = (
                root_payload.get("h1000_best_periodic_median") is not None
                and float(root_payload["h1000_best_periodic_median"]) < args.good_threshold
            )
            root_payload["all_seeds_good_at_h1000"] = (
                root_payload.get("h1000_best_periodic_max") is not None
                and float(root_payload["h1000_best_periodic_max"]) < args.good_threshold
            )
            root_summary[root] = root_payload
            row_payload[prefix] = root_payload

        anchor = row_payload.get("generic_sparse")
        dense = row_payload.get("lista_dense_promoted")
        blockdiag = row_payload.get("lista_blockdiag")
        if isinstance(anchor, dict):
            anchor_h1000 = _safe_float(anchor.get("h1000_best_periodic_median"))
        else:
            anchor_h1000 = None
        for prefix, payload in (("lista_dense_promoted", dense), ("lista_blockdiag", blockdiag)):
            metric = _safe_float(payload.get("h1000_best_periodic_median")) if isinstance(payload, dict) else None
            row_payload[f"{prefix}_vs_generic_ratio_h1000"] = (
                metric / anchor_h1000 if metric is not None and anchor_h1000 not in (None, 0.0) else None
            )
            row_payload[f"{prefix}_vs_generic_delta_h1000"] = (
                metric - anchor_h1000 if metric is not None and anchor_h1000 is not None else None
            )
        dimension_summary.append(row_payload)

    out_json = out_dir / "kuramoto_dimension_summary.json"
    out_md = out_dir / "kuramoto_dimension_summary.md"
    out_json.write_text(
        json.dumps(
            {
                "good_threshold": args.good_threshold,
                "horizons": horizons,
                "dimension_summary": dimension_summary,
                "root_summary": root_summary,
            },
            indent=2,
        )
    )

    lines = [
        "# Kuramoto Dimension Sweep Summary",
        "",
        f"- Good threshold: H1000 best-periodic median < {args.good_threshold}",
        f"- Horizons: {', '.join(f'H{h}' for h in horizons)}",
        "",
        "## H1000 Scaling Table",
        "",
        "| N | generic_sparse | dense promoted | dense/generic | blockdiag | blockdiag/generic | best root | dense all-seeds-good | blockdiag all-seeds-good |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]

    for row in dimension_summary:
        generic = row.get("generic_sparse")
        dense = row.get("lista_dense_promoted")
        blockdiag = row.get("lista_blockdiag")
        metrics = []
        for label, payload in (
            ("generic_sparse", generic),
            ("lista_dense_promoted", dense),
            ("lista_blockdiag", blockdiag),
        ):
            if isinstance(payload, dict):
                metric = _safe_float(payload.get("h1000_best_periodic_median"))
                if metric is not None:
                    metrics.append((metric, label))
        best_root = min(metrics)[1] if metrics else "N/A"
        lines.append(
            f"| {row['dimension']} | "
            f"{_fmt(_safe_float(generic.get('h1000_best_periodic_median')) if isinstance(generic, dict) else None)} | "
            f"{_fmt(_safe_float(dense.get('h1000_best_periodic_median')) if isinstance(dense, dict) else None)} | "
            f"{_fmt(_safe_float(row.get('lista_dense_promoted_vs_generic_ratio_h1000')))} | "
            f"{_fmt(_safe_float(blockdiag.get('h1000_best_periodic_median')) if isinstance(blockdiag, dict) else None)} | "
            f"{_fmt(_safe_float(row.get('lista_blockdiag_vs_generic_ratio_h1000')))} | "
            f"{best_root} | "
            f"{int(bool(isinstance(dense, dict) and dense.get('all_seeds_good_at_h1000')))} | "
            f"{int(bool(isinstance(blockdiag, dict) and blockdiag.get('all_seeds_good_at_h1000')))} |"
        )

    lines.extend(
        [
            "",
            "## Per-Root Seed Robustness at H1000",
            "",
            "| N | root | seeds | min | median | max | median every-step | good | all seeds good |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dimension in sorted(grouped):
        roots = grouped[dimension]
        for root in sorted(roots):
            payload = root_summary[root]
            lines.append(
                f"| {dimension} | {root} | {payload['n_seeds']} | "
                f"{_fmt(_safe_float(payload.get('h1000_best_periodic_min')))} | "
                f"{_fmt(_safe_float(payload.get('h1000_best_periodic_median')))} | "
                f"{_fmt(_safe_float(payload.get('h1000_best_periodic_max')))} | "
                f"{_fmt(_safe_float(payload.get('h1000_every_step_median')))} | "
                f"{int(bool(payload.get('good_at_h1000')))} | "
                f"{int(bool(payload.get('all_seeds_good_at_h1000')))} |"
            )
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
