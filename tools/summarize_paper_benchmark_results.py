#!/usr/bin/env python3
"""Summarize the final paper benchmark rows into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
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


def _parse_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _seed_name(row: Dict[str, object]) -> str:
    seed_name = str(row.get("seed_name", "")).strip()
    if seed_name and seed_name != "None":
        return seed_name
    seed_value = _safe_float(row.get("seed"))
    return f"seed_{int(round(seed_value or 0))}"


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
    return sorted(latest.values(), key=lambda row: (row.get("root_label", ""), row.get("system_key", ""), _seed_name(row)))


def _system_stats(rows: List[Dict[str, object]], horizons: List[int]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    latest = _latest_rows(rows)
    by_root_system: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in latest:
        by_root_system[(str(row["root_label"]), str(row["system_key"]))].append(row)

    for (root, system), system_rows in sorted(by_root_system.items()):
        out: Dict[str, object] = {"root_label": root, "system_key": system}
        for horizon in horizons:
            for prefix in ("every_step", "best_periodic"):
                for suffix in ("mean", "per_dim_mean"):
                    key = f"h{horizon}_{prefix}_{suffix}"
                    values = [_safe_float(row.get(key)) for row in system_rows]
                    values = [value for value in values if value is not None]
                    out[f"{key}_median"] = median(values) if values else None
            modes = [str(row.get(f"h{horizon}_best_periodic_mode")) for row in system_rows if row.get(f"h{horizon}_best_periodic_mode")]
            out[f"h{horizon}_best_periodic_mode_consensus"] = Counter(modes).most_common(1)[0][0] if modes else None
        grouped[root].append(out)
    return grouped


def _root_summary(system_rows: Dict[str, List[Dict[str, object]]], horizons: List[int]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for root, rows in system_rows.items():
        stats: Dict[str, object] = {"n_systems": len(rows)}
        for horizon in horizons:
            for prefix in ("every_step", "best_periodic"):
                for suffix in ("mean", "per_dim_mean"):
                    key = f"h{horizon}_{prefix}_{suffix}_median"
                    values = [_safe_float(row.get(key)) for row in rows]
                    values = [value for value in values if value is not None]
                    stats[f"{key}_across_systems"] = median(values) if values else None
            catastrophic = [
                _safe_float(row.get(f"h{horizon}_best_periodic_mean_median"))
                for row in rows
            ]
            catastrophic = [value for value in catastrophic if value is not None and value >= 1000.0]
            stats[f"h{horizon}_catastrophic_systems"] = len(catastrophic)
        summary[root] = stats
    return summary


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the paper benchmark rows CSV.")
    parser.add_argument("--rows_csv", required=True, help="Rows CSV from collect_forecasting_roots.py")
    parser.add_argument("--output_dir", required=True, help="Directory for summary artifacts")
    parser.add_argument("--horizons", nargs="+", type=int, default=[100, 500, 1000], help="Horizons to summarize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _parse_rows(Path(args.rows_csv))
    system_rows = _system_stats(rows, horizons=args.horizons)
    root_summary = _root_summary(system_rows, horizons=args.horizons)

    payload = {
        "paper_benchmark": True,
        "horizons": args.horizons,
        "root_summary": root_summary,
        "system_rows": system_rows,
    }
    out_json = out_dir / "paper_benchmark_summary.json"
    out_md = out_dir / "paper_benchmark_summary.md"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Paper Benchmark Summary",
        "",
        "These are the canonical experiments intended for the research paper.",
        "",
    ]
    for horizon in args.horizons:
        lines.append(f"## H{horizon} Cross-System Medians")
        lines.append("")
        lines.append(
            f"| root | median every-step | median every-step/dim | median best-periodic | median best-periodic/dim | catastrophic systems |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for root, stats in sorted(root_summary.items()):
            lines.append(
                f"| {root} | "
                f"{_fmt(_safe_float(stats.get(f'h{horizon}_every_step_mean_median_across_systems')))} | "
                f"{_fmt(_safe_float(stats.get(f'h{horizon}_every_step_per_dim_mean_median_across_systems')))} | "
                f"{_fmt(_safe_float(stats.get(f'h{horizon}_best_periodic_mean_median_across_systems')))} | "
                f"{_fmt(_safe_float(stats.get(f'h{horizon}_best_periodic_per_dim_mean_median_across_systems')))} | "
                f"{stats.get(f'h{horizon}_catastrophic_systems', 0)} |"
            )
        lines.append("")

    lines.append("## H1000 System Medians")
    lines.append("")
    lines.append("| root | system | H1000 every-step | H1000 every-step/dim | H1000 best-periodic | H1000 best-periodic/dim | best mode |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for root, rows_for_root in sorted(system_rows.items()):
        rows_sorted = sorted(
            rows_for_root,
            key=lambda row: (_safe_float(row.get("h1000_best_periodic_mean_median")) is None, _safe_float(row.get("h1000_best_periodic_mean_median")) or float("inf")),
        )
        for row in rows_sorted:
            lines.append(
                f"| {root} | {row['system_key']} | "
                f"{_fmt(_safe_float(row.get('h1000_every_step_mean_median')))} | "
                f"{_fmt(_safe_float(row.get('h1000_every_step_per_dim_mean_median')))} | "
                f"{_fmt(_safe_float(row.get('h1000_best_periodic_mean_median')))} | "
                f"{_fmt(_safe_float(row.get('h1000_best_periodic_per_dim_mean_median')))} | "
                f"{row.get('h1000_best_periodic_mode_consensus') or 'N/A'} |"
            )
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
