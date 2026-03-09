#!/usr/bin/env python3
"""Resolve per-system dt choices for the paper benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

from skae.benchmarks.paper_benchmark_manifest import (
    PAPER_BENCHMARK_DT_HALVING_FACTORS,
    paper_benchmark_systems,
    resolve_system_default_dt,
)


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


def _read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _run_sort_key(row: Dict[str, object]) -> Tuple[str, str]:
    return (str(row.get("run_id", "")), str(row.get("run_dir", "")))


def _seed_name(row: Dict[str, object]) -> str:
    seed_name = str(row.get("seed_name", "")).strip()
    if seed_name and seed_name != "None":
        return seed_name
    seed_value = _safe_float(row.get("seed"))
    if seed_value is None:
        return "seed_0"
    return f"seed_{int(round(seed_value))}"


def _dt_key(value: float) -> float:
    return round(float(value), 12)


def _latest_rows_by_system_dt_seed(rows: List[Dict[str, object]]) -> Dict[Tuple[str, float, str], Dict[str, object]]:
    latest: Dict[Tuple[str, float, str], Dict[str, object]] = {}
    for row in rows:
        system_key = str(row.get("system_key", "")).strip()
        env_dt = _safe_float(row.get("env_dt"))
        if not system_key or env_dt is None:
            continue
        key = (system_key, _dt_key(env_dt), _seed_name(row))
        prev = latest.get(key)
        if prev is None or _run_sort_key(row) > _run_sort_key(prev):
            latest[key] = row
    return latest


def _median_metric(rows: List[Dict[str, object]], column: str) -> Optional[float]:
    values = [_safe_float(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    return median(values) if values else None


def _write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve per-system dt for the paper benchmark.")
    parser.add_argument("--rows_csv", required=True, help="Forecasting rows CSV from collect_forecasting_roots.py")
    parser.add_argument("--output_dir", required=True, help="Directory for resolution artifacts")
    parser.add_argument("--threshold", type=float, default=1.0, help="Acceptable H1000 every-step per-dim median")
    parser.add_argument("--current_pass", type=int, default=0, help="Current dt rescue pass index")
    parser.add_argument("--max_halvings", type=int, default=2, help="Maximum number of dt halvings to allow")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(Path(args.rows_csv))
    latest = _latest_rows_by_system_dt_seed(rows)
    by_system_dt: Dict[Tuple[str, float], List[Dict[str, object]]] = defaultdict(list)
    for (system_key, env_dt, _seed_name), row in latest.items():
        by_system_dt[(system_key, env_dt)].append(row)

    selected_rows: List[Dict[str, object]] = []
    request_rows: List[Dict[str, object]] = []
    report: Dict[str, object] = {
        "threshold": args.threshold,
        "current_pass": args.current_pass,
        "max_halvings": args.max_halvings,
        "systems": [],
    }

    for system_spec in paper_benchmark_systems():
        default_dt = resolve_system_default_dt(system_spec.system_key)
        schedule = [default_dt * factor for factor in PAPER_BENCHMARK_DT_HALVING_FACTORS[: args.max_halvings + 1]]
        dt_rows: List[Dict[str, object]] = []
        selected_dt = None
        selected_status = None
        next_request_dt = None

        for idx, dt in enumerate(schedule):
            rows_for_dt = by_system_dt.get((system_spec.system_key, _dt_key(dt)), [])
            median_every = _median_metric(rows_for_dt, "h1000_every_step_per_dim_mean")
            median_best = _median_metric(rows_for_dt, "h1000_best_periodic_per_dim_mean")
            row_summary = {
                "dt": dt,
                "seed_count": len(rows_for_dt),
                "median_h1000_every_step_per_dim": median_every,
                "median_h1000_best_periodic_per_dim": median_best,
                "accepted": bool(
                    len(rows_for_dt) >= 3
                    and median_every is not None
                    and median_every < args.threshold
                ),
            }
            dt_rows.append(row_summary)

            if row_summary["accepted"] and selected_dt is None:
                selected_dt = dt
                selected_status = "accepted_default" if idx == 0 else f"accepted_halving_{idx}"

        if selected_dt is None:
            for idx, dt in enumerate(schedule):
                rows_for_dt = by_system_dt.get((system_spec.system_key, _dt_key(dt)), [])
                if len(rows_for_dt) < 3:
                    if idx <= args.max_halvings:
                        next_request_dt = dt
                    break

        if selected_dt is None and next_request_dt is None:
            available = [entry for entry in dt_rows if entry["seed_count"] >= 3]
            if available:
                selected_dt = float(available[-1]["dt"])
            else:
                selected_dt = float(schedule[min(args.current_pass, len(schedule) - 1)])
            selected_status = "integration_hard"

        if next_request_dt is not None:
            request_rows.append(
                {
                    "system_key": system_spec.system_key,
                    "env_name": system_spec.env_name,
                    "requested_dt": next_request_dt,
                    "pass_index": args.current_pass + 1,
                }
            )

        selected_rows.append(
            {
                "system_key": system_spec.system_key,
                "env_name": system_spec.env_name,
                "selected_dt": selected_dt,
                "status": selected_status or f"pending_halving_{args.current_pass + 1}",
                "default_dt": default_dt,
                "current_pass": args.current_pass,
            }
        )
        report["systems"].append(
            {
                "system_key": system_spec.system_key,
                "env_name": system_spec.env_name,
                "default_dt": default_dt,
                "schedule": schedule,
                "dt_rows": dt_rows,
                "selected_dt": selected_dt,
                "selected_status": selected_status,
                "next_request_dt": next_request_dt,
            }
        )

    selected_tsv = output_dir / "selected_dt.tsv"
    request_tsv = output_dir / f"dt_rescue_request_pass{args.current_pass + 1}.tsv"
    report_json = output_dir / "dt_resolution.json"
    report_md = output_dir / "dt_resolution.md"

    _write_tsv(selected_tsv, selected_rows)
    _write_tsv(request_tsv, request_rows)
    report_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Paper Benchmark DT Resolution",
        "",
        "- This resolution is for the research-paper benchmark runs.",
        f"- Acceptance gate: median H1000 every-step per-dim < {args.threshold}",
        f"- Current pass: {args.current_pass}",
        "",
        "| system | default dt | selected dt | status | next request |",
        "|---|---:|---:|---|---:|",
    ]
    for item in report["systems"]:
        next_request_str = ""
        if item["next_request_dt"] is not None:
            next_request_str = f"{float(item['next_request_dt']):.8g}"
        selected_dt_str = ""
        if item["selected_dt"] is not None:
            selected_dt_str = f"{float(item['selected_dt']):.8g}"
        lines.append(
            f"| {item['system_key']} | {item['default_dt']:.8g} | "
            f"{selected_dt_str} | {item['selected_status'] or f'pending_halving_{args.current_pass + 1}'} | "
            f"{next_request_str} |"
        )
    report_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote: {selected_tsv}")
    print(f"Wrote: {request_tsv}")
    print(f"Wrote: {report_json}")
    print(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()
