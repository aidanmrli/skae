#!/usr/bin/env python3
"""Resolve per-arm dt rescue requests for the transition-rich LISTA sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    TRANSITION_RICH_BASIN_PARTITION_H1000_THRESHOLD,
    TRANSITION_RICH_BASIN_PARTITION_MAX_HALVINGS,
    transition_rich_basin_partition_models,
    transition_rich_basin_partition_systems,
    transition_rich_dt_halving_schedule,
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


def _seed_name(row: Dict[str, object]) -> str:
    seed_name = str(row.get("seed_name", "")).strip()
    if seed_name and seed_name != "None":
        return seed_name
    seed_value = _safe_float(row.get("seed"))
    if seed_value is None:
        return "seed_0"
    return f"seed_{int(round(seed_value))}"


def _run_sort_key(row: Dict[str, object]) -> Tuple[str, str]:
    return (str(row.get("run_id", "")), str(row.get("run_dir", "")))


def _dt_key(value: float) -> float:
    return round(float(value), 12)


def _latest_rows_by_arm_dt_seed(
    rows: List[Dict[str, object]],
) -> Dict[Tuple[str, str, float, str], Dict[str, object]]:
    latest: Dict[Tuple[str, str, float, str], Dict[str, object]] = {}
    for row in rows:
        model_variant = str(row.get("root_label", "")).strip()
        system_key = str(row.get("system_key", "")).strip()
        env_dt = _safe_float(row.get("env_dt"))
        if not model_variant or not system_key or env_dt is None:
            continue
        key = (model_variant, system_key, _dt_key(env_dt), _seed_name(row))
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


def resolve_rows(
    rows: List[Dict[str, object]],
    *,
    threshold: float,
    current_pass: int,
    max_halvings: int,
    min_seeds: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    latest = _latest_rows_by_arm_dt_seed(rows)
    by_arm_dt: Dict[Tuple[str, str, float], List[Dict[str, object]]] = defaultdict(list)
    for (model_variant, system_key, env_dt, _seed_name), row in latest.items():
        by_arm_dt[(model_variant, system_key, env_dt)].append(row)

    selected_rows: List[Dict[str, object]] = []
    request_rows: List[Dict[str, object]] = []
    report: Dict[str, object] = {
        "threshold": threshold,
        "current_pass": current_pass,
        "max_halvings": max_halvings,
        "min_seeds": min_seeds,
        "arms": [],
    }

    model_specs = {
        spec.variant: spec
        for spec in transition_rich_basin_partition_models()
    }

    for model_spec in transition_rich_basin_partition_models():
        for system_spec in transition_rich_basin_partition_systems():
            schedule = transition_rich_dt_halving_schedule(
                system_spec.system_key,
                max_halvings=max_halvings,
            )
            dt_rows: List[Dict[str, object]] = []
            selected_dt = None
            selected_status = None
            next_request_dt = None

            for pass_index, dt in enumerate(schedule):
                rows_for_dt = by_arm_dt.get(
                    (model_spec.variant, system_spec.system_key, _dt_key(dt)),
                    [],
                )
                median_h1000 = _median_metric(rows_for_dt, "h1000_best_periodic_mean")
                row_summary = {
                    "dt": dt,
                    "seed_count": len(rows_for_dt),
                    "median_h1000_best_periodic_mean": median_h1000,
                    "accepted": bool(
                        len(rows_for_dt) >= min_seeds
                        and median_h1000 is not None
                        and median_h1000 < threshold
                    ),
                }
                dt_rows.append(row_summary)
                if row_summary["accepted"] and selected_dt is None:
                    selected_dt = dt
                    selected_status = "accepted_default" if pass_index == 0 else f"accepted_halving_{pass_index}"

            if selected_dt is None:
                for dt in schedule:
                    rows_for_dt = by_arm_dt.get(
                        (model_spec.variant, system_spec.system_key, _dt_key(dt)),
                        [],
                    )
                    if len(rows_for_dt) < min_seeds:
                        next_request_dt = dt
                        break

            if selected_dt is None and next_request_dt is None:
                available = [entry for entry in dt_rows if entry["seed_count"] >= min_seeds]
                if available:
                    selected_dt = float(available[-1]["dt"])
                    selected_status = "max_halvings_exhausted"
                else:
                    selected_dt = float(schedule[min(current_pass, len(schedule) - 1)])
                    selected_status = f"pending_halving_{current_pass + 1}"

            if next_request_dt is not None:
                request_rows.append(
                    {
                        "model_variant": model_spec.variant,
                        "system_key": system_spec.system_key,
                        "env_name": system_spec.env_name,
                        "requested_dt": next_request_dt,
                        "pass_index": current_pass + 1,
                        "default_dt": schedule[0],
                    }
                )

            selected_rows.append(
                {
                    "model_variant": model_spec.variant,
                    "system_key": system_spec.system_key,
                    "env_name": system_spec.env_name,
                    "default_dt": schedule[0],
                    "selected_dt": selected_dt,
                    "status": selected_status or f"pending_halving_{current_pass + 1}",
                    "current_pass": current_pass,
                }
            )
            report["arms"].append(
                {
                    "model_variant": model_spec.variant,
                    "system_key": system_spec.system_key,
                    "env_name": system_spec.env_name,
                    "default_dt": schedule[0],
                    "schedule": schedule,
                    "dt_rows": dt_rows,
                    "selected_dt": selected_dt,
                    "selected_status": selected_status,
                    "next_request_dt": next_request_dt,
                    "k_structure": model_specs[model_spec.variant].k_structure,
                }
            )

    return selected_rows, request_rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve dt rescue requests for transition-rich basin-partition arms."
    )
    parser.add_argument("--rows_csv", required=True, help="forecasting_rows.csv from collection.")
    parser.add_argument("--output_dir", required=True, help="Directory for resolution artifacts.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=TRANSITION_RICH_BASIN_PARTITION_H1000_THRESHOLD,
        help="Acceptable H1000 best-periodic mean threshold.",
    )
    parser.add_argument("--current_pass", type=int, default=0, help="Current dt rescue pass index.")
    parser.add_argument(
        "--max_halvings",
        type=int,
        default=TRANSITION_RICH_BASIN_PARTITION_MAX_HALVINGS,
        help="Maximum number of dt halvings to allow.",
    )
    parser.add_argument(
        "--min_seeds",
        type=int,
        default=1,
        help="Minimum number of seeds required before judging a dt arm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(Path(args.rows_csv))
    selected_rows, request_rows, report = resolve_rows(
        rows,
        threshold=args.threshold,
        current_pass=args.current_pass,
        max_halvings=args.max_halvings,
        min_seeds=args.min_seeds,
    )

    selected_tsv = output_dir / "selected_dt.tsv"
    request_tsv = output_dir / f"dt_rescue_request_pass{args.current_pass + 1}.tsv"
    report_json = output_dir / "dt_resolution.json"
    report_md = output_dir / "dt_resolution.md"

    _write_tsv(selected_tsv, selected_rows)
    _write_tsv(request_tsv, request_rows)
    report_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Transition-Rich DT Resolution",
        "",
        f"- Acceptance gate: `H1000 best-periodic mean < {args.threshold}`",
        f"- Minimum seeds per arm: `{args.min_seeds}`",
        f"- Current pass: `{args.current_pass}`",
        "",
        "| model | system | default dt | selected dt | status | next request |",
        "|---|---|---:|---:|---|---:|",
    ]
    for item in report["arms"]:
        next_request = item["next_request_dt"]
        selected_dt = item["selected_dt"]
        lines.append(
            f"| {item['model_variant']} | {item['system_key']} | {item['default_dt']:.8g} | "
            f"{'' if selected_dt is None else f'{float(selected_dt):.8g}'} | "
            f"{item['selected_status'] or f'pending_halving_{args.current_pass + 1}'} | "
            f"{'' if next_request is None else f'{float(next_request):.8g}'} |"
        )
    report_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote: {selected_tsv}")
    print(f"Wrote: {request_tsv}")
    print(f"Wrote: {report_json}")
    print(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()
