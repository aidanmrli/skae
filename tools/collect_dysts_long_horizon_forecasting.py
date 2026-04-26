#!/usr/bin/env python3
"""Collect long-horizon Dysts reevaluation metrics from a task manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_HORIZONS: Sequence[int] = (5000, 10000, 20000, 30000)


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
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


def _read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_tasks(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _horizon_mean(system_data: Dict, mode_name: str, horizon: int) -> Optional[float]:
    return _safe_float(
        system_data.get("modes", {})
        .get(mode_name, {})
        .get("horizons", {})
        .get(str(horizon), {})
        .get("mean")
    )


def _extract_row(
    task_row: Dict[str, str],
    *,
    horizons: Sequence[int],
    output_tag: str,
    checkpoint_name: str,
) -> Dict[str, object]:
    run_dir = Path(task_row["run_dir"])
    eval_json = run_dir / f"reeval_{output_tag}" / f"evaluation_results_{checkpoint_name}.json"
    row: Dict[str, object] = {
        "task_id": int(task_row["task_id"]),
        "root_label": task_row["root_label"],
        "root_display_name": task_row["root_display_name"],
        "model_family": task_row["model_family"],
        "system_key": task_row["system_key"],
        "system_slug": task_row["system_slug"],
        "seed": int(task_row["seed"]),
        "run_dir": str(run_dir),
        "reeval_results_json": str(eval_json),
        "status": "missing",
        "selected_rollout_artifacts": "",
    }
    for horizon in horizons:
        row[f"h{horizon}_no_reencode_mean"] = None
        row[f"h{horizon}_every_step_mean"] = None
        row[f"h{horizon}_best_periodic_mean"] = None
        row[f"h{horizon}_best_periodic_mode"] = None
        row[f"h{horizon}_best_reset_mean"] = None
        row[f"h{horizon}_best_reset_mode"] = None

    if not eval_json.exists():
        return row

    payload = _read_json(eval_json)
    if not isinstance(payload, dict):
        row["status"] = "invalid_json"
        return row

    system_data = payload.get(task_row["system_key"])
    if not isinstance(system_data, dict):
        row["status"] = "missing_system"
        return row

    files = system_data.get("files", {})
    selected_path = files.get("selected_rollout_artifacts")
    if isinstance(selected_path, str):
        row["selected_rollout_artifacts"] = selected_path

    complete = True
    for horizon in horizons:
        horizon_key = str(horizon)
        row[f"h{horizon}_no_reencode_mean"] = _horizon_mean(system_data, "no_reencode", horizon)
        row[f"h{horizon}_every_step_mean"] = _horizon_mean(system_data, "every_step", horizon)

        best_periodic = system_data.get("best_periodic", {}).get(horizon_key, {})
        row[f"h{horizon}_best_periodic_mean"] = _safe_float(best_periodic.get("mean"))
        row[f"h{horizon}_best_periodic_mode"] = best_periodic.get("mode")

        best_reset = system_data.get("best_reset", {}).get(horizon_key, {})
        row[f"h{horizon}_best_reset_mean"] = _safe_float(best_reset.get("mean"))
        row[f"h{horizon}_best_reset_mode"] = best_reset.get("mode")

        if row[f"h{horizon}_best_periodic_mean"] is None:
            complete = False

    row["status"] = "complete" if complete else "partial"
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _per_root_summary(rows: Iterable[Dict[str, object]], horizons: Sequence[int]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["root_label"])].append(row)

    summary: Dict[str, Dict[str, object]] = {}
    for root_label, group in grouped.items():
        complete_rows = [row for row in group if row["status"] == "complete"]
        payload: Dict[str, object] = {
            "n_tasks": len(group),
            "n_complete": len(complete_rows),
            "n_pending": len(group) - len(complete_rows),
        }
        for horizon in horizons:
            vals = [
                _safe_float(row.get(f"h{horizon}_best_periodic_mean"))
                for row in complete_rows
            ]
            vals = [value for value in vals if value is not None]
            payload[f"h{horizon}_median_best_periodic_mean"] = median(vals) if vals else None
        summary[root_label] = payload
    return summary


def _write_markdown(
    path: Path,
    *,
    rows: Sequence[Dict[str, object]],
    summary: Dict[str, Dict[str, object]],
    horizons: Sequence[int],
) -> None:
    complete = [row for row in rows if row["status"] == "complete"]
    pending = [row for row in rows if row["status"] != "complete"]
    lines = [
        "# Dysts Long-Horizon Reevaluation Summary",
        "",
        f"- Tasks: {len(rows)}",
        f"- Complete: {len(complete)}",
        f"- Pending/invalid: {len(pending)}",
        f"- Horizons: {', '.join(f'H{int(h)}' for h in horizons)}",
        "",
        "## Root Summary",
        "",
    ]
    for root_label, payload in sorted(summary.items()):
        lines.append(
            f"- `{root_label}`: complete `{payload['n_complete']}/{payload['n_tasks']}`"
        )
        for horizon in horizons:
            value = payload.get(f"h{horizon}_median_best_periodic_mean")
            if value is None:
                continue
            lines.append(
                f"  median best-periodic MSE at `H{int(horizon)}`: `{value:.6g}`"
            )
    if pending:
        lines.extend(
            [
                "",
                "## Pending Or Invalid Tasks",
                "",
            ]
        )
        for row in pending[:50]:
            lines.append(
                f"- `{row['root_label']}` / `{row['system_key']}` / seed `{row['seed']}`: `{row['status']}`"
            )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect long-horizon Dysts reevaluation metrics.")
    parser.add_argument("--task-tsv", required=True, help="Task TSV produced by build_dysts_long_horizon_eval_tasks.py")
    parser.add_argument("--out-dir", required=True, help="Collector output directory")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Horizons to summarize",
    )
    parser.add_argument(
        "--output-tag",
        default="dysts_long_horizon_h5000_h10000_h20000_h30000",
        help="Reevaluation output tag",
    )
    parser.add_argument("--checkpoint-name", default="checkpoint", help="Checkpoint stem that was reevaluated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_rows = _read_tasks(Path(args.task_tsv))
    horizons = tuple(sorted({int(h) for h in args.horizons}))
    rows = [
        _extract_row(
            task_row,
            horizons=horizons,
            output_tag=str(args.output_tag),
            checkpoint_name=str(args.checkpoint_name),
        )
        for task_row in task_rows
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = out_dir / "forecasting_rows.csv"
    pending_csv = out_dir / "pending_rows.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    _write_csv(rows_csv, rows)
    _write_csv(pending_csv, [row for row in rows if row["status"] != "complete"])
    summary = {
        "n_tasks": len(rows),
        "n_complete": sum(1 for row in rows if row["status"] == "complete"),
        "n_pending": sum(1 for row in rows if row["status"] != "complete"),
        "horizons": list(horizons),
        "per_root": _per_root_summary(rows, horizons),
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    _write_markdown(summary_md, rows=rows, summary=summary["per_root"], horizons=horizons)
    print(f"Wrote long-horizon Dysts collector outputs to {out_dir}")


if __name__ == "__main__":
    main()
