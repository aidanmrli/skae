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

import numpy as np


DEFAULT_HORIZONS: Sequence[int] = (100, 500, 1000, 1500, 2000, 3000, 4000, 5000)


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
    return _horizon_metric(system_data, mode_name, horizon, "mean")


def _horizon_metric(
    system_data: Dict,
    mode_name: str,
    horizon: int,
    key: str,
) -> Optional[float]:
    return _safe_float(
        system_data.get("modes", {})
        .get(mode_name, {})
        .get("horizons", {})
        .get(str(horizon), {})
        .get(key)
    )


def _best_summary_metric(summary: Dict, key: str) -> Optional[float]:
    return _safe_float(summary.get(key))


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
        row[f"h{horizon}_no_reencode_strict_full_horizon_mean"] = None
        row[f"h{horizon}_no_reencode_strict_full_horizon_num_valid"] = None
        row[f"h{horizon}_no_reencode_full_finite_fraction"] = None
        row[f"h{horizon}_no_reencode_finite_step_fraction"] = None
        row[f"h{horizon}_every_step_mean"] = None
        row[f"h{horizon}_every_step_full_finite_fraction"] = None
        row[f"h{horizon}_every_step_finite_step_fraction"] = None
        row[f"h{horizon}_best_periodic_mean"] = None
        row[f"h{horizon}_best_periodic_mode"] = None
        row[f"h{horizon}_best_periodic_full_finite_fraction"] = None
        row[f"h{horizon}_best_periodic_finite_step_fraction"] = None
        row[f"h{horizon}_best_periodic_num_full_horizon_finite"] = None
        row[f"h{horizon}_best_periodic_median_finite_prefix_length"] = None
        row[f"h{horizon}_best_periodic_min_finite_prefix_length"] = None
        row[f"h{horizon}_best_reset_mean"] = None
        row[f"h{horizon}_best_reset_mode"] = None
        row[f"h{horizon}_best_reset_full_finite_fraction"] = None
        row[f"h{horizon}_best_reset_finite_step_fraction"] = None
        row[f"h{horizon}_best_reset_num_full_horizon_finite"] = None
        row[f"h{horizon}_best_reset_median_finite_prefix_length"] = None
        row[f"h{horizon}_best_reset_min_finite_prefix_length"] = None

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
        row[f"h{horizon}_no_reencode_strict_full_horizon_mean"] = _horizon_metric(
            system_data, "no_reencode", horizon, "strict_full_horizon_mean"
        )
        row[f"h{horizon}_no_reencode_strict_full_horizon_num_valid"] = _horizon_metric(
            system_data, "no_reencode", horizon, "strict_full_horizon_num_valid"
        )
        row[f"h{horizon}_no_reencode_full_finite_fraction"] = _horizon_metric(
            system_data, "no_reencode", horizon, "full_horizon_finite_fraction"
        )
        row[f"h{horizon}_no_reencode_finite_step_fraction"] = _horizon_metric(
            system_data, "no_reencode", horizon, "finite_step_fraction"
        )
        row[f"h{horizon}_every_step_mean"] = _horizon_mean(system_data, "every_step", horizon)
        row[f"h{horizon}_every_step_full_finite_fraction"] = _horizon_metric(
            system_data, "every_step", horizon, "full_horizon_finite_fraction"
        )
        row[f"h{horizon}_every_step_finite_step_fraction"] = _horizon_metric(
            system_data, "every_step", horizon, "finite_step_fraction"
        )

        best_periodic = system_data.get("best_periodic", {}).get(horizon_key, {})
        row[f"h{horizon}_best_periodic_mean"] = _safe_float(best_periodic.get("mean"))
        row[f"h{horizon}_best_periodic_mode"] = best_periodic.get("mode")
        row[f"h{horizon}_best_periodic_full_finite_fraction"] = _best_summary_metric(
            best_periodic, "full_horizon_finite_fraction"
        )
        row[f"h{horizon}_best_periodic_finite_step_fraction"] = _best_summary_metric(
            best_periodic, "finite_step_fraction"
        )
        row[f"h{horizon}_best_periodic_num_full_horizon_finite"] = _best_summary_metric(
            best_periodic, "num_full_horizon_finite"
        )
        row[f"h{horizon}_best_periodic_median_finite_prefix_length"] = _best_summary_metric(
            best_periodic, "median_finite_prefix_length"
        )
        row[f"h{horizon}_best_periodic_min_finite_prefix_length"] = _best_summary_metric(
            best_periodic, "min_finite_prefix_length"
        )

        best_reset = system_data.get("best_reset", {}).get(horizon_key, {})
        row[f"h{horizon}_best_reset_mean"] = _safe_float(best_reset.get("mean"))
        row[f"h{horizon}_best_reset_mode"] = best_reset.get("mode")
        row[f"h{horizon}_best_reset_full_finite_fraction"] = _best_summary_metric(
            best_reset, "full_horizon_finite_fraction"
        )
        row[f"h{horizon}_best_reset_finite_step_fraction"] = _best_summary_metric(
            best_reset, "finite_step_fraction"
        )
        row[f"h{horizon}_best_reset_num_full_horizon_finite"] = _best_summary_metric(
            best_reset, "num_full_horizon_finite"
        )
        row[f"h{horizon}_best_reset_median_finite_prefix_length"] = _best_summary_metric(
            best_reset, "median_finite_prefix_length"
        )
        row[f"h{horizon}_best_reset_min_finite_prefix_length"] = _best_summary_metric(
            best_reset, "min_finite_prefix_length"
        )

        direct_strict = row[f"h{horizon}_no_reencode_strict_full_horizon_mean"]
        direct_coverage = row[f"h{horizon}_no_reencode_full_finite_fraction"]
        if (
            direct_strict is None
            or direct_coverage is None
            or not math.isclose(float(direct_coverage), 1.0)
        ):
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
            mse_vals = [
                _safe_float(
                    row.get(f"h{horizon}_no_reencode_strict_full_horizon_mean")
                )
                for row in complete_rows
            ]
            mse_vals = [value for value in mse_vals if value is not None]
            payload[f"h{horizon}_median_direct_strict_mean"] = median(mse_vals) if mse_vals else None
            coverage_vals = [
                _safe_float(row.get(f"h{horizon}_no_reencode_full_finite_fraction"))
                for row in complete_rows
            ]
            coverage_vals = [value for value in coverage_vals if value is not None]
            payload[f"h{horizon}_median_direct_full_finite_fraction"] = (
                median(coverage_vals) if coverage_vals else None
            )
        summary[root_label] = payload
    return summary


def _direct_system_effects(
    rows: Sequence[Dict[str, object]],
    horizons: Sequence[int],
    *,
    dense_label: str = "dense_mlp_tanh",
) -> Dict[str, Dict[str, object]]:
    """Pair seeds within systems, then aggregate over systems."""

    indexed = {
        (str(row["root_label"]), str(row["system_key"]), int(row["seed"])): row
        for row in rows
    }
    labels = sorted({str(row["root_label"]) for row in rows})
    systems = sorted({str(row["system_key"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    output: Dict[str, Dict[str, object]] = {}
    for horizon_index, horizon in enumerate(horizons):
        endpoint: Dict[str, object] = {}
        metric = f"h{horizon}_no_reencode_strict_full_horizon_mean"
        coverage_key = f"h{horizon}_no_reencode_full_finite_fraction"
        for label_index, label in enumerate(labels):
            if label == dense_label:
                continue
            system_effects: Dict[str, float] = {}
            failures = []
            for system in systems:
                paired = []
                for seed in seeds:
                    candidate = indexed.get((label, system, seed))
                    dense = indexed.get((dense_label, system, seed))
                    if candidate is None or dense is None:
                        failures.append(f"{system}/seed{seed}:missing_pair")
                        continue
                    candidate_value = _safe_float(candidate.get(metric))
                    dense_value = _safe_float(dense.get(metric))
                    candidate_coverage = _safe_float(candidate.get(coverage_key))
                    dense_coverage = _safe_float(dense.get(coverage_key))
                    if (
                        candidate_value is None
                        or dense_value is None
                        or candidate_value <= 0.0
                        or dense_value <= 0.0
                        or candidate_coverage != 1.0
                        or dense_coverage != 1.0
                    ):
                        failures.append(f"{system}/seed{seed}:nonfinite_or_incomplete")
                        continue
                    paired.append(math.log(candidate_value / dense_value))
                if len(paired) == len(seeds) and len(seeds) == 15:
                    system_effects[system] = float(sum(paired) / len(paired))
            record: Dict[str, object] = {
                "status": "available" if len(system_effects) == 10 and not failures else "unavailable",
                "metric": "mean paired-seed log(direct strict MSE candidate/dense) within system",
                "n_expected_seeds_per_system": 15,
                "n_systems": len(system_effects),
                "system_effects": system_effects,
                "failures": failures,
            }
            if record["status"] == "available":
                values = np.asarray(list(system_effects.values()), dtype=np.float64)
                rng = np.random.default_rng(
                    20260722 + 1009 * horizon_index + 97 * label_index
                )
                draws = values[
                    rng.integers(0, len(values), size=(100_000, len(values)))
                ].mean(axis=1)
                mean_effect = float(values.mean())
                wins = int((values < 0.0).sum())
                record.update(
                    {
                        "mean_system_log_ratio": mean_effect,
                        "geometric_mean_mse_ratio": math.exp(mean_effect),
                        "bootstrap_95_interval_log_ratio": [
                            float(np.quantile(draws, 0.025)),
                            float(np.quantile(draws, 0.975)),
                        ],
                        "system_wins": wins,
                        "one_sided_exact_sign_p": sum(
                            math.comb(len(values), k)
                            for k in range(wins, len(values) + 1)
                        )
                        / (2 ** len(values)),
                        "inference_unit": "system",
                    }
                )
            endpoint[label] = record
        output[str(horizon)] = endpoint
    return output


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
            value = payload.get(f"h{horizon}_median_direct_strict_mean")
            coverage = payload.get(f"h{horizon}_median_direct_full_finite_fraction")
            if value is None:
                continue
            suffix = ""
            if coverage is not None:
                suffix = f", median full-finite coverage: `{coverage:.3g}`"
            lines.append(
                f"  median direct strict MSE at `H{int(horizon)}`: `{value:.6g}`{suffix}"
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
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write diagnostics, then fail if any requested row is incomplete.",
    )
    parser.add_argument(
        "--expected-task-count",
        type=int,
        default=None,
        help="Expected number of task rows for a sealed packet.",
    )
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
        "direct_system_level_effects_vs_dense": _direct_system_effects(
            rows, horizons
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    _write_markdown(summary_md, rows=rows, summary=summary["per_root"], horizons=horizons)
    print(f"Wrote long-horizon Dysts collector outputs to {out_dir}")

    if args.require_complete:
        expected = (
            len(rows)
            if args.expected_task_count is None
            else int(args.expected_task_count)
        )
        if len(rows) != expected or summary["n_pending"] != 0:
            raise RuntimeError(
                "Incomplete Dysts evaluation packet after writing diagnostics: "
                f"tasks={len(rows)}/{expected}, pending={summary['n_pending']}."
            )


if __name__ == "__main__":
    main()
