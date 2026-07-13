#!/usr/bin/env python3
"""Summarize support-family hyperparameter sweeps for local Koopman routing."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=25)
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Iterable[object]) -> Optional[float]:
    clean = [number for number in (_float(value) for value in values) if number is not None]
    return float(sum(clean) / len(clean)) if clean else None


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6g}"


def _fmt_md(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4g}"


def _first(rows: Sequence[Dict[str, str]], key: str, default: str = "") -> str:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _unassigned_total(rows: Sequence[Dict[str, str]]) -> int:
    total = 0
    for row in rows:
        raw = row.get("assignment_mode", "")
        marker = "jaccard_unassigned="
        if marker not in raw:
            continue
        try:
            total += int(float(raw.split(marker, 1)[1].split()[0]))
        except (TypeError, ValueError, IndexError):
            continue
    return total


def _summarize_combo(row_path: Path, sweep_dir: Path) -> Dict[str, object]:
    rows = _read_csv(row_path)
    manifest = _read_json(row_path.parent / "manifest.json")
    failures = _read_json(row_path.parent / "failures.json")
    learned = [row for row in rows if row.get("route_kind") == "learned_support_family"]
    oracle = [row for row in rows if row.get("route_kind") == "oracle_basin"]
    global_rows = [row for row in rows if row.get("route_kind") == "global_k"]
    reference_rows = learned or oracle or global_rows or rows

    learned_ratio = _mean(row.get("partition_over_global_on_covered") for row in learned)
    oracle_ratio = _mean(row.get("partition_over_global_on_covered") for row in oracle)
    learned_over_oracle = _ratio(learned_ratio, oracle_ratio)
    oracle_gap = None
    if learned_ratio is not None and oracle_ratio is not None:
        oracle_gap = learned_ratio - oracle_ratio

    failure_count = 0
    if isinstance(failures, list):
        failure_count = len(failures)
    elif isinstance(failures, dict):
        failure_count = int(_float(failures.get("num_failures")) or 0)
    else:
        failure_count = int(_float(manifest.get("num_failures")) or 0)

    return {
        "combo_id": str(row_path.parent.relative_to(sweep_dir)),
        "output_dir": str(row_path.parent),
        "status": str(manifest.get("status", "")),
        "completed_runs": int(_float(manifest.get("completed_runs")) or len(global_rows)),
        "num_runs": int(_float(manifest.get("num_runs")) or len(global_rows)),
        "num_failures": failure_count,
        "num_rows": len(rows),
        "support_definition": _first(reference_rows, "support_definition"),
        "family_jaccard_threshold": _first(reference_rows, "family_jaccard_threshold"),
        "min_operator_transitions": _first(reference_rows, "min_operator_transitions"),
        "ridge_lambda": _first(reference_rows, "ridge_lambda"),
        "train_fraction": _first(reference_rows, "train_fraction"),
        "learned_latent_over_global": learned_ratio,
        "oracle_latent_over_global": oracle_ratio,
        "learned_minus_oracle": oracle_gap,
        "learned_over_oracle": learned_over_oracle,
        "learned_coverage": _mean(row.get("test_coverage_fraction") for row in learned),
        "learned_basin_ari": _mean(row.get("basin_ari") for row in learned),
        "learned_basin_nmi": _mean(row.get("basin_nmi") for row in learned),
        "learned_basin_purity": _mean(row.get("basin_purity") for row in learned),
        "learned_class_count_total": _mean(row.get("class_count_total") for row in learned),
        "learned_class_count_fit": _mean(row.get("class_count_fit") for row in learned),
        "learned_unassigned_total": _unassigned_total(learned),
        "oracle_coverage": _mean(row.get("test_coverage_fraction") for row in oracle),
        "oracle_basin_ari": _mean(row.get("basin_ari") for row in oracle),
        "oracle_basin_nmi": _mean(row.get("basin_nmi") for row in oracle),
        "oracle_basin_purity": _mean(row.get("basin_purity") for row in oracle),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def _write_markdown(path: Path, rows: Sequence[Dict[str, object]], *, sweep_dir: Path, top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def is_complete(row: Dict[str, object]) -> bool:
        completed = int(row.get("completed_runs") or 0)
        total = int(row.get("num_runs") or 0)
        return str(row.get("status", "")) == "complete" and total > 0 and completed >= total

    complete = [
        row
        for row in rows
        if is_complete(row)
        and row.get("learned_latent_over_global") is not None
        and int(row.get("num_failures") or 0) == 0
    ]
    partial = [
        row
        for row in rows
        if not is_complete(row)
        and row.get("learned_latent_over_global") is not None
        and int(row.get("num_failures") or 0) == 0
    ]
    best = complete[0] if complete else None
    baseline = None
    for row in complete:
        if (
            row.get("support_definition") == "topk:8"
            and _float(row.get("family_jaccard_threshold")) == 0.5
            and _float(row.get("min_operator_transitions")) == 128.0
        ):
            baseline = row
            break

    lines = [
        "# Support-Family Local-K Hyperparameter Summary",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"Sweep directory: `{sweep_dir}`",
        "",
        "Sorted by learned support-family one-step latent MSE divided by the checkpoint global-K latent MSE. Lower is better; oracle basin is an evaluation-only upper-bound route.",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best Complete Setting",
                "",
                (
                    f"- `{best['support_definition']}`, J=`{best['family_jaccard_threshold']}`, "
                    f"min transitions=`{best['min_operator_transitions']}`, ridge=`{best['ridge_lambda']}`: "
                    f"learned/global `{_fmt_md(best.get('learned_latent_over_global'))}`, "
                    f"oracle/global `{_fmt_md(best.get('oracle_latent_over_global'))}`, "
                    f"learned/oracle `{_fmt_md(best.get('learned_over_oracle'))}`, "
                    f"coverage `{_fmt_md(best.get('learned_coverage'))}`."
                ),
                "",
            ]
        )
    if baseline is not None:
        lines.extend(
            [
                "## Current Baseline Setting",
                "",
                (
                    f"- `topk:8`, J=`0.5`, min transitions=`128`: "
                    f"learned/global `{_fmt_md(baseline.get('learned_latent_over_global'))}`, "
                    f"oracle/global `{_fmt_md(baseline.get('oracle_latent_over_global'))}`, "
                    f"learned/oracle `{_fmt_md(baseline.get('learned_over_oracle'))}`, "
                    f"coverage `{_fmt_md(baseline.get('learned_coverage'))}`."
                ),
                "",
            ]
        )
    if not complete and partial:
        lines.extend(
            [
                "## Partial Diagnostic",
                "",
                "No full hyperparameter setting is complete yet. The table below is based on partial worker outputs and must not be treated as a final result.",
                "",
            ]
        )

    lines.extend(
        [
            f"## Top {min(top_k, len(complete or partial))} Settings",
            "",
            "| rank | support | J | min trans | ridge | train frac | learned/global | oracle/global | learned/oracle | gap | coverage | ARI | NMI | purity | fit/total families | failures |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    display_rows = complete if complete else partial
    for rank, row in enumerate(display_rows[:top_k], start=1):
        lines.append(
            f"| {rank} | `{row.get('support_definition', '')}` | "
            f"{row.get('family_jaccard_threshold', '')} | {row.get('min_operator_transitions', '')} | "
            f"{row.get('ridge_lambda', '')} | {row.get('train_fraction', '')} | "
            f"{_fmt_md(row.get('learned_latent_over_global'))} | "
            f"{_fmt_md(row.get('oracle_latent_over_global'))} | "
            f"{_fmt_md(row.get('learned_over_oracle'))} | "
            f"{_fmt_md(row.get('learned_minus_oracle'))} | "
            f"{_fmt_md(row.get('learned_coverage'))} | "
            f"{_fmt_md(row.get('learned_basin_ari'))} | "
            f"{_fmt_md(row.get('learned_basin_nmi'))} | "
            f"{_fmt_md(row.get('learned_basin_purity'))} | "
            f"{_fmt_md(row.get('learned_class_count_fit'))}/{_fmt_md(row.get('learned_class_count_total'))} | "
            f"{row.get('num_failures', '')} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir)
    row_paths = sorted(sweep_dir.rglob("regime_discovery_local_koopman_rows.csv"))
    summaries = [_summarize_combo(path, sweep_dir) for path in row_paths]
    summaries.sort(
        key=lambda row: (
            row.get("learned_latent_over_global") is None,
            float(row.get("learned_latent_over_global") or float("inf")),
            str(row.get("support_definition", "")),
            float(_float(row.get("family_jaccard_threshold")) or 0.0),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "support_family_hparam_summary.csv", summaries)
    _write_markdown(
        output_dir / "support_family_hparam_summary.md",
        summaries,
        sweep_dir=sweep_dir,
        top_k=args.top_k,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "sweep_dir": str(sweep_dir),
                "num_combo_outputs": len(summaries),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
