#!/usr/bin/env python3
"""Summarize the final transition-rich basin-partition comparison packet.

This tool combines forecasting rows with a selected slice of the interpretability
rows so the final paper-facing `17`-system comparison can be read from one
markdown artifact instead of separate ad hoc tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FORECAST_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("h100_best_periodic_mean", "H100 best", "lower"),
    ("h500_best_periodic_mean", "H500 best", "lower"),
    ("h1000_best_periodic_mean", "H1000 best", "lower"),
)

INTERPRETABILITY_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("h_basin_given_support", "H(B|S)", "lower"),
    ("h_support_given_basin", "H(S|B)", "lower"),
    ("support_nmi", "NMI", "higher"),
    ("u_exact", "U_exact", "higher"),
    ("family_h_family_given_basin", "H(F|B)", "lower"),
    ("support_projection_self_over_base", "own/base", "lower"),
    ("support_freeze_self_over_base_h20", "freeze/base@20", "lower"),
    ("support_persistence", "persistence", "higher"),
    ("operator_between_over_within", "op between/within", "higher"),
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


def _seed_name(row: Dict[str, object]) -> str:
    seed_name = str(row.get("seed_name", "")).strip()
    if seed_name and seed_name != "None":
        return seed_name
    raw_seed = _safe_float(row.get("seed"))
    return f"seed_{int(round(raw_seed or 0))}"


def _run_sort_key(row: Dict[str, object]) -> Tuple[str, str]:
    return (str(row.get("run_id", "")), str(row.get("run_dir", "")))


def _read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _latest_forecasting_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("root_label", "")).strip(),
            str(row.get("system_key", "")).strip(),
            _seed_name(row),
        )
        prev = latest.get(key)
        if prev is None or _run_sort_key(row) > _run_sort_key(prev):
            latest[key] = row
    return sorted(latest.values(), key=lambda row: (row.get("root_label", ""), row.get("system_key", ""), _seed_name(row)))


def _latest_interpretability_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[Tuple[str, str, str, str, str], Dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("root_label", "")).strip(),
            str(row.get("system_key", "")).strip(),
            str(row.get("seed", "")).strip(),
            str(row.get("support_scheme", "")).strip(),
            str(row.get("subset", "")).strip(),
        )
        prev = latest.get(key)
        if prev is None or _run_sort_key(row) > _run_sort_key(prev):
            latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            row.get("root_label", ""),
            row.get("system_key", ""),
            row.get("seed", ""),
            row.get("support_scheme", ""),
            row.get("subset", ""),
        ),
    )


def _median_by_root_system(
    rows: Sequence[Dict[str, object]],
    *,
    root_label: str,
    value_key: str,
    extra_filters: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    extra_filters = extra_filters or {}
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("root_label", "")).strip() != root_label:
            continue
        if any(str(row.get(key, "")).strip() != expected for key, expected in extra_filters.items()):
            continue
        value = _safe_float(row.get(value_key))
        if value is None:
            continue
        grouped[str(row.get("system_key", "")).strip()].append(value)
    return {system: median(values) for system, values in grouped.items() if values}


def _root_metric_summary(
    rows: Sequence[Dict[str, object]],
    *,
    roots: Sequence[str],
    metrics: Sequence[Tuple[str, str, str]],
    extra_filters: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for root in roots:
        root_stats: Dict[str, object] = {}
        for value_key, label, _direction in metrics:
            medians = _median_by_root_system(rows, root_label=root, value_key=value_key, extra_filters=extra_filters)
            values = list(medians.values())
            root_stats[label] = median(values) if values else None
            root_stats[f"{label} system_count"] = len(values)
        summary[root] = root_stats
    return summary


def _pairwise_wins(
    rows: Sequence[Dict[str, object]],
    *,
    left_root: str,
    right_root: str,
    value_key: str,
    direction: str,
    extra_filters: Optional[Dict[str, str]] = None,
    tie_tol: float = 1e-12,
) -> Dict[str, object]:
    left = _median_by_root_system(rows, root_label=left_root, value_key=value_key, extra_filters=extra_filters)
    right = _median_by_root_system(rows, root_label=right_root, value_key=value_key, extra_filters=extra_filters)
    shared_systems = sorted(set(left) & set(right))
    left_better = 0
    right_better = 0
    ties = 0
    for system in shared_systems:
        left_value = left[system]
        right_value = right[system]
        if abs(left_value - right_value) <= tie_tol:
            ties += 1
            continue
        if direction == "lower":
            if left_value < right_value:
                left_better += 1
            else:
                right_better += 1
        else:
            if left_value > right_value:
                left_better += 1
            else:
                right_better += 1
    return {
        "left_root": left_root,
        "right_root": right_root,
        "systems_compared": len(shared_systems),
        "left_better": left_better,
        "right_better": right_better,
        "ties": ties,
    }


def build_summary(
    *,
    forecasting_rows: Sequence[Dict[str, object]],
    interpretability_rows: Sequence[Dict[str, object]],
    candidate_roots: Sequence[str],
    control_roots: Sequence[str],
    support_scheme: str,
    subset: str,
    good_threshold: float,
) -> Dict[str, object]:
    roots = list(candidate_roots) + list(control_roots)
    forecast_summary = _root_metric_summary(
        forecasting_rows,
        roots=roots,
        metrics=FORECAST_METRICS,
    )

    for root in roots:
        systems = _median_by_root_system(
            forecasting_rows,
            root_label=root,
            value_key="h1000_best_periodic_mean",
        )
        forecast_summary[root]["good systems (H1000 best)"] = sum(value < good_threshold for value in systems.values())

    interpretability_summary = _root_metric_summary(
        interpretability_rows,
        roots=roots,
        metrics=INTERPRETABILITY_METRICS,
        extra_filters={"support_scheme": support_scheme, "subset": subset},
    )

    pairwise: Dict[str, Dict[str, Dict[str, object]]] = {}
    for candidate in candidate_roots:
        candidate_vs_controls: Dict[str, Dict[str, object]] = {}
        for control in control_roots:
            metrics: Dict[str, object] = {}
            for value_key, label, direction in FORECAST_METRICS:
                metrics[label] = _pairwise_wins(
                    forecasting_rows,
                    left_root=candidate,
                    right_root=control,
                    value_key=value_key,
                    direction=direction,
                )
            for value_key, label, direction in INTERPRETABILITY_METRICS:
                metrics[label] = _pairwise_wins(
                    interpretability_rows,
                    left_root=candidate,
                    right_root=control,
                    value_key=value_key,
                    direction=direction,
                    extra_filters={"support_scheme": support_scheme, "subset": subset},
                )
            candidate_vs_controls[control] = metrics
        pairwise[candidate] = candidate_vs_controls

    return {
        "support_scheme": support_scheme,
        "subset": subset,
        "good_threshold": good_threshold,
        "candidate_roots": list(candidate_roots),
        "control_roots": list(control_roots),
        "forecast_summary": forecast_summary,
        "interpretability_summary": interpretability_summary,
        "pairwise": pairwise,
    }


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def _render_markdown(summary: Dict[str, object]) -> str:
    forecast_summary = summary["forecast_summary"]
    interpretability_summary = summary["interpretability_summary"]
    candidate_roots = summary["candidate_roots"]
    control_roots = summary["control_roots"]
    roots = list(candidate_roots) + list(control_roots)

    lines = [
        "# Transition-Rich Final Comparison",
        "",
        f"- Support slice: `{summary['support_scheme']}` / `{summary['subset']}`",
        f"- Good-forecast threshold: `H1000 best < {summary['good_threshold']}`",
        "",
        "## Root Summary",
        "",
        "| root | H100 best | H500 best | H1000 best | good systems | H(B|S) | H(S|B) | U_exact | H(F|B) | own/base | freeze/base@20 | op between/within |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for root in roots:
        fstats = forecast_summary.get(root, {})
        istats = interpretability_summary.get(root, {})
        lines.append(
            f"| {root} | "
            f"{_fmt(fstats.get('H100 best'))} | "
            f"{_fmt(fstats.get('H500 best'))} | "
            f"{_fmt(fstats.get('H1000 best'))} | "
            f"{fstats.get('good systems (H1000 best)', 0)} | "
            f"{_fmt(istats.get('H(B|S)'))} | "
            f"{_fmt(istats.get('H(S|B)'))} | "
            f"{_fmt(istats.get('U_exact'))} | "
            f"{_fmt(istats.get('H(F|B)'))} | "
            f"{_fmt(istats.get('own/base'))} | "
            f"{_fmt(istats.get('freeze/base@20'))} | "
            f"{_fmt(istats.get('op between/within'))} |"
        )

    pairwise = summary["pairwise"]
    for candidate in candidate_roots:
        for control in control_roots:
            lines.extend(
                [
                    "",
                    f"## {candidate} vs {control}",
                    "",
                    "| metric | systems compared | candidate better | control better | ties |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            metrics = pairwise[candidate][control]
            ordered_labels = [label for _key, label, _dir in FORECAST_METRICS + INTERPRETABILITY_METRICS]
            for label in ordered_labels:
                stats = metrics[label]
                lines.append(
                    f"| {label} | {stats['systems_compared']} | {stats['left_better']} | {stats['right_better']} | {stats['ties']} |"
                )

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast_rows_csv", required=True)
    parser.add_argument("--interpretability_rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--candidate_roots",
        default=(
            "lista_blockdiag_signsplit_hardinit_basin_partition,"
            "lista_dense_softblock_signsplit_p64_hardinit_basin_partition"
        ),
    )
    parser.add_argument(
        "--control_roots",
        default="mlp_sparse_basin_partition_control,mlp_zero_sparse_basin_partition_control",
    )
    parser.add_argument("--support_scheme", default="absolute:0.001")
    parser.add_argument("--subset", default="deep")
    parser.add_argument("--good_threshold", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasting_rows = _latest_forecasting_rows(_read_csv(Path(args.forecast_rows_csv)))
    interpretability_rows = _latest_interpretability_rows(_read_csv(Path(args.interpretability_rows_csv)))
    candidate_roots = [item.strip() for item in args.candidate_roots.split(",") if item.strip()]
    control_roots = [item.strip() for item in args.control_roots.split(",") if item.strip()]

    payload = build_summary(
        forecasting_rows=forecasting_rows,
        interpretability_rows=interpretability_rows,
        candidate_roots=candidate_roots,
        control_roots=control_roots,
        support_scheme=args.support_scheme,
        subset=args.subset,
        good_threshold=args.good_threshold,
    )

    out_json = output_dir / "transition_rich_final_comparison.json"
    out_md = output_dir / "transition_rich_final_comparison.md"
    out_json.write_text(json.dumps(payload, indent=2))
    out_md.write_text(_render_markdown(payload) + "\n")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
