#!/usr/bin/env python
"""Collect and summarize block-loss balance sweeps (top1_margin + balance).

Aggregates cosine separation and fixed-threshold metrics, grouped by
configuration (ts, balance loss, weights, margin).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_BASE_DIRS = [
    "/network/scratch/l/lia/skae/lyapunov_block_loss_balance_phase1",
    "/network/scratch/l/lia/skae/lyapunov_block_loss_balance_phase2",
]

EXP_NAME_RE = re.compile(
    r"dim(?P<dim>\d+)_nb(?P<nb>\d+)_ts(?P<ts>\d+)_blkdiag_"
    r"top1m(?P<margin>[\d.]+)_one(?P<one>[\d.]+)_bal(?P<bal>[\d.]+)_"
    r"(?P<balance>[^_]+)_seed(?P<seed>\d+)"
)


def _find_latest_timestamp_dir(exp_dir: Path) -> Optional[Path]:
    subdirs = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: d.name,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


def _load_json(path: Path) -> Optional[Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _load_cosine_metrics(exp_dir: Path, run_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    candidates = [exp_dir / "support_eval" / "cosine_metrics.json"]
    if run_dir is not None:
        candidates.append(run_dir / "support_eval" / "cosine_metrics.json")
    for path in candidates:
        data = _load_json(path)
        if data is not None:
            return data
    return None


def _load_threshold_sweep(exp_dir: Path, run_dir: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    candidates = [exp_dir / "support_eval" / "threshold_sweep.json"]
    if run_dir is not None:
        candidates.append(run_dir / "support_eval" / "threshold_sweep.json")
    for path in candidates:
        data = _load_json(path)
        if data is not None:
            return data
    return None


def _pick_threshold_entry(sweep: List[Dict[str, Any]], threshold: float) -> Optional[Dict[str, Any]]:
    for entry in sweep:
        if float(entry.get("support_threshold", -1.0)) == threshold:
            return entry
    return None


def _group_key(entry: Dict[str, Any]) -> Tuple:
    return (
        entry["target_size"],
        entry["balance"],
        entry["one_weight"],
        entry["bal_weight"],
        entry["margin"],
    )


def collect_all(base_dirs: List[str], threshold: float) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue

        for exp_dir in sorted(base_path.iterdir()):
            if not exp_dir.is_dir():
                continue

            match = EXP_NAME_RE.match(exp_dir.name)
            if not match:
                continue

            run_dir = _find_latest_timestamp_dir(exp_dir)
            cosine = _load_cosine_metrics(exp_dir, run_dir)
            sweep = _load_threshold_sweep(exp_dir, run_dir)
            if cosine is None and sweep is None:
                continue

            fixed = _pick_threshold_entry(sweep, threshold) if sweep else None

            entry: Dict[str, Any] = {
                "experiment": exp_dir.name,
                "dim": int(match.group("dim")),
                "num_basins": int(match.group("nb")),
                "target_size": int(match.group("ts")),
                "balance": match.group("balance"),
                "one_weight": float(match.group("one")),
                "bal_weight": float(match.group("bal")),
                "margin": float(match.group("margin")),
                "seed": int(match.group("seed")),
                "run_dir": str(run_dir) if run_dir is not None else "",
            }

            if cosine is not None:
                entry["cosine_sep"] = cosine.get("cosine_separation_score")
                entry["cosine_intra"] = cosine.get("mean_intra_basin_cosine")
                entry["cosine_inter"] = cosine.get("mean_inter_basin_cosine")

            if fixed is not None:
                entry["consistency"] = fixed.get("mean_basin_consistency")
                entry["uniqueness"] = fixed.get("mode_uniqueness_rate")
                entry["jaccard"] = fixed.get("mean_pairwise_jaccard")
                entry["support_size"] = fixed.get("mean_mode_support_size")

            results.append(entry)

    return results


def _mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def aggregate(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple, List[Dict[str, Any]]] = {}
    for entry in results:
        grouped.setdefault(_group_key(entry), []).append(entry)

    agg: List[Dict[str, Any]] = []
    for key, entries in grouped.items():
        ts, balance, one_w, bal_w, margin = key
        cos_sep_mean, cos_sep_std = _mean_std([e.get("cosine_sep") for e in entries])
        cons_mean, _ = _mean_std([e.get("consistency") for e in entries])
        uniq_mean, _ = _mean_std([e.get("uniqueness") for e in entries])
        jac_mean, _ = _mean_std([e.get("jaccard") for e in entries])
        support_mean, _ = _mean_std([e.get("support_size") for e in entries])
        agg.append({
            "target_size": ts,
            "balance": balance,
            "one_weight": one_w,
            "bal_weight": bal_w,
            "margin": margin,
            "num_runs": len(entries),
            "cos_sep_mean": cos_sep_mean,
            "cos_sep_std": cos_sep_std,
            "consistency_mean": cons_mean,
            "uniqueness_mean": uniq_mean,
            "jaccard_mean": jac_mean,
            "support_mean": support_mean,
        })
    return agg


def _format_float(val: Any, fmt: str = ".3f") -> str:
    if val is None:
        return "--"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return "--"


def render_summary(agg: List[Dict[str, Any]]) -> str:
    if not agg:
        return "No results to display."

    lines: List[str] = []
    lines.append("# Block Loss Balance Summary")
    lines.append("")
    lines.append("| ts | balance | one_w | bal_w | margin | cos_sep | std | cons | uniq | jaccard | support | n |")
    lines.append("|---:|:--------|------:|------:|-------:|--------:|----:|-----:|-----:|--------:|--------:|--:|")

    agg_sorted = sorted(agg, key=lambda r: (r["target_size"], -(r["cos_sep_mean"] or -1e9)))
    for entry in agg_sorted:
        lines.append(
            "| {ts} | {bal} | {one} | {bw} | {m} | {sep} | {std} | {cons} | {uniq} | {jac} | {supp} | {n} |".format(
                ts=entry["target_size"],
                bal=entry["balance"],
                one=_format_float(entry["one_weight"], ".2f"),
                bw=_format_float(entry["bal_weight"], ".2f"),
                m=_format_float(entry["margin"], ".2f"),
                sep=_format_float(entry["cos_sep_mean"]),
                std=_format_float(entry["cos_sep_std"]),
                cons=_format_float(entry["consistency_mean"]),
                uniq=_format_float(entry["uniqueness_mean"]),
                jac=_format_float(entry["jaccard_mean"]),
                supp=_format_float(entry["support_mean"], ".1f"),
                n=entry["num_runs"],
            )
        )

    lines.append("")
    lines.append("Best config per target size (by cosine separation):")
    for ts in sorted({e["target_size"] for e in agg}):
        group = [e for e in agg if e["target_size"] == ts]
        best = max(group, key=lambda r: (r["cos_sep_mean"] or -1e9))
        lines.append(
            "- ts={ts}: {bal} one_w={one} bal_w={bw} m={m} sep={sep}".format(
                ts=ts,
                bal=best["balance"],
                one=_format_float(best["one_weight"], ".2f"),
                bw=_format_float(best["bal_weight"], ".2f"),
                m=_format_float(best["margin"], ".2f"),
                sep=_format_float(best["cos_sep_mean"]),
            )
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect block loss balance sweep results")
    parser.add_argument("--base_dir", type=str, action="append", default=None,
                        help="Base directory to scan (can be provided multiple times)")
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--summary_md", type=str, default=None)
    parser.add_argument("--no_print", action="store_true", help="Do not print summary to stdout")
    args = parser.parse_args()

    base_dirs = args.base_dir if args.base_dir else DEFAULT_BASE_DIRS
    results = collect_all(base_dirs, args.threshold)
    agg = aggregate(results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"runs": results, "aggregate": agg}, f, indent=2)

    summary = render_summary(agg)

    if args.summary_md:
        with open(args.summary_md, "w") as f:
            f.write(summary + "\n")

    if not args.no_print:
        print(summary)


if __name__ == "__main__":
    main()
