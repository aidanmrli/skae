#!/usr/bin/env python
"""Collect and summarize block-loss ablation sweep results.

Aggregates both:
  - cosine separation (threshold-free)
  - threshold-based support metrics (consistency/uniqueness/Jaccard)

Usage:
  python tools/collect_block_loss_ablation.py
  python tools/collect_block_loss_ablation.py --base_dir /path --output_json out.json --summary_md summary.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = "/network/scratch/l/lia/skae/lyapunov_block_loss_sweep"
EXP_NAME_RE = re.compile(
    r"dim(?P<dim>\d+)_nb(?P<nb>\d+)_ts(?P<ts>\d+)_blkdiag_(?P<loss>.+)"
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


def _pick_threshold_entry(
    sweep: List[Dict[str, Any]],
    threshold: float,
) -> Optional[Dict[str, Any]]:
    for entry in sweep:
        if float(entry.get("support_threshold", -1.0)) == threshold:
            return entry
    return None


def _best_consistency_entry(
    sweep: List[Dict[str, Any]],
    require_full_unique: bool,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    if not sweep:
        return None, False

    def is_full_unique(entry: Dict[str, Any]) -> bool:
        unique = entry.get("unique_mode_supports")
        total = entry.get("num_basins")
        if unique is None or total is None:
            return False
        return int(unique) == int(total)

    full_unique_entries = [e for e in sweep if is_full_unique(e)]
    if require_full_unique and full_unique_entries:
        best = max(full_unique_entries, key=lambda e: e.get("mean_basin_consistency", 0.0))
        return best, True

    best = max(sweep, key=lambda e: e.get("mean_basin_consistency", 0.0))
    return best, is_full_unique(best)


def collect_all(base_dir: str, threshold: float, require_full_unique: bool) -> List[Dict[str, Any]]:
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory not found: {base_dir}")
        return []

    results: List[Dict[str, Any]] = []

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
            print(f"  {exp_dir.name}: no cosine or threshold sweep results found")
            continue

        fixed = _pick_threshold_entry(sweep, threshold) if sweep else None
        best, best_full_unique = _best_consistency_entry(sweep, require_full_unique) if sweep else (None, False)

        entry: Dict[str, Any] = {
            "experiment": exp_dir.name,
            "dim": int(match.group("dim")),
            "num_basins": int(match.group("nb")),
            "target_size": int(match.group("ts")),
            "loss": match.group("loss"),
            "run_dir": str(run_dir) if run_dir is not None else "",
        }

        if cosine is not None:
            entry["cosine"] = {
                "mean_intra": cosine.get("mean_intra_basin_cosine"),
                "mean_inter": cosine.get("mean_inter_basin_cosine"),
                "separation": cosine.get("cosine_separation_score"),
                "aggregation": cosine.get("cosine_aggregation"),
                "demean": cosine.get("cosine_demean"),
                "remove_pc1": cosine.get("cosine_remove_pc1"),
            }

        if fixed is not None:
            entry["fixed"] = {
                "threshold": fixed.get("support_threshold"),
                "consistency": fixed.get("mean_basin_consistency"),
                "uniqueness_rate": fixed.get("mode_uniqueness_rate"),
                "unique_mode_supports": fixed.get("unique_mode_supports"),
                "mean_support_size": fixed.get("mean_mode_support_size"),
                "mean_jaccard": fixed.get("mean_pairwise_jaccard"),
            }

        if best is not None:
            entry["best_consistency"] = {
                "threshold": best.get("support_threshold"),
                "consistency": best.get("mean_basin_consistency"),
                "uniqueness_rate": best.get("mode_uniqueness_rate"),
                "unique_mode_supports": best.get("unique_mode_supports"),
                "mean_support_size": best.get("mean_mode_support_size"),
                "mean_jaccard": best.get("mean_pairwise_jaccard"),
                "full_unique": best_full_unique,
            }

        results.append(entry)

    return results


def _format_float(val: Any, fmt: str = ".3f") -> str:
    if val is None:
        return "--"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return "--"


def render_summary(results: List[Dict[str, Any]], threshold: float) -> str:
    if not results:
        return "No results to display."

    lines: List[str] = []
    lines.append("# Block Loss Ablation Summary")
    lines.append("")
    lines.append(f"Fixed threshold: {threshold:.1e} (secondary metric)")
    lines.append("")
    lines.append("| ts | loss | cos_sep | intra | inter | cons | uniq | jaccard | support |")
    lines.append("|---:|:-----|--------:|------:|------:|-----:|-----:|--------:|--------:|")

    results_sorted = sorted(results, key=lambda r: (r["target_size"], r["loss"]))
    for entry in results_sorted:
        cos = entry.get("cosine", {})
        fixed = entry.get("fixed", {})
        lines.append(
            "| {ts} | {loss} | {sep} | {intra} | {inter} | {cons} | {uniq} | {jac} | {supp} |".format(
                ts=entry["target_size"],
                loss=entry["loss"],
                sep=_format_float(cos.get("separation")),
                intra=_format_float(cos.get("mean_intra")),
                inter=_format_float(cos.get("mean_inter")),
                cons=_format_float(fixed.get("consistency")),
                uniq=_format_float(fixed.get("uniqueness_rate")),
                jac=_format_float(fixed.get("mean_jaccard")),
                supp=_format_float(fixed.get("mean_support_size"), ".1f"),
            )
        )

    lines.append("")
    lines.append("Best consistency (prefer full uniqueness if available):")
    lines.append("")
    lines.append("| ts | loss | best_tau | cons | uniq | jaccard | full_unique |")
    lines.append("|---:|:-----|---------:|-----:|-----:|--------:|:-----------:|")
    for entry in results_sorted:
        best = entry.get("best_consistency", {})
        if not best:
            lines.append(
                f"| {entry['target_size']} | {entry['loss']} | -- | -- | -- | -- | -- |"
            )
            continue
        lines.append(
            "| {ts} | {loss} | {tau} | {cons} | {uniq} | {jac} | {full} |".format(
                ts=entry["target_size"],
                loss=entry["loss"],
                tau=_format_float(best.get("threshold"), ".1e"),
                cons=_format_float(best.get("consistency")),
                uniq=_format_float(best.get("uniqueness_rate")),
                jac=_format_float(best.get("mean_jaccard")),
                full="yes" if best.get("full_unique") else "no",
            )
        )

    lines.append("")
    lines.append("Per-target-size best (by cosine separation) vs control:")
    for ts in sorted({r["target_size"] for r in results_sorted}):
        group = [r for r in results_sorted if r["target_size"] == ts]
        control = next((r for r in group if r["loss"] == "control"), None)
        if not control:
            continue
        best = max(group, key=lambda r: (r.get("cosine", {}).get("separation", 0.0)))
        ctrl_sep = control.get("cosine", {}).get("separation")
        best_sep = best.get("cosine", {}).get("separation")
        delta = None
        if ctrl_sep is not None and best_sep is not None:
            delta = float(best_sep) - float(ctrl_sep)
        lines.append(
            "- ts={ts}: best={loss} sep={best} (Δ={delta} vs control)".format(
                ts=ts,
                loss=best["loss"],
                best=_format_float(best_sep),
                delta=_format_float(delta),
            )
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect block loss ablation sweep results")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--summary_md", type=str, default=None)
    parser.add_argument("--require_full_unique", action="store_true",
                        help="Prefer thresholds with full uniqueness when selecting best consistency")
    parser.add_argument("--no_print", action="store_true", help="Do not print summary to stdout")
    args = parser.parse_args()

    results = collect_all(args.base_dir, args.threshold, args.require_full_unique)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)

    summary = render_summary(results, args.threshold)

    if args.summary_md:
        with open(args.summary_md, "w") as f:
            f.write(summary + "\n")

    if not args.no_print:
        print(summary)


if __name__ == "__main__":
    main()
