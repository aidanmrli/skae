#!/usr/bin/env python3
"""Compute Pareto frontier from summarize_encoder_comparison output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def _band_violation(sparsity: float, low: float, high: float) -> float:
    if sparsity < low:
        return low - sparsity
    if sparsity > high:
        return sparsity - high
    return 0.0


def _dominates(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> bool:
    return all(a[k] <= b[k] for k in keys) and any(a[k] < b[k] for k in keys)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.6e}" if abs(v) < 1e-3 else f"{v:.6f}"


def _render_md(
    records: List[Dict[str, object]],
    frontier: List[Dict[str, object]],
    target_sparsity: float,
    band_low: float,
    band_high: float,
) -> str:
    lines: List[str] = []
    lines.append("# Pareto Frontier (Forecasting vs Sparsity)")
    lines.append("")
    lines.append(f"- Target sparsity: `{target_sparsity:.3f}`")
    lines.append(f"- Sparsity band: `[{band_low:.3f}, {band_high:.3f}]`")
    lines.append("- Objectives minimized: `H1000`, `H500`, `quick_best`, `|sparsity-target|`")
    lines.append("")

    lines.append("## All Arms")
    lines.append("")
    lines.append(
        "| arm | H1000 | H500 | quick_best | sparsity | |s-0.8| | band_violation | in_band |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for r in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["arm"]),
                    _fmt(_safe_float(r["h1000_bp_mean"])),
                    _fmt(_safe_float(r["h500_bp_mean"])),
                    _fmt(_safe_float(r["quick_best_mean"])),
                    _fmt(_safe_float(r["sparsity_ratio_final_mean"])),
                    _fmt(_safe_float(r["sparsity_target_gap"])),
                    _fmt(_safe_float(r["sparsity_band_violation"])),
                    "Y" if r.get("in_band") else "N",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Pareto Frontier")
    lines.append("")
    lines.append(
        "| arm | H1000 | H500 | quick_best | sparsity | |s-0.8| | band_violation | in_band |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for r in frontier:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["arm"]),
                    _fmt(_safe_float(r["h1000_bp_mean"])),
                    _fmt(_safe_float(r["h500_bp_mean"])),
                    _fmt(_safe_float(r["quick_best_mean"])),
                    _fmt(_safe_float(r["sparsity_ratio_final_mean"])),
                    _fmt(_safe_float(r["sparsity_target_gap"])),
                    _fmt(_safe_float(r["sparsity_band_violation"])),
                    "Y" if r.get("in_band") else "N",
                ]
            )
            + " |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Pareto frontier from encoder summary JSON.")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    parser.add_argument("--target_sparsity", type=float, default=0.8)
    parser.add_argument("--band_low", type=float, default=0.7)
    parser.add_argument("--band_high", type=float, default=0.9)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    payload = json.loads(summary_path.read_text())
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError(f"Invalid summary payload: {summary_path}")

    records: List[Dict[str, object]] = []
    for arm, stats in summary.items():
        if not isinstance(stats, dict):
            continue
        quick = _safe_float(stats.get("quick_best_mean"))
        h500 = _safe_float(stats.get("h500_bp_mean"))
        h1000 = _safe_float(stats.get("h1000_bp_mean"))
        sparsity = _safe_float(stats.get("sparsity_ratio_final_mean"))
        if quick is None or h500 is None or h1000 is None or sparsity is None:
            continue

        rec: Dict[str, object] = {
            "arm": arm,
            "quick_best_mean": quick,
            "h500_bp_mean": h500,
            "h1000_bp_mean": h1000,
            "sparsity_ratio_final_mean": sparsity,
            "sparsity_target_gap": abs(sparsity - args.target_sparsity),
            "sparsity_band_violation": _band_violation(sparsity, args.band_low, args.band_high),
            "in_band": args.band_low <= sparsity <= args.band_high,
        }
        records.append(rec)

    objective_keys = [
        "h1000_bp_mean",
        "h500_bp_mean",
        "quick_best_mean",
        "sparsity_target_gap",
    ]

    frontier: List[Dict[str, object]] = []
    for cand in records:
        cand_obj = {k: float(cand[k]) for k in objective_keys}
        dominated = False
        for other in records:
            if other["arm"] == cand["arm"]:
                continue
            other_obj = {k: float(other[k]) for k in objective_keys}
            if _dominates(other_obj, cand_obj, objective_keys):
                dominated = True
                break
        if not dominated:
            frontier.append(cand)

    records_sorted = sorted(
        records,
        key=lambda r: (
            float(r["h1000_bp_mean"]),
            float(r["h500_bp_mean"]),
            float(r["quick_best_mean"]),
            float(r["sparsity_target_gap"]),
        ),
    )
    frontier_sorted = sorted(
        frontier,
        key=lambda r: (
            float(r["h1000_bp_mean"]),
            float(r["h500_bp_mean"]),
            float(r["quick_best_mean"]),
            float(r["sparsity_target_gap"]),
        ),
    )

    out_payload = {
        "summary_json": str(summary_path),
        "target_sparsity": args.target_sparsity,
        "band_low": args.band_low,
        "band_high": args.band_high,
        "objective_keys": objective_keys,
        "records": records_sorted,
        "pareto_frontier": frontier_sorted,
    }

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = output_prefix.with_name(output_prefix.name + "_pareto_frontier.json")
    out_md = output_prefix.with_name(output_prefix.name + "_pareto_frontier.md")
    out_json.write_text(json.dumps(out_payload, indent=2))
    out_md.write_text(
        _render_md(
            records=records_sorted,
            frontier=frontier_sorted,
            target_sparsity=args.target_sparsity,
            band_low=args.band_low,
            band_high=args.band_high,
        )
    )

    print(f"Wrote Pareto JSON: {out_json}")
    print(f"Wrote Pareto Markdown: {out_md}")


if __name__ == "__main__":
    main()
