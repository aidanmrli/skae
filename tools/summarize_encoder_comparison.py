#!/usr/bin/env python3
"""Summarize encoder comparison runs (arm-based layout).

Expected layout:
  <base_root>/<arm_name>/<system>/seed_<seed>/<run_id>/

Produces a markdown + JSON summary comparing arms across seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


def _read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _latest_run_dir(seed_dir: Path) -> Optional[Path]:
    runs = sorted([p for p in seed_dir.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def _quick_metrics(
    run_dir: Path,
) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[float]]:
    metrics_history = run_dir / "metrics_history.jsonl"
    quick_best: Optional[float] = None
    quick_best_step: Optional[int] = None

    if metrics_history.exists():
        with metrics_history.open("r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("name") != "eval/final_error":
                    continue
                value = _safe_float(row.get("value"))
                step = row.get("step")
                if value is None or not isinstance(step, int):
                    continue
                if quick_best is None or value < quick_best:
                    quick_best = value
                    quick_best_step = step

    quick_last: Optional[float] = None
    sparsity_final: Optional[float] = None
    metrics_summary = _read_json(run_dir / "metrics_summary.json")
    if isinstance(metrics_summary, dict):
        eval_final = metrics_summary.get("eval/final_error")
        if isinstance(eval_final, dict):
            quick_last = _safe_float(eval_final.get("final"))
        sparsity_ratio = metrics_summary.get("train/sparsity_ratio")
        if isinstance(sparsity_ratio, dict):
            sparsity_final = _safe_float(sparsity_ratio.get("final"))

    return quick_best, quick_best_step, quick_last, sparsity_final


def _best_periodic(eval_json: Dict, system: str, horizon: int) -> Tuple[Optional[float], Optional[str]]:
    system_data = eval_json.get(system)
    if not isinstance(system_data, dict):
        return None, None
    best = system_data.get("best_periodic", {}).get(str(horizon))
    if not isinstance(best, dict):
        return None, None
    return _safe_float(best.get("mean")), best.get("mode")


def _collect_rows(
    base_root: Path,
    system: str,
    arms: List[str],
    seeds: List[int],
    eval_file_name: str,
    horizons: List[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for arm in arms:
        for seed in seeds:
            seed_dir = base_root / arm / system / f"seed_{seed}"
            if not seed_dir.exists():
                continue
            run_dir = _latest_run_dir(seed_dir)
            if run_dir is None:
                continue

            quick_best, quick_best_step, quick_last, sparsity_final = _quick_metrics(run_dir)
            eval_data = _read_json(run_dir / eval_file_name)
            if not isinstance(eval_data, dict):
                eval_data = {}

            row: Dict[str, object] = {
                "arm": arm,
                "seed": seed,
                "run_dir": str(run_dir),
                "quick_best": quick_best,
                "quick_best_step": quick_best_step,
                "quick_last": quick_last,
                "sparsity_ratio_final": sparsity_final,
            }

            for h in horizons:
                h_mean, h_mode = _best_periodic(eval_data, system=system, horizon=h)
                row[f"h{h}_bp"] = h_mean
                row[f"h{h}_mode"] = h_mode

            rows.append(row)
    return rows


def _aggregate(rows: List[Dict[str, object]], arms: List[str], horizons: List[int]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for arm in arms:
        grp = [r for r in rows if r["arm"] == arm]
        quick_best_vals = [v for v in (_safe_float(r.get("quick_best")) for r in grp) if v is not None]
        quick_last_vals = [v for v in (_safe_float(r.get("quick_last")) for r in grp) if v is not None]
        sparsity_vals = [v for v in (_safe_float(r.get("sparsity_ratio_final")) for r in grp) if v is not None]

        stats: Dict[str, object] = {
            "n": len(grp),
            "quick_best_mean": mean(quick_best_vals) if quick_best_vals else None,
            "quick_best_median": median(quick_best_vals) if quick_best_vals else None,
            "quick_last_mean": mean(quick_last_vals) if quick_last_vals else None,
            "quick_last_median": median(quick_last_vals) if quick_last_vals else None,
            "sparsity_ratio_final_mean": mean(sparsity_vals) if sparsity_vals else None,
            "sparsity_ratio_final_median": median(sparsity_vals) if sparsity_vals else None,
        }

        for h in horizons:
            values = [v for v in (_safe_float(r.get(f"h{h}_bp")) for r in grp) if v is not None]
            stats[f"h{h}_bp_mean"] = mean(values) if values else None
            stats[f"h{h}_bp_median"] = median(values) if values else None
            mode_counter = Counter(
                str(mode)
                for mode in (r.get(f"h{h}_mode") for r in grp)
                if isinstance(mode, str) and mode
            )
            total_modes = sum(mode_counter.values())
            stats[f"h{h}_mode_distribution"] = (
                {mode: count / total_modes for mode, count in mode_counter.most_common()}
                if total_modes > 0
                else {}
            )

        out[arm] = stats
    return out


def _seedwise_ratios(
    rows: List[Dict[str, object]],
    candidate: str,
    anchor: str,
    horizons: List[int],
) -> List[Dict[str, object]]:
    cand_by_seed = {int(r["seed"]): r for r in rows if r["arm"] == candidate}
    anch_by_seed = {int(r["seed"]): r for r in rows if r["arm"] == anchor}
    shared = sorted(set(cand_by_seed) & set(anch_by_seed))
    out: List[Dict[str, object]] = []
    for seed in shared:
        c = cand_by_seed[seed]
        a = anch_by_seed[seed]
        row: Dict[str, object] = {"seed": seed}

        quick_a = _safe_float(a.get("quick_best"))
        quick_c = _safe_float(c.get("quick_best"))
        if quick_a and quick_c is not None:
            row["quick_best_ratio"] = quick_c / quick_a
        else:
            row["quick_best_ratio"] = None

        for h in horizons:
            a_val = _safe_float(a.get(f"h{h}_bp"))
            c_val = _safe_float(c.get(f"h{h}_bp"))
            if a_val and c_val is not None:
                row[f"h{h}_ratio"] = c_val / a_val
            else:
                row[f"h{h}_ratio"] = None
        out.append(row)
    return out


def _fmt(value: Optional[float], kind: str = "auto") -> str:
    if value is None:
        return "N/A"
    if kind == "fixed":
        return f"{value:.6f}"
    return f"{value:.6e}" if abs(value) < 1e-3 else f"{value:.6f}"


def _render_md(
    rows: List[Dict[str, object]],
    summary: Dict[str, Dict[str, object]],
    ratio_sections: Dict[str, List[Dict[str, object]]],
    base_root: str,
    arms: List[str],
    horizons: List[int],
) -> str:
    lines: List[str] = []
    lines.append(f"# Duffing Encoder Comparison ({len(rows)} rows)")
    lines.append("")
    lines.append(f"- Base root: `{base_root}`")
    lines.append(f"- Arms: {', '.join(arms)}")
    lines.append("")

    # Seed-wise table
    lines.append("## Seed-wise rows")
    lines.append("")
    header = "| arm | seed | quick best | quick last | sparsity(final) | " + " | ".join(
        [f"H{h} best-periodic" for h in horizons]
    ) + " |"
    sep = "|---|---:|---:|---:|---:|" + "---:|" * len(horizons)
    lines.append(header)
    lines.append(sep)

    for row in sorted(rows, key=lambda r: (arms.index(r["arm"]), int(r["seed"]))):
        values = [
            str(row["arm"]),
            str(int(row["seed"])),
            f"{_fmt(_safe_float(row.get('quick_best')), kind='fixed')} @ {row.get('quick_best_step')}",
            _fmt(_safe_float(row.get("quick_last")), kind="fixed"),
            _fmt(_safe_float(row.get("sparsity_ratio_final")), kind="fixed"),
        ]
        for h in horizons:
            h_val = _safe_float(row.get(f"h{h}_bp"))
            h_mode = row.get(f"h{h}_mode")
            values.append(f"{_fmt(h_val)} ({h_mode})")
        lines.append("| " + " | ".join(values) + " |")

    # Aggregate
    lines.append("")
    lines.append("## Aggregate (mean / median)")
    lines.append("")
    for arm in arms:
        s = summary.get(arm, {})
        part = (
            f"- **{arm}**: "
            f"quick_best={_fmt(_safe_float(s.get('quick_best_mean')), kind='fixed')}/"
            f"{_fmt(_safe_float(s.get('quick_best_median')), kind='fixed')}, "
            f"quick_last={_fmt(_safe_float(s.get('quick_last_mean')), kind='fixed')}/"
            f"{_fmt(_safe_float(s.get('quick_last_median')), kind='fixed')}, "
            f"sparsity={_fmt(_safe_float(s.get('sparsity_ratio_final_mean')), kind='fixed')}/"
            f"{_fmt(_safe_float(s.get('sparsity_ratio_final_median')), kind='fixed')}"
        )
        for h in horizons:
            part += (
                f", H{h}="
                f"{_fmt(_safe_float(s.get(f'h{h}_bp_mean')))}"
                f"/{_fmt(_safe_float(s.get(f'h{h}_bp_median')))}"
            )
        lines.append(part)

    lines.append("")
    lines.append("## Best-period mode distribution")
    lines.append("")
    for arm in arms:
        s = summary.get(arm, {})
        for h in horizons:
            mode_dist = s.get(f"h{h}_mode_distribution")
            if not isinstance(mode_dist, dict) or not mode_dist:
                lines.append(f"- **{arm}**, H{h}: N/A")
                continue
            dist_parts = ", ".join([f"{k}={v:.2f}" for k, v in mode_dist.items()])
            lines.append(f"- **{arm}**, H{h}: {dist_parts}")

    # Ratios
    for label, ratios in ratio_sections.items():
        lines.append("")
        lines.append(f"## {label} (lower favors candidate)")
        lines.append("")
        for row in ratios:
            part = f"- seed {int(row['seed'])}: quick_best={_fmt(_safe_float(row.get('quick_best_ratio')))}"
            for h in horizons:
                part += f", H{h}={_fmt(_safe_float(row.get(f'h{h}_ratio')))}"
            lines.append(part)

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize arm-based encoder comparison runs."
    )
    parser.add_argument("--base_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="encoder_comparison")
    parser.add_argument("--system", type=str, default="duffing")
    parser.add_argument("--arms", type=str, nargs="+",
                        default=["lista_current", "lista_matched", "generic_sparse"])
    parser.add_argument("--anchor", type=str, default="generic_sparse",
                        help="Anchor arm for ratio computation")
    parser.add_argument("--eval_file_name", type=str, default="evaluation_results_best.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--horizons", type=int, nargs="+", default=[100, 500, 1000])
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = list(args.horizons)
    arms = list(args.arms)

    rows = _collect_rows(
        base_root=base_root,
        system=args.system,
        arms=arms,
        seeds=args.seeds,
        eval_file_name=args.eval_file_name,
        horizons=horizons,
    )
    summary = _aggregate(rows=rows, arms=arms, horizons=horizons)

    # Compute ratios: each non-anchor arm vs anchor
    ratio_sections: Dict[str, List[Dict[str, object]]] = {}
    for arm in arms:
        if arm == args.anchor:
            continue
        ratios = _seedwise_ratios(rows=rows, candidate=arm, anchor=args.anchor, horizons=horizons)
        ratio_sections[f"{arm} / {args.anchor}"] = ratios

    payload = {
        "base_root": str(base_root),
        "arms": arms,
        "anchor": args.anchor,
        "rows": rows,
        "summary": summary,
        "ratio_sections": ratio_sections,
    }

    json_path = output_dir / f"{args.output_prefix}_summary.json"
    md_path = output_dir / f"{args.output_prefix}_summary.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(
        _render_md(
            rows=rows,
            summary=summary,
            ratio_sections=ratio_sections,
            base_root=str(base_root),
            arms=arms,
            horizons=horizons,
        )
    )

    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote Markdown summary: {md_path}")


if __name__ == "__main__":
    main()
