#!/usr/bin/env python3
"""Summarize Duffing pairwise-vs-sequence parity runs (L=1 vs L=8).

Expected layout:
  <base_root>/L1/<system>/seed_<seed>/<run_id>/
  <base_root>/L8/<system>/seed_<seed>/<run_id>/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple


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
    if not runs:
        return None
    return runs[-1]


def _quick_metrics(run_dir: Path) -> Tuple[Optional[float], Optional[int], Optional[float]]:
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
    metrics_summary = _read_json(run_dir / "metrics_summary.json")
    if isinstance(metrics_summary, dict):
        eval_final = metrics_summary.get("eval/final_error")
        if isinstance(eval_final, dict):
            quick_last = _safe_float(eval_final.get("final"))

    return quick_best, quick_best_step, quick_last


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
    sequence_lengths: Iterable[int],
    seeds: Iterable[int],
    eval_file_name: str,
    horizons: Iterable[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seq_len in sequence_lengths:
        for seed in seeds:
            seed_dir = base_root / f"L{seq_len}" / system / f"seed_{seed}"
            if not seed_dir.exists():
                continue
            run_dir = _latest_run_dir(seed_dir)
            if run_dir is None:
                continue

            quick_best, quick_best_step, quick_last = _quick_metrics(run_dir)
            eval_data = _read_json(run_dir / eval_file_name)
            if not isinstance(eval_data, dict):
                eval_data = {}

            row: Dict[str, object] = {
                "L": seq_len,
                "seed": seed,
                "run_dir": str(run_dir),
                "quick_best": quick_best,
                "quick_best_step": quick_best_step,
                "quick_last": quick_last,
            }

            for h in horizons:
                h_mean, h_mode = _best_periodic(eval_data, system=system, horizon=h)
                row[f"h{h}_bp"] = h_mean
                row[f"h{h}_mode"] = h_mode

            rows.append(row)
    return rows


def _aggregate(rows: List[Dict[str, object]], horizons: Iterable[int]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for seq_len in sorted({int(r["L"]) for r in rows}):
        grp = [r for r in rows if int(r["L"]) == seq_len]
        quick_best_vals = [_safe_float(r.get("quick_best")) for r in grp]
        quick_best_vals = [v for v in quick_best_vals if v is not None]
        quick_last_vals = [_safe_float(r.get("quick_last")) for r in grp]
        quick_last_vals = [v for v in quick_last_vals if v is not None]

        stats: Dict[str, object] = {
            "n": len(grp),
            "quick_best_mean": mean(quick_best_vals) if quick_best_vals else None,
            "quick_best_median": median(quick_best_vals) if quick_best_vals else None,
            "quick_last_mean": mean(quick_last_vals) if quick_last_vals else None,
            "quick_last_median": median(quick_last_vals) if quick_last_vals else None,
        }

        for h in horizons:
            values = [_safe_float(r.get(f"h{h}_bp")) for r in grp]
            values = [v for v in values if v is not None]
            stats[f"h{h}_bp_mean"] = mean(values) if values else None
            stats[f"h{h}_bp_median"] = median(values) if values else None

        out[f"L{seq_len}"] = stats
    return out


def _seedwise_ratios(
    rows: List[Dict[str, object]],
    l_num: int,
    r_num: int,
    horizons: Iterable[int],
) -> List[Dict[str, object]]:
    left_by_seed = {int(r["seed"]): r for r in rows if int(r["L"]) == l_num}
    right_by_seed = {int(r["seed"]): r for r in rows if int(r["L"]) == r_num}
    shared = sorted(set(left_by_seed) & set(right_by_seed))
    out: List[Dict[str, object]] = []
    for seed in shared:
        left = left_by_seed[seed]
        right = right_by_seed[seed]
        row: Dict[str, object] = {"seed": seed}

        quick_l = _safe_float(left.get("quick_best"))
        quick_r = _safe_float(right.get("quick_best"))
        if quick_l is not None and quick_l > 0 and quick_r is not None:
            row[f"quick_best_ratio_l{r_num}_over_l{l_num}"] = quick_r / quick_l
        else:
            row[f"quick_best_ratio_l{r_num}_over_l{l_num}"] = None

        for h in horizons:
            l_val = _safe_float(left.get(f"h{h}_bp"))
            r_val = _safe_float(right.get(f"h{h}_bp"))
            key = f"h{h}_ratio_l{r_num}_over_l{l_num}"
            if l_val is not None and l_val > 0 and r_val is not None:
                row[key] = r_val / l_val
            else:
                row[key] = None
        out.append(row)
    return out


def _fmt(value: Optional[float], kind: str = "auto") -> str:
    if value is None:
        return "N/A"
    if kind == "fixed":
        return f"{value:.6f}"
    return f"{value:.6e}" if abs(value) < 1e-3 else f"{value:.6f}"


def _render_md(
    payload: Dict[str, object],
    base_root: str,
    l_num: int,
    r_num: int,
    horizons: List[int],
) -> str:
    rows = payload["rows"]
    summary = payload["summary"]
    ratios = payload["seedwise_l8_over_l1_ratios"]

    lines: List[str] = []
    lines.append(f"# Duffing PairSeq L{l_num} vs L{r_num} ({len(rows)} rows)")
    lines.append("")
    lines.append(f"- Base root: `{base_root}`")
    lines.append("")
    lines.append("## Seed-wise rows")
    lines.append("")
    header = "| L | seed | quick best | quick last | " + " | ".join(
        [f"H{h} best-periodic" for h in horizons]
    ) + " |"
    sep = "|---|---:|---:|---:|" + "---:|" * len(horizons)
    lines.append(header)
    lines.append(sep)

    for row in sorted(rows, key=lambda r: (int(r["L"]), int(r["seed"]))):
        values = [
            str(int(row["L"])),
            str(int(row["seed"])),
            f"{_fmt(_safe_float(row.get('quick_best')), kind='fixed')} @ {row.get('quick_best_step')}",
            _fmt(_safe_float(row.get("quick_last")), kind="fixed"),
        ]
        for h in horizons:
            h_val = _safe_float(row.get(f"h{h}_bp"))
            h_mode = row.get(f"h{h}_mode")
            values.append(f"{_fmt(h_val)} ({h_mode})")
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## Aggregate (mean / median)")
    lines.append("")
    for key in [f"L{l_num}", f"L{r_num}"]:
        s = summary.get(key, {})
        part = (
            f"- {key}: "
            f"quick_best={_fmt(_safe_float(s.get('quick_best_mean')), kind='fixed')}/"
            f"{_fmt(_safe_float(s.get('quick_best_median')), kind='fixed')}, "
            f"quick_last={_fmt(_safe_float(s.get('quick_last_mean')), kind='fixed')}/"
            f"{_fmt(_safe_float(s.get('quick_last_median')), kind='fixed')}"
        )
        for h in horizons:
            part += (
                f", H{h}="
                f"{_fmt(_safe_float(s.get(f'h{h}_bp_mean')))}"
                f"/{_fmt(_safe_float(s.get(f'h{h}_bp_median')))}"
            )
        lines.append(part)

    lines.append("")
    lines.append(f"## L{r_num}/L{l_num} ratio by seed (lower is better)")
    lines.append("")
    for row in ratios:
        part = (
            f"- seed {int(row['seed'])}: "
            f"quick_best={_fmt(_safe_float(row.get(f'quick_best_ratio_l{r_num}_over_l{l_num}')))}"
        )
        for h in horizons:
            part += f", H{h}={_fmt(_safe_float(row.get(f'h{h}_ratio_l{r_num}_over_l{l_num}')))}"
        lines.append(part)

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Duffing L1-vs-L8 pairwise/sequence runs."
    )
    parser.add_argument("--base_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="duffing_lista_pairseq")
    parser.add_argument("--system", type=str, default="duffing")
    parser.add_argument("--eval_file_name", type=str, default="evaluation_results_best.json")
    parser.add_argument("--l_values", type=int, nargs=2, default=[1, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--horizons", type=int, nargs="+", default=[100, 500, 1000])
    args = parser.parse_args()

    base_root = Path(args.base_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    l_num, r_num = args.l_values
    horizons = list(args.horizons)

    rows = _collect_rows(
        base_root=base_root,
        system=args.system,
        sequence_lengths=[l_num, r_num],
        seeds=args.seeds,
        eval_file_name=args.eval_file_name,
        horizons=horizons,
    )
    summary = _aggregate(rows=rows, horizons=horizons)
    ratios = _seedwise_ratios(rows=rows, l_num=l_num, r_num=r_num, horizons=horizons)

    payload: Dict[str, object] = {
        "base_root": str(base_root),
        "rows": rows,
        "summary": summary,
        "seedwise_l8_over_l1_ratios": ratios,
    }

    json_path = output_dir / f"{args.output_prefix}_summary.json"
    md_path = output_dir / f"{args.output_prefix}_summary.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(
        _render_md(
            payload=payload,
            base_root=str(base_root),
            l_num=l_num,
            r_num=r_num,
            horizons=horizons,
        )
    )

    print(f"Wrote JSON summary: {json_path}")
    print(f"Wrote Markdown summary: {md_path}")


if __name__ == "__main__":
    main()
