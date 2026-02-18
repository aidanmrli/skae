#!/usr/bin/env python3
"""Compare dysts forecasting roots from collected rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


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


def _as_int(value: float) -> int:
    return int(round(value))


def _parse_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _group_by_root(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        root = str(row.get("root_label", ""))
        if root:
            grouped[root].append(row)
    return grouped


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    idx = (len(vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    w = idx - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def _representative_by_system(rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        system = str(row.get("system_name", ""))
        run_id = str(row.get("run_id", ""))
        if not system:
            continue
        prev = out.get(system)
        if prev is None or str(prev.get("run_id", "")) < run_id:
            out[system] = row
    return out


def _root_summary(
    rows: List[Dict[str, object]],
    horizon: int,
    good_threshold: float,
    catastrophic_threshold: float,
) -> Dict[str, object]:
    best_key = f"h{horizon}_best_periodic_mean"
    no_re_key = f"h{horizon}_no_reencode_mean"
    every_key = f"h{horizon}_every_step_mean"
    mode_key = f"h{horizon}_best_periodic_mode"

    best_vals: List[float] = []
    improved_vs_no_re = 0
    improved_vs_every = 0
    mode_counts: Counter[str] = Counter()

    for row in rows:
        bp = _safe_float(row.get(best_key))
        nr = _safe_float(row.get(no_re_key))
        es = _safe_float(row.get(every_key))
        mode = row.get(mode_key)
        if bp is not None:
            best_vals.append(bp)
        if bp is not None and nr is not None and bp < nr:
            improved_vs_no_re += 1
        if bp is not None and es is not None and bp < es:
            improved_vs_every += 1
        if mode:
            mode_counts[str(mode)] += 1

    good_count = sum(1 for v in best_vals if v < good_threshold)
    catastrophic_count = sum(1 for v in best_vals if v >= catastrophic_threshold)
    n = len(rows)
    n_systems = len({str(r.get("system_name", "")) for r in rows if r.get("system_name")})

    return {
        "n_rows": n,
        "n_systems": n_systems,
        "h_best_median": median(best_vals) if best_vals else None,
        "h_best_mean": mean(best_vals) if best_vals else None,
        "h_best_p90": _percentile(best_vals, 0.90),
        "h_best_max": max(best_vals) if best_vals else None,
        "good_count": good_count,
        "good_fraction": (good_count / n) if n else None,
        "catastrophic_count": catastrophic_count,
        "catastrophic_fraction": (catastrophic_count / n) if n else None,
        "improved_vs_no_reencode": improved_vs_no_re,
        "improved_vs_every_step": improved_vs_every,
        "top_modes": [{"mode": m, "count": c} for m, c in mode_counts.most_common(5)],
    }


def _paired_comparison(
    candidate_rows: List[Dict[str, object]],
    anchor_rows: List[Dict[str, object]],
    horizon: int,
    good_threshold: float,
) -> Dict[str, object]:
    best_key = f"h{horizon}_best_periodic_mean"

    cand_repr = _representative_by_system(candidate_rows)
    anch_repr = _representative_by_system(anchor_rows)
    shared_systems = sorted(set(cand_repr) & set(anch_repr))

    deltas: List[float] = []
    ratios: List[float] = []
    candidate_wins = 0
    anchor_wins = 0
    ties = 0
    candidate_good = 0
    anchor_good = 0
    both_good = 0
    candidate_fail_anchor_pass: List[Dict[str, object]] = []
    candidate_pass_anchor_fail: List[Dict[str, object]] = []

    for system in shared_systems:
        cand = cand_repr[system]
        anch = anch_repr[system]
        c = _safe_float(cand.get(best_key))
        a = _safe_float(anch.get(best_key))
        if c is None or a is None:
            continue
        deltas.append(c - a)
        if a > 0:
            ratios.append(c / a)
        if c < a:
            candidate_wins += 1
        elif c > a:
            anchor_wins += 1
        else:
            ties += 1

        c_good = c < good_threshold
        a_good = a < good_threshold
        if c_good:
            candidate_good += 1
        if a_good:
            anchor_good += 1
        if c_good and a_good:
            both_good += 1
        if (not c_good) and a_good:
            candidate_fail_anchor_pass.append(
                {"system": system, "candidate_best": c, "anchor_best": a}
            )
        if c_good and (not a_good):
            candidate_pass_anchor_fail.append(
                {"system": system, "candidate_best": c, "anchor_best": a}
            )

    candidate_fail_anchor_pass = sorted(
        candidate_fail_anchor_pass,
        key=lambda x: float(x["candidate_best"]),
        reverse=True,
    )
    candidate_pass_anchor_fail = sorted(
        candidate_pass_anchor_fail,
        key=lambda x: float(x["anchor_best"]),
        reverse=True,
    )

    return {
        "n_shared_systems": len(shared_systems),
        "candidate_wins": candidate_wins,
        "anchor_wins": anchor_wins,
        "ties": ties,
        "median_delta_candidate_minus_anchor": median(deltas) if deltas else None,
        "median_ratio_candidate_over_anchor": median(ratios) if ratios else None,
        "candidate_good_shared": candidate_good,
        "anchor_good_shared": anchor_good,
        "both_good_shared": both_good,
        "candidate_fail_anchor_pass": candidate_fail_anchor_pass,
        "candidate_pass_anchor_fail": candidate_pass_anchor_fail,
    }


def _candidate_seed_robustness(
    candidate_rows: List[Dict[str, object]],
    horizon: int,
    good_threshold: float,
    catastrophic_threshold: float,
) -> Dict[str, object]:
    best_key = f"h{horizon}_best_periodic_mean"
    per_system: Dict[str, List[float]] = defaultdict(list)
    for row in candidate_rows:
        system = str(row.get("system_name", ""))
        if not system:
            continue
        bp = _safe_float(row.get(best_key))
        if bp is not None:
            per_system[system].append(bp)

    system_rows: List[Dict[str, object]] = []
    all_seed_good = 0
    any_seed_bad = 0
    any_seed_catastrophic = 0
    any_seed_good = 0
    worst_seed_values: List[float] = []

    for system, vals in sorted(per_system.items()):
        if not vals:
            continue
        min_v = min(vals)
        med_v = median(vals)
        max_v = max(vals)
        std_v = pstdev(vals) if len(vals) > 1 else 0.0
        n = len(vals)
        n_good = sum(1 for v in vals if v < good_threshold)
        success_rate = n_good / n

        if max_v < good_threshold:
            all_seed_good += 1
        else:
            any_seed_bad += 1
        if max_v >= catastrophic_threshold:
            any_seed_catastrophic += 1
        if min_v < good_threshold:
            any_seed_good += 1
        worst_seed_values.append(max_v)

        system_rows.append(
            {
                "system": system,
                "n_runs": n,
                "best_periodic_min": min_v,
                "best_periodic_median": med_v,
                "best_periodic_max": max_v,
                "best_periodic_std": std_v,
                "seed_success_rate": success_rate,
            }
        )

    system_rows = sorted(system_rows, key=lambda x: float(x["best_periodic_max"]), reverse=True)
    return {
        "n_systems": len(system_rows),
        "all_seeds_good_systems": all_seed_good,
        "any_seed_bad_systems": any_seed_bad,
        "any_seed_catastrophic_systems": any_seed_catastrophic,
        "any_seed_good_systems": any_seed_good,
        "median_worst_seed_best_periodic": median(worst_seed_values) if worst_seed_values else None,
        "system_rows": system_rows,
    }


def _write_markdown(
    path: Path,
    output: Dict[str, object],
    horizon: int,
    good_threshold: float,
    catastrophic_threshold: float,
    top_k: int,
) -> None:
    lines: List[str] = []
    lines.append("# Dysts Forecast Root Comparison")
    lines.append("")
    lines.append(f"- Horizon: H{horizon}")
    lines.append(f"- Good threshold: H{horizon} best-periodic < {good_threshold}")
    lines.append(f"- Catastrophic threshold: H{horizon} best-periodic >= {catastrophic_threshold}")
    lines.append("")

    lines.append("## Root Summary")
    lines.append("")
    lines.append(
        "| root | n_rows | n_systems | median best | p90 best | max best | good | catastrophic | improved vs no-re | improved vs every-step |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    root_summary = output["root_summary"]
    for root in sorted(root_summary):
        row = root_summary[root]
        lines.append(
            f"| {root} | {_as_int(float(row['n_rows']))} | {_as_int(float(row['n_systems']))} | "
            f"{_fmt(_safe_float(row['h_best_median']))} | {_fmt(_safe_float(row['h_best_p90']))} | "
            f"{_fmt(_safe_float(row['h_best_max']))} | {_as_int(float(row['good_count']))} | "
            f"{_as_int(float(row['catastrophic_count']))} | "
            f"{_as_int(float(row['improved_vs_no_reencode']))} | {_as_int(float(row['improved_vs_every_step']))} |"
        )

    lines.append("")
    lines.append("## Candidate vs Anchors")
    lines.append("")
    lines.append(
        "| anchor | shared systems | candidate wins | anchor wins | ties | median (cand-anchor) | median (cand/anchor) | candidate good | anchor good | both good |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for anchor, stats in sorted(output["candidate_vs_anchors"].items()):
        lines.append(
            f"| {anchor} | {_as_int(float(stats['n_shared_systems']))} | {_as_int(float(stats['candidate_wins']))} | "
            f"{_as_int(float(stats['anchor_wins']))} | {_as_int(float(stats['ties']))} | "
            f"{_fmt(_safe_float(stats['median_delta_candidate_minus_anchor']))} | "
            f"{_fmt(_safe_float(stats['median_ratio_candidate_over_anchor']))} | "
            f"{_as_int(float(stats['candidate_good_shared']))} | {_as_int(float(stats['anchor_good_shared']))} | "
            f"{_as_int(float(stats['both_good_shared']))} |"
        )

    seed_stats = output["candidate_seed_robustness"]
    lines.append("")
    lines.append("## Candidate Seed Robustness")
    lines.append("")
    lines.append(
        f"- Systems: {_as_int(float(seed_stats['n_systems']))}, "
        f"all-seeds-good: {_as_int(float(seed_stats['all_seeds_good_systems']))}, "
        f"any-seed-bad: {_as_int(float(seed_stats['any_seed_bad_systems']))}, "
        f"any-seed-catastrophic: {_as_int(float(seed_stats['any_seed_catastrophic_systems']))}, "
        f"median worst-seed H{horizon}: {_fmt(_safe_float(seed_stats['median_worst_seed_best_periodic']))}"
    )
    lines.append("")
    lines.append(f"Top {top_k} worst systems by candidate worst-seed H{horizon}:")
    lines.append("")
    lines.append("| system | n_runs | min | median | max | std | seed success rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in seed_stats["system_rows"][:top_k]:
        lines.append(
            f"| {row['system']} | {_as_int(float(row['n_runs']))} | "
            f"{_fmt(_safe_float(row['best_periodic_min']))} | {_fmt(_safe_float(row['best_periodic_median']))} | "
            f"{_fmt(_safe_float(row['best_periodic_max']))} | {_fmt(_safe_float(row['best_periodic_std']))} | "
            f"{_fmt(_safe_float(row['seed_success_rate']))} |"
        )

    lines.append("")
    lines.append("## Candidate Failures vs Anchors")
    lines.append("")
    for anchor, stats in sorted(output["candidate_vs_anchors"].items()):
        lines.append(f"### Candidate fails but {anchor} passes")
        rows = stats["candidate_fail_anchor_pass"][:top_k]
        if not rows:
            lines.append("- None")
        else:
            for row in rows:
                lines.append(
                    f"- {row['system']}: candidate={_fmt(_safe_float(row['candidate_best']))}, "
                    f"{anchor}={_fmt(_safe_float(row['anchor_best']))}"
                )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare forecasting roots from collected dysts rows CSV.")
    parser.add_argument("--rows_csv", type=str, required=True, help="Path to dysts_forecasting_rows.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write comparison outputs")
    parser.add_argument("--candidate_root", type=str, required=True, help="Candidate root label to evaluate")
    parser.add_argument(
        "--anchor_roots",
        type=str,
        nargs="+",
        default=["generic_sparse", "lista_nonlinear"],
        help="Anchor root labels for paired comparison",
    )
    parser.add_argument("--horizon", type=int, default=1000, help="Horizon used in rows file")
    parser.add_argument("--good_threshold", type=float, default=10.0, help="Good best-periodic threshold")
    parser.add_argument(
        "--catastrophic_threshold",
        type=float,
        default=1_000.0,
        help="Catastrophic best-periodic threshold",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of systems to show in markdown lists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _parse_rows(Path(args.rows_csv))
    grouped = _group_by_root(rows)

    if args.candidate_root not in grouped:
        raise ValueError(f"Candidate root '{args.candidate_root}' not found in {args.rows_csv}")

    root_summary = {
        root: _root_summary(
            root_rows,
            horizon=args.horizon,
            good_threshold=args.good_threshold,
            catastrophic_threshold=args.catastrophic_threshold,
        )
        for root, root_rows in grouped.items()
    }

    candidate_rows = grouped[args.candidate_root]
    candidate_vs_anchors: Dict[str, Dict[str, object]] = {}
    for anchor in args.anchor_roots:
        if anchor not in grouped:
            continue
        candidate_vs_anchors[anchor] = _paired_comparison(
            candidate_rows=candidate_rows,
            anchor_rows=grouped[anchor],
            horizon=args.horizon,
            good_threshold=args.good_threshold,
        )

    seed_robustness = _candidate_seed_robustness(
        candidate_rows=candidate_rows,
        horizon=args.horizon,
        good_threshold=args.good_threshold,
        catastrophic_threshold=args.catastrophic_threshold,
    )

    output = {
        "rows_csv": str(args.rows_csv),
        "candidate_root": args.candidate_root,
        "anchor_roots": args.anchor_roots,
        "horizon": args.horizon,
        "good_threshold": args.good_threshold,
        "catastrophic_threshold": args.catastrophic_threshold,
        "root_summary": root_summary,
        "candidate_vs_anchors": candidate_vs_anchors,
        "candidate_seed_robustness": seed_robustness,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "dysts_forecasting_comparison.json"
    out_md = out_dir / "dysts_forecasting_comparison.md"
    out_json.write_text(json.dumps(output, indent=2))
    _write_markdown(
        path=out_md,
        output=output,
        horizon=args.horizon,
        good_threshold=args.good_threshold,
        catastrophic_threshold=args.catastrophic_threshold,
        top_k=args.top_k,
    )
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
