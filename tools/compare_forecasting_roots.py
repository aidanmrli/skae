#!/usr/bin/env python3
"""Compare forecasting roots from mixed-system collected rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
        root = str(row.get("root_label", "")).strip()
        if root:
            grouped[root].append(row)
    return grouped


def _system_id(row: Dict[str, object]) -> str:
    system_key = str(row.get("system_key", "")).strip()
    if system_key and system_key != "None":
        return system_key
    return str(row.get("system_name", "")).strip()


def _seed_id(row: Dict[str, object]) -> str:
    seed_name = str(row.get("seed_name", "")).strip()
    if seed_name and seed_name != "None":
        return seed_name

    seed_value = _safe_float(row.get("seed"))
    if seed_value is not None:
        return f"seed_{int(round(seed_value))}"

    run_dir = str(row.get("run_dir", ""))
    match = re.search(r"/(seed_\d+)(?:/|$)", run_dir)
    if match:
        return match.group(1)

    return "seed_0"


def _run_sort_key(row: Dict[str, object]) -> Tuple[str, str]:
    return (str(row.get("run_id", "")), str(row.get("run_dir", "")))


def _latest_rows_by_system_seed(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        system = _system_id(row)
        if not system:
            continue
        seed = _seed_id(row)
        key = (system, seed)
        prev = latest.get(key)
        if prev is None or _run_sort_key(row) > _run_sort_key(prev):
            latest[key] = row
    return sorted(latest.values(), key=lambda r: (_system_id(r), _seed_id(r), _run_sort_key(r)))


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


def _system_rows(rows: List[Dict[str, object]], horizon: int) -> List[Dict[str, object]]:
    best_key = f"h{horizon}_best_periodic_mean"
    no_re_key = f"h{horizon}_no_reencode_mean"
    every_key = f"h{horizon}_every_step_mean"
    mode_key = f"h{horizon}_best_periodic_mode"

    latest = _latest_rows_by_system_seed(rows)
    by_system: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in latest:
        by_system[_system_id(row)].append(row)

    output: List[Dict[str, object]] = []
    for system, items in sorted(by_system.items()):
        bp_vals = [_safe_float(r.get(best_key)) for r in items]
        bp_vals = [v for v in bp_vals if v is not None]
        if not bp_vals:
            continue

        nr_vals = [_safe_float(r.get(no_re_key)) for r in items]
        nr_vals = [v for v in nr_vals if v is not None]
        es_vals = [_safe_float(r.get(every_key)) for r in items]
        es_vals = [v for v in es_vals if v is not None]
        modes = [str(r.get(mode_key)) for r in items if r.get(mode_key)]
        consensus_mode = Counter(modes).most_common(1)[0][0] if modes else None

        output.append(
            {
                "system": system,
                "n_seeds": len(bp_vals),
                "best_periodic_min": min(bp_vals),
                "best_periodic_median": median(bp_vals),
                "best_periodic_max": max(bp_vals),
                "best_periodic_std": pstdev(bp_vals) if len(bp_vals) > 1 else 0.0,
                "no_reencode_median": median(nr_vals) if nr_vals else None,
                "every_step_median": median(es_vals) if es_vals else None,
                "consensus_mode": consensus_mode,
                "seed_success_rate": 0.0,  # filled later where thresholds are known
            }
        )
    return output


def _root_summary(
    rows: List[Dict[str, object]],
    horizon: int,
    good_threshold: float,
    catastrophic_threshold: float,
) -> Dict[str, object]:
    system_rows = _system_rows(rows, horizon)

    medians = [float(r["best_periodic_median"]) for r in system_rows]
    good_count = sum(1 for r in system_rows if float(r["best_periodic_median"]) < good_threshold)
    catastrophic_count = sum(
        1 for r in system_rows if float(r["best_periodic_median"]) >= catastrophic_threshold
    )
    all_seeds_good = sum(1 for r in system_rows if float(r["best_periodic_max"]) < good_threshold)
    any_seed_catastrophic = sum(
        1 for r in system_rows if float(r["best_periodic_max"]) >= catastrophic_threshold
    )
    periodic_1_count = sum(1 for r in system_rows if r.get("consensus_mode") == "periodic_1")
    improved_vs_no_re = sum(
        1
        for r in system_rows
        if _safe_float(r.get("no_reencode_median")) is not None
        and float(r["best_periodic_median"]) < float(r["no_reencode_median"])
    )
    improved_vs_every = sum(
        1
        for r in system_rows
        if _safe_float(r.get("every_step_median")) is not None
        and float(r["best_periodic_median"]) < float(r["every_step_median"])
    )

    mode_counts = Counter(str(r["consensus_mode"]) for r in system_rows if r.get("consensus_mode"))

    latest = _latest_rows_by_system_seed(rows)
    return {
        "n_rows_collected": len(latest),
        "n_systems": len(system_rows),
        "h_best_median": median(medians) if medians else None,
        "h_best_mean": mean(medians) if medians else None,
        "h_best_p90": _percentile(medians, 0.90),
        "h_best_max": max(medians) if medians else None,
        "good_systems": good_count,
        "catastrophic_systems": catastrophic_count,
        "all_seeds_good_systems": all_seeds_good,
        "any_seed_catastrophic_systems": any_seed_catastrophic,
        "periodic_1_systems": periodic_1_count,
        "improved_vs_no_reencode": improved_vs_no_re,
        "improved_vs_every_step": improved_vs_every,
        "top_consensus_modes": [{"mode": m, "count": c} for m, c in mode_counts.most_common(5)],
        "system_rows": sorted(system_rows, key=lambda x: float(x["best_periodic_max"]), reverse=True),
    }


def _system_median_map(rows: List[Dict[str, object]], horizon: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in _system_rows(rows, horizon):
        out[str(row["system"])] = float(row["best_periodic_median"])
    return out


def _paired_comparison(
    candidate_rows: List[Dict[str, object]],
    anchor_rows: List[Dict[str, object]],
    horizon: int,
    good_threshold: float,
) -> Dict[str, object]:
    cand_map = _system_median_map(candidate_rows, horizon)
    anchor_map = _system_median_map(anchor_rows, horizon)
    shared_systems = sorted(set(cand_map) & set(anchor_map))

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
        c = cand_map[system]
        a = anchor_map[system]
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
    system_rows = _system_rows(candidate_rows, horizon)
    out_rows: List[Dict[str, object]] = []
    worst_seed_values: List[float] = []

    all_seed_good = 0
    any_seed_bad = 0
    any_seed_catastrophic = 0
    any_seed_good = 0

    for row in system_rows:
        best_min = float(row["best_periodic_min"])
        best_median = float(row["best_periodic_median"])
        best_max = float(row["best_periodic_max"])
        n_seeds = int(row["n_seeds"])
        n_good = int(round(n_seeds * 0))  # overwritten below

        # Recover latest per-system-per-seed rows to compute exact success rate.
        # This avoids relying on rounded statistics.
        n_good = 0
        for seed_row in _latest_rows_by_system_seed(candidate_rows):
            if _system_id(seed_row) != str(row["system"]):
                continue
            bp = _safe_float(seed_row.get(f"h{horizon}_best_periodic_mean"))
            if bp is not None and bp < good_threshold:
                n_good += 1

        success_rate = (n_good / n_seeds) if n_seeds > 0 else 0.0

        if best_max < good_threshold:
            all_seed_good += 1
        else:
            any_seed_bad += 1
        if best_max >= catastrophic_threshold:
            any_seed_catastrophic += 1
        if best_min < good_threshold:
            any_seed_good += 1
        worst_seed_values.append(best_max)

        out_rows.append(
            {
                "system": row["system"],
                "n_runs": n_seeds,
                "best_periodic_min": best_min,
                "best_periodic_median": best_median,
                "best_periodic_max": best_max,
                "best_periodic_std": float(row["best_periodic_std"]),
                "seed_success_rate": success_rate,
            }
        )

    out_rows = sorted(out_rows, key=lambda x: float(x["best_periodic_max"]), reverse=True)
    return {
        "n_systems": len(out_rows),
        "all_seeds_good_systems": all_seed_good,
        "any_seed_bad_systems": any_seed_bad,
        "any_seed_catastrophic_systems": any_seed_catastrophic,
        "any_seed_good_systems": any_seed_good,
        "median_worst_seed_best_periodic": median(worst_seed_values) if worst_seed_values else None,
        "system_rows": out_rows,
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
    lines.append("# Forecast Root Comparison")
    lines.append("")
    lines.append(f"- Horizon: H{horizon}")
    lines.append(f"- Good threshold: H{horizon} best-periodic < {good_threshold}")
    lines.append(f"- Catastrophic threshold: H{horizon} best-periodic >= {catastrophic_threshold}")
    lines.append("")

    lines.append("## Root Summary")
    lines.append("")
    lines.append(
        "| root | runs collected | systems | good systems | catastrophic systems | median best | p90 best | periodic_1 systems | all-seeds-good | any-seed-catastrophic |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    root_summary = output["root_summary"]
    for root in sorted(root_summary):
        row = root_summary[root]
        lines.append(
            f"| {root} | {_as_int(float(row['n_rows_collected']))} | {_as_int(float(row['n_systems']))} | "
            f"{_as_int(float(row['good_systems']))} | {_as_int(float(row['catastrophic_systems']))} | "
            f"{_fmt(_safe_float(row['h_best_median']))} | {_fmt(_safe_float(row['h_best_p90']))} | "
            f"{_as_int(float(row['periodic_1_systems']))} | {_as_int(float(row['all_seeds_good_systems']))} | "
            f"{_as_int(float(row['any_seed_catastrophic_systems']))} |"
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
    parser = argparse.ArgumentParser(
        description="Compare forecasting roots from collected mixed-system rows CSV."
    )
    parser.add_argument("--rows_csv", type=str, required=True, help="Path to forecasting_rows.csv")
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
    out_json = out_dir / "forecasting_comparison.json"
    out_md = out_dir / "forecasting_comparison.md"
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
