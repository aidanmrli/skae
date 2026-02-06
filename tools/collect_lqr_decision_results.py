#!/usr/bin/env python
"""Collect LQR-readiness runs and apply the pre-registered decision rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

RUN_NAME_RE = re.compile(
    r"stage(?P<stage>\d+)_(?P<system>[a-z0-9_]+)_(?P<arm>[a-z0-9_]+)_"
    r"bp(?P<b_proxy>\d+)_ts(?P<ts>\d+)_seed(?P<seed>\d+)"
)

METRICS = [
    "m1_local_fit_nrmse_1_step",
    "m1_local_fit_nrmse_h_step",
    "m2_lqr_feasibility_rate",
    "m3_closed_loop_stability_rate",
    "m4_closed_loop_cost_reduction",
    "m4_recovery_success_rate",
]


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None, None
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def _bootstrap_ci(
    values: Sequence[float],
    num_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    vals = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None, None
    if vals.size == 1:
        return float(vals[0]), float(vals[0])

    rng = np.random.default_rng(seed)
    boots = np.empty((num_bootstrap,), dtype=np.float64)
    n = vals.size
    for i in range(num_bootstrap):
        idx = rng.integers(low=0, high=n, size=n)
        boots[i] = float(np.mean(vals[idx]))

    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def _paired_bootstrap_diff_ci(
    diffs: Sequence[float],
    num_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = np.asarray([float(v) for v in diffs if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None, None, None

    mean = float(np.mean(vals))
    if vals.size == 1:
        return mean, float(vals[0]), float(vals[0])

    rng = np.random.default_rng(seed)
    boots = np.empty((num_bootstrap,), dtype=np.float64)
    n = vals.size
    for i in range(num_bootstrap):
        idx = rng.integers(low=0, high=n, size=n)
        boots[i] = float(np.mean(vals[idx]))

    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return mean, lo, hi


def _ci_excludes_zero(ci_low: Optional[float], ci_high: Optional[float]) -> bool:
    if ci_low is None or ci_high is None:
        return False
    return ci_low > 0.0 or ci_high < 0.0


def _parse_from_path(path: Path) -> Dict[str, Any]:
    for parent in [path.parent.name, path.parent.parent.name, path.parent.parent.parent.name]:
        m = RUN_NAME_RE.match(parent)
        if m:
            return {
                "stage": int(m.group("stage")),
                "system": m.group("system"),
                "arm": m.group("arm"),
                "b_proxy": int(m.group("b_proxy")),
                "target_size": int(m.group("ts")),
                "seed": int(m.group("seed")),
            }
    return {}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def collect_run_rows(base_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Use os.walk with followlinks=True so symlinked run directories are found.
    import os
    summary_paths: List[Path] = []
    for root, _dirs, files in os.walk(base_dir, followlinks=True):
        if "lqr_readiness_summary.json" in files:
            summary_paths.append(Path(root) / "lqr_readiness_summary.json")
    for summary_path in sorted(summary_paths):
        data = _load_json(summary_path)
        meta = dict(data.get("metadata", {}))
        agg = dict(data.get("aggregate_metrics", {}))

        # Fallback to parse naming if metadata is incomplete.
        parsed = _parse_from_path(summary_path)
        stage = meta.get("stage", parsed.get("stage", -1))
        system = meta.get("system", parsed.get("system", "unknown"))
        arm = meta.get("arm", parsed.get("arm", "unknown"))
        target_size = meta.get("target_size", parsed.get("target_size"))
        b_proxy = meta.get("b_proxy", parsed.get("b_proxy"))
        seed = meta.get("run_seed", parsed.get("seed"))

        row: Dict[str, Any] = {
            "summary_path": str(summary_path),
            "run_dir": str(summary_path.parent.parent),
            "stage": int(stage) if stage is not None else -1,
            "system": str(system),
            "arm": str(arm),
            "target_size": int(target_size) if target_size is not None else -1,
            "b_proxy": int(b_proxy) if b_proxy is not None else -1,
            "seed": int(seed) if seed is not None else -1,
            "num_regimes_total": int(agg.get("num_regimes_total", 0) or 0),
            "num_regimes_evaluable": int(agg.get("num_regimes_evaluable", 0) or 0),
        }

        for metric in METRICS:
            row[metric] = _to_float(agg.get(metric))

        rows.append(row)

    return rows


def _group_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return grouped


def build_summary_table(
    rows: Sequence[Dict[str, Any]],
    num_bootstrap: int,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    grouped = _group_rows(rows, keys=["arm", "target_size", "b_proxy"])

    out: List[Dict[str, Any]] = []
    for (arm, target_size, b_proxy), group in sorted(grouped.items()):
        entry: Dict[str, Any] = {
            "arm": arm,
            "target_size": target_size,
            "b_proxy": b_proxy,
            "n_runs": len(group),
        }
        for mi, metric in enumerate(METRICS):
            vals = [r.get(metric) for r in group]
            mean, std = _mean_std(vals)
            ci_low, ci_high = _bootstrap_ci(
                vals,
                num_bootstrap=num_bootstrap,
                seed=bootstrap_seed + 101 * mi + 17,
            )
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
            entry[f"{metric}_ci_low"] = ci_low
            entry[f"{metric}_ci_high"] = ci_high
        out.append(entry)

    # Overall arm-level rows.
    arm_grouped = _group_rows(rows, keys=["arm"])
    for (arm,), group in sorted(arm_grouped.items()):
        entry = {
            "arm": arm,
            "target_size": "all",
            "b_proxy": "all",
            "n_runs": len(group),
        }
        for mi, metric in enumerate(METRICS):
            vals = [r.get(metric) for r in group]
            mean, std = _mean_std(vals)
            ci_low, ci_high = _bootstrap_ci(
                vals,
                num_bootstrap=num_bootstrap,
                seed=bootstrap_seed + 503 * mi + 31,
            )
            entry[f"{metric}_mean"] = mean
            entry[f"{metric}_std"] = std
            entry[f"{metric}_ci_low"] = ci_low
            entry[f"{metric}_ci_high"] = ci_high
        out.append(entry)

    return out


def _arm_m5_sensitivity(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Compute robustness sensitivity to B_proxy (lower is better).

    For each (arm, target_size, seed), compute the range over B_proxy of
    score = 0.5 * (M2 + M3), then average ranges.
    """
    grouped = _group_rows(rows, keys=["arm", "target_size", "seed"])
    by_arm: Dict[str, List[float]] = defaultdict(list)

    for (arm, _target_size, _seed), group in grouped.items():
        vals: List[float] = []
        for r in group:
            m2 = r.get("m2_lqr_feasibility_rate")
            m3 = r.get("m3_closed_loop_stability_rate")
            if m2 is None or m3 is None:
                continue
            vals.append(0.5 * (float(m2) + float(m3)))
        if len(vals) >= 2:
            by_arm[arm].append(float(max(vals) - min(vals)))

    out: Dict[str, Optional[float]] = {}
    all_arms = sorted({r["arm"] for r in rows})
    for arm in all_arms:
        if by_arm[arm]:
            out[arm] = float(np.mean(by_arm[arm]))
        else:
            out[arm] = None
    return out


def _pair_rows(
    rows: Sequence[Dict[str, Any]],
    arm_a: str,
    arm_b: str,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    a_map: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    b_map: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    for row in rows:
        key = (int(row["target_size"]), int(row["b_proxy"]), int(row["seed"]))
        if row["arm"] == arm_a:
            a_map[key] = row
        elif row["arm"] == arm_b:
            b_map[key] = row

    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    return [(a_map[k], b_map[k]) for k in common]


def _pick_simple_arm(arms: Sequence[str]) -> str:
    for arm in arms:
        if "bd" in arm:
            return arm
    return arms[0]


def decide_winner(
    decision_rows: Sequence[Dict[str, Any]],
    arm_a: str,
    arm_b: str,
    num_bootstrap: int,
    bootstrap_seed: int,
    threshold: float,
) -> Dict[str, Any]:
    pairs = _pair_rows(decision_rows, arm_a=arm_a, arm_b=arm_b)

    pairwise: Dict[str, Dict[str, Optional[float]]] = {}
    for mi, metric in enumerate([
        "m2_lqr_feasibility_rate",
        "m3_closed_loop_stability_rate",
        "m4_closed_loop_cost_reduction",
    ]):
        diffs = []
        for a, b in pairs:
            av = a.get(metric)
            bv = b.get(metric)
            if av is None or bv is None:
                continue
            diffs.append(float(av) - float(bv))

        mean, ci_low, ci_high = _paired_bootstrap_diff_ci(
            diffs,
            num_bootstrap=num_bootstrap,
            seed=bootstrap_seed + 1003 * mi,
        )
        pairwise[metric] = {
            "diff_mean": mean,
            "diff_ci_low": ci_low,
            "diff_ci_high": ci_high,
            "ci_excludes_zero": _ci_excludes_zero(ci_low, ci_high),
            "n_pairs": len(diffs),
        }

    m5_by_arm = _arm_m5_sensitivity(decision_rows)
    m5_a = m5_by_arm.get(arm_a)
    m5_b = m5_by_arm.get(arm_b)

    winner: Optional[str] = None
    decided_by = "none"

    # Lexicographic decision rule.
    m2 = pairwise["m2_lqr_feasibility_rate"]
    if m2["ci_excludes_zero"] and m2["diff_mean"] is not None:
        winner = arm_a if m2["diff_mean"] > 0 else arm_b
        decided_by = "M2"

    if winner is None:
        m3 = pairwise["m3_closed_loop_stability_rate"]
        if m3["ci_excludes_zero"] and m3["diff_mean"] is not None:
            winner = arm_a if m3["diff_mean"] > 0 else arm_b
            decided_by = "M3"

    if winner is None:
        m4 = pairwise["m4_closed_loop_cost_reduction"]
        if m4["ci_excludes_zero"] and m4["diff_mean"] is not None:
            winner = arm_a if m4["diff_mean"] > 0 else arm_b
            decided_by = "M4"

    if winner is None and m5_a is not None and m5_b is not None and not math.isclose(m5_a, m5_b):
        winner = arm_a if m5_a < m5_b else arm_b
        decided_by = "M5"

    # Minimum practical threshold: +10% absolute gain in M2 or M3 with CI excluding 0.
    passes_threshold = False
    m2_gain = pairwise["m2_lqr_feasibility_rate"]
    m3_gain = pairwise["m3_closed_loop_stability_rate"]
    if winner is not None:
        sign = 1.0 if winner == arm_a else -1.0

        if (
            m2_gain["diff_mean"] is not None
            and m2_gain["ci_excludes_zero"]
            and sign * m2_gain["diff_mean"] >= threshold
        ):
            passes_threshold = True
        if (
            m3_gain["diff_mean"] is not None
            and m3_gain["ci_excludes_zero"]
            and sign * m3_gain["diff_mean"] >= threshold
        ):
            passes_threshold = True

    clear_winner = winner if (winner is not None and passes_threshold) else None

    arms = [arm_a, arm_b]
    if clear_winner is None:
        fallback = _pick_simple_arm(arms)
        decision_label = "no_clear_winner"
        final_choice = fallback
    else:
        fallback = None
        decision_label = "winner"
        final_choice = clear_winner

    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "paired_samples": len(pairs),
        "pairwise": pairwise,
        "m5_sensitivity": {arm_a: m5_a, arm_b: m5_b},
        "lexicographic_winner": winner,
        "decided_by": decided_by,
        "passes_practical_threshold": passes_threshold,
        "decision": decision_label,
        "final_choice": final_choice,
        "fallback_simpler_arm": fallback,
        "threshold": threshold,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return

    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_final_decision_md(
    output_path: Path,
    decision_result: Dict[str, Any],
    arm_summary: List[Dict[str, Any]],
    decision_stage: int,
    decision_system: str,
) -> None:
    lines: List[str] = []
    lines.append("# LQR Decision")
    lines.append("")
    lines.append(f"Decision stage: {decision_stage}")
    lines.append(f"System: {decision_system}")
    lines.append("")

    if decision_result["decision"] == "winner":
        lines.append(f"Final choice: **{decision_result['final_choice']}**")
        lines.append(f"Decided by: {decision_result['decided_by']}")
    else:
        lines.append("Final choice: **no clear winner**")
        lines.append(
            "Fallback (simpler arm): "
            f"**{decision_result['final_choice']}**"
        )
    lines.append("")

    lines.append("## Primary Pairwise Metrics")
    lines.append("")
    lines.append("| Metric (arm_a - arm_b) | Mean Diff | 95% CI | CI excludes 0 |")
    lines.append("|---|---:|---:|:---:|")

    for metric in [
        "m2_lqr_feasibility_rate",
        "m3_closed_loop_stability_rate",
        "m4_closed_loop_cost_reduction",
    ]:
        item = decision_result["pairwise"][metric]
        mean = item.get("diff_mean")
        lo = item.get("diff_ci_low")
        hi = item.get("diff_ci_high")
        mean_s = "--" if mean is None else f"{mean:.4f}"
        ci_s = "--" if lo is None or hi is None else f"[{lo:.4f}, {hi:.4f}]"
        excl = "yes" if item.get("ci_excludes_zero") else "no"
        lines.append(f"| {metric} | {mean_s} | {ci_s} | {excl} |")

    lines.append("")
    lines.append("## Robustness (M5)")
    lines.append("")
    m5 = decision_result.get("m5_sensitivity", {})
    lines.append(f"- arm_a ({decision_result['arm_a']}): {m5.get(decision_result['arm_a'])}")
    lines.append(f"- arm_b ({decision_result['arm_b']}): {m5.get(decision_result['arm_b'])}")
    lines.append("")

    lines.append("## Arm-Level Summary")
    lines.append("")
    lines.append("| arm | n_runs | M2 mean | M3 mean | M4 mean |")
    lines.append("|---|---:|---:|---:|---:|")

    for row in arm_summary:
        if row["target_size"] != "all" or row["b_proxy"] != "all":
            continue
        m2 = row.get("m2_lqr_feasibility_rate_mean")
        m3 = row.get("m3_closed_loop_stability_rate_mean")
        m4 = row.get("m4_closed_loop_cost_reduction_mean")
        lines.append(
            f"| {row['arm']} | {row['n_runs']} | "
            f"{('--' if m2 is None else f'{m2:.4f}')} | "
            f"{('--' if m3 is None else f'{m3:.4f}')} | "
            f"{('--' if m4 is None else f'{m4:.4f}')} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LQR decision results and apply decision rule")
    parser.add_argument("--base_dir", type=str, required=True, help="Base sweep directory to scan")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for decision artifacts")
    parser.add_argument("--decision_stage", type=int, default=2, help="Stage to use for final decision")
    parser.add_argument("--decision_system", type=str, default="lyapunov")
    parser.add_argument("--arms", type=str, default=None,
                        help="Comma-separated finalist arms (default: infer top-2 by run count)")
    parser.add_argument("--num_bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Minimum absolute gain threshold on M2 or M3")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_run_rows(base_dir)
    if not rows:
        raise ValueError(f"No lqr_readiness_summary.json files found under {base_dir}")

    decision_rows = [
        r for r in rows
        if int(r["stage"]) == int(args.decision_stage)
        and str(r["system"]) == str(args.decision_system)
    ]
    if not decision_rows:
        raise ValueError(
            f"No rows for decision stage={args.decision_stage} and system={args.decision_system}"
        )

    if args.arms is not None:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        counts: Dict[str, int] = defaultdict(int)
        for r in decision_rows:
            counts[r["arm"]] += 1
        arms = [k for k, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]

    if len(arms) < 2:
        raise ValueError("Need at least two arms for pairwise decision.")
    arm_a, arm_b = arms[0], arms[1]

    summary_table = build_summary_table(
        rows=decision_rows,
        num_bootstrap=args.num_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    decision_result = decide_winner(
        decision_rows=decision_rows,
        arm_a=arm_a,
        arm_b=arm_b,
        num_bootstrap=args.num_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        threshold=args.threshold,
    )

    summary_csv = output_dir / "summary_decision_table.csv"
    write_csv(summary_csv, summary_table)

    payload = {
        "base_dir": str(base_dir),
        "output_dir": str(output_dir),
        "decision_stage": args.decision_stage,
        "decision_system": args.decision_system,
        "num_runs_total": len(rows),
        "num_runs_decision_subset": len(decision_rows),
        "finalist_arms": [arm_a, arm_b],
        "summary_table": summary_table,
        "decision_result": decision_result,
    }

    summary_json = output_dir / "lqr_readiness_summary.json"
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    final_md = output_dir / "final_decision.md"
    render_final_decision_md(
        output_path=final_md,
        decision_result=decision_result,
        arm_summary=summary_table,
        decision_stage=args.decision_stage,
        decision_system=args.decision_system,
    )

    print(f"Saved summary table: {summary_csv}")
    print(f"Saved summary json:  {summary_json}")
    print(f"Saved final decision:{final_md}")


if __name__ == "__main__":
    main()
