#!/usr/bin/env python
"""Select BD* (bd_c1 vs bd_c2) from Stage 1 LQR-readiness outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ARMS = ("bd_c1", "bd_c2")


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _safe_mean(vals: List[Optional[float]]) -> Optional[float]:
    usable = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not usable:
        return None
    return float(np.mean(usable))


def _score_m5(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Lower is better: average range across B_proxy per (seed, target_size)."""
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    for r in rows:
        m2 = _to_float(r.get("m2"))
        m3 = _to_float(r.get("m3"))
        if m2 is None or m3 is None:
            continue
        key = (int(r["seed"]), int(r["target_size"]))
        grouped[key].append(0.5 * (m2 + m3))

    ranges = []
    for vals in grouped.values():
        if len(vals) >= 2:
            ranges.append(float(max(vals) - min(vals)))
    if not ranges:
        return None
    return float(np.mean(ranges))


def collect_stage1_rows(base_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for summary_path in sorted(base_dir.rglob("lqr_readiness_summary.json")):
        try:
            with open(summary_path) as f:
                data = json.load(f)
        except Exception:
            continue

        meta = data.get("metadata", {})
        agg = data.get("aggregate_metrics", {})

        if int(meta.get("stage", -1)) != 1:
            continue
        if str(meta.get("system", "")) != "lyapunov":
            continue
        arm = str(meta.get("arm", ""))
        if arm not in ARMS:
            continue

        rows.append(
            {
                "summary_path": str(summary_path),
                "arm": arm,
                "seed": int(meta.get("run_seed", -1)),
                "target_size": int(meta.get("target_size", -1)),
                "b_proxy": int(meta.get("b_proxy", -1)),
                "m2": _to_float(agg.get("m2_lqr_feasibility_rate")),
                "m3": _to_float(agg.get("m3_closed_loop_stability_rate")),
                "m4": _to_float(agg.get("m4_closed_loop_cost_reduction")),
            }
        )

    return rows


def choose_bd_star(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    summary: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        group = by_arm.get(arm, [])
        summary[arm] = {
            "n_runs": len(group),
            "m2_mean": _safe_mean([r.get("m2") for r in group]),
            "m3_mean": _safe_mean([r.get("m3") for r in group]),
            "m4_mean": _safe_mean([r.get("m4") for r in group]),
            "m5_sensitivity": _score_m5(group),
        }

    # Lexicographic: M2, M3, M4, then lower M5.
    eps = 1e-9

    def cmp(a: str, b: str, key: str) -> int:
        av = summary[a].get(key)
        bv = summary[b].get(key)
        if av is None and bv is None:
            return 0
        if av is None:
            return -1
        if bv is None:
            return 1
        if av > bv + eps:
            return 1
        if bv > av + eps:
            return -1
        return 0

    winner = None
    decided_by = "none"

    c = cmp("bd_c1", "bd_c2", "m2_mean")
    if c != 0:
        winner = "bd_c1" if c > 0 else "bd_c2"
        decided_by = "m2"

    if winner is None:
        c = cmp("bd_c1", "bd_c2", "m3_mean")
        if c != 0:
            winner = "bd_c1" if c > 0 else "bd_c2"
            decided_by = "m3"

    if winner is None:
        c = cmp("bd_c1", "bd_c2", "m4_mean")
        if c != 0:
            winner = "bd_c1" if c > 0 else "bd_c2"
            decided_by = "m4"

    if winner is None:
        m51 = summary["bd_c1"].get("m5_sensitivity")
        m52 = summary["bd_c2"].get("m5_sensitivity")
        if m51 is not None and m52 is not None and not math.isclose(m51, m52, rel_tol=0.0, abs_tol=eps):
            winner = "bd_c1" if m51 < m52 else "bd_c2"
            decided_by = "m5"

    if winner is None:
        winner = "bd_c1"
        decided_by = "tie_default"

    return {
        "bd_star": winner,
        "decided_by": decided_by,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose BD* from Stage 1 outputs")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--expected_runs", type=int, default=24,
                        help="Expected Stage 1 runs for BD arms (2 arms x 3 B_proxy x 4 seeds)")
    parser.add_argument("--min_rows", type=int, default=20,
                        help="Fail if fewer than this many BD Stage-1 summaries are found")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rows = collect_stage1_rows(base_dir)
    if len(rows) < int(args.min_rows):
        raise SystemExit(
            f"Not enough Stage-1 BD rows to decide BD*: found {len(rows)}, "
            f"required at least {args.min_rows}"
        )

    result = choose_bd_star(rows)
    result["num_rows_used"] = len(rows)
    result["expected_runs"] = int(args.expected_runs)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
