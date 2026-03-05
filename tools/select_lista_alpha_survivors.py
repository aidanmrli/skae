#!/usr/bin/env python3
"""Select LISTA alpha survivors from a gate-stage summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            out = float(raw)
        except ValueError:
            return None
        if math.isfinite(out):
            return out
    return None


def _parse_alpha(arm_name: str) -> Optional[float]:
    raw = arm_name
    if raw.startswith("alpha_"):
        raw = raw[len("alpha_"):]
    raw = raw.replace("p", ".")
    return _safe_float(raw)


def _alpha_token(arm_name: str) -> Optional[str]:
    raw = arm_name
    if raw.startswith("alpha_"):
        raw = raw[len("alpha_"):]
    raw = raw.replace("p", ".").strip()
    return raw if _safe_float(raw) is not None else None


def _score_candidate(
    stats: Dict[str, object],
    alpha: Optional[float],
    target_low: float,
    target_high: float,
    min_runs: int,
) -> Tuple[float, float, float, float, float, float]:
    n_runs = int(stats.get("n", 0) or 0)
    h1000 = _safe_float(stats.get("h1000_bp_mean"))
    h500 = _safe_float(stats.get("h500_bp_mean"))
    sparsity = _safe_float(stats.get("sparsity_ratio_final_median"))

    in_band = 0.0
    distance_to_band = 10.0
    if sparsity is not None:
        if target_low <= sparsity <= target_high:
            in_band = 0.0
            distance_to_band = 0.0
        else:
            in_band = 1.0
            distance_to_band = min(abs(sparsity - target_low), abs(sparsity - target_high))
    else:
        in_band = 1.0

    missing_runs = 1.0 if n_runs < min_runs else 0.0
    h1000_score = h1000 if h1000 is not None else float("inf")
    h500_score = h500 if h500 is not None else float("inf")
    alpha_score = alpha if alpha is not None else float("inf")

    return (
        missing_runs,
        in_band,
        h1000_score,
        h500_score,
        distance_to_band,
        alpha_score,
    )


def _load_summary(path: Path) -> Dict[str, Dict[str, object]]:
    data = json.loads(path.read_text())
    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Summary payload missing 'summary': {path}")
    out: Dict[str, Dict[str, object]] = {}
    for arm, stats in summary.items():
        if isinstance(arm, str) and isinstance(stats, dict):
            out[arm] = stats
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select survivor alpha values from gate-stage LISTA sweep summary."
    )
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_survivors", type=int, default=3)
    parser.add_argument("--target_low", type=float, default=0.7)
    parser.add_argument("--target_high", type=float, default=0.9)
    parser.add_argument("--min_runs", type=int, default=3)
    args = parser.parse_args()

    if args.max_survivors < 1:
        raise ValueError("--max_survivors must be >= 1")
    if args.target_low > args.target_high:
        raise ValueError("--target_low must be <= --target_high")

    summary = _load_summary(Path(args.summary_json))
    candidates: List[Dict[str, object]] = []
    for arm, stats in summary.items():
        alpha = _parse_alpha(arm)
        if alpha is None:
            continue
        score = _score_candidate(
            stats=stats,
            alpha=alpha,
            target_low=args.target_low,
            target_high=args.target_high,
            min_runs=args.min_runs,
        )
        candidates.append(
            {
                "arm": arm,
                "alpha": alpha,
                "n": int(stats.get("n", 0) or 0),
                "h1000_bp_mean": _safe_float(stats.get("h1000_bp_mean")),
                "h500_bp_mean": _safe_float(stats.get("h500_bp_mean")),
                "sparsity_ratio_final_median": _safe_float(stats.get("sparsity_ratio_final_median")),
                "score": list(score),
            }
        )

    candidates_sorted = sorted(candidates, key=lambda row: tuple(row["score"]))  # type: ignore[arg-type]
    selected = candidates_sorted[: args.max_survivors]

    selected_arms = [str(row["arm"]) for row in selected]
    selected_alpha_tokens = [_alpha_token(arm) for arm in selected_arms]
    selected_alpha_tokens = [token for token in selected_alpha_tokens if isinstance(token, str)]
    selected_alphas = [row["alpha"] for row in selected if isinstance(row.get("alpha"), float)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not candidates_sorted:
        raise ValueError("No alpha-like arms found in summary.")

    payload = {
        "summary_json": str(Path(args.summary_json)),
        "target_sparsity_band": [args.target_low, args.target_high],
        "min_runs": args.min_runs,
        "max_survivors": args.max_survivors,
        "candidates_ranked": candidates_sorted,
        "selected_arms": selected_arms,
        "selected_alpha_tokens": selected_alpha_tokens,
        "selected_alphas": selected_alphas,
    }

    json_path = output_dir / "lista_alpha_survivors.json"
    txt_path = output_dir / "lista_alpha_survivors.txt"
    csv_path = output_dir / "lista_alpha_survivors.csv"

    json_path.write_text(json.dumps(payload, indent=2))
    txt_path.write_text("\n".join(selected_arms) + ("\n" if selected_arms else ""))
    csv_path.write_text(",".join(selected_alpha_tokens) + "\n")

    print(f"Wrote survivor JSON: {json_path}")
    print(f"Wrote survivor arms: {txt_path}")
    print(f"Wrote survivor alphas CSV: {csv_path}")
    if selected_arms:
        print(f"Selected survivors: {', '.join(selected_arms)}")
    else:
        print("Warning: no survivors selected.")


if __name__ == "__main__":
    main()
