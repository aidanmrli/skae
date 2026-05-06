#!/usr/bin/env python3
"""Select the best stage-2 routed local-map setting from aggregation output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


CONTROLLED_TO_DYSTS_ROOT = {
    "lista_dense_signsplit_p256_hardinit_basin_partition": "lista",
    "lista_blockdiag_signsplit_hardinit_basin_partition": "lista_bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", help="routed_forecasting_iqm_summary.csv from stage-2 aggregation")
    parser.add_argument("--dataset", default="multibasin")
    parser.add_argument(
        "--root_labels",
        default=",".join(CONTROLLED_TO_DYSTS_ROOT),
        help="comma-separated candidate controlled root labels",
    )
    parser.add_argument("--horizons", default="100,500,1000")
    parser.add_argument("--min_systems", type=int, default=15)
    parser.add_argument(
        "--score",
        default="geomean_mse",
        choices=["geomean_mse", "mean_mse", "mean_ratio", "h1000_mse"],
    )
    parser.add_argument("--json_output", default="")
    parser.add_argument("--env_output", default="")
    return parser.parse_args()


def parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_horizons(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_items(raw)]


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def geometric_mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if float(value) > 0 and math.isfinite(float(value))]
    if not clean:
        return float("inf")
    return float(math.exp(sum(math.log(value) for value in clean) / len(clean)))


def arithmetic_mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return float("inf")
    return float(sum(clean) / len(clean))


def shell_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def main() -> None:
    args = parse_args()
    candidate_roots = set(parse_csv_items(args.root_labels))
    horizons = parse_horizons(args.horizons)
    grouped: dict[tuple[str, int, str], dict[int, dict[str, object]]] = defaultdict(dict)

    with Path(args.summary_csv).open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("dataset") != args.dataset:
                continue
            root = str(row.get("root_label", ""))
            if root not in candidate_roots:
                continue
            if str(row.get("local_map_source", "")) != "stage2_rollout_trained":
                continue
            horizon = int(float(row.get("horizon", 0)))
            if horizon not in horizons:
                continue
            n_systems = int(float(row.get("n_systems", 0)))
            if n_systems < int(args.min_systems):
                continue
            key = (
                root,
                int(float(row.get("reencode_period", 0))),
                str(row.get("route_freeze_mode", "")),
            )
            grouped[key][horizon] = dict(row)

    candidates = []
    for (root, period, freeze_mode), by_horizon in grouped.items():
        if any(horizon not in by_horizon for horizon in horizons):
            continue
        routed_mse = [finite_float(by_horizon[h]["routed_iqm_mean"]) for h in horizons]
        ratios = [finite_float(by_horizon[h]["ratio_routed_over_best_periodic"]) for h in horizons]
        if any(value is None for value in routed_mse):
            continue
        if args.score == "geomean_mse":
            score = geometric_mean(value for value in routed_mse if value is not None)
        elif args.score == "mean_mse":
            score = arithmetic_mean(value for value in routed_mse if value is not None)
        elif args.score == "mean_ratio":
            if any(value is None for value in ratios):
                continue
            score = arithmetic_mean(value for value in ratios if value is not None)
        else:
            score = finite_float(by_horizon[1000]["routed_iqm_mean"])
            if score is None:
                continue
        candidates.append(
            {
                "score": score,
                "score_name": args.score,
                "root_label": root,
                "dysts_root_label": CONTROLLED_TO_DYSTS_ROOT.get(root, root),
                "reencode_period": period,
                "route_freeze_mode": freeze_mode,
                "horizons": horizons,
                "routed_mse_by_horizon": {
                    str(horizon): finite_float(by_horizon[horizon]["routed_iqm_mean"])
                    for horizon in horizons
                },
                "best_periodic_mse_by_horizon": {
                    str(horizon): finite_float(by_horizon[horizon]["best_periodic_iqm_mean"])
                    for horizon in horizons
                },
                "ratio_over_best_by_horizon": {
                    str(horizon): finite_float(by_horizon[horizon]["ratio_routed_over_best_periodic"])
                    for horizon in horizons
                },
                "wins_over_best_by_horizon": {
                    str(horizon): int(float(by_horizon[horizon]["systems_routed_better_than_best_periodic"]))
                    for horizon in horizons
                },
            }
        )

    if not candidates:
        raise SystemExit("No complete candidate settings found")

    candidates.sort(key=lambda item: (float(item["score"]), str(item["root_label"]), int(item["reencode_period"])))
    best = candidates[0]
    payload = {"best": best, "candidates": candidates}

    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")
    if args.env_output:
        path = Path(args.env_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"BEST_ROOT_LABEL={shell_quote(best['root_label'])}",
            f"BEST_DYSTS_ROOT_LABEL={shell_quote(best['dysts_root_label'])}",
            f"BEST_REENCODE_PERIOD={shell_quote(best['reencode_period'])}",
            f"BEST_ROUTE_FREEZE_MODE={shell_quote(best['route_freeze_mode'])}",
            f"BEST_SCORE={shell_quote(best['score'])}",
            f"BEST_SCORE_NAME={shell_quote(best['score_name'])}",
        ]
        path.write_text("\n".join(lines) + "\n")

    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
