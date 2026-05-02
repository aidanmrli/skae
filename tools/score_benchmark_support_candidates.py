#!/usr/bin/env python3
"""Score visual support-family/basin agreement for fixed-17 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    TRANSITION_RICH_BASIN_PARTITION_SYSTEMS,
    get_transition_rich_basin_count,
)
from skae.data import _infer_reset_bounds
from tools.make_benchmark_support_dysts_composite import (
    DEFAULT_INTERPRETABILITY_CSV,
    ROOT_LISTA_SUPPORT,
    SUPPORT_PANELS,
    _basin_labels_for_states,
    _encode_latents,
    _family_to_dominant_basin,
    _find_run_dir,
    _grid_states,
    _known_attractor_centers,
    _load_model_and_env,
    _read_rows,
    _topk_support_mask,
)
from tools.reduce_transition_rich_interpretability_metrics import support_family_labels


DEFAULT_OUTPUT = Path(
    "docs/figures/neurips_paper_2026/fig_benchmark_support_grid_agreement_candidates.json"
)


def _bounds_for_system(env: Any, system: str) -> tuple[tuple[float, float], tuple[float, float], str]:
    display_bounds = {spec.system: (spec.xlim, spec.ylim) for spec in SUPPORT_PANELS}
    if system in display_bounds:
        xlim, ylim = display_bounds[system]
        return xlim, ylim, "current_display_override"

    base_env = getattr(env, "unwrapped", env)
    try:
        low, high = _infer_reset_bounds(base_env)
        return (float(low[0]), float(high[0])), (float(low[1]), float(high[1])), "inferred_reset_bounds"
    except Exception:
        init_range = float(getattr(base_env, "init_range", 3.5))
        return (-init_range, init_range), (-init_range, init_range), "init_range_fallback"


def score_candidates(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_rows(args.interpretability_csv)
    requested_systems = {
        item.strip()
        for item in str(args.systems).split(",")
        if item.strip()
    }
    scores = []
    for item in TRANSITION_RICH_BASIN_PARTITION_SYSTEMS:
        system = item.system_key
        if requested_systems and system not in requested_systems:
            continue
        print(f"Scoring support candidate: {system}", flush=True)
        run_dir = _find_run_dir(
            rows,
            root_label=args.root_label,
            system=system,
            seed=args.seed,
        )
        model, env = _load_model_and_env(run_dir, system, args.device)
        xlim, ylim, bounds_source = _bounds_for_system(env, system)
        _xx, _yy, states = _grid_states(xlim, ylim, args.grid_points)
        if args.skip_endpoint_fallback and not hasattr(env, "basin_label"):
            basin_count = int(get_transition_rich_basin_count(system))
            centers = _known_attractor_centers(env, states, basin_count)
            if centers is None:
                print(f"Skipping {system}: no explicit basin map or attractor-center convention", flush=True)
                continue
        basin_labels_t, basin_label_source = _basin_labels_for_states(
            env,
            system,
            states,
            endpoint_rollout_steps=args.endpoint_rollout_steps,
        )
        basin_labels = basin_labels_t.numpy()
        latents = _encode_latents(model, states, args.device)
        support_mask = _topk_support_mask(latents, args.topk)
        families = support_family_labels(
            support_mask[:, None, :],
            min_jaccard=args.family_jaccard,
        ).reshape(-1)
        support_basin, family_map = _family_to_dominant_basin(families, basin_labels)
        agreement = float(np.mean(support_basin == basin_labels))
        scores.append(
            {
                "system": system,
                "agreement": agreement,
                "support_family_count": int(len(set(families.tolist()))),
                "basin_count": int(
                    max(get_transition_rich_basin_count(system), int(basin_labels.max()) + 1)
                ),
                "basin_label_source": basin_label_source,
                "bounds_source": bounds_source,
                "xlim": [float(xlim[0]), float(xlim[1])],
                "ylim": [float(ylim[0]), float(ylim[1])],
                "run_dir": str(run_dir),
                "family_to_dominant_basin": family_map,
            }
        )

    scores = sorted(scores, key=lambda row: row["agreement"], reverse=True)
    output = {
        "root_label": args.root_label,
        "seed": int(args.seed),
        "support_definition": f"topk:{args.topk}",
        "family_jaccard": float(args.family_jaccard),
        "grid_points": int(args.grid_points),
        "agreement_definition": (
            "Fraction of grid points whose evaluation basin label equals the "
            "dominant evaluation basin assigned post hoc to that point's "
            "learned support family."
        ),
        "rows": scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpretability-csv", type=Path, default=DEFAULT_INTERPRETABILITY_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root-label", default=ROOT_LISTA_SUPPORT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grid-points", type=int, default=84)
    parser.add_argument("--endpoint-rollout-steps", type=int, default=360)
    parser.add_argument("--family-jaccard", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument(
        "--skip-endpoint-fallback",
        action="store_true",
        help="skip systems that would need endpoint rollouts to construct basin labels",
    )
    return parser.parse_args()


def main() -> None:
    output = score_candidates(parse_args())
    print(json.dumps(output["rows"], indent=2))


if __name__ == "__main__":
    main()
