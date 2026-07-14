#!/usr/bin/env python3
"""Plot support-coordinate intervention trajectories from saved initial states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from tools.evaluate_support_coordinate_interventions import (
    REDUCER,
    _configure_paper_style,
    _make_drop_z,
    _make_random_support_z,
    _parse_support_definition,
    _plot_trajectories,
    _rollout_decode,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--plot_format", default="pdf,png")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max_points", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--intervention",
        default=None,
        help="Condition to plot. Defaults to run_summary worst_intervention_condition.",
    )
    parser.add_argument(
        "--filename_stem",
        default="fig_support_coordinate_trajectories",
        help="Output filename stem.",
    )
    return parser.parse_args()


def _read_initial_points(path: Path, *, max_points: int) -> tuple[torch.Tensor, List[List[int]]]:
    xs: List[List[float]] = []
    top_indices: List[List[int]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            xs.append([float(item) for item in str(row["x0"]).split()])
            top_indices.append([int(item) for item in str(row["top_indices"]).split()])
            if len(xs) >= max_points:
                break
    if not xs:
        raise RuntimeError(f"No initial states found in {path}")
    return torch.tensor(xs, dtype=torch.float32), top_indices


def _true_future(env, x0: torch.Tensor, *, steps: int) -> torch.Tensor:
    current = x0
    out: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(steps):
            current = env.step(current)
            out.append(current.detach().cpu())
    return torch.stack(out, dim=1)


def _make_intervention_z(
    *,
    condition: str,
    z0: torch.Tensor,
    top_indices: Sequence[Sequence[int]],
    support_definition: str,
) -> torch.Tensor:
    if condition.startswith("drop_top_"):
        drop_count = int(condition.rsplit("_", 1)[1])
        return _make_drop_z(z0, top_indices, drop_count)
    if condition.startswith("random_support_"):
        support_scheme, support_value = _parse_support_definition(support_definition)
        active_mask = REDUCER._support_mask(
            z0.detach().cpu().numpy(),
            scheme=support_scheme,
            value=support_value,
        )
        repeat = int(condition.rsplit("_", 1)[1])
        z_random, _moves = _make_random_support_z(
            z0,
            active_mask,
            rng=np.random.default_rng(123 + repeat),
        )
        return z_random
    raise ValueError(f"Unsupported intervention condition for trajectory plot: {condition}")


def main() -> None:
    args = _parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_formats = [item.strip().lstrip(".") for item in args.plot_format.split(",") if item.strip()]

    summary: Dict[str, object] = json.loads((result_dir / "run_summary.json").read_text())
    condition = args.intervention or str(summary["worst_intervention_condition"])
    checkpoint_path = Path(str(summary["checkpoint_path"]))
    system_key = str(summary["system_key"])
    support_definition = str(summary["support_definition"])

    _configure_paper_style()
    _cfg, env, model = REDUCER._load_checkpoint_model(checkpoint_path, system_key, args.device)
    x0_cpu, top_indices = _read_initial_points(
        result_dir / "initial_points.csv",
        max_points=int(args.max_points),
    )
    x0 = x0_cpu.to(args.device)
    with torch.no_grad():
        z0 = model.encode(x0.to(args.device, dtype=torch.float32)).detach()
    z_intervention = _make_intervention_z(
        condition=condition,
        z0=z0,
        top_indices=top_indices,
        support_definition=support_definition,
    )
    condition_predictions = {
        "baseline": _rollout_decode(model, z0, max_horizon=int(args.steps)),
        condition: _rollout_decode(model, z_intervention, max_horizon=int(args.steps)),
    }
    true_future = _true_future(env, x0_cpu, steps=int(args.steps))
    _plot_trajectories(
        output_dir=output_dir,
        plot_formats=plot_formats,
        env=env,
        x0=x0_cpu,
        true_future=true_future,
        condition_predictions=condition_predictions,
        worst_condition=condition,
    )
    for ext in plot_formats:
        src = output_dir / f"trajectory_vector_field_worst_intervention.{ext}"
        dst = output_dir / f"{args.filename_stem}.{ext}"
        if src.exists() and src != dst:
            src.replace(dst)
    print(f"Wrote trajectory plot for {condition} to {output_dir}")


if __name__ == "__main__":
    main()
