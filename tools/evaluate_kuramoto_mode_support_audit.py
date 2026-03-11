#!/usr/bin/env python
"""Run the Kuramoto mode-support audit for one checkpoint and sampling regime."""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch

from skae.basin_utils import BasinLabeledDataset
from skae.config import Config
from skae.data import make_env
from skae.model import make_model
from tools.evaluate_support_uniqueness import (
    compute_cosine_basin_similarity,
    compute_support_uniqueness,
)


def _parse_csv_strings(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _attach_cosine_metrics(result, cosine_metrics: Dict[str, float]) -> Dict[str, object]:
    result.mean_intra_basin_cosine = cosine_metrics["mean_intra_basin_cosine"]
    result.mean_inter_basin_cosine = cosine_metrics["mean_inter_basin_cosine"]
    result.cosine_separation_score = cosine_metrics["cosine_separation_score"]
    return asdict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--system", default="kuramoto")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sampling_strategy", default="random", choices=("random", "balanced"))
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectories_per_basin", type=int, default=None)
    parser.add_argument("--target_raw_labels_csv", default="")
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--long_rollout_steps", type=int, default=5000)
    parser.add_argument("--support_threshold", type=float, default=1e-3)
    parser.add_argument("--support_modes_csv", default="mean,majority,modal")
    parser.add_argument("--threshold_sweep_modes_csv", default="mean,modal")
    parser.add_argument("--thresholds_csv", default="1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1")
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint": args.checkpoint,
        "system": args.system,
        "sampling_strategy": args.sampling_strategy,
        "num_trajectories": args.num_trajectories,
        "trajectories_per_basin": args.trajectories_per_basin,
        "target_raw_labels": _parse_csv_ints(args.target_raw_labels_csv),
        "trajectory_length": args.trajectory_length,
        "long_rollout_steps": args.long_rollout_steps,
        "support_threshold": args.support_threshold,
        "support_modes": _parse_csv_strings(args.support_modes_csv),
        "threshold_sweep_modes": _parse_csv_strings(args.threshold_sweep_modes_csv),
        "thresholds": _parse_csv_floats(args.thresholds_csv),
        "max_attempts": args.max_attempts,
        "seed": args.seed,
        "device": args.device,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    analysis_path = output_dir / "analysis_results.json"

    try:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        cfg = Config.from_dict(checkpoint["config"])
        cfg.ENV.ENV_NAME = args.system

        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(args.device)
        model.eval()

        dataset = BasinLabeledDataset(
            system=args.system,
            cfg=cfg,
            num_trajectories=args.num_trajectories,
            trajectory_length=args.trajectory_length,
            long_rollout_steps=args.long_rollout_steps,
            seed=args.seed,
            sampling_strategy=args.sampling_strategy,
            target_raw_labels=_parse_csv_ints(args.target_raw_labels_csv),
            trajectories_per_basin=args.trajectories_per_basin,
            max_attempts=args.max_attempts,
        )

        cosine_metrics = compute_cosine_basin_similarity(model, dataset, args.device)

        primary_results = {}
        for support_mode in _parse_csv_strings(args.support_modes_csv):
            result = compute_support_uniqueness(
                model,
                dataset,
                device=args.device,
                support_threshold=args.support_threshold,
                support_mode=support_mode,
            )
            primary_results[support_mode] = _attach_cosine_metrics(result, cosine_metrics)

        threshold_sweeps = {}
        thresholds = _parse_csv_floats(args.thresholds_csv)
        for support_mode in _parse_csv_strings(args.threshold_sweep_modes_csv):
            sweep_rows = []
            for threshold in thresholds:
                result = compute_support_uniqueness(
                    model,
                    dataset,
                    device=args.device,
                    support_threshold=threshold,
                    support_mode=support_mode,
                )
                sweep_rows.append(_attach_cosine_metrics(result, cosine_metrics))
            threshold_sweeps[support_mode] = sweep_rows

        payload = {
            "status": "ok",
            "system_name": args.system,
            "model_name": type(model).__name__,
            "num_trajectories": len(dataset.trajectories),
            "num_basins": dataset.num_basins,
            "basin_names": list(dataset.basin_names),
            "sampling_strategy": dataset.sampling_strategy,
            "raw_basin_labels": list(dataset.raw_basin_labels),
            "raw_to_mapped_label": dict(dataset.raw_to_mapped_label),
            "raw_basin_distribution": dict(dataset.raw_basin_distribution),
            "mapped_basin_distribution": dict(dataset.mapped_basin_distribution),
            "support_threshold": args.support_threshold,
            "support_modes": _parse_csv_strings(args.support_modes_csv),
            "threshold_sweep_modes": _parse_csv_strings(args.threshold_sweep_modes_csv),
            "thresholds": thresholds,
            "cosine_metrics": cosine_metrics,
            "primary_results": primary_results,
            "threshold_sweeps": threshold_sweeps,
        }
    except Exception as exc:
        payload = {
            "status": "failed",
            "system_name": args.system,
            "sampling_strategy": args.sampling_strategy,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        analysis_path.write_text(json.dumps(payload, indent=2) + "\n")
        raise

    analysis_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Saved Kuramoto mode-support audit to {analysis_path} "
        f"(status={payload['status']})"
    )


if __name__ == "__main__":
    main()
