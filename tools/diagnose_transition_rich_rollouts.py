#!/usr/bin/env python3
"""Diagnose transition-rich forecast rollouts from saved evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.benchmarks.transition_rich_basin_partition_manifest import get_transition_rich_basin_count
from skae.config import Config
from skae.data import make_env
from skae.model import make_model
from skae.transition_diagnostics import compare_label_sequences


def _ensure_matplotlib():
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")


def _load_rollout_artifact(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if "predictions" not in payload or "true_sequences" not in payload:
        raise ValueError(f"{path} is not a rollout artifact produced by skae.evaluation.")
    return payload


def _infer_basin_count(system: str, explicit: Optional[int]) -> int:
    if explicit is not None and explicit > 0:
        return int(explicit)
    try:
        return int(get_transition_rich_basin_count(system))
    except KeyError as exc:
        raise ValueError(
            f"No fixed benchmark basin count is registered for '{system}'. "
            "Pass --expected_num_basins explicitly."
        ) from exc


def _kmeans_centers(points: torch.Tensor, num_centers: int, num_iters: int = 25) -> torch.Tensor:
    """Small deterministic k-means used for label fallback on unlabeled systems."""
    if points.ndim != 2:
        raise ValueError("points must have shape [N, dim]")
    if num_centers <= 0:
        raise ValueError("num_centers must be positive")
    if points.shape[0] < num_centers:
        raise ValueError("Need at least as many points as requested centers")

    centers = [points[0]]
    while len(centers) < num_centers:
        current = torch.stack(centers, dim=0)
        dists = torch.cdist(points, current)
        min_dists = dists.min(dim=1).values
        centers.append(points[min_dists.argmax()])
    centers = torch.stack(centers, dim=0).clone()

    for _ in range(num_iters):
        assignments = torch.cdist(points, centers).argmin(dim=1)
        new_centers = []
        for center_idx in range(num_centers):
            mask = assignments == center_idx
            if bool(mask.any()):
                new_centers.append(points[mask].mean(dim=0))
            else:
                new_centers.append(centers[center_idx])
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


def _long_rollout(env, states: torch.Tensor, steps: int) -> torch.Tensor:
    current = states.clone()
    for _ in range(max(0, steps)):
        current = env.step(current)
    return current


def _label_from_native_method(method, sequences: torch.Tensor) -> torch.Tensor:
    flat = sequences.reshape(-1, sequences.shape[-1])
    labels = method(flat)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    return labels.reshape(sequences.shape[:-1]).to(dtype=torch.long)


def _estimate_basin_centers(
    env,
    true_sequences: torch.Tensor,
    basin_count: int,
    endpoint_rollout_steps: int,
) -> torch.Tensor:
    endpoint_states = true_sequences[:, -1, :]
    converged = _long_rollout(env, endpoint_states, endpoint_rollout_steps)
    return _kmeans_centers(converged, basin_count)


def _assign_nearest_centers(sequences: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    flat = sequences.reshape(-1, sequences.shape[-1])
    labels = torch.cdist(flat, centers).argmin(dim=1)
    return labels.reshape(sequences.shape[:-1]).to(dtype=torch.long)


def _label_sequences(
    *,
    env,
    sequences: torch.Tensor,
    basin_count: int,
    endpoint_rollout_steps: int,
    estimated_centers: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, str, Optional[torch.Tensor]]:
    """Return basin labels for sequences and describe the label source."""
    if hasattr(env, "basin_label"):
        return _label_from_native_method(env.basin_label, sequences), "native_basin_label", None
    if hasattr(env, "points"):
        centers = env.points
        if centers.ndim == 2 and centers.shape[1] == sequences.shape[-1]:
            return _assign_nearest_centers(sequences, centers.to(dtype=sequences.dtype)), "env_points", centers
    if estimated_centers is None:
        estimated_centers = _estimate_basin_centers(env, sequences, basin_count, endpoint_rollout_steps)
    return _assign_nearest_centers(sequences, estimated_centers), "estimated_centers", estimated_centers


def _label_regions_if_available(env, sequences: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    if hasattr(env, "region_label"):
        return _label_from_native_method(env.region_label, sequences), "native_region_label"
    return None, None


def _support_tuple(z: torch.Tensor, threshold: float) -> Tuple[int, ...]:
    votes = (z.abs() > threshold).float().mean(dim=0)
    return tuple(int(item) for item in (votes > 0.5).cpu().tolist())


def _compute_support_metrics(
    model,
    sequences: torch.Tensor,
    basin_labels: torch.Tensor,
    threshold: float,
    recurring_min_count: int,
) -> Dict[str, object]:
    model.eval()
    support_counts: Counter[Tuple[int, ...]] = Counter()
    support_to_basins: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    support_switch_counts: List[int] = []
    support_sizes: List[int] = []

    with torch.no_grad():
        for traj, basin_path in zip(sequences, basin_labels):
            z = model.encode(traj)
            support = _support_tuple(z, threshold)
            support_counts[support] += 1
            support_to_basins[support].append(int(basin_path[-1].item()))
            support_sizes.append(int(sum(support)))

            step_supports = [
                tuple(int(item) for item in row.tolist())
                for row in (z.abs() > threshold).to(dtype=torch.int32).cpu()
            ]
            support_switch_counts.append(
                sum(1 for prev, nxt in zip(step_supports, step_supports[1:]) if prev != nxt)
            )

    recurring_supports = {
        support: count for support, count in support_counts.items() if count >= recurring_min_count
    }
    recurring_coverage = (
        float(sum(recurring_supports.values())) / float(max(len(sequences), 1))
        if recurring_supports
        else 0.0
    )

    purity_values = []
    group_rows = []
    for support, count in support_counts.items():
        basin_counter = Counter(support_to_basins[support])
        dominant_basin, dominant_count = basin_counter.most_common(1)[0]
        purity = float(dominant_count) / float(count)
        purity_values.append(purity)
        group_rows.append(
            {
                "support": "".join(str(bit) for bit in support),
                "count": count,
                "dominant_basin": int(dominant_basin),
                "purity": purity,
            }
        )

    return {
        "support_threshold": threshold,
        "support_group_count": len(support_counts),
        "recurring_support_group_count": len(recurring_supports),
        "retained_trajectory_coverage": recurring_coverage,
        "mean_support_size": float(np.mean(support_sizes)) if support_sizes else 0.0,
        "mean_support_switch_count": float(np.mean(support_switch_counts)) if support_switch_counts else 0.0,
        "mean_support_group_purity": float(np.mean(purity_values)) if purity_values else 0.0,
        "support_groups": sorted(group_rows, key=lambda row: (-row["count"], row["support"])),
    }


def _save_endpoint_confusion_heatmap(
    endpoint_pairs: Sequence[Tuple[Optional[int], Optional[int]]],
    basin_count: int,
    path: Path,
) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    matrix = np.zeros((basin_count, basin_count), dtype=np.float32)
    for true_label, pred_label in endpoint_pairs:
        if true_label is None or pred_label is None:
            continue
        if true_label < 0 or pred_label < 0:
            continue
        matrix[int(true_label), int(pred_label)] += 1.0

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted final basin")
    ax.set_ylabel("True final basin")
    ax.set_title("Endpoint Basin Confusion")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_transition_histogram(
    true_counts: Sequence[int],
    pred_counts: Sequence[int],
    path: Path,
) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    max_count = max(list(true_counts) + list(pred_counts) + [0])
    bins = np.arange(max_count + 2) - 0.5

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(true_counts, bins=bins, alpha=0.65, label="true")
    ax.hist(pred_counts, bins=bins, alpha=0.65, label="pred")
    ax.set_xlabel("Transition count")
    ax.set_ylabel("Trajectories")
    ax.set_title("Transition Count Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_phase_overlay(
    true_sequences: torch.Tensor,
    pred_sequences: torch.Tensor,
    path: Path,
    max_trajectories: int = 20,
) -> None:
    if true_sequences.shape[-1] < 2:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(max_trajectories, true_sequences.shape[0])
    indices = torch.arange(count)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for idx in indices.tolist():
        gt = true_sequences[idx, :, :2].cpu().numpy()
        pred = pred_sequences[idx, :, :2].cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], color="#6b7280", alpha=0.35, linewidth=1.4)
        ax.plot(pred[:, 0], pred[:, 1], color="#1d4ed8", alpha=0.8, linewidth=1.0)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Ground Truth vs Forecast Trajectories")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_support_group_heatmap(
    support_groups: Sequence[Dict[str, object]],
    path: Path,
) -> None:
    if not support_groups:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    counts = np.array([row["count"] for row in support_groups], dtype=np.float32)[:, None]
    purity = np.array([row["purity"] for row in support_groups], dtype=np.float32)[:, None]
    matrix = np.concatenate([counts, purity], axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4, max(4, 0.3 * len(support_groups))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks([0, 1], labels=["count", "purity"])
    ax.set_yticks(np.arange(len(support_groups)), labels=[row["support"][:24] for row in support_groups])
    ax.set_title("Support Groups")
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout_artifact", required=True, help="Path to rollout_artifacts.pt")
    parser.add_argument("--rollout_mode", default="no_reencode", help="Prediction mode to diagnose")
    parser.add_argument("--output_dir", default=None, help="Directory for diagnostic JSON and plots")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for support diagnostics")
    parser.add_argument(
        "--expected_num_basins",
        type=int,
        default=None,
        help="Optional override for the benchmark basin count",
    )
    parser.add_argument(
        "--endpoint_rollout_steps",
        type=int,
        default=1000,
        help="Extra env rollout steps used when estimating unlabeled endpoint basins",
    )
    parser.add_argument(
        "--support_threshold",
        type=float,
        default=1e-3,
        help="Absolute threshold for latent support membership",
    )
    parser.add_argument(
        "--recurring_min_count",
        type=int,
        default=2,
        help="Minimum number of trajectories required for a support group to count as recurring",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for optional checkpoint-based support diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.rollout_artifact)
    payload = _load_rollout_artifact(artifact_path)

    system = str(payload["system"])
    cfg = Config.from_dict(payload["config"])
    cfg.ENV.ENV_NAME = system
    env = make_env(cfg)

    init_states = payload["init_states"].float()
    true_sequences = payload["true_sequences"].float()
    predictions = payload["predictions"]
    if args.rollout_mode not in predictions:
        raise KeyError(
            f"Rollout mode '{args.rollout_mode}' not found. Available: {sorted(predictions.keys())}"
        )
    pred_future = predictions[args.rollout_mode].float().transpose(0, 1).contiguous()
    pred_sequences = torch.cat([init_states.unsqueeze(1), pred_future], dim=1)

    basin_count = _infer_basin_count(system, args.expected_num_basins)
    true_basin_labels, basin_label_source, estimated_centers = _label_sequences(
        env=env,
        sequences=true_sequences,
        basin_count=basin_count,
        endpoint_rollout_steps=args.endpoint_rollout_steps,
    )
    pred_basin_labels, _, _ = _label_sequences(
        env=env,
        sequences=pred_sequences,
        basin_count=basin_count,
        endpoint_rollout_steps=args.endpoint_rollout_steps,
        estimated_centers=estimated_centers,
    )
    basin_comparison = compare_label_sequences(
        list(true_basin_labels),
        list(pred_basin_labels),
    )

    true_region_labels, region_label_source = _label_regions_if_available(env, true_sequences)
    pred_region_labels = None
    region_comparison = None
    if true_region_labels is not None and region_label_source is not None:
        pred_region_labels, _ = _label_regions_if_available(env, pred_sequences)
        region_comparison = compare_label_sequences(
            list(true_region_labels),
            list(pred_region_labels),
        )

    support_metrics = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model = make_model(cfg, env.observation_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(args.device)
        support_metrics = _compute_support_metrics(
            model=model,
            sequences=true_sequences.to(args.device),
            basin_labels=true_basin_labels.to(args.device),
            threshold=args.support_threshold,
            recurring_min_count=args.recurring_min_count,
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else artifact_path.parent / f"diagnostics_{args.rollout_mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "system": system,
        "rollout_mode": args.rollout_mode,
        "rollout_artifact": str(artifact_path),
        "basin_count": basin_count,
        "basin_label_source": basin_label_source,
        "region_label_source": region_label_source,
        "basin_path_metrics": basin_comparison.to_dict(),
        "region_path_metrics": region_comparison.to_dict() if region_comparison is not None else None,
        "support_metrics": support_metrics,
    }

    summary_path = output_dir / "diagnostics.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    _save_endpoint_confusion_heatmap(
        [
            (true_traj.final_label, pred_traj.final_label)
            for true_traj, pred_traj in zip(
                basin_comparison.true_summary.trajectory_summaries,
                basin_comparison.pred_summary.trajectory_summaries,
            )
        ],
        basin_count,
        output_dir / "endpoint_confusion.png",
    )
    _save_transition_histogram(
        [traj.transition_count for traj in basin_comparison.true_summary.trajectory_summaries],
        [traj.transition_count for traj in basin_comparison.pred_summary.trajectory_summaries],
        output_dir / "transition_count_histogram.png",
    )
    _save_phase_overlay(
        true_sequences,
        pred_sequences,
        output_dir / "phase_overlay.png",
    )
    if support_metrics is not None:
        _save_support_group_heatmap(
            support_metrics["support_groups"],
            output_dir / "support_groups.png",
        )

    print(f"Wrote transition-rich rollout diagnostics to {summary_path}")


if __name__ == "__main__":
    main()
