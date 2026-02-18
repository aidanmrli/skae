"""
Evaluate whether each basin of attraction maps to a unique sparse support pattern.

This is a mechanistic-style diagnostic: do different basins consistently
activate different sets of latent coordinates?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from skae.config import Config
from skae.data import make_env
from skae.model import make_model
from skae.basin_utils import BasinLabeledDataset, BasinLabeledTrajectory


# ---------------------------------------------------------------------------
# Support Uniqueness Metrics
# ---------------------------------------------------------------------------


@dataclass
class SupportUniquenessResults:
    system_name: str
    model_name: str
    num_trajectories: int
    num_basins: int
    latent_dim: int
    support_threshold: float
    support_mode: str
    unique_mode_supports: int
    mode_collision_pairs: int
    mode_uniqueness_rate: float
    mean_basin_consistency: float
    mean_mode_support_size: float
    mean_pairwise_jaccard: float
    # Cosine similarity metrics (threshold-free)
    mean_intra_basin_cosine: float = 0.0
    mean_inter_basin_cosine: float = 0.0
    cosine_separation_score: float = 0.0
    per_basin_consistency: Dict[int, float] = field(default_factory=dict)
    per_basin_support_size: Dict[int, float] = field(default_factory=dict)
    per_basin_active_indices: Dict[int, List[int]] = field(default_factory=dict)


def _support_from_latents(
    latents: torch.Tensor,
    threshold: float,
    mode: str,
) -> np.ndarray:
    if mode == "mean":
        z = latents.mean(dim=0)
        support = (z.abs() > threshold).cpu().numpy()
    elif mode == "last":
        z = latents[-1]
        support = (z.abs() > threshold).cpu().numpy()
    elif mode == "median":
        z = latents.median(dim=0).values
        support = (z.abs() > threshold).cpu().numpy()
    elif mode == "majority":
        votes = (latents.abs() > threshold).float().mean(dim=0)
        support = (votes > 0.5).cpu().numpy()
    else:
        raise ValueError(f"Unknown support mode '{mode}'")
    return support.astype(np.int8)


def _aggregate_latents_for_cosine(
    latents: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    if aggregation == "mean":
        return latents.mean(dim=0)
    if aggregation == "median":
        return latents.median(dim=0).values
    if aggregation == "mean_abs":
        return latents.abs().mean(dim=0)
    raise ValueError(f"Unknown cosine aggregation '{aggregation}'")


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _compute_cosine_metrics(
    vectors: np.ndarray,
    basin_ids: np.ndarray,
    num_basins: int,
) -> Dict[str, float]:
    basin_latents: Dict[int, List[np.ndarray]] = {b: [] for b in range(num_basins)}
    for vec, basin in zip(vectors, basin_ids):
        basin_latents[int(basin)].append(vec)

    # Intra-basin: mean pairwise cosine within each basin
    intra_cosines = []
    for basin, vecs in basin_latents.items():
        if len(vecs) < 2:
            continue
        pairs = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                pairs.append(_cosine(vecs[i], vecs[j]))
        if pairs:
            intra_cosines.append(float(np.mean(pairs)))

    # Inter-basin: cosine between basin centroids
    centroids: Dict[int, np.ndarray] = {}
    for basin, vecs in basin_latents.items():
        if vecs:
            centroids[basin] = np.mean(vecs, axis=0)
    inter_cosines = []
    basins_sorted = sorted(centroids.keys())
    for i, bi in enumerate(basins_sorted):
        for bj in basins_sorted[i + 1:]:
            inter_cosines.append(_cosine(centroids[bi], centroids[bj]))

    mean_intra = float(np.mean(intra_cosines)) if intra_cosines else 0.0
    mean_inter = float(np.mean(inter_cosines)) if inter_cosines else 0.0

    return {
        "mean_intra_basin_cosine": mean_intra,
        "mean_inter_basin_cosine": mean_inter,
        "cosine_separation_score": mean_intra - mean_inter,
    }


def _demean_vectors(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean_vec = vectors.mean(axis=0)
    return vectors - mean_vec, mean_vec


def _remove_pc1(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    if vectors.shape[0] < 2:
        return vectors, np.zeros((vectors.shape[1],), dtype=vectors.dtype), 0.0
    u, s, vt = np.linalg.svd(vectors, full_matrices=False)
    pc1 = vt[0]
    projected = (vectors @ pc1)[:, None] * pc1[None, :]
    removed = vectors - projected
    total_var = float((s ** 2).sum())
    pc1_var = float(s[0] ** 2) if s.size > 0 else 0.0
    pc1_ratio = (pc1_var / total_var) if total_var > 0 else 0.0
    return removed, pc1, pc1_ratio


def compute_cosine_basin_similarity(
    model,
    dataset: BasinLabeledDataset,
    device: str,
    aggregation: str = "mean",
    demean: bool = False,
    remove_pc1: bool = False,
) -> Dict[str, float]:
    """Compute cosine similarity metrics on continuous latent activations.

    Operates on the raw (unthresholded) encoded vectors, avoiding the
    sensitivity to hard threshold choices that plagues binary support metrics.

    Returns dict with:
        mean_intra_basin_cosine: mean cosine similarity within each basin
        mean_inter_basin_cosine: mean cosine similarity between basin centroids
        cosine_separation_score: intra - inter (higher = better)
    """
    model.eval()
    vectors: List[np.ndarray] = []
    basin_ids: List[int] = []

    with torch.no_grad():
        for traj in dataset.trajectories:
            states = traj.states.to(device)
            z = model.encode(states)
            z_vec = _aggregate_latents_for_cosine(z, aggregation).cpu().numpy()
            vectors.append(z_vec)
            basin_ids.append(traj.final_basin)

    if not vectors:
        return {
            "mean_intra_basin_cosine": 0.0,
            "mean_inter_basin_cosine": 0.0,
            "cosine_separation_score": 0.0,
        }

    vectors_np = np.stack(vectors, axis=0)
    if demean or remove_pc1:
        vectors_np, _ = _demean_vectors(vectors_np)
    if remove_pc1:
        vectors_np, _, _ = _remove_pc1(vectors_np)

    return _compute_cosine_metrics(
        vectors_np,
        np.array(basin_ids, dtype=np.int64),
        dataset.num_basins,
    )


def compute_cosine_diagnostics(
    model,
    dataset: BasinLabeledDataset,
    device: str,
    aggregations: Sequence[str] = ("mean", "median", "mean_abs"),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute cosine separation diagnostics across aggregations and transforms."""
    model.eval()
    diagnostics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for agg in aggregations:
        vectors: List[np.ndarray] = []
        basin_ids: List[int] = []
        with torch.no_grad():
            for traj in dataset.trajectories:
                states = traj.states.to(device)
                z = model.encode(states)
                z_vec = _aggregate_latents_for_cosine(z, agg).cpu().numpy()
                vectors.append(z_vec)
                basin_ids.append(traj.final_basin)

        if not vectors:
            diagnostics[agg] = {}
            continue

        vectors_np = np.stack(vectors, axis=0)
        basin_ids_np = np.array(basin_ids, dtype=np.int64)

        def _stats(vecs: np.ndarray) -> Dict[str, float]:
            norms = np.linalg.norm(vecs, axis=1)
            centroid_norms = []
            for b in range(dataset.num_basins):
                mask = basin_ids_np == b
                if not np.any(mask):
                    continue
                centroid = vecs[mask].mean(axis=0)
                centroid_norms.append(float(np.linalg.norm(centroid)))
            centroid_norms = np.array(centroid_norms, dtype=np.float64)
            return {
                "mean_vector_norm": float(np.mean(norms)),
                "mean_abs_value": float(np.mean(np.abs(vecs))),
                "global_mean_norm": float(np.linalg.norm(vecs.mean(axis=0))),
                "centroid_norm_mean": float(np.mean(centroid_norms)) if centroid_norms.size else 0.0,
                "centroid_norm_std": float(np.std(centroid_norms)) if centroid_norms.size else 0.0,
                "centroid_norm_min": float(np.min(centroid_norms)) if centroid_norms.size else 0.0,
                "centroid_norm_max": float(np.max(centroid_norms)) if centroid_norms.size else 0.0,
            }

        diagnostics[agg] = {}

        raw_metrics = _compute_cosine_metrics(vectors_np, basin_ids_np, dataset.num_basins)
        diagnostics[agg]["raw"] = {**raw_metrics, **_stats(vectors_np)}

        demeaned, mean_vec = _demean_vectors(vectors_np)
        demean_metrics = _compute_cosine_metrics(demeaned, basin_ids_np, dataset.num_basins)
        diagnostics[agg]["demean"] = {
            **demean_metrics,
            **_stats(demeaned),
            "global_mean_norm": float(np.linalg.norm(mean_vec)),
        }

        pc_removed, pc1, pc1_ratio = _remove_pc1(demeaned)
        pc_metrics = _compute_cosine_metrics(pc_removed, basin_ids_np, dataset.num_basins)
        diagnostics[agg]["pc1_removed"] = {
            **pc_metrics,
            **_stats(pc_removed),
            "pc1_norm": float(np.linalg.norm(pc1)),
            "pc1_explained_variance_ratio": float(pc1_ratio),
        }

    return diagnostics


def compute_support_uniqueness(
    model,
    dataset: BasinLabeledDataset,
    device: str,
    support_threshold: float,
    support_mode: str,
) -> SupportUniquenessResults:
    model.eval()
    basin_supports: Dict[int, List[Tuple[int, ...]]] = {b: [] for b in range(dataset.num_basins)}

    with torch.no_grad():
        for traj in dataset.trajectories:
            states = traj.states.to(device)
            z = model.encode(states)
            support = _support_from_latents(z, support_threshold, support_mode)
            basin_supports[traj.final_basin].append(tuple(support.tolist()))

    # Mode support per basin + consistency
    per_basin_consistency: Dict[int, float] = {}
    per_basin_support_size: Dict[int, float] = {}
    per_basin_active_indices: Dict[int, List[int]] = {}
    basin_mode_supports: Dict[int, Tuple[int, ...]] = {}
    for basin, supports in basin_supports.items():
        if not supports:
            per_basin_consistency[basin] = 0.0
            per_basin_support_size[basin] = 0.0
            continue
        counts: Dict[Tuple[int, ...], int] = {}
        for s in supports:
            counts[s] = counts.get(s, 0) + 1
        mode_support, mode_count = max(counts.items(), key=lambda kv: kv[1])
        basin_mode_supports[basin] = mode_support
        per_basin_consistency[basin] = mode_count / max(1, len(supports))
        per_basin_support_size[basin] = float(np.sum(mode_support))
        per_basin_active_indices[basin] = [i for i, v in enumerate(mode_support) if v == 1]

    # Uniqueness across basins
    mode_support_list = list(basin_mode_supports.values())
    unique_mode_supports = len(set(mode_support_list))
    total_pairs = dataset.num_basins * (dataset.num_basins - 1) // 2
    collision_pairs = 0
    jaccards = []
    basins = sorted(basin_mode_supports.keys())
    for i, bi in enumerate(basins):
        si = np.array(basin_mode_supports[bi], dtype=np.int8)
        for bj in basins[i + 1:]:
            sj = np.array(basin_mode_supports[bj], dtype=np.int8)
            if np.array_equal(si, sj):
                collision_pairs += 1
            jaccards.append(_jaccard(si, sj))

    mean_jaccard = float(np.mean(jaccards)) if jaccards else 0.0
    uniqueness_rate = 1.0 - (collision_pairs / max(1, total_pairs))

    # Aggregate stats
    mean_consistency = float(np.mean(list(per_basin_consistency.values()))) if per_basin_consistency else 0.0
    mean_support_size = float(np.mean(list(per_basin_support_size.values()))) if per_basin_support_size else 0.0

    return SupportUniquenessResults(
        system_name=dataset.system,
        model_name=type(model).__name__,
        num_trajectories=len(dataset.trajectories),
        num_basins=dataset.num_basins,
        latent_dim=model.target_size,
        support_threshold=support_threshold,
        support_mode=support_mode,
        unique_mode_supports=unique_mode_supports,
        mode_collision_pairs=collision_pairs,
        mode_uniqueness_rate=uniqueness_rate,
        mean_basin_consistency=mean_consistency,
        mean_mode_support_size=mean_support_size,
        mean_pairwise_jaccard=mean_jaccard,
        per_basin_consistency=per_basin_consistency,
        per_basin_support_size=per_basin_support_size,
        per_basin_active_indices=per_basin_active_indices,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate support uniqueness across basins"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (last.pt or checkpoint.pt)')
    parser.add_argument('--system', type=str, default=None,
                        help='System to evaluate (defaults to checkpoint env). Supported: duffing, lyapunov, multiwell*')
    parser.add_argument('--num_trajectories', type=int, default=100,
                        help='Number of test trajectories')
    parser.add_argument('--trajectory_length', type=int, default=500,
                        help='Length of each trajectory')
    parser.add_argument('--long_rollout_steps', type=int, default=5000,
                        help='Steps for basin identification after trajectory end')
    parser.add_argument('--support_threshold', type=float, default=1e-3,
                        help='Threshold for nonzero support')
    parser.add_argument('--support_mode', type=str, default='mean',
                        choices=['mean', 'last', 'median', 'majority'],
                        help='How to aggregate support over a trajectory')
    parser.add_argument('--cosine_aggregation', type=str, default='mean',
                        choices=['mean', 'median', 'mean_abs'],
                        help='Aggregation for cosine similarity metrics')
    parser.add_argument('--cosine_report_all', action='store_true',
                        help='Report cosine metrics for mean/median/mean_abs')
    parser.add_argument('--cosine_diag', action='store_true',
                        help='Save extended cosine diagnostics to JSON')
    parser.add_argument('--cosine_demean', action='store_true',
                        help='Demean trajectory vectors before cosine metrics')
    parser.add_argument('--cosine_remove_pc1', action='store_true',
                        help='Remove top principal component before cosine metrics')
    parser.add_argument('--cosine_only', action='store_true',
                        help='Only compute cosine metrics (skip support thresholding)')
    parser.add_argument('--output_dir', type=str, default='results/support_uniqueness',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for trajectory generation')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run on')
    parser.add_argument('--threshold_sweep', action='store_true',
                        help='Run evaluation across multiple thresholds')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = Config.from_dict(checkpoint['config'])

    if args.system is not None:
        cfg.ENV.ENV_NAME = args.system
    system = cfg.ENV.ENV_NAME
    print(f"Evaluating on system: {system}")

    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    dataset = BasinLabeledDataset(
        system=system,
        cfg=cfg,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        long_rollout_steps=args.long_rollout_steps,
        seed=args.seed,
    )

    # Cosine similarity metrics (threshold-free)
    cosine_metrics = compute_cosine_basin_similarity(
        model,
        dataset,
        args.device,
        aggregation=args.cosine_aggregation,
        demean=args.cosine_demean,
        remove_pc1=args.cosine_remove_pc1,
    )
    print("\nCosine similarity metrics (threshold-free):")
    print(f"  Aggregation: {args.cosine_aggregation}")
    if args.cosine_demean:
        print("  Demean: True")
    if args.cosine_remove_pc1:
        print("  Remove PC1: True")
    print(f"  Mean intra-basin cosine: {cosine_metrics['mean_intra_basin_cosine']:.4f}")
    print(f"  Mean inter-basin cosine: {cosine_metrics['mean_inter_basin_cosine']:.4f}")
    print(f"  Cosine separation score: {cosine_metrics['cosine_separation_score']:.4f}")
    if args.cosine_only and args.threshold_sweep:
        print("\nNote: --cosine_only set; skipping threshold sweep.")

    if args.cosine_report_all:
        print("\nCosine metrics by aggregation (raw):")
        for agg in ("mean", "median", "mean_abs"):
            metrics = compute_cosine_basin_similarity(
                model, dataset, args.device, aggregation=agg
            )
            print(
                f"  {agg:>8}: intra={metrics['mean_intra_basin_cosine']:.4f} "
                f"inter={metrics['mean_inter_basin_cosine']:.4f} "
                f"sep={metrics['cosine_separation_score']:.4f}"
            )

    if args.cosine_diag:
        diagnostics = compute_cosine_diagnostics(model, dataset, args.device)
        diag_path = output_dir / "cosine_diagnostics.json"
        with open(diag_path, "w") as f:
            json.dump(diagnostics, f, indent=2)
        print(f"\nSaved cosine diagnostics to {diag_path}")

    if args.threshold_sweep and not args.cosine_only:
        thresholds = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        sweep_results = []
        print(f"\nThreshold sweep across {len(thresholds)} values:")
        print(f"{'Threshold':>12} {'Consistency':>12} {'Uniqueness':>11} "
              f"{'Jaccard':>8} {'SupportSize':>12} {'UniqueSupp':>11}")
        print("-" * 70)
        for thresh in thresholds:
            res = compute_support_uniqueness(
                model, dataset, device=args.device,
                support_threshold=thresh, support_mode=args.support_mode,
            )
            # Attach cosine metrics
            res.mean_intra_basin_cosine = cosine_metrics['mean_intra_basin_cosine']
            res.mean_inter_basin_cosine = cosine_metrics['mean_inter_basin_cosine']
            res.cosine_separation_score = cosine_metrics['cosine_separation_score']
            sweep_results.append(asdict(res))
            print(f"{thresh:>12.1e} {res.mean_basin_consistency:>12.3f} "
                  f"{res.mode_uniqueness_rate:>11.3f} {res.mean_pairwise_jaccard:>8.3f} "
                  f"{res.mean_mode_support_size:>12.1f} "
                  f"{res.unique_mode_supports:>5}/{res.num_basins}")

        sweep_path = output_dir / "threshold_sweep.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSaved threshold sweep to {sweep_path}")
    elif not args.cosine_only:
        results = compute_support_uniqueness(
            model, dataset, device=args.device,
            support_threshold=args.support_threshold,
            support_mode=args.support_mode,
        )
        results.mean_intra_basin_cosine = cosine_metrics['mean_intra_basin_cosine']
        results.mean_inter_basin_cosine = cosine_metrics['mean_inter_basin_cosine']
        results.cosine_separation_score = cosine_metrics['cosine_separation_score']

        results_path = output_dir / "support_uniqueness.json"
        with open(results_path, "w") as f:
            json.dump(asdict(results), f, indent=2)

        print("\nSupport uniqueness results:")
        print(f"  Unique mode supports: {results.unique_mode_supports}/{results.num_basins}")
        print(f"  Mode collisions (pairs): {results.mode_collision_pairs}")
        print(f"  Mode uniqueness rate: {results.mode_uniqueness_rate:.3f}")
        print(f"  Mean basin consistency: {results.mean_basin_consistency:.3f}")
        print(f"  Mean mode support size: {results.mean_mode_support_size:.1f}")
        print(f"  Mean pairwise Jaccard: {results.mean_pairwise_jaccard:.3f}")
        print(f"\nSaved results to {results_path}")

    # Always save cosine metrics
    cosine_path = output_dir / "cosine_metrics.json"
    cosine_payload = {
        "system_name": system,
        "model_name": cfg.MODEL.MODEL_NAME,
        "num_trajectories": args.num_trajectories,
        "trajectory_length": args.trajectory_length,
        "long_rollout_steps": args.long_rollout_steps,
        "cosine_aggregation": args.cosine_aggregation,
        "cosine_demean": args.cosine_demean,
        "cosine_remove_pc1": args.cosine_remove_pc1,
        "seed": args.seed,
        **cosine_metrics,
    }
    with open(cosine_path, "w") as f:
        json.dump(cosine_payload, f, indent=2)
    print(f"\nSaved cosine metrics to {cosine_path}")


if __name__ == "__main__":
    main()
