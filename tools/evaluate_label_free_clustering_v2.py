#!/usr/bin/env python
"""Label-free clustering evaluation with multiple feature extraction strategies.

Evaluates whether unsupervised clustering of latent representations can recover
basin-of-attraction labels without any supervision.  Unlike the v1 evaluation
which only used trajectory-mean cosine features, this version tests multiple
feature views that better preserve the discrete support structure.

Feature views:
  - modal_support:    Per-timestep binary support → mode (most common) per traj
  - majority_support: Per-dim majority vote of binary support across timesteps
  - last_step_support: Binary support of the last encoded timestep
  - last_step_cosine:  Cosine-normalised encoding of the last timestep
  - traj_mean_support: Binarised trajectory-mean encoding
  - traj_mean_cosine:  Cosine-normalised trajectory-mean (original v1 approach)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score

from skae.basin_utils import BasinLabeledDataset
from skae.config import Config
from skae.data import make_env
from skae.model import KoopmanMachine, make_model


# ---- dataclasses -----------------------------------------------------------

@dataclass
class ViewResult:
    feature_view: str
    adjusted_rand_index: float
    normalized_mutual_info: float
    silhouette_score: float
    kmeans_purity: float
    linear_classifier_accuracy: float
    linear_classifier_cv_std: float


@dataclass
class LabelFreeClusteringResultsV2:
    system_name: str
    model_name: str
    num_trajectories: int
    num_ground_truth_basins: int
    latent_dim: int
    support_threshold: float
    basin_distribution: Dict[str, int]
    mean_sparsity: float
    mean_l1_norm: float
    view_results: List[ViewResult] = field(default_factory=list)


# ---- feature extraction ---------------------------------------------------

def encode_all_timesteps(
    model: KoopmanMachine,
    dataset: BasinLabeledDataset,
    device: str,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Encode every timestep in every trajectory.

    Returns:
        per_traj_latents: list of [T, D] arrays (one per trajectory)
        labels: [N] int array of basin labels
    """
    model.eval()
    per_traj_latents: List[np.ndarray] = []
    labels: List[int] = []

    with torch.no_grad():
        for item in dataset.trajectories:
            states = item.states.to(device)
            z = model.encode(states)  # [T, D]
            per_traj_latents.append(z.cpu().numpy())
            labels.append(item.final_basin)

    return per_traj_latents, np.asarray(labels, dtype=np.int64)


def extract_features(
    per_traj_latents: List[np.ndarray],
    feature_view: str,
    support_threshold: float,
) -> np.ndarray:
    """Extract a single feature vector per trajectory under the given view."""

    features: List[np.ndarray] = []

    for z in per_traj_latents:
        # z is [T, D]
        if feature_view == "modal_support":
            # Binary support at each timestep → find the mode
            supports = (np.abs(z) >= support_threshold).astype(np.int8)
            # Convert each row to a hashable tuple, find most common
            support_tuples = [tuple(row) for row in supports]
            counter = Counter(support_tuples)
            mode_support = np.array(counter.most_common(1)[0][0], dtype=np.float32)
            features.append(mode_support)

        elif feature_view == "majority_support":
            # Per-dimension majority vote across timesteps
            votes = (np.abs(z) >= support_threshold).astype(np.float32).mean(axis=0)
            features.append((votes > 0.5).astype(np.float32))

        elif feature_view == "last_step_support":
            support = (np.abs(z[-1]) >= support_threshold).astype(np.float32)
            features.append(support)

        elif feature_view == "last_step_cosine":
            vec = z[-1]
            norm = np.linalg.norm(vec)
            features.append(vec / max(norm, 1e-12))

        elif feature_view == "traj_mean_support":
            mean_z = z.mean(axis=0)
            support = (np.abs(mean_z) >= support_threshold).astype(np.float32)
            features.append(support)

        elif feature_view == "traj_mean_cosine":
            mean_z = z.mean(axis=0)
            norm = np.linalg.norm(mean_z)
            features.append(mean_z / max(norm, 1e-12))

        else:
            raise ValueError(f"Unknown feature_view: {feature_view}")

    return np.stack(features, axis=0)


# ---- clustering metrics ----------------------------------------------------

def compute_clustering_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    use_pca: bool = False,
    pca_dim: int = 20,
) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    if num_clusters < 2:
        return {
            "silhouette_score": 0.0,
            "adjusted_rand_index": 1.0,
            "normalized_mutual_info": 1.0,
            "kmeans_purity": 1.0,
        }

    work_features = features
    if use_pca and features.shape[1] > pca_dim:
        n_components = min(pca_dim, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        work_features = pca.fit_transform(features)

    try:
        sil = float(silhouette_score(work_features, labels))
    except ValueError:
        sil = 0.0

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    pred = kmeans.fit_predict(work_features)
    ari = float(adjusted_rand_score(labels, pred))
    nmi = float(normalized_mutual_info_score(labels, pred))

    purity_hits = 0
    for cluster_id in range(num_clusters):
        mask = pred == cluster_id
        if not np.any(mask):
            continue
        cluster_labels = labels[mask]
        counts = np.bincount(cluster_labels)
        purity_hits += int(counts.max())
    purity = float(purity_hits / len(labels))

    return {
        "silhouette_score": sil,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "kmeans_purity": purity,
    }


def compute_linear_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    use_pca: bool = False,
    pca_dim: int = 20,
) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "linear_classifier_accuracy": 1.0,
            "linear_classifier_cv_std": 0.0,
        }

    work_features = features
    if use_pca and features.shape[1] > pca_dim:
        n_components = min(pca_dim, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        work_features = pca.fit_transform(features)

    # Drop classes with fewer than 2 samples to enable proper CV
    counts = np.bincount(labels)
    keep_classes = np.where(counts >= 2)[0]
    if len(keep_classes) < 2:
        return {
            "linear_classifier_accuracy": float("nan"),
            "linear_classifier_cv_std": float("nan"),
            "linear_classifier_dropped_classes": int(len(counts) - len(keep_classes)),
        }
    if len(keep_classes) < len(counts):
        mask = np.isin(labels, keep_classes)
        work_features = work_features[mask]
        labels = labels[mask]
        # Remap to contiguous indices
        remap = {old: new for new, old in enumerate(keep_classes)}
        labels = np.array([remap[l] for l in labels])

    clf = LogisticRegression(max_iter=1000, random_state=42)
    min_count = min(np.bincount(labels))
    folds = min(5, min_count)

    scores = cross_val_score(clf, work_features, labels, cv=folds)
    result = {
        "linear_classifier_accuracy": float(scores.mean()),
        "linear_classifier_cv_std": float(scores.std()),
    }
    if len(keep_classes) < len(counts):
        result["linear_classifier_dropped_classes"] = int(len(counts) - len(keep_classes))
    return result


# ---- main ------------------------------------------------------------------

ALL_VIEWS = [
    "modal_support",
    "majority_support",
    "last_step_support",
    "last_step_cosine",
    "traj_mean_support",
    "traj_mean_cosine",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--system", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--feature_views",
        default=",".join(ALL_VIEWS),
        help="Comma-separated list of feature views to evaluate",
    )
    parser.add_argument("--support_threshold", type=float, default=1e-3)
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--long_rollout_steps", type=int, default=5000)
    parser.add_argument("--pca_dim", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    views = [v.strip() for v in args.feature_views.split(",")]

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = Config.from_dict(checkpoint["config"])
    if args.system is not None:
        cfg.ENV.ENV_NAME = args.system
    system = cfg.ENV.ENV_NAME

    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    # Generate basin-labeled trajectories
    dataset = BasinLabeledDataset(
        system=system,
        cfg=cfg,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        long_rollout_steps=args.long_rollout_steps,
        seed=args.seed,
    )

    # Encode every timestep once (shared across all views)
    per_traj_latents, labels = encode_all_timesteps(model, dataset, device=args.device)

    # Sparsity stats from raw latents
    all_latents = np.concatenate(per_traj_latents, axis=0)
    mean_sparsity = float((np.abs(all_latents) < args.support_threshold).mean())
    mean_l1_norm = float(np.abs(all_latents).sum(axis=-1).mean())
    latent_dim = int(per_traj_latents[0].shape[1])

    # Basin distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    basin_dist = {str(int(l)): int(c) for l, c in zip(unique_labels, counts)}

    # Evaluate each view
    view_results: List[ViewResult] = []
    for view in views:
        features = extract_features(per_traj_latents, view, args.support_threshold)

        # Determine whether to use PCA (for high-dim continuous features)
        is_binary = view.endswith("_support")
        use_pca = not is_binary and features.shape[1] > args.pca_dim

        clustering = compute_clustering_metrics(features, labels, use_pca=use_pca, pca_dim=args.pca_dim)
        linear = compute_linear_accuracy(features, labels, use_pca=use_pca, pca_dim=args.pca_dim)

        vr = ViewResult(
            feature_view=view,
            adjusted_rand_index=clustering["adjusted_rand_index"],
            normalized_mutual_info=clustering["normalized_mutual_info"],
            silhouette_score=clustering["silhouette_score"],
            kmeans_purity=clustering["kmeans_purity"],
            linear_classifier_accuracy=linear["linear_classifier_accuracy"],
            linear_classifier_cv_std=linear["linear_classifier_cv_std"],
        )
        view_results.append(vr)
        print(
            f"  {view:25s}  ARI={vr.adjusted_rand_index:.4f}  NMI={vr.normalized_mutual_info:.4f}  "
            f"sil={vr.silhouette_score:.4f}  purity={vr.kmeans_purity:.4f}  "
            f"lin_acc={vr.linear_classifier_accuracy:.4f}"
        )

    results = LabelFreeClusteringResultsV2(
        system_name=system,
        model_name=type(model).__name__,
        num_trajectories=len(dataset.trajectories),
        num_ground_truth_basins=dataset.num_basins,
        latent_dim=latent_dim,
        support_threshold=args.support_threshold,
        basin_distribution=basin_dist,
        mean_sparsity=mean_sparsity,
        mean_l1_norm=mean_l1_norm,
        view_results=view_results,
    )

    analysis_path = output_dir / "analysis_results.json"
    analysis_path.write_text(json.dumps(asdict(results), indent=2) + "\n")

    metadata = {
        "checkpoint": args.checkpoint,
        "system": system,
        "feature_views": views,
        "support_threshold": args.support_threshold,
        "num_trajectories": args.num_trajectories,
        "trajectory_length": args.trajectory_length,
        "long_rollout_steps": args.long_rollout_steps,
        "pca_dim": args.pca_dim,
        "seed": args.seed,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(asdict(results), indent=2))


if __name__ == "__main__":
    main()
