#!/usr/bin/env python
"""Evaluate label-free clustering quality for a single checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
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


@dataclass
class LabelFreeClusteringResults:
    system_name: str
    model_name: str
    feature_view: str
    num_trajectories: int
    num_ground_truth_basins: int
    latent_dim: int
    silhouette_score: float
    adjusted_rand_index: float
    normalized_mutual_info: float
    kmeans_purity: float
    linear_classifier_accuracy: float
    linear_classifier_cv_std: float
    mean_sparsity: float
    mean_l1_norm: float
    basin_distribution: Dict[str, int]


def encode_trajectory_means(
    model: KoopmanMachine,
    dataset: BasinLabeledDataset,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for item in dataset.trajectories:
            states = item.states.to(device)
            z = model.encode(states)
            latents.append(z.mean(dim=0).cpu().numpy())
            labels.append(item.final_basin)

    return np.stack(latents, axis=0), np.asarray(labels, dtype=np.int64)


def transform_features(
    latents: np.ndarray,
    feature_view: str,
    support_threshold: float,
) -> np.ndarray:
    if feature_view == "raw":
        return latents
    if feature_view == "cosine":
        norms = np.linalg.norm(latents, axis=-1, keepdims=True)
        return latents / np.clip(norms, a_min=1e-12, a_max=None)
    if feature_view == "support":
        return (np.abs(latents) >= support_threshold).astype(np.float32)
    raise ValueError(f"Unsupported feature_view: {feature_view}")


def compute_sparsity_metrics(latents: np.ndarray, threshold: float) -> Dict[str, float]:
    return {
        "mean_sparsity": float((np.abs(latents) < threshold).mean()),
        "mean_l1_norm": float(np.abs(latents).sum(axis=-1).mean()),
    }


def compute_clustering_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    if num_clusters < 2:
        return {
            "silhouette_score": 0.0,
            "adjusted_rand_index": 1.0,
            "normalized_mutual_info": 1.0,
            "kmeans_purity": 1.0,
        }

    try:
        sil = float(silhouette_score(features, labels))
    except ValueError:
        sil = 0.0

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    pred = kmeans.fit_predict(features)
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


def compute_linear_accuracy(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "linear_classifier_accuracy": 1.0,
            "linear_classifier_cv_std": 0.0,
        }

    clf = LogisticRegression(max_iter=1000, random_state=42)
    min_count = min(np.bincount(labels))
    folds = min(5, min_count)
    if folds < 2:
        clf.fit(features, labels)
        return {
            "linear_classifier_accuracy": float(clf.score(features, labels)),
            "linear_classifier_cv_std": 0.0,
        }

    scores = cross_val_score(clf, features, labels, cv=folds)
    return {
        "linear_classifier_accuracy": float(scores.mean()),
        "linear_classifier_cv_std": float(scores.std()),
    }


def basin_distribution(labels: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--system", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_view", default="cosine", choices=("raw", "cosine", "support"))
    parser.add_argument("--support_threshold", type=float, default=1e-3)
    parser.add_argument("--num_trajectories", type=int, default=128)
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--long_rollout_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = Config.from_dict(checkpoint["config"])
    if args.system is not None:
        cfg.ENV.ENV_NAME = args.system
    system = cfg.ENV.ENV_NAME

    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint["model_state_dict"])
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

    latents, labels = encode_trajectory_means(model, dataset, device=args.device)
    features = transform_features(latents, args.feature_view, args.support_threshold)

    clustering = compute_clustering_metrics(features, labels)
    linear = compute_linear_accuracy(features, labels)
    sparsity = compute_sparsity_metrics(latents, args.support_threshold)

    results = LabelFreeClusteringResults(
        system_name=system,
        model_name=type(model).__name__,
        feature_view=args.feature_view,
        num_trajectories=len(dataset.trajectories),
        num_ground_truth_basins=dataset.num_basins,
        latent_dim=int(latents.shape[1]),
        silhouette_score=clustering["silhouette_score"],
        adjusted_rand_index=clustering["adjusted_rand_index"],
        normalized_mutual_info=clustering["normalized_mutual_info"],
        kmeans_purity=clustering["kmeans_purity"],
        linear_classifier_accuracy=linear["linear_classifier_accuracy"],
        linear_classifier_cv_std=linear["linear_classifier_cv_std"],
        mean_sparsity=sparsity["mean_sparsity"],
        mean_l1_norm=sparsity["mean_l1_norm"],
        basin_distribution=basin_distribution(labels),
    )

    analysis_path = output_dir / "analysis_results.json"
    analysis_path.write_text(json.dumps(asdict(results), indent=2) + "\n")

    metadata = {
        "checkpoint": args.checkpoint,
        "system": system,
        "feature_view": args.feature_view,
        "support_threshold": args.support_threshold,
        "num_trajectories": args.num_trajectories,
        "trajectory_length": args.trajectory_length,
        "long_rollout_steps": args.long_rollout_steps,
        "seed": args.seed,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(asdict(results), indent=2))


if __name__ == "__main__":
    main()
