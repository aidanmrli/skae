"""Offline support-alignment evaluation for the competitive_lv paper workstream.

Runs support uniqueness, cosine separation, and latent clustering metrics on an
explicit list of checkpoints and writes both per-run JSON artifacts and an
aggregate summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_val_score

from skae.basin_utils import BasinLabeledDataset
from skae.config import Config
from skae.data import make_env
from skae.model import make_model


def _parse_entry(raw: str) -> Tuple[str, int, Path]:
    parts = raw.split("::", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Entries must use ROOT_LABEL::SEED::CHECKPOINT_PATH format."
        )
    root_label, seed_raw, checkpoint_raw = parts
    try:
        seed = int(seed_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed '{seed_raw}'.") from exc
    checkpoint = Path(checkpoint_raw)
    return root_label, seed, checkpoint


def _load_checkpoint(checkpoint_path: Path, device: str) -> Dict:
    return torch.load(checkpoint_path, map_location=device)


def _dataset_cache_key(
    checkpoint_dict: Dict,
    system: str,
    num_trajectories: int,
    trajectory_length: int,
    long_rollout_steps: int,
    eval_seed: int,
) -> str:
    key = {
        "env": checkpoint_dict["config"]["ENV"],
        "system": system,
        "num_trajectories": num_trajectories,
        "trajectory_length": trajectory_length,
        "long_rollout_steps": long_rollout_steps,
        "eval_seed": eval_seed,
    }
    return json.dumps(key, sort_keys=True)


def _build_dataset(
    checkpoint_dict: Dict,
    system: str,
    num_trajectories: int,
    trajectory_length: int,
    long_rollout_steps: int,
    eval_seed: int,
) -> Tuple[Config, BasinLabeledDataset]:
    cfg = Config.from_dict(checkpoint_dict["config"])
    cfg.ENV.ENV_NAME = system
    dataset = BasinLabeledDataset(
        system=system,
        cfg=cfg,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        long_rollout_steps=long_rollout_steps,
        seed=eval_seed,
    )
    return cfg, dataset


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

    intra_cosines = []
    for vecs in basin_latents.values():
        if len(vecs) < 2:
            continue
        pairs = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                pairs.append(_cosine(vecs[i], vecs[j]))
        if pairs:
            intra_cosines.append(float(np.mean(pairs)))

    centroids: Dict[int, np.ndarray] = {}
    for basin, vecs in basin_latents.items():
        if vecs:
            centroids[basin] = np.mean(vecs, axis=0)
    inter_cosines = []
    basins_sorted = sorted(centroids.keys())
    for i, bi in enumerate(basins_sorted):
        for bj in basins_sorted[i + 1 :]:
            inter_cosines.append(_cosine(centroids[bi], centroids[bj]))

    mean_intra = float(np.mean(intra_cosines)) if intra_cosines else 0.0
    mean_inter = float(np.mean(inter_cosines)) if inter_cosines else 0.0
    return {
        "mean_intra_basin_cosine": mean_intra,
        "mean_inter_basin_cosine": mean_inter,
        "cosine_separation_score": mean_intra - mean_inter,
    }


def _compute_support_metrics(
    *,
    model,
    dataset: BasinLabeledDataset,
    device: str,
    support_threshold: float,
    support_mode: str,
) -> Dict[str, float]:
    model.eval()
    basin_supports: Dict[int, List[Tuple[int, ...]]] = {
        b: [] for b in range(dataset.num_basins)
    }

    with torch.no_grad():
        for traj in dataset.trajectories:
            states = traj.states.to(device)
            z = model.encode(states)
            support = _support_from_latents(z, support_threshold, support_mode)
            basin_supports[traj.final_basin].append(tuple(support.tolist()))

    per_basin_consistency: Dict[int, float] = {}
    per_basin_support_size: Dict[int, float] = {}
    basin_mode_supports: Dict[int, Tuple[int, ...]] = {}
    for basin, supports in basin_supports.items():
        if not supports:
            per_basin_consistency[basin] = 0.0
            per_basin_support_size[basin] = 0.0
            continue
        counts: Dict[Tuple[int, ...], int] = {}
        for support in supports:
            counts[support] = counts.get(support, 0) + 1
        mode_support, mode_count = max(counts.items(), key=lambda item: item[1])
        basin_mode_supports[basin] = mode_support
        per_basin_consistency[basin] = mode_count / max(1, len(supports))
        per_basin_support_size[basin] = float(np.sum(mode_support))

    mode_support_list = list(basin_mode_supports.values())
    unique_mode_supports = len(set(mode_support_list))
    total_pairs = dataset.num_basins * (dataset.num_basins - 1) // 2
    collision_pairs = 0
    jaccards = []
    basins = sorted(basin_mode_supports.keys())
    for i, bi in enumerate(basins):
        si = np.array(basin_mode_supports[bi], dtype=np.int8)
        for bj in basins[i + 1 :]:
            sj = np.array(basin_mode_supports[bj], dtype=np.int8)
            if np.array_equal(si, sj):
                collision_pairs += 1
            jaccards.append(_jaccard(si, sj))

    return {
        "mode_uniqueness_rate": 1.0 - (collision_pairs / max(1, total_pairs)),
        "mean_basin_consistency": float(np.mean(list(per_basin_consistency.values())))
        if per_basin_consistency
        else 0.0,
        "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "unique_mode_supports": int(unique_mode_supports),
        "mode_collision_pairs": int(collision_pairs),
        "mean_mode_support_size": float(np.mean(list(per_basin_support_size.values())))
        if per_basin_support_size
        else 0.0,
    }


def _compute_cosine_basin_similarity(
    *,
    model,
    dataset: BasinLabeledDataset,
    device: str,
    aggregation: str,
) -> Dict[str, float]:
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

    return _compute_cosine_metrics(
        np.stack(vectors, axis=0),
        np.array(basin_ids, dtype=np.int64),
        dataset.num_basins,
    )


def _encode_trajectories(
    model,
    dataset: BasinLabeledDataset,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for traj in dataset.trajectories:
            states = traj.states.to(device)
            z = model.encode(states)
            latents.append(z.mean(dim=0).cpu().numpy())
            labels.append(traj.final_basin)
    return np.stack(latents, axis=0), np.array(labels, dtype=np.int64)


def _compute_clustering_metrics(
    latents: np.ndarray,
    labels: np.ndarray,
    num_clusters: int,
) -> Dict[str, float]:
    sil_score = silhouette_score(latents, labels) if len(np.unique(labels)) > 1 else 0.0
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(latents)
    ari = adjusted_rand_score(labels, kmeans_labels)

    purity = 0.0
    for cluster_id in range(num_clusters):
        mask = kmeans_labels == cluster_id
        if mask.sum() > 0:
            cluster_labels = labels[mask]
            counts = np.bincount(cluster_labels)
            purity += counts.max()
    purity /= len(labels)

    return {
        "silhouette_score": float(sil_score),
        "adjusted_rand_index": float(ari),
        "kmeans_purity": float(purity),
    }


def _compute_separability_metrics(
    latents: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    if len(np.unique(labels)) < 2:
        return {
            "linear_classifier_accuracy": 1.0,
            "linear_classifier_cv_std": 0.0,
        }

    clf = LogisticRegression(max_iter=1000, random_state=42)
    min_samples = min(np.bincount(labels))
    n_folds = min(5, min_samples)
    if n_folds < 2:
        clf.fit(latents, labels)
        return {
            "linear_classifier_accuracy": float(clf.score(latents, labels)),
            "linear_classifier_cv_std": 0.0,
        }

    scores = cross_val_score(clf, latents, labels, cv=n_folds)
    return {
        "linear_classifier_accuracy": float(scores.mean()),
        "linear_classifier_cv_std": float(scores.std()),
    }


def _evaluate_entry(
    *,
    checkpoint_path: Path,
    checkpoint_dict: Dict,
    root_label: str,
    seed: int,
    system: str,
    device: str,
    dataset: BasinLabeledDataset,
    support_threshold: float,
    support_mode: str,
    cosine_aggregation: str,
) -> Dict:
    cfg = Config.from_dict(checkpoint_dict["config"])
    cfg.ENV.ENV_NAME = system

    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    model = model.to(device)
    model.eval()

    support = _compute_support_metrics(
        model=model,
        dataset=dataset,
        device=device,
        support_threshold=support_threshold,
        support_mode=support_mode,
    )
    cosine = _compute_cosine_basin_similarity(
        model=model,
        dataset=dataset,
        device=device,
        aggregation=cosine_aggregation,
    )
    latents_mean, labels = _encode_trajectories(
        model=model,
        dataset=dataset,
        device=device,
    )
    clustering = _compute_clustering_metrics(
        latents=latents_mean,
        labels=labels,
        num_clusters=dataset.num_basins,
    )
    separability = _compute_separability_metrics(
        latents=latents_mean,
        labels=labels,
    )

    return {
        "root_label": root_label,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "system": system,
        "num_trajectories": len(dataset.trajectories),
        "num_basins": dataset.num_basins,
        "basin_names": list(dataset.basin_names),
        "basin_distribution": {str(b): sum(1 for t in dataset.trajectories if t.final_basin == b) for b in range(dataset.num_basins)},
        "support_threshold": support_threshold,
        "support_mode": support_mode,
        "cosine_aggregation": cosine_aggregation,
        "metrics": {
            "mode_uniqueness_rate": float(support["mode_uniqueness_rate"]),
            "mean_basin_consistency": float(support["mean_basin_consistency"]),
            "mean_pairwise_jaccard": float(support["mean_pairwise_jaccard"]),
            "unique_mode_supports": int(support["unique_mode_supports"]),
            "mode_collision_pairs": int(support["mode_collision_pairs"]),
            "mean_mode_support_size": float(support["mean_mode_support_size"]),
            "mean_intra_basin_cosine": float(cosine["mean_intra_basin_cosine"]),
            "mean_inter_basin_cosine": float(cosine["mean_inter_basin_cosine"]),
            "cosine_separation_score": float(cosine["cosine_separation_score"]),
            "silhouette_score": float(clustering["silhouette_score"]),
            "adjusted_rand_index": float(clustering["adjusted_rand_index"]),
            "kmeans_purity": float(clustering["kmeans_purity"]),
            "linear_classifier_accuracy": float(
                separability["linear_classifier_accuracy"]
            ),
            "linear_classifier_cv_std": float(
                separability["linear_classifier_cv_std"]
            ),
        },
    }


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _aggregate(results: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for item in results:
        grouped[item["root_label"]].append(item)

    aggregates: Dict[str, Dict] = {}
    metric_names = list(results[0]["metrics"].keys()) if results else []
    for root_label, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: item["seed"])
        metrics = {
            name: _median([item["metrics"][name] for item in items])
            for name in metric_names
        }
        aggregates[root_label] = {
            "num_seeds": len(items),
            "seeds": [item["seed"] for item in items],
            "num_basins": items[0]["num_basins"],
            "basin_distribution": items[0]["basin_distribution"],
            "median_metrics": metrics,
        }
    return aggregates


def _write_markdown(
    *,
    summary_path: Path,
    results: List[Dict],
    aggregates: Dict[str, Dict],
) -> None:
    metric_columns = [
        "mode_uniqueness_rate",
        "mean_basin_consistency",
        "mean_pairwise_jaccard",
        "cosine_separation_score",
        "silhouette_score",
        "adjusted_rand_index",
        "kmeans_purity",
        "linear_classifier_accuracy",
    ]

    lines = [
        "# competitive_lv Support Alignment Summary",
        "",
        "## Root Medians",
        "",
        "| root | seeds | basins | uniqueness | consistency | jaccard | cosine_sep | silhouette | ari | purity | linear_acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for root_label, payload in sorted(aggregates.items()):
        m = payload["median_metrics"]
        lines.append(
            "| {root} | {num_seeds} | {num_basins} | {mode_uniqueness_rate:.4f} | "
            "{mean_basin_consistency:.4f} | {mean_pairwise_jaccard:.4f} | "
            "{cosine_separation_score:.4f} | {silhouette_score:.4f} | "
            "{adjusted_rand_index:.4f} | {kmeans_purity:.4f} | "
            "{linear_classifier_accuracy:.4f} |".format(
                root=root_label,
                num_seeds=payload["num_seeds"],
                num_basins=payload["num_basins"],
                **{name: m[name] for name in metric_columns},
            )
        )

    lines.extend(
        [
            "",
            "## Per-Seed Runs",
            "",
            "| root | seed | uniqueness | consistency | jaccard | cosine_sep | silhouette | ari | purity | linear_acc | checkpoint |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in sorted(results, key=lambda row: (row["root_label"], row["seed"])):
        m = item["metrics"]
        lines.append(
            "| {root} | {seed} | {mode_uniqueness_rate:.4f} | "
            "{mean_basin_consistency:.4f} | {mean_pairwise_jaccard:.4f} | "
            "{cosine_separation_score:.4f} | {silhouette_score:.4f} | "
            "{adjusted_rand_index:.4f} | {kmeans_purity:.4f} | "
            "{linear_classifier_accuracy:.4f} | `{checkpoint}` |".format(
                root=item["root_label"],
                seed=item["seed"],
                checkpoint=item["checkpoint_path"],
                **{name: m[name] for name in metric_columns},
            )
        )

    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate competitive_lv support alignment across explicit checkpoints."
    )
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help="Evaluation entry in ROOT_LABEL::SEED::CHECKPOINT_PATH format.",
    )
    parser.add_argument("--system", type=str, default="competitive_lv")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--trajectory-length", type=int, default=500)
    parser.add_argument("--long-rollout-steps", type=int, default=5000)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--support-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--support-mode",
        type=str,
        default="mean",
        choices=["mean", "last", "median", "majority"],
    )
    parser.add_argument(
        "--cosine-aggregation",
        type=str,
        default="mean",
        choices=["mean", "median", "mean_abs"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    if not args.entry:
        raise SystemExit("At least one --entry is required.")

    output_dir = Path(args.output_dir)
    per_run_dir = output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache: Dict[str, BasinLabeledDataset] = {}
    results: List[Dict] = []

    parsed_entries = [_parse_entry(raw) for raw in args.entry]
    print(f"Evaluating {len(parsed_entries)} checkpoints on {args.system}...")

    for root_label, seed, checkpoint_path in parsed_entries:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_dict = _load_checkpoint(checkpoint_path, args.device)
        cache_key = _dataset_cache_key(
            checkpoint_dict,
            args.system,
            args.num_trajectories,
            args.trajectory_length,
            args.long_rollout_steps,
            args.eval_seed,
        )

        if cache_key not in dataset_cache:
            _, dataset_cache[cache_key] = _build_dataset(
                checkpoint_dict,
                args.system,
                args.num_trajectories,
                args.trajectory_length,
                args.long_rollout_steps,
                args.eval_seed,
            )
        dataset = dataset_cache[cache_key]

        result = _evaluate_entry(
            checkpoint_path=checkpoint_path,
            checkpoint_dict=checkpoint_dict,
            root_label=root_label,
            seed=seed,
            system=args.system,
            device=args.device,
            dataset=dataset,
            support_threshold=args.support_threshold,
            support_mode=args.support_mode,
            cosine_aggregation=args.cosine_aggregation,
        )
        results.append(result)

        run_dir = per_run_dir / root_label / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"  {root_label} seed {seed}: "
            f"cosine_sep={result['metrics']['cosine_separation_score']:.4f}, "
            f"linear_acc={result['metrics']['linear_classifier_accuracy']:.4f}"
        )

    aggregates = _aggregate(results)
    summary = {
        "system": args.system,
        "num_entries": len(results),
        "num_trajectories": args.num_trajectories,
        "trajectory_length": args.trajectory_length,
        "long_rollout_steps": args.long_rollout_steps,
        "eval_seed": args.eval_seed,
        "support_threshold": args.support_threshold,
        "support_mode": args.support_mode,
        "cosine_aggregation": args.cosine_aggregation,
        "results": sorted(results, key=lambda item: (item["root_label"], item["seed"])),
        "aggregates": aggregates,
    }
    summary_json = output_dir / "competitive_lv_support_alignment_summary.json"
    summary_md = output_dir / "competitive_lv_support_alignment_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary_path=summary_md, results=results, aggregates=aggregates)

    print(f"Wrote summary JSON to {summary_json}")
    print(f"Wrote summary Markdown to {summary_md}")


if __name__ == "__main__":
    main()
