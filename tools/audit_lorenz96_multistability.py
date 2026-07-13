#!/usr/bin/env python3
"""Numerically audit Lorenz-96 parameter pairs for candidate multistability.

The script is intentionally conservative: it does not declare a Lorenz-96
system multibasin from theory or from a single trajectory. For each requested
``(D, F)`` pair, it launches several diverse initial-condition families, runs a
long transient, summarizes the asymptotic tail using translation-invariant
statistics, and clusters the resulting tail features. A pair is only marked as
``candidate_multistable`` when at least two reproducible asymptotic clusters are
found under the configured silhouette and cluster-size thresholds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from experiments.benchmark_suite.data import rk4_step_lorenz96


@dataclass(frozen=True)
class AuditConfig:
    dimensions: List[int]
    forcings: List[float]
    seed: int = 0
    initials_per_pair: int = 48
    dt: float = 0.005
    sample_every: int = 10
    warmup_observations: int = 600
    tail_observations: int = 256
    initial_scale: float = 2.0
    max_clusters: int = 6
    min_cluster_size: int = 3
    min_silhouette: float = 0.25


def _parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dimensions", default="4,5,6,8,16,32")
    parser.add_argument("--forcings", default="0.5,0.75,1,1.25,1.5,2,2.5,3,4,5,6,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initials_per_pair", type=int, default=48)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--warmup_observations", type=int, default=600)
    parser.add_argument("--tail_observations", type=int, default=256)
    parser.add_argument("--initial_scale", type=float, default=2.0)
    parser.add_argument("--max_clusters", type=int, default=6)
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--min_silhouette", type=float, default=0.25)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def make_initial_state(
    *,
    dimension: int,
    forcing: float,
    index: int,
    rng: np.random.Generator,
    scale: float,
) -> tuple[np.ndarray, str]:
    grid = np.arange(dimension, dtype=np.float32)
    family = index % 5
    if family == 0:
        return (forcing + scale * rng.standard_normal(dimension)).astype(np.float32), "random_gaussian"
    if family == 1:
        mode = 1 + (index // 5) % max(1, dimension // 2)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        state = forcing + scale * np.cos(2.0 * np.pi * mode * grid / dimension + phase)
        return state.astype(np.float32), f"cos_mode_{mode}"
    if family == 2:
        mode = 1 + (index // 5) % max(1, dimension // 2)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        state = forcing + scale * np.sin(2.0 * np.pi * mode * grid / dimension + phase)
        return state.astype(np.float32), f"sin_mode_{mode}"
    if family == 3:
        state = forcing + 0.05 * rng.standard_normal(dimension)
        width = max(1, dimension // 16)
        center = int(rng.integers(0, dimension))
        for offset in range(-width, width + 1):
            state[(center + offset) % dimension] += scale * rng.choice([-1.0, 1.0])
        return state.astype(np.float32), "localized_pulse"
    state = rng.uniform(-abs(forcing) - scale, abs(forcing) + scale, size=dimension)
    return state.astype(np.float32), "wide_uniform"


def simulate_tail(
    initial: np.ndarray,
    *,
    forcing: float,
    dt: float,
    sample_every: int,
    warmup_observations: int,
    tail_observations: int,
) -> np.ndarray:
    state = initial.astype(np.float32, copy=True)
    for _ in range(int(warmup_observations) * int(sample_every)):
        state = rk4_step_lorenz96(state, forcing=forcing, dt=dt)
    tail = np.empty((int(tail_observations), initial.shape[0]), dtype=np.float32)
    for t in range(int(tail_observations)):
        tail[t] = state
        for _ in range(int(sample_every)):
            state = rk4_step_lorenz96(state, forcing=forcing, dt=dt)
    return tail


def _safe_entropy(power: np.ndarray) -> float:
    total = float(np.sum(power))
    if total <= 1e-12:
        return 0.0
    p = np.asarray(power, dtype=np.float64) / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)) / np.log(max(2, p.size)))


def classify_tail(tail: np.ndarray) -> str:
    centered = tail - tail.mean(axis=0, keepdims=True)
    temporal_std = float(centered.std())
    if temporal_std < 1e-4:
        return "steady"
    spatial_mean = tail.mean(axis=1)
    power = np.abs(np.fft.rfft(spatial_mean - spatial_mean.mean())) ** 2
    entropy = _safe_entropy(power[1:])
    normalized_tail = centered / max(float(centered.std()), 1e-6)
    final = normalized_tail[-1]
    previous = normalized_tail[:-8] if normalized_tail.shape[0] > 16 else normalized_tail[:-1]
    recurrence = float(np.sqrt(np.mean((previous - final[None, :]) ** 2, axis=1)).min()) if previous.size else float("inf")
    if recurrence < 0.1 and entropy < 0.35:
        return "periodic_candidate"
    if entropy < 0.65:
        return "quasiperiodic_or_few_frequency_candidate"
    return "broadband_or_chaotic_candidate"


def tail_features(tail: np.ndarray, *, max_spatial_modes: int = 10, max_temporal_modes: int = 10) -> np.ndarray:
    tail = np.asarray(tail, dtype=np.float64)
    centered = tail - tail.mean(axis=0, keepdims=True)
    energy = np.mean(tail**2)
    temporal_std = np.mean(tail.std(axis=0))
    spatial_mean = tail.mean(axis=1)
    spatial_std = tail.std(axis=1)

    spatial_power = np.abs(np.fft.rfft(tail, axis=1)) ** 2
    spatial_power = spatial_power.mean(axis=0)
    spatial_power = spatial_power[1 : max_spatial_modes + 1]
    spatial_power = spatial_power / max(float(spatial_power.sum()), 1e-12)

    temporal_power = np.abs(np.fft.rfft(spatial_mean - spatial_mean.mean())) ** 2
    temporal_power = temporal_power[1 : max_temporal_modes + 1]
    temporal_power = temporal_power / max(float(temporal_power.sum()), 1e-12)

    return np.concatenate(
        [
            np.asarray(
                [
                    tail.mean(),
                    tail.std(),
                    energy,
                    temporal_std,
                    spatial_mean.std(),
                    spatial_std.mean(),
                    _safe_entropy(spatial_power),
                    _safe_entropy(temporal_power),
                    centered[-1].std(),
                ],
                dtype=np.float64,
            ),
            spatial_power,
            temporal_power,
        ]
    )


def choose_clusters(
    features: np.ndarray,
    *,
    max_clusters: int,
    min_cluster_size: int,
) -> tuple[np.ndarray, int, float]:
    n = int(features.shape[0])
    if n < 2 * int(min_cluster_size):
        return np.zeros(n, dtype=np.int64), 1, float("nan")
    scaled = StandardScaler().fit_transform(features)
    best_labels = np.zeros(n, dtype=np.int64)
    best_k = 1
    best_score = float("-inf")
    for k in range(2, min(int(max_clusters), n - 1) + 1):
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(scaled)
        counts = np.bincount(labels)
        if int(counts.min()) < int(min_cluster_size):
            continue
        score = float(silhouette_score(scaled, labels))
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels.astype(np.int64)
    if best_k == 1:
        return np.zeros(n, dtype=np.int64), 1, float("nan")
    return best_labels, best_k, best_score


def summarize_pair(
    *,
    dimension: int,
    forcing: float,
    cfg: AuditConfig,
    rng: np.random.Generator,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    feature_rows: list[dict[str, object]] = []
    feature_vectors: list[np.ndarray] = []
    for init_idx in range(int(cfg.initials_per_pair)):
        initial, family = make_initial_state(
            dimension=dimension,
            forcing=forcing,
            index=init_idx,
            rng=rng,
            scale=float(cfg.initial_scale),
        )
        tail = simulate_tail(
            initial,
            forcing=forcing,
            dt=float(cfg.dt),
            sample_every=int(cfg.sample_every),
            warmup_observations=int(cfg.warmup_observations),
            tail_observations=int(cfg.tail_observations),
        )
        finite = bool(np.isfinite(tail).all())
        vector = tail_features(tail if finite else np.nan_to_num(tail, nan=0.0, posinf=0.0, neginf=0.0))
        feature_vectors.append(vector)
        feature_rows.append(
            {
                "dimension": int(dimension),
                "forcing": float(forcing),
                "initial_index": int(init_idx),
                "initial_family": family,
                "finite": finite,
                "tail_type": classify_tail(tail) if finite else "nonfinite",
                "tail_mean": float(np.nanmean(tail)),
                "tail_std": float(np.nanstd(tail)),
                "tail_energy": float(np.nanmean(tail**2)),
            }
        )

    features = np.stack(feature_vectors, axis=0)
    finite_mask = np.isfinite(features).all(axis=1)
    labels = np.full(int(cfg.initials_per_pair), -1, dtype=np.int64)
    if int(finite_mask.sum()) >= max(2, 2 * int(cfg.min_cluster_size)):
        finite_labels, k, silhouette = choose_clusters(
            features[finite_mask],
            max_clusters=int(cfg.max_clusters),
            min_cluster_size=int(cfg.min_cluster_size),
        )
        labels[finite_mask] = finite_labels
    else:
        k = 0
        silhouette = float("nan")
    for row, label in zip(feature_rows, labels):
        row["cluster"] = int(label)

    cluster_counts = {str(int(label)): int((labels == label).sum()) for label in sorted(set(labels.tolist()))}
    tail_type_by_cluster: dict[str, dict[str, int]] = {}
    for label in sorted(set(labels.tolist())):
        if label < 0:
            continue
        type_counts: dict[str, int] = {}
        for row in feature_rows:
            if int(row["cluster"]) == int(label):
                key = str(row["tail_type"])
                type_counts[key] = type_counts.get(key, 0) + 1
        tail_type_by_cluster[str(int(label))] = type_counts
    candidate = bool(k >= 2 and np.isfinite(silhouette) and silhouette >= float(cfg.min_silhouette))
    summary = {
        "dimension": int(dimension),
        "forcing": float(forcing),
        "finite_trajectories": int(finite_mask.sum()),
        "initials_per_pair": int(cfg.initials_per_pair),
        "num_clusters": int(k),
        "silhouette": float(silhouette),
        "candidate_multistable": candidate,
        "cluster_counts": cluster_counts,
        "tail_type_by_cluster": tail_type_by_cluster,
    }
    return summary, feature_rows, features


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    scatter = ax.scatter(
        summary["forcing"],
        summary["dimension"],
        c=summary["num_clusters"],
        s=np.where(summary["candidate_multistable"], 95, 45),
        cmap="viridis",
        vmin=1,
        vmax=max(2, int(summary["num_clusters"].max())),
        edgecolor=np.where(summary["candidate_multistable"], "black", "none"),
        linewidth=0.8,
    )
    ax.set_xlabel("Lorenz-96 forcing F")
    ax.set_ylabel("state dimension D")
    ax.set_title("Lorenz-96 candidate multistability audit")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("asymptotic feature clusters")
    ax.grid(True, alpha=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"lorenz96_multistability_audit.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = AuditConfig(
        dimensions=_parse_ints(args.dimensions),
        forcings=_parse_floats(args.forcings),
        seed=int(args.seed),
        initials_per_pair=int(args.initials_per_pair),
        dt=float(args.dt),
        sample_every=int(args.sample_every),
        warmup_observations=int(args.warmup_observations),
        tail_observations=int(args.tail_observations),
        initial_scale=float(args.initial_scale),
        max_clusters=int(args.max_clusters),
        min_cluster_size=int(args.min_cluster_size),
        min_silhouette=float(args.min_silhouette),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(cfg.seed))
    summary_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    for dimension in cfg.dimensions:
        for forcing in cfg.forcings:
            summary, features, _vectors = summarize_pair(
                dimension=int(dimension),
                forcing=float(forcing),
                cfg=cfg,
                rng=rng,
            )
            summary_rows.append(summary)
            feature_rows.extend(features)
            print(json.dumps(summary, sort_keys=True), flush=True)

    summary_df = pd.DataFrame(summary_rows)
    feature_df = pd.DataFrame(feature_rows)
    summary_df.to_csv(output_dir / "lorenz96_multistability_summary.csv", index=False)
    feature_df.to_csv(output_dir / "lorenz96_multistability_features.csv", index=False)
    plot_summary(summary_df, output_dir)
    manifest = {
        "status": "completed",
        "config": asdict(cfg),
        "num_parameter_pairs": int(len(summary_rows)),
        "num_candidate_multistable_pairs": int(summary_df["candidate_multistable"].sum()) if not summary_df.empty else 0,
        "candidate_pairs": summary_df[summary_df["candidate_multistable"]][
            ["dimension", "forcing", "num_clusters", "silhouette"]
        ].to_dict(orient="records")
        if not summary_df.empty
        else [],
        "caution": (
            "Candidate labels are numerical screening results. A pair should only "
            "be promoted to a benchmark after rerunning with longer transients, "
            "more initial conditions, and qualitative inspection of representative tails."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
