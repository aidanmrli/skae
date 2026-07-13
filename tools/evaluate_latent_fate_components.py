#!/usr/bin/env python3
"""Post-hoc continuous latent fate clustering diagnostic.

This is a control for stable support components. Instead of building a
transition graph over discrete support states, it clusters continuous latent
tail summaries for each trajectory and assigns every state on the trajectory
to that latent-fate cluster.

The unsupervised variant selects the number of clusters by silhouette score.
The oracle-k variant uses the represented benchmark basin count only as an
evaluation upper bound, never as a proposed training-time method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, make_env
from skae.model import make_model
from tools import reduce_transition_rich_interpretability_metrics as REDUCER

EPS = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--root_labels", default="mlp_zero_sparse_hardinit_basin_partition_control")
    parser.add_argument("--systems", default="")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--num_trajectories", type=int, default=64)
    parser.add_argument("--trajectory_length", type=int, default=96)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--tail_window", type=int, default=16)
    parser.add_argument("--max_clusters", type=int, default=12)
    parser.add_argument("--min_silhouette", type=float, default=0.05)
    parser.add_argument("--pca_components", type=int, default=16)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _system_aliases(system_key: str, system_name: str) -> set[str]:
    tail = system_key.split(":")[-1]
    return {
        system_key,
        system_name,
        tail,
        system_key.replace(":", "_"),
        system_name.replace(":", "_"),
        tail.replace(":", "_"),
    }


def _timestamp_key(run_dir: str) -> Tuple[str, str]:
    stem = Path(run_dir).name
    if len(stem) == 15 and stem[8] == "-":
        return stem, run_dir
    return "", run_dir


def _load_specs(
    rows_csv: Path,
    *,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
) -> List[Dict[str, object]]:
    root_set = set(root_labels)
    system_set = set(systems)
    seed_set = set(seeds)
    best: Dict[Tuple[str, str, int], Dict[str, str]] = {}

    with rows_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            root_label = str(row.get("root_label", "")).strip()
            if root_set and root_label not in root_set:
                continue
            system_key = str(row.get("system_key", "")).strip()
            system_name = str(row.get("system_name", system_key)).strip()
            if system_set and not (_system_aliases(system_key, system_name) & system_set):
                continue
            seed = int(row.get("seed", 0))
            if seed_set and seed not in seed_set:
                continue
            key = (root_label, system_key, seed)
            incumbent = best.get(key)
            if incumbent is None or _timestamp_key(row["run_dir"]) > _timestamp_key(incumbent["run_dir"]):
                best[key] = row

    specs = []
    for row in best.values():
        specs.append(
            {
                "root_label": row["root_label"],
                "system_key": row["system_key"],
                "system_name": row.get("system_name", row["system_key"]),
                "seed": int(row["seed"]),
                "run_dir": row["run_dir"],
            }
        )
    return sorted(specs, key=lambda item: (str(item["root_label"]), str(item["system_key"]), int(item["seed"])))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue())


def _load_model(checkpoint_path: Path, system_key: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = Config.from_dict(checkpoint["config"])
    cfg.ENV.ENV_NAME = system_key
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    load_model_state_dict_compat(model, checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return env, model


def _generate_trajectories(env, *, num_trajectories: int, trajectory_length: int, seed: int) -> torch.Tensor:
    vec_env = VectorWrapper(env, int(num_trajectories))
    rng = torch.Generator().manual_seed(int(seed))
    return vec_env.generate_sequence_batch(rng=rng, window_length=int(trajectory_length)).float()


def _encode(model, trajectories: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        flat = trajectories.reshape(-1, trajectories.shape[-1]).to(device)
        z = model.encode(flat).reshape(*trajectories.shape[:2], -1)
    return z.detach().cpu().numpy().astype(np.float32, copy=False)


def _entropy(counter: Counter[object]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        prob = float(count) / total
        entropy -= prob * math.log(prob + EPS)
    return entropy


def _conditional_entropy(x: Sequence[object], y: Sequence[object]) -> float:
    return _entropy(Counter(zip(x, y))) - _entropy(Counter(y))


def _nmi(x: Sequence[object], y: Sequence[object]) -> float:
    hx = _entropy(Counter(x))
    hy = _entropy(Counter(y))
    if hx <= 0.0 or hy <= 0.0:
        return 0.0
    mi = hx + hy - _entropy(Counter(zip(x, y)))
    return float(mi / max(math.sqrt(hx * hy), EPS))


def _dominant_accuracy(classes: Sequence[object], basins: Sequence[int]) -> Optional[float]:
    by_class: Dict[object, Counter[int]] = defaultdict(Counter)
    for cls, basin in zip(classes, basins):
        if int(basin) >= 0:
            by_class[cls][int(basin)] += 1
    if not by_class:
        return None
    class_to_basin = {cls: counter.most_common(1)[0][0] for cls, counter in by_class.items() if counter}
    total = 0
    correct = 0
    for cls, basin in zip(classes, basins):
        basin_i = int(basin)
        if basin_i < 0 or cls not in class_to_basin:
            continue
        total += 1
        correct += int(class_to_basin[cls] == basin_i)
    return float(correct) / float(total) if total else None


def _object_metrics(
    object_labels: np.ndarray,
    basin_labels: np.ndarray,
    subset_mask: np.ndarray,
    *,
    object_kind: str,
) -> Dict[str, object]:
    flat_obj = object_labels.reshape(-1)
    flat_basins = basin_labels.reshape(-1)
    flat_subset = subset_mask.reshape(-1)
    valid = np.logical_and(flat_subset, flat_basins >= 0)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return {
            "object_kind": object_kind,
            "eligible_state_count": 0,
            "assigned_state_count": 0,
            "coverage": 0.0,
            "h_basin_given_object": None,
            "h_object_given_basin": None,
            "object_basin_nmi": None,
            "dominant_basin_accuracy": None,
            "object_count": 0,
            "represented_basin_count": 0,
        }
    obj = flat_obj[valid].tolist()
    basins = [int(item) for item in flat_basins[valid].tolist()]
    return {
        "object_kind": object_kind,
        "eligible_state_count": valid_count,
        "assigned_state_count": valid_count,
        "coverage": 1.0,
        "h_basin_given_object": _conditional_entropy(basins, obj),
        "h_object_given_basin": _conditional_entropy(obj, basins),
        "object_basin_nmi": _nmi(obj, basins),
        "dominant_basin_accuracy": _dominant_accuracy(obj, basins),
        "object_count": int(len(set(obj))),
        "represented_basin_count": int(len(set(basins))),
    }


def _tail_features(latents: np.ndarray, tail_window: int) -> np.ndarray:
    tail = latents[:, -max(1, min(int(tail_window), latents.shape[1])) :, :]
    mean = tail.mean(axis=1)
    std = tail.std(axis=1)
    final = tail[:, -1, :]
    return np.concatenate([mean, std, final], axis=1).astype(np.float32, copy=False)


def _preprocess_features(features: np.ndarray, pca_components: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    n_components = min(int(pca_components), scaled.shape[0] - 1, scaled.shape[1])
    if n_components >= 2:
        return PCA(n_components=n_components, random_state=0).fit_transform(scaled)
    return scaled


def _kmeans_labels(features: np.ndarray, k: int, seed: int) -> np.ndarray:
    if k <= 1:
        return np.zeros(features.shape[0], dtype=np.int64)
    return KMeans(n_clusters=int(k), random_state=int(seed), n_init=20).fit_predict(features).astype(np.int64)


def _select_unsupervised_labels(
    features: np.ndarray,
    *,
    max_clusters: int,
    min_silhouette: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    max_k = min(int(max_clusters), features.shape[0] - 1)
    best_score = -1.0
    best_labels = np.zeros(features.shape[0], dtype=np.int64)
    best_k = 1
    scores: Dict[int, float] = {}
    for k in range(2, max_k + 1):
        labels = _kmeans_labels(features, k, seed)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(silhouette_score(features, labels))
        scores[int(k)] = score
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = int(k)
    if best_score < float(min_silhouette):
        return np.zeros(features.shape[0], dtype=np.int64), {
            "selected_k": 1,
            "silhouette": best_score if best_score >= 0.0 else None,
            "silhouette_scores": scores,
            "selection_rule": "silhouette_below_threshold",
        }
    return best_labels, {
        "selected_k": best_k,
        "silhouette": best_score,
        "silhouette_scores": scores,
        "selection_rule": "best_silhouette",
    }


def _evaluate_spec(spec: Dict[str, object], args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    checkpoint_path = Path(str(spec["run_dir"])) / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    env, model = _load_model(checkpoint_path, str(spec["system_key"]), args.device)
    trajectories = _generate_trajectories(
        env,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        seed=args.eval_seed + int(spec["seed"]),
    )
    basin_labels_t, centers, label_method = REDUCER._label_sequences_and_centers(
        env,
        trajectories,
        system_key=str(spec["system_key"]),
        endpoint_rollout_steps=args.endpoint_rollout_steps,
    )
    basin_labels = basin_labels_t.detach().cpu().numpy().astype(np.int64, copy=False)
    subsets = REDUCER._margin_subsets(
        trajectories,
        centers,
        basin_labels=basin_labels,
        depth_slice_mode="per_basin",
    )
    latents = _encode(model, trajectories, args.device)
    features = _preprocess_features(_tail_features(latents, args.tail_window), args.pca_components)

    unsup_labels, unsup_info = _select_unsupervised_labels(
        features,
        max_clusters=args.max_clusters,
        min_silhouette=args.min_silhouette,
        seed=args.eval_seed + int(spec["seed"]),
    )
    represented_basin_count = int(len(set(int(item) for item in basin_labels.reshape(-1).tolist() if int(item) >= 0)))
    oracle_k = max(1, min(represented_basin_count, features.shape[0]))
    oracle_labels = _kmeans_labels(features, oracle_k, args.eval_seed + int(spec["seed"]))

    label_sets = {
        "latent_tail_fate_unsupervised_k": (unsup_labels, unsup_info),
        "latent_tail_fate_oracle_k": (
            oracle_labels,
            {
                "selected_k": oracle_k,
                "silhouette": (
                    float(silhouette_score(features, oracle_labels))
                    if len(set(oracle_labels.tolist())) > 1
                    else None
                ),
                "selection_rule": "represented_basin_count_evaluation_upper_bound",
            },
        ),
    }

    rows: List[Dict[str, object]] = []
    for object_kind, (trajectory_labels, info) in label_sets.items():
        object_labels = np.repeat(trajectory_labels[:, None], basin_labels.shape[1], axis=1)
        for subset_name, subset_mask in subsets.items():
            metrics = _object_metrics(object_labels, basin_labels, subset_mask, object_kind=object_kind)
            rows.append(
                {
                    "root_label": spec["root_label"],
                    "system_key": spec["system_key"],
                    "system_name": spec["system_name"],
                    "seed": spec["seed"],
                    "run_dir": spec["run_dir"],
                    "checkpoint_path": str(checkpoint_path),
                    "subset": subset_name,
                    "label_method": label_method,
                    "tail_window": args.tail_window,
                    "feature_kind": "tail_mean_std_final",
                    "pca_components": features.shape[1],
                    "max_clusters": args.max_clusters,
                    "min_silhouette": args.min_silhouette,
                    "selected_k": info["selected_k"],
                    "selection_rule": info["selection_rule"],
                    "silhouette": info["silhouette"],
                    **metrics,
                }
            )
    diagnostic = {
        "root_label": spec["root_label"],
        "system_key": spec["system_key"],
        "seed": spec["seed"],
        "unsupervised": unsup_info,
        "oracle_k": oracle_k,
        "represented_basin_count_all_states": represented_basin_count,
    }
    return rows, diagnostic


def _summary_markdown(rows: Sequence[Dict[str, object]], failures: Sequence[Dict[str, object]]) -> str:
    def fmt(value: object) -> str:
        if value in (None, ""):
            return ""
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "# Latent Fate Component Diagnostic",
        "",
        f"- Rows: `{len(rows)}`",
        f"- Failures: `{len(failures)}`",
        "",
    ]
    if rows:
        groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[(str(row["root_label"]), str(row["object_kind"]), str(row["subset"]))].append(row)
        lines.extend(
            [
                "## Aggregate Metrics",
                "",
                "| root | object | subset | n | coverage | H(B|obj) | H(obj|B) | NMI | mean k | count match |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (root, object_kind, subset), group in sorted(groups.items()):
            def mean(name: str) -> Optional[float]:
                values = []
                for item in group:
                    value = item.get(name)
                    if value in (None, ""):
                        continue
                    values.append(float(value))
                return float(np.mean(values)) if values else None

            count_match = sum(
                int(float(str(item.get("object_count", -1))))
                == int(float(str(item.get("represented_basin_count", -2))))
                for item in group
            )
            lines.append(
                "| {root} | {object_kind} | {subset} | {n} | {coverage} | {hb} | {hc} | {nmi} | {k} | {match}/{n} |".format(
                    root=root,
                    object_kind=object_kind,
                    subset=subset,
                    n=len(group),
                    coverage=fmt(mean("coverage")),
                    hb=fmt(mean("h_basin_given_object")),
                    hc=fmt(mean("h_object_given_basin")),
                    nmi=fmt(mean("object_basin_nmi")),
                    k=fmt(mean("selected_k")),
                    match=count_match,
                )
            )
        lines.extend(["", "## Deep Rows", ""])
        lines.extend(
            [
                "| root | system | seed | object | k | basin count | H(B|obj) | H(obj|B) | NMI |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row.get("subset") != "deep":
                continue
            lines.append(
                "| {root} | {system} | {seed} | {object_kind} | {k} | {basins} | {hb} | {hc} | {nmi} |".format(
                    root=row.get("root_label", ""),
                    system=row.get("system_key", ""),
                    seed=row.get("seed", ""),
                    object_kind=row.get("object_kind", ""),
                    k=row.get("object_count", ""),
                    basins=row.get("represented_basin_count", ""),
                    hb=fmt(row.get("h_basin_given_object")),
                    hc=fmt(row.get("h_object_given_basin")),
                    nmi=fmt(row.get("object_basin_nmi")),
                )
            )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure.get('system_key')}` seed `{failure.get('seed')}`: {failure.get('error')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_specs(
        Path(args.rows_csv),
        root_labels=_parse_csv_strings(args.root_labels),
        systems=_parse_csv_strings(args.systems),
        seeds=_parse_csv_ints(args.seeds),
    )
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []

    for index, spec in enumerate(specs, start=1):
        print(f"[{index}/{len(specs)}] {spec['system_key']} seed={spec['seed']}", flush=True)
        try:
            spec_rows, spec_diagnostic = _evaluate_spec(spec, args)
            rows.extend(spec_rows)
            diagnostics.append(spec_diagnostic)
            _write_csv(output_dir / "latent_fate_component_rows.csv", rows)
            (output_dir / "latent_fate_component_diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2)
            )
        except Exception as exc:  # noqa: BLE001 - batch diagnostics should capture all errors.
            failures.append(
                {
                    "root_label": spec["root_label"],
                    "system_key": spec["system_key"],
                    "seed": spec["seed"],
                    "run_dir": spec["run_dir"],
                    "error": repr(exc),
                }
            )
            print(f"  failed: {exc!r}", flush=True)

    manifest = {
        "rows_csv": args.rows_csv,
        "output_dir": str(output_dir),
        "num_specs": len(specs),
        "num_rows": len(rows),
        "num_failures": len(failures),
        "arguments": vars(args),
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_csv(output_dir / "latent_fate_component_rows.csv", rows)
    _write_csv(output_dir / "latent_fate_component_failures.csv", failures)
    (output_dir / "latent_fate_component_summary.md").write_text(_summary_markdown(rows, failures))
    print(json.dumps({"rows": len(rows), "failures": len(failures), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
