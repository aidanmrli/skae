#!/usr/bin/env python3
"""Post-hoc stable support component evaluation.

This script builds a label-free dynamical support object from existing trained
checkpoints:

1. encode sampled trajectories,
2. compute high-resolution support states,
3. build a support-transition graph,
4. identify recurrent support components,
5. assign each support state by empirical absorption fate,
6. compare the resulting stable support components to benchmark basin labels
   and to the current greedy Jaccard support-family baseline.

The implementation is intentionally post-hoc and diagnostic. Basin labels are
used only for evaluation metrics, never to build the support components.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, make_env
from skae.model import make_model
from tools import reduce_transition_rich_interpretability_metrics as REDUCER

EPS = 1e-12
UNCERTAIN = -1


@dataclass(frozen=True)
class RunSpec:
    root_label: str
    system_key: str
    system_name: str
    seed: int
    run_dir: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--root_labels", default="lista_dense_signsplit_p256_hardinit_basin_partition")
    parser.add_argument("--systems", default="")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--support_definition", default="absolute:0.001")
    parser.add_argument(
        "--base_object",
        default="family",
        choices=("exact", "family"),
        help="High-resolution support nodes: exact masks or high-Jaccard families.",
    )
    parser.add_argument("--base_family_jaccard", type=float, default=0.8)
    parser.add_argument("--comparison_family_jaccard", type=float, default=0.5)
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=192)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--tail_window", type=int, default=32)
    parser.add_argument("--min_edge_count", type=int, default=2)
    parser.add_argument("--min_edge_probability", type=float, default=0.02)
    parser.add_argument("--max_recurrent_out_probability", type=float, default=0.05)
    parser.add_argument("--min_tail_count", type=int, default=8)
    parser.add_argument("--min_absorption_observations", type=int, default=8)
    parser.add_argument("--min_absorption_confidence", type=float, default=0.80)
    parser.add_argument("--fit_fraction", type=float, default=0.5)
    parser.add_argument("--ridge_lambda", type=float, default=1e-4)
    parser.add_argument("--min_operator_transitions", type=int, default=128)
    parser.add_argument("--linear_map", choices=("linear", "affine"), default="affine")
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
    path = Path(run_dir)
    stem = path.name
    if len(stem) == 15 and stem[8] == "-":
        return stem, run_dir
    return "", run_dir


def _load_specs(
    rows_csv: Path,
    *,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
) -> List[RunSpec]:
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

    specs = [
        RunSpec(
            root_label=row["root_label"],
            system_key=row["system_key"],
            system_name=row.get("system_name", row["system_key"]),
            seed=int(row["seed"]),
            run_dir=row["run_dir"],
        )
        for row in best.values()
    ]
    return sorted(specs, key=lambda spec: (spec.root_label, spec.system_key, spec.seed))


def _parse_support_definition(raw: str) -> Tuple[str, float]:
    if ":" not in raw:
        raise ValueError(f"Support definition must be scheme:value, got {raw!r}")
    scheme, value = raw.split(":", 1)
    scheme = scheme.strip()
    if scheme not in {"absolute", "relative", "topk"}:
        raise ValueError(f"Unsupported support scheme {scheme!r}")
    return scheme, float(value)


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
    for row in rows:
        writer.writerow(row)
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
    return cfg, env, model


def _generate_trajectories(env, *, num_trajectories: int, trajectory_length: int, seed: int) -> torch.Tensor:
    vec_env = VectorWrapper(env, int(num_trajectories))
    rng = torch.Generator().manual_seed(int(seed))
    return vec_env.generate_sequence_batch(rng=rng, window_length=int(trajectory_length)).float()


def _encode(model, trajectories: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        flat = trajectories.reshape(-1, trajectories.shape[-1]).to(device)
        z = model.encode(flat).reshape(*trajectories.shape[:2], -1)
    return z.detach().cpu().numpy().astype(np.float32, copy=False)


def _remap_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[int, object]]:
    flat = labels.reshape(-1)
    mapping: Dict[object, int] = {}
    reverse: Dict[int, object] = {}
    out = np.empty(flat.shape[0], dtype=np.int64)
    for idx, item in enumerate(flat.tolist()):
        if item not in mapping:
            new_id = len(mapping)
            mapping[item] = new_id
            reverse[new_id] = item
        out[idx] = mapping[item]
    return out.reshape(labels.shape), reverse


def _strongly_connected_components(nodes: Iterable[int], adjacency: Dict[int, List[int]]) -> List[List[int]]:
    index = 0
    stack: List[int] = []
    on_stack: set[int] = set()
    indices: Dict[int, int] = {}
    lowlink: Dict[int, int] = {}
    components: List[List[int]] = []

    def visit(node: int) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for nxt in adjacency.get(node, []):
            if nxt not in indices:
                visit(nxt)
                lowlink[node] = min(lowlink[node], lowlink[nxt])
            elif nxt in on_stack:
                lowlink[node] = min(lowlink[node], indices[nxt])

        if lowlink[node] == indices[node]:
            component = []
            while True:
                popped = stack.pop()
                on_stack.remove(popped)
                component.append(popped)
                if popped == node:
                    break
            components.append(component)

    for node in sorted(set(nodes)):
        if node not in indices:
            visit(node)
    return components


def _stable_support_components(
    base_labels: np.ndarray,
    *,
    tail_window: int,
    min_edge_count: int,
    min_edge_probability: float,
    max_recurrent_out_probability: float,
    min_tail_count: int,
    min_absorption_observations: int,
    min_absorption_confidence: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if base_labels.ndim != 2:
        raise ValueError("base_labels must have shape [num_trajectories, trajectory_length]")

    transition_counts: Dict[int, Counter[int]] = defaultdict(Counter)
    source_counts: Counter[int] = Counter()
    for src, dst in zip(base_labels[:, :-1].reshape(-1).tolist(), base_labels[:, 1:].reshape(-1).tolist()):
        src_i = int(src)
        dst_i = int(dst)
        transition_counts[src_i][dst_i] += 1
        source_counts[src_i] += 1

    nodes = {int(item) for item in base_labels.reshape(-1).tolist()}
    adjacency: Dict[int, List[int]] = defaultdict(list)
    for src, dst_counter in transition_counts.items():
        total = float(sum(dst_counter.values()))
        if total <= 0.0:
            continue
        for dst, count in dst_counter.items():
            if int(count) >= int(min_edge_count) and float(count) / total >= float(min_edge_probability):
                adjacency[int(src)].append(int(dst))

    sccs = _strongly_connected_components(nodes, adjacency)

    tail_width = max(1, min(int(tail_window), base_labels.shape[1]))
    tail_counts = Counter(int(item) for item in base_labels[:, -tail_width:].reshape(-1).tolist())
    recurrent_components: List[List[int]] = []
    recurrent_stats: List[Dict[str, object]] = []
    for component in sccs:
        comp_set = set(component)
        total = 0
        internal = 0
        for src in component:
            dst_counter = transition_counts.get(src, Counter())
            total += sum(dst_counter.values())
            internal += sum(count for dst, count in dst_counter.items() if int(dst) in comp_set)
        outbound = 0.0 if total == 0 else 1.0 - (float(internal) / float(total))
        tail_count = sum(tail_counts.get(int(node), 0) for node in component)
        if tail_count >= int(min_tail_count) and outbound <= float(max_recurrent_out_probability):
            recurrent_components.append(sorted(component))
            recurrent_stats.append(
                {
                    "nodes": sorted(component),
                    "tail_count": int(tail_count),
                    "out_probability": float(outbound),
                    "transition_count": int(total),
                }
            )

    if not recurrent_components:
        for node, count in tail_counts.most_common():
            if int(count) < int(min_tail_count):
                continue
            recurrent_components.append([int(node)])
            recurrent_stats.append(
                {
                    "nodes": [int(node)],
                    "tail_count": int(count),
                    "out_probability": None,
                    "transition_count": int(source_counts.get(int(node), 0)),
                    "fallback_singleton_tail_node": True,
                }
            )

    recurrent_by_node: Dict[int, int] = {}
    for comp_idx, component in enumerate(recurrent_components):
        for node in component:
            recurrent_by_node[int(node)] = int(comp_idx)

    absorption: Dict[int, Counter[int]] = defaultdict(Counter)
    for trajectory in base_labels:
        future_component: Optional[int] = None
        for label in reversed(trajectory.tolist()):
            node = int(label)
            if node in recurrent_by_node:
                future_component = recurrent_by_node[node]
            if future_component is not None:
                absorption[node][future_component] += 1

    node_to_stable: Dict[int, int] = {}
    confidence_by_node: Dict[int, float] = {}
    for node, counter in absorption.items():
        total = sum(counter.values())
        if total < int(min_absorption_observations):
            continue
        best_component, best_count = counter.most_common(1)[0]
        confidence = float(best_count) / float(total)
        confidence_by_node[int(node)] = confidence
        if confidence >= float(min_absorption_confidence):
            node_to_stable[int(node)] = int(best_component)

    stable = np.full(base_labels.shape, UNCERTAIN, dtype=np.int64)
    for node, component in node_to_stable.items():
        stable[base_labels == int(node)] = int(component)

    diagnostics = {
        "base_node_count": int(len(nodes)),
        "edge_count": int(sum(len(values) for values in adjacency.values())),
        "scc_count": int(len(sccs)),
        "recurrent_component_count": int(len(recurrent_components)),
        "assigned_base_node_count": int(len(node_to_stable)),
        "uncertain_base_node_count": int(len(nodes) - len(node_to_stable)),
        "mean_absorption_confidence": (
            float(np.mean(list(confidence_by_node.values()))) if confidence_by_node else None
        ),
        "recurrent_components": recurrent_stats,
    }
    return stable, diagnostics


def _entropy(counter: Counter[object]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    out = 0.0
    for count in counter.values():
        prob = float(count) / total
        out -= prob * math.log(prob + EPS)
    return out


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
    if not class_to_basin:
        return None
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
    eligible = np.logical_and(flat_subset, flat_basins >= 0)
    valid = np.logical_and(eligible, flat_obj != UNCERTAIN)
    eligible_count = int(eligible.sum())
    valid_count = int(valid.sum())
    coverage = float(valid_count) / float(eligible_count) if eligible_count else 0.0
    if valid_count == 0:
        return {
            "object_kind": object_kind,
            "eligible_state_count": eligible_count,
            "assigned_state_count": valid_count,
            "coverage": coverage,
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
        "eligible_state_count": eligible_count,
        "assigned_state_count": valid_count,
        "coverage": coverage,
        "h_basin_given_object": _conditional_entropy(basins, obj),
        "h_object_given_basin": _conditional_entropy(obj, basins),
        "object_basin_nmi": _nmi(obj, basins),
        "dominant_basin_accuracy": _dominant_accuracy(obj, basins),
        "object_count": int(len(set(obj))),
        "represented_basin_count": int(len(set(basins))),
    }


def _fit_map(x: np.ndarray, y: np.ndarray, *, ridge_lambda: float, affine: bool) -> Optional[np.ndarray]:
    if x.shape[0] == 0 or y.shape[0] == 0 or x.shape[0] != y.shape[0]:
        return None
    design = x
    if affine:
        design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    reg = float(ridge_lambda) * np.eye(design.shape[1], dtype=np.float64)
    lhs = design.astype(np.float64).T @ design.astype(np.float64) + reg
    rhs = design.astype(np.float64).T @ y.astype(np.float64)
    try:
        return np.linalg.solve(lhs, rhs).astype(np.float32)
    except np.linalg.LinAlgError:
        return None


def _apply_map(x: np.ndarray, operator: np.ndarray, *, affine: bool) -> np.ndarray:
    if affine:
        design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        return design @ operator
    return x @ operator


def _local_operator_probe(
    latents: np.ndarray,
    object_labels: np.ndarray,
    global_k: np.ndarray,
    *,
    fit_fraction: float,
    ridge_lambda: float,
    min_transitions: int,
    affine: bool,
) -> Dict[str, object]:
    num_traj = int(latents.shape[0])
    split = int(round(num_traj * float(fit_fraction)))
    split = min(max(split, 1), max(num_traj - 1, 1))

    x_fit = latents[:split, :-1, :].reshape(-1, latents.shape[-1])
    y_fit = latents[:split, 1:, :].reshape(-1, latents.shape[-1])
    c_fit = object_labels[:split, :-1].reshape(-1)

    operators: Dict[int, np.ndarray] = {}
    transition_counts = Counter(int(item) for item in c_fit.tolist() if int(item) != UNCERTAIN)
    for class_id, count in transition_counts.items():
        if int(count) < int(min_transitions):
            continue
        mask = c_fit == int(class_id)
        operator = _fit_map(x_fit[mask], y_fit[mask], ridge_lambda=ridge_lambda, affine=affine)
        if operator is not None:
            operators[int(class_id)] = operator

    x_eval = latents[split:, :-1, :].reshape(-1, latents.shape[-1])
    y_eval = latents[split:, 1:, :].reshape(-1, latents.shape[-1])
    c_eval = object_labels[split:, :-1].reshape(-1)
    if x_eval.shape[0] == 0:
        return {
            "operator_count": len(operators),
            "operator_eval_transition_count": 0,
            "operator_local_coverage": 0.0,
            "latent_mse_global": None,
            "latent_mse_local_or_fallback": None,
            "latent_mse_ratio_local_over_global": None,
        }

    global_pred = x_eval @ global_k
    local_pred = global_pred.copy()
    local_used = np.zeros(x_eval.shape[0], dtype=bool)
    for class_id, operator in operators.items():
        mask = c_eval == int(class_id)
        if not np.any(mask):
            continue
        local_pred[mask] = _apply_map(x_eval[mask], operator, affine=affine)
        local_used[mask] = True

    global_mse = float(np.mean((global_pred - y_eval) ** 2))
    local_mse = float(np.mean((local_pred - y_eval) ** 2))
    return {
        "operator_count": int(len(operators)),
        "operator_eval_transition_count": int(x_eval.shape[0]),
        "operator_local_coverage": float(local_used.mean()) if local_used.size else 0.0,
        "latent_mse_global": global_mse,
        "latent_mse_local_or_fallback": local_mse,
        "latent_mse_ratio_local_over_global": local_mse / max(global_mse, EPS),
    }


def _summary_markdown(rows: Sequence[Dict[str, object]], failures: Sequence[Dict[str, object]]) -> str:
    def fmt(value: object) -> str:
        if value in (None, ""):
            return ""
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "# Stable Support Component Post-Hoc Trial",
        "",
        f"- Rows: `{len(rows)}`",
        f"- Failures: `{len(failures)}`",
        "",
    ]
    if rows:
        aggregate: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            key = (
                str(row.get("root_label", "")),
                str(row.get("object_kind", "")),
                str(row.get("subset", "")),
            )
            aggregate[key].append(row)
        lines.extend(
            [
                "## Aggregate Metrics",
                "",
                "| root | object | subset | n | coverage | H(B|obj) | H(obj|B) | NMI | count match | local/global |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (root, object_kind, subset), group in sorted(aggregate.items()):
            def mean_field(name: str) -> float:
                values = [float(row[name]) for row in group if row.get(name) not in (None, "")]
                return float(np.mean(values)) if values else float("nan")

            count_match = sum(
                int(row.get("object_count", -1)) == int(row.get("represented_basin_count", -2))
                for row in group
            )
            lines.append(
                "| {root} | {obj} | {subset} | {n} | {cov} | {hb} | {hc} | {nmi} | {match}/{n} | {ratio} |".format(
                    root=root,
                    obj=object_kind,
                    subset=subset,
                    n=len(group),
                    cov=fmt(mean_field("coverage")),
                    hb=fmt(mean_field("h_basin_given_object")),
                    hc=fmt(mean_field("h_object_given_basin")),
                    nmi=fmt(mean_field("object_basin_nmi")),
                    match=count_match,
                    ratio=fmt(mean_field("latent_mse_ratio_local_over_global")),
                )
            )
        lines.extend(["", "## Deep-Slice Rows", ""])
        lines.extend(
            [
                "| root | system | seed | object | subset | coverage | H(B|obj) | H(obj|B) | count | local/global |",
                "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if row.get("subset") != "deep":
                continue
            lines.append(
                "| {root} | {system} | {seed} | {obj} | {subset} | {cov} | {hb} | {hc} | {count} | {ratio} |".format(
                    root=row.get("root_label", ""),
                    system=row.get("system_key", ""),
                    seed=row.get("seed", ""),
                    obj=row.get("object_kind", ""),
                    subset=row.get("subset", ""),
                    cov=fmt(row.get("coverage")),
                    hb=fmt(row.get("h_basin_given_object")),
                    hc=fmt(row.get("h_object_given_basin")),
                    count=fmt(row.get("object_count")),
                    ratio=fmt(row.get("latent_mse_ratio_local_over_global")),
                )
            )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure.get('system_key')}` seed `{failure.get('seed')}`: {failure.get('error')}")
    return "\n".join(lines) + "\n"


def _evaluate_spec(spec: RunSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    scheme, support_value = _parse_support_definition(args.support_definition)
    _cfg, env, model = _load_model(checkpoint_path, spec.system_key, args.device)
    trajectories = _generate_trajectories(
        env,
        num_trajectories=args.num_trajectories,
        trajectory_length=args.trajectory_length,
        seed=args.eval_seed + spec.seed,
    )
    basin_labels_t, centers, label_method = REDUCER._label_sequences_and_centers(
        env,
        trajectories,
        system_key=spec.system_key,
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
    support_mask = REDUCER._support_mask(latents, scheme=scheme, value=support_value)

    comparison_labels = REDUCER.support_family_labels(
        support_mask,
        min_jaccard=float(args.comparison_family_jaccard),
    ).astype(np.int64, copy=False)

    if args.base_object == "exact":
        base_raw = REDUCER._support_keys(support_mask)
        base_labels, _reverse = _remap_labels(base_raw)
    else:
        base_labels = REDUCER.support_family_labels(
            support_mask,
            min_jaccard=float(args.base_family_jaccard),
        ).astype(np.int64, copy=False)

    stable_labels, stable_diagnostics = _stable_support_components(
        base_labels,
        tail_window=args.tail_window,
        min_edge_count=args.min_edge_count,
        min_edge_probability=args.min_edge_probability,
        max_recurrent_out_probability=args.max_recurrent_out_probability,
        min_tail_count=args.min_tail_count,
        min_absorption_observations=args.min_absorption_observations,
        min_absorption_confidence=args.min_absorption_confidence,
    )

    global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
    affine = args.linear_map == "affine"
    operator_by_object = {
        "current_fabs": _local_operator_probe(
            latents,
            comparison_labels,
            global_k,
            fit_fraction=args.fit_fraction,
            ridge_lambda=args.ridge_lambda,
            min_transitions=args.min_operator_transitions,
            affine=affine,
        ),
        "stable_support_component": _local_operator_probe(
            latents,
            stable_labels,
            global_k,
            fit_fraction=args.fit_fraction,
            ridge_lambda=args.ridge_lambda,
            min_transitions=args.min_operator_transitions,
            affine=affine,
        ),
    }

    rows: List[Dict[str, object]] = []
    for object_kind, labels in (
        ("current_fabs", comparison_labels),
        ("stable_support_component", stable_labels),
    ):
        operator_metrics = operator_by_object[object_kind]
        for subset_name, subset_mask in subsets.items():
            metrics = _object_metrics(labels, basin_labels, subset_mask, object_kind=object_kind)
            row: Dict[str, object] = {
                "root_label": spec.root_label,
                "system_key": spec.system_key,
                "system_name": spec.system_name,
                "seed": spec.seed,
                "run_dir": spec.run_dir,
                "checkpoint_path": str(checkpoint_path),
                "support_definition": args.support_definition,
                "base_object": args.base_object,
                "base_family_jaccard": args.base_family_jaccard,
                "comparison_family_jaccard": args.comparison_family_jaccard,
                "subset": subset_name,
                "label_method": label_method,
                "linear_map": args.linear_map,
                **metrics,
                **operator_metrics,
            }
            if object_kind == "stable_support_component":
                row.update(
                    {
                        "stable_base_node_count": stable_diagnostics["base_node_count"],
                        "stable_edge_count": stable_diagnostics["edge_count"],
                        "stable_scc_count": stable_diagnostics["scc_count"],
                        "stable_recurrent_component_count": stable_diagnostics[
                            "recurrent_component_count"
                        ],
                        "stable_assigned_base_node_count": stable_diagnostics[
                            "assigned_base_node_count"
                        ],
                        "stable_uncertain_base_node_count": stable_diagnostics[
                            "uncertain_base_node_count"
                        ],
                        "stable_mean_absorption_confidence": stable_diagnostics[
                            "mean_absorption_confidence"
                        ],
                    }
                )
            rows.append(row)

    diagnostic = {
        "root_label": spec.root_label,
        "system_key": spec.system_key,
        "seed": spec.seed,
        "stable_diagnostics": stable_diagnostics,
    }
    return rows, diagnostic


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
        print(f"[{index}/{len(specs)}] {spec.system_key} seed={spec.seed}", flush=True)
        try:
            spec_rows, spec_diagnostic = _evaluate_spec(spec, args)
            rows.extend(spec_rows)
            diagnostics.append(spec_diagnostic)
            _write_csv(output_dir / "stable_support_component_rows.csv", rows)
            (output_dir / "stable_support_component_diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2)
            )
        except Exception as exc:  # noqa: BLE001 - write all failures for batch diagnostics.
            failures.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
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
    _write_csv(output_dir / "stable_support_component_rows.csv", rows)
    _write_csv(output_dir / "stable_support_component_failures.csv", failures)
    (output_dir / "stable_support_component_summary.md").write_text(_summary_markdown(rows, failures))
    print(json.dumps({"rows": len(rows), "failures": len(failures), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
