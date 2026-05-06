#!/usr/bin/env python3
"""Sweep support-family Jaccard thresholds on existing trained runs.

This is a lightweight companion to
``tools/reduce_transition_rich_interpretability_metrics.py``.  It keeps the
same state generation, basin labeling, support-mask extraction, and greedy
support-family construction, but skips the expensive intervention, Jacobian,
and local-operator diagnostics.  The intended use is threshold sensitivity:
how much do ``F_abs`` and ``F_top8`` basin-alignment metrics change when the
Jaccard merge threshold is varied at evaluation time?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tools.reduce_transition_rich_interpretability_metrics import (
    RunSpec,
    _encode_trajectories,
    _generate_observation_trajectories,
    _label_sequences_and_centers,
    _load_checkpoint_model,
    _margin_subsets,
    _masked_class_metrics,
    _run_timestamp_key,
    _stringify_value,
    _support_keys,
    _support_mask,
    _write_csv,
)


DEFAULT_THRESHOLDS = "0.2,0.32,0.4,0.45,0.5,0.6,0.7,0.8,0.9"
DEFAULT_SUPPORTS = "absolute:0.001,topk:8"
DEFAULT_EXCLUDED_SYSTEMS = (
    "multiwell_strong_transition,"
    "claude_checkerboard_potential,"
    "claude:checkerboard_potential"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows_csvs",
        required=True,
        help="comma-separated forecasting_rows.csv files used to discover trained runs",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--root_labels",
        default="",
        help="optional comma-separated root labels; defaults to all roots in rows_csvs",
    )
    parser.add_argument("--systems", default="", help="optional comma-separated system_key or system_name filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated seed filter")
    parser.add_argument(
        "--exclude_systems",
        default=DEFAULT_EXCLUDED_SYSTEMS,
        help="comma-separated system_key/system_name/train_env_name entries to exclude",
    )
    parser.add_argument("--num_trajectories", type=int, default=128)
    parser.add_argument("--trajectory_length", type=int, default=128)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--support_definitions",
        default=DEFAULT_SUPPORTS,
        help="comma-separated support definitions, e.g. absolute:0.001,topk:8",
    )
    parser.add_argument(
        "--family_jaccard_thresholds",
        default=DEFAULT_THRESHOLDS,
        help="comma-separated Jaccard thresholds to sweep",
    )
    parser.add_argument(
        "--depth_slice_mode",
        choices=["global", "per_basin"],
        default="global",
        help="match the reducer's deep/boundary slice convention",
    )
    parser.add_argument(
        "--subsets",
        default="all,deep,boundary",
        help="comma-separated subset names to score from the reducer subset map",
    )
    parser.add_argument(
        "--family_fit_scope",
        choices=["all", "subset"],
        default="all",
        help=(
            "where support-family prototypes are formed. 'all' matches the full "
            "reducer convention; 'subset' forms families only on the scored subset "
            "and is useful for focused threshold sensitivity on the deep slice."
        ),
    )
    parser.add_argument("--progress_every_runs", type=int, default=5)
    parser.add_argument(
        "--flush_every_runs",
        type=int,
        default=25,
        help="write partial CSV/JSON artifacts after every N completed run specs; 0 disables partial flush",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume from output_dir/progress.json and existing CSV/JSON outputs. "
            "Use only with the same inputs, filters, support definitions, and thresholds."
        ),
    )
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_support_definitions(raw: str) -> List[Tuple[str, float]]:
    definitions: List[Tuple[str, float]] = []
    for item in _parse_csv_strings(raw):
        if ":" not in item:
            raise ValueError(f"Support definition must be scheme:value, got {item!r}")
        scheme, value = item.split(":", 1)
        scheme = scheme.strip()
        if scheme not in {"absolute", "relative", "topk"}:
            raise ValueError(f"Unsupported support scheme {scheme!r}")
        definitions.append((scheme, float(value)))
    if not definitions:
        raise ValueError("At least one support definition is required")
    return definitions


def _load_latest_specs_from_csvs(
    rows_csvs: Sequence[Path],
    *,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    exclude_systems: Sequence[str],
) -> List[RunSpec]:
    selected_roots = set(root_labels)
    selected_systems = set(systems)
    selected_seeds = set(seeds)
    excluded = set(exclude_systems)
    best_rows: Dict[Tuple[str, str, int], Dict[str, str]] = {}

    for rows_csv in rows_csvs:
        with rows_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                root_label = str(row.get("root_label", "")).strip()
                if selected_roots and root_label not in selected_roots:
                    continue
                system_key = str(row.get("system_key", "")).strip()
                system_name = str(row.get("system_name", "")).strip()
                train_env_name = str(row.get("train_env_name", "")).strip()
                if selected_systems and system_key not in selected_systems and system_name not in selected_systems:
                    continue
                if {system_key, system_name, train_env_name} & excluded:
                    continue
                seed = int(row.get("seed", 0))
                if selected_seeds and seed not in selected_seeds:
                    continue
                run_dir = str(row.get("run_dir", "")).strip()
                if not run_dir:
                    continue
                key = (root_label, system_key, seed)
                incumbent = best_rows.get(key)
                if incumbent is None or _run_timestamp_key(run_dir) > _run_timestamp_key(incumbent["run_dir"]):
                    best_rows[key] = row

    specs = [
        RunSpec(
            root_label=row["root_label"],
            system_key=row["system_key"],
            system_name=row.get("system_name", row["system_key"]),
            seed=int(row["seed"]),
            run_dir=row["run_dir"],
        )
        for row in best_rows.values()
    ]
    return sorted(specs, key=lambda item: (item.root_label, item.system_key, item.seed))


def _finite_values(values: Iterable[object]) -> np.ndarray:
    clean: List[float] = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(float_value):
            clean.append(float_value)
    return np.asarray(clean, dtype=float)


def _iqm(values: Iterable[object]) -> Optional[float]:
    clean = _finite_values(values)
    if clean.size == 0:
        return None
    if clean.size < 4:
        return float(np.mean(clean))
    lo, hi = np.percentile(clean, [25, 75])
    middle = clean[(clean >= lo) & (clean <= hi)]
    if middle.size == 0:
        return float(np.median(clean))
    return float(np.mean(middle))


def _mean(values: Iterable[object]) -> Optional[float]:
    clean = _finite_values(values)
    return float(np.mean(clean)) if clean.size else None


def _std(values: Iterable[object]) -> Optional[float]:
    clean = _finite_values(values)
    return float(np.std(clean, ddof=1)) if clean.size > 1 else None


def _dominant_basin_accuracy(
    class_labels: np.ndarray,
    basin_labels: np.ndarray,
    subset_mask: np.ndarray,
) -> Optional[float]:
    class_flat = class_labels.reshape(-1)[subset_mask]
    basin_flat = basin_labels.reshape(-1)[subset_mask]
    if class_flat.size == 0:
        return None

    class_to_basin: Dict[object, int] = {}
    counters: Dict[object, Counter[int]] = defaultdict(Counter)
    for class_id, basin in zip(class_flat.tolist(), basin_flat.tolist()):
        basin_int = int(basin)
        if basin_int < 0:
            continue
        counters[class_id][basin_int] += 1
    for class_id, counter in counters.items():
        if counter:
            class_to_basin[class_id] = counter.most_common(1)[0][0]
    if not class_to_basin:
        return None

    correct = 0
    total = 0
    for class_id, basin in zip(class_flat.tolist(), basin_flat.tolist()):
        basin_int = int(basin)
        if basin_int < 0 or class_id not in class_to_basin:
            continue
        total += 1
        correct += int(class_to_basin[class_id] == basin_int)
    return float(correct) / float(total) if total else None


def _represented_basin_count(basin_labels: np.ndarray, subset_mask: np.ndarray) -> int:
    basin_flat = basin_labels.reshape(-1)[subset_mask]
    return len({int(item) for item in basin_flat.tolist() if int(item) >= 0})


def _support_family_labels_fast(support_mask: np.ndarray, *, min_jaccard: float) -> np.ndarray:
    """Vectorized equivalent of the reducer's greedy Jaccard family merge."""
    if support_mask.ndim != 3:
        raise ValueError("support_mask must have shape [num_trajectories, trajectory_length, latent_dim]")

    flat_mask = support_mask.reshape(-1, support_mask.shape[-1]).astype(bool, copy=False)
    flat_keys = _support_keys(support_mask).reshape(-1)
    key_counts = Counter(flat_keys.tolist())
    key_masks: Dict[object, np.ndarray] = {}
    for key, mask in zip(flat_keys.tolist(), flat_mask):
        if key not in key_masks:
            key_masks[key] = mask.astype(bool, copy=True)

    ordered_keys = [key for key, _count in sorted(key_counts.items(), key=lambda item: (-item[1], item[0]))]
    max_families = len(ordered_keys)
    packed_dim = np.packbits(np.zeros(flat_mask.shape[-1], dtype=bool)).shape[0]
    prototypes = np.zeros((max_families, packed_dim), dtype=np.uint8)
    prototype_sums = np.zeros(max_families, dtype=np.float64)
    num_families = 0
    key_to_family: Dict[object, int] = {}

    for key in ordered_keys:
        mask = key_masks[key]
        mask_sum = float(mask.sum())
        packed_mask = np.packbits(mask)
        if num_families > 0:
            prototype_view = prototypes[:num_families]
            intersections = np.bitwise_count(np.bitwise_and(prototype_view, packed_mask)).sum(axis=1)
            unions = prototype_sums[:num_families] + mask_sum - intersections
            similarities = np.ones(num_families, dtype=np.float64)
            valid = unions > 0.0
            similarities[valid] = intersections[valid] / unions[valid]
            best_family = int(np.argmax(similarities))
            best_similarity = float(similarities[best_family])
        else:
            best_family = -1
            best_similarity = -1.0

        if best_family >= 0 and best_similarity >= float(min_jaccard):
            key_to_family[key] = best_family
        else:
            key_to_family[key] = num_families
            prototypes[num_families] = packed_mask
            prototype_sums[num_families] = mask_sum
            num_families += 1

    labels = np.asarray([key_to_family[key] for key in flat_keys.tolist()], dtype=np.int64)
    return labels.reshape(support_mask.shape[:-1])


def _row_metrics_for_subset(
    *,
    spec: RunSpec,
    support_name: str,
    subset_name: str,
    subset_mask: np.ndarray,
    basin_labels_np: np.ndarray,
    support_mask: np.ndarray,
    support_codes: np.ndarray,
    family_labels: np.ndarray,
    family_jaccard_threshold: float,
    family_fit_scope: str,
    label_source: str,
) -> Dict[str, object]:
    support_metrics = _masked_class_metrics(support_codes, basin_labels_np, subset_mask)
    family_metrics = _masked_class_metrics(family_labels.astype(object), basin_labels_np, subset_mask)
    represented_basin_count = _represented_basin_count(basin_labels_np, subset_mask)
    family_unique_count = family_metrics["unique_class_count"]
    family_count_minus_basin_count = None
    family_count_over_basin_count = None
    if family_unique_count is not None:
        family_count_minus_basin_count = float(family_unique_count) - float(represented_basin_count)
        if represented_basin_count > 0:
            family_count_over_basin_count = float(family_unique_count) / float(represented_basin_count)
    mean_support_size = None
    if int(subset_mask.sum()) > 0:
        mean_support_size = float(
            support_mask.reshape(-1, support_mask.shape[-1])[subset_mask].sum(axis=1).mean()
        )

    return {
        "root_label": spec.root_label,
        "system_key": spec.system_key,
        "system_name": spec.system_name,
        "seed": spec.seed,
        "run_dir": spec.run_dir,
        "support_scheme": support_name,
        "subset": subset_name,
        "label_source": label_source,
        "family_fit_scope": family_fit_scope,
        "family_jaccard_threshold": float(family_jaccard_threshold),
        "num_states": int(subset_mask.sum()),
        "represented_basin_count": int(represented_basin_count),
        "h_basin_given_support": support_metrics["h_basin_given_class"],
        "h_support_given_basin": support_metrics["h_class_given_basin"],
        "support_nmi": support_metrics["class_nmi"],
        "u_exact": support_metrics["u_exact"],
        "unique_support_count": support_metrics["unique_class_count"],
        "mean_support_size": mean_support_size,
        "family_h_basin_given_family": family_metrics["h_basin_given_class"],
        "family_h_family_given_basin": family_metrics["h_class_given_basin"],
        "family_nmi": family_metrics["class_nmi"],
        "family_u": family_metrics["u_exact"],
        "family_unique_count": family_unique_count,
        "family_dominant_basin_accuracy": _dominant_basin_accuracy(
            family_labels.astype(object),
            basin_labels_np,
            subset_mask,
        ),
        "family_count_minus_basin_count": family_count_minus_basin_count,
        "family_count_over_basin_count": family_count_over_basin_count,
    }


def _reduce_run(
    spec: RunSpec,
    *,
    num_trajectories: int,
    trajectory_length: int,
    eval_seed: int,
    endpoint_rollout_steps: int,
    device: str,
    support_definitions: Sequence[Tuple[str, float]],
    family_jaccard_thresholds: Sequence[float],
    depth_slice_mode: str,
    subsets: Sequence[str],
    family_fit_scope: str,
) -> List[Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    _cfg, env, model = _load_checkpoint_model(checkpoint_path, spec.system_key, device)
    trajectories = _generate_observation_trajectories(
        env,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        eval_seed=eval_seed,
    )
    basin_labels, centers, label_source = _label_sequences_and_centers(
        env,
        trajectories,
        system_key=spec.system_key,
        endpoint_rollout_steps=endpoint_rollout_steps,
    )
    latents = _encode_trajectories(model, trajectories, device)
    basin_labels_np = basin_labels.cpu().numpy()
    subset_masks = _margin_subsets(
        trajectories,
        centers,
        basin_labels=basin_labels_np if depth_slice_mode == "per_basin" else None,
        depth_slice_mode=depth_slice_mode,
    )
    selected_subset_masks = {
        name: mask for name, mask in subset_masks.items() if not subsets or name in set(subsets)
    }
    if not selected_subset_masks:
        raise ValueError(f"No requested subsets {list(subsets)!r} found in {sorted(subset_masks)}")

    rows: List[Dict[str, object]] = []
    for scheme, value in support_definitions:
        support_name = f"{scheme}:{_stringify_value(scheme, value)}"
        support_mask = _support_mask(latents, scheme=scheme, value=value)
        support_codes = _support_keys(support_mask)
        for threshold in family_jaccard_thresholds:
            if family_fit_scope == "all":
                family_labels = _support_family_labels_fast(support_mask, min_jaccard=threshold)
                for subset_name, subset_mask in selected_subset_masks.items():
                    rows.append(
                        _row_metrics_for_subset(
                            spec=spec,
                            support_name=support_name,
                            subset_name=subset_name,
                            subset_mask=subset_mask,
                            basin_labels_np=basin_labels_np,
                            support_mask=support_mask,
                            support_codes=support_codes,
                            family_labels=family_labels,
                            family_jaccard_threshold=threshold,
                            family_fit_scope=family_fit_scope,
                            label_source=label_source,
                        )
                    )
            else:
                flat_support_mask = support_mask.reshape(-1, support_mask.shape[-1])
                flat_basin_labels = basin_labels_np.reshape(-1)
                for subset_name, subset_mask in selected_subset_masks.items():
                    subset_support_mask = flat_support_mask[subset_mask].reshape(1, -1, support_mask.shape[-1])
                    subset_support_codes = _support_keys(subset_support_mask)
                    subset_basin_labels = flat_basin_labels[subset_mask].reshape(1, -1)
                    subset_family_labels = _support_family_labels_fast(
                        subset_support_mask,
                        min_jaccard=threshold,
                    )
                    rows.append(
                        _row_metrics_for_subset(
                            spec=spec,
                            support_name=support_name,
                            subset_name=subset_name,
                            subset_mask=np.ones(subset_support_mask.shape[1], dtype=bool),
                            basin_labels_np=subset_basin_labels,
                            support_mask=subset_support_mask,
                            support_codes=subset_support_codes,
                            family_labels=subset_family_labels,
                            family_jaccard_threshold=threshold,
                            family_fit_scope=family_fit_scope,
                            label_source=label_source,
                        )
                    )
    return rows


def _summarize_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            row["root_label"],
            row["support_scheme"],
            row["subset"],
            row.get("family_fit_scope", "all"),
            row["family_jaccard_threshold"],
        )
        grouped[key].append(row)

    metric_summary_modes = {
        "represented_basin_count": "mean",
        "h_basin_given_support": "iqm",
        "h_support_given_basin": "iqm",
        "support_nmi": "iqm",
        "u_exact": "iqm",
        "unique_support_count": "mean",
        "mean_support_size": "mean",
        "family_h_basin_given_family": "iqm",
        "family_h_family_given_basin": "iqm",
        "family_nmi": "iqm",
        "family_u": "iqm",
        "family_unique_count": "mean",
        "family_dominant_basin_accuracy": "iqm",
        "family_count_minus_basin_count": "mean",
        "family_count_over_basin_count": "mean",
    }

    summary_rows: List[Dict[str, object]] = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        root_label, support_scheme, subset, family_fit_scope, threshold = key
        systems = sorted({str(row["system_name"]) for row in group_rows})
        seeds_by_system: Dict[str, set[int]] = defaultdict(set)
        for row in group_rows:
            seeds_by_system[str(row["system_name"])].add(int(row["seed"]))

        summary: Dict[str, object] = {
            "root_label": root_label,
            "support_scheme": support_scheme,
            "subset": subset,
            "family_fit_scope": family_fit_scope,
            "family_jaccard_threshold": threshold,
            "num_rows": len(group_rows),
            "num_systems": len(systems),
            "seed_count_min": min((len(v) for v in seeds_by_system.values()), default=0),
            "seed_count_max": max((len(v) for v in seeds_by_system.values()), default=0),
        }

        for metric, mode in metric_summary_modes.items():
            per_system_values: List[float] = []
            for system in systems:
                sys_rows = [row for row in group_rows if str(row["system_name"]) == system]
                value = _mean(row.get(metric) for row in sys_rows) if mode == "mean" else _iqm(
                    row.get(metric) for row in sys_rows
                )
                if value is not None:
                    per_system_values.append(float(value))
            summary[f"{metric}_{mode}_over_seeds_mean_over_systems"] = _mean(per_system_values)
            summary[f"{metric}_{mode}_over_seeds_std_over_systems"] = _std(per_system_values)
        summary_rows.append(summary)
    return summary_rows


def _summarize_threshold_sensitivity(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[(row["root_label"], row["support_scheme"], row["subset"], row.get("family_fit_scope", "all"))].append(row)

    out: List[Dict[str, object]] = []
    h_key = "family_h_basin_given_family_iqm_over_seeds_mean_over_systems"
    f_key = "family_unique_count_mean_over_seeds_mean_over_systems"
    acc_key = "family_dominant_basin_accuracy_iqm_over_seeds_mean_over_systems"
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        valid = [
            row
            for row in rows
            if row.get(h_key) is not None and row.get(f_key) is not None
        ]
        if not valid:
            continue
        by_threshold = {float(row["family_jaccard_threshold"]): row for row in valid}
        reference = by_threshold.get(0.5)
        best_h = min(valid, key=lambda row: float(row[h_key]))
        min_count = min(float(row[f_key]) for row in valid)
        max_count = max(float(row[f_key]) for row in valid)
        min_h = min(float(row[h_key]) for row in valid)
        max_h = max(float(row[h_key]) for row in valid)
        min_acc = min(
            float(row[acc_key])
            for row in valid
            if row.get(acc_key) is not None
        )
        max_acc = max(
            float(row[acc_key])
            for row in valid
            if row.get(acc_key) is not None
        )
        record: Dict[str, object] = {
            "root_label": key[0],
            "support_scheme": key[1],
            "subset": key[2],
            "family_fit_scope": key[3],
            "num_thresholds": len(valid),
            "best_h_threshold": best_h["family_jaccard_threshold"],
            "best_h_basin_given_family": best_h[h_key],
            "best_h_family_unique_count": best_h[f_key],
            "family_unique_count_min": min_count,
            "family_unique_count_max": max_count,
            "family_unique_count_range": max_count - min_count,
            "family_h_basin_given_family_min": min_h,
            "family_h_basin_given_family_max": max_h,
            "family_h_basin_given_family_range": max_h - min_h,
            "family_dominant_basin_accuracy_min": min_acc,
            "family_dominant_basin_accuracy_max": max_acc,
            "family_dominant_basin_accuracy_range": max_acc - min_acc,
        }
        if reference is not None:
            record.update(
                {
                    "jaccard_0p5_family_unique_count": reference[f_key],
                    "jaccard_0p5_h_basin_given_family": reference[h_key],
                    "jaccard_0p5_family_dominant_basin_accuracy": reference.get(acc_key),
                }
            )
        out.append(record)
    return out


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _flush(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    rows_csvs: Sequence[Path],
    support_definitions: Sequence[Tuple[str, float]],
    thresholds: Sequence[float],
    specs: Sequence[RunSpec],
    completed_runs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = _summarize_rows(rows)
    sensitivity_rows = _summarize_threshold_sensitivity(summary_rows)
    _write_csv(output_dir / "jaccard_sweep_rows.csv", rows)
    _write_csv(output_dir / "jaccard_sweep_summary.csv", summary_rows)
    _write_csv(output_dir / "jaccard_sweep_threshold_sensitivity.csv", sensitivity_rows)
    _write_json(
        output_dir / "failures.json",
        {"failure_count": len(failures), "failures": list(failures)},
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "rows_csvs": [str(path) for path in rows_csvs],
            "root_labels": _parse_csv_strings(args.root_labels),
            "systems": _parse_csv_strings(args.systems),
            "seeds": _parse_csv_ints(args.seeds),
            "exclude_systems": _parse_csv_strings(args.exclude_systems),
            "num_trajectories": args.num_trajectories,
            "trajectory_length": args.trajectory_length,
            "eval_seed": args.eval_seed,
            "endpoint_rollout_steps": args.endpoint_rollout_steps,
            "device": args.device,
            "support_definitions": [
                {"scheme": scheme, "value": value} for scheme, value in support_definitions
            ],
            "family_jaccard_thresholds": list(thresholds),
            "depth_slice_mode": args.depth_slice_mode,
            "subsets": _parse_csv_strings(args.subsets),
            "family_fit_scope": args.family_fit_scope,
            "num_runs": len(specs),
            "completed_runs": completed_runs,
            "remaining_runs": max(0, len(specs) - completed_runs),
            "num_rows": len(rows),
            "num_summary_rows": len(summary_rows),
            "num_sensitivity_rows": len(sensitivity_rows),
            "num_failures": len(failures),
            "status": status,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    _write_json(
        output_dir / "progress.json",
        {
            "num_runs": len(specs),
            "completed_runs": completed_runs,
            "remaining_runs": max(0, len(specs) - completed_runs),
            "num_rows": len(rows),
            "num_failures": len(failures),
            "status": status,
            "elapsed_seconds": elapsed_seconds,
        },
    )


def _load_resume_state(output_dir: Path) -> Tuple[int, List[Dict[str, object]], List[Dict[str, object]]]:
    progress_path = output_dir / "progress.json"
    rows_path = output_dir / "jaccard_sweep_rows.csv"
    failures_path = output_dir / "failures.json"

    completed_runs = 0
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
            completed_runs = int(progress.get("completed_runs", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            completed_runs = 0

    rows: List[Dict[str, object]] = []
    if rows_path.exists() and rows_path.stat().st_size > 0:
        with rows_path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))

    failures: List[Dict[str, object]] = []
    if failures_path.exists() and failures_path.stat().st_size > 0:
        try:
            payload = json.loads(failures_path.read_text())
            failures = list(payload.get("failures", []))
        except (OSError, json.JSONDecodeError, TypeError):
            failures = []

    return max(0, completed_runs), rows, failures


def main() -> None:
    args = _parse_args()
    rows_csvs = [Path(item) for item in _parse_csv_strings(args.rows_csvs)]
    support_definitions = _parse_support_definitions(args.support_definitions)
    thresholds = _parse_csv_floats(args.family_jaccard_thresholds)
    subsets = _parse_csv_strings(args.subsets)
    if not thresholds:
        raise ValueError("At least one Jaccard threshold is required")

    specs = _load_latest_specs_from_csvs(
        rows_csvs,
        root_labels=_parse_csv_strings(args.root_labels),
        systems=_parse_csv_strings(args.systems),
        seeds=_parse_csv_ints(args.seeds),
        exclude_systems=_parse_csv_strings(args.exclude_systems),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        completed_before, rows, failures = _load_resume_state(output_dir)
        completed_before = min(completed_before, len(specs))
    else:
        completed_before = 0
        rows = []
        failures = []
    start_time = time.time()
    progress_every_runs = max(1, int(args.progress_every_runs))
    flush_every_runs = max(0, int(args.flush_every_runs))

    _flush(
        output_dir,
        args=args,
        rows_csvs=rows_csvs,
        support_definitions=support_definitions,
        thresholds=thresholds,
        specs=specs,
        completed_runs=completed_before,
        rows=rows,
        failures=failures,
        status="running",
        elapsed_seconds=0.0,
    )

    for spec_index, spec in enumerate(specs[completed_before:], start=completed_before + 1):
        status = "ok"
        error: Optional[str] = None
        try:
            rows.extend(
                _reduce_run(
                    spec,
                    num_trajectories=args.num_trajectories,
                    trajectory_length=args.trajectory_length,
                    eval_seed=args.eval_seed,
                    endpoint_rollout_steps=args.endpoint_rollout_steps,
                    device=args.device,
                    support_definitions=support_definitions,
                    family_jaccard_thresholds=thresholds,
                    depth_slice_mode=args.depth_slice_mode,
                    subsets=subsets,
                    family_fit_scope=args.family_fit_scope,
                )
            )
        except Exception as exc:  # pragma: no cover - long sweeps should continue across bad runs.
            status = "failed"
            error = repr(exc)
            failures.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "system_name": spec.system_name,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
                    "error": error,
                }
            )

        elapsed = time.time() - start_time
        if spec_index % progress_every_runs == 0 or spec_index == len(specs):
            print(
                (
                    f"[{spec_index}/{len(specs)}] {status} "
                    f"root={spec.root_label} system={spec.system_key} seed={spec.seed} "
                    f"rows={len(rows)} failures={len(failures)} elapsed_s={elapsed:.1f}"
                ),
                flush=True,
            )
            if error:
                print(f"  error={error}", flush=True)
        if flush_every_runs > 0 and (spec_index % flush_every_runs == 0 or spec_index == len(specs)):
            _flush(
                output_dir,
                args=args,
                rows_csvs=rows_csvs,
                support_definitions=support_definitions,
                thresholds=thresholds,
                specs=specs,
                completed_runs=spec_index,
                rows=rows,
                failures=failures,
                status="running" if spec_index < len(specs) else "completed",
                elapsed_seconds=elapsed,
            )

    elapsed = time.time() - start_time
    _flush(
        output_dir,
        args=args,
        rows_csvs=rows_csvs,
        support_definitions=support_definitions,
        thresholds=thresholds,
        specs=specs,
        completed_runs=len(specs),
        rows=rows,
        failures=failures,
        status="completed",
        elapsed_seconds=elapsed,
    )


if __name__ == "__main__":
    main()
