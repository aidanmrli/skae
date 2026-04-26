#!/usr/bin/env python3
"""Diagnose why support-conditioned local-law evidence is inconsistent.

This tool focuses on two mechanism questions that the operator-selection packet
cannot answer cleanly on its own:

1. Does the learned global Koopman matrix actually keep predicted latent mass on
   the currently active support (or support family), or does it immediately spill
   energy into off-support coordinates?
2. On systems with native basin labels, how far do proxy label modes
   (`env_points`, `estimated_centers`) drift from the native basin partition?

The first diagnostic addresses the architectural concern that support may be an
encoder-side phenomenon while the dynamics remain governed by one global dense
operator. The second diagnostic addresses whether proxy basin labels are
plausible substitutes for native labels on transition-rich systems.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

EPS = 1e-12


@dataclass(frozen=True)
class RunSpec:
    root_label: str
    system_key: str
    system_name: str
    seed: int
    run_dir: str


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REDUCER = _load_module(
    "reduce_transition_rich_interpretability_metrics.py",
    "reduce_transition_rich_interpretability_metrics_support_flow",
)
OPSEL = _load_module(
    "evaluate_transition_rich_operator_selection.py",
    "evaluate_transition_rich_operator_selection_support_flow",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows_csvs",
        required=True,
        help="comma-separated forecasting_rows.csv files used to discover runs",
    )
    parser.add_argument("--output_dir", required=True, help="directory for diagnostic artifacts")
    parser.add_argument("--root_labels", required=True, help="comma-separated root labels to include")
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated integer seed filter")
    parser.add_argument(
        "--support_definitions",
        default="absolute:0.001,relative:0.1,topk:8",
        help="comma-separated support definitions formatted as scheme:value",
    )
    parser.add_argument(
        "--subsets",
        default="all,deep,boundary",
        help="comma-separated subset names to evaluate",
    )
    parser.add_argument("--num_trajectories", type=int, default=256)
    parser.add_argument("--trajectory_length", type=int, default=256)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.5)
    parser.add_argument(
        "--label_modes",
        default="env_points,estimated_centers",
        help="proxy label modes to compare against native basin labels when available",
    )
    parser.add_argument("--progress_every_runs", type=int, default=1)
    parser.add_argument("--flush_every_runs", type=int, default=0)
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_support_definitions(raw: str) -> List[Tuple[str, float]]:
    definitions: List[Tuple[str, float]] = []
    for item in _parse_csv_strings(raw):
        if ":" not in item:
            raise ValueError(f"Support definition must be scheme:value, got '{item}'")
        scheme, raw_value = item.split(":", 1)
        scheme = scheme.strip()
        raw_value = raw_value.strip()
        if scheme == "topk":
            definitions.append((scheme, float(int(raw_value))))
        else:
            definitions.append((scheme, float(raw_value)))
    return definitions


def _load_latest_specs(
    rows_csvs: Sequence[Path],
    *,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
) -> List[RunSpec]:
    selected_roots = set(root_labels)
    selected_systems = set(systems)
    selected_seeds = set(seeds)
    best_rows: Dict[Tuple[str, str, int], Dict[str, str]] = {}

    for rows_csv in rows_csvs:
        with rows_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                root_label = str(row.get("root_label", "")).strip()
                if root_label not in selected_roots:
                    continue
                system_key = str(row.get("system_key", "")).strip()
                if selected_systems and system_key not in selected_systems:
                    continue
                seed = int(row.get("seed", 0))
                if selected_seeds and seed not in selected_seeds:
                    continue
                key = (root_label, system_key, seed)
                incumbent = best_rows.get(key)
                if incumbent is None or OPSEL._run_timestamp_key(row["run_dir"]) > OPSEL._run_timestamp_key(
                    incumbent["run_dir"]
                ):
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


def _best_permutation_accuracy(reference: np.ndarray, predicted: np.ndarray) -> float:
    if reference.shape != predicted.shape:
        raise ValueError("reference and predicted must have the same shape")
    if reference.size == 0:
        return float("nan")
    ref_labels = sorted({int(item) for item in reference.tolist()})
    pred_labels = sorted({int(item) for item in predicted.tolist()})
    best = 0.0
    for mapped_targets in itertools.permutations(ref_labels, r=min(len(pred_labels), len(ref_labels))):
        mapping = {pred: target for pred, target in zip(pred_labels, mapped_targets)}
        mapped = np.asarray([mapping.get(int(item), -1) for item in predicted.tolist()], dtype=np.int64)
        best = max(best, float(np.mean(mapped == reference)))
    return best


def _family_prototype_masks(support_mask: np.ndarray, family_labels: np.ndarray) -> np.ndarray:
    flat_support = support_mask.reshape(-1, support_mask.shape[-1])
    flat_family = family_labels.reshape(-1)
    flat_keys = REDUCER._support_keys(support_mask).reshape(-1)

    counts_by_family: Dict[int, Counter[object]] = defaultdict(Counter)
    mask_by_key: Dict[object, np.ndarray] = {}
    for mask, family, key in zip(flat_support, flat_family.tolist(), flat_keys.tolist()):
        family_int = int(family)
        counts_by_family[family_int][key] += 1
        if key not in mask_by_key:
            mask_by_key[key] = mask.astype(bool, copy=True)

    prototype_by_family: Dict[int, np.ndarray] = {}
    for family, counter in counts_by_family.items():
        prototype_key = counter.most_common(1)[0][0]
        prototype_by_family[family] = mask_by_key[prototype_key]

    flat_masks = np.stack([prototype_by_family[int(family)] for family in flat_family.tolist()], axis=0)
    return flat_masks.reshape(support_mask.shape)


def _flow_metrics(z: np.ndarray, kmat: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if z.shape != mask.shape:
        raise ValueError("z and mask must have the same shape")
    if z.ndim != 2:
        raise ValueError("z and mask must have shape [N, dim]")

    z_active = np.where(mask, z, 0.0)
    z_inactive = np.where(~mask, z, 0.0)
    full_pred = z @ kmat
    active_pred = z_active @ kmat
    inactive_pred = z_inactive @ kmat

    full_energy = np.sum(full_pred**2, axis=1) + EPS
    active_energy = np.sum(active_pred**2, axis=1) + EPS

    full_on_mask = np.where(mask, full_pred, 0.0)
    full_off_mask = np.where(~mask, full_pred, 0.0)
    active_on_mask = np.where(mask, active_pred, 0.0)
    active_off_mask = np.where(~mask, active_pred, 0.0)

    support_sizes = mask.sum(axis=1).astype(np.float64)
    return {
        "mean_support_size": float(np.mean(support_sizes)),
        "full_output_on_mask_energy_fraction": float(np.mean(np.sum(full_on_mask**2, axis=1) / full_energy)),
        "full_output_off_mask_energy_fraction": float(np.mean(np.sum(full_off_mask**2, axis=1) / full_energy)),
        "active_source_energy_fraction_of_full": float(np.mean(np.sum(active_pred**2, axis=1) / full_energy)),
        "inactive_source_energy_fraction_of_full": float(np.mean(np.sum(inactive_pred**2, axis=1) / full_energy)),
        "active_source_on_mask_energy_fraction": float(np.mean(np.sum(active_on_mask**2, axis=1) / active_energy)),
        "active_source_off_mask_energy_fraction": float(np.mean(np.sum(active_off_mask**2, axis=1) / active_energy)),
    }


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _format_float(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "N/A"
    return f"{float(value):.4f}"


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_support_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["root_label"]),
                str(row["system_key"]),
                str(row["partition_kind"]),
                str(row["support_definition"]),
                str(row["subset"]),
            )
        ].append(row)

    lines = [
        "# Support-Flow Summary",
        "",
        "How much of the learned global Koopman prediction stays on the current support/family mask?",
        "",
        "| root | system | partition | support | subset | coverage | on-mask(full) | off-mask(full) | active-src/full | active-src->on | active-src->off | inactive-src/full |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(grouped):
        group_rows = grouped[key]
        lines.append(
            f"| `{key[0]}` | `{key[1]}` | `{key[2]}` | `{key[3]}` | `{key[4]}` | "
            f"{_format_float(_safe_mean(row.get('coverage_fraction') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('full_output_on_mask_energy_fraction') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('full_output_off_mask_energy_fraction') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('active_source_energy_fraction_of_full') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('active_source_on_mask_energy_fraction') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('active_source_off_mask_energy_fraction') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('inactive_source_energy_fraction_of_full') for row in group_rows))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_label_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["root_label"]),
                str(row["system_key"]),
                str(row["proxy_label_mode"]),
                str(row["subset"]),
            )
        ].append(row)

    lines = [
        "# Label-Agreement Summary",
        "",
        "How well do proxy label modes align with native basin labels on systems that expose them?",
        "",
        "| root | system | proxy mode | subset | matched acc | NMI | H(native|proxy) | H(proxy|native) |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for key in sorted(grouped):
        group_rows = grouped[key]
        lines.append(
            f"| `{key[0]}` | `{key[1]}` | `{key[2]}` | `{key[3]}` | "
            f"{_format_float(_safe_mean(row.get('matched_accuracy') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('nmi') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('h_native_given_proxy') for row in group_rows))} | "
            f"{_format_float(_safe_mean(row.get('h_proxy_given_native') for row in group_rows))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    rows_csvs = [Path(item) for item in _parse_csv_strings(args.rows_csvs)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_latest_specs(
        rows_csvs,
        root_labels=_parse_csv_strings(args.root_labels),
        systems=_parse_csv_strings(args.systems),
        seeds=_parse_csv_ints(args.seeds),
    )
    support_definitions = _parse_support_definitions(args.support_definitions)
    subset_names = _parse_csv_strings(args.subsets)
    proxy_modes = _parse_csv_strings(args.label_modes)

    support_rows: List[Dict[str, object]] = []
    label_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    progress: List[Dict[str, object]] = []

    for run_index, spec in enumerate(specs, start=1):
        run_started = time.time()
        try:
            checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
            _cfg, env, model = REDUCER._load_checkpoint_model(checkpoint_path, spec.system_key, args.device)
            trajectories = REDUCER._generate_observation_trajectories(
                env,
                num_trajectories=args.num_trajectories,
                trajectory_length=args.trajectory_length,
                eval_seed=args.eval_seed,
            )
            latents = REDUCER._encode_trajectories(model, trajectories, args.device)
            kmat = model.kmatrix().detach().cpu().numpy()

            basin_labels, centers, label_source = OPSEL._label_sequences_for_mode(
                env,
                trajectories,
                system_key=spec.system_key,
                endpoint_rollout_steps=args.endpoint_rollout_steps,
                label_mode="auto",
            )
            subset_masks = REDUCER._margin_subsets(trajectories, centers)

            z_t = latents[:, :-1, :].reshape(-1, latents.shape[-1])
            transition_subset_masks = {
                subset_name: subset_mask.reshape(trajectories.shape[0], trajectories.shape[1])[:, :-1].reshape(-1)
                for subset_name, subset_mask in subset_masks.items()
            }

            for scheme, value in support_definitions:
                support_name = f"{scheme}:{int(value) if scheme == 'topk' else value:g}"
                support_mask = REDUCER._support_mask(latents, scheme=scheme, value=value)
                family_labels = REDUCER.support_family_labels(
                    support_mask,
                    min_jaccard=args.family_jaccard_threshold,
                )
                family_masks = _family_prototype_masks(support_mask, family_labels)

                support_t = support_mask[:, :-1, :].reshape(-1, support_mask.shape[-1])
                family_t = family_masks[:, :-1, :].reshape(-1, family_masks.shape[-1])

                for partition_kind, mask_array in (("support", support_t), ("family", family_t)):
                    valid_mask = mask_array.any(axis=1)
                    for subset_name in subset_names:
                        if subset_name not in transition_subset_masks:
                            continue
                        select_mask = np.logical_and(transition_subset_masks[subset_name], valid_mask)
                        coverage = float(np.mean(select_mask)) if select_mask.size else 0.0
                        if not bool(np.any(select_mask)):
                            support_rows.append(
                                {
                                    "root_label": spec.root_label,
                                    "system_key": spec.system_key,
                                    "system_name": spec.system_name,
                                    "seed": spec.seed,
                                    "label_source": label_source,
                                    "support_definition": support_name,
                                    "subset": subset_name,
                                    "partition_kind": partition_kind,
                                    "coverage_fraction": coverage,
                                    "mean_support_size": None,
                                    "full_output_on_mask_energy_fraction": None,
                                    "full_output_off_mask_energy_fraction": None,
                                    "active_source_energy_fraction_of_full": None,
                                    "inactive_source_energy_fraction_of_full": None,
                                    "active_source_on_mask_energy_fraction": None,
                                    "active_source_off_mask_energy_fraction": None,
                                }
                            )
                            continue
                        metrics = _flow_metrics(z_t[select_mask], kmat, mask_array[select_mask])
                        support_rows.append(
                            {
                                "root_label": spec.root_label,
                                "system_key": spec.system_key,
                                "system_name": spec.system_name,
                                "seed": spec.seed,
                                "label_source": label_source,
                                "support_definition": support_name,
                                "subset": subset_name,
                                "partition_kind": partition_kind,
                                "coverage_fraction": coverage,
                                **metrics,
                            }
                        )

            if hasattr(env, "basin_label"):
                native_labels, native_centers, _ = OPSEL._label_sequences_for_mode(
                    env,
                    trajectories,
                    system_key=spec.system_key,
                    endpoint_rollout_steps=args.endpoint_rollout_steps,
                    label_mode="native",
                )
                native_subset_masks = REDUCER._margin_subsets(trajectories, native_centers)
                native_flat = native_labels.reshape(-1).cpu().numpy().astype(np.int64)
                for proxy_mode in proxy_modes:
                    proxy_labels, _proxy_centers, _proxy_source = OPSEL._label_sequences_for_mode(
                        env,
                        trajectories,
                        system_key=spec.system_key,
                        endpoint_rollout_steps=args.endpoint_rollout_steps,
                        label_mode=proxy_mode,
                    )
                    proxy_flat = proxy_labels.reshape(-1).cpu().numpy().astype(np.int64)
                    for subset_name in subset_names:
                        if subset_name not in native_subset_masks:
                            continue
                        subset_mask = native_subset_masks[subset_name].reshape(-1)
                        ref = native_flat[subset_mask]
                        pred = proxy_flat[subset_mask]
                        if ref.size == 0:
                            matched_accuracy = None
                            nmi = None
                            h_native_given_proxy = None
                            h_proxy_given_native = None
                        else:
                            matched_accuracy = _best_permutation_accuracy(ref, pred)
                            nmi = REDUCER.normalized_mutual_information(ref.tolist(), pred.tolist())
                            h_native_given_proxy = REDUCER.conditional_entropy(ref.tolist(), pred.tolist())
                            h_proxy_given_native = REDUCER.conditional_entropy(pred.tolist(), ref.tolist())
                        label_rows.append(
                            {
                                "root_label": spec.root_label,
                                "system_key": spec.system_key,
                                "system_name": spec.system_name,
                                "seed": spec.seed,
                                "proxy_label_mode": proxy_mode,
                                "subset": subset_name,
                                "matched_accuracy": matched_accuracy,
                                "nmi": nmi,
                                "h_native_given_proxy": h_native_given_proxy,
                                "h_proxy_given_native": h_proxy_given_native,
                            }
                        )

            progress.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "seed": spec.seed,
                    "elapsed_sec": time.time() - run_started,
                }
            )
            if args.progress_every_runs > 0 and run_index % args.progress_every_runs == 0:
                print(
                    f"[{run_index}/{len(specs)}] {spec.root_label} {spec.system_key} seed={spec.seed} "
                    f"support_rows={len(support_rows)} label_rows={len(label_rows)} failures={len(failures)}"
                )
            if args.flush_every_runs > 0 and run_index % args.flush_every_runs == 0:
                _write_csv(output_dir / "support_flow_rows.csv", support_rows)
                _write_csv(output_dir / "label_agreement_rows.csv", label_rows)
                _write_csv(output_dir / "progress.csv", progress)
                (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
        except Exception as exc:  # pragma: no cover - diagnostic script
            failures.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
                    "error": repr(exc),
                }
            )
            print(f"[error] {spec.root_label} {spec.system_key} seed={spec.seed}: {exc}")

    _write_csv(output_dir / "support_flow_rows.csv", support_rows)
    _write_csv(output_dir / "label_agreement_rows.csv", label_rows)
    _write_csv(output_dir / "progress.csv", progress)
    _write_support_summary(output_dir / "support_flow_summary.md", support_rows)
    _write_label_summary(output_dir / "label_agreement_summary.md", label_rows)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows_csvs": [str(path) for path in rows_csvs],
                "root_labels": _parse_csv_strings(args.root_labels),
                "systems": _parse_csv_strings(args.systems),
                "seeds": _parse_csv_ints(args.seeds),
                "support_definitions": [f"{scheme}:{value}" for scheme, value in support_definitions],
                "subsets": subset_names,
                "num_trajectories": args.num_trajectories,
                "trajectory_length": args.trajectory_length,
                "eval_seed": args.eval_seed,
                "endpoint_rollout_steps": args.endpoint_rollout_steps,
                "family_jaccard_threshold": args.family_jaccard_threshold,
                "label_modes": proxy_modes,
                "support_row_count": len(support_rows),
                "label_row_count": len(label_rows),
                "failure_count": len(failures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
