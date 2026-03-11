"""Batch basin-support evaluation for labelable paper-benchmark systems.

This script reads a collected benchmark CSV, resolves the corresponding
`checkpoint.pt` paths, runs basin-support uniqueness and cosine separation on
the selected systems, and writes per-seed plus aggregated summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from skae.basin_utils import BasinLabeledDataset
from skae.config import Config
from skae.data import make_env
from skae.model import make_model
from tools.evaluate_support_uniqueness import (
    compute_cosine_basin_similarity,
    compute_support_uniqueness,
)


DEFAULT_LABELABLE_SYSTEMS: Sequence[str] = (
    "duffing",
    "multiwell_gradient",
    "multiwell_rotational",
    "multiwell_energy",
    "multiwell_strong_transition",
    "multiwell_gradient_hd",
    "multiwell_rotational_hd",
    "multiwell_energy_hd",
    "multiwell_strong_transition_hd",
    "kuramoto",
    "hopfield",
    "competitive_lv",
)

DEFAULT_ROOT_LABELS: Sequence[str] = (
    "generic_sparse",
    "lista_dense",
    "lista_diagonal",
    "lista_blockdiag",
)

SUMMARY_METRIC_NAMES: Sequence[str] = (
    "mode_uniqueness_rate",
    "mean_basin_consistency",
    "mean_pairwise_jaccard",
    "mean_mode_support_size",
    "trajectory_unique_support_rate",
    "mean_within_basin_hamming",
    "mean_between_basin_hamming",
    "between_over_within_hamming_ratio",
    "mean_intra_basin_cosine",
    "mean_inter_basin_cosine",
    "cosine_separation_score",
)


def _parse_csv_strings(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str | None) -> List[int]:
    if not raw:
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _median(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(statistics.median(values_list))


def _dataset_cache_key(
    checkpoint_dict: Dict,
    system: str,
    num_trajectories: int,
    trajectory_length: int,
    long_rollout_steps: int,
    eval_seed: int,
    sampling_strategy: str,
    target_raw_labels: Sequence[int],
    trajectories_per_basin: int | None,
    max_attempts: int | None,
) -> str:
    key = {
        "env": checkpoint_dict["config"]["ENV"],
        "system": system,
        "num_trajectories": num_trajectories,
        "trajectory_length": trajectory_length,
        "long_rollout_steps": long_rollout_steps,
        "eval_seed": eval_seed,
        "sampling_strategy": sampling_strategy,
        "target_raw_labels": list(target_raw_labels),
        "trajectories_per_basin": trajectories_per_basin,
        "max_attempts": max_attempts,
    }
    return json.dumps(key, sort_keys=True)


def _build_dataset(
    checkpoint_dict: Dict,
    system: str,
    num_trajectories: int,
    trajectory_length: int,
    long_rollout_steps: int,
    eval_seed: int,
    sampling_strategy: str,
    target_raw_labels: Sequence[int],
    trajectories_per_basin: int | None,
    max_attempts: int | None,
) -> BasinLabeledDataset:
    cfg = Config.from_dict(checkpoint_dict["config"])
    cfg.ENV.ENV_NAME = system
    return BasinLabeledDataset(
        system=system,
        cfg=cfg,
        num_trajectories=num_trajectories,
        trajectory_length=trajectory_length,
        long_rollout_steps=long_rollout_steps,
        seed=eval_seed,
        sampling_strategy=sampling_strategy,
        target_raw_labels=list(target_raw_labels),
        trajectories_per_basin=trajectories_per_basin,
        max_attempts=max_attempts,
    )


def _load_checkpoint(checkpoint_path: Path, device: str) -> Dict:
    return torch.load(checkpoint_path, map_location=device)


def _select_rows(
    rows: List[Dict[str, str]],
    systems: Sequence[str],
    root_labels: Sequence[str],
    seeds: Sequence[int],
    skip_invalid_competitive_lv: bool,
) -> Tuple[List[Dict[str, str]], List[str]]:
    system_set = set(systems)
    root_set = set(root_labels)
    seed_set = set(seeds)
    selected: List[Dict[str, str]] = []
    skipped_messages: List[str] = []

    for row in rows:
        system = row["system_key"]
        if system not in system_set:
            continue
        if row["root_label"] not in root_set:
            continue
        if seed_set and int(row["seed"]) not in seed_set:
            continue
        if skip_invalid_competitive_lv and system == "competitive_lv":
            continue
        selected.append(row)

    if skip_invalid_competitive_lv and "competitive_lv" in system_set:
        skipped_messages.append(
            "Skipped `competitive_lv` from the canonical v4 matrix because the "
            "March 8 benchmark used the invalidated 1-basin configuration."
        )

    return selected, skipped_messages


def _evaluate_entry(
    *,
    row: Dict[str, str],
    checkpoint_dict: Dict,
    dataset: BasinLabeledDataset,
    device: str,
    support_threshold: float,
    support_mode: str,
    cosine_aggregation: str,
) -> Dict[str, object]:
    cfg = Config.from_dict(checkpoint_dict["config"])
    cfg.ENV.ENV_NAME = row["system_key"]

    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    model = model.to(device)
    model.eval()

    support_results = compute_support_uniqueness(
        model=model,
        dataset=dataset,
        device=device,
        support_threshold=support_threshold,
        support_mode=support_mode,
    )
    cosine_metrics = compute_cosine_basin_similarity(
        model=model,
        dataset=dataset,
        device=device,
        aggregation=cosine_aggregation,
    )
    support_results.mean_intra_basin_cosine = cosine_metrics["mean_intra_basin_cosine"]
    support_results.mean_inter_basin_cosine = cosine_metrics["mean_inter_basin_cosine"]
    support_results.cosine_separation_score = cosine_metrics["cosine_separation_score"]

    payload = asdict(support_results)
    payload.update(
        {
            "root_label": row["root_label"],
            "seed": int(row["seed"]),
            "run_dir": row["run_dir"],
            "checkpoint_path": str(Path(row["run_dir"]) / "checkpoint.pt"),
            "train_env_name": row["train_env_name"],
            "env_dt": float(row["env_dt"]),
            "h1000_best_periodic_mean": float(row["h1000_best_periodic_mean"]),
        }
    )
    return payload


def _aggregate_by_system_and_root(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for item in results:
        grouped[(str(item["system_name"]), str(item["root_label"]))].append(item)

    rows: List[Dict[str, object]] = []
    for (system, root_label), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: int(item["seed"]))
        row: Dict[str, object] = {
            "system_name": system,
            "root_label": root_label,
            "num_seeds": len(items),
            "seed_list": ",".join(str(int(item["seed"])) for item in items),
            "num_basins": int(items[0]["num_basins"]),
            "raw_basin_labels": json.dumps(items[0]["raw_basin_labels"]),
            "raw_basin_distribution": json.dumps(items[0]["raw_basin_distribution"], sort_keys=True),
            "mapped_basin_distribution": json.dumps(items[0]["mapped_basin_distribution"], sort_keys=True),
        }
        for metric_name in SUMMARY_METRIC_NAMES:
            row[metric_name] = _median(float(item[metric_name]) for item in items)
        rows.append(row)
    return rows


def _aggregate_overall_root_medians(system_root_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in system_root_rows:
        grouped[str(row["root_label"])].append(row)

    rows: List[Dict[str, object]] = []
    for root_label, items in sorted(grouped.items()):
        row: Dict[str, object] = {
            "root_label": root_label,
            "num_systems": len(items),
            "systems": ",".join(sorted(str(item["system_name"]) for item in items)),
        }
        for metric_name in SUMMARY_METRIC_NAMES:
            row[metric_name] = _median(float(item[metric_name]) for item in items)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_markdown(
    *,
    path: Path,
    results: Sequence[Dict[str, object]],
    system_root_rows: Sequence[Dict[str, object]],
    overall_rows: Sequence[Dict[str, object]],
    skipped_messages: Sequence[str],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Paper Benchmark Support Alignment Summary",
        "",
        "## Evaluation Setup",
        "",
        f"- Input rows: `{args.rows_csv}`",
        f"- Systems: `{','.join(args.systems)}`",
        f"- Root labels: `{','.join(args.root_labels)}`",
        f"- Seeds: `{','.join(str(seed) for seed in args.seeds) if args.seeds else 'all in collector'}`",
        f"- Trajectories per system: `{args.num_trajectories}`",
        f"- Trajectory length: `{args.trajectory_length}`",
        f"- Long-rollout steps: `{args.long_rollout_steps}`",
        f"- Support threshold/mode: `{args.support_threshold}` / `{args.support_mode}`",
        f"- Cosine aggregation: `{args.cosine_aggregation}`",
        f"- Device: `{args.device}`",
        f"- Evaluated checkpoints: `{len(results)}`",
    ]

    if skipped_messages:
        lines.extend(["", "## Skipped / Excluded", ""])
        lines.extend(f"- {message}" for message in skipped_messages)

    lines.extend(
        [
            "",
            "## Overall Root Medians",
            "",
            "| root | systems | uniq | consistency | traj_unique | cos_sep | intra_cos | inter_cos |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overall_rows:
        lines.append(
            "| {root_label} | {num_systems} | {mode_uniqueness_rate:.3f} | "
            "{mean_basin_consistency:.3f} | {trajectory_unique_support_rate:.3f} | "
            "{cosine_separation_score:.3f} | {mean_intra_basin_cosine:.3f} | "
            "{mean_inter_basin_cosine:.3f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## System x Root Medians",
            "",
            "| system | root | basins | uniq | consistency | traj_unique | hamming_ratio | cos_sep | intra_cos | inter_cos |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in system_root_rows:
        lines.append(
            "| {system_name} | {root_label} | {num_basins} | {mode_uniqueness_rate:.3f} | "
            "{mean_basin_consistency:.3f} | {trajectory_unique_support_rate:.3f} | "
            "{between_over_within_hamming_ratio:.3f} | {cosine_separation_score:.3f} | "
            "{mean_intra_basin_cosine:.3f} | {mean_inter_basin_cosine:.3f} |".format(**row)
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate basin-support alignment on labelable paper-benchmark systems."
    )
    parser.add_argument(
        "--rows-csv",
        type=str,
        default="results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_rows.csv",
        help="Collector CSV used to resolve benchmark checkpoints.",
    )
    parser.add_argument(
        "--systems-csv",
        type=str,
        default=",".join(DEFAULT_LABELABLE_SYSTEMS),
        help="Comma-separated systems to evaluate.",
    )
    parser.add_argument(
        "--root-labels-csv",
        type=str,
        default=",".join(DEFAULT_ROOT_LABELS),
        help="Comma-separated root labels to evaluate.",
    )
    parser.add_argument(
        "--seed-csv",
        type=str,
        default="",
        help="Optional comma-separated seed filter.",
    )
    parser.add_argument(
        "--skip-invalid-competitive-lv",
        action="store_true",
        help="Skip the invalidated v4 competitive_lv checkpoints.",
    )
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--trajectory-length", type=int, default=500)
    parser.add_argument("--long-rollout-steps", type=int, default=5000)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="random",
        choices=["random", "balanced"],
    )
    parser.add_argument("--target-raw-labels-csv", type=str, default="")
    parser.add_argument("--trajectories-per-basin", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--support-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--support-mode",
        type=str,
        default="mean",
        choices=["mean", "last", "median", "majority", "modal"],
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

    args.systems = _parse_csv_strings(args.systems_csv)
    args.root_labels = _parse_csv_strings(args.root_labels_csv)
    args.seeds = _parse_csv_ints(args.seed_csv)
    target_raw_labels = _parse_csv_ints(args.target_raw_labels_csv)

    rows_csv = Path(args.rows_csv)
    with rows_csv.open() as handle:
        rows = list(csv.DictReader(handle))

    selected_rows, skipped_messages = _select_rows(
        rows=rows,
        systems=args.systems,
        root_labels=args.root_labels,
        seeds=args.seeds,
        skip_invalid_competitive_lv=args.skip_invalid_competitive_lv,
    )
    if not selected_rows:
        raise SystemExit("No rows matched the requested filters.")

    output_dir = Path(args.output_dir)
    per_run_dir = output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache: Dict[str, BasinLabeledDataset] = {}
    checkpoint_cache: Dict[Path, Dict] = {}
    results: List[Dict[str, object]] = []

    print(f"Evaluating {len(selected_rows)} checkpoints...")
    for idx, row in enumerate(selected_rows, start=1):
        checkpoint_path = Path(row["run_dir"]) / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        if checkpoint_path not in checkpoint_cache:
            checkpoint_cache[checkpoint_path] = _load_checkpoint(checkpoint_path, args.device)
        checkpoint_dict = checkpoint_cache[checkpoint_path]

        cache_key = _dataset_cache_key(
            checkpoint_dict=checkpoint_dict,
            system=row["system_key"],
            num_trajectories=args.num_trajectories,
            trajectory_length=args.trajectory_length,
            long_rollout_steps=args.long_rollout_steps,
            eval_seed=args.eval_seed,
            sampling_strategy=args.sampling_strategy,
            target_raw_labels=target_raw_labels,
            trajectories_per_basin=args.trajectories_per_basin,
            max_attempts=args.max_attempts,
        )
        if cache_key not in dataset_cache:
            dataset_cache[cache_key] = _build_dataset(
                checkpoint_dict=checkpoint_dict,
                system=row["system_key"],
                num_trajectories=args.num_trajectories,
                trajectory_length=args.trajectory_length,
                long_rollout_steps=args.long_rollout_steps,
                eval_seed=args.eval_seed,
                sampling_strategy=args.sampling_strategy,
                target_raw_labels=target_raw_labels,
                trajectories_per_basin=args.trajectories_per_basin,
                max_attempts=args.max_attempts,
            )

        dataset = dataset_cache[cache_key]
        result = _evaluate_entry(
            row=row,
            checkpoint_dict=checkpoint_dict,
            dataset=dataset,
            device=args.device,
            support_threshold=args.support_threshold,
            support_mode=args.support_mode,
            cosine_aggregation=args.cosine_aggregation,
        )
        results.append(result)

        run_output_dir = per_run_dir / row["system_key"] / row["root_label"] / f"seed_{row['seed']}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        (run_output_dir / "support_alignment.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(
            f"[{idx:03d}/{len(selected_rows):03d}] "
            f"{row['system_key']} {row['root_label']} seed={row['seed']} "
            f"uniq={result['mode_uniqueness_rate']:.3f} "
            f"cons={result['mean_basin_consistency']:.3f} "
            f"cos_sep={result['cosine_separation_score']:.3f}"
        )

        del result
        if args.device == "cuda":
            torch.cuda.empty_cache()

    results_sorted = sorted(
        results,
        key=lambda item: (
            str(item["system_name"]),
            str(item["root_label"]),
            int(item["seed"]),
        ),
    )
    system_root_rows = _aggregate_by_system_and_root(results_sorted)
    overall_rows = _aggregate_overall_root_medians(system_root_rows)

    (output_dir / "support_alignment_rows.json").write_text(
        json.dumps(results_sorted, indent=2) + "\n"
    )
    _write_csv(output_dir / "support_alignment_rows.csv", results_sorted)
    (output_dir / "system_root_medians.json").write_text(
        json.dumps(system_root_rows, indent=2) + "\n"
    )
    _write_csv(output_dir / "system_root_medians.csv", system_root_rows)
    (output_dir / "overall_root_medians.json").write_text(
        json.dumps(overall_rows, indent=2) + "\n"
    )
    _write_csv(output_dir / "overall_root_medians.csv", overall_rows)
    _write_summary_markdown(
        path=output_dir / "summary.md",
        results=results_sorted,
        system_root_rows=system_root_rows,
        overall_rows=overall_rows,
        skipped_messages=skipped_messages,
        args=args,
    )

    print(f"Wrote summary to {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
