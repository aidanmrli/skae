#!/usr/bin/env python
"""Collect Kuramoto mode-support audit outputs into summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List


def _load_rows(task_tsv: Path) -> List[Dict[str, str]]:
    with task_tsv.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _bool_flag(value: bool) -> str:
    return "yes" if value else "no"


def _dump_json_string(payload) -> str:
    return json.dumps(payload, sort_keys=True)


def _extract_primary_summary(row: Dict[str, str], result: Dict[str, object]) -> Dict[str, object]:
    mean_result = result.get("primary_results", {}).get("mean", {})
    modal_result = result.get("primary_results", {}).get("modal", {})
    majority_result = result.get("primary_results", {}).get("majority", {})
    mode_counts = {
        int(key): int(value)
        for key, value in mean_result.get("per_basin_mode_count", {}).items()
    }
    mode_ties = {
        int(key): int(value)
        for key, value in mean_result.get("per_basin_mode_tie_count", {}).items()
    }
    all_mode_counts_ge_2 = bool(mode_counts) and all(value >= 2 for value in mode_counts.values())
    singleton_mode_basins = sum(1 for value in mode_counts.values() if value <= 1)
    tied_mode_basins = sum(1 for value in mode_ties.values() if value > 1)

    return {
        "task_id": row["task_id"],
        "family": row["family"],
        "root_label": row["root_label"],
        "seed": int(row["seed"]),
        "sampling_strategy": row["sampling_strategy"],
        "status": result.get("status", "missing"),
        "num_basins": result.get("num_basins", ""),
        "num_trajectories": result.get("num_trajectories", ""),
        "mean_unique_mode_supports": mean_result.get("unique_mode_supports", ""),
        "mean_mode_uniqueness_rate": mean_result.get("mode_uniqueness_rate", ""),
        "mean_basin_consistency": mean_result.get("mean_basin_consistency", ""),
        "mean_mode_support_size": mean_result.get("mean_mode_support_size", ""),
        "mean_pairwise_jaccard": mean_result.get("mean_pairwise_jaccard", ""),
        "mean_trajectory_unique_support_rate": mean_result.get("trajectory_unique_support_rate", ""),
        "mean_hamming_ratio": mean_result.get("between_over_within_hamming_ratio", ""),
        "mean_singleton_mode_basins": singleton_mode_basins,
        "mean_tied_mode_basins": tied_mode_basins,
        "mean_all_mode_counts_ge_2": all_mode_counts_ge_2,
        "modal_unique_mode_supports": modal_result.get("unique_mode_supports", ""),
        "majority_unique_mode_supports": majority_result.get("unique_mode_supports", ""),
        "raw_basin_distribution": _dump_json_string(result.get("raw_basin_distribution", {})),
        "error_type": result.get("error_type", ""),
        "error": result.get("error", ""),
        "result_path": str(Path(row["output_dir"]) / "analysis_results.json"),
    }


def _extract_threshold_rows(row: Dict[str, str], result: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for support_mode, sweep_rows in result.get("threshold_sweeps", {}).items():
        for sweep_row in sweep_rows:
            mode_counts = {
                int(key): int(value)
                for key, value in sweep_row.get("per_basin_mode_count", {}).items()
            }
            rows.append(
                {
                    "task_id": row["task_id"],
                    "family": row["family"],
                    "root_label": row["root_label"],
                    "seed": int(row["seed"]),
                    "sampling_strategy": row["sampling_strategy"],
                    "support_mode": support_mode,
                    "threshold": sweep_row.get("support_threshold", ""),
                    "num_basins": sweep_row.get("num_basins", ""),
                    "unique_mode_supports": sweep_row.get("unique_mode_supports", ""),
                    "mode_uniqueness_rate": sweep_row.get("mode_uniqueness_rate", ""),
                    "mean_basin_consistency": sweep_row.get("mean_basin_consistency", ""),
                    "trajectory_unique_support_rate": sweep_row.get("trajectory_unique_support_rate", ""),
                    "hamming_ratio": sweep_row.get("between_over_within_hamming_ratio", ""),
                    "all_mode_counts_ge_2": bool(mode_counts) and all(value >= 2 for value in mode_counts.values()),
                }
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_ok_rows(summary_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        if row["status"] == "ok":
            grouped[(row["family"], row["sampling_strategy"])].append(row)

    aggregate_rows = []
    for (family, sampling_strategy), rows in sorted(grouped.items()):
        full_unique = sum(
            1 for row in rows
            if row["mean_unique_mode_supports"] == row["num_basins"]
        )
        full_unique_nontrivial = sum(
            1 for row in rows
            if row["mean_unique_mode_supports"] == row["num_basins"]
            and row["mean_all_mode_counts_ge_2"]
        )
        aggregate_rows.append(
            {
                "family": family,
                "sampling_strategy": sampling_strategy,
                "num_ok": len(rows),
                "seeds_full_unique": full_unique,
                "seeds_full_unique_nontrivial": full_unique_nontrivial,
                "median_unique_mode_supports": median(
                    row["mean_unique_mode_supports"] for row in rows
                ),
                "median_basin_consistency": median(
                    row["mean_basin_consistency"] for row in rows
                ),
                "median_trajectory_unique_support_rate": median(
                    row["mean_trajectory_unique_support_rate"] for row in rows
                ),
                "median_hamming_ratio": median(
                    row["mean_hamming_ratio"] for row in rows
                ),
            }
        )
    return aggregate_rows


def _best_threshold_rows(threshold_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in threshold_rows:
        grouped[(row["family"], row["sampling_strategy"], row["support_mode"])].append(row)

    best_rows = []
    for key, rows in sorted(grouped.items()):
        best = max(
            rows,
            key=lambda row: (
                float(row["unique_mode_supports"]),
                int(bool(row["all_mode_counts_ge_2"])),
                float(row["mean_basin_consistency"]),
                -float(row["trajectory_unique_support_rate"]),
            ),
        )
        best_rows.append(best)
    return best_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--summary_dir", type=Path, required=True)
    args = parser.parse_args()

    task_rows = _load_rows(args.task_tsv)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    threshold_rows = []
    for row in task_rows:
        result_path = Path(row["output_dir"]) / "analysis_results.json"
        if not result_path.exists():
            summary_rows.append(
                {
                    "task_id": row["task_id"],
                    "family": row["family"],
                    "root_label": row["root_label"],
                    "seed": int(row["seed"]),
                    "sampling_strategy": row["sampling_strategy"],
                    "status": "missing",
                    "num_basins": "",
                    "num_trajectories": "",
                    "mean_unique_mode_supports": "",
                    "mean_mode_uniqueness_rate": "",
                    "mean_basin_consistency": "",
                    "mean_mode_support_size": "",
                    "mean_pairwise_jaccard": "",
                    "mean_trajectory_unique_support_rate": "",
                    "mean_hamming_ratio": "",
                    "mean_singleton_mode_basins": "",
                    "mean_tied_mode_basins": "",
                    "mean_all_mode_counts_ge_2": "",
                    "modal_unique_mode_supports": "",
                    "majority_unique_mode_supports": "",
                    "raw_basin_distribution": "",
                    "error_type": "",
                    "error": "",
                    "result_path": str(result_path),
                }
            )
            continue

        result = json.loads(result_path.read_text())
        summary_rows.append(_extract_primary_summary(row, result))
        if result.get("status") == "ok":
            threshold_rows.extend(_extract_threshold_rows(row, result))

    aggregate_rows = _aggregate_ok_rows(summary_rows)
    best_threshold_rows = _best_threshold_rows(threshold_rows)

    _write_csv(args.summary_dir / "kuramoto_mode_support_audit_rows.csv", summary_rows)
    (args.summary_dir / "kuramoto_mode_support_audit_rows.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n"
    )
    _write_csv(args.summary_dir / "kuramoto_mode_support_threshold_rows.csv", threshold_rows)
    (args.summary_dir / "kuramoto_mode_support_threshold_rows.json").write_text(
        json.dumps(threshold_rows, indent=2) + "\n"
    )
    _write_csv(args.summary_dir / "kuramoto_mode_support_aggregate_rows.csv", aggregate_rows)
    _write_csv(args.summary_dir / "kuramoto_mode_support_best_thresholds.csv", best_threshold_rows)

    lines = [
        "# Kuramoto Mode-Support Audit Summary",
        "",
        "## Primary Mean-Support Results",
        "",
        "| family | sampling | seed | status | uniq | basins | cons | traj_unique | hamming_ratio | singleton_basins | tied_basins | all_mode_counts_ge_2 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {family} | {sampling_strategy} | {seed} | {status} | {mean_unique_mode_supports} | "
            "{num_basins} | {mean_basin_consistency} | {mean_trajectory_unique_support_rate} | "
            "{mean_hamming_ratio} | {mean_singleton_mode_basins} | {mean_tied_mode_basins} | "
            "{all_ge_2} |".format(
                **row,
                all_ge_2=_bool_flag(bool(row["mean_all_mode_counts_ge_2"])),
            )
        )

    lines.extend(
        [
            "",
            "## Aggregate by Family × Sampling",
            "",
            "| family | sampling | ok | full_unique_seeds | full_unique_nontrivial_seeds | median_uniq | median_cons | median_traj_unique | median_hamming_ratio |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {family} | {sampling_strategy} | {num_ok} | {seeds_full_unique} | "
            "{seeds_full_unique_nontrivial} | {median_unique_mode_supports} | "
            "{median_basin_consistency} | {median_trajectory_unique_support_rate} | "
            "{median_hamming_ratio} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Best Threshold per Family × Sampling × Mode",
            "",
            "| family | sampling | mode | seed | threshold | uniq | cons | traj_unique | all_mode_counts_ge_2 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in best_threshold_rows:
        lines.append(
            "| {family} | {sampling_strategy} | {support_mode} | {seed} | {threshold} | "
            "{unique_mode_supports} | {mean_basin_consistency} | {trajectory_unique_support_rate} | "
            "{all_ge_2} |".format(
                **row,
                all_ge_2=_bool_flag(bool(row["all_mode_counts_ge_2"])),
            )
        )

    (args.summary_dir / "kuramoto_mode_support_audit_summary.md").write_text(
        "\n".join(lines) + "\n"
    )

    completed = sum(1 for row in summary_rows if row["status"] == "ok")
    print(
        f"Collected {completed}/{len(summary_rows)} completed Kuramoto mode-support audit "
        f"outputs into {args.summary_dir}"
    )


if __name__ == "__main__":
    main()
