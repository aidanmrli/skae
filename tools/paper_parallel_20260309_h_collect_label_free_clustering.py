#!/usr/bin/env python
"""Collect Subagent H label-free clustering outputs into one summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RESULT_METRICS = {
    "adjusted_rand_index": "adjusted_rand_index",
    "normalized_mutual_info": "normalized_mutual_info",
    "silhouette_score": "silhouette_score",
    "kmeans_purity": "kmeans_purity",
    "linear_classifier_accuracy": "linear_classifier_accuracy",
    "linear_classifier_cv_std": "linear_classifier_cv_std",
    "mean_sparsity": "mean_sparsity",
    "mean_l1_norm": "mean_l1_norm",
    "num_ground_truth_basins": "num_ground_truth_basins",
    "num_trajectories": "result_num_trajectories",
    "latent_dim": "latent_dim",
}


def load_rows(task_tsv: Path):
    with task_tsv.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--summary_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.task_tsv)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for row in rows:
        result_path = Path(row["output_dir"]) / "analysis_results.json"
        summary_row = dict(row)
        summary_row["result_path"] = str(result_path)
        summary_row["status"] = "missing"
        for summary_key in RESULT_METRICS.values():
            summary_row[summary_key] = ""

        if result_path.exists():
            result = json.loads(result_path.read_text())
            summary_row["status"] = "ok"
            for result_key, summary_key in RESULT_METRICS.items():
                summary_row[summary_key] = result.get(result_key, "")
        summary_rows.append(summary_row)

    csv_path = args.summary_dir / "label_free_clustering_rows.csv"
    json_path = args.summary_dir / "label_free_clustering_rows.json"
    md_path = args.summary_dir / "label_free_clustering_summary.md"

    fieldnames = list(summary_rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path.write_text(json.dumps(summary_rows, indent=2) + "\n")

    lines = [
        "# Label-Free Clustering Summary",
        "",
        "| system | family | root_label | seed | status | ARI | NMI | silhouette | purity | linear_acc |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {system} | {family} | {root_label} | {seed} | {status} | {adjusted_rand_index} | "
            "{normalized_mutual_info} | {silhouette_score} | {kmeans_purity} | "
            "{linear_classifier_accuracy} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n")

    completed = sum(1 for row in summary_rows if row["status"] == "ok")
    print(f"Collected {completed}/{len(summary_rows)} completed outputs into {args.summary_dir}")


if __name__ == "__main__":
    main()
