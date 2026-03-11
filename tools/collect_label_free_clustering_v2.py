#!/usr/bin/env python
"""Collect label-free clustering v2 results into a summary table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_tsv", required=True, help="TSV with task specs")
    parser.add_argument("--base_out", required=True, help="Base output directory")
    parser.add_argument("--output_dir", required=True, help="Directory for summary files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_out = Path(args.base_out)

    # Read task specs
    tasks = []
    with open(args.task_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tasks.append(row)

    # Collect results
    all_rows = []
    for task in tasks:
        task_id = task["task_id"]
        result_dir = base_out / "eval" / task["system"] / task["family"] / f"seed_{task['seed']}"
        result_file = result_dir / "analysis_results.json"

        if not result_file.exists():
            print(f"MISSING: {result_file}")
            continue

        with open(result_file) as f:
            result = json.load(f)

        for vr in result.get("view_results", []):
            row = {
                "task_id": task_id,
                "system": task["system"],
                "family": task["family"],
                "root_label": task["root_label"],
                "seed": task["seed"],
                "feature_view": vr["feature_view"],
                "num_basins": result["num_ground_truth_basins"],
                "num_trajectories": result["num_trajectories"],
                "ARI": vr["adjusted_rand_index"],
                "NMI": vr["normalized_mutual_info"],
                "silhouette": vr["silhouette_score"],
                "purity": vr["kmeans_purity"],
                "linear_acc": vr["linear_classifier_accuracy"],
                "linear_acc_std": vr["linear_classifier_cv_std"],
                "mean_sparsity": result["mean_sparsity"],
                "mean_l1_norm": result["mean_l1_norm"],
                "status": "ok",
            }
            all_rows.append(row)

    if not all_rows:
        print("No results found.")
        return

    # Write CSV
    csv_path = output_dir / "label_free_clustering_v2_rows.csv"
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {csv_path}")

    # Write markdown summary (best view per system×family)
    md_path = output_dir / "label_free_clustering_v2_summary.md"
    lines = ["# Label-Free Clustering V2 Summary\n"]

    # Group by view for a cross-view comparison table
    views = sorted(set(r["feature_view"] for r in all_rows))
    systems = sorted(set(r["system"] for r in all_rows))
    families = sorted(set(r["family"] for r in all_rows))

    for view in views:
        lines.append(f"\n## Feature View: `{view}`\n")
        lines.append("| system | family | seed | basins | ARI | NMI | silhouette | purity | linear_acc |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

        for system in systems:
            for family in families:
                matching = [
                    r for r in all_rows
                    if r["system"] == system
                    and r["family"] == family
                    and r["feature_view"] == view
                ]
                for r in matching:
                    lines.append(
                        f"| {r['system']} | {r['family']} | {r['seed']} | {r['num_basins']} "
                        f"| {r['ARI']:.4f} | {r['NMI']:.4f} | {r['silhouette']:.4f} "
                        f"| {r['purity']:.4f} | {r['linear_acc']:.4f} |"
                    )

    # Cross-view best-ARI summary
    lines.append("\n## Best View per System × Family (by ARI)\n")
    lines.append("| system | family | best_view | ARI | NMI | purity | linear_acc |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")

    for system in systems:
        for family in families:
            matching = [
                r for r in all_rows
                if r["system"] == system and r["family"] == family
            ]
            if not matching:
                continue
            best = max(matching, key=lambda r: r["ARI"])
            lines.append(
                f"| {system} | {family} | {best['feature_view']} "
                f"| {best['ARI']:.4f} | {best['NMI']:.4f} "
                f"| {best['purity']:.4f} | {best['linear_acc']:.4f} |"
            )

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote summary to {md_path}")


if __name__ == "__main__":
    main()
