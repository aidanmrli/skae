"""Summarize control world-model run directories into CSV and JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from skae.benchmarks.control_world_model import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(args.runs_root.rglob("final_metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = payload.get("config", {})
        test_metrics = payload.get("test_metrics", {})
        row: Dict[str, Any] = {
            "run_dir": str(metrics_path.parent),
            "task": config.get("dataset_metadata", {}).get("task", ""),
            "variant": config.get("variant", ""),
            "seed": config.get("seed", ""),
            "data_fraction": config.get("data_fraction", ""),
            "best_validation_score": payload.get("best_validation_score", ""),
            "selection_horizon": payload.get("selection_horizon", ""),
            "checkpoint": payload.get("checkpoint", ""),
        }
        for key, value in flatten_mapping(test_metrics, prefix="test").items():
            row[key] = value
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        with args.output_csv.open("w", encoding="utf-8") as handle:
            handle.write("")
    write_json(args.output_json, {"run_count": len(rows), "rows": rows})
    print(f"Summarized {len(rows)} runs", flush=True)


def flatten_mapping(values: Dict[str, Any], *, prefix: str) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            flat.update(flatten_mapping(value, prefix=f"{prefix}/{key}"))
        else:
            flat[f"{prefix}/{key}"] = value
    return flat


if __name__ == "__main__":
    main()
