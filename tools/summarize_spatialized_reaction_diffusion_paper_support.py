#!/usr/bin/env python3
"""Summarize paper-object support evaluations for the spatialized PDE benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_tsv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def _read_tasks(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if value == value]
    return float(mean(finite)) if finite else float("nan")


def _row_from_collection(task: Dict[str, str], payload: Dict[str, object], collection: str) -> Dict[str, object]:
    paper = payload["paper_support_objects"]  # type: ignore[index]
    collections = paper["collections"]  # type: ignore[index]
    item = collections[collection]  # type: ignore[index]
    s_abs = item["s_abs"]  # type: ignore[index]
    f_abs = item["f_abs"]  # type: ignore[index]
    forecast = payload.get("forecast", {})
    h4 = forecast.get("4", {}) if isinstance(forecast, dict) else {}
    return {
        "task_id": int(task["task_id"]),
        "source_system": task["source_system"],
        "seed": int(task["seed"]),
        "model_variant": task["model_variant"],
        "collection": collection,
        "num_states": int(item["num_states"]),  # type: ignore[index]
        "num_represented_basins": int(item["num_represented_basins"]),  # type: ignore[index]
        "h4_field_mse": float(h4.get("field_mse", float("nan"))) if isinstance(h4, dict) else float("nan"),
        "s_abs_exact_support_count": int(s_abs["exact_support_count"]),  # type: ignore[index]
        "s_abs_h_basin_given": float(s_abs["h_basin_given_s_abs"]),  # type: ignore[index]
        "s_abs_h_object_given_basin": float(s_abs["h_s_abs_given_basin"]),  # type: ignore[index]
        "s_abs_u_exact": float(s_abs["u_exact"]),  # type: ignore[index]
        "s_abs_support_size_mean": float(s_abs["support_size_mean"]),  # type: ignore[index]
        "s_abs_zero_support_fraction": float(s_abs["zero_support_fraction"]),  # type: ignore[index]
        "f_abs_family_count": int(f_abs["family_count"]),  # type: ignore[index]
        "f_abs_h_basin_given": float(f_abs["h_basin_given_f_abs"]),  # type: ignore[index]
        "f_abs_h_object_given_basin": float(f_abs["h_f_abs_given_basin"]),  # type: ignore[index]
        "f_abs_u_family": float(f_abs["u_family"]),  # type: ignore[index]
        "f_abs_purity": float(f_abs["purity"]),  # type: ignore[index]
        "f_abs_nmi": float(f_abs["nmi"]),  # type: ignore[index]
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for task in _read_tasks(Path(args.task_tsv)):
        output_path = Path(task["output_path"])
        if not output_path.exists():
            missing.append(str(output_path))
            continue
        payload = json.loads(output_path.read_text())
        collections = payload["paper_support_objects"]["collections"].keys()
        for collection in collections:
            rows.append(_row_from_collection(task, payload, collection))

    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_variant"]), str(row["collection"]))].append(row)

    summary_rows = []
    metric_names = [
        "h4_field_mse",
        "s_abs_exact_support_count",
        "s_abs_h_basin_given",
        "s_abs_h_object_given_basin",
        "s_abs_u_exact",
        "s_abs_support_size_mean",
        "s_abs_zero_support_fraction",
        "f_abs_family_count",
        "f_abs_h_basin_given",
        "f_abs_h_object_given_basin",
        "f_abs_u_family",
        "f_abs_purity",
        "f_abs_nmi",
    ]
    for (model_variant, collection), group_rows in sorted(grouped.items()):
        summary = {
            "model_variant": model_variant,
            "collection": collection,
            "num_runs": len(group_rows),
        }
        for metric in metric_names:
            summary[f"{metric}_mean"] = _finite_mean(float(row[metric]) for row in group_rows)
        summary_rows.append(summary)

    _write_csv(Path(args.output_csv), rows)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(
            {
                "status": "completed" if not missing else "incomplete",
                "num_rows": len(rows),
                "missing_outputs": missing,
                "summary": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"rows": len(rows), "missing": len(missing)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
