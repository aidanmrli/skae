#!/usr/bin/env python3
"""Merge stage-2 support-family local-map shard outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List


def _load_tool():
    tool_path = Path(__file__).with_name("train_support_family_local_maps.py")
    spec = importlib.util.spec_from_file_location("train_support_family_local_maps_merge", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load stage-2 helpers from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = _parse_args()
    shards_dir = Path(args.shards_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    completed_runs = 0
    total_runs = 0
    for shard_dir in sorted(path for path in shards_dir.iterdir() if path.is_dir()):
        rows.extend(_read_rows(shard_dir / "self_routed_forecasting_rows.csv"))
        failure_path = shard_dir / "failures.json"
        if failure_path.exists():
            failures.extend(json.loads(failure_path.read_text()))
        manifest_path = shard_dir / "manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text())
            completed_runs += int(payload.get("completed_runs", 0))
            total_runs += int(payload.get("total_runs", payload.get("num_runs", 0)))

    horizons = []
    for row in rows:
        for key in row.keys():
            if key.startswith("h") and key.endswith("_mean"):
                try:
                    horizons.append(int(key[1:-5]))
                except ValueError:
                    continue
    horizons = sorted(set(horizons))
    TOOL._write_csv(output_dir / "self_routed_forecasting_rows.csv", rows)
    TOOL._write_summary(output_dir / "self_routed_forecasting_summary.md", rows, horizons)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete" if not failures else "complete_with_failures",
                "completed_runs": completed_runs,
                "total_runs": total_runs,
                "num_rows": len(rows),
                "num_failures": len(failures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
