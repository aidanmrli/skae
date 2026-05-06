#!/usr/bin/env python3
"""Merge explicit regime-discovery local-Koopman shard outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List


def _load_tool():
    tool_path = Path(__file__).with_name("evaluate_regime_discovery_local_koopman.py")
    spec = importlib.util.spec_from_file_location("evaluate_regime_discovery_local_koopman_merge", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load tool helpers from {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rows_csvs", default="", help="comma-separated forecasting_rows.csv files used to build shards")
    parser.add_argument("--root_labels", default="", help="comma-separated root labels included in shards")
    parser.add_argument("--systems", default="", help="optional comma-separated system filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated seed filter")
    return parser.parse_args()


def _read_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def main() -> None:
    args = _parse_args()
    shards_dir = Path(args.shards_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    completed_runs = 0
    num_runs = 0

    for shard_dir in sorted(path for path in shards_dir.iterdir() if path.is_dir()):
        rows.extend(_read_rows(shard_dir / "regime_discovery_local_koopman_rows.csv"))
        failure_path = shard_dir / "failures.json"
        if failure_path.exists():
            failures.extend(json.loads(failure_path.read_text()))
        manifest_path = shard_dir / "manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text())
            completed_runs += int(payload.get("completed_runs", 0))
            num_runs += int(payload.get("num_runs", 0))

    TOOL._write_csv(output_dir / "regime_discovery_local_koopman_rows.csv", rows)
    TOOL._write_summary(output_dir / "regime_discovery_local_koopman_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows_csvs": TOOL._parse_csv_strings(args.rows_csvs),
                "root_labels": TOOL._parse_csv_strings(args.root_labels),
                "systems": TOOL._parse_csv_strings(args.systems),
                "seeds": TOOL._parse_csv_ints(args.seeds),
                "num_runs": num_runs,
                "completed_runs": completed_runs,
                "remaining_runs": max(0, num_runs - completed_runs),
                "num_rows": len(rows),
                "num_failures": len(failures),
                "status": "complete" if not failures else "complete_with_failures",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
