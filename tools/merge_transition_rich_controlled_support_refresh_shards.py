#!/usr/bin/env python3
"""Merge controlled support-refresh shard outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List


def _load_tool():
    tool_path = Path(__file__).with_name("evaluate_transition_rich_controlled_support_refresh.py")
    spec = importlib.util.spec_from_file_location("evaluate_transition_rich_controlled_support_refresh_merge", tool_path)
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
        return list(csv.DictReader(handle))


def main() -> None:
    args = _parse_args()
    shards_dir = Path(args.shards_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    completed_specs = 0
    specs_count = 0
    shard_manifests: List[Dict[str, object]] = []

    for shard_dir in sorted(path for path in shards_dir.iterdir() if path.is_dir()):
        rows.extend(_read_rows(shard_dir / "controlled_support_refresh_rows.csv"))
        failure_path = shard_dir / "failures.json"
        if failure_path.exists():
            failures.extend(json.loads(failure_path.read_text()))
        manifest_path = shard_dir / "manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text())
            shard_manifests.append(
                {
                    "shard": shard_dir.name,
                    "status": payload.get("status"),
                    "row_count": payload.get("row_count"),
                    "failure_count": payload.get("failure_count"),
                    "completed_specs": payload.get("completed_specs"),
                    "specs_count": payload.get("specs_count"),
                }
            )
            completed_specs += int(payload.get("completed_specs", 0))
            specs_count += int(payload.get("specs_count", 0))

    TOOL._write_csv(output_dir / "controlled_support_refresh_rows.csv", rows)
    TOOL._write_summary(output_dir / "controlled_support_refresh_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows_csvs": TOOL.REFRESH._parse_csv_strings(args.rows_csvs),
                "root_labels": TOOL.REFRESH._parse_csv_strings(args.root_labels),
                "systems": TOOL.REFRESH._parse_csv_strings(args.systems),
                "seeds": TOOL.REFRESH._parse_csv_ints(args.seeds),
                "specs_count": specs_count,
                "completed_specs": completed_specs,
                "remaining_specs": max(0, specs_count - completed_specs),
                "row_count": len(rows),
                "failure_count": len(failures),
                "status": "complete" if not failures else "complete_with_failures",
                "shards": shard_manifests,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
