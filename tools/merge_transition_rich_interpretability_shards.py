#!/usr/bin/env python3
"""Merge per-root interpretability shards into the standard final artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards_dir", required=True, help="parent directory containing per-root shard outputs")
    parser.add_argument("--output_dir", required=True, help="directory for merged interpretability artifacts")
    parser.add_argument("--rows_csv", default="", help="source forecasting_rows.csv used to build the shards")
    parser.add_argument("--root_labels", default="", help="comma-separated root labels expected in the shard set")
    parser.add_argument("--systems", default="", help="optional comma-separated system filter used by shards")
    parser.add_argument("--seeds", default="", help="optional comma-separated seed filter used by shards")
    return parser.parse_args()


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _read_rows(path: Path) -> List[Dict[str, object]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized: List[Dict[str, object]] = []
    for row in rows:
        normalized.append(
            {
                key: (None if value == "" else value)
                for key, value in row.items()
            }
        )
    return normalized


def _read_failures(path: Path) -> List[Dict[str, object]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    return json.loads(path.read_text())


def _load_reducer_helpers():
    reducer_path = Path(__file__).with_name("reduce_transition_rich_interpretability_metrics.py")
    spec = importlib.util.spec_from_file_location("reduce_transition_rich_interpretability_metrics", reducer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load reducer helpers from {reducer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = _parse_args()
    shards_dir = Path(args.shards_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(path for path in shards_dir.iterdir() if path.is_dir())
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    shard_summaries: List[Dict[str, object]] = []
    for shard_dir in shard_dirs:
        rows.extend(_read_rows(shard_dir / "interpretability_rows.csv"))
        failures.extend(_read_failures(shard_dir / "failures.json"))
        manifest_path = shard_dir / "manifest.json"
        manifest: Dict[str, object] = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
        shard_summaries.append(
            {
                "path": str(shard_dir),
                "num_rows": int(manifest.get("num_rows", 0)),
                "num_failures": int(manifest.get("num_failures", 0)),
                "completed_runs": int(manifest.get("completed_runs", 0)),
                "num_runs": int(manifest.get("num_runs", 0)),
                "status": str(manifest.get("status", "")),
            }
        )

    reducer = _load_reducer_helpers()
    reducer._write_csv(output_dir / "interpretability_rows.csv", rows)
    reducer._write_summary(output_dir / "interpretability_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows_csv": args.rows_csv,
                "root_labels": _parse_csv(args.root_labels),
                "systems": _parse_csv(args.systems),
                "seeds": _parse_csv(args.seeds),
                "shards_dir": str(shards_dir),
                "shard_count": len(shard_dirs),
                "num_rows": len(rows),
                "num_failures": len(failures),
                "status": "complete",
                "shards": shard_summaries,
            },
            indent=2,
        )
    )
    (output_dir / "progress.json").write_text(
        json.dumps(
            {
                "completed_shards": len(shard_dirs),
                "num_shards": len(shard_dirs),
                "num_rows": len(rows),
                "num_failures": len(failures),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "num_shards": len(shard_dirs),
                "num_rows": len(rows),
                "num_failures": len(failures),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
