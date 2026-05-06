#!/usr/bin/env python3
"""Combine selected stage-2 routed rows across seed batches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector_json", required=True)
    parser.add_argument("--input_csvs", required=True, help="comma-separated self_routed_forecasting_rows.csv files")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument(
        "--target_root_label",
        default="",
        help="override root label to match; otherwise root_label and dysts_root_label from selector are both accepted",
    )
    return parser.parse_args()


def parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_selector(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    return dict(payload["best"] if "best" in payload else payload)


def matches(row: dict[str, str], selector: dict[str, object], *, target_root_label: str = "") -> bool:
    if target_root_label:
        root_labels = {target_root_label}
    else:
        root_labels = {str(selector["root_label"])}
        if selector.get("dysts_root_label"):
            root_labels.add(str(selector["dysts_root_label"]))
    return (
        str(row.get("root_label", "")) in root_labels
        and int(float(row.get("reencode_period", 0))) == int(selector["reencode_period"])
        and str(row.get("route_freeze_mode", "")) == str(selector["route_freeze_mode"])
        and str(row.get("local_map_source", "")) == "stage2_rollout_trained"
    )


def read_matching_rows(
    paths: Iterable[str],
    selector: dict[str, object],
    *,
    target_root_label: str = "",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in paths:
        path = Path(item)
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open("r", newline="") as handle:
            for row in csv.DictReader(handle):
                if matches(row, selector, target_root_label=target_root_label):
                    rows.append(dict(row))
    return rows


def write_rows(path: str | Path, rows: list[dict[str, str]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    selector = load_selector(args.selector_json)
    rows = read_matching_rows(
        parse_csv_items(args.input_csvs),
        selector,
        target_root_label=args.target_root_label,
    )
    if not rows:
        raise SystemExit("No matching rows found")
    write_rows(args.output_csv, rows)
    print(f"Wrote {args.output_csv} with {len(rows)} rows")


if __name__ == "__main__":
    main()
