#!/usr/bin/env python3
"""Merge latent-fate component shard outputs into one readout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_prefix", default="")
    parser.add_argument("--expected_tasks", type=int, default=0)
    return parser.parse_args()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _maybe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(rows: Iterable[Dict[str, object]], key: str) -> float | None:
    values = [_maybe_float(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _fmt(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _summary_markdown(
    *,
    input_root: Path,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    completed_manifests: int,
    expected_tasks: int,
) -> str:
    lines = [
        "# Dense Latent-Fate Component Merge",
        "",
        f"- Input root: `{input_root}`",
        f"- Completed shard manifests: `{completed_manifests}`",
        f"- Expected tasks: `{expected_tasks or 'not specified'}`",
        f"- Metric rows: `{len(rows)}`",
        f"- Failure rows: `{len(failures)}`",
        "",
    ]
    if rows:
        groups: Dict[tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            groups[
                (
                    str(row.get("root_label", "")),
                    str(row.get("object_kind", "")),
                    str(row.get("subset", "")),
                )
            ].append(row)
        lines.extend(
            [
                "## Aggregate Metrics",
                "",
                "| root | object | subset | n | H(B|obj) | H(obj|B) | NMI | mean selected k | count match |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (root, object_kind, subset), group in sorted(groups.items()):
            count_match = 0
            for row in group:
                try:
                    count_match += int(float(str(row["object_count"]))) == int(
                        float(str(row["represented_basin_count"]))
                    )
                except (KeyError, ValueError):
                    pass
            lines.append(
                "| {root} | {object_kind} | {subset} | {n} | {hb} | {hc} | {nmi} | {k} | {match}/{n} |".format(
                    root=root,
                    object_kind=object_kind,
                    subset=subset,
                    n=len(group),
                    hb=_fmt(_mean(group, "h_basin_given_object")),
                    hc=_fmt(_mean(group, "h_object_given_basin")),
                    nmi=_fmt(_mean(group, "object_basin_nmi")),
                    k=_fmt(_mean(group, "selected_k")),
                    match=count_match,
                )
            )
        lines.extend(
            [
                "",
                "## Deep Rows",
                "",
                "| root | system | seed | object | selected k | object count | basin count | H(B|obj) | H(obj|B) | NMI |",
                "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(
            [item for item in rows if item.get("subset") == "deep"],
            key=lambda item: (
                str(item.get("root_label", "")),
                str(item.get("system_key", "")),
                str(item.get("object_kind", "")),
                int(float(str(item.get("seed", 0)))),
            ),
        ):
            lines.append(
                "| {root} | {system} | {seed} | {object_kind} | {selected_k} | {count} | {basins} | {hb} | {hc} | {nmi} |".format(
                    root=row.get("root_label", ""),
                    system=row.get("system_key", ""),
                    seed=row.get("seed", ""),
                    object_kind=row.get("object_kind", ""),
                    selected_k=row.get("selected_k", ""),
                    count=row.get("object_count", ""),
                    basins=row.get("represented_basin_count", ""),
                    hb=_fmt(_maybe_float(row.get("h_basin_given_object"))),
                    hc=_fmt(_maybe_float(row.get("h_object_given_basin"))),
                    nmi=_fmt(_maybe_float(row.get("object_basin_nmi"))),
                )
            )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures[:100]:
            lines.append(
                f"- `{failure.get('root_label', '')}` `{failure.get('system_key', '')}` seed `{failure.get('seed', '')}`: {failure.get('error', '')}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root)
    output_prefix = Path(args.output_prefix) if args.output_prefix else input_root / "merged"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    manifests = []
    for manifest_path in sorted(input_root.glob("shards/**/manifest.json")):
        try:
            manifests.append(json.loads(manifest_path.read_text()))
        except json.JSONDecodeError:
            continue
        shard_dir = manifest_path.parent
        rows.extend(_read_csv(shard_dir / "latent_fate_component_rows.csv"))
        failures.extend(_read_csv(shard_dir / "latent_fate_component_failures.csv"))

    _write_csv(Path(f"{output_prefix}_rows.csv"), rows)
    _write_csv(Path(f"{output_prefix}_failures.csv"), failures)
    Path(f"{output_prefix}_manifest.json").write_text(
        json.dumps(
            {
                "input_root": str(input_root),
                "expected_tasks": args.expected_tasks,
                "completed_manifests": len(manifests),
                "metric_rows": len(rows),
                "failure_rows": len(failures),
            },
            indent=2,
        )
    )
    Path(f"{output_prefix}_summary.md").write_text(
        _summary_markdown(
            input_root=input_root,
            rows=rows,
            failures=failures,
            completed_manifests=len(manifests),
            expected_tasks=args.expected_tasks,
        )
    )
    print(
        json.dumps(
            {
                "completed_manifests": len(manifests),
                "metric_rows": len(rows),
                "failure_rows": len(failures),
                "summary": f"{output_prefix}_summary.md",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
