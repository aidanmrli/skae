#!/usr/bin/env python3
"""Merge stable-support-component shard outputs into one readout."""

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
    if not math.isfinite(out):
        return None
    return out


def _mean(rows: Iterable[Dict[str, object]], key: str) -> float | None:
    values = [_maybe_float(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    return float(np.mean(finite)) if finite else None


def _fmt(value: object, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _is_deep_perfect(row: Dict[str, object]) -> bool:
    if row.get("subset") != "deep" or row.get("object_kind") != "stable_support_component":
        return False
    coverage = _maybe_float(row.get("coverage"))
    h_basin = _maybe_float(row.get("h_basin_given_object"))
    h_object = _maybe_float(row.get("h_object_given_basin"))
    nmi = _maybe_float(row.get("object_basin_nmi"))
    try:
        object_count = int(float(str(row.get("object_count", ""))))
        basin_count = int(float(str(row.get("represented_basin_count", ""))))
    except ValueError:
        return False
    return (
        coverage is not None
        and coverage >= 0.999999
        and h_basin is not None
        and h_basin <= 1e-9
        and h_object is not None
        and h_object <= 1e-9
        and nmi is not None
        and nmi >= 0.999999
        and object_count == basin_count
    )


def _summary_markdown(
    *,
    input_root: Path,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    manifests: Sequence[Dict[str, object]],
    expected_tasks: int,
) -> str:
    completed_tasks = len(manifests)
    lines = [
        "# Retained-15 Stable Support Component Merge",
        "",
        f"- Input root: `{input_root}`",
        f"- Completed shard manifests: `{completed_tasks}`",
        f"- Expected tasks: `{expected_tasks or 'not specified'}`",
        f"- Metric rows: `{len(rows)}`",
        f"- Failure rows: `{len(failures)}`",
        "",
    ]

    if rows:
        by_group: Dict[tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_group[
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
                "| root | object | subset | n | coverage | H(B|obj) | H(obj|B) | NMI | count match | perfect deep | local/global |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for (root, object_kind, subset), group in sorted(by_group.items()):
            count_match = 0
            for row in group:
                try:
                    count_match += int(float(str(row["object_count"]))) == int(
                        float(str(row["represented_basin_count"]))
                    )
                except (KeyError, ValueError):
                    pass
            perfect = sum(_is_deep_perfect(row) for row in group)
            lines.append(
                "| {root} | {object_kind} | {subset} | {n} | {coverage} | {hb} | {hc} | {nmi} | {match}/{n} | {perfect}/{n} | {ratio} |".format(
                    root=root,
                    object_kind=object_kind,
                    subset=subset,
                    n=len(group),
                    coverage=_fmt(_mean(group, "coverage")),
                    hb=_fmt(_mean(group, "h_basin_given_object")),
                    hc=_fmt(_mean(group, "h_object_given_basin")),
                    nmi=_fmt(_mean(group, "object_basin_nmi")),
                    match=count_match,
                    perfect=perfect if subset == "deep" and object_kind == "stable_support_component" else "",
                    ratio=_fmt(_mean(group, "latent_mse_ratio_local_over_global")),
                )
            )
        lines.append("")

        stable_deep = [
            row
            for row in rows
            if row.get("object_kind") == "stable_support_component" and row.get("subset") == "deep"
        ]
        if stable_deep:
            lines.extend(
                [
                    "## Stable Deep Rows",
                    "",
                    "| root | system | seed | C_stab count | basin count | coverage | H(B|C) | H(C|B) | NMI | perfect |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in sorted(
                stable_deep,
                key=lambda item: (
                    str(item.get("root_label", "")),
                    str(item.get("system_key", "")),
                    int(float(str(item.get("seed", 0)))),
                ),
            ):
                lines.append(
                    "| {root} | {system} | {seed} | {count} | {basins} | {coverage} | {hb} | {hc} | {nmi} | {perfect} |".format(
                        root=row.get("root_label", ""),
                        system=row.get("system_key", ""),
                        seed=row.get("seed", ""),
                        count=row.get("object_count", ""),
                        basins=row.get("represented_basin_count", ""),
                        coverage=_fmt(_maybe_float(row.get("coverage"))),
                        hb=_fmt(_maybe_float(row.get("h_basin_given_object"))),
                        hc=_fmt(_maybe_float(row.get("h_object_given_basin"))),
                        nmi=_fmt(_maybe_float(row.get("object_basin_nmi"))),
                        perfect="yes" if _is_deep_perfect(row) else "no",
                    )
                )
            lines.append("")

    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures[:100]:
            lines.append(
                f"- `{failure.get('root_label', '')}` `{failure.get('system_key', '')}` seed `{failure.get('seed', '')}`: {failure.get('error', '')}"
            )
        if len(failures) > 100:
            lines.append(f"- ... {len(failures) - 100} additional failures omitted")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root)
    output_prefix = Path(args.output_prefix) if args.output_prefix else input_root / "merged"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    manifests: List[Dict[str, object]] = []
    for manifest_path in sorted(input_root.glob("shards/**/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        manifest["manifest_path"] = str(manifest_path)
        manifests.append(manifest)
        shard_dir = manifest_path.parent
        rows.extend(_read_csv(shard_dir / "stable_support_component_rows.csv"))
        failures.extend(_read_csv(shard_dir / "stable_support_component_failures.csv"))

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
            manifests=manifests,
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
