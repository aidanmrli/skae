"""Derive staged local-operator tasks from the frozen controlled task table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from experiments.neurips_2026.local_operators.contract import (
    ROUTE_PROTOCOL,
    TOTAL_TRAINING_STEPS,
    route_protocol_metadata,
)


def _tagify(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _completed_run(
    row: dict[str, str],
    *,
    base_out: Path,
    phase_label: str,
    target_variant: str,
) -> Path | None:
    seed_dir = (
        base_out
        / phase_label
        / target_variant
        / row["system_slug"]
        / f"dt_{_tagify(row['env_dt'])}"
        / f"seed_{int(row['seed'])}"
    )
    candidates = [
        path
        for path in seed_dir.glob("20*")
        if path.is_dir() and (path / "evaluation_results_best.json").is_file()
    ]
    return sorted(candidates)[-1] if candidates else None


def prepare_tasks(
    *,
    base_task_tsv: Path,
    base_manifest_json: Path,
    output_tsv: Path,
    output_manifest_json: Path,
    source_variant: str,
    target_variant: str,
    phase_label: str,
    base_out: Path,
    skip_completed: bool,
) -> list[dict[str, str]]:
    """Build the staged task table and its canonical protocol manifest."""

    with base_task_tsv.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        source_rows = [dict(row) for row in reader]
    if not fields:
        raise ValueError(f"Base task table has no header: {base_task_tsv}")

    rows: list[dict[str, str]] = []
    skipped: list[dict[str, object]] = []
    for source in source_rows:
        row = dict(source)
        row["phase"] = phase_label
        row["model_variant"] = target_variant
        prior = (
            _completed_run(
                row,
                base_out=base_out,
                phase_label=phase_label,
                target_variant=target_variant,
            )
            if skip_completed
            else None
        )
        if prior is not None:
            skipped.append(
                {
                    "system_key": row["system_key"],
                    "seed": int(row["seed"]),
                    "completed_run": str(prior),
                }
            )
            continue
        if int(row["num_steps"]) != TOTAL_TRAINING_STEPS:
            raise ValueError(
                f"Expected {TOTAL_TRAINING_STEPS} steps, got {row['num_steps']}"
            )
        row["task_id"] = str(len(rows))
        rows.append(row)

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    manifest = json.loads(base_manifest_json.read_text())
    manifest.update(
        {
            "experiment_family": ROUTE_PROTOCOL,
            "source_variant": source_variant,
            "target_variant": target_variant,
            "task_tsv": str(output_tsv),
            "task_count": len(rows),
            "counts_by_system": dict(
                Counter(row["system_key"] for row in rows)
            ),
            "skipped_completed_count": len(skipped),
            "skipped_completed_rows": skipped,
            "staged_protocol": route_protocol_metadata(),
        }
    )
    output_manifest_json.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-task-tsv", type=Path, required=True)
    parser.add_argument("--base-manifest-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-manifest-json", type=Path, required=True)
    parser.add_argument("--source-variant", required=True)
    parser.add_argument("--target-variant", required=True)
    parser.add_argument("--phase-label", required=True)
    parser.add_argument("--base-out", type=Path, required=True)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rows = prepare_tasks(
        base_task_tsv=args.base_task_tsv,
        base_manifest_json=args.base_manifest_json,
        output_tsv=args.output_tsv,
        output_manifest_json=args.output_manifest_json,
        source_variant=args.source_variant,
        target_variant=args.target_variant,
        phase_label=args.phase_label,
        base_out=args.base_out,
        skip_completed=args.skip_completed,
    )
    print(f"Wrote {len(rows)} staged local-operator tasks to {args.output_tsv}")


if __name__ == "__main__":
    main()
