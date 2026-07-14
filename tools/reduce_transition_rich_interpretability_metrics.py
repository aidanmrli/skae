#!/usr/bin/env python3
"""Reduce the exact controlled-paper basin/support alignment protocol.

The numerical protocol lives in ``skae.benchmarks.controlled_alignment``.
This entry point only discovers checkpoints, executes the fixed evaluator,
and writes compact restart-friendly artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from skae.benchmarks.controlled_alignment import (
    DEFAULT_ROOT_LABELS,
    FAMILY_JACCARD_THRESHOLD,
    OUTPUT_COLUMNS,
    SUBSET,
    SUPPORT_SCHEME,
    alignment_protocol_metadata,
    evaluate_checkpoint_alignment,
)
from skae.benchmarks.paper_protocol import CONTROLLED_MODEL_ROW_IDS


@dataclass(frozen=True)
class RunSpec:
    root_label: str
    system_key: str
    system_name: str
    seed: int
    run_dir: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--root_labels", default=DEFAULT_ROOT_LABELS)
    parser.add_argument("--systems", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
    )
    parser.add_argument("--progress_every_runs", type=int, default=1)
    parser.add_argument("--flush_every_runs", type=int, default=0)
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _run_timestamp_key(run_dir: str) -> Tuple[str, str]:
    stem = Path(run_dir).name
    return (stem if re.fullmatch(r"\d{8}-\d{6}", stem) else "", run_dir)


def _load_latest_specs(
    rows_csv: Path,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
) -> List[RunSpec]:
    selected_roots = set(root_labels)
    selected_systems = set(systems)
    selected_seeds = set(seeds)
    latest: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    with rows_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            root_label = str(row.get("root_label", "")).strip()
            system_key = str(row.get("system_key", "")).strip()
            if root_label not in selected_roots:
                continue
            if selected_systems and system_key not in selected_systems:
                continue
            seed = int(row.get("seed", 0))
            if selected_seeds and seed not in selected_seeds:
                continue
            run_dir = str(row.get("run_dir", "")).strip()
            if not run_dir:
                continue
            key = (root_label, system_key, seed)
            incumbent = latest.get(key)
            if incumbent is None or _run_timestamp_key(run_dir) > _run_timestamp_key(
                incumbent["run_dir"]
            ):
                latest[key] = row
    specs = [
        RunSpec(
            root_label=str(row["root_label"]),
            system_key=str(row["system_key"]),
            system_name=str(row.get("system_name", row["system_key"])),
            seed=int(row["seed"]),
            run_dir=str(row["run_dir"]),
        )
        for row in latest.values()
    ]
    return sorted(specs, key=lambda item: (item.root_label, item.system_key, item.seed))


def reduce_run(
    spec: RunSpec,
    *,
    device: str,
) -> Dict[str, object]:
    metrics = evaluate_checkpoint_alignment(
        Path(spec.run_dir) / "checkpoint.pt",
        spec.system_key,
        device=device,
    )
    return {
        "root_label": spec.root_label,
        "system_name": spec.system_name,
        "seed": spec.seed,
        "support_scheme": SUPPORT_SCHEME,
        "subset": SUBSET,
        "num_states": metrics["num_states"],
        "observed_label_count": metrics["observed_label_count"],
        "family_jaccard_threshold": FAMILY_JACCARD_THRESHOLD,
        "family_h_basin_given_family": metrics[
            "family_h_basin_given_family"
        ],
        "family_unique_count": metrics["family_unique_count"],
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(OUTPUT_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in OUTPUT_COLUMNS})


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    num_specs: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "protocol": alignment_protocol_metadata(),
                "rows_csv": args.rows_csv,
                "root_labels": list(root_labels),
                "systems": list(systems),
                "seeds": list(seeds),
                "num_runs": num_specs,
                "completed_runs": completed_specs,
                "remaining_runs": max(0, num_specs - completed_specs),
                "num_rows": len(rows),
                "num_failures": len(failures),
                "status": status,
            },
            indent=2,
        )
        + "\n"
    )


def _flush_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    num_specs: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
    last_spec: Optional[RunSpec],
    last_status: Optional[str],
    last_error: Optional[str],
) -> None:
    _write_csv(output_dir / "interpretability_rows.csv", rows)
    (output_dir / "failures.json").write_text(
        json.dumps(list(failures), indent=2) + "\n"
    )
    _write_manifest(
        output_dir / "manifest.json",
        args=args,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        num_specs=num_specs,
        completed_specs=completed_specs,
        rows=rows,
        failures=failures,
        status=status,
    )
    progress: Dict[str, object] = {
        "completed_runs": completed_specs,
        "num_runs": num_specs,
        "remaining_runs": max(0, num_specs - completed_specs),
        "num_rows": len(rows),
        "num_failures": len(failures),
        "elapsed_seconds": elapsed_seconds,
    }
    if last_spec is not None:
        progress["last_completed_spec"] = {
            "root_label": last_spec.root_label,
            "system_key": last_spec.system_key,
            "seed": last_spec.seed,
            "run_dir": last_spec.run_dir,
            "status": last_status,
        }
    if last_error:
        progress["last_error"] = last_error
    (output_dir / "progress.json").write_text(
        json.dumps(progress, indent=2) + "\n"
    )


def _flush(
    output_dir: Path,
    args: argparse.Namespace,
    roots: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    specs: Sequence[RunSpec],
    completed: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    started: float,
    last_spec: Optional[RunSpec],
    last_status: Optional[str],
    last_error: Optional[str],
) -> None:
    _flush_outputs(
        output_dir,
        args=args,
        root_labels=roots,
        systems=systems,
        seeds=seeds,
        num_specs=len(specs),
        completed_specs=completed,
        rows=rows,
        failures=failures,
        status=status,
        elapsed_seconds=time.time() - started,
        last_spec=last_spec,
        last_status=last_status,
        last_error=last_error,
    )


def main() -> None:
    args = _parse_args()
    roots = _parse_csv_strings(args.root_labels)
    unknown_roots = sorted(set(roots) - set(CONTROLLED_MODEL_ROW_IDS))
    if unknown_roots:
        raise ValueError(f"Rows outside the controlled paper roster: {unknown_roots}")
    systems = _parse_csv_strings(args.systems)
    seeds = _parse_csv_ints(args.seeds)
    specs = _load_latest_specs(Path(args.rows_csv), roots, systems, seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    started = time.time()
    progress_every = max(1, int(args.progress_every_runs))
    flush_every = max(0, int(args.flush_every_runs))
    last_spec: Optional[RunSpec] = None
    last_status: Optional[str] = None
    last_error: Optional[str] = None
    _flush(
        output_dir,
        args,
        roots,
        systems,
        seeds,
        specs,
        0,
        rows,
        failures,
        "running",
        started,
        None,
        None,
        None,
    )

    for index, spec in enumerate(specs, start=1):
        last_spec, last_status, last_error = spec, "ok", None
        try:
            rows.append(
                reduce_run(
                    spec,
                    device=args.device,
                )
            )
        except Exception as exc:  # pragma: no cover - retain other run rows
            last_status, last_error = "failed", repr(exc)
            failures.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
                    "error": last_error,
                }
            )
        if index % progress_every == 0 or index == len(specs):
            print(
                f"[{index}/{len(specs)}] {last_status} root={spec.root_label} "
                f"system={spec.system_key} seed={spec.seed} rows={len(rows)} "
                f"failures={len(failures)} elapsed_s={time.time() - started:.1f}",
                flush=True,
            )
        if flush_every and (index % flush_every == 0 or index == len(specs)):
            _flush(
                output_dir,
                args,
                roots,
                systems,
                seeds,
                specs,
                index,
                rows,
                failures,
                (
                    "running"
                    if index < len(specs)
                    else ("failed" if failures else "complete")
                ),
                started,
                last_spec,
                last_status,
                last_error,
            )

    final_status = "failed" if failures else "complete"
    _flush(
        output_dir,
        args,
        roots,
        systems,
        seeds,
        specs,
        len(specs),
        rows,
        failures,
        final_status,
        started,
        last_spec,
        last_status,
        last_error,
    )
    print(
        json.dumps(
            {
                "num_runs": len(specs),
                "num_rows": len(rows),
                "num_failures": len(failures),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
