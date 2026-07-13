#!/usr/bin/env python3
"""Build support-threshold sweep tasks for existing spatialized PDE checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _parse_csv_strings(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _find_dataset_path(checkpoint_path: Path) -> Path:
    for parent in checkpoint_path.parents:
        candidate = parent / "dataset.pt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find dataset.pt above checkpoint {checkpoint_path}")


def _read_existing_evaluation(task_context: Path) -> Dict[str, object]:
    candidate = task_context / "evaluation.json"
    if not candidate.is_file():
        return {}
    try:
        return json.loads(candidate.read_text())
    except json.JSONDecodeError:
        return {}


def _metadata_from_relative(rel_context: Path) -> Dict[str, str]:
    parts = rel_context.parts
    if "runs" not in parts:
        return {"model_variant": "", "source_system": "", "grid": "", "seed": "", "setting_slug": ""}
    start = parts.index("runs")
    tail = parts[start + 1 :]
    out = {"model_variant": "", "source_system": "", "grid": "", "seed": "", "setting_slug": ""}
    if len(tail) >= 4:
        out["model_variant"] = tail[0]
        out["source_system"] = tail[1]
        out["grid"] = tail[2]
        out["seed"] = tail[3]
    if len(tail) >= 5:
        out["setting_slug"] = "/".join(tail[4:])
    return out


def _iter_checkpoints(roots: Sequence[Path]) -> Iterable[tuple[Path, Path, Path]]:
    seen: set[Path] = set()
    for root in roots:
        for checkpoint in sorted(root.glob("runs/**/model/checkpoint.pt")):
            checkpoint = checkpoint.resolve()
            if checkpoint in seen:
                continue
            seen.add(checkpoint)
            yield root.resolve(), checkpoint, checkpoint.parent.parent.resolve()


def _write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_roots_csv", required=True)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_manifest_json", default=None)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--support_thresholds_csv", default="0.01,0.03,0.05,0.1,0.2,0.3")
    parser.add_argument("--family_jaccards_csv", default="0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--deep_threshold", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(item) for item in _parse_csv_strings(args.input_roots_csv)]
    output_root = Path(args.output_root)
    rows: List[Dict[str, object]] = []
    for task_id, (root, checkpoint, task_context) in enumerate(_iter_checkpoints(roots)):
        rel_context = task_context.relative_to(root)
        meta = _metadata_from_relative(rel_context)
        existing_eval = _read_existing_evaluation(task_context)
        dataset = _find_dataset_path(checkpoint)
        source_system = str(existing_eval.get("source_system") or meta["source_system"])
        output = output_root / rel_context / "support_threshold_sweep.json"
        rows.append(
            {
                "task_id": task_id,
                "source_system": source_system,
                "seed": meta["seed"].replace("seed_", ""),
                "model_variant": meta["model_variant"],
                "grid": meta["grid"],
                "setting_slug": meta["setting_slug"],
                "dataset": str(dataset),
                "checkpoint": str(checkpoint),
                "output": str(output),
                "support_thresholds_csv": args.support_thresholds_csv,
                "family_jaccards_csv": args.family_jaccards_csv,
                "batch_size": int(args.batch_size),
                "deep_threshold": float(args.deep_threshold),
            }
        )
    _write_tsv(Path(args.output_tsv), rows)
    if args.output_manifest_json:
        Path(args.output_manifest_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_manifest_json).write_text(
            json.dumps(
                {
                    "num_tasks": len(rows),
                    "input_roots": [str(root) for root in roots],
                    "output_root": str(output_root),
                    "support_thresholds_csv": args.support_thresholds_csv,
                    "family_jaccards_csv": args.family_jaccards_csv,
                    "deep_threshold": float(args.deep_threshold),
                    "batch_size": int(args.batch_size),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(f"Wrote {len(rows)} support-sweep tasks to {args.output_tsv}")


if __name__ == "__main__":
    main()
