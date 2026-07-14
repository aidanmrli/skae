"""Build task tables for long-horizon Dysts forecasting reevaluation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from experiments.neurips_2026.protocol import DYSTS_MODEL_ROW_IDS, DYSTS_PAPER_PROTOCOL

DYSTS_SYSTEMS: Sequence[str] = DYSTS_PAPER_PROTOCOL.system_keys
DEFAULT_SEEDS: Sequence[int] = DYSTS_PAPER_PROTOCOL.seeds


@dataclass(frozen=True)
class RootSpec:
    label: str
    display_name: str
    model_family: str
    root_dir: Path


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _normalize_system(name: str) -> str:
    raw = name.strip()
    if not raw:
        raise ValueError("Empty system name")
    if raw.startswith("dysts:"):
        return raw
    return f"dysts:{raw}"


def _system_dir_name(system: str) -> str:
    if system.startswith("dysts:"):
        return f"dysts_{system.split(':', 1)[1]}"
    return system


def _system_slug(system: str) -> str:
    return system.split(":", 1)[1] if system.startswith("dysts:") else system


def _raw_system_name(system: str) -> str:
    return system.split(":", 1)[1] if system.startswith("dysts:") else system


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "checkpoint.pt").exists()


def _latest_seed_run(system_dir: Path, seed: int) -> Path | None:
    run_dirs: List[Path] = []
    for seed_dir in sorted(system_dir.glob(f"**/seed_{seed}")):
        if not seed_dir.is_dir():
            continue
        for child in sorted(seed_dir.iterdir()):
            if _is_run_dir(child):
                run_dirs.append(child)
        if _is_run_dir(seed_dir):
            run_dirs.append(seed_dir)
    if not run_dirs:
        return None
    return sorted(set(run_dirs))[-1]


def _read_root_specs(path: Path) -> List[RootSpec]:
    rows: List[RootSpec] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            label = str(row.get("label", "")).strip()
            display_name = str(row.get("display_name", "")).strip()
            model_family = str(row.get("model_family", "")).strip()
            root_dir = str(row.get("root_dir", "")).strip()
            if not label or not display_name or not model_family or not root_dir:
                raise ValueError(f"Malformed root spec row: {row}")
            rows.append(
                RootSpec(
                    label=label,
                    display_name=display_name,
                    model_family=model_family,
                    root_dir=Path(root_dir),
                )
            )
    labels = tuple(spec.label for spec in rows)
    if labels != DYSTS_MODEL_ROW_IDS:
        raise ValueError(
            "The paper reevaluation requires exactly the six global-K root rows "
            f"in protocol order {DYSTS_MODEL_ROW_IDS}; got {labels}"
        )
    return rows


def _write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_root_specs_snapshot(path: Path, root_specs: Sequence[RootSpec]) -> None:
    rows = [
        {
            "label": spec.label,
            "display_name": spec.display_name,
            "model_family": spec.model_family,
            "root_dir": str(spec.root_dir),
        }
        for spec in root_specs
    ]
    _write_tsv(path, rows)


def _write_systems_snapshot(path: Path, systems: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{_raw_system_name(system)}\n" for system in systems]
    path.write_text("".join(lines))


def _build_rows(
    root_specs: Sequence[RootSpec],
    systems: Sequence[str],
    seeds: Sequence[int],
    output_tag: str,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    missing: List[Dict[str, object]] = []
    task_id = 0

    for root_spec in root_specs:
        for system in systems:
            system_dir = root_spec.root_dir / _system_dir_name(system)
            if not system_dir.exists():
                for seed in seeds:
                    missing.append(
                        {
                            "root_label": root_spec.label,
                            "root_display_name": root_spec.display_name,
                            "model_family": root_spec.model_family,
                            "system_key": system,
                            "seed": seed,
                            "reason": "missing_system_dir",
                            "path": str(system_dir),
                        }
                    )
                continue

            for seed in seeds:
                run_dir = _latest_seed_run(system_dir, seed)
                if run_dir is None:
                    missing.append(
                        {
                            "root_label": root_spec.label,
                            "root_display_name": root_spec.display_name,
                            "model_family": root_spec.model_family,
                            "system_key": system,
                            "seed": seed,
                            "reason": "missing_seed_run",
                            "path": str(system_dir),
                        }
                    )
                    continue

                dt_dir = next(
                    (parent for parent in run_dir.parents if parent.name.startswith("dt_")),
                    None,
                )
                reeval_dir = run_dir / f"reeval_{output_tag}"
                rows.append(
                    {
                        "task_id": task_id,
                        "root_label": root_spec.label,
                        "root_display_name": root_spec.display_name,
                        "model_family": root_spec.model_family,
                        "root_dir": str(root_spec.root_dir),
                        "system_key": system,
                        "system_slug": _system_slug(system),
                        "seed": seed,
                        "dt_dir": "" if dt_dir is None else str(dt_dir),
                        "run_dir": str(run_dir),
                        "checkpoint_path": str(run_dir / "checkpoint.pt"),
                        "config_path": str(run_dir / "config.json"),
                        "reeval_dir": str(reeval_dir),
                        "reeval_results_json": str(reeval_dir / "evaluation_results_checkpoint.json"),
                        "selected_rollout_artifacts": str(
                            reeval_dir
                            / "evaluation_checkpoint"
                            / system
                            / "selected_rollout_artifacts.pt"
                        ),
                    }
                )
                task_id += 1

    return rows, missing


def _summary_payload(
    *,
    rows: Sequence[Dict[str, object]],
    missing: Sequence[Dict[str, object]],
    root_specs: Sequence[RootSpec],
    systems: Sequence[str],
    seeds: Sequence[int],
    output_tag: str,
) -> Dict[str, object]:
    expected_rows = len(root_specs) * len(systems) * len(seeds)
    coverage_by_root: Dict[str, Dict[str, int]] = {}
    for root_spec in root_specs:
        actual = sum(1 for row in rows if row["root_label"] == root_spec.label)
        missing_count = sum(1 for item in missing if item["root_label"] == root_spec.label)
        coverage_by_root[root_spec.label] = {
            "actual_rows": actual,
            "expected_rows": len(systems) * len(seeds),
            "missing_rows": missing_count,
        }

    return {
        "output_tag": output_tag,
        "n_rows": len(rows),
        "expected_rows": expected_rows,
        "n_missing": len(missing),
        "root_specs": [asdict(spec) | {"root_dir": str(spec.root_dir)} for spec in root_specs],
        "systems": list(systems),
        "seeds": list(seeds),
        "coverage_by_root": coverage_by_root,
        "missing": list(missing),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build long-horizon Dysts reevaluation task tables.")
    parser.add_argument("--output_tsv", required=True, help="Path to the task TSV to write.")
    parser.add_argument("--output_summary_json", default=None, help="Optional JSON summary path.")
    parser.add_argument("--output_root_specs_tsv", default=None, help="Optional root-spec snapshot TSV.")
    parser.add_argument("--output_systems_file", default=None, help="Optional raw Dysts system list file.")
    parser.add_argument(
        "--root_specs_tsv",
        required=True,
        help="Explicit root-spec TSV generated by the retained Dysts training queue.",
    )
    parser.add_argument("--systems_csv", default=None, help="Optional comma-separated Dysts systems.")
    parser.add_argument("--seeds_csv", default=None, help="Optional comma-separated seeds.")
    parser.add_argument(
        "--output_tag",
        default="dysts_dt30_h100_to_h5000_paper",
        help="Output tag used to derive expected reevaluation subdirectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_specs = _read_root_specs(Path(args.root_specs_tsv))
    systems = (
        [_normalize_system(item) for item in _parse_csv_list(args.systems_csv)]
        if args.systems_csv
        else list(DYSTS_SYSTEMS)
    )
    unknown_systems = sorted(set(systems) - set(DYSTS_PAPER_PROTOCOL.system_keys))
    if unknown_systems:
        raise ValueError(
            f"Systems outside the retained Dysts paper roster: {unknown_systems}"
        )
    seeds = (
        [int(item) for item in _parse_csv_list(args.seeds_csv)]
        if args.seeds_csv
        else list(DEFAULT_SEEDS)
    )

    rows, missing = _build_rows(
        root_specs=root_specs,
        systems=systems,
        seeds=seeds,
        output_tag=str(args.output_tag),
    )
    _write_tsv(Path(args.output_tsv), rows)

    if args.output_summary_json:
        summary = _summary_payload(
            rows=rows,
            missing=missing,
            root_specs=root_specs,
            systems=systems,
            seeds=seeds,
            output_tag=str(args.output_tag),
        )
        Path(args.output_summary_json).write_text(json.dumps(summary, indent=2))

    if args.output_root_specs_tsv:
        _write_root_specs_snapshot(Path(args.output_root_specs_tsv), root_specs)
    if args.output_systems_file:
        _write_systems_snapshot(Path(args.output_systems_file), systems)

    print(
        f"Wrote {len(rows)} long-horizon Dysts reevaluation tasks "
        f"to {args.output_tsv} (missing={len(missing)})"
    )


if __name__ == "__main__":
    main()
