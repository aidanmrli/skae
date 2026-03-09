#!/usr/bin/env python3
"""Build task tables for the paper benchmark experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from skae.benchmarks.paper_benchmark_manifest import (
    PAPER_BENCHMARK_BATCH_SIZE,
    PAPER_BENCHMARK_NUM_STEPS,
    PAPER_BENCHMARK_SEQUENCE_LENGTH,
    PAPER_BENCHMARK_SEEDS,
    PAPER_BENCHMARK_TARGET_SIZE,
    get_paper_benchmark_model,
    get_paper_benchmark_system,
    paper_benchmark_manifest_jsonable,
    paper_benchmark_models,
    paper_benchmark_systems,
    resolve_system_default_dt,
)


SMOKE_SYSTEM_KEYS: Sequence[str] = (
    "duffing",
    "multiwell_rotational",
    "kuramoto",
    "dysts:LorenzCoupled",
)


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _read_dt_table(path: Path, value_column: str) -> Dict[str, float]:
    dt_map: Dict[str, float] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            system_key = str(row.get("system_key", "")).strip()
            value = row.get(value_column)
            if not system_key or value in (None, ""):
                continue
            dt_map[system_key] = float(value)
    return dt_map


def _system_specs_for_phase(phase: str, system_keys: Sequence[str]) -> List:
    if system_keys:
        return [get_paper_benchmark_system(key) for key in system_keys]
    if phase == "smoke":
        return [get_paper_benchmark_system(key) for key in SMOKE_SYSTEM_KEYS]
    return paper_benchmark_systems()


def _model_specs_for_phase(phase: str, model_variants: Sequence[str]) -> List:
    if model_variants:
        return [get_paper_benchmark_model(variant) for variant in model_variants]
    if phase in {"anchor", "rescue"}:
        return [get_paper_benchmark_model("generic_sparse")]
    return paper_benchmark_models()


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    requested_system_keys = _parse_csv_list(args.systems_csv)
    system_specs = _system_specs_for_phase(args.phase, requested_system_keys)
    model_specs = _model_specs_for_phase(args.phase, _parse_csv_list(args.model_variants_csv))
    dt_map: Dict[str, float] = {}
    if args.dt_table is not None:
        column = "requested_dt" if args.phase == "rescue" else "selected_dt"
        dt_map = _read_dt_table(Path(args.dt_table), value_column=column)
        if not requested_system_keys:
            allowed_keys = set(dt_map.keys())
            system_specs = [spec for spec in system_specs if spec.system_key in allowed_keys]

    seeds = [int(value) for value in _parse_csv_list(args.seeds_csv)] if args.seeds_csv else list(PAPER_BENCHMARK_SEEDS)
    if args.phase == "smoke" and not args.seeds_csv:
        seeds = [0]

    num_steps = args.num_steps
    if num_steps is None:
        num_steps = 2_000 if args.phase == "smoke" else PAPER_BENCHMARK_NUM_STEPS

    eval_profile = args.eval_profile or ("smoke" if args.phase == "smoke" else "full")

    rows: List[Dict[str, object]] = []
    task_id = 0
    phase_label = str(args.phase_label)
    for system_spec in system_specs:
        env_dt = dt_map.get(system_spec.system_key, resolve_system_default_dt(system_spec.system_key))
        for model_spec in model_specs:
            for seed in seeds:
                rows.append(
                    {
                        "task_id": task_id,
                        "phase": phase_label,
                        "model_variant": model_spec.variant,
                        "config_name": model_spec.config_name,
                        "system_key": system_spec.system_key,
                        "system_slug": system_spec.system_slug,
                        "system_group": system_spec.system_group,
                        "env_name": system_spec.env_name,
                        "seed": seed,
                        "num_steps": num_steps,
                        "batch_size": PAPER_BENCHMARK_BATCH_SIZE,
                        "target_size": PAPER_BENCHMARK_TARGET_SIZE,
                        "sequence_length": PAPER_BENCHMARK_SEQUENCE_LENGTH,
                        "res_coeff": model_spec.res_coeff,
                        "reconst_coeff": model_spec.reconst_coeff,
                        "pred_coeff": model_spec.pred_coeff,
                        "sparsity_coeff": model_spec.sparsity_coeff,
                        "lista_alpha": model_spec.lista_alpha or "",
                        "lista_num_loops": model_spec.lista_num_loops or "",
                        "lista_final_op": model_spec.lista_final_op or "",
                        "k_structure": model_spec.k_structure or "",
                        "k_block_size": model_spec.k_block_size or "",
                        "env_dt": env_dt,
                        "eval_profile": eval_profile,
                        "standardize": 1 if system_spec.is_dysts else 0,
                        "dysts_native_cache": 1 if system_spec.is_dysts else 0,
                        "dysts_cache_profile": "full" if system_spec.is_dysts else "",
                        "dysts_cache_reuse": 1 if system_spec.is_dysts else 0,
                        "dysts_ic_noise_scale": 0.2 if system_spec.is_dysts else "",
                    }
                )
                task_id += 1
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build task tables for the paper benchmark.")
    parser.add_argument("--phase", choices=["smoke", "anchor", "rescue", "full"], required=True)
    parser.add_argument("--phase_label", default=None, help="Optional output phase label to embed in the task rows.")
    parser.add_argument("--output_tsv", required=True, help="Path to the task TSV to write.")
    parser.add_argument("--output_manifest_json", default=None, help="Optional JSON snapshot of the manifest.")
    parser.add_argument("--systems_csv", default=None, help="Optional comma-separated system keys.")
    parser.add_argument("--model_variants_csv", default=None, help="Optional comma-separated model variants.")
    parser.add_argument("--seeds_csv", default=None, help="Optional comma-separated seeds.")
    parser.add_argument("--num_steps", type=int, default=None, help="Override training steps.")
    parser.add_argument("--eval_profile", default=None, help="Override eval profile.")
    parser.add_argument(
        "--dt_table",
        default=None,
        help="Optional TSV with per-system dt values. Uses requested_dt for rescue and selected_dt for full.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase_label is None:
        args.phase_label = args.phase
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        Path(args.output_manifest_json).write_text(json.dumps(paper_benchmark_manifest_jsonable(), indent=2))

    print(f"Wrote {len(rows)} paper benchmark tasks to {output_tsv}")


if __name__ == "__main__":
    main()
