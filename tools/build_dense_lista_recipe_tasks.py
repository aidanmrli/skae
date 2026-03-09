#!/usr/bin/env python3
"""Build exact dense-LISTA recipe task tables for validation or reruns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from skae.benchmarks.paper_benchmark_manifest import (
    PAPER_BENCHMARK_BATCH_SIZE,
    PAPER_BENCHMARK_SEEDS,
    PAPER_BENCHMARK_SEQUENCE_LENGTH,
    PAPER_BENCHMARK_TARGET_SIZE,
    get_paper_benchmark_model,
    get_paper_benchmark_system,
    resolve_system_default_dt,
)


DEFAULT_SYSTEM_KEYS: Sequence[str] = (
    "blended",
    "competitive_lv",
    "duffing",
    "dysts:Dadras",
    "dysts:Hadley",
    "dysts:LuChenCheng",
    "dysts:SanUmSrisuchinwong",
    "multiwell_gradient",
)


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str | None, default: Sequence[int]) -> List[int]:
    if raw is None:
        return list(default)
    return [int(item) for item in _parse_csv_list(raw)]


def _read_dt_table(path: Path, value_column: str = "selected_dt") -> Dict[str, float]:
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


def _parse_recipe_specs(raw: str) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    for item in _parse_csv_list(raw):
        parts = item.split(":")
        if len(parts) != 8:
            raise ValueError(
                f"Invalid recipe spec '{item}'. Expected "
                "'label:num_steps:lr:k_matrix_lr:weight_decay:reconst_coeff:pred_coeff:sparsity_coeff'."
            )
        (
            label,
            num_steps_raw,
            lr_raw,
            k_lr_raw,
            weight_decay_raw,
            reconst_coeff_raw,
            pred_coeff_raw,
            sparsity_coeff_raw,
        ) = parts
        specs.append(
            {
                "model_variant": label,
                "num_steps": int(num_steps_raw),
                "lr": float(lr_raw),
                "k_matrix_lr": float(k_lr_raw),
                "weight_decay": float(weight_decay_raw),
                "reconst_coeff": float(reconst_coeff_raw),
                "pred_coeff": float(pred_coeff_raw),
                "sparsity_coeff": float(sparsity_coeff_raw),
            }
        )
    if not specs:
        raise ValueError("No recipe specs were provided.")
    return specs


def _build_rows(args: argparse.Namespace) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    dense_spec = get_paper_benchmark_model("lista_dense")
    systems = _parse_csv_list(args.systems_csv) or list(DEFAULT_SYSTEM_KEYS)
    seeds = _parse_int_csv(args.seeds_csv, [int(seed) for seed in PAPER_BENCHMARK_SEEDS])
    recipe_specs = _parse_recipe_specs(args.recipe_specs_csv)
    dt_map = _read_dt_table(Path(args.dt_table)) if args.dt_table else {}

    rows: List[Dict[str, object]] = []
    task_id = 0
    for recipe in recipe_specs:
        for system_key in systems:
            system_spec = get_paper_benchmark_system(system_key)
            env_dt = dt_map.get(system_key, resolve_system_default_dt(system_key))
            for seed in seeds:
                rows.append(
                    {
                        "task_id": task_id,
                        "phase": args.phase_label,
                        "model_variant": recipe["model_variant"],
                        "config_name": dense_spec.config_name,
                        "system_key": system_spec.system_key,
                        "system_slug": system_spec.system_slug,
                        "system_group": system_spec.system_group,
                        "env_name": system_spec.env_name,
                        "seed": seed,
                        "num_steps": recipe["num_steps"],
                        "batch_size": PAPER_BENCHMARK_BATCH_SIZE,
                        "target_size": PAPER_BENCHMARK_TARGET_SIZE,
                        "sequence_length": PAPER_BENCHMARK_SEQUENCE_LENGTH,
                        "res_coeff": dense_spec.res_coeff,
                        "reconst_coeff": recipe["reconst_coeff"],
                        "pred_coeff": recipe["pred_coeff"],
                        "sparsity_coeff": recipe["sparsity_coeff"],
                        "lista_alpha": dense_spec.lista_alpha or "",
                        "lista_num_loops": dense_spec.lista_num_loops or "",
                        "lista_final_op": dense_spec.lista_final_op or "",
                        "k_structure": dense_spec.k_structure or "",
                        "k_block_size": dense_spec.k_block_size or "",
                        "lr": recipe["lr"],
                        "k_matrix_lr": recipe["k_matrix_lr"],
                        "weight_decay": recipe["weight_decay"],
                        "env_dt": env_dt,
                        "eval_profile": args.eval_profile,
                        "standardize": 1 if system_spec.is_dysts else 0,
                        "dysts_native_cache": 1 if system_spec.is_dysts else 0,
                        "dysts_cache_profile": "full" if system_spec.is_dysts else "",
                        "dysts_cache_reuse": 1 if system_spec.is_dysts else 0,
                        "dysts_ic_noise_scale": 0.2 if system_spec.is_dysts else "",
                    }
                )
                task_id += 1
    return rows, recipe_specs


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


def _manifest_payload(
    *,
    phase_label: str,
    systems: Iterable[str],
    seeds: Iterable[int],
    recipe_specs: Sequence[Dict[str, object]],
    task_count: int,
) -> Dict[str, object]:
    return {
        "experiment": "dense_lista_recipe_validation",
        "phase_label": phase_label,
        "systems": list(systems),
        "seeds": list(seeds),
        "recipe_specs": list(recipe_specs),
        "task_count": task_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact dense-LISTA recipe validation tasks.")
    parser.add_argument("--output_tsv", required=True, help="Path to the task TSV to write.")
    parser.add_argument("--output_manifest_json", default=None, help="Optional JSON manifest path.")
    parser.add_argument("--phase_label", default="recipe_validation", help="Phase label embedded in task rows.")
    parser.add_argument("--systems_csv", default=None, help="Optional comma-separated system keys.")
    parser.add_argument("--seeds_csv", default=None, help="Optional comma-separated seeds.")
    parser.add_argument(
        "--recipe_specs_csv",
        required=True,
        help=(
            "Comma-separated recipe specs as "
            "'label:num_steps:lr:k_matrix_lr:weight_decay:reconst_coeff:pred_coeff:sparsity_coeff'."
        ),
    )
    parser.add_argument(
        "--dt_table",
        default=None,
        help="Optional TSV with per-system dt values. Uses the selected_dt column when provided.",
    )
    parser.add_argument("--eval_profile", default="full", help="Evaluation profile to embed in task rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, recipe_specs = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        payload = _manifest_payload(
            phase_label=args.phase_label,
            systems=_parse_csv_list(args.systems_csv) or DEFAULT_SYSTEM_KEYS,
            seeds=_parse_int_csv(args.seeds_csv, [int(seed) for seed in PAPER_BENCHMARK_SEEDS]),
            recipe_specs=recipe_specs,
            task_count=len(rows),
        )
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(rows)} dense-LISTA recipe tasks to {output_tsv}")
    print(f"Recipes: {len(recipe_specs)}")


if __name__ == "__main__":
    main()
