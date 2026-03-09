#!/usr/bin/env python3
"""Build task tables for the Kuramoto dimension scaling sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from skae.benchmarks.paper_benchmark_manifest import (
    PAPER_BENCHMARK_BATCH_SIZE,
    PAPER_BENCHMARK_SEQUENCE_LENGTH,
    PAPER_BENCHMARK_TARGET_SIZE,
    get_paper_benchmark_model,
    get_paper_benchmark_system,
)


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_csv_strings(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _model_row(variant: str) -> Dict[str, object]:
    if variant == "lista_dense_promoted":
        return {
            "variant": "lista_dense_promoted",
            "config_name": "lista_parity_generic_sparse",
            "res_coeff": 1.0,
            "reconst_coeff": 0.03,
            "pred_coeff": 1.0,
            "sparsity_coeff": 0.003,
            "lista_alpha": 0.15,
            "lista_num_loops": 1,
            "lista_final_op": "relu",
            "k_structure": "dense",
            "k_block_size": "",
            "lr": 5e-5,
            "k_matrix_lr": 5e-6,
            "weight_decay": 1e-4,
        }

    spec = get_paper_benchmark_model(variant)
    return {
        "variant": spec.variant,
        "config_name": spec.config_name,
        "res_coeff": spec.res_coeff,
        "reconst_coeff": spec.reconst_coeff,
        "pred_coeff": spec.pred_coeff,
        "sparsity_coeff": spec.sparsity_coeff,
        "lista_alpha": spec.lista_alpha or "",
        "lista_num_loops": spec.lista_num_loops or "",
        "lista_final_op": spec.lista_final_op or "",
        "k_structure": spec.k_structure or "",
        "k_block_size": spec.k_block_size or "",
        "lr": "",
        "k_matrix_lr": "",
        "weight_decay": "",
    }


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    system_spec = get_paper_benchmark_system("kuramoto")
    dimensions = _parse_csv_ints(args.dimensions_csv)
    seeds = _parse_csv_ints(args.seeds_csv)
    model_variants = _parse_csv_strings(args.model_variants_csv)

    rows: List[Dict[str, object]] = []
    task_id = 0
    for n_oscillators in dimensions:
        for variant in model_variants:
            model_row = _model_row(variant)
            for seed in seeds:
                rows.append(
                    {
                        "task_id": task_id,
                        "phase": args.phase_label,
                        "model_variant": model_row["variant"],
                        "config_name": model_row["config_name"],
                        "system_key": system_spec.system_key,
                        "system_slug": system_spec.system_slug,
                        "system_group": system_spec.system_group,
                        "env_name": system_spec.env_name,
                        "seed": seed,
                        "num_steps": args.num_steps,
                        "batch_size": PAPER_BENCHMARK_BATCH_SIZE,
                        "target_size": PAPER_BENCHMARK_TARGET_SIZE,
                        "sequence_length": PAPER_BENCHMARK_SEQUENCE_LENGTH,
                        "res_coeff": model_row["res_coeff"],
                        "reconst_coeff": model_row["reconst_coeff"],
                        "pred_coeff": model_row["pred_coeff"],
                        "sparsity_coeff": model_row["sparsity_coeff"],
                        "lista_alpha": model_row["lista_alpha"],
                        "lista_num_loops": model_row["lista_num_loops"],
                        "lista_final_op": model_row["lista_final_op"],
                        "k_structure": model_row["k_structure"],
                        "k_block_size": model_row["k_block_size"],
                        "lr": model_row["lr"],
                        "k_matrix_lr": model_row["k_matrix_lr"],
                        "weight_decay": model_row["weight_decay"],
                        "env_dt": args.env_dt,
                        "eval_profile": args.eval_profile,
                        "standardize": 0,
                        "dysts_native_cache": 0,
                        "dysts_cache_profile": "",
                        "dysts_cache_reuse": 0,
                        "dysts_ic_noise_scale": "",
                        "kuramoto_num_oscillators": n_oscillators,
                        "hopfield_num_neurons": "",
                        "hopfield_num_patterns": "",
                        "competitive_lv_num_species": "",
                    }
                )
                task_id += 1
    return rows


def _write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build task tables for the Kuramoto dimension sweep.")
    parser.add_argument("--phase_label", default="kuramoto_dimension_sweep_dt00625_200k")
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_manifest_json", default=None)
    parser.add_argument("--dimensions_csv", default="8,16,24,32,64")
    parser.add_argument("--model_variants_csv", default="generic_sparse,lista_dense_promoted,lista_blockdiag")
    parser.add_argument("--seeds_csv", default="0,1,2,3,4")
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--env_dt", type=float, default=0.00625)
    parser.add_argument("--eval_profile", default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        payload = {
            "system_key": "kuramoto",
            "phase_label": args.phase_label,
            "dimensions": _parse_csv_ints(args.dimensions_csv),
            "model_variants": _parse_csv_strings(args.model_variants_csv),
            "seeds": _parse_csv_ints(args.seeds_csv),
            "num_steps": args.num_steps,
            "env_dt": args.env_dt,
            "eval_profile": args.eval_profile,
        }
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(rows)} Kuramoto dimension-sweep tasks to {output_tsv}")


if __name__ == "__main__":
    main()
