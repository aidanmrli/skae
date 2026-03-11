#!/usr/bin/env python3
"""Build task tables for a Hopfield basin-count sweep."""

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
)


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_csv_strings(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _model_row(variant: str) -> Dict[str, object]:
    if variant == "generic_sparse":
        return {
            "variant": "generic_sparse",
            "config_name": "generic_sparse",
            "res_coeff": 1.0,
            "reconst_coeff": 0.03,
            "pred_coeff": 1.0,
            "sparsity_coeff": 0.0005,
            "lista_alpha": "",
            "lista_num_loops": "",
            "lista_final_op": "",
            "k_structure": "",
            "k_block_size": "",
            "lr": "",
            "k_matrix_lr": "",
            "weight_decay": "",
        }

    if variant == "lista_dense_promoted_stage4":
        return {
            "variant": "lista_dense_promoted_stage4",
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

    if variant == "lista_blockdiag_targeted":
        return {
            "variant": "lista_blockdiag_targeted",
            "config_name": "lista_parity_generic_sparse",
            "res_coeff": 1.0,
            "reconst_coeff": 0.03,
            "pred_coeff": 1.0,
            "sparsity_coeff": 0.001,
            "lista_alpha": 0.15,
            "lista_num_loops": 1,
            "lista_final_op": "relu",
            "k_structure": "block_diagonal",
            "k_block_size": 16,
            "lr": "",
            "k_matrix_lr": "",
            "weight_decay": "",
        }

    raise KeyError(f"Unknown model variant '{variant}'")


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    basin_counts = _parse_csv_ints(args.num_basins_csv)
    seeds = _parse_csv_ints(args.seeds_csv)
    model_variants = _parse_csv_strings(args.model_variants_csv)

    rows: List[Dict[str, object]] = []
    task_id = 0
    for basin_count in basin_counts:
        system_key = f"hopfield_n{args.num_neurons}_p{basin_count:02d}"
        for variant in model_variants:
            model_row = _model_row(variant)
            for seed in seeds:
                rows.append(
                    {
                        "task_id": task_id,
                        "phase": args.phase_label,
                        "model_variant": model_row["variant"],
                        "config_name": model_row["config_name"],
                        "system_key": system_key,
                        "system_slug": system_key,
                        "system_group": "builtin_high_dim",
                        "env_name": "hopfield",
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
                        "kuramoto_num_oscillators": "",
                        "hopfield_num_neurons": args.num_neurons,
                        "hopfield_num_patterns": basin_count,
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
    parser = argparse.ArgumentParser(description="Build task tables for the Hopfield basin-count sweep.")
    parser.add_argument("--phase_label", default="hopfield_basin_sweep")
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_manifest_json", default=None)
    parser.add_argument("--num_basins_csv", default="8,10,12,14,16")
    parser.add_argument(
        "--model_variants_csv",
        default="generic_sparse,lista_dense_promoted_stage4,lista_blockdiag_targeted",
    )
    parser.add_argument("--seeds_csv", default="0,1,2")
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--env_dt", type=float, default=0.00625)
    parser.add_argument("--num_neurons", type=int, default=64)
    parser.add_argument("--eval_profile", default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        payload = {
            "system_env": "hopfield",
            "phase_label": args.phase_label,
            "num_neurons": args.num_neurons,
            "num_basins": _parse_csv_ints(args.num_basins_csv),
            "model_variants": _parse_csv_strings(args.model_variants_csv),
            "seeds": _parse_csv_ints(args.seeds_csv),
            "num_steps": args.num_steps,
            "env_dt": args.env_dt,
            "eval_profile": args.eval_profile,
        }
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(rows)} Hopfield basin-sweep tasks to {output_tsv}")


if __name__ == "__main__":
    main()
