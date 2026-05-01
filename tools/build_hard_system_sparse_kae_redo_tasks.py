#!/usr/bin/env python3
"""Build task tables for the hard-system sparse-KAE forecasting redo."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


SYSTEM_SPECS: Dict[str, Dict[str, object]] = {
    "competitive_lv_fixed_8basin_dt0p005": {
        "system_key": "competitive_lv_fixed_8basin_dt0p005",
        "system_slug": "competitive_lv_fixed_8basin",
        "system_group": "builtin_high_dim",
        "env_name": "competitive_lv",
        "env_dt": 0.005,
        "basin_count": 8,
        "competitive_lv_num_species": 15,
        "competitive_lv_interaction_scale": 0.83,
        "competitive_lv_system_seed": 0,
        "description": "Fixed 8-basin competitive Lotka-Volterra system used in the hard-system paper audit.",
    },
    "hopfield_n16_p16_dt0p00625": {
        "system_key": "hopfield_n16_p16_dt0p00625",
        "system_slug": "hopfield_n16_p16",
        "system_group": "builtin_high_dim",
        "env_name": "hopfield",
        "env_dt": 0.00625,
        "basin_count": 16,
        "hopfield_num_neurons": 16,
        "hopfield_num_patterns": 16,
        "description": "Continuous Hopfield N=16, P=16 memory-pattern benchmark.",
    },
    "kuramoto_n16_identical_dt0p00625": {
        "system_key": "kuramoto_n16_identical_dt0p00625",
        "system_slug": "kuramoto_n16_identical",
        "system_group": "builtin_high_dim",
        "env_name": "kuramoto",
        "env_dt": 0.00625,
        "basin_count": 5,
        "kuramoto_num_oscillators": 16,
        "kuramoto_topology": "ring",
        "kuramoto_omega_mode": "identical",
        "description": "Kuramoto ring N=16 with identical natural frequencies; basin labels are winding numbers.",
    },
}


MODEL_FIELDS = {
    "config_name",
    "res_coeff",
    "reconst_coeff",
    "pred_coeff",
    "sparsity_coeff",
    "lista_alpha",
    "lista_num_loops",
    "lista_final_op",
    "k_structure",
    "k_block_size",
    "k_num_blocks",
    "lr",
    "k_matrix_lr",
    "weight_decay",
    "soft_block",
    "soft_block_num_blocks",
    "soft_block_weight",
    "soft_block_norm",
}


FIELDNAMES = [
    "task_id",
    "phase",
    "model_variant",
    "config_name",
    "system_key",
    "system_slug",
    "system_group",
    "env_name",
    "seed",
    "num_steps",
    "batch_size",
    "target_size",
    "sequence_length",
    "res_coeff",
    "reconst_coeff",
    "pred_coeff",
    "sparsity_coeff",
    "lista_alpha",
    "lista_num_loops",
    "lista_final_op",
    "k_structure",
    "k_block_size",
    "k_num_blocks",
    "lr",
    "k_matrix_lr",
    "weight_decay",
    "env_dt",
    "eval_profile",
    "standardize",
    "dysts_native_cache",
    "dysts_cache_profile",
    "dysts_cache_reuse",
    "dysts_ic_noise_scale",
    "kuramoto_num_oscillators",
    "kuramoto_topology",
    "kuramoto_omega_mode",
    "kuramoto_omega_spread",
    "hopfield_num_neurons",
    "hopfield_num_patterns",
    "competitive_lv_num_species",
    "competitive_lv_interaction_scale",
    "competitive_lv_system_seed",
    "soft_block",
    "soft_block_num_blocks",
    "soft_block_weight",
    "soft_block_norm",
]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_csv_strings(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def _base_recipe(
    *,
    config_name: str,
    sparsity_coeff: float,
    k_structure: str = "dense",
    lr: float = 5e-5,
    k_matrix_lr: float = 5e-6,
    weight_decay: float = 1e-4,
) -> Dict[str, object]:
    return {
        "config_name": config_name,
        "res_coeff": 1.0,
        "reconst_coeff": 0.03,
        "pred_coeff": 1.0,
        "sparsity_coeff": sparsity_coeff,
        "lista_alpha": "",
        "lista_num_loops": "",
        "lista_final_op": "",
        "k_structure": k_structure,
        "k_block_size": "",
        "k_num_blocks": "",
        "lr": lr,
        "k_matrix_lr": k_matrix_lr,
        "weight_decay": weight_decay,
        "soft_block": 0,
        "soft_block_num_blocks": "",
        "soft_block_weight": "",
        "soft_block_norm": "",
    }


def _recipes(args: argparse.Namespace, basin_count: int) -> Dict[str, Dict[str, object]]:
    sparse_coeff = args.sparsity_coeff
    dense = _base_recipe(
        config_name="generic_no_shrink",
        sparsity_coeff=args.dense_sparsity_coeff,
        k_structure="dense",
        lr=args.generic_lr,
        k_matrix_lr=args.generic_k_matrix_lr,
    )

    sparse_mlp = _base_recipe(
        config_name="generic_sparse",
        sparsity_coeff=sparse_coeff,
        k_structure="dense",
        lr=args.generic_lr,
        k_matrix_lr=args.generic_k_matrix_lr,
    )

    sparse_mlp_bd = _base_recipe(
        config_name="generic_sparse",
        sparsity_coeff=sparse_coeff,
        k_structure="block_diagonal",
        lr=args.generic_lr,
        k_matrix_lr=args.generic_k_matrix_lr,
    )
    sparse_mlp_bd["k_num_blocks"] = basin_count

    lista = _base_recipe(
        config_name="lista_parity_generic_sparse",
        sparsity_coeff=sparse_coeff,
        k_structure="dense",
        lr=args.lista_lr,
        k_matrix_lr=args.lista_k_matrix_lr,
    )
    lista.update(
        {
            "lista_alpha": args.lista_alpha,
            "lista_num_loops": 1,
            "lista_final_op": "relu",
        }
    )

    lista_bd = dict(lista)
    lista_bd["k_structure"] = "block_diagonal"
    lista_bd["k_num_blocks"] = basin_count

    lista_sb = dict(lista)
    lista_sb.update(
        {
            "lista_num_loops": 2,
            "lista_final_op": "sign_split",
            "soft_block": 1,
            "soft_block_num_blocks": basin_count,
            "soft_block_weight": args.soft_block_weight,
            "soft_block_norm": args.soft_block_norm,
        }
    )

    return {
        "dense_mlp_tanh": dense,
        "sparse_mlp": sparse_mlp,
        "sparse_mlp_bd": sparse_mlp_bd,
        "lista": lista,
        "lista_bd": lista_bd,
        "lista_sb": lista_sb,
    }


def _row_value(row: Dict[str, object], field: str) -> object:
    value = row.get(field, "")
    return "" if value is None else value


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    system_keys = _parse_csv_strings(args.systems_csv)
    model_variants = _parse_csv_strings(args.model_variants_csv)
    seeds = _parse_csv_ints(args.seeds_csv)

    rows: List[Dict[str, object]] = []
    task_id = 0
    for system_key in system_keys:
        if system_key not in SYSTEM_SPECS:
            raise KeyError(f"Unknown system key '{system_key}'. Available: {sorted(SYSTEM_SPECS)}")
        system = SYSTEM_SPECS[system_key]
        recipes = _recipes(args, int(system["basin_count"]))

        for model_variant in model_variants:
            if model_variant not in recipes:
                raise KeyError(f"Unknown model variant '{model_variant}'. Available: {sorted(recipes)}")
            recipe = recipes[model_variant]
            for seed in seeds:
                row: Dict[str, object] = {field: "" for field in FIELDNAMES}
                row.update(
                    {
                        "task_id": task_id,
                        "phase": args.phase_label,
                        "model_variant": model_variant,
                        "system_key": system["system_key"],
                        "system_slug": system["system_slug"],
                        "system_group": system["system_group"],
                        "env_name": system["env_name"],
                        "seed": seed,
                        "num_steps": args.num_steps,
                        "batch_size": args.batch_size,
                        "target_size": args.target_size,
                        "sequence_length": args.sequence_length,
                        "env_dt": system["env_dt"],
                        "eval_profile": args.eval_profile,
                        "standardize": 0,
                        "dysts_native_cache": 0,
                        "dysts_cache_profile": "",
                        "dysts_cache_reuse": 0,
                        "dysts_ic_noise_scale": "",
                    }
                )
                row.update({field: recipe[field] for field in MODEL_FIELDS})
                for field, value in system.items():
                    if field in row:
                        row[field] = value
                rows.append({field: _row_value(row, field) for field in FIELDNAMES})
                task_id += 1
    return rows


def _write_tsv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_label", default="hard_system_sparse_kae_redo")
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_manifest_json", default=None)
    parser.add_argument(
        "--systems_csv",
        default="competitive_lv_fixed_8basin_dt0p005,hopfield_n16_p16_dt0p00625,kuramoto_n16_identical_dt0p00625",
    )
    parser.add_argument(
        "--model_variants_csv",
        default="dense_mlp_tanh,sparse_mlp,sparse_mlp_bd,lista,lista_bd,lista_sb",
    )
    parser.add_argument("--seeds_csv", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14")
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--target_size", type=int, default=1024)
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--sparsity_coeff", type=float, default=0.006)
    parser.add_argument("--dense_sparsity_coeff", type=float, default=0.0)
    parser.add_argument("--generic_lr", type=float, default=5e-5)
    parser.add_argument("--generic_k_matrix_lr", type=float, default=5e-6)
    parser.add_argument("--lista_alpha", type=float, default=0.15)
    parser.add_argument("--lista_lr", type=float, default=2.5e-5)
    parser.add_argument("--lista_k_matrix_lr", type=float, default=2.5e-6)
    parser.add_argument("--soft_block_weight", type=float, default=1e-4)
    parser.add_argument("--soft_block_norm", default="l1")
    parser.add_argument("--eval_profile", default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        system_keys = _parse_csv_strings(args.systems_csv)
        payload = {
            "phase_label": args.phase_label,
            "systems": {key: SYSTEM_SPECS[key] for key in system_keys},
            "model_variants": _parse_csv_strings(args.model_variants_csv),
            "seeds": _parse_csv_ints(args.seeds_csv),
            "num_steps": args.num_steps,
            "batch_size": args.batch_size,
            "target_size": args.target_size,
            "sequence_length": args.sequence_length,
            "sparsity_coeff": args.sparsity_coeff,
            "dense_sparsity_coeff": args.dense_sparsity_coeff,
            "generic_lr": args.generic_lr,
            "generic_k_matrix_lr": args.generic_k_matrix_lr,
            "lista_alpha": args.lista_alpha,
            "lista_lr": args.lista_lr,
            "lista_k_matrix_lr": args.lista_k_matrix_lr,
            "soft_block_weight": args.soft_block_weight,
            "soft_block_norm": args.soft_block_norm,
            "eval_profile": args.eval_profile,
            "task_count": len(rows),
            "notes": [
                "dense_mlp_tanh uses generic_no_shrink: tanh hidden activation, no final ReLU, sparsity_coeff=0.",
                "This redo uses sequence_length=8, 100k training steps, and half the previous learning rates.",
                "Block-diagonal K and soft-block recipes set the number of blocks to the benchmark basin count.",
                "LISTA+SB follows the current Dysts soft-block add-on convention: 2 LISTA loops and sign_split final op.",
            ],
        }
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {len(rows)} hard-system redo tasks to {output_tsv}")


if __name__ == "__main__":
    main()
