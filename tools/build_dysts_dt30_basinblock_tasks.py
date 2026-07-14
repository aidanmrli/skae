#!/usr/bin/env python3
"""Build task tables for the Dysts dt-x30 basin-block rerun."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from skae.benchmarks.paper_protocol import (
    DYSTS_MODEL_ROW_IDS,
    DYSTS_PAPER_PROTOCOL,
    DYSTS_PAPER_ROW_OVERRIDES,
)


LISTA_SB_PAPER_OVERRIDE = next(
    override
    for override in DYSTS_PAPER_ROW_OVERRIDES
    if override.variant == "lista_sb"
)


DYSTS_SYSTEM_SPECS: Dict[str, Dict[str, object]] = {
    "dysts:Chua": {
        "system_key": "dysts:Chua",
        "system_slug": "dysts_Chua",
        "system_group": "dysts_dt30",
        "env_name": "dysts:Chua",
        "base_dt": 0.0002847474579095888,
        "diagnostic_structure_count": 2,
        "structure_count_note": "Double-scroll/lobe count used as the block count.",
    },
    "dysts:Dadras": {
        "system_key": "dysts:Dadras",
        "system_slug": "dysts_Dadras",
        "system_group": "dysts_dt30",
        "env_name": "dysts:Dadras",
        "base_dt": 0.0006578296382730287,
        "diagnostic_structure_count": 2,
        "structure_count_note": "Bistable/multiple-attractor system; two blocks used for this diagnostic rerun.",
    },
    "dysts:DequanLi": {
        "system_key": "dysts:DequanLi",
        "system_slug": "dysts_DequanLi",
        "system_group": "dysts_dt30",
        "env_name": "dysts:DequanLi",
        "base_dt": 1.6763993998996084e-05,
        "diagnostic_structure_count": 3,
        "structure_count_note": "Three-scroll attractor count used as the block count.",
    },
    "dysts:Hadley": {
        "system_key": "dysts:Hadley",
        "system_slug": "dysts_Hadley",
        "system_group": "dysts_dt30",
        "env_name": "dysts:Hadley",
        "base_dt": 0.00029086847807974437,
        "diagnostic_structure_count": 3,
        "structure_count_note": "Multiple-equilibrium convective system; three blocks used for the lobe/equilibrium structure.",
    },
    "dysts:LuChenCheng": {
        "system_key": "dysts:LuChenCheng",
        "system_slug": "dysts_LuChenCheng",
        "system_group": "dysts_dt30",
        "env_name": "dysts:LuChenCheng",
        "base_dt": 0.00018469678279714685,
        "diagnostic_structure_count": 4,
        "structure_count_note": "Four-scroll attractor count used as the block count.",
    },
    "dysts:QiChen": {
        "system_key": "dysts:QiChen",
        "system_slug": "dysts_QiChen",
        "system_group": "dysts_dt30",
        "env_name": "dysts:QiChen",
        "base_dt": 7.837106184364728e-05,
        "diagnostic_structure_count": 2,
        "structure_count_note": "Double-wing/bistable attractor count used as the block count.",
    },
    "dysts:Sakarya": {
        "system_key": "dysts:Sakarya",
        "system_slug": "dysts_Sakarya",
        "system_group": "dysts_dt30",
        "env_name": "dysts:Sakarya",
        "base_dt": 0.0009970461743625946,
        "diagnostic_structure_count": 2,
        "structure_count_note": "Merging of two disjoint bistable attractors; two blocks used.",
    },
    "dysts:SanUmSrisuchinwong": {
        "system_key": "dysts:SanUmSrisuchinwong",
        "system_slug": "dysts_SanUmSrisuchinwong",
        "system_group": "dysts_dt30",
        "env_name": "dysts:SanUmSrisuchinwong",
        "base_dt": 0.0014933288881479711,
        "diagnostic_structure_count": 2,
        "structure_count_note": "Two-scroll attractor count used as the block count.",
    },
    "dysts:ShimizuMorioka": {
        "system_key": "dysts:ShimizuMorioka",
        "system_slug": "dysts_ShimizuMorioka",
        "system_group": "dysts_dt30",
        "env_name": "dysts:ShimizuMorioka",
        "base_dt": 0.002408001333556058,
        "diagnostic_structure_count": 2,
        "structure_count_note": "Symmetric/asymmetric Lorenz-like two-lobe structure; two blocks used.",
    },
    "dysts:WangSun": {
        "system_key": "dysts:WangSun",
        "system_slug": "dysts_WangSun",
        "system_group": "dysts_dt30",
        "env_name": "dysts:WangSun",
        "base_dt": 0.005392498749791912,
        "diagnostic_structure_count": 4,
        "structure_count_note": "Four-scroll attractor count used as the block count.",
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
    "base_dt",
    "dt_multiplier",
    "diagnostic_structure_count",
    "structure_count_note",
    "eval_profile",
    "standardize",
    "dysts_native_cache",
    "dysts_cache_profile",
    "dysts_cache_reuse",
    "dysts_ic_noise_scale",
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
    lr: float,
    k_matrix_lr: float,
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


def _recipes(
    args: argparse.Namespace,
    diagnostic_structure_count: int,
) -> Dict[str, Dict[str, object]]:
    dense = _base_recipe(
        config_name="generic_no_shrink",
        sparsity_coeff=args.dense_sparsity_coeff,
        k_structure="dense",
        lr=args.generic_lr,
        k_matrix_lr=args.generic_k_matrix_lr,
        weight_decay=args.weight_decay,
    )

    sparse_mlp = _base_recipe(
        config_name="generic_sparse",
        sparsity_coeff=args.sparsity_coeff,
        k_structure="dense",
        lr=args.generic_lr,
        k_matrix_lr=args.generic_k_matrix_lr,
        weight_decay=args.weight_decay,
    )

    sparse_mlp_bd = _base_recipe(
        config_name="generic_sparse",
        sparsity_coeff=args.sparsity_coeff,
        k_structure="block_diagonal",
        lr=args.generic_lr,
        k_matrix_lr=args.generic_k_matrix_lr,
        weight_decay=args.weight_decay,
    )
    sparse_mlp_bd["k_num_blocks"] = diagnostic_structure_count

    lista = _base_recipe(
        config_name="lista_parity_generic_sparse",
        sparsity_coeff=args.sparsity_coeff,
        k_structure="dense",
        lr=args.lista_lr,
        k_matrix_lr=args.lista_k_matrix_lr,
        weight_decay=args.weight_decay,
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
    lista_bd["k_num_blocks"] = diagnostic_structure_count

    lista_sb = dict(lista)
    lista_sb.update(
        {
            "lista_num_loops": LISTA_SB_PAPER_OVERRIDE.lista_num_loops,
            "lista_final_op": LISTA_SB_PAPER_OVERRIDE.lista_final_op,
            "soft_block": 1,
            "soft_block_num_blocks": diagnostic_structure_count,
            "soft_block_weight": args.soft_block_weight,
            "soft_block_norm": args.soft_block_norm,
        }
    )

    return {
        "lista": lista,
        "lista_bd": lista_bd,
        "lista_sb": lista_sb,
        "sparse_mlp": sparse_mlp,
        "sparse_mlp_bd": sparse_mlp_bd,
        "dense_mlp_tanh": dense,
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
        if system_key not in DYSTS_SYSTEM_SPECS:
            raise KeyError(
                f"Unknown system key '{system_key}'. Available: {sorted(DYSTS_SYSTEM_SPECS)}"
            )
        system = DYSTS_SYSTEM_SPECS[system_key]
        diagnostic_structure_count = int(system["diagnostic_structure_count"])
        recipes = _recipes(args, diagnostic_structure_count)
        env_dt = float(system["base_dt"]) * float(args.dt_multiplier)

        for model_variant in model_variants:
            if model_variant not in recipes:
                raise KeyError(
                    f"Unknown model variant '{model_variant}'. Available: {sorted(recipes)}"
                )
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
                        "env_dt": f"{env_dt:.17g}",
                        "base_dt": f"{float(system['base_dt']):.17g}",
                        "dt_multiplier": f"{float(args.dt_multiplier):.17g}",
                        "diagnostic_structure_count": diagnostic_structure_count,
                        "structure_count_note": system["structure_count_note"],
                        "eval_profile": args.eval_profile,
                        "standardize": 1,
                        "dysts_native_cache": 1,
                        "dysts_cache_profile": args.dysts_cache_profile,
                        "dysts_cache_reuse": 1,
                        "dysts_ic_noise_scale": args.dysts_ic_noise_scale,
                    }
                )
                row.update({field: recipe[field] for field in MODEL_FIELDS})
                rows.append({field: _row_value(row, field) for field in FIELDNAMES})
                task_id += 1
    return rows


def _write_tsv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_label", default="dysts_dt30_basinblock_p256_seq10_100k")
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_manifest_json", default=None)
    parser.add_argument(
        "--systems_csv",
        default=",".join(DYSTS_PAPER_PROTOCOL.system_keys),
    )
    parser.add_argument(
        "--model_variants_csv",
        default=",".join(DYSTS_MODEL_ROW_IDS),
    )
    parser.add_argument(
        "--seeds_csv",
        default=",".join(str(seed) for seed in DYSTS_PAPER_PROTOCOL.seeds),
    )
    parser.add_argument("--num_steps", type=int, default=DYSTS_PAPER_PROTOCOL.num_steps)
    parser.add_argument("--batch_size", type=int, default=DYSTS_PAPER_PROTOCOL.batch_size)
    parser.add_argument("--target_size", type=int, default=DYSTS_PAPER_PROTOCOL.target_size)
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=DYSTS_PAPER_PROTOCOL.sequence_length,
    )
    parser.add_argument(
        "--dt_multiplier",
        type=float,
        default=DYSTS_PAPER_PROTOCOL.dt_multiplier,
    )
    parser.add_argument("--sparsity_coeff", type=float, default=0.006)
    parser.add_argument("--dense_sparsity_coeff", type=float, default=0.0)
    parser.add_argument("--generic_lr", type=float, default=1e-4)
    parser.add_argument("--generic_k_matrix_lr", type=float, default=1e-5)
    parser.add_argument("--lista_alpha", type=float, default=0.15)
    parser.add_argument("--lista_lr", type=float, default=5e-5)
    parser.add_argument("--lista_k_matrix_lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--soft_block_weight", type=float, default=1e-4)
    parser.add_argument("--soft_block_norm", default="l1")
    parser.add_argument("--eval_profile", default="full")
    parser.add_argument("--dysts_cache_profile", default="full")
    parser.add_argument("--dysts_ic_noise_scale", type=float, default=0.2)
    return parser


def parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        system_keys = _parse_csv_strings(args.systems_csv)
        payload = {
            "protocol_id": DYSTS_PAPER_PROTOCOL.protocol_id,
            "phase_label": args.phase_label,
            "systems": {key: DYSTS_SYSTEM_SPECS[key] for key in system_keys},
            "model_variants": _parse_csv_strings(args.model_variants_csv),
            "seeds": _parse_csv_ints(args.seeds_csv),
            "num_steps": args.num_steps,
            "paper_row_overrides": [
                {
                    "variant": LISTA_SB_PAPER_OVERRIDE.variant,
                    "lista_num_loops": LISTA_SB_PAPER_OVERRIDE.lista_num_loops,
                    "lista_final_op": LISTA_SB_PAPER_OVERRIDE.lista_final_op,
                    "source_campaign_system_count": (
                        LISTA_SB_PAPER_OVERRIDE.source_campaign_system_count
                    ),
                    "retained_paper_system_count": (
                        LISTA_SB_PAPER_OVERRIDE.retained_paper_system_count
                    ),
                }
            ],
            "batch_size": args.batch_size,
            "target_size": args.target_size,
            "sequence_length": args.sequence_length,
            "dt_multiplier": args.dt_multiplier,
            "sparsity_coeff": args.sparsity_coeff,
            "dense_sparsity_coeff": args.dense_sparsity_coeff,
            "generic_lr": args.generic_lr,
            "generic_k_matrix_lr": args.generic_k_matrix_lr,
            "lista_alpha": args.lista_alpha,
            "lista_lr": args.lista_lr,
            "lista_k_matrix_lr": args.lista_k_matrix_lr,
            "weight_decay": args.weight_decay,
            "soft_block_weight": args.soft_block_weight,
            "soft_block_norm": args.soft_block_norm,
            "eval_profile": args.eval_profile,
            "dysts_cache_profile": args.dysts_cache_profile,
            "dysts_ic_noise_scale": args.dysts_ic_noise_scale,
            "task_count": len(rows),
            "notes": [
                "All recipes use matched latent size d_z=256.",
                "Dysts dt is multiplied by 30, and the long-horizon evaluation uses horizons reduced by roughly 30x.",
                "Block-diagonal K and soft-block recipes use the listed hand-set lobe, scroll, or equilibrium count.",
                "The diagnostic structure count only sizes K blocks; it is not a basin count and no training-time labels are used.",
                "dense_mlp_tanh uses generic_no_shrink: tanh hidden activation, no final ReLU, sparsity_coeff=0.",
                "LISTA-SB is a retained ablation row with two LISTA loops and sign-split output; LISTA and LISTA-BD use one loop and ReLU.",
                "The LISTA-SB source campaign contained 12 systems; paper summaries retain the frozen 10-system Dysts cohort.",
            ],
        }
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote {len(rows)} Dysts dt-x30 tasks to {output_tsv}")


if __name__ == "__main__":
    main()
