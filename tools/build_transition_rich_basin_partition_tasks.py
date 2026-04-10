#!/usr/bin/env python3
"""Build task tables for the fixed transition-rich basin-partition packet."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE,
    TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS,
    TRANSITION_RICH_BASIN_PARTITION_SEEDS,
    TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH,
    TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE,
    TransitionRichBasinPartitionModel,
    TransitionRichBasinPartitionSystem,
    get_transition_rich_basin_partition_model,
    get_transition_rich_basin_partition_system,
    resolve_transition_rich_default_dt,
    transition_rich_basin_partition_models,
    transition_rich_basin_partition_systems,
)


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str | None, default: Sequence[int]) -> List[int]:
    if raw is None:
        return list(default)
    return [int(item) for item in _parse_csv_list(raw)]


def _read_dt_table(path: Path, value_column: str) -> Dict[Tuple[str, str], float]:
    dt_map: Dict[Tuple[str, str], float] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            model_variant = str(row.get("model_variant", "")).strip()
            system_key = str(row.get("system_key", "")).strip()
            value = row.get(value_column)
            if not model_variant or not system_key or value in (None, ""):
                continue
            dt_map[(model_variant, system_key)] = float(value)
    return dt_map


def _selected_system_specs(args: argparse.Namespace) -> List[TransitionRichBasinPartitionSystem]:
    if args.systems_csv:
        return [
            get_transition_rich_basin_partition_system(system_key)
            for system_key in _parse_csv_list(args.systems_csv)
        ]
    return transition_rich_basin_partition_systems()


def _selected_model_specs(args: argparse.Namespace) -> List[TransitionRichBasinPartitionModel]:
    if args.model_variants_csv:
        return [
            get_transition_rich_basin_partition_model(variant)
            for variant in _parse_csv_list(args.model_variants_csv)
        ]
    return transition_rich_basin_partition_models()


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    systems = _selected_system_specs(args)
    models = _selected_model_specs(args)
    num_steps_override = int(args.num_steps_override) if args.num_steps_override is not None else None
    dt_map: Dict[Tuple[str, str], float] = {}
    if args.dt_table is not None:
        dt_map = _read_dt_table(Path(args.dt_table), value_column=args.dt_column)
        if not args.systems_csv:
            allowed_systems = {system_key for _, system_key in dt_map.keys()}
            systems = [spec for spec in systems if spec.system_key in allowed_systems]
        if not args.model_variants_csv:
            allowed_models = {model_variant for model_variant, _ in dt_map.keys()}
            models = [spec for spec in models if spec.variant in allowed_models]

    seeds = _parse_int_csv(
        args.seeds_csv,
        [int(seed) for seed in TRANSITION_RICH_BASIN_PARTITION_SEEDS],
    )

    rows: List[Dict[str, object]] = []
    task_id = 0
    for model in models:
        for system in systems:
            env_dt = dt_map.get((model.variant, system.system_key))
            if args.dt_table is not None and env_dt is None:
                continue
            if env_dt is None:
                env_dt = resolve_transition_rich_default_dt(system.system_key)
            target_size = (
                int(model.target_size)
                if model.target_size is not None
                else TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE
            )
            k_num_blocks: object = ""
            if model.use_basin_count_for_blocks:
                k_num_blocks = (
                    system.basin_count * model.basin_count_block_multiplier
                    + model.basin_count_block_offset
                )
            structured_num_basins: object = ""
            if model.use_basin_count_for_structured_num_basins:
                structured_num_basins = system.basin_count
            elif model.structured:
                structured_num_basins = (
                    model.structured_num_basins if model.structured_num_basins is not None else ""
                )
            structured_d_basin: object = (
                model.structured_d_basin if model.structured_d_basin is not None else ""
            )
            if model.structured_use_remaining_target:
                if model.structured_d_global is None:
                    raise ValueError(
                        f"{model.variant} requests structured remaining-target sizing without structured_d_global"
                    )
                if structured_num_basins in ("", 0):
                    raise ValueError(
                        f"{model.variant} requests structured remaining-target sizing without a basin count"
                    )
                remaining = target_size - int(model.structured_d_global)
                if remaining <= 0 or remaining % int(structured_num_basins) != 0:
                    raise ValueError(
                        f"{model.variant} cannot evenly split remaining target size {remaining} "
                        f"across {structured_num_basins} basins"
                    )
                structured_d_basin = remaining // int(structured_num_basins)
            soft_block_num_blocks: object = ""
            if model.use_basin_count_for_soft_block_num_blocks:
                soft_block_num_blocks = (
                    system.basin_count * model.soft_block_num_blocks_multiplier
                    + model.soft_block_num_blocks_offset
                )
            elif model.soft_block_num_blocks is not None:
                soft_block_num_blocks = model.soft_block_num_blocks
            for seed in seeds:
                rows.append(
                    {
                        "task_id": task_id,
                        "phase": args.phase_label,
                        "model_variant": model.variant,
                        "config_name": model.config_name,
                        "system_key": system.system_key,
                        "system_slug": system.system_slug,
                        "system_group": system.system_group,
                        "paper_role": system.paper_role,
                        "env_name": system.env_name,
                        "basin_count": system.basin_count,
                        "seed": seed,
                        "num_steps": (
                            num_steps_override if num_steps_override is not None else model.num_steps
                        ),
                        "batch_size": TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE,
                        "target_size": target_size,
                        "sequence_length": TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH,
                        "hard_init_oversample": (
                            str(model.hard_init_oversample).lower()
                            if model.hard_init_oversample is not None
                            else ""
                        ),
                        "hard_init_fraction": (
                            model.hard_init_fraction if model.hard_init_fraction is not None else ""
                        ),
                        "hard_init_pool_size": (
                            model.hard_init_pool_size if model.hard_init_pool_size is not None else ""
                        ),
                        "hard_init_num_candidates": (
                            model.hard_init_num_candidates
                            if model.hard_init_num_candidates is not None
                            else ""
                        ),
                        "hard_init_probe_steps": (
                            model.hard_init_probe_steps if model.hard_init_probe_steps is not None else ""
                        ),
                        "hard_init_num_perturbations": (
                            model.hard_init_num_perturbations
                            if model.hard_init_num_perturbations is not None
                            else ""
                        ),
                        "hard_init_perturb_scale": (
                            model.hard_init_perturb_scale
                            if model.hard_init_perturb_scale is not None
                            else ""
                        ),
                        "hard_init_transient_window": (
                            model.hard_init_transient_window
                            if model.hard_init_transient_window is not None
                            else ""
                        ),
                        "hard_init_transient_weight": (
                            model.hard_init_transient_weight
                            if model.hard_init_transient_weight is not None
                            else ""
                        ),
                        "hard_init_jitter_scale": (
                            model.hard_init_jitter_scale
                            if model.hard_init_jitter_scale is not None
                            else ""
                        ),
                        "res_coeff": model.res_coeff,
                        "reconst_coeff": model.reconst_coeff,
                        "pred_coeff": model.pred_coeff,
                        "sparsity_coeff": model.sparsity_coeff,
                        "lista_alpha": model.lista_alpha if model.lista_alpha is not None else "",
                        "lista_num_loops": (
                            model.lista_num_loops if model.lista_num_loops is not None else ""
                        ),
                        "lista_use_momentum": (
                            str(model.lista_use_momentum).lower()
                            if model.lista_use_momentum is not None
                            else ""
                        ),
                        "lista_momentum_beta": (
                            model.lista_momentum_beta
                            if model.lista_momentum_beta is not None
                            else ""
                        ),
                        "lista_linear_encoder": (
                            str(model.lista_linear_encoder).lower()
                            if model.lista_linear_encoder is not None
                            else ""
                        ),
                        "lista_final_op": model.lista_final_op or "",
                        "lista_precode_mode": model.lista_precode_mode or "",
                        "lista_precode_residual_scale": (
                            model.lista_precode_residual_scale
                            if model.lista_precode_residual_scale is not None
                            else ""
                        ),
                        "lista_adaptive_thresholds": (
                            str(model.lista_adaptive_thresholds).lower()
                            if model.lista_adaptive_thresholds is not None
                            else ""
                        ),
                        "lista_alpha_residual_coeff": (
                            model.lista_alpha_residual_coeff
                            if model.lista_alpha_residual_coeff is not None
                            else ""
                        ),
                        "lista_alpha_prior_coeff": (
                            model.lista_alpha_prior_coeff
                            if model.lista_alpha_prior_coeff is not None
                            else ""
                        ),
                        "lista_groupwise_thresholds": (
                            str(model.lista_groupwise_thresholds).lower()
                            if model.lista_groupwise_thresholds is not None
                            else ""
                        ),
                        "encoder_group_shrinkage": (
                            str(model.encoder_group_shrinkage).lower()
                            if model.encoder_group_shrinkage is not None
                            else ""
                        ),
                        "encoder_group_threshold_scale": (
                            model.encoder_group_threshold_scale
                            if model.encoder_group_threshold_scale is not None
                            else ""
                        ),
                        "encoder_topk_groups": (
                            model.encoder_topk_groups
                            if model.encoder_topk_groups is not None
                            else ""
                        ),
                        "decoder_coherence_weight": (
                            model.decoder_coherence_weight
                            if model.decoder_coherence_weight is not None
                            else ""
                        ),
                        "k_structure": model.k_structure,
                        "k_block_size": "",
                        "k_num_blocks": k_num_blocks,
                        "block_loss": 1 if model.block_loss else 0,
                        "block_one_block_loss": model.block_one_block_loss or "",
                        "block_one_block_weight": (
                            model.block_one_block_weight
                            if model.block_one_block_weight is not None
                            else ""
                        ),
                        "block_top1_margin": (
                            model.block_top1_margin if model.block_top1_margin is not None else ""
                        ),
                        "block_balance_loss": model.block_balance_loss or "",
                        "block_balance_weight": (
                            model.block_balance_weight
                            if model.block_balance_weight is not None
                            else ""
                        ),
                        "block_energy_norm": model.block_energy_norm or "",
                        "hyperlista_c_theta": (
                            model.hyperlista_c_theta if model.hyperlista_c_theta is not None else ""
                        ),
                        "hyperlista_c_beta": (
                            model.hyperlista_c_beta if model.hyperlista_c_beta is not None else ""
                        ),
                        "hyperlista_c_ss": (
                            model.hyperlista_c_ss if model.hyperlista_c_ss is not None else ""
                        ),
                        "hyperlista_use_ss": (
                            str(model.hyperlista_use_ss).lower()
                            if model.hyperlista_use_ss is not None
                            else ""
                        ),
                        "hyperlista_use_momentum": (
                            str(model.hyperlista_use_momentum).lower()
                            if model.hyperlista_use_momentum is not None
                            else ""
                        ),
                        "eval_use_dynamics_prior": (
                            str(model.eval_use_dynamics_prior).lower()
                        ),
                        "eval_event_trigger_proj_threshold": (
                            model.eval_event_trigger_proj_threshold
                            if model.eval_event_trigger_proj_threshold is not None
                            else ""
                        ),
                        "eval_event_trigger_ambiguity_threshold": (
                            model.eval_event_trigger_ambiguity_threshold
                            if model.eval_event_trigger_ambiguity_threshold is not None
                            else ""
                        ),
                        "eval_event_trigger_spillover_threshold": (
                            model.eval_event_trigger_spillover_threshold
                            if model.eval_event_trigger_spillover_threshold is not None
                            else ""
                        ),
                        "eval_event_trigger_support_margin_min_ratio": (
                            model.eval_event_trigger_support_margin_min_ratio
                            if model.eval_event_trigger_support_margin_min_ratio is not None
                            else ""
                        ),
                        "eval_event_trigger_support_threshold": (
                            model.eval_event_trigger_support_threshold
                            if model.eval_event_trigger_support_threshold is not None
                            else ""
                        ),
                        "eval_event_trigger_min_dwell": model.eval_event_trigger_min_dwell,
                        "eval_event_trigger_max_interval": model.eval_event_trigger_max_interval,
                        "structured": 1 if model.structured else 0,
                        "structured_d_global": (
                            model.structured_d_global if model.structured_d_global is not None else ""
                        ),
                        "structured_num_basins": structured_num_basins,
                        "structured_d_basin": structured_d_basin,
                        "lambda_global": (
                            model.lambda_global if model.lambda_global is not None else ""
                        ),
                        "lambda_local": (
                            model.lambda_local if model.lambda_local is not None else ""
                        ),
                        "lambda_exclusivity": (
                            model.lambda_exclusivity if model.lambda_exclusivity is not None else ""
                        ),
                        "lambda_sparsity": (
                            model.lambda_sparsity if model.lambda_sparsity is not None else ""
                        ),
                        "lambda_entropy": (
                            model.lambda_entropy if model.lambda_entropy is not None else ""
                        ),
                        "lambda_dominance": (
                            model.lambda_dominance if model.lambda_dominance is not None else ""
                        ),
                        "lambda_temporal": (
                            model.lambda_temporal if model.lambda_temporal is not None else ""
                        ),
                        "excl_warmup_steps": (
                            model.excl_warmup_steps if model.excl_warmup_steps is not None else ""
                        ),
                        "soft_block": 1 if model.soft_block else 0,
                        "soft_block_num_blocks": soft_block_num_blocks,
                        "soft_block_weight": (
                            model.soft_block_weight if model.soft_block_weight is not None else ""
                        ),
                        "soft_block_norm": model.soft_block_norm or "",
                        "lr": model.lr if model.lr is not None else "",
                        "k_matrix_lr": model.k_matrix_lr if model.k_matrix_lr is not None else "",
                        "weight_decay": model.weight_decay if model.weight_decay is not None else "",
                        "env_dt": env_dt,
                        "eval_profile": args.eval_profile,
                        "standardize": 0,
                        "dysts_native_cache": 0,
                        "dysts_cache_profile": "",
                        "dysts_cache_reuse": 0,
                        "dysts_ic_noise_scale": "",
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


def _manifest_payload(
    *,
    phase_label: str,
    systems: Sequence[TransitionRichBasinPartitionSystem],
    models: Sequence[TransitionRichBasinPartitionModel],
    seeds: Iterable[int],
    task_count: int,
    eval_profile: str,
    num_steps: int,
) -> Dict[str, object]:
    return {
        "experiment": "transition_rich_basin_partition",
        "phase_label": phase_label,
        "task_count": task_count,
        "seeds": list(seeds),
        "eval_profile": eval_profile,
        "packet_recipe": {
            "default_num_steps": num_steps,
            "batch_size": TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE,
            "target_size": TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE,
            "sequence_length": TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH,
        },
        "systems": [spec.system_key for spec in systems],
        "models": [spec.variant for spec in models],
        "selected_systems": [
            {
                **asdict(spec),
                "system_slug": spec.system_slug,
                "resolved_default_dt": resolve_transition_rich_default_dt(spec.system_key),
            }
            for spec in systems
        ],
        "selected_models": [asdict(spec) for spec in models],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build transition-rich basin-partition LISTA sweep tasks."
    )
    parser.add_argument("--output_tsv", required=True, help="Path to the task TSV to write.")
    parser.add_argument("--output_manifest_json", default=None, help="Optional JSON manifest path.")
    parser.add_argument(
        "--phase_label",
        default="transition_rich_basin_partition",
        help="Phase label embedded in task rows.",
    )
    parser.add_argument("--systems_csv", default=None, help="Optional comma-separated system keys.")
    parser.add_argument(
        "--model_variants_csv",
        default=None,
        help="Optional comma-separated model variants.",
    )
    parser.add_argument("--seeds_csv", default=None, help="Optional comma-separated seeds.")
    parser.add_argument("--eval_profile", default="full", help="Evaluation profile to embed in task rows.")
    parser.add_argument(
        "--num_steps_override",
        type=int,
        default=None,
        help="Optional global num_steps override for every emitted task row.",
    )
    parser.add_argument(
        "--dt_table",
        default=None,
        help="Optional TSV with per-arm dt values keyed by model_variant and system_key.",
    )
    parser.add_argument(
        "--dt_column",
        default="requested_dt",
        help="Column to read from --dt_table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        selected_system_keys = {str(row["system_key"]) for row in rows}
        selected_model_variants = {str(row["model_variant"]) for row in rows}
        systems = [
            spec for spec in _selected_system_specs(args)
            if spec.system_key in selected_system_keys
        ]
        models = [
            spec for spec in _selected_model_specs(args)
            if spec.variant in selected_model_variants
        ]
        payload = _manifest_payload(
            phase_label=args.phase_label,
            systems=systems,
            models=models,
            seeds=_parse_int_csv(
                args.seeds_csv,
                [int(seed) for seed in TRANSITION_RICH_BASIN_PARTITION_SEEDS],
            ),
            task_count=len(rows),
            eval_profile=args.eval_profile,
            num_steps=(
                int(args.num_steps_override)
                if args.num_steps_override is not None
                else TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS
            ),
        )
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(rows)} transition-rich basin-partition tasks to {output_tsv}")


if __name__ == "__main__":
    main()
