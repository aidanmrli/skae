#!/usr/bin/env python3
"""Build task tables for the fixed transition-rich basin-partition LISTA sweep."""

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
            k_num_blocks = system.basin_count if model.use_basin_count_for_blocks else ""
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
                        "num_steps": model.num_steps,
                        "batch_size": TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE,
                        "target_size": TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE,
                        "sequence_length": TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH,
                        "res_coeff": model.res_coeff,
                        "reconst_coeff": model.reconst_coeff,
                        "pred_coeff": model.pred_coeff,
                        "sparsity_coeff": model.sparsity_coeff,
                        "lista_alpha": model.lista_alpha,
                        "lista_num_loops": model.lista_num_loops,
                        "lista_final_op": model.lista_final_op,
                        "k_structure": model.k_structure,
                        "k_block_size": "",
                        "k_num_blocks": k_num_blocks,
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
) -> Dict[str, object]:
    return {
        "experiment": "transition_rich_basin_partition",
        "phase_label": phase_label,
        "task_count": task_count,
        "seeds": list(seeds),
        "eval_profile": eval_profile,
        "packet_recipe": {
            "default_num_steps": TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS,
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
        )
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(rows)} transition-rich basin-partition tasks to {output_tsv}")


if __name__ == "__main__":
    main()
