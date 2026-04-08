#!/usr/bin/env python3
"""Build task tables for the first Claude-catalog training packet."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from skae.benchmarks.claude_catalog_packet_manifest import (
    CLAUDE_CATALOG_PACKET_BATCH_SIZE,
    CLAUDE_CATALOG_PACKET_NUM_STEPS,
    CLAUDE_CATALOG_PACKET_SEEDS,
    CLAUDE_CATALOG_PACKET_SEQUENCE_LENGTH,
    CLAUDE_CATALOG_PACKET_TARGET_SIZE,
    ClaudeCatalogPacketModel,
    ClaudeCatalogPacketSystem,
    claude_catalog_packet_models,
    claude_catalog_packet_systems,
    get_claude_catalog_packet_model,
    get_claude_catalog_packet_system,
    resolve_claude_catalog_packet_dt,
)


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str | None, default: Sequence[int]) -> List[int]:
    if raw is None:
        return list(default)
    return [int(item) for item in _parse_csv_list(raw)]


def _selected_system_specs(args: argparse.Namespace) -> List[ClaudeCatalogPacketSystem]:
    if args.systems_csv:
        return [
            get_claude_catalog_packet_system(system_key)
            for system_key in _parse_csv_list(args.systems_csv)
        ]
    return claude_catalog_packet_systems(include_second_wave=args.include_second_wave)


def _selected_model_specs(args: argparse.Namespace) -> List[ClaudeCatalogPacketModel]:
    if args.model_variants_csv:
        return [
            get_claude_catalog_packet_model(variant)
            for variant in _parse_csv_list(args.model_variants_csv)
        ]
    return claude_catalog_packet_models()


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    systems = _selected_system_specs(args)
    models = _selected_model_specs(args)

    seeds = _parse_int_csv(args.seeds_csv, [int(seed) for seed in CLAUDE_CATALOG_PACKET_SEEDS])

    rows: List[Dict[str, object]] = []
    task_id = 0
    for model in models:
        for system in systems:
            env_dt = resolve_claude_catalog_packet_dt(system.system_key)
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
                        "env_name": system.env_name,
                        "seed": seed,
                        "num_steps": model.num_steps,
                        "batch_size": CLAUDE_CATALOG_PACKET_BATCH_SIZE,
                        "target_size": CLAUDE_CATALOG_PACKET_TARGET_SIZE,
                        "sequence_length": CLAUDE_CATALOG_PACKET_SEQUENCE_LENGTH,
                        "res_coeff": model.res_coeff,
                        "reconst_coeff": model.reconst_coeff,
                        "pred_coeff": model.pred_coeff,
                        "sparsity_coeff": model.sparsity_coeff,
                        "lista_alpha": model.lista_alpha or "",
                        "lista_num_loops": model.lista_num_loops or "",
                        "lista_final_op": model.lista_final_op or "",
                        "k_structure": model.k_structure or "",
                        "k_block_size": model.k_block_size or "",
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
    systems: Sequence[ClaudeCatalogPacketSystem],
    models: Sequence[ClaudeCatalogPacketModel],
    seeds: Iterable[int],
    task_count: int,
    include_second_wave: bool,
    eval_profile: str,
) -> Dict[str, object]:
    return {
        "experiment": "claude_catalog_packet",
        "phase_label": phase_label,
        "task_count": task_count,
        "seeds": list(seeds),
        "include_second_wave": include_second_wave,
        "eval_profile": eval_profile,
        "packet_recipe": {
            "default_num_steps": CLAUDE_CATALOG_PACKET_NUM_STEPS,
            "batch_size": CLAUDE_CATALOG_PACKET_BATCH_SIZE,
            "target_size": CLAUDE_CATALOG_PACKET_TARGET_SIZE,
            "sequence_length": CLAUDE_CATALOG_PACKET_SEQUENCE_LENGTH,
        },
        "systems": [spec.system_key for spec in systems],
        "models": [spec.variant for spec in models],
        "selected_systems": [
            {
                **asdict(spec),
                "system_slug": spec.system_slug,
                "resolved_default_dt": resolve_claude_catalog_packet_dt(spec.system_key),
            }
            for spec in systems
        ],
        "selected_models": [asdict(spec) for spec in models],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Claude-catalog packet tasks.")
    parser.add_argument("--output_tsv", required=True, help="Path to the task TSV to write.")
    parser.add_argument("--output_manifest_json", default=None, help="Optional JSON manifest path.")
    parser.add_argument("--phase_label", default="claude_catalog_packet", help="Phase label embedded in task rows.")
    parser.add_argument("--systems_csv", default=None, help="Optional comma-separated system keys.")
    parser.add_argument("--model_variants_csv", default=None, help="Optional comma-separated model variants.")
    parser.add_argument("--seeds_csv", default=None, help="Optional comma-separated seeds.")
    parser.add_argument("--include_second_wave", action="store_true", help="Include the second-wave three-system extension.")
    parser.add_argument("--eval_profile", default="full", help="Evaluation profile to embed in task rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        systems = _selected_system_specs(args)
        models = _selected_model_specs(args)
        payload = _manifest_payload(
            phase_label=args.phase_label,
            systems=systems,
            models=models,
            seeds=_parse_int_csv(args.seeds_csv, [int(seed) for seed in CLAUDE_CATALOG_PACKET_SEEDS]),
            task_count=len(rows),
            include_second_wave=args.include_second_wave,
            eval_profile=args.eval_profile,
        )
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(rows)} Claude catalog packet tasks to {output_tsv}")


if __name__ == "__main__":
    main()
