"""Build the frozen controlled-multibasin paper training matrix."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from experiments.neurips_2026.protocol import CONTROLLED_PAPER_PROTOCOL
from experiments.neurips_2026.controlled import (
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
    values: Dict[Tuple[str, str], float] = {}
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            variant = str(row.get("model_variant", "")).strip()
            system = str(row.get("system_key", "")).strip()
            value = row.get(value_column)
            if variant and system and value not in (None, ""):
                values[(variant, system)] = float(value)
    return values


def _selected_system_specs(
    args: argparse.Namespace,
) -> List[TransitionRichBasinPartitionSystem]:
    requested = _parse_csv_list(args.systems_csv)
    if not requested:
        return transition_rich_basin_partition_systems()
    return [get_transition_rich_basin_partition_system(key) for key in requested]


def _selected_model_specs(
    args: argparse.Namespace,
) -> List[TransitionRichBasinPartitionModel]:
    requested = _parse_csv_list(args.model_variants_csv)
    if not requested:
        return transition_rich_basin_partition_models()
    return [get_transition_rich_basin_partition_model(key) for key in requested]


def _optional(value: object | None) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _row(
    *,
    task_id: int,
    args: argparse.Namespace,
    model: TransitionRichBasinPartitionModel,
    system: TransitionRichBasinPartitionSystem,
    seed: int,
    env_dt: float,
) -> Dict[str, object]:
    k_num_blocks: object = (
        system.basin_count if model.use_basin_count_for_blocks else ""
    )
    soft_block_num_blocks: object = (
        system.basin_count
        if model.use_basin_count_for_soft_block_num_blocks
        else ""
    )
    num_steps = (
        int(args.num_steps_override)
        if args.num_steps_override is not None
        else model.num_steps
    )
    return {
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
        "num_steps": num_steps,
        "batch_size": CONTROLLED_PAPER_PROTOCOL.batch_size,
        "target_size": model.target_size,
        "sequence_length": CONTROLLED_PAPER_PROTOCOL.sequence_length,
        "hard_init_oversample": _optional(model.hard_init_oversample),
        "hard_init_fraction": model.hard_init_fraction,
        "hard_init_pool_size": model.hard_init_pool_size,
        "hard_init_num_candidates": model.hard_init_num_candidates,
        "hard_init_probe_steps": model.hard_init_probe_steps,
        "hard_init_num_perturbations": model.hard_init_num_perturbations,
        "hard_init_perturb_scale": model.hard_init_perturb_scale,
        "hard_init_transient_window": model.hard_init_transient_window,
        "hard_init_transient_weight": model.hard_init_transient_weight,
        "hard_init_jitter_scale": model.hard_init_jitter_scale,
        "res_coeff": model.res_coeff,
        "reconst_coeff": model.reconst_coeff,
        "pred_coeff": model.pred_coeff,
        "sparsity_coeff": model.sparsity_coeff,
        "lista_alpha": _optional(model.lista_alpha),
        "lista_num_loops": _optional(model.lista_num_loops),
        "lista_final_op": _optional(model.lista_final_op),
        "k_structure": model.k_structure,
        "k_block_size": "",
        "k_num_blocks": k_num_blocks,
        "soft_block": int(model.soft_block),
        "soft_block_num_blocks": soft_block_num_blocks,
        "soft_block_weight": _optional(model.soft_block_weight),
        "soft_block_norm": _optional(model.soft_block_norm),
        "lr": model.lr,
        "k_matrix_lr": model.k_matrix_lr,
        "weight_decay": model.weight_decay,
        "env_dt": env_dt,
        "eval_profile": args.eval_profile,
        "standardize": 0,
        "dysts_native_cache": 0,
        "dysts_cache_profile": "",
        "dysts_cache_reuse": 0,
        "dysts_ic_noise_scale": "",
    }


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    systems = _selected_system_specs(args)
    models = _selected_model_specs(args)
    dt_map: Dict[Tuple[str, str], float] = {}
    if args.dt_table is not None:
        dt_map = _read_dt_table(Path(args.dt_table), args.dt_column)
        if not args.systems_csv:
            allowed = {system for _, system in dt_map}
            systems = [spec for spec in systems if spec.system_key in allowed]
        if not args.model_variants_csv:
            allowed = {variant for variant, _ in dt_map}
            models = [spec for spec in models if spec.variant in allowed]

    seeds = _parse_int_csv(args.seeds_csv, CONTROLLED_PAPER_PROTOCOL.seeds)
    rows: List[Dict[str, object]] = []
    for model in models:
        for system in systems:
            env_dt = dt_map.get((model.variant, system.system_key))
            if args.dt_table is not None and env_dt is None:
                continue
            if env_dt is None:
                env_dt = resolve_transition_rich_default_dt(system.system_key)
            for seed in seeds:
                rows.append(
                    _row(
                        task_id=len(rows),
                        args=args,
                        model=model,
                        system=system,
                        seed=seed,
                        env_dt=env_dt,
                    )
                )
    return rows


def _write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _manifest_payload(
    *,
    phase_label: str,
    systems: Sequence[TransitionRichBasinPartitionSystem],
    models: Sequence[TransitionRichBasinPartitionModel],
    seeds: Iterable[int],
    task_count: int,
    eval_profile: str,
    num_steps: int,
    paper_protocol: bool = True,
) -> Dict[str, object]:
    del paper_protocol  # Retained for callers of the former exploratory builder.
    return {
        "protocol_id": CONTROLLED_PAPER_PROTOCOL.protocol_id,
        "experiment": "controlled_multibasin_paper",
        "phase_label": phase_label,
        "task_count": task_count,
        "seeds": list(seeds),
        "eval_profile": eval_profile,
        "packet_recipe": {
            "default_num_steps": num_steps,
            "batch_size": CONTROLLED_PAPER_PROTOCOL.batch_size,
            "target_size": CONTROLLED_PAPER_PROTOCOL.target_size,
            "sequence_length": CONTROLLED_PAPER_PROTOCOL.sequence_length,
        },
        "systems": [spec.system_key for spec in systems],
        "models": [spec.variant for spec in models],
        "selected_systems": [
            {
                **asdict(spec),
                "system_slug": spec.system_slug,
                "resolved_default_dt": resolve_transition_rich_default_dt(
                    spec.system_key
                ),
            }
            for spec in systems
        ],
        "selected_models": [asdict(spec) for spec in models],
        "count_semantics": (
            "Known basin counts size block-diagonal and soft-block diagnostics "
            "only; no basin labels are supplied during training."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_manifest_json", default=None)
    parser.add_argument(
        "--phase_label",
        default="transition_rich_basin_partition",
    )
    parser.add_argument("--systems_csv", default=None)
    parser.add_argument("--model_variants_csv", default=None)
    parser.add_argument("--seeds_csv", default=None)
    parser.add_argument(
        "--paper_protocol",
        action="store_true",
        help="Explicitly document use of the only retained protocol (default).",
    )
    parser.add_argument("--eval_profile", default="full")
    parser.add_argument("--num_steps_override", type=int, default=None)
    parser.add_argument("--dt_table", default=None)
    parser.add_argument("--dt_column", default="requested_dt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        system_keys = {str(row["system_key"]) for row in rows}
        model_keys = {str(row["model_variant"]) for row in rows}
        systems = [
            spec
            for spec in _selected_system_specs(args)
            if spec.system_key in system_keys
        ]
        models = [
            spec
            for spec in _selected_model_specs(args)
            if spec.variant in model_keys
        ]
        seeds = _parse_int_csv(args.seeds_csv, CONTROLLED_PAPER_PROTOCOL.seeds)
        num_steps = (
            int(args.num_steps_override)
            if args.num_steps_override is not None
            else CONTROLLED_PAPER_PROTOCOL.num_steps
        )
        payload = _manifest_payload(
            phase_label=args.phase_label,
            systems=systems,
            models=models,
            seeds=seeds,
            task_count=len(rows),
            eval_profile=args.eval_profile,
            num_steps=num_steps,
        )
        Path(args.output_manifest_json).write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    print(f"Wrote {len(rows)} controlled paper tasks to {output_tsv}")


if __name__ == "__main__":
    main()
