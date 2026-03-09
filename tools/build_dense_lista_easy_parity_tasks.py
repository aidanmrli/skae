#!/usr/bin/env python3
"""Build the dense-LISTA easy-system parity stage-1 task table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from skae.benchmarks.paper_benchmark_manifest import (
    PAPER_BENCHMARK_BATCH_SIZE,
    PAPER_BENCHMARK_SEEDS,
    PAPER_BENCHMARK_SEQUENCE_LENGTH,
    PAPER_BENCHMARK_TARGET_SIZE,
    get_paper_benchmark_model,
    get_paper_benchmark_system,
    resolve_system_default_dt,
)


EASY_SYSTEM_KEYS: Sequence[str] = (
    "blended",
    "competitive_lv",
    "duffing",
    "dysts:Dadras",
    "dysts:Hadley",
    "dysts:LuChenCheng",
    "dysts:SanUmSrisuchinwong",
    "multiwell_gradient",
)

STAGE1_NUM_STEPS: Sequence[int] = (50_000, 100_000, 200_000)
STAGE1_LR_PAIRS: Sequence[Tuple[float, float]] = (
    (1e-4, 1e-5),
    (3e-4, 3e-5),
    (5e-5, 5e-6),
)
DEFAULT_WEIGHT_DECAY = 1e-4


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_steps(raw: str | None) -> List[int]:
    if raw is None:
        return list(STAGE1_NUM_STEPS)
    return [int(item) for item in _parse_csv_list(raw)]


def _parse_lr_pairs(raw: str | None) -> List[Tuple[float, float]]:
    if raw is None:
        return list(STAGE1_LR_PAIRS)

    pairs: List[Tuple[float, float]] = []
    for item in _parse_csv_list(raw):
        lr_raw, sep, k_lr_raw = item.partition(":")
        if not sep:
            raise ValueError(
                f"Invalid lr pair '{item}'. Expected format 'lr:k_matrix_lr'."
            )
        pairs.append((float(lr_raw), float(k_lr_raw)))
    return pairs


def _step_tag(num_steps: int) -> str:
    if num_steps % 1000 == 0:
        return f"{num_steps // 1000}k"
    return str(num_steps)


def _float_tag(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    exp_int = int(exponent)
    sign_tag = "m" if exp_int < 0 else "p"
    return f"{mantissa}e{sign_tag}{abs(exp_int)}"


def _arm_label(num_steps: int, lr: float, k_matrix_lr: float, weight_decay: float) -> str:
    return (
        f"lista_dense_ns{_step_tag(num_steps)}"
        f"_lr{_float_tag(lr)}"
        f"_klr{_float_tag(k_matrix_lr)}"
        f"_wd{_float_tag(weight_decay)}"
    )


def _build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    dense_spec = get_paper_benchmark_model("lista_dense")
    systems = _parse_csv_list(args.systems_csv) or list(EASY_SYSTEM_KEYS)
    seeds = [int(item) for item in (_parse_csv_list(args.seeds_csv) or [str(seed) for seed in PAPER_BENCHMARK_SEEDS])]
    num_steps_grid = _parse_steps(args.num_steps_csv)
    lr_pairs = _parse_lr_pairs(args.lr_pairs_csv)
    weight_decay = float(args.weight_decay)

    rows: List[Dict[str, object]] = []
    arm_specs: List[Dict[str, object]] = []
    task_id = 0

    for num_steps in num_steps_grid:
        for lr, k_matrix_lr in lr_pairs:
            arm_label = _arm_label(
                num_steps=num_steps,
                lr=lr,
                k_matrix_lr=k_matrix_lr,
                weight_decay=weight_decay,
            )
            arm_specs.append(
                {
                    "model_variant": arm_label,
                    "num_steps": num_steps,
                    "lr": lr,
                    "k_matrix_lr": k_matrix_lr,
                    "weight_decay": weight_decay,
                }
            )
            for system_key in systems:
                system_spec = get_paper_benchmark_system(system_key)
                env_dt = resolve_system_default_dt(system_key)
                for seed in seeds:
                    rows.append(
                        {
                            "task_id": task_id,
                            "phase": args.phase_label,
                            "model_variant": arm_label,
                            "config_name": dense_spec.config_name,
                            "system_key": system_spec.system_key,
                            "system_slug": system_spec.system_slug,
                            "system_group": system_spec.system_group,
                            "env_name": system_spec.env_name,
                            "seed": seed,
                            "num_steps": num_steps,
                            "batch_size": PAPER_BENCHMARK_BATCH_SIZE,
                            "target_size": PAPER_BENCHMARK_TARGET_SIZE,
                            "sequence_length": PAPER_BENCHMARK_SEQUENCE_LENGTH,
                            "res_coeff": dense_spec.res_coeff,
                            "reconst_coeff": dense_spec.reconst_coeff,
                            "pred_coeff": dense_spec.pred_coeff,
                            "sparsity_coeff": dense_spec.sparsity_coeff,
                            "lista_alpha": dense_spec.lista_alpha or "",
                            "lista_num_loops": dense_spec.lista_num_loops or "",
                            "lista_final_op": dense_spec.lista_final_op or "",
                            "k_structure": dense_spec.k_structure or "",
                            "k_block_size": dense_spec.k_block_size or "",
                            "lr": lr,
                            "k_matrix_lr": k_matrix_lr,
                            "weight_decay": weight_decay,
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

    return rows, arm_specs


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
    arm_specs: Sequence[Dict[str, object]],
    task_count: int,
) -> Dict[str, object]:
    return {
        "experiment": "dense_lista_easy_system_parity_stage1",
        "phase_label": phase_label,
        "systems": list(systems),
        "seeds": list(seeds),
        "arm_specs": list(arm_specs),
        "task_count": task_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build task tables for the dense-LISTA easy-system parity stage-1 sweep."
    )
    parser.add_argument("--output_tsv", required=True, help="Path to the task TSV to write.")
    parser.add_argument("--output_manifest_json", default=None, help="Optional JSON manifest path.")
    parser.add_argument("--phase_label", default="stage1", help="Phase label embedded in task rows.")
    parser.add_argument("--systems_csv", default=None, help="Optional comma-separated system keys.")
    parser.add_argument("--seeds_csv", default=None, help="Optional comma-separated seeds.")
    parser.add_argument("--num_steps_csv", default=None, help="Optional comma-separated training steps.")
    parser.add_argument(
        "--lr_pairs_csv",
        default=None,
        help="Optional comma-separated lr:k_matrix_lr pairs.",
    )
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay for all arms.")
    parser.add_argument("--eval_profile", default="full", help="Evaluation profile to embed in task rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, arm_specs = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        payload = _manifest_payload(
            phase_label=args.phase_label,
            systems=_parse_csv_list(args.systems_csv) or EASY_SYSTEM_KEYS,
            seeds=[int(item) for item in (_parse_csv_list(args.seeds_csv) or [str(seed) for seed in PAPER_BENCHMARK_SEEDS])],
            arm_specs=arm_specs,
            task_count=len(rows),
        )
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(rows)} dense-LISTA easy-system parity tasks to {output_tsv}")


if __name__ == "__main__":
    main()
