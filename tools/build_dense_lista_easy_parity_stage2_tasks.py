#!/usr/bin/env python3
"""Build coefficient-only Stage-2 task tables for dense-LISTA easy-system parity."""

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


HOLDOUT_SYSTEM_KEYS: Sequence[str] = ("competitive_lv", "duffing")
DEFAULT_BASE_ARM_SPECS: Sequence[Tuple[int, float, float, float]] = (
    (100_000, 5e-5, 5e-6, 1e-4),
    (200_000, 5e-5, 5e-6, 1e-4),
)
DEFAULT_SPARSITY_COEFFS: Sequence[float] = (0.003, 0.006, 0.012)
DEFAULT_RECONST_COEFFS: Sequence[float] = (0.01, 0.03, 0.1)
DEFAULT_PRED_COEFFS: Sequence[float] = (0.5, 1.0, 2.0)


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_float_csv(raw: str | None, default: Sequence[float]) -> List[float]:
    if raw is None:
        return list(default)
    return [float(item) for item in _parse_csv_list(raw)]


def _parse_int_csv(raw: str | None, default: Sequence[int]) -> List[int]:
    if raw is None:
        return list(default)
    return [int(item) for item in _parse_csv_list(raw)]


def _parse_base_arm_specs(raw: str | None) -> List[Tuple[int, float, float, float]]:
    if raw is None:
        return list(DEFAULT_BASE_ARM_SPECS)

    specs: List[Tuple[int, float, float, float]] = []
    for item in _parse_csv_list(raw):
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid base arm '{item}'. Expected "
                "'num_steps:lr:k_matrix_lr:weight_decay'."
            )
        num_steps_raw, lr_raw, k_lr_raw, weight_decay_raw = parts
        specs.append(
            (
                int(num_steps_raw),
                float(lr_raw),
                float(k_lr_raw),
                float(weight_decay_raw),
            )
        )
    return specs


def _step_tag(num_steps: int) -> str:
    if num_steps % 1000 == 0:
        return f"{num_steps // 1000}k"
    return str(num_steps)


def _float_tag(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    exp_int = int(exponent)
    sign_tag = "m" if exp_int < 0 else "p"
    return f"{mantissa}e{sign_tag}{abs(exp_int)}"


def _base_arm_label(
    num_steps: int, lr: float, k_matrix_lr: float, weight_decay: float
) -> str:
    return (
        f"lista_dense_ns{_step_tag(num_steps)}"
        f"_lr{_float_tag(lr)}"
        f"_klr{_float_tag(k_matrix_lr)}"
        f"_wd{_float_tag(weight_decay)}"
    )


def _coeff_variant_label(
    *,
    base_arm_label: str,
    reconst_coeff: float,
    pred_coeff: float,
    sparsity_coeff: float,
) -> str:
    return (
        f"{base_arm_label}"
        f"_rc{_float_tag(reconst_coeff)}"
        f"_pc{_float_tag(pred_coeff)}"
        f"_sc{_float_tag(sparsity_coeff)}"
    )


def _coefficient_variants(args: argparse.Namespace, dense_spec) -> List[Tuple[str, float, float, float]]:
    seen = set()
    variants: List[Tuple[str, float, float, float]] = []

    def add_variant(label: str, reconst_coeff: float, pred_coeff: float, sparsity_coeff: float) -> None:
        key = (reconst_coeff, pred_coeff, sparsity_coeff)
        if key in seen:
            return
        seen.add(key)
        variants.append((label, reconst_coeff, pred_coeff, sparsity_coeff))

    add_variant(
        "baseline",
        dense_spec.reconst_coeff,
        dense_spec.pred_coeff,
        dense_spec.sparsity_coeff,
    )

    for sparsity_coeff in _parse_float_csv(args.sparsity_coeffs_csv, DEFAULT_SPARSITY_COEFFS):
        add_variant(
            f"sp{_float_tag(sparsity_coeff)}",
            dense_spec.reconst_coeff,
            dense_spec.pred_coeff,
            sparsity_coeff,
        )
    for reconst_coeff in _parse_float_csv(args.reconst_coeffs_csv, DEFAULT_RECONST_COEFFS):
        add_variant(
            f"rc{_float_tag(reconst_coeff)}",
            reconst_coeff,
            dense_spec.pred_coeff,
            dense_spec.sparsity_coeff,
        )
    for pred_coeff in _parse_float_csv(args.pred_coeffs_csv, DEFAULT_PRED_COEFFS):
        add_variant(
            f"pc{_float_tag(pred_coeff)}",
            dense_spec.reconst_coeff,
            pred_coeff,
            dense_spec.sparsity_coeff,
        )

    return variants


def _build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    dense_spec = get_paper_benchmark_model("lista_dense")
    systems = _parse_csv_list(args.systems_csv) or list(HOLDOUT_SYSTEM_KEYS)
    seeds = _parse_int_csv(
        args.seeds_csv,
        [int(seed) for seed in PAPER_BENCHMARK_SEEDS],
    )
    base_arm_specs = _parse_base_arm_specs(args.base_arms_csv)
    coeff_variants = _coefficient_variants(args, dense_spec)

    rows: List[Dict[str, object]] = []
    arm_specs: List[Dict[str, object]] = []
    task_id = 0

    for num_steps, lr, k_matrix_lr, weight_decay in base_arm_specs:
        base_arm_label = _base_arm_label(
            num_steps=num_steps,
            lr=lr,
            k_matrix_lr=k_matrix_lr,
            weight_decay=weight_decay,
        )
        for coeff_label, reconst_coeff, pred_coeff, sparsity_coeff in coeff_variants:
            model_variant = _coeff_variant_label(
                base_arm_label=base_arm_label,
                reconst_coeff=reconst_coeff,
                pred_coeff=pred_coeff,
                sparsity_coeff=sparsity_coeff,
            )
            arm_specs.append(
                {
                    "model_variant": model_variant,
                    "base_arm_label": base_arm_label,
                    "coeff_label": coeff_label,
                    "num_steps": num_steps,
                    "lr": lr,
                    "k_matrix_lr": k_matrix_lr,
                    "weight_decay": weight_decay,
                    "res_coeff": dense_spec.res_coeff,
                    "reconst_coeff": reconst_coeff,
                    "pred_coeff": pred_coeff,
                    "sparsity_coeff": sparsity_coeff,
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
                            "model_variant": model_variant,
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
                            "reconst_coeff": reconst_coeff,
                            "pred_coeff": pred_coeff,
                            "sparsity_coeff": sparsity_coeff,
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
    base_arm_specs: Sequence[Tuple[int, float, float, float]],
    coeff_variants: Sequence[Tuple[str, float, float, float]],
    task_count: int,
) -> Dict[str, object]:
    return {
        "experiment": "dense_lista_easy_system_parity_stage2",
        "phase_label": phase_label,
        "systems": list(systems),
        "seeds": list(seeds),
        "base_arm_specs": [
            {
                "num_steps": num_steps,
                "lr": lr,
                "k_matrix_lr": k_matrix_lr,
                "weight_decay": weight_decay,
            }
            for num_steps, lr, k_matrix_lr, weight_decay in base_arm_specs
        ],
        "coefficient_variants": [
            {
                "label": label,
                "reconst_coeff": reconst_coeff,
                "pred_coeff": pred_coeff,
                "sparsity_coeff": sparsity_coeff,
            }
            for label, reconst_coeff, pred_coeff, sparsity_coeff in coeff_variants
        ],
        "task_count": task_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build coefficient-only Stage-2 tasks for dense-LISTA easy-system parity."
    )
    parser.add_argument("--output_tsv", required=True, help="Path to the task TSV to write.")
    parser.add_argument("--output_manifest_json", default=None, help="Optional JSON manifest path.")
    parser.add_argument("--phase_label", default="stage2", help="Phase label embedded in task rows.")
    parser.add_argument("--systems_csv", default=None, help="Optional comma-separated system keys.")
    parser.add_argument("--seeds_csv", default=None, help="Optional comma-separated seeds.")
    parser.add_argument(
        "--base_arms_csv",
        default=None,
        help="Optional comma-separated base arms as num_steps:lr:k_matrix_lr:weight_decay.",
    )
    parser.add_argument(
        "--sparsity_coeffs_csv",
        default=None,
        help="Optional comma-separated sparsity coefficients for the one-axis sweep.",
    )
    parser.add_argument(
        "--reconst_coeffs_csv",
        default=None,
        help="Optional comma-separated reconstruction coefficients for the one-axis sweep.",
    )
    parser.add_argument(
        "--pred_coeffs_csv",
        default=None,
        help="Optional comma-separated prediction coefficients for the one-axis sweep.",
    )
    parser.add_argument("--eval_profile", default="full", help="Evaluation profile to embed in task rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, arm_specs = _build_rows(args)
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)

    if args.output_manifest_json:
        dense_spec = get_paper_benchmark_model("lista_dense")
        coeff_variants = _coefficient_variants(args, dense_spec)
        payload = _manifest_payload(
            phase_label=args.phase_label,
            systems=_parse_csv_list(args.systems_csv) or HOLDOUT_SYSTEM_KEYS,
            seeds=_parse_int_csv(
                args.seeds_csv,
                [int(seed) for seed in PAPER_BENCHMARK_SEEDS],
            ),
            base_arm_specs=_parse_base_arm_specs(args.base_arms_csv),
            coeff_variants=coeff_variants,
            task_count=len(rows),
        )
        Path(args.output_manifest_json).write_text(json.dumps(payload, indent=2))

    print(f"Wrote {len(rows)} dense-LISTA easy-system Stage-2 tasks to {output_tsv}")
    print(f"Arms: {len(arm_specs)}")


if __name__ == "__main__":
    main()
