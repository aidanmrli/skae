#!/usr/bin/env python3
"""Build task tables for the spatialized multibasin PDE benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_SYSTEMS: Sequence[str] = (
    "cal_square_4",
    "cal_high_cross_3",
    "transition_routes_4",
    "var_l_shape_5",
    "cal_pentagon_5",
)
DEFAULT_MODELS: Sequence[str] = ("conv_lista", "conv_dense", "conv_sparse_mlp")
DEFAULT_SEEDS: Sequence[int] = (0, 1, 2)
PDE_CHANNELS = 2
DEFAULT_MIN_LATENT_STATE_RATIO = 4.0


def _parse_csv_strings(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str | None) -> List[int]:
    return [int(item) for item in _parse_csv_strings(raw)]


def _parse_csv_floats(raw: str | None) -> List[float]:
    return [float(item) for item in _parse_csv_strings(raw)]


def _tagify(raw: str) -> str:
    out = raw.replace(":", "_").replace("/", "_").replace(".", "p").replace("-", "_")
    return out


def _write_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _state_dim_for_grid(grid_size: int) -> int:
    return int(PDE_CHANNELS) * int(grid_size) * int(grid_size)


def _resolve_target_size(args: argparse.Namespace) -> int:
    state_dim = _state_dim_for_grid(int(args.grid_size))
    ratio = float(getattr(args, "min_latent_state_ratio", DEFAULT_MIN_LATENT_STATE_RATIO))
    min_target_size = math.ceil(ratio * state_dim)
    requested = int(args.target_size)
    if requested <= 0:
        return int(min_target_size)
    if requested < min_target_size:
        raise ValueError(
            "Spatialized PDE Koopman lifts must be overcomplete: "
            f"target_size={requested} is below min_latent_state_ratio={ratio:g} "
            f"* state_dim={state_dim}, so the minimum is {min_target_size}."
        )
    return requested


def _validate_requested_horizons(args: argparse.Namespace) -> None:
    horizons = _parse_csv_ints(str(args.eval_horizons))
    too_long = [horizon for horizon in horizons if horizon > int(args.trajectory_length)]
    if too_long:
        raise ValueError(
            "Requested eval_horizons exceed trajectory_length. "
            f"trajectory_length={int(args.trajectory_length)}, too_long={too_long}. "
            "Increase trajectory_length or remove those horizons so long-horizon "
            "scores are not clipped to the final stored observation."
        )


def _build_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    _validate_requested_horizons(args)
    systems = _parse_csv_strings(args.systems_csv) or list(DEFAULT_SYSTEMS)
    model_variants = _parse_csv_strings(args.model_variants_csv) or list(DEFAULT_MODELS)
    seeds = _parse_csv_ints(args.seeds_csv) or list(DEFAULT_SEEDS)
    lista_num_loops_values = _parse_csv_ints(getattr(args, "lista_num_loops_csv", None)) or [int(args.lista_num_loops)]
    lista_alphas = _parse_csv_floats(getattr(args, "lista_alpha_csv", None)) or [float(args.lista_alpha)]
    sparsity_coeffs = _parse_csv_floats(getattr(args, "sparsity_coeff_csv", None)) or [float(args.sparsity_coeff)]
    support_thresholds = _parse_csv_floats(getattr(args, "support_threshold_csv", None)) or [float(args.support_threshold)]
    family_jaccards = _parse_csv_floats(getattr(args, "family_jaccard_csv", None)) or [float(args.family_jaccard)]
    state_dim = _state_dim_for_grid(int(args.grid_size))
    target_size = _resolve_target_size(args)
    min_target_size = math.ceil(float(getattr(args, "min_latent_state_ratio", DEFAULT_MIN_LATENT_STATE_RATIO)) * state_dim)
    is_hparam_sweep = (
        len(lista_num_loops_values)
        * len(lista_alphas)
        * len(sparsity_coeffs)
        * len(support_thresholds)
        * len(family_jaccards)
    ) > 1
    rows: List[Dict[str, object]] = []
    task_id = 0
    for source_system in systems:
        system_slug = _tagify(source_system)
        for seed in seeds:
            for model_variant in model_variants:
                if model_variant not in DEFAULT_MODELS and not model_variant.startswith("flat_"):
                    raise ValueError(f"Unknown model_variant '{model_variant}'")
                model_slug = _tagify(model_variant)
                for lista_num_loops in lista_num_loops_values:
                    for lista_alpha in lista_alphas:
                        for sparsity_coeff in sparsity_coeffs:
                            for support_threshold in support_thresholds:
                                for family_jaccard in family_jaccards:
                                    activation_slug = _tagify(str(getattr(args, "conv_activation", "") or "default"))
                                    setting_slug = (
                                        "loops_{}_alpha_{}_sp_{}_tau_{}_jac_{}_act_{}".format(
                                            _tagify(f"{int(lista_num_loops)}"),
                                            _tagify(f"{lista_alpha:g}"),
                                            _tagify(f"{sparsity_coeff:g}"),
                                            _tagify(f"{support_threshold:g}"),
                                            _tagify(f"{family_jaccard:g}"),
                                            activation_slug,
                                        )
                                    )
                                    task_root = (
                                        Path(args.base_out)
                                        / "runs"
                                        / model_slug
                                        / system_slug
                                        / f"grid{args.grid_size}"
                                        / f"seed_{seed}"
                                    )
                                    dataset_base_out = (
                                        Path(args.dataset_base_out)
                                        if str(getattr(args, "dataset_base_out", "") or "").strip()
                                        else Path(args.base_out)
                                    )
                                    if bool(getattr(args, "share_dataset_by_seed", False)):
                                        dataset_root = (
                                            dataset_base_out
                                            / "datasets"
                                            / system_slug
                                            / f"grid{args.grid_size}"
                                            / f"seed_{seed}"
                                        )
                                    else:
                                        dataset_root = task_root
                                    if is_hparam_sweep:
                                        task_root = task_root / setting_slug
                                    dataset_path = dataset_root / "dataset.pt"
                                    run_dir = task_root / "model"
                                    eval_path = task_root / "evaluation.json"
                                    trainer = "flat" if model_variant.startswith("flat_") else "conv"
                                    config_name = "lista_parity_generic_sparse"
                                    if model_variant == "flat_dense":
                                        config_name = "generic_no_shrink"
                                    rows.append(
                                        {
                                            "task_id": task_id,
                                            "source_system": source_system,
                                            "seed": seed,
                                            "model_variant": model_variant,
                                            "trainer": trainer,
                                            "config_name": config_name,
                                            "setting_slug": setting_slug if is_hparam_sweep else "",
                                            "grid_size": args.grid_size,
                                            "diffusion": args.diffusion,
                                            "rk4_dt": args.rk4_dt,
                                            "substeps_per_observation": args.substeps_per_observation,
                                            "trajectory_length": args.trajectory_length,
                                            "label_extra_observations": args.label_extra_observations,
                                            "train_trajectories": args.train_trajectories,
                                            "val_trajectories": args.val_trajectories,
                                            "test_trajectories": args.test_trajectories,
                                            "laplacian_scaling": args.laplacian_scaling,
                                            "state_dim": state_dim,
                                            "target_size": target_size,
                                            "min_target_size": min_target_size,
                                            "latent_state_ratio": float(target_size) / float(state_dim),
                                            "hidden_channels": args.hidden_channels,
                                            "num_blocks": args.num_blocks,
                                            "conv_activation": getattr(args, "conv_activation", ""),
                                            "num_steps": args.num_steps,
                                            "batch_size": args.batch_size,
                                            "sequence_length": args.sequence_length,
                                            "train_observation_limit": args.train_observation_limit,
                                            "lista_num_loops": int(lista_num_loops),
                                            "lista_alpha": lista_alpha,
                                            "sparsity_coeff": sparsity_coeff,
                                            "support_threshold": support_threshold,
                                            "family_jaccard": family_jaccard,
                                            "max_validation_reps": args.max_validation_reps,
                                            "deep_threshold": args.deep_threshold,
                                            "eval_horizons": args.eval_horizons,
                                            "eval_horizon": args.eval_horizon,
                                            "dataset_path": str(dataset_path),
                                            "run_dir": str(run_dir),
                                            "eval_path": str(eval_path),
                                        }
                                    )
                                    task_id += 1
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_manifest_json", default=None)
    parser.add_argument("--base_out", required=True)
    parser.add_argument(
        "--dataset_base_out",
        default="",
        help=(
            "Optional base directory containing shared datasets. Run directories "
            "are still written under base_out."
        ),
    )
    parser.add_argument("--systems_csv", default=",".join(DEFAULT_SYSTEMS))
    parser.add_argument("--model_variants_csv", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds_csv", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--diffusion", type=float, default=0.01)
    parser.add_argument("--rk4_dt", type=float, default=0.005)
    parser.add_argument("--substeps_per_observation", type=int, default=10)
    parser.add_argument("--trajectory_length", type=int, default=24)
    parser.add_argument("--label_extra_observations", type=int, default=24)
    parser.add_argument("--train_trajectories", type=int, default=96)
    parser.add_argument("--val_trajectories", type=int, default=24)
    parser.add_argument("--test_trajectories", type=int, default=24)
    parser.add_argument("--laplacian_scaling", default="continuum", choices=["continuum", "graph"])
    parser.add_argument(
        "--target_size",
        type=int,
        default=0,
        help="Latent dimension d_z. Use 0 to auto-set to min_latent_state_ratio * state_dim.",
    )
    parser.add_argument("--min_latent_state_ratio", type=float, default=DEFAULT_MIN_LATENT_STATE_RATIO)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--conv_activation", default="")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=4)
    parser.add_argument(
        "--train_observation_limit",
        type=int,
        default=0,
        help=(
            "If >0, training samples sequence windows only from the first this many "
            "observation intervals while evaluation can use the full trajectory."
        ),
    )
    parser.add_argument("--lista_num_loops", type=int, default=2)
    parser.add_argument("--lista_num_loops_csv", default=None)
    parser.add_argument("--lista_alpha", type=float, default=1e-3)
    parser.add_argument("--lista_alpha_csv", default=None)
    parser.add_argument("--sparsity_coeff", type=float, default=0.0)
    parser.add_argument("--sparsity_coeff_csv", default=None)
    parser.add_argument("--support_threshold", type=float, default=1e-4)
    parser.add_argument("--support_threshold_csv", default=None)
    parser.add_argument("--family_jaccard", type=float, default=0.7)
    parser.add_argument("--family_jaccard_csv", default=None)
    parser.add_argument("--max_validation_reps", type=int, default=256)
    parser.add_argument("--deep_threshold", type=float, default=0.7)
    parser.add_argument("--eval_horizons", default="1,4,8,12")
    parser.add_argument("--eval_horizon", type=int, default=8)
    parser.add_argument(
        "--share_dataset_by_seed",
        action="store_true",
        help="Store one generated dataset per system/grid/seed and reuse it across model variants.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(args)
    if rows:
        args.target_size = int(rows[0]["target_size"])
    output_tsv = Path(args.output_tsv)
    _write_tsv(output_tsv, rows)
    if args.output_manifest_json:
        Path(args.output_manifest_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_manifest_json).write_text(
            json.dumps(
                {
                    "num_tasks": len(rows),
                    "systems": _parse_csv_strings(args.systems_csv),
                    "model_variants": _parse_csv_strings(args.model_variants_csv),
                    "seeds": _parse_csv_ints(args.seeds_csv),
                    "config": vars(args),
                    "state_dim": _state_dim_for_grid(int(args.grid_size)),
                    "min_target_size": int(rows[0]["min_target_size"]) if rows else None,
                    "latent_state_ratio": float(rows[0]["latent_state_ratio"]) if rows else None,
                    "label_policy": "Basin labels and attractor centers are stored for evaluation only.",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    print(f"Wrote {len(rows)} spatialized PDE tasks to {output_tsv}")


if __name__ == "__main__":
    main()
