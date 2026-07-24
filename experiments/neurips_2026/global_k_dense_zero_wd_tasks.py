#!/usr/bin/env python3
"""Build frozen task tables for the prospective zero-weight-decay dense control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.neurips_2026.controlled import resolve_controlled_default_dt


DEFAULT_CARD = Path(__file__).with_name("global_k_dense_zero_wd_card.json")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_card(path: Path = DEFAULT_CARD) -> tuple[dict[str, Any], str]:
    card = json.loads(path.read_text())
    if card.get("status") != "preregistered_before_zero_weight_decay_training":
        raise RuntimeError("Dense-control card is not in its frozen preregistration state")
    sparse = card["frozen_sparse_reference"]
    sparse_card_path = Path(sparse["card"])
    if sha256_path(sparse_card_path) != sparse["card_sha256"]:
        raise RuntimeError("Frozen sparse card hash mismatch")
    dense_reference = card["disqualified_historical_dense_reference"]
    for key, hash_key in (
        ("rows_csv", "rows_csv_sha256"),
        ("representative_config", "representative_config_sha256"),
    ):
        if sha256_path(Path(dense_reference[key])) != dense_reference[hash_key]:
            raise RuntimeError(f"Historical dense reference hash mismatch: {key}")
    matched = card["matched_evaluation"]
    if sha256_path(Path(matched["sparse_rows_csv"])) != matched["sparse_rows_csv_sha256"]:
        raise RuntimeError("Frozen sparse roster CSV hash mismatch")
    return card, sha256_path(path)


def _row(
    card: dict[str, Any], *, task_id: int, system_key: str, seed: int,
    num_steps: int,
) -> dict[str, object]:
    train = card["training"]
    hard = train["hard_initial_condition_oversampling"]
    return {
        "task_id": task_id,
        "phase": train["phase"],
        "model_variant": train["model_variant"],
        "config_name": train["config_name"],
        "system_key": system_key,
        "system_slug": system_key.replace(":", "_"),
        "system_group": "throughput_smoke" if seed >= 9000 else "controlled",
        "paper_role": "dense_zero_weight_decay_specificity_control",
        "env_name": system_key,
        "basin_count": "",
        "seed": seed,
        "num_steps": num_steps,
        "batch_size": train["batch_size"],
        "target_size": train["latent_dim"],
        "sequence_length": train["sequence_length"],
        "hard_init_oversample": str(hard["enabled"]).lower(),
        "hard_init_fraction": hard["fraction"],
        "hard_init_pool_size": hard["pool_size"],
        "hard_init_num_candidates": hard["num_candidates"],
        "hard_init_probe_steps": hard["probe_steps"],
        "hard_init_num_perturbations": hard["num_perturbations"],
        "hard_init_perturb_scale": hard["perturb_scale"],
        "hard_init_transient_window": hard["transient_window"],
        "hard_init_transient_weight": hard["transient_weight"],
        "hard_init_jitter_scale": hard["jitter_scale"],
        "res_coeff": train["residual_coefficient"],
        "reconst_coeff": train["reconstruction_coefficient"],
        "pred_coeff": train["prediction_coefficient"],
        "sparsity_coeff": train["sparsity_coefficient"],
        "lista_alpha": "",
        "lista_num_loops": "",
        "lista_final_op": "",
        "k_structure": train["koopman_structure"],
        "k_block_size": "",
        "k_num_blocks": "",
        "block_loss": 0,
        "soft_block": 0,
        "lr": train["learning_rate"],
        "k_matrix_lr": train["koopman_learning_rate"],
        "weight_decay": train["weight_decay"],
        "eval_every": train["eval_every"],
        "eval_num_steps": train["eval_num_steps"],
        "env_dt": resolve_controlled_default_dt(system_key),
        "eval_profile": "smoke",
        "standardize": 0,
        "dysts_native_cache": 0,
        "dysts_cache_profile": "",
        "dysts_cache_reuse": 0,
        "dysts_ic_noise_scale": "",
    }


def build_rows(card: dict[str, Any], mode: str) -> list[dict[str, object]]:
    if mode == "smoke":
        smoke = card["gpu_smoke"]
        systems = [smoke["system"]]
        seeds = smoke["seeds"]
        num_steps = int(smoke["num_steps"])
    else:
        train = card["training"]
        systems = train["systems"]
        seeds = train["seeds"]
        num_steps = int(train["num_steps"])
    rows = [
        _row(card, task_id=index, system_key=system, seed=int(seed), num_steps=num_steps)
        for index, (system, seed) in enumerate(
            (system, seed) for system in systems for seed in seeds
        )
    ]
    expected = len(card["gpu_smoke"]["seeds"]) if mode == "smoke" else int(
        card["training"]["expected_run_count"]
    )
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} {mode} tasks, built {len(rows)}")
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output_tsv", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_manifest.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_manifest}")
    card, card_hash = load_card(args.card)
    rows = build_rows(card, args.mode)
    write_tsv(args.output_tsv, rows)
    manifest = {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "mode": args.mode,
        "card_path": str(args.card),
        "card_sha256": card_hash,
        "task_tsv": str(args.output_tsv),
        "task_tsv_sha256": sha256_path(args.output_tsv),
        "task_count": len(rows),
        "systems": sorted({str(row["system_key"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "outcomes_quarantined": args.mode == "smoke",
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
