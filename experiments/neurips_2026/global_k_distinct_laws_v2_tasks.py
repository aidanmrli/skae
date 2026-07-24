#!/usr/bin/env python3
"""Build frozen mixed sparse/dense task tables for distinct-law V2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CARD = Path(__file__).with_name("global_k_distinct_laws_v2_card.json")
APPROVED_CARD_SHA256 = (
    "663fd03ddf9bfacabeef616f2a74f24998460d78b28413fdfeb42b012712f45b"
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_card(path: Path = DEFAULT_CARD) -> tuple[dict[str, Any], str]:
    card_hash = sha256_path(path)
    if card_hash != APPROVED_CARD_SHA256:
        raise RuntimeError(
            f"V2 card drift: approved={APPROVED_CARD_SHA256}, observed={card_hash}"
        )
    card = json.loads(path.read_text())
    if card.get("status") != (
        "preregistered_before_new_seed_training_or_checkpoint_evaluation"
    ):
        raise RuntimeError("V2 card is not in its approved preregistration state")
    if card.get("protocol_id") != (
        "global_k_distinct_laws_gated_local_linear_v2_new_seeds"
    ):
        raise RuntimeError("Unexpected V2 protocol ID")
    sparse = card["training_arms"]["sparse"]
    if sha256_path(Path(sparse["representative_frozen_config"])) != sparse[
        "representative_frozen_config_sha256"
    ]:
        raise RuntimeError("Frozen sparse representative config hash mismatch")
    dense = card["training_arms"]["dense"]
    if sha256_path(Path(dense["source_recipe_card"])) != dense[
        "source_recipe_card_sha256"
    ]:
        raise RuntimeError("Frozen dense source-recipe card hash mismatch")
    broad = card["direct_active_code_cloud_closure"]["broad_complementary_evidence"]
    for path_key, hash_key in (("card", "card_sha256"), ("decision", "decision_sha256")):
        if sha256_path(Path(broad[path_key])) != broad[hash_key]:
            raise RuntimeError(f"Broad closure source hash mismatch: {path_key}")
    for shard in broad["old_gated_local_linear_shards"]:
        if sha256_path(Path(shard["path"])) != shard["sha256"]:
            raise RuntimeError(f"Old closure shard hash mismatch: seed {shard['seed']}")
    return card, card_hash


def _common_row(
    card: dict[str, Any], *, task_id: int, arm: str, seed: int,
    num_steps: int, mode: str,
) -> dict[str, object]:
    recipe = card["training_arms"][arm]
    hard = card["training_arms"]["matched_hard_initial_condition_oversampling"]
    loss = recipe["loss_weights"]
    is_sparse = arm == "sparse"
    return {
        "task_id": task_id,
        "arm": arm,
        "phase": f"neurips_2026_global_k_distinct_laws_v2_{mode}",
        "model_variant": recipe["arm_id"],
        "config_name": recipe["config_name"],
        "system_key": "gated_local_linear",
        "system_slug": "gated_local_linear",
        "system_group": "throughput_smoke" if mode == "smoke" else "controlled",
        "paper_role": f"global_k_distinct_laws_v2_{arm}",
        "env_name": "gated_local_linear",
        "basin_count": "",
        "seed": seed,
        "num_steps": num_steps,
        "batch_size": recipe["batch_size"],
        "target_size": recipe["latent_dim"],
        "sequence_length": recipe["sequence_length"],
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
        "res_coeff": loss["residual"],
        "reconst_coeff": loss["reconstruction"],
        "pred_coeff": loss["prediction"],
        "sparsity_coeff": loss["sparsity"],
        "lista_alpha": recipe["encoder"]["alpha"] if is_sparse else "",
        "lista_num_loops": recipe["encoder"]["loops"] if is_sparse else "",
        "lista_linear_encoder": "false" if is_sparse else "",
        "lista_final_op": recipe["encoder"]["final_op"] if is_sparse else "",
        "encoder_group_shrinkage": "false",
        "encoder_topk_groups": 0,
        "decoder_coherence_weight": 0,
        "normalize_decoder_atoms": "false",
        "k_structure": recipe["koopman_structure"],
        "k_block_size": "",
        "k_num_blocks": "",
        "block_loss": 0,
        "soft_block": 0,
        "lr": recipe["learning_rate"],
        "k_matrix_lr": recipe["koopman_learning_rate"],
        "weight_decay": recipe["weight_decay"],
        "eval_every": recipe["eval_every"],
        "eval_num_steps": recipe["eval_num_steps"],
        "env_dt": card["benchmark"]["dt"],
        "eval_profile": "smoke",
        "standardize": 0,
        "dysts_native_cache": 0,
        "dysts_cache_profile": "",
        "dysts_cache_reuse": 0,
        "dysts_ic_noise_scale": "",
    }


def build_rows(card: dict[str, Any], mode: str) -> list[dict[str, object]]:
    if mode == "smoke":
        num_steps = int(card["gpu_utilization_and_schedule"]["smoke"]["num_steps"])
        seed_pairs = (
            [("sparse", int(seed)) for seed in card["new_seed_contract"]["smoke_seeds_sparse"]]
            + [("dense", int(seed)) for seed in card["new_seed_contract"]["smoke_seeds_dense"]]
        )
    elif mode == "full":
        seeds = [int(seed) for seed in card["new_seed_contract"]["scientific_seeds"]]
        seed_pairs = [("sparse", seed) for seed in seeds] + [
            ("dense", seed) for seed in seeds
        ]
        num_steps = int(card["training_arms"]["sparse"]["num_steps"])
        if num_steps != int(card["training_arms"]["dense"]["num_steps"]):
            raise RuntimeError("Sparse/dense scientific budgets differ")
    else:
        raise ValueError(mode)
    rows = [
        _common_row(
            card, task_id=index, arm=arm, seed=seed,
            num_steps=num_steps, mode=mode,
        )
        for index, (arm, seed) in enumerate(seed_pairs)
    ]
    contract = card["task_table_contract"]
    if mode == "smoke" and num_steps != int(contract["smoke_num_steps"]):
        raise RuntimeError("Smoke budget disagrees with the task-table contract")
    expected = int(contract[f"{mode}_task_count"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} {mode} tasks, built {len(rows)}")
    if [row["task_id"] for row in rows] != list(range(expected)):
        raise AssertionError("Task IDs are not contiguous")
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
        "arms": [str(row["arm"]) for row in rows],
        "seeds": [int(row["seed"]) for row in rows],
        "outcomes_quarantined": args.mode == "smoke",
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
