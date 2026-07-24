#!/usr/bin/env python3
"""Fail-closed checkpoint/config audit for distinct-law V2."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    load_card,
    sha256_path,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_math import (
    decoder_linearity_diagnostics,
)
from experiments.neurips_2026.global_k_support_invariance import (
    assert_sign_split_layout,
)
from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import make_env
from skae.model import make_model
from skae.training.runner import build_optimizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrainedRun:
    task_id: int
    arm: str
    seed: int
    system_key: str
    run_dir: Path
    attempt_count: int
    incomplete_attempt_count: int


def _tagify(value: str) -> str:
    return value.replace("-", "m").replace(".", "p")


def load_task_rows(path: Path, card: dict[str, Any]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = int(card["task_table_contract"]["full_task_count"])
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} scientific tasks, found {len(rows)}")
    if [int(row["task_id"]) for row in rows] != list(range(expected)):
        raise RuntimeError("Scientific task IDs/order drifted")
    expected_seeds = [int(seed) for seed in card["new_seed_contract"]["scientific_seeds"]]
    observed = [(row["arm"], int(row["seed"])) for row in rows]
    roster = [("sparse", seed) for seed in expected_seeds] + [
        ("dense", seed) for seed in expected_seeds
    ]
    if observed != roster:
        raise RuntimeError(f"Scientific task roster drift: {observed}")
    if any(row["system_key"] != "gated_local_linear" for row in rows):
        raise RuntimeError("Scientific task system drift")
    return rows


def discover_trained_runs(task_tsv: Path, base_out: Path, card: dict[str, Any]) -> list[TrainedRun]:
    specs: list[TrainedRun] = []
    for row in load_task_rows(task_tsv, card):
        parent = (
            base_out
            / row["phase"]
            / row["model_variant"]
            / row["system_slug"]
            / f"dt_{_tagify(row['env_dt'])}"
            / f"seed_{row['seed']}"
        )
        attempts = sorted(
            path for path in parent.glob("20*")
            if path.is_dir() and (path / "checkpoint.pt").is_file()
        )
        completed = [
            path for path in attempts
            if (path / "evaluation_summary.json").is_file()
        ]
        if len(completed) != 1:
            raise RuntimeError(
                f"Expected one completed task under {parent}; "
                f"completed={completed}, attempts={attempts}"
            )
        specs.append(
            TrainedRun(
                task_id=int(row["task_id"]),
                arm=row["arm"],
                seed=int(row["seed"]),
                system_key=row["system_key"],
                run_dir=completed[0],
                attempt_count=len(attempts),
                incomplete_attempt_count=len(attempts) - 1,
            )
        )
    return specs


def load_trained_model(spec: TrainedRun, device: str = "cpu"):
    checkpoint_path = spec.run_dir / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = Config.from_dict(checkpoint["config"])
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    load_model_state_dict_compat(model, checkpoint["model_state_dict"])
    return cfg, env, model.to(device).eval(), checkpoint, checkpoint_path


def _optimizer_weight_decays(model: nn.Module, cfg: Config) -> list[float]:
    return [
        float(group.get("weight_decay", 0.0))
        for group in build_optimizer(model, cfg).param_groups
    ]


def trainable_parameter_counts(model: nn.Module) -> dict[str, int]:
    counts = {"koopman": 0, "encoder": 0, "decoder": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        size = int(parameter.numel())
        if "kmat" in name or name.startswith("K_"):
            category = "koopman"
        elif name.startswith("encoder."):
            category = "encoder"
        elif name == "dict" or name.startswith("decoder"):
            category = "decoder"
        else:
            category = "other"
        counts[category] += size
    counts["total"] = sum(counts.values())
    return counts


def _schema_changes(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        changes = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left:
                value = right[key]
                rendered = (
                    str(value).lower() if isinstance(value, bool)
                    else str(value)
                )
                changes.append(f"insert {path}={rendered}")
            elif key not in right:
                suffix = (
                    " with ENABLED=false"
                    if isinstance(left[key], dict) and left[key].get("ENABLED") is False
                    else ""
                )
                changes.append(f"remove {path}{suffix}")
            else:
                changes.extend(_schema_changes(left[key], right[key], path))
        return changes
    return [] if left == right else [f"change {prefix}: {left!r}->{right!r}"]


def _audit_shared(
    cfg: Config, model: nn.Module, checkpoint: dict[str, Any],
    recipe: dict[str, Any], card: dict[str, Any], spec: TrainedRun,
) -> dict[str, bool]:
    hard = card["training_arms"]["matched_hard_initial_condition_oversampling"]
    loss = recipe["loss_weights"]
    checkpoint_step = int(checkpoint.get("step", -1))
    eval_every = int(recipe["eval_every"])
    serialized = checkpoint.get("optimizer_state_dict")
    return {
        "seed": int(cfg.SEED) == spec.seed,
        "system": str(cfg.ENV.ENV_NAME) == "gated_local_linear",
        "dt": float(cfg.ENV.GATED_LOCAL_LINEAR.DT) == float(card["benchmark"]["dt"]),
        "model_name": str(cfg.MODEL.MODEL_NAME) == recipe["model_name"],
        "latent_dim": int(cfg.MODEL.TARGET_SIZE) == int(recipe["latent_dim"]),
        "identity_norm": str(cfg.MODEL.NORM_FN).lower() == "id",
        "dense_k": str(cfg.MODEL.K_STRUCTURE).lower() == "dense",
        "linear_decoder_config": list(cfg.MODEL.DECODER.LAYERS) == [],
        "decoder_bias_disabled": not bool(cfg.MODEL.DECODER.USE_BIAS)
        and not bool(cfg.MODEL.DECODER.AFFINE_BIAS),
        "decoder_normalize_atoms_config_false": not bool(
            cfg.MODEL.DECODER.NORMALIZE_ATOMS
        ),
        "homogeneous_off": not bool(cfg.MODEL.USE_HOMOGENEOUS),
        "block_loss_off": not bool(cfg.MODEL.BLOCK_LOSS.ENABLED),
        "soft_block_off": not bool(cfg.MODEL.SOFT_BLOCK.ENABLED),
        "coherence_off": float(cfg.MODEL.DECODER_COHERENCE_WEIGHT) == 0.0,
        "budget": int(cfg.TRAIN.NUM_STEPS) == int(recipe["num_steps"]),
        "batch": int(cfg.TRAIN.BATCH_SIZE) == int(recipe["batch_size"]),
        "data_size": int(cfg.TRAIN.DATA_SIZE) == int(recipe["data_size"]),
        "sequence": int(cfg.TRAIN.SEQUENCE_LENGTH) == int(recipe["sequence_length"]),
        "eval_every": int(cfg.TRAIN.EVAL_EVERY) == eval_every,
        "eval_steps": int(cfg.TRAIN.EVAL_NUM_STEPS) == int(recipe["eval_num_steps"]),
        "lr": float(cfg.TRAIN.LR) == float(recipe["learning_rate"]),
        "k_lr": float(cfg.TRAIN.K_MATRIX_LR) == float(recipe["koopman_learning_rate"]),
        "weight_decay": float(cfg.TRAIN.WEIGHT_DECAY) == float(recipe["weight_decay"]),
        "residual_loss": float(cfg.MODEL.RES_COEFF) == float(loss["residual"]),
        "reconstruction_loss": float(cfg.MODEL.RECONST_COEFF) == float(loss["reconstruction"]),
        "prediction_loss": float(cfg.MODEL.PRED_COEFF) == float(loss["prediction"]),
        "sparsity_loss": float(cfg.MODEL.SPARSITY_COEFF) == float(loss["sparsity"]),
        "hard_enabled": bool(cfg.TRAIN.HARD_INIT_OVERSAMPLE.ENABLED) == bool(hard["enabled"]),
        "hard_fraction": float(cfg.TRAIN.HARD_INIT_OVERSAMPLE.FRACTION) == float(hard["fraction"]),
        "hard_pool": int(cfg.TRAIN.HARD_INIT_OVERSAMPLE.POOL_SIZE) == int(hard["pool_size"]),
        "hard_candidates": int(cfg.TRAIN.HARD_INIT_OVERSAMPLE.NUM_CANDIDATES) == int(hard["num_candidates"]),
        "hard_probe": int(cfg.TRAIN.HARD_INIT_OVERSAMPLE.PROBE_STEPS) == int(hard["probe_steps"]),
        "hard_perturbations": int(cfg.TRAIN.HARD_INIT_OVERSAMPLE.NUM_PERTURBATIONS) == int(hard["num_perturbations"]),
        "hard_perturb_scale": float(cfg.TRAIN.HARD_INIT_OVERSAMPLE.PERTURB_SCALE) == float(hard["perturb_scale"]),
        "hard_transient_window": int(cfg.TRAIN.HARD_INIT_OVERSAMPLE.TRANSIENT_WINDOW) == int(hard["transient_window"]),
        "hard_transient_weight": float(cfg.TRAIN.HARD_INIT_OVERSAMPLE.TRANSIENT_WEIGHT) == float(hard["transient_weight"]),
        "hard_jitter": float(cfg.TRAIN.HARD_INIT_OVERSAMPLE.JITTER_SCALE) == float(hard["jitter_scale"]),
        "hard_build_chunk": int(cfg.TRAIN.HARD_INIT_OVERSAMPLE.BUILD_CHUNK_SIZE) == int(hard["build_chunk_size"]),
        "checkpoint_at_frozen_selection_step": checkpoint_step == int(recipe["num_steps"]) - 1
        or (checkpoint_step > 0 and checkpoint_step % eval_every == 0),
        "compact_checkpoint_optimizer_absent": serialized is None,
    }


def assert_exact_checkpoint(
    cfg: Config, model: nn.Module, checkpoint: dict[str, Any],
    card: dict[str, Any], spec: TrainedRun,
) -> dict[str, Any]:
    recipe = card["training_arms"][spec.arm]
    checks = _audit_shared(cfg, model, checkpoint, recipe, card, spec)
    optimizer_decays = _optimizer_weight_decays(model, cfg)
    compatibility_normalization: dict[str, Any] | None = None
    decoder_linearity: dict[str, float | bool] | None = None
    canonical_current = Config.from_dict(checkpoint["config"]).to_dict()
    checks["current_checkpoint_schema_is_canonical"] = (
        checkpoint["config"] == canonical_current
    )
    if spec.arm == "sparse":
        latent_dim = int(recipe["latent_dim"])
        decoder_linearity = decoder_linearity_diagnostics(model, latent_dim)
        checks.update(
            {
                "sign_split_layout": assert_sign_split_layout(cfg, model) == 128,
                "lista_encoder": str(cfg.MODEL.ENCODER.ENCODER_TYPE).lower() == "lista",
                "lista_loops": int(cfg.MODEL.ENCODER.LISTA.NUM_LOOPS) == int(recipe["encoder"]["loops"]),
                "lista_alpha": float(cfg.MODEL.ENCODER.LISTA.ALPHA) == float(recipe["encoder"]["alpha"]),
                "lista_final_op": str(cfg.MODEL.ENCODER.LISTA.FINAL_OP).lower() == recipe["encoder"]["final_op"],
                "lista_group_shrinkage_off": not bool(cfg.MODEL.ENCODER.LISTA.GROUP_SHRINKAGE),
                "lista_topk_off": int(cfg.MODEL.ENCODER.LISTA.TOPK_GROUPS) == 0,
                "optimizer_sparse_wd_exact": sorted(optimizer_decays) == [0.0, 0.0001],
                "normalized_dictionary_decoder_shape": tuple(model.dict.shape)
                == (latent_dim, 2),
                "normalized_dictionary_decoder_trainable": bool(model.dict.requires_grad),
                "sparse_decoder_bias_disabled": not bool(model.use_decoder_bias)
                and not bool(model.decoder_bias.requires_grad)
                and bool(torch.count_nonzero(model.decoder_bias) == 0),
                "sparse_decoder_zero_preserving": bool(
                    decoder_linearity["zero_preserving"]
                ),
                "sparse_decoder_additive_in_latent": bool(
                    decoder_linearity["additive"]
                ),
                "sparse_decoder_homogeneous_in_latent": bool(
                    decoder_linearity["homogeneous"]
                ),
                "sparse_decoder_linear_in_latent": bool(decoder_linearity["linear"]),
            }
        )
        representative_raw = json.loads(
            Path(recipe["representative_frozen_config"]).read_text()
        )
        representative = Config.from_dict(representative_raw).to_dict()
        observed = canonical_current
        changes = _schema_changes(representative_raw, representative)
        allowlist = recipe["representative_schema_compatibility_normalization"][
            "exact_allowlist"
        ]
        checks["historical_schema_normalization_exact_allowlist"] = changes == allowlist
        representative["SEED"] = observed["SEED"]
        checks["normalized_full_config_matches_representative_except_seed"] = (
            observed == representative
        )
        compatibility_normalization = {
            "observed_changes": changes,
            "exact_allowlist": allowlist,
            "behavior_equivalence": recipe[
                "representative_schema_compatibility_normalization"
            ]["behavior_equivalence"],
        }
    else:
        modules = list(model.encoder.network)
        decoder_modules = list(model.decoder.network)
        checks.update(
            {
                "tanh_hidden_activation": str(cfg.MODEL.ENCODER.ACTIVATION).lower() == "tanh",
                "linear_output": not bool(cfg.MODEL.ENCODER.LAST_RELU)
                and bool(modules) and isinstance(modules[-1], nn.Linear),
                "encoder_layers": list(cfg.MODEL.ENCODER.LAYERS) == [64, 64],
                "encoder_only_linear_tanh": all(isinstance(module, (nn.Linear, nn.Tanh)) for module in modules),
                "exactly_two_tanh": sum(isinstance(module, nn.Tanh) for module in modules) == 2,
                "no_dropout": not any(isinstance(module, nn.Dropout) for module in model.modules()),
                "zero_sparsity": float(cfg.MODEL.SPARSITY_COEFF) == 0.0,
                "zero_weight_decay": float(cfg.TRAIN.WEIGHT_DECAY) == 0.0,
                "optimizer_zero_wd": bool(optimizer_decays) and all(value == 0.0 for value in optimizer_decays),
                "lista_group_shrinkage_off": not bool(cfg.MODEL.ENCODER.LISTA.GROUP_SHRINKAGE),
                "lista_topk_off": int(cfg.MODEL.ENCODER.LISTA.TOPK_GROUPS) == 0,
                "dense_decoder_one_linear": len(decoder_modules) == 1
                and isinstance(decoder_modules[0], nn.Linear),
                "dense_decoder_linear_bias_absent": len(decoder_modules) == 1
                and isinstance(decoder_modules[0], nn.Linear)
                and decoder_modules[0].bias is None,
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"{spec.arm} checkpoint audit failed: {failed}")
    return {
        "checks": checks,
        "optimizer_param_group_weight_decays": optimizer_decays,
        "checkpoint_step": int(checkpoint["step"]),
        "compatibility_normalization": compatibility_normalization,
        "decoder_linearity_diagnostics": decoder_linearity,
    }


def audit_one(
    spec: TrainedRun, card: dict[str, Any], card_hash: str,
    task_tsv: Path,
) -> dict[str, Any]:
    cfg, _env, model, checkpoint, checkpoint_path = load_trained_model(spec)
    audit = assert_exact_checkpoint(cfg, model, checkpoint, card, spec)
    config_path = spec.run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if json.loads(config_path.read_text()) != checkpoint["config"]:
        raise RuntimeError("config.json and checkpoint config differ")
    parameter_counts = trainable_parameter_counts(model)
    expected_counts = card["training_arms"][spec.arm][
        "expected_trainable_parameter_counts"
    ]
    if parameter_counts != expected_counts:
        raise AssertionError(
            f"Trainable parameter-count drift: {parameter_counts} != {expected_counts}"
        )
    runner_source = REPOSITORY_ROOT / "skae/training/runner.py"
    evaluation_source = REPOSITORY_ROOT / "skae/evaluation.py"
    runner_text, evaluation_text = runner_source.read_text(), evaluation_source.read_text()
    selector_semantics = bool(
        "rollout_every_step_reencode" in runner_text
        and "eval_results['final_error'] < best_eval_final_error" in runner_text
        and "period=1" in evaluation_text
        and "def rollout_every_step_reencode" in evaluation_text
    )
    if not selector_semantics:
        raise AssertionError("Inherited every-step-reencoding selector source drift")
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "task_tsv_sha256": sha256_path(task_tsv),
        "task_id": spec.task_id,
        "arm": spec.arm,
        "seed": spec.seed,
        "status": "passed",
        "run_dir": str(spec.run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "config_sha256": sha256_path(config_path),
        "attempt_count": spec.attempt_count,
        "incomplete_attempt_count": spec.incomplete_attempt_count,
        "trainable_parameter_counts": parameter_counts,
        "checkpoint_selector": {
            "selected_step": int(checkpoint["step"]),
            "metric": "fixed_validation_final_physical_state_error",
            "rollout": "rollout_every_step_reencode",
            "physical_state_reencoded_every_step": True,
            "strict_improvement_retains_earliest_exact_tie": True,
            "not_autonomous_repeated_k_selection": True,
            "source_semantics_verified": selector_semantics,
            "runner_source_sha256": sha256_path(runner_source),
            "evaluation_source_sha256": sha256_path(evaluation_source),
        },
        "audit": audit,
    }


def summarize_audits(output_dir: Path, card: dict[str, Any], card_hash: str) -> dict[str, Any]:
    expected = int(card["task_table_contract"]["full_task_count"])
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted((output_dir / "shards").glob("task_*.json")):
        payload = json.loads(path.read_text())
        if payload.get("protocol_id") != card["protocol_id"] or payload.get("card_sha256") != card_hash:
            raise RuntimeError(f"Audit source mismatch: {path}")
        task_id = int(payload["task_id"])
        if task_id in rows:
            raise RuntimeError(f"Duplicate audit task {task_id}")
        rows[task_id] = payload
    if set(rows) != set(range(expected)):
        raise RuntimeError(f"Incomplete checkpoint audit: {sorted(rows)}")
    ordered = [rows[index] for index in range(expected)]
    task_hashes = {row["task_tsv_sha256"] for row in ordered}
    if len(task_hashes) != 1:
        raise RuntimeError("Task-table hash drift across checkpoint audits")
    count_sets = {
        arm: {
            json.dumps(row["trainable_parameter_counts"], sort_keys=True)
            for row in ordered if row["arm"] == arm
        }
        for arm in ("sparse", "dense")
    }
    if any(len(values) != 1 for values in count_sets.values()):
        raise RuntimeError(f"Parameter-count drift within arm: {count_sets}")
    return {
        "schema_version": 1,
        "protocol_id": card["protocol_id"],
        "card_sha256": card_hash,
        "status": "passed" if all(row["status"] == "passed" for row in ordered) else "failed",
        "task_count": expected,
        "passed_count": sum(row["status"] == "passed" for row in ordered),
        "task_tsv_sha256": next(iter(task_hashes)),
        "arm_counts": {
            arm: sum(row["arm"] == arm for row in ordered)
            for arm in ("sparse", "dense")
        },
        "parameter_counts_by_arm": {
            arm: json.loads(next(iter(count_sets[arm])))
            for arm in ("sparse", "dense")
        },
        "rows": [
            {
                "task_id": row["task_id"], "arm": row["arm"], "seed": row["seed"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "trainable_parameter_counts": row["trainable_parameter_counts"],
                "checkpoint_selector": row["checkpoint_selector"],
            }
            for row in ordered
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--task_tsv", type=Path, required=True)
    parser.add_argument("--base_out", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--task_index", type=int)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    card, card_hash = load_card(args.card)
    if args.summarize:
        payload = summarize_audits(args.output_dir, card, card_hash)
        output = args.output_dir / "summary.json"
    else:
        if args.task_index is None:
            raise ValueError("--task_index is required unless --summarize")
        roster = discover_trained_runs(args.task_tsv, args.base_out, card)
        if not 0 <= args.task_index < len(roster):
            raise IndexError(args.task_index)
        payload = audit_one(roster[args.task_index], card, card_hash, args.task_tsv)
        output = args.output_dir / "shards" / f"task_{args.task_index:02d}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
