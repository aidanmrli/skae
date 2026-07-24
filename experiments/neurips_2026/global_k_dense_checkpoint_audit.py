"""Fail-closed architecture and optimizer audit for the exact-dense control."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.neurips_2026.global_k_dense_zero_wd_tasks import sha256_path
from skae.config import Config
from skae.training.runner import build_optimizer


def assert_exact_dense_control(
    cfg: Config,
    model: nn.Module,
    card: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Reject any checkpoint that departs from the frozen no-sparsity recipe."""

    train = card["training"]
    hard = train["hard_initial_condition_oversampling"]
    encoder_layers = list(model.encoder.network)
    decoder_layers = list(model.decoder.network)
    decoder_linear = decoder_layers[0] if len(decoder_layers) == 1 else None
    reconstructed_optimizer = build_optimizer(model, cfg)
    reconstructed_weight_decays = [
        float(group.get("weight_decay", 0.0))
        for group in reconstructed_optimizer.param_groups
    ]
    optimizer_state = checkpoint.get("optimizer_state_dict")
    serialized_weight_decays = (
        None
        if optimizer_state is None
        else [
            float(group.get("weight_decay", 0.0))
            for group in optimizer_state.get("param_groups", [])
        ]
    )
    checks = {
        "model": cfg.MODEL.MODEL_NAME == "GenericKM",
        "latent": int(cfg.MODEL.TARGET_SIZE) == int(train["latent_dim"]),
        "identity_latent_normalization": str(cfg.MODEL.NORM_FN).lower() == "id",
        "activation": str(cfg.MODEL.ENCODER.ACTIVATION).lower() == "tanh",
        "linear_latent": not bool(cfg.MODEL.ENCODER.LAST_RELU),
        "encoder_layers": list(cfg.MODEL.ENCODER.LAYERS)
        == train["encoder_hidden_layers"],
        "encoder_modules_only_linear_tanh": all(
            isinstance(module, (nn.Linear, nn.Tanh)) for module in encoder_layers
        ),
        "encoder_tanh_count": sum(
            isinstance(module, nn.Tanh) for module in encoder_layers
        )
        == 2,
        "encoder_final_module_linear": bool(encoder_layers)
        and isinstance(encoder_layers[-1], nn.Linear),
        "linear_decoder": list(cfg.MODEL.DECODER.LAYERS) == [],
        "decoder_one_linear": isinstance(decoder_linear, nn.Linear),
        "decoder_bias_absent": isinstance(decoder_linear, nn.Linear)
        and decoder_linear.bias is None,
        "decoder_atom_normalization_off": not bool(
            cfg.MODEL.DECODER.NORMALIZE_ATOMS
        ),
        "dense_k": str(cfg.MODEL.K_STRUCTURE).lower() == "dense",
        "zero_sparsity": float(cfg.MODEL.SPARSITY_COEFF) == 0.0,
        "zero_weight_decay": float(cfg.TRAIN.WEIGHT_DECAY) == 0.0,
        "reconstructed_optimizer_zero_weight_decay": bool(
            reconstructed_weight_decays
        )
        and all(value == 0.0 for value in reconstructed_weight_decays),
        "serialized_optimizer_zero_or_compact_absent": serialized_weight_decays
        is None
        or (
            bool(serialized_weight_decays)
            and all(value == 0.0 for value in serialized_weight_decays)
        ),
        "optimizer_is_adamw": isinstance(
            reconstructed_optimizer, torch.optim.AdamW
        ),
        "zero_group_shrinkage": not bool(
            cfg.MODEL.ENCODER.LISTA.GROUP_SHRINKAGE
        ),
        "zero_topk": int(cfg.MODEL.ENCODER.LISTA.TOPK_GROUPS) == 0,
        "block_loss_off": not bool(cfg.MODEL.BLOCK_LOSS.ENABLED),
        "soft_block_off": not bool(cfg.MODEL.SOFT_BLOCK.ENABLED),
        "zero_coherence": float(cfg.MODEL.DECODER_COHERENCE_WEIGHT) == 0.0,
        "homogeneous_coordinates_off": not bool(cfg.MODEL.USE_HOMOGENEOUS),
        "no_dropout": not any(
            isinstance(module, nn.Dropout) for module in model.modules()
        ),
        "budget": int(cfg.TRAIN.NUM_STEPS) == int(train["num_steps"]),
        "batch": int(cfg.TRAIN.BATCH_SIZE) == int(train["batch_size"]),
        "data": int(cfg.TRAIN.DATA_SIZE) == int(train["data_size"]),
        "sequence": int(cfg.TRAIN.SEQUENCE_LENGTH)
        == int(train["sequence_length"]),
        "eval_every": int(cfg.TRAIN.EVAL_EVERY) == int(train["eval_every"]),
        "eval_num_steps": int(cfg.TRAIN.EVAL_NUM_STEPS)
        == int(train["eval_num_steps"]),
        "learning_rate": float(cfg.TRAIN.LR) == float(train["learning_rate"]),
        "koopman_learning_rate": float(cfg.TRAIN.K_MATRIX_LR)
        == float(train["koopman_learning_rate"]),
        "residual_coefficient": float(cfg.MODEL.RES_COEFF)
        == float(train["residual_coefficient"]),
        "reconstruction_coefficient": float(cfg.MODEL.RECONST_COEFF)
        == float(train["reconstruction_coefficient"]),
        "prediction_coefficient": float(cfg.MODEL.PRED_COEFF)
        == float(train["prediction_coefficient"]),
        "hard_init_enabled": bool(cfg.TRAIN.HARD_INIT_OVERSAMPLE.ENABLED),
        "hard_init_fraction": float(cfg.TRAIN.HARD_INIT_OVERSAMPLE.FRACTION)
        == float(hard["fraction"]),
        "hard_init_pool": int(cfg.TRAIN.HARD_INIT_OVERSAMPLE.POOL_SIZE)
        == int(hard["pool_size"]),
        "hard_init_candidates": int(
            cfg.TRAIN.HARD_INIT_OVERSAMPLE.NUM_CANDIDATES
        )
        == int(hard["num_candidates"]),
        "hard_init_probe": int(cfg.TRAIN.HARD_INIT_OVERSAMPLE.PROBE_STEPS)
        == int(hard["probe_steps"]),
        "hard_init_perturbations": int(
            cfg.TRAIN.HARD_INIT_OVERSAMPLE.NUM_PERTURBATIONS
        )
        == int(hard["num_perturbations"]),
        "hard_init_perturb_scale": float(
            cfg.TRAIN.HARD_INIT_OVERSAMPLE.PERTURB_SCALE
        )
        == float(hard["perturb_scale"]),
        "hard_init_transient_window": int(
            cfg.TRAIN.HARD_INIT_OVERSAMPLE.TRANSIENT_WINDOW
        )
        == int(hard["transient_window"]),
        "hard_init_transient_weight": float(
            cfg.TRAIN.HARD_INIT_OVERSAMPLE.TRANSIENT_WEIGHT
        )
        == float(hard["transient_weight"]),
        "hard_init_jitter_scale": float(
            cfg.TRAIN.HARD_INIT_OVERSAMPLE.JITTER_SCALE
        )
        == float(hard["jitter_scale"]),
        "hard_init_build_chunk": int(
            cfg.TRAIN.HARD_INIT_OVERSAMPLE.BUILD_CHUNK_SIZE
        )
        == int(hard["build_chunk_size"]),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Dense-control checkpoint assertion failure: {failed}")
    return {
        "compact_best_checkpoint_omits_optimizer_state": optimizer_state is None,
        "reconstructed_param_group_weight_decays": reconstructed_weight_decays,
        "serialized_param_group_weight_decays": serialized_weight_decays,
        "training_runner_sha256": sha256_path(Path("skae/training/runner.py")),
        "optimizer_class": type(reconstructed_optimizer).__name__,
    }

