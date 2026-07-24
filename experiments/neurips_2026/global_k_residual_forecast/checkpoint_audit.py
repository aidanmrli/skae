"""Checkpoint audit that consumes only already-authenticated in-memory objects."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from experiments.neurips_2026.global_k_distinct_laws_v2_checkpoint_audit import (
    TrainedRun,
    _audit_shared,
    _optimizer_weight_decays,
    _schema_changes,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_math import (
    decoder_linearity_diagnostics,
)
from experiments.neurips_2026.global_k_support_invariance import (
    assert_sign_split_layout,
)
from skae.config import Config


def audit_authenticated_checkpoint(
    cfg: Config,
    model: nn.Module,
    checkpoint: dict[str, Any],
    card: dict[str, Any],
    spec: TrainedRun,
    representative_raw: dict[str, Any],
) -> dict[str, Any]:
    """Mirror the frozen V2 audit without opening any external path."""

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
                "lista_encoder": str(cfg.MODEL.ENCODER.ENCODER_TYPE).lower()
                == "lista",
                "lista_loops": int(cfg.MODEL.ENCODER.LISTA.NUM_LOOPS)
                == int(recipe["encoder"]["loops"]),
                "lista_alpha": float(cfg.MODEL.ENCODER.LISTA.ALPHA)
                == float(recipe["encoder"]["alpha"]),
                "lista_final_op": str(cfg.MODEL.ENCODER.LISTA.FINAL_OP).lower()
                == recipe["encoder"]["final_op"],
                "lista_group_shrinkage_off": not bool(
                    cfg.MODEL.ENCODER.LISTA.GROUP_SHRINKAGE
                ),
                "lista_topk_off": int(cfg.MODEL.ENCODER.LISTA.TOPK_GROUPS) == 0,
                "optimizer_sparse_wd_exact": sorted(optimizer_decays)
                == [0.0, 0.0001],
                "normalized_dictionary_decoder_shape": tuple(model.dict.shape)
                == (latent_dim, 2),
                "normalized_dictionary_decoder_trainable": bool(
                    model.dict.requires_grad
                ),
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
                "sparse_decoder_linear_in_latent": bool(
                    decoder_linearity["linear"]
                ),
            }
        )
        raw = copy.deepcopy(representative_raw)
        representative = Config.from_dict(raw).to_dict()
        observed = canonical_current
        changes = _schema_changes(raw, representative)
        allowlist = recipe["representative_schema_compatibility_normalization"][
            "exact_allowlist"
        ]
        checks["historical_schema_normalization_exact_allowlist"] = (
            changes == allowlist
        )
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
                "tanh_hidden_activation": str(cfg.MODEL.ENCODER.ACTIVATION).lower()
                == "tanh",
                "linear_output": not bool(cfg.MODEL.ENCODER.LAST_RELU)
                and bool(modules)
                and isinstance(modules[-1], nn.Linear),
                "encoder_layers": list(cfg.MODEL.ENCODER.LAYERS) == [64, 64],
                "encoder_only_linear_tanh": all(
                    isinstance(module, (nn.Linear, nn.Tanh)) for module in modules
                ),
                "exactly_two_tanh": sum(
                    isinstance(module, nn.Tanh) for module in modules
                )
                == 2,
                "no_dropout": not any(
                    isinstance(module, nn.Dropout) for module in model.modules()
                ),
                "zero_sparsity": float(cfg.MODEL.SPARSITY_COEFF) == 0.0,
                "zero_weight_decay": float(cfg.TRAIN.WEIGHT_DECAY) == 0.0,
                "optimizer_zero_wd": bool(optimizer_decays)
                and all(value == 0.0 for value in optimizer_decays),
                "lista_group_shrinkage_off": not bool(
                    cfg.MODEL.ENCODER.LISTA.GROUP_SHRINKAGE
                ),
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
