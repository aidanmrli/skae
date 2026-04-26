"""Compatibility helpers for loading older model checkpoints."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import torch.nn as nn


_LEGACY_OPTIONAL_MISSING_KEYS = frozenset(
    {
        "encoder.dict_param",
        "encoder.encoder_dict_init",
    }
)


def remap_legacy_model_state_dict(model: nn.Module, state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Remap known legacy checkpoint key prefixes to current model names."""

    remapped: Dict[str, Any] = dict(state_dict)
    model_keys = set(model.state_dict().keys())
    has_encoder_in_model = any(key.startswith("encoder.") for key in model_keys)
    has_encoder_in_ckpt = any(key.startswith("encoder.") for key in remapped.keys())
    has_lista_in_ckpt = any(key.startswith("lista.") for key in remapped.keys())

    # Older LISTAKM checkpoints used `lista.*`; current code expects `encoder.*`.
    if has_encoder_in_model and has_lista_in_ckpt and not has_encoder_in_ckpt:
        remapped = {
            (f"encoder.{key[len('lista.'):]}" if key.startswith("lista.") else key): value
            for key, value in remapped.items()
        }

    # Older LISTA checkpoints stored the pre-code module under `encoder.We.*`.
    if any(key.startswith("encoder.precode_module.") for key in model_keys) and any(
        key.startswith("encoder.We.") for key in remapped.keys()
    ):
        remapped = {
            (
                f"encoder.precode_module.{key[len('encoder.We.'):]}"
                if key.startswith("encoder.We.")
                else key
            ): value
            for key, value in remapped.items()
        }

    return remapped


def load_model_state_dict_compat(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    allowed_missing_keys: Iterable[str] = _LEGACY_OPTIONAL_MISSING_KEYS,
) -> Tuple[Iterable[str], Iterable[str]]:
    """Load a checkpoint state dict while tolerating known legacy omissions."""

    remapped = remap_legacy_model_state_dict(model, state_dict)
    incompat = model.load_state_dict(remapped, strict=False)
    missing_keys = set(incompat.missing_keys)
    unexpected_keys = set(incompat.unexpected_keys)
    allowed_missing = set(allowed_missing_keys)

    if unexpected_keys or not missing_keys.issubset(allowed_missing):
        details = []
        if missing_keys:
            details.append(f"missing={sorted(missing_keys)}")
        if unexpected_keys:
            details.append(f"unexpected={sorted(unexpected_keys)}")
        raise RuntimeError(
            "Checkpoint compatibility load failed after legacy remap: " + ", ".join(details)
        )

    return sorted(missing_keys), sorted(unexpected_keys)
