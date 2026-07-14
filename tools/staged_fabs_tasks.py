"""Task-table parsing for the staged ``F_abs`` training entry point."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from skae.config import Config, apply_env_dt_override, get_config


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _maybe_float(row: Mapping[str, str], key: str) -> Optional[float]:
    raw = _safe_str(row.get(key))
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _maybe_int(row: Mapping[str, str], key: str) -> Optional[int]:
    value = _maybe_float(row, key)
    return None if value is None else int(round(value))


def _maybe_bool(row: Mapping[str, str], key: str) -> Optional[bool]:
    raw = _safe_str(row.get(key)).lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _tagify(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _read_task_row(
    task_tsv: Path,
    *,
    array_index: int,
    array_offset: int,
) -> Dict[str, str]:
    target = int(array_index) + int(array_offset)
    with task_tsv.open("r", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            if index == target:
                return dict(row)
    raise IndexError(f"No task row for data index {target} in {task_tsv}")


def _apply_task_row_to_config(row: Dict[str, str]) -> Config:
    cfg = get_config(_safe_str(row.get("config_name")) or "lista_parity_generic_sparse")
    cfg.ENV.ENV_NAME = _safe_str(row.get("env_name")) or _safe_str(row.get("system_key"))
    cfg.SEED = _maybe_int(row, "seed") or 0
    for key, attr in (
        ("num_steps", "NUM_STEPS"),
        ("batch_size", "BATCH_SIZE"),
        ("sequence_length", "SEQUENCE_LENGTH"),
    ):
        value = _maybe_int(row, key)
        if value is not None:
            setattr(cfg.TRAIN, attr, value)
    for key, attr in (
        ("lr", "LR"),
        ("k_matrix_lr", "K_MATRIX_LR"),
        ("weight_decay", "WEIGHT_DECAY"),
    ):
        value = _maybe_float(row, key)
        if value is not None:
            setattr(cfg.TRAIN, attr, value)

    hard_enabled = _maybe_bool(row, "hard_init_oversample")
    if hard_enabled is not None:
        cfg.TRAIN.HARD_INIT_OVERSAMPLE.ENABLED = hard_enabled
    for key, attr in {
        "hard_init_fraction": "FRACTION",
        "hard_init_perturb_scale": "PERTURB_SCALE",
        "hard_init_transient_weight": "TRANSIENT_WEIGHT",
        "hard_init_jitter_scale": "JITTER_SCALE",
    }.items():
        value = _maybe_float(row, key)
        if value is not None:
            setattr(cfg.TRAIN.HARD_INIT_OVERSAMPLE, attr, value)
    for key, attr in {
        "hard_init_pool_size": "POOL_SIZE",
        "hard_init_num_candidates": "NUM_CANDIDATES",
        "hard_init_probe_steps": "PROBE_STEPS",
        "hard_init_num_perturbations": "NUM_PERTURBATIONS",
        "hard_init_transient_window": "TRANSIENT_WINDOW",
    }.items():
        value = _maybe_int(row, key)
        if value is not None:
            setattr(cfg.TRAIN.HARD_INIT_OVERSAMPLE, attr, value)

    target_size = _maybe_int(row, "target_size")
    if target_size is not None:
        cfg.MODEL.TARGET_SIZE = target_size
    for key, attr in {
        "res_coeff": "RES_COEFF",
        "reconst_coeff": "RECONST_COEFF",
        "pred_coeff": "PRED_COEFF",
        "sparsity_coeff": "SPARSITY_COEFF",
        "decoder_coherence_weight": "DECODER_COHERENCE_WEIGHT",
    }.items():
        value = _maybe_float(row, key)
        if value is not None:
            setattr(cfg.MODEL, attr, value)
    structure = _safe_str(row.get("k_structure"))
    if structure:
        cfg.MODEL.K_STRUCTURE = structure
    for key, attr in (("k_block_size", "K_BLOCK_SIZE"), ("k_num_blocks", "K_NUM_BLOCKS")):
        value = _maybe_int(row, key)
        if value is not None:
            setattr(cfg.MODEL, attr, value)

    lista = cfg.MODEL.ENCODER.LISTA
    alpha = _maybe_float(row, "lista_alpha")
    if alpha is not None:
        lista.ALPHA = alpha
    loops = _maybe_int(row, "lista_num_loops")
    if loops is not None:
        lista.NUM_LOOPS = loops
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = loops
    for key, attr in (("lista_final_op", "FINAL_OP"), ("lista_precode_mode", "PRECODE_MODE")):
        value = _safe_str(row.get(key))
        if value:
            setattr(lista, attr, value)
    linear = _maybe_bool(row, "lista_linear_encoder")
    if linear is not None:
        lista.LINEAR_ENCODER = linear
    residual_scale = _maybe_float(row, "lista_precode_residual_scale")
    if residual_scale is not None:
        lista.PRECODE_RESIDUAL_SCALE = residual_scale
    env_dt = _maybe_float(row, "env_dt")
    if env_dt is not None:
        apply_env_dt_override(cfg, dt=env_dt, env_name=cfg.ENV.ENV_NAME)
    return cfg
