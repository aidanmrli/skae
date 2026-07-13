#!/usr/bin/env python3
"""Train staged LISTA support-family local transition maps.

Protocol:
1. Train encoder, decoder, and global K jointly for the first half of the
   requested training budget.
2. Freeze the encoder and decoder, build F_abs support families or matched
   route-control partitions from training batches seen by the run itself, and
   initialize one local K_c per route from the learned global K.
3. Train all local K_c matrices, optionally with a learned affine target
   center/intercept per support family, for the second half of the budget on
   the same training data stream.

The script is intentionally task-TSV driven so it can reuse the exact Table 1
LISTA dense p256 hard-init recipe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from skae.config import Config, apply_env_dt_override, get_config
from skae.data import VectorWrapper, generate_trajectory, make_env, wrap_training_env
from skae.evaluation import EvaluationSettings, evaluate_model
from skae.model import make_model

from tools.train import (
    MetricsLogger,
    build_optimizer,
    generate_sequence_batch_for_device,
    get_device,
    train_step,
)
from tools.train_support_family_local_maps import (
    OPSEL,
    REDUCER,
    _build_route_codebook,
    _route_indices_np,
    _step_routes_for_torch,
)
from tools.evaluate_stable_support_components import (
    UNCERTAIN,
    _stable_support_components,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_tsv", required=True)
    parser.add_argument("--array_index", type=int, default=0)
    parser.add_argument("--array_offset", type=int, default=0)
    parser.add_argument("--base_out", required=True)
    parser.add_argument("--support_definition", default="absolute:0.001")
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.4)
    parser.add_argument("--support_fit_batches", type=int, default=16)
    parser.add_argument("--min_family_transitions", type=int, default=1)
    parser.add_argument(
        "--support_family_fit_source",
        default="stage1_buffer",
        choices=("stage1_buffer", "stable_fit_trajectories"),
        help=(
            "Fit source for support_family routes. The historical stage1_buffer "
            "uses retained stage-1 minibatches; stable_fit_trajectories uses the "
            "same long training-distribution route-fit trajectories as C_stab."
        ),
    )
    parser.add_argument(
        "--routing_object",
        default="support_family",
        choices=(
            "support_family",
            "stable_support_component",
            "oracle_basin",
            "latent_kmeans",
            "latent_tail_fate",
            "random_matched",
        ),
        help=(
            "Object used to build and route local maps. support_family keeps the "
            "historical instantaneous Jaccard F_abs families. "
            "stable_support_component builds C_stab from support-flow fate and "
            "routes local maps by the resulting stable support component. "
            "oracle_basin uses privileged benchmark basin labels. latent_kmeans "
            "uses k-means in the frozen latent space with C_stab-matched cluster "
            "count by default. latent_tail_fate clusters continuous latent tail "
            "summaries per route-fit trajectory and assigns all transitions from "
            "a trajectory to that fate label. random_matched shuffles C_stab route "
            "labels on the same fit transitions and routes by nearest latent route center."
        ),
    )
    parser.add_argument(
        "--stable_base_object",
        default="family",
        choices=("exact", "family"),
        help="High-resolution support-flow nodes used before building C_stab.",
    )
    parser.add_argument("--stable_base_family_jaccard", type=float, default=0.8)
    parser.add_argument("--stable_tail_window", type=int, default=32)
    parser.add_argument("--stable_min_edge_count", type=int, default=2)
    parser.add_argument("--stable_min_edge_probability", type=float, default=0.02)
    parser.add_argument("--stable_max_recurrent_out_probability", type=float, default=0.05)
    parser.add_argument("--stable_min_tail_count", type=int, default=8)
    parser.add_argument("--stable_min_absorption_observations", type=int, default=8)
    parser.add_argument("--stable_min_absorption_confidence", type=float, default=0.80)
    parser.add_argument(
        "--stable_fit_trajectories",
        type=int,
        default=256,
        help="Number of training-distribution trajectories used to build C_stab after stage 1.",
    )
    parser.add_argument(
        "--stable_fit_trajectory_length",
        type=int,
        default=192,
        help="Trajectory length used to build support-flow components for C_stab.",
    )
    parser.add_argument("--stable_fit_seed_offset", type=int, default=271828)
    parser.add_argument(
        "--baseline_route_seed_offset",
        type=int,
        default=314159,
        help="Seed offset for stochastic matched-route controls such as random_matched and latent_kmeans.",
    )
    parser.add_argument(
        "--baseline_latent_cluster_count",
        type=int,
        default=0,
        help=(
            "Number of latent k-means routes. The default 0 matches the number "
            "of fitted C_stab routes for the same stage-1 model and fit dataset."
        ),
    )
    parser.add_argument(
        "--baseline_kmeans_n_init",
        type=int,
        default=10,
        help="Number of k-means initializations for the latent_kmeans route baseline.",
    )
    parser.add_argument("--latent_fate_tail_window", type=int, default=16)
    parser.add_argument("--latent_fate_max_clusters", type=int, default=12)
    parser.add_argument("--latent_fate_min_silhouette", type=float, default=0.05)
    parser.add_argument("--latent_fate_pca_components", type=int, default=16)
    parser.add_argument(
        "--local_map_parameterization",
        default="source_target_affine_global_init",
        choices=("source_target_affine_global_init", "source_target_affine_learned_intercept"),
        help=(
            "source_target_affine_global_init uses z_next = d_c + "
            "(z - c_c) @ K_c with d_c = c_c @ K_global, so stage 2 starts "
            "exactly at the frozen global-K map instead of assuming c_c is fixed. "
            "source_target_affine_learned_intercept uses the same initialization "
            "but makes d_c trainable, giving a fully affine local chart."
        ),
    )
    parser.add_argument("--local_lr", type=float, default=None)
    parser.add_argument(
        "--num_steps_override",
        type=int,
        default=None,
        help="Override the task-table training budget for short interactive probes.",
    )
    parser.add_argument(
        "--stage1_steps_override",
        type=int,
        default=None,
        help="Override the joint-training stage length for short interactive probes.",
    )
    parser.add_argument(
        "--eval_every_override",
        type=int,
        default=None,
        help="Override TRAIN.EVAL_EVERY for short interactive probes.",
    )
    parser.add_argument(
        "--eval_num_steps_override",
        type=int,
        default=None,
        help="Override TRAIN.EVAL_NUM_STEPS for short interactive probes.",
    )
    parser.add_argument(
        "--stage2_selection_metric",
        default="every_step_final_l2",
        choices=("every_step_final_l2", "best_periodic_mse", "best_periodic_horizon_mse"),
        help=(
            "Metric used to choose the staged local-K checkpoint. The default "
            "preserves historical behavior; best_periodic_mse selects by the "
            "best periodic-reset MSE over --stage2_selection_periods; "
            "best_periodic_horizon_mse selects by displayed-horizon periodic MSE."
        ),
    )
    parser.add_argument(
        "--stage2_selection_periods",
        default="1,2,5,10,20,25,50,100",
        help="Comma-separated periodic re-encoding periods for best_periodic_mse selection.",
    )
    parser.add_argument(
        "--stage2_selection_horizons",
        default="100,500,1000",
        help="Comma-separated horizons for best_periodic_horizon_mse selection.",
    )
    parser.add_argument(
        "--stage2_selection_batch_size",
        type=int,
        default=32,
        help=(
            "Validation batch size for best_periodic_horizon_mse selection. "
            "The historical selectors keep their original 16-start batch."
        ),
    )
    parser.add_argument(
        "--stage2_selection_seed_offset",
        type=int,
        default=12345,
        help=(
            "Seed offset for best_periodic_horizon_mse selection; the default "
            "matches EvaluationSettings.seed_offset."
        ),
    )
    parser.add_argument(
        "--eval_periodic_periods_override",
        default=None,
        help="Comma-separated periodic re-encoding periods for final standardized evaluation.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--eval_profile", default="full", choices=("full", "smoke"))
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--resume_from_latest", action="store_true", default=True)
    parser.add_argument("--no_resume_from_latest", dest="resume_from_latest", action="store_false")
    parser.add_argument(
        "--save_metrics_history",
        action="store_true",
        help="Write raw per-step metrics_history.jsonl. Off by default.",
    )
    parser.add_argument(
        "--save_last_checkpoint",
        action="store_true",
        help="Write last.pt for resumability/debugging. checkpoint.pt is always saved.",
    )
    parser.add_argument(
        "--save_stage2_artifacts",
        action="store_true",
        help="Write stage2_artifacts.pt. Off by default because checkpoint.pt includes route metadata.",
    )
    parser.add_argument(
        "--save_eval_rollout_artifacts",
        action="store_true",
        help="Save raw standardized-evaluation rollout tensors.",
    )
    parser.add_argument(
        "--save_eval_plots",
        action="store_true",
        help="Render standardized-evaluation qualitative plots.",
    )
    parser.add_argument(
        "--save_eval_per_ic_values",
        action="store_true",
        help="Include per-initial-condition metric arrays in evaluation JSON.",
    )
    parser.add_argument(
        "--save_eval_error_curves",
        action="store_true",
        help="Include long per-step error curves in evaluation JSON.",
    )
    return parser.parse_args()


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _maybe_float(row: Dict[str, str], key: str) -> Optional[float]:
    raw = _safe_str(row.get(key))
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _maybe_int(row: Dict[str, str], key: str) -> Optional[int]:
    value = _maybe_float(row, key)
    if value is None:
        return None
    return int(round(value))


def _maybe_bool(row: Dict[str, str], key: str) -> Optional[bool]:
    raw = _safe_str(row.get(key)).lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _tagify(value: object) -> str:
    raw = str(value)
    raw = raw.replace("-", "m").replace(".", "p")
    return raw


def _read_task_row(task_tsv: Path, *, array_index: int, array_offset: int) -> Dict[str, str]:
    target_data_index = int(array_index) + int(array_offset)
    with task_tsv.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for data_index, row in enumerate(reader):
            if data_index == target_data_index:
                return dict(row)
    raise IndexError(f"No task row for data index {target_data_index} in {task_tsv}")


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
    hard_float_fields = {
        "hard_init_fraction": "FRACTION",
        "hard_init_perturb_scale": "PERTURB_SCALE",
        "hard_init_transient_weight": "TRANSIENT_WEIGHT",
        "hard_init_jitter_scale": "JITTER_SCALE",
    }
    for key, attr in hard_float_fields.items():
        value = _maybe_float(row, key)
        if value is not None:
            setattr(cfg.TRAIN.HARD_INIT_OVERSAMPLE, attr, value)
    hard_int_fields = {
        "hard_init_pool_size": "POOL_SIZE",
        "hard_init_num_candidates": "NUM_CANDIDATES",
        "hard_init_probe_steps": "PROBE_STEPS",
        "hard_init_num_perturbations": "NUM_PERTURBATIONS",
        "hard_init_transient_window": "TRANSIENT_WINDOW",
    }
    for key, attr in hard_int_fields.items():
        value = _maybe_int(row, key)
        if value is not None:
            setattr(cfg.TRAIN.HARD_INIT_OVERSAMPLE, attr, value)

    int_model_fields = {"target_size": "TARGET_SIZE"}
    for key, attr in int_model_fields.items():
        value = _maybe_int(row, key)
        if value is not None:
            setattr(cfg.MODEL, attr, value)
    float_model_fields = {
        "res_coeff": "RES_COEFF",
        "reconst_coeff": "RECONST_COEFF",
        "pred_coeff": "PRED_COEFF",
        "sparsity_coeff": "SPARSITY_COEFF",
        "decoder_coherence_weight": "DECODER_COHERENCE_WEIGHT",
    }
    for key, attr in float_model_fields.items():
        value = _maybe_float(row, key)
        if value is not None:
            setattr(cfg.MODEL, attr, value)

    k_structure = _safe_str(row.get("k_structure"))
    if k_structure:
        cfg.MODEL.K_STRUCTURE = k_structure
    k_block_size = _maybe_int(row, "k_block_size")
    if k_block_size is not None:
        cfg.MODEL.K_BLOCK_SIZE = k_block_size
    k_num_blocks = _maybe_int(row, "k_num_blocks")
    if k_num_blocks is not None:
        cfg.MODEL.K_NUM_BLOCKS = k_num_blocks

    lista_alpha = _maybe_float(row, "lista_alpha")
    if lista_alpha is not None:
        cfg.MODEL.ENCODER.LISTA.ALPHA = lista_alpha
    lista_num_loops = _maybe_int(row, "lista_num_loops")
    if lista_num_loops is not None:
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = lista_num_loops
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = lista_num_loops
    lista_final_op = _safe_str(row.get("lista_final_op"))
    if lista_final_op:
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = lista_final_op
    lista_linear = _maybe_bool(row, "lista_linear_encoder")
    if lista_linear is not None:
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = lista_linear
    lista_precode_mode = _safe_str(row.get("lista_precode_mode"))
    if lista_precode_mode:
        cfg.MODEL.ENCODER.LISTA.PRECODE_MODE = lista_precode_mode
    lista_precode_residual_scale = _maybe_float(row, "lista_precode_residual_scale")
    if lista_precode_residual_scale is not None:
        cfg.MODEL.ENCODER.LISTA.PRECODE_RESIDUAL_SCALE = lista_precode_residual_scale

    env_dt = _maybe_float(row, "env_dt")
    if env_dt is not None:
        apply_env_dt_override(cfg, dt=env_dt, env_name=cfg.ENV.ENV_NAME)
    return cfg


def _support_definition(raw: str) -> Tuple[str, float]:
    if ":" not in raw:
        raise ValueError(f"Support definition must be scheme:value, got {raw!r}")
    scheme, value = raw.split(":", 1)
    scheme = scheme.strip()
    if scheme not in {"absolute", "relative", "topk"}:
        raise ValueError(f"Unsupported support scheme {scheme!r}")
    return scheme, float(value)


def _parse_int_csv(raw: Optional[str]) -> Tuple[int, ...]:
    if raw is None:
        return ()
    values: List[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Periodic re-encoding periods must be positive, got {value}")
        values.append(value)
    return tuple(dict.fromkeys(values))


def _freeze_autoencoder(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "kmat" in name or name.startswith("K_"):
            continue
        param.requires_grad_(False)


def _encode_sequence_batches(model: nn.Module, batches: Sequence[torch.Tensor], device: str) -> np.ndarray:
    if not batches:
        raise RuntimeError("No training batches were retained for support-family construction")
    model.eval()
    latents: List[np.ndarray] = []
    model_device = next(model.parameters()).device
    with torch.no_grad():
        for batch in batches:
            x_seq = batch.to(model_device)
            batch_size, seq_len, obs_dim = x_seq.shape
            z = model.encode(x_seq.reshape(batch_size * seq_len, obs_dim))
            z = z.reshape(batch_size, seq_len, -1)
            latents.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
    del device
    return np.concatenate(latents, axis=0)


def _generate_support_fit_batches(
    train_env: VectorWrapper,
    *,
    num_trajectories: int,
    trajectory_length: int,
    seed: int,
) -> List[torch.Tensor]:
    target_count = max(1, int(num_trajectories))
    length = max(2, int(trajectory_length))
    rng = torch.Generator().manual_seed(int(seed))
    batches: List[torch.Tensor] = []
    total = 0
    while total < target_count:
        batch = train_env.generate_sequence_batch(rng, window_length=length).float().cpu()
        batches.append(batch)
        total += int(batch.shape[0])
    if total == target_count:
        return batches
    concatenated = torch.cat(batches, dim=0)[:target_count].contiguous()
    return [concatenated]


def _target_centers_from_global(
    centers: Dict[object, np.ndarray],
    family_ids: Sequence[object],
    global_k: np.ndarray,
) -> Dict[object, np.ndarray]:
    targets: Dict[object, np.ndarray] = {}
    for family_id in family_ids:
        source = np.asarray(centers[family_id], dtype=np.float32)
        targets[family_id] = (source @ global_k).astype(np.float32, copy=False)
    return targets


def _remap_object_labels(labels: np.ndarray) -> np.ndarray:
    flat = labels.reshape(-1)
    mapping: Dict[object, int] = {}
    out = np.empty(flat.shape[0], dtype=np.int64)
    for idx, item in enumerate(flat.tolist()):
        if item not in mapping:
            mapping[item] = len(mapping)
        out[idx] = mapping[item]
    return out.reshape(labels.shape)


def _build_stable_route_codebook(
    fit_latents: np.ndarray,
    *,
    scheme: str,
    value: float,
    min_operator_transitions: int,
    route_jaccard_threshold: float,
    stable_base_object: str,
    stable_base_family_jaccard: float,
    stable_tail_window: int,
    stable_min_edge_count: int,
    stable_min_edge_probability: float,
    stable_max_recurrent_out_probability: float,
    stable_min_tail_count: int,
    stable_min_absorption_observations: int,
    stable_min_absorption_confidence: float,
) -> Dict[str, object]:
    """Build a route codebook whose classes are stable support components.

    The runtime router still starts from a current support mask. Exact support
    keys seen during fitting map directly to their stable component; unseen
    supports fall back to the representative mask for each stable component
    using the same Jaccard threshold machinery as the historical family route.
    """

    support_mask = REDUCER._support_mask(fit_latents, scheme=scheme, value=value)
    support_keys = REDUCER._support_keys(support_mask)
    if stable_base_object == "exact":
        base_labels = _remap_object_labels(support_keys)
    elif stable_base_object == "family":
        base_labels = REDUCER.support_family_labels(
            support_mask,
            min_jaccard=float(stable_base_family_jaccard),
        ).astype(np.int64, copy=False)
    else:
        raise ValueError(f"Unsupported stable_base_object={stable_base_object!r}")

    stable_labels, stable_diagnostics = _stable_support_components(
        base_labels,
        tail_window=int(stable_tail_window),
        min_edge_count=int(stable_min_edge_count),
        min_edge_probability=float(stable_min_edge_probability),
        max_recurrent_out_probability=float(stable_max_recurrent_out_probability),
        min_tail_count=int(stable_min_tail_count),
        min_absorption_observations=int(stable_min_absorption_observations),
        min_absorption_confidence=float(stable_min_absorption_confidence),
    )

    x_fit = fit_latents[:, :-1, :].reshape(-1, fit_latents.shape[-1]).astype(np.float32, copy=False)
    support_cur = support_keys[:, :-1].reshape(-1).astype(object)
    stable_cur = stable_labels[:, :-1].reshape(-1).astype(np.int64, copy=False)
    valid = stable_cur != int(UNCERTAIN)
    counts = Counter(stable_cur[valid].tolist())
    fitted_family_ids = sorted(
        [
            int(component_id)
            for component_id, count in counts.items()
            if int(count) >= int(min_operator_transitions)
        ],
        key=lambda item: str(item),
    )
    centers: Dict[object, np.ndarray] = {}
    for component_id in fitted_family_ids:
        centers[component_id] = x_fit[valid & (stable_cur == int(component_id))].mean(axis=0).astype(
            np.float32,
            copy=False,
        )

    flat_support_mask = support_mask[:, :-1, :].reshape(-1, support_mask.shape[-1])
    stable_cur_obj = stable_cur[valid].astype(object)
    support_cur_valid = support_cur[valid]
    flat_support_valid = flat_support_mask[valid]
    family_prototypes = OPSEL._prototype_masks_from_exact_support(
        stable_cur_obj,
        support_cur_valid,
        flat_support_valid,
        class_kind="family",
    )

    support_to_component_counts: Dict[object, Counter[int]] = {}
    for support_key, component_id in zip(support_cur_valid.tolist(), stable_cur[valid].tolist()):
        support_to_component_counts.setdefault(support_key, Counter())[int(component_id)] += 1
    support_key_to_family: Dict[object, object] = {
        support_key: counter.most_common(1)[0][0]
        for support_key, counter in support_to_component_counts.items()
    }

    return {
        "support_mask": support_mask,
        "family_labels": stable_labels,
        "base_labels": base_labels,
        "family_counts": counts,
        "fitted_family_ids": fitted_family_ids,
        "centers": centers,
        "family_prototypes": family_prototypes,
        "support_key_to_family": support_key_to_family,
        "routing_object": "stable_support_component",
        "route_jaccard_threshold": float(route_jaccard_threshold),
        "stable_base_object": stable_base_object,
        "stable_base_family_jaccard": float(stable_base_family_jaccard),
        "stable_diagnostics": stable_diagnostics,
    }


def _runtime_routing_kind(route_codebook: Dict[str, object]) -> str:
    routing_kind = route_codebook.get("runtime_routing_kind")
    if routing_kind:
        return str(routing_kind)
    routing_object = str(route_codebook.get("routing_object", "support_family"))
    if routing_object in {"support_family", "stable_support_component"}:
        return "support_jaccard"
    if routing_object == "oracle_basin":
        return "oracle_basin_label"
    if routing_object in {"latent_kmeans", "latent_tail_fate", "random_matched"}:
        return "nearest_latent_center"
    return routing_object


def _base_env(env: object) -> object:
    return getattr(env, "unwrapped", env)


def _label_owners(env: object) -> List[object]:
    base = _base_env(env)
    owners = [base]
    system = getattr(base, "system", None)
    if system is not None and system is not base:
        owners.append(system)
    return owners


def _call_labeler(labeler: object, flat_states: torch.Tensor) -> torch.Tensor:
    try:
        labels = labeler(flat_states)
    except Exception:
        per_state = [labeler(state) for state in flat_states]
        labels = torch.as_tensor(
            [int(item.item()) if isinstance(item, torch.Tensor) else int(item) for item in per_state],
            dtype=torch.long,
        )
    if isinstance(labels, int):
        labels = torch.tensor([labels], dtype=torch.long)
    elif not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    return labels.to(dtype=torch.long).reshape(-1)


def _coerce_center_matrix(value: object, state_dim: int) -> Optional[torch.Tensor]:
    try:
        centers = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] < 1:
        return None
    centers = centers[:, : int(state_dim)]
    if centers.shape[1] < int(state_dim):
        pad = torch.zeros(
            centers.shape[0],
            int(state_dim) - centers.shape[1],
            dtype=centers.dtype,
            device=centers.device,
        )
        centers = torch.cat([centers, pad], dim=1)
    return centers


def _special_basin_centers(owner: object, state_dim: int) -> Optional[torch.Tensor]:
    system_name = str(getattr(owner, "name", "")).lower()
    class_name = owner.__class__.__name__.lower()
    if system_name == "arrested_spiral" or class_name == "arrestedspiral":
        well_centers = _coerce_center_matrix(getattr(owner, "well_centers", None), state_dim)
        if well_centers is not None:
            origin = torch.zeros(1, int(state_dim), dtype=well_centers.dtype)
            return torch.cat([well_centers, origin], dim=0)

    if system_name == "duffing_triple_well" or class_name == "duffingtriplewell":
        a = float(getattr(owner, "a", 0.3))
        outer_sq = max(0.0, 1.0 + float(np.sqrt(max(0.0, 1.0 - 2.0 * a))))
        outer = float(np.sqrt(outer_sq))
        centers = torch.zeros(3, int(state_dim), dtype=torch.float32)
        centers[:, 0] = torch.tensor([-outer, 0.0, outer], dtype=torch.float32)
        return centers

    if system_name == "snic_multi" or class_name == "snicmulti":
        a = float(getattr(owner, "a", 1.2))
        omega = float(getattr(owner, "omega", 0.5))
        eps = float(getattr(owner, "eps", 0.3))
        cos_value = float(np.clip((1.0 - omega) / max(abs(a), 1e-6), -1.0, 1.0))
        theta0 = -float(np.arccos(cos_value)) / 3.0
        radius = 1.0
        radial_rhs = eps * cos_value
        for candidate in np.linspace(0.2, 1.5, 512):
            if candidate * (1.0 - candidate**2) <= radial_rhs:
                radius = float(candidate)
                break
        centers = torch.zeros(3, int(state_dim), dtype=torch.float32)
        for idx in range(3):
            theta = theta0 + 2.0 * np.pi * idx / 3.0
            centers[idx, 0] = float(radius * np.cos(theta))
            if int(state_dim) > 1:
                centers[idx, 1] = float(radius * np.sin(theta))
        return centers

    if hasattr(owner, "well_x") and hasattr(owner, "well_y"):
        x = torch.as_tensor(getattr(owner, "well_x"), dtype=torch.float32).reshape(-1)
        y = torch.as_tensor(getattr(owner, "well_y"), dtype=torch.float32).reshape(-1)
        if x.numel() == y.numel() and x.numel() > 0:
            return _coerce_center_matrix(torch.stack([x, y], dim=1), state_dim)
    return None


def _extract_basin_centers(env: object, state_dim: int) -> Optional[torch.Tensor]:
    center_attrs = (
        "points",
        "points_2d",
        "centers",
        "well_centers",
        "room_centers",
        "wells",
        "_wells",
    )
    parts_attrs = (("wells_inner", "wells_outer"),)
    for owner in _label_owners(env):
        special = _special_basin_centers(owner, state_dim)
        if special is not None:
            return special
        for attr in center_attrs:
            if not hasattr(owner, attr):
                continue
            centers = _coerce_center_matrix(getattr(owner, attr), state_dim)
            if centers is not None:
                return centers
        for attrs in parts_attrs:
            if not all(hasattr(owner, attr) for attr in attrs):
                continue
            pieces = [
                _coerce_center_matrix(getattr(owner, attr), state_dim)
                for attr in attrs
            ]
            valid_pieces = [piece for piece in pieces if piece is not None]
            if valid_pieces:
                return torch.cat(valid_pieces, dim=0)
    return None


def _basin_labels_np(env: object, states: np.ndarray) -> np.ndarray:
    """Return benchmark basin labels for an array of states.

    This is intentionally a privileged evaluation/control route. It is only
    valid on benchmark systems that expose ground-truth basin structure.
    """

    state_array = np.asarray(states, dtype=np.float32)
    if state_array.ndim < 2:
        raise ValueError("states must have shape [..., observation_dim]")
    leading_shape = state_array.shape[:-1]
    flat_states = torch.as_tensor(state_array.reshape(-1, state_array.shape[-1]), dtype=torch.float32)
    with torch.no_grad():
        labels = None
        for env_obj in _label_owners(env):
            if hasattr(env_obj, "basin_label"):
                labels = _call_labeler(getattr(env_obj, "basin_label"), flat_states)
                break
            if hasattr(env_obj, "get_basin_label"):
                labels = _call_labeler(getattr(env_obj, "get_basin_label"), flat_states)
                break

        env_obj = _base_env(env)
        if labels is not None:
            pass
        elif env_obj.__class__.__name__ == "Duffing":
            labels = (flat_states[:, 0] >= 0.0).to(dtype=torch.long)
        else:
            points = _extract_basin_centers(env, flat_states.shape[-1])
            if points is None:
                raise ValueError(
                    "oracle_basin requires an environment with basin_label, get_basin_label, "
                    "Duffing state sign labels, or attractor points"
                )
            points = points.to(dtype=flat_states.dtype)
            diff = flat_states.unsqueeze(1) - points.unsqueeze(0)
            labels = torch.linalg.vector_norm(diff, dim=-1).argmin(dim=-1).to(dtype=torch.long)
    if labels.numel() != flat_states.shape[0]:
        raise ValueError(
            f"Basin labeler returned {labels.numel()} labels for {flat_states.shape[0]} states"
        )
    return labels.detach().cpu().numpy().astype(np.int64, copy=False).reshape(leading_shape)


def _route_codebook_from_current_labels(
    fit_latents: np.ndarray,
    current_labels: np.ndarray,
    *,
    full_labels: Optional[np.ndarray],
    min_operator_transitions: int,
    routing_object: str,
    runtime_routing_kind: str,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    x_fit = fit_latents[:, :-1, :].reshape(-1, fit_latents.shape[-1]).astype(np.float32, copy=False)
    labels_flat = np.asarray(current_labels).reshape(-1).astype(np.int64, copy=False)
    if labels_flat.shape[0] != x_fit.shape[0]:
        raise ValueError(
            f"Expected one current route label per source transition, got {labels_flat.shape[0]} "
            f"labels for {x_fit.shape[0]} transitions"
        )
    valid = labels_flat != int(UNCERTAIN)
    counts = Counter(labels_flat[valid].tolist())
    fitted_family_ids = sorted(
        [
            int(family_id)
            for family_id, count in counts.items()
            if int(count) >= int(min_operator_transitions)
        ],
        key=lambda item: str(item),
    )
    centers: Dict[object, np.ndarray] = {}
    for family_id in fitted_family_ids:
        centers[family_id] = x_fit[valid & (labels_flat == int(family_id))].mean(axis=0).astype(
            np.float32,
            copy=False,
        )
    if full_labels is None:
        label_table = np.full(fit_latents.shape[:2], int(UNCERTAIN), dtype=np.int64)
        label_table[:, :-1] = labels_flat.reshape(fit_latents.shape[0], fit_latents.shape[1] - 1)
    else:
        label_table = np.asarray(full_labels, dtype=np.int64)
    route_codebook: Dict[str, object] = {
        "family_labels": label_table,
        "family_counts": counts,
        "fitted_family_ids": fitted_family_ids,
        "centers": centers,
        "family_prototypes": {},
        "support_key_to_family": {},
        "routing_object": routing_object,
        "runtime_routing_kind": runtime_routing_kind,
    }
    if runtime_routing_kind == "nearest_latent_center":
        route_codebook["runtime_family_ids"] = list(fitted_family_ids)
        if fitted_family_ids:
            route_codebook["runtime_latent_centers"] = np.stack(
                [centers[family_id] for family_id in fitted_family_ids],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            route_codebook["runtime_latent_centers"] = np.empty(
                (0, fit_latents.shape[-1]),
                dtype=np.float32,
            )
    if extra_metadata:
        route_codebook.update(extra_metadata)
    return route_codebook


def _build_oracle_basin_route_codebook(
    fit_latents: np.ndarray,
    fit_states: np.ndarray,
    *,
    env: object,
    min_operator_transitions: int,
) -> Dict[str, object]:
    full_labels = _basin_labels_np(env, fit_states)
    return _route_codebook_from_current_labels(
        fit_latents,
        full_labels[:, :-1],
        full_labels=full_labels,
        min_operator_transitions=min_operator_transitions,
        routing_object="oracle_basin",
        runtime_routing_kind="oracle_basin_label",
        extra_metadata={"oracle_label_source": "benchmark_basin_label"},
    )


def _build_latent_kmeans_route_codebook(
    fit_latents: np.ndarray,
    *,
    n_clusters: int,
    min_operator_transitions: int,
    seed: int,
    n_init: int,
    cluster_count_source: str,
) -> Dict[str, object]:
    if int(n_clusters) < 1:
        raise ValueError("latent_kmeans requires at least one cluster")
    x_fit = fit_latents[:, :-1, :].reshape(-1, fit_latents.shape[-1]).astype(np.float32, copy=False)
    if x_fit.shape[0] < int(n_clusters):
        raise ValueError(
            f"latent_kmeans requested {n_clusters} clusters from only {x_fit.shape[0]} fit transitions"
        )
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init=max(1, int(n_init)))
    current_labels = kmeans.fit_predict(x_fit).astype(np.int64, copy=False)
    full_labels = kmeans.predict(
        fit_latents.reshape(-1, fit_latents.shape[-1]).astype(np.float32, copy=False)
    ).reshape(fit_latents.shape[:2]).astype(np.int64, copy=False)
    return _route_codebook_from_current_labels(
        fit_latents,
        current_labels,
        full_labels=full_labels,
        min_operator_transitions=min_operator_transitions,
        routing_object="latent_kmeans",
        runtime_routing_kind="nearest_latent_center",
        extra_metadata={
            "baseline_cluster_count_source": cluster_count_source,
            "latent_cluster_count": int(n_clusters),
            "baseline_route_seed": int(seed),
            "baseline_kmeans_n_init": int(n_init),
        },
    )


def _latent_fate_tail_features(latents: np.ndarray, tail_window: int) -> np.ndarray:
    window = max(1, min(int(tail_window), latents.shape[1]))
    tail = latents[:, -window:, :]
    mean = tail.mean(axis=1)
    std = tail.std(axis=1)
    final = tail[:, -1, :]
    return np.concatenate([mean, std, final], axis=1).astype(np.float32, copy=False)


def _preprocess_latent_fate_features(features: np.ndarray, pca_components: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(features)
    n_components = min(int(pca_components), scaled.shape[0] - 1, scaled.shape[1])
    if n_components >= 2:
        return PCA(n_components=n_components, random_state=0).fit_transform(scaled)
    return scaled


def _latent_fate_kmeans_labels(features: np.ndarray, k: int, seed: int, n_init: int) -> np.ndarray:
    if int(k) <= 1:
        return np.zeros(features.shape[0], dtype=np.int64)
    from sklearn.cluster import KMeans

    return KMeans(
        n_clusters=int(k),
        random_state=int(seed),
        n_init=max(1, int(n_init)),
    ).fit_predict(features).astype(np.int64, copy=False)


def _select_latent_tail_fate_labels(
    features: np.ndarray,
    *,
    max_clusters: int,
    min_silhouette: float,
    seed: int,
    n_init: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    from sklearn.metrics import silhouette_score

    max_k = min(int(max_clusters), features.shape[0] - 1)
    best_score = -1.0
    best_labels = np.zeros(features.shape[0], dtype=np.int64)
    best_k = 1
    scores: Dict[int, float] = {}
    for k in range(2, max_k + 1):
        labels = _latent_fate_kmeans_labels(features, k, seed, n_init)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(silhouette_score(features, labels))
        scores[int(k)] = score
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = int(k)
    if best_score < float(min_silhouette):
        return np.zeros(features.shape[0], dtype=np.int64), {
            "selected_k": 1,
            "silhouette": best_score if best_score >= 0.0 else None,
            "silhouette_scores": scores,
            "selection_rule": "silhouette_below_threshold",
        }
    return best_labels, {
        "selected_k": best_k,
        "silhouette": best_score,
        "silhouette_scores": scores,
        "selection_rule": "best_silhouette",
    }


def _build_latent_tail_fate_route_codebook(
    fit_latents: np.ndarray,
    *,
    min_operator_transitions: int,
    seed: int,
    n_init: int,
    tail_window: int,
    max_clusters: int,
    min_silhouette: float,
    pca_components: int,
) -> Dict[str, object]:
    features = _preprocess_latent_fate_features(
        _latent_fate_tail_features(fit_latents, tail_window),
        pca_components,
    )
    trajectory_labels, info = _select_latent_tail_fate_labels(
        features,
        max_clusters=max_clusters,
        min_silhouette=min_silhouette,
        seed=seed,
        n_init=n_init,
    )
    full_labels = np.repeat(
        trajectory_labels[:, None],
        fit_latents.shape[1],
        axis=1,
    ).astype(np.int64, copy=False)
    codebook = _route_codebook_from_current_labels(
        fit_latents,
        full_labels[:, :-1],
        full_labels=full_labels,
        min_operator_transitions=min_operator_transitions,
        routing_object="latent_tail_fate",
        runtime_routing_kind="nearest_latent_center",
        extra_metadata={
            "latent_tail_fate_feature_kind": "tail_mean_std_final",
            "latent_tail_fate_tail_window": int(tail_window),
            "latent_tail_fate_pca_components": int(features.shape[1]),
            "latent_tail_fate_max_clusters": int(max_clusters),
            "latent_tail_fate_min_silhouette": float(min_silhouette),
            "latent_tail_fate_selected_k": int(info["selected_k"]),
            "latent_tail_fate_selection_rule": str(info["selection_rule"]),
            "latent_tail_fate_silhouette": info["silhouette"],
            "latent_tail_fate_silhouette_scores": info["silhouette_scores"],
            "baseline_route_seed": int(seed),
            "baseline_kmeans_n_init": int(n_init),
        },
    )
    return codebook


def _build_random_matched_route_codebook(
    fit_latents: np.ndarray,
    reference_codebook: Dict[str, object],
    *,
    min_operator_transitions: int,
    seed: int,
) -> Dict[str, object]:
    ref_labels = np.asarray(reference_codebook["family_labels"])[:, :-1].reshape(-1)
    ref_fitted = {int(item) for item in reference_codebook["fitted_family_ids"]}
    valid_ref = np.asarray(
        [int(label) in ref_fitted for label in ref_labels.tolist()],
        dtype=bool,
    )
    ref_counts = Counter(int(label) for label in ref_labels[valid_ref].tolist())
    if not ref_counts:
        raise RuntimeError("random_matched requires at least one fitted reference C_stab route")
    shuffled = np.full(ref_labels.shape[0], int(UNCERTAIN), dtype=np.int64)
    valid_indices = np.flatnonzero(valid_ref)
    rng = np.random.default_rng(int(seed))
    permuted_indices = rng.permutation(valid_indices)
    cursor = 0
    for family_id in sorted(ref_counts.keys(), key=lambda item: str(item)):
        count = int(ref_counts[family_id])
        selected = permuted_indices[cursor : cursor + count]
        shuffled[selected] = int(family_id)
        cursor += count
    return _route_codebook_from_current_labels(
        fit_latents,
        shuffled,
        full_labels=None,
        min_operator_transitions=min_operator_transitions,
        routing_object="random_matched",
        runtime_routing_kind="nearest_latent_center",
        extra_metadata={
            "random_matched_source": "stable_support_component",
            "reference_route_count": int(len(ref_counts)),
            "baseline_route_seed": int(seed),
        },
    )


def _nearest_latent_center_route_indices_for_torch(
    z: torch.Tensor,
    *,
    route_codebook: Dict[str, object],
    family_to_index: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    family_ids = route_codebook.get("runtime_family_ids") or route_codebook["fitted_family_ids"]
    centers = route_codebook.get("runtime_latent_centers")
    if centers is None:
        centers = np.stack(
            [route_codebook["centers"][family_id] for family_id in family_ids],
            axis=0,
        ).astype(np.float32, copy=False)
    if len(family_ids) == 0:
        return torch.full((z.shape[0],), -1, dtype=torch.long, device=device)
    centers_tensor = torch.as_tensor(centers, dtype=z.dtype, device=device)
    route_indices = torch.as_tensor(
        [int(family_to_index.get(str(family_id), -1)) for family_id in family_ids],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        distances = torch.cdist(z.detach(), centers_tensor)
        nearest = distances.argmin(dim=1)
    return route_indices[nearest]


def _oracle_basin_route_indices_for_torch(
    z: torch.Tensor,
    *,
    model: nn.Module,
    route_env: object,
    family_to_index: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    if route_env is None:
        raise RuntimeError("oracle_basin routing requires route_env")
    with torch.no_grad():
        decoded = model.decode(z.detach()).detach().cpu().numpy().astype(np.float32, copy=False)
        labels = _basin_labels_np(route_env, decoded).reshape(-1)
    route_np = np.full(labels.shape[0], -1, dtype=np.int64)
    for idx, label in enumerate(labels.tolist()):
        mapped = family_to_index.get(str(int(label)))
        if mapped is not None:
            route_np[idx] = int(mapped)
    return torch.from_numpy(route_np).to(device=device, dtype=torch.long)


def _route_indices_for_torch(
    z: torch.Tensor,
    *,
    model: nn.Module,
    route_env: object,
    route_codebook: Dict[str, object],
    scheme: str,
    support_value: float,
    family_jaccard_threshold: float,
    family_to_index: Dict[str, int],
    family_cache: Dict[object, object],
    device: torch.device,
) -> torch.Tensor:
    routing_kind = _runtime_routing_kind(route_codebook)
    if routing_kind == "support_jaccard":
        return _step_routes_for_torch(
            z,
            scheme=scheme,
            value=support_value,
            family_jaccard_threshold=family_jaccard_threshold,
            support_key_to_family=route_codebook["support_key_to_family"],
            family_prototypes=route_codebook["family_prototypes"],
            family_to_index=family_to_index,
            family_cache=family_cache,
            device=device,
        )
    if routing_kind == "nearest_latent_center":
        return _nearest_latent_center_route_indices_for_torch(
            z,
            route_codebook=route_codebook,
            family_to_index=family_to_index,
            device=device,
        )
    if routing_kind == "oracle_basin_label":
        return _oracle_basin_route_indices_for_torch(
            z,
            model=model,
            route_env=route_env,
            family_to_index=family_to_index,
            device=device,
        )
    raise ValueError(f"Unsupported runtime routing kind {routing_kind!r}")


def _learn_target_centers(parameterization: str) -> bool:
    return parameterization == "source_target_affine_learned_intercept"


def _target_center_rule(parameterization: str) -> str:
    if _learn_target_centers(parameterization):
        return "learned target center initialized as source_center @ frozen_global_k"
    return "fixed source_center @ frozen_global_k"


class SourceTargetLocalMapBundle(nn.Module):
    """Affine local charts initialized to reproduce the frozen global map."""

    def __init__(
        self,
        *,
        family_ids: Sequence[object],
        source_centers: Dict[object, np.ndarray],
        target_centers: Dict[object, np.ndarray],
        global_k: np.ndarray,
        device: str,
        learn_target_centers: bool = False,
    ) -> None:
        super().__init__()
        self.family_ids = [str(item) for item in family_ids]
        self.family_to_index = {family_id: idx for idx, family_id in enumerate(self.family_ids)}
        source_array = np.stack([source_centers[item] for item in family_ids], axis=0).astype(
            np.float32,
            copy=False,
        )
        target_array = np.stack([target_centers[item] for item in family_ids], axis=0).astype(
            np.float32,
            copy=False,
        )
        self.register_buffer("source_centers", torch.from_numpy(source_array).to(device=device))
        target_tensor = torch.from_numpy(target_array).to(device=device)
        if learn_target_centers:
            self.target_centers = nn.Parameter(target_tensor)
        else:
            self.register_buffer("target_centers", target_tensor)
        init_k = torch.from_numpy(global_k.astype(np.float32, copy=False)).to(device=device)
        self.local_maps = nn.Parameter(init_k.unsqueeze(0).repeat(len(self.family_ids), 1, 1))
        self.register_buffer("global_k", init_k)

    def forward(self, z: torch.Tensor, route_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = route_index >= 0
        out = z @ self.global_k
        if bool(valid.any()):
            selected = route_index[valid]
            source_centers = self.source_centers[selected]
            target_centers = self.target_centers[selected]
            maps = self.local_maps[selected]
            z_valid = z[valid]
            out[valid] = target_centers + torch.bmm(
                (z_valid - source_centers).unsqueeze(1),
                maps,
            ).squeeze(1)
        return out, valid


class StagedLocalKoopmanWrapper(nn.Module):
    """Evaluation wrapper that routes latent steps through trained local K_c maps."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        local_bundle: SourceTargetLocalMapBundle,
        route_codebook: Dict[str, object],
        route_env: object,
        scheme: str,
        support_value: float,
        family_jaccard_threshold: float,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.local_bundle = local_bundle
        self.route_codebook = route_codebook
        self.route_env = route_env
        self.scheme = scheme
        self.support_value = float(support_value)
        self.family_jaccard_threshold = float(family_jaccard_threshold)
        self.family_cache: Dict[object, object] = {}
        self.cfg = getattr(base_model, "cfg")
        self.observation_size = int(getattr(base_model, "observation_size"))
        self.target_size = int(getattr(base_model, "target_size"))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model.encode(x)

    def encode_with_prior(
        self,
        x: torch.Tensor,
        latent_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.base_model.encode_with_prior(x, latent_prior=latent_prior)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.base_model.decode(z)

    def kmatrix(self) -> torch.Tensor:
        return self.local_bundle.global_k

    def step_latent(self, z: torch.Tensor) -> torch.Tensor:
        route_index = _route_indices_for_torch(
            z,
            model=self.base_model,
            route_env=self.route_env,
            route_codebook=self.route_codebook,
            scheme=self.scheme,
            support_value=self.support_value,
            family_jaccard_threshold=self.family_jaccard_threshold,
            family_to_index=self.local_bundle.family_to_index,
            family_cache=self.family_cache,
            device=z.device,
        )
        z_next, _used_local = self.local_bundle(z, route_index)
        return z_next

    def step_env(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.step_latent(self.encode(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step_env(x)


def _make_wrapped_model(
    model: nn.Module,
    bundle: SourceTargetLocalMapBundle,
    route_codebook: Dict[str, object],
    *,
    route_env: object,
    scheme: str,
    support_value: float,
    family_jaccard_threshold: float,
) -> StagedLocalKoopmanWrapper:
    wrapped = StagedLocalKoopmanWrapper(
        base_model=model,
        local_bundle=bundle,
        route_codebook=route_codebook,
        route_env=route_env,
        scheme=scheme,
        support_value=support_value,
        family_jaccard_threshold=family_jaccard_threshold,
    )
    return wrapped.to(next(model.parameters()).device)


def _local_train_step(
    *,
    model: nn.Module,
    bundle: SourceTargetLocalMapBundle,
    route_codebook: Dict[str, object],
    route_env: object,
    x_seq: torch.Tensor,
    scheme: str,
    support_value: float,
    family_jaccard_threshold: float,
    optimizer: torch.optim.Optimizer,
    family_cache: Dict[object, object],
    step: int,
) -> Dict[str, float]:
    model.eval()
    bundle.train()
    optimizer.zero_grad()

    if x_seq.ndim != 3 or x_seq.shape[1] < 2:
        raise ValueError("x_seq must have shape [batch, horizon+1, obs]")
    batch_size, seq_len, obs_dim = x_seq.shape
    horizon = seq_len - 1
    x_true = x_seq[:, 1:, :]

    with torch.no_grad():
        z_all = model.encode(x_seq.reshape(batch_size * seq_len, obs_dim)).reshape(batch_size, seq_len, -1)
        z_true = z_all[:, 1:, :]
        x_recon_true = model.decode(z_true.reshape(batch_size * horizon, -1)).reshape(
            batch_size, horizon, obs_dim
        )
        reconstruction_error = torch.norm(x_true - x_recon_true, dim=-1).mean()

    z = z_all[:, 0, :]
    z_preds: List[torch.Tensor] = []
    used_local: List[torch.Tensor] = []
    route_counts = torch.zeros(len(bundle.family_ids), dtype=torch.long, device=z.device)
    fallback_count = torch.zeros((), dtype=torch.long, device=z.device)
    for _ in range(horizon):
        route_index = _route_indices_for_torch(
            z,
            model=model,
            route_env=route_env,
            route_codebook=route_codebook,
            scheme=scheme,
            support_value=support_value,
            family_jaccard_threshold=family_jaccard_threshold,
            family_to_index=bundle.family_to_index,
            family_cache=family_cache,
            device=z.device,
        )
        valid_route = route_index >= 0
        if bool(valid_route.any()):
            route_counts += torch.bincount(
                route_index[valid_route],
                minlength=len(bundle.family_ids),
            )[: len(bundle.family_ids)]
        fallback_count += (~valid_route).sum()
        z, used = bundle(z, route_index)
        z_preds.append(z)
        used_local.append(used.float())

    z_pred = torch.stack(z_preds, dim=1)
    x_pred = model.decode(z_pred.reshape(batch_size * horizon, -1)).reshape(batch_size, horizon, obs_dim)
    loss, metrics = model.loss(
        x_pred=x_pred,
        x_true=x_true,
        x0=x_seq[:, 0, :],
        z0=z_all[:, 0, :],
        z_pred=z_pred,
        z_true=z_true,
        reconstruction_error=reconstruction_error,
        sparsity_latent=z_pred,
        step=step,
    )
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        metrics["route_coverage"] = float(torch.stack(used_local, dim=1).mean().detach().cpu().item())
        metrics["fallback_fraction"] = 1.0 - metrics["route_coverage"]
        metrics["route_total_count"] = float(batch_size * horizon)
        metrics["route_fallback_count"] = float(fallback_count.detach().cpu().item())
        for idx, family_id in enumerate(bundle.family_ids):
            metrics[f"route_family_{family_id}_count"] = float(route_counts[idx].detach().cpu().item())
    return metrics


def _save_checkpoint(
    path: Path,
    *,
    stage: str,
    next_step: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    bundle: Optional[SourceTargetLocalMapBundle],
    local_optimizer: Optional[torch.optim.Optimizer],
    best_eval_final_error: float,
    metrics: Dict[str, float],
    cfg: Config,
    route_metadata: Optional[Dict[str, object]] = None,
    route_codebook: Optional[Dict[str, object]] = None,
    target_centers: Optional[Dict[object, np.ndarray]] = None,
    support_batches: Optional[Sequence[torch.Tensor]] = None,
    include_optimizer_state: bool = False,
) -> None:
    payload: Dict[str, object] = {
        "stage": stage,
        "next_step": int(next_step),
        "step": int(next_step) - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if include_optimizer_state and optimizer is not None
            else None
        ),
        "local_bundle_state_dict": bundle.state_dict() if bundle is not None else None,
        "local_optimizer_state_dict": (
            local_optimizer.state_dict()
            if include_optimizer_state and local_optimizer is not None
            else None
        ),
        "best_eval_final_error": float(best_eval_final_error),
        "metrics": dict(metrics),
        "config": cfg.to_dict(),
        "route_metadata": route_metadata or {},
        "route_codebook": route_codebook,
        "target_centers": target_centers,
        "support_batches": [batch.cpu() for batch in support_batches] if support_batches else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def _write_torch(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _log_phase(run_dir: Path, phase: str, *, device: str, **payload: object) -> None:
    record: Dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        **payload,
    }
    if str(device) == "cuda" and torch.cuda.is_available():
        record["cuda_device_name"] = torch.cuda.get_device_name(0)
        record["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1_000_000, 3)
        record["cuda_max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1_000_000, 3)
    phase_path = run_dir / "phase_status.jsonl"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    with phase_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    details = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"[phase] {phase} {details}".rstrip(), flush=True)


def _route_codebook_metadata(route_codebook: Dict[str, object]) -> Dict[str, object]:
    prototypes = route_codebook.get("family_prototypes", {})
    metadata = {
        "family_counts": {str(key): int(value) for key, value in route_codebook["family_counts"].items()},
        "fitted_family_ids": [str(item) for item in route_codebook["fitted_family_ids"]],
        "family_class_count_total": int(len(route_codebook["family_counts"])),
        "family_class_count_fit": int(len(route_codebook["fitted_family_ids"])),
        "family_prototypes": {
            str(key): np.asarray(value, dtype=bool).astype(int).tolist()
            for key, value in prototypes.items()
        },
    }
    for key in (
        "routing_object",
        "runtime_routing_kind",
        "route_jaccard_threshold",
        "stable_base_object",
        "stable_base_family_jaccard",
        "stable_diagnostics",
        "oracle_label_source",
        "baseline_cluster_count_source",
        "latent_cluster_count",
        "baseline_route_seed",
        "baseline_kmeans_n_init",
        "latent_tail_fate_feature_kind",
        "latent_tail_fate_tail_window",
        "latent_tail_fate_pca_components",
        "latent_tail_fate_max_clusters",
        "latent_tail_fate_min_silhouette",
        "latent_tail_fate_selected_k",
        "latent_tail_fate_selection_rule",
        "latent_tail_fate_silhouette",
        "latent_tail_fate_silhouette_scores",
        "random_matched_source",
        "reference_route_count",
    ):
        if key in route_codebook:
            metadata[key] = route_codebook[key]
    return metadata


def _find_resume_run(seed_dir: Path) -> Optional[Path]:
    candidates = [
        path
        for path in seed_dir.glob("20*")
        if path.is_dir()
        and ((path / "last.pt").exists() or (path / "checkpoint.pt").exists())
        and not (path / "evaluation_results_best.json").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (p.name, str(p)))[-1]


def _find_completed_run(seed_dir: Path) -> Optional[Path]:
    candidates = [
        path
        for path in seed_dir.glob("20*")
        if path.is_dir() and (path / "evaluation_results_best.json").exists()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (p.name, str(p)))[-1]


def _make_eval_settings(
    profile: str,
    cfg: Config,
    *,
    periodic_periods_override: Sequence[int] = (),
    save_rollout_artifacts: bool = False,
    save_plots: bool = False,
    include_per_ic_values: bool = False,
    include_error_curves: bool = False,
) -> EvaluationSettings:
    settings = EvaluationSettings()
    settings.systems = [cfg.ENV.ENV_NAME]
    settings.save_rollout_artifacts = bool(save_rollout_artifacts)
    settings.save_plots = bool(save_plots)
    settings.include_per_ic_values = bool(include_per_ic_values)
    settings.include_error_curves = bool(include_error_curves)
    if profile == "smoke":
        settings.batch_size = 32
        settings.horizons = (100, 500, 1000)
        settings.periodic_reencode_periods = (10, 25)
        settings.phase_portrait_samples = 8
        settings.phase_portrait_length = 100
        settings.phase_portrait_batch_size = 64
        settings.phase_portrait_reencode_periods = (0, 1, 10, 25)
    if periodic_periods_override:
        settings.periodic_reencode_periods = tuple(int(period) for period in periodic_periods_override)
        settings.phase_portrait_reencode_periods = tuple(
            dict.fromkeys((0, 1, *settings.periodic_reencode_periods))
        )
    return settings


def _quick_eval_final_error(
    eval_model: nn.Module,
    *,
    val_x: torch.Tensor,
    eval_env: VectorWrapper,
    num_steps: int,
) -> float:
    eval_model.eval()
    device = next(eval_model.parameters()).device
    with torch.no_grad():
        true_traj = generate_trajectory(lambda state: eval_env.step(state), val_x.cpu(), length=num_steps)
        latent = eval_model.encode(val_x.to(device))
        predictions: List[torch.Tensor] = []
        for _ in range(num_steps):
            latent = eval_model.step_latent(latent)
            pred = eval_model.decode(latent)
            predictions.append(pred)
            latent = eval_model.encode(pred)
        pred_traj = torch.stack(predictions, dim=0).detach().cpu()
        step_error = torch.norm(pred_traj - true_traj, dim=-1)
        final_error = torch.nanmean(step_error[-1])
    return float(final_error.item())


def _quick_eval_best_periodic_mse(
    eval_model: nn.Module,
    *,
    val_x: torch.Tensor,
    eval_env: VectorWrapper,
    num_steps: int,
    periods: Sequence[int],
) -> Tuple[float, int]:
    eval_model.eval()
    device = next(eval_model.parameters()).device
    true_traj = generate_trajectory(lambda state: eval_env.step(state), val_x.cpu(), length=num_steps)
    best_value = float("inf")
    best_period = -1
    with torch.no_grad():
        for period in periods:
            latent = eval_model.encode(val_x.to(device))
            predictions: List[torch.Tensor] = []
            for step in range(num_steps):
                latent_pred = eval_model.step_latent(latent)
                pred = eval_model.decode(latent_pred)
                predictions.append(pred)
                if (step + 1) % int(period) == 0:
                    latent = eval_model.encode(pred)
                else:
                    latent = latent_pred
            pred_traj = torch.stack(predictions, dim=0).detach().cpu()
            squared_error = torch.sum((pred_traj - true_traj) ** 2, dim=-1)
            score = float(torch.nanmean(squared_error).item())
            if math.isfinite(score) and score < best_value:
                best_value = score
                best_period = int(period)
    return best_value, best_period


def _quick_eval_best_periodic_horizon_mse(
    eval_model: nn.Module,
    *,
    val_x: torch.Tensor,
    eval_env: VectorWrapper,
    horizons: Sequence[int],
    periods: Sequence[int],
) -> Tuple[float, Dict[int, Tuple[float, int]]]:
    eval_model.eval()
    clean_horizons = sorted({int(horizon) for horizon in horizons if int(horizon) > 0})
    if not clean_horizons:
        raise ValueError("--stage2_selection_horizons must include at least one positive horizon")
    max_horizon = max(clean_horizons)
    device = next(eval_model.parameters()).device
    true_traj = generate_trajectory(lambda state: eval_env.step(state), val_x.cpu(), length=max_horizon)
    best_by_horizon: Dict[int, Tuple[float, int]] = {
        horizon: (float("inf"), -1) for horizon in clean_horizons
    }

    with torch.no_grad():
        for period in periods:
            latent = eval_model.encode(val_x.to(device))
            predictions: List[torch.Tensor] = []
            for step in range(max_horizon):
                latent_pred = eval_model.step_latent(latent)
                pred = eval_model.decode(latent_pred)
                predictions.append(pred)
                if (step + 1) % int(period) == 0:
                    latent = eval_model.encode(pred)
                else:
                    latent = latent_pred
            pred_traj = torch.stack(predictions, dim=0).detach().cpu()
            squared_error = torch.sum((pred_traj - true_traj) ** 2, dim=-1)
            squared_error = torch.where(torch.isfinite(squared_error), squared_error, torch.nan)
            for horizon in clean_horizons:
                per_ic = torch.nanmean(squared_error[:horizon], dim=0)
                finite = per_ic[torch.isfinite(per_ic)]
                if finite.numel() == 0:
                    continue
                score = float(finite.mean().item())
                if math.isfinite(score) and score < best_by_horizon[horizon][0]:
                    best_by_horizon[horizon] = (score, int(period))

    finite_scores = [
        value for value, _period in best_by_horizon.values() if math.isfinite(value) and value > 0.0
    ]
    if not finite_scores:
        return float("inf"), best_by_horizon
    aggregate = float(sum(finite_scores) / len(finite_scores))
    return aggregate, best_by_horizon


def main() -> None:
    args = _parse_args()
    row = _read_task_row(
        Path(args.task_tsv),
        array_index=int(args.array_index),
        array_offset=int(args.array_offset),
    )
    cfg = _apply_task_row_to_config(row)
    if args.num_steps_override is not None:
        cfg.TRAIN.NUM_STEPS = int(args.num_steps_override)
    if args.eval_every_override is not None:
        cfg.TRAIN.EVAL_EVERY = int(args.eval_every_override)
    if args.eval_num_steps_override is not None:
        cfg.TRAIN.EVAL_NUM_STEPS = int(args.eval_num_steps_override)
    total_steps = int(cfg.TRAIN.NUM_STEPS)
    if total_steps < 2:
        raise ValueError("Staged local-K training requires at least two total steps")
    stage1_steps = (
        int(args.stage1_steps_override)
        if args.stage1_steps_override is not None
        else total_steps // 2
    )
    if stage1_steps <= 0 or stage1_steps >= total_steps:
        raise ValueError("--stage1_steps_override must be between 1 and total_steps - 1")
    stage2_start = stage1_steps
    scheme, support_value = _support_definition(args.support_definition)
    stage2_selection_periods = _parse_int_csv(args.stage2_selection_periods)
    stage2_selection_horizons = _parse_int_csv(args.stage2_selection_horizons)
    eval_periodic_periods = _parse_int_csv(args.eval_periodic_periods_override)
    device = get_device(args.device)

    phase = _safe_str(row.get("phase")) or "transition_rich_basin_partition"
    model_variant = _safe_str(row.get("model_variant")) or "lista_fabs_local_k_staged"
    system_slug = _safe_str(row.get("system_slug")) or (_safe_str(row.get("system_key")).replace(":", "_"))
    seed = int(cfg.SEED)
    env_dt = _maybe_float(row, "env_dt")
    dt_tag = _tagify(env_dt if env_dt is not None else "default")
    seed_dir = Path(args.base_out) / phase / model_variant / system_slug / f"dt_{dt_tag}" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    completed_run = _find_completed_run(seed_dir)
    if args.skip_completed and completed_run is not None:
        print(f"Completed staged run already exists: {completed_run}", flush=True)
        return

    resume_run = _find_resume_run(seed_dir) if args.resume_from_latest else None
    run_dir = resume_run or (seed_dir / datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(str(run_dir / "config.json"))
    _log_phase(
        run_dir,
        "init",
        device=device,
        task_id=row.get("task_id"),
        system=cfg.ENV.ENV_NAME,
        seed=seed,
        total_steps=total_steps,
        stage1_steps=stage1_steps,
        routing_object=args.routing_object,
    )
    _write_json(
        run_dir / "staged_local_k_config.json",
        {
            "task_tsv": str(args.task_tsv),
            "task_id": row.get("task_id"),
            "source_model_variant": row.get("model_variant"),
            "total_steps": total_steps,
            "stage1_joint_steps": stage1_steps,
            "stage2_local_steps": total_steps - stage1_steps,
            "support_definition": args.support_definition,
            "family_jaccard_threshold": float(args.family_jaccard_threshold),
            "support_fit_batches": int(args.support_fit_batches),
            "min_family_transitions": int(args.min_family_transitions),
            "support_family_fit_source": args.support_family_fit_source,
            "routing_object": args.routing_object,
            "stable_base_object": args.stable_base_object,
            "stable_base_family_jaccard": float(args.stable_base_family_jaccard),
            "stable_tail_window": int(args.stable_tail_window),
            "stable_min_edge_count": int(args.stable_min_edge_count),
            "stable_min_edge_probability": float(args.stable_min_edge_probability),
            "stable_max_recurrent_out_probability": float(args.stable_max_recurrent_out_probability),
            "stable_min_tail_count": int(args.stable_min_tail_count),
            "stable_min_absorption_observations": int(args.stable_min_absorption_observations),
            "stable_min_absorption_confidence": float(args.stable_min_absorption_confidence),
            "stable_fit_trajectories": int(args.stable_fit_trajectories),
            "stable_fit_trajectory_length": int(args.stable_fit_trajectory_length),
            "stable_fit_seed_offset": int(args.stable_fit_seed_offset),
            "baseline_route_seed_offset": int(args.baseline_route_seed_offset),
            "baseline_latent_cluster_count": int(args.baseline_latent_cluster_count),
            "baseline_kmeans_n_init": int(args.baseline_kmeans_n_init),
            "latent_fate_tail_window": int(args.latent_fate_tail_window),
            "latent_fate_max_clusters": int(args.latent_fate_max_clusters),
            "latent_fate_min_silhouette": float(args.latent_fate_min_silhouette),
            "latent_fate_pca_components": int(args.latent_fate_pca_components),
            "local_map_parameterization": args.local_map_parameterization,
            "local_lr": args.local_lr,
            "stage2_selection_metric": args.stage2_selection_metric,
            "stage2_selection_periods": list(stage2_selection_periods),
            "stage2_selection_horizons": list(stage2_selection_horizons),
            "stage2_selection_batch_size": int(args.stage2_selection_batch_size),
            "stage2_selection_seed_offset": int(args.stage2_selection_seed_offset),
            "eval_periodic_periods_override": list(eval_periodic_periods),
            "save_metrics_history": bool(args.save_metrics_history),
            "save_last_checkpoint": bool(args.save_last_checkpoint),
            "save_stage2_artifacts": bool(args.save_stage2_artifacts),
            "save_eval_rollout_artifacts": bool(args.save_eval_rollout_artifacts),
            "save_eval_plots": bool(args.save_eval_plots),
            "save_eval_per_ic_values": bool(args.save_eval_per_ic_values),
            "save_eval_error_curves": bool(args.save_eval_error_curves),
            "device": device,
        },
    )

    print("=============================================", flush=True)
    print("Staged support-family local-K training", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"System: {cfg.ENV.ENV_NAME}", flush=True)
    print(f"Seed: {cfg.SEED}", flush=True)
    print(f"Total steps: {total_steps} ({stage1_steps} joint + {total_steps - stage1_steps} local)", flush=True)
    print(
        f"Support: {args.support_definition}, route={args.routing_object}, "
        f"J={args.family_jaccard_threshold}",
        flush=True,
    )
    print(f"Device: {device}", flush=True)
    print("=============================================", flush=True)

    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)

    _log_phase(run_dir, "construct_env_model_start", device=device)
    base_env = make_env(cfg)
    train_env = VectorWrapper(wrap_training_env(base_env, cfg), cfg.TRAIN.BATCH_SIZE)
    eval_env = VectorWrapper(base_env, cfg.TRAIN.BATCH_SIZE)
    model = make_model(cfg, base_env.observation_size).to(device)
    optimizer = build_optimizer(model, cfg)
    logger = MetricsLogger(run_dir, save_history=bool(args.save_metrics_history))
    _log_phase(
        run_dir,
        "construct_env_model_end",
        device=device,
        observation_size=base_env.observation_size,
        batch_size=cfg.TRAIN.BATCH_SIZE,
    )

    if args.stage2_selection_metric == "best_periodic_horizon_mse":
        selection_batch_size = max(1, int(args.stage2_selection_batch_size))
        selection_rng = torch.Generator().manual_seed(cfg.SEED + int(args.stage2_selection_seed_offset))
        selection_env = VectorWrapper(base_env, selection_batch_size)
        val_x = selection_env.reset(selection_rng).to(device)
    else:
        val_rng = torch.Generator().manual_seed(cfg.SEED + 999999)
        val_seq = generate_sequence_batch_for_device(
            eval_env,
            val_rng,
            window_length=max(1, cfg.TRAIN.SEQUENCE_LENGTH),
            device=device,
        )
        val_x = val_seq[:16, 0, :]

    num_batches = max(1, cfg.TRAIN.DATA_SIZE // cfg.TRAIN.BATCH_SIZE)
    rngs = [torch.Generator().manual_seed(cfg.SEED + i * cfg.TRAIN.BATCH_SIZE) for i in range(num_batches)]

    start_step = 0
    best_eval_final_error = float("inf")
    support_buffer: Deque[torch.Tensor] = deque(maxlen=max(1, int(args.support_fit_batches)))
    route_codebook: Optional[Dict[str, object]] = None
    bundle: Optional[SourceTargetLocalMapBundle] = None
    local_optimizer: Optional[torch.optim.Optimizer] = None
    local_family_cache: Dict[object, object] = {}
    route_metadata: Dict[str, object] = {}
    target_centers: Optional[Dict[object, np.ndarray]] = None
    resume_payload: Optional[Dict[str, object]] = None

    last_path = run_dir / "last.pt"
    checkpoint_path = run_dir / "checkpoint.pt"
    stage2_artifact_path = run_dir / "stage2_artifacts.pt"
    resume_path: Optional[Path] = None
    if args.resume_from_latest:
        if last_path.exists():
            resume_path = last_path
        elif checkpoint_path.exists():
            resume_path = checkpoint_path
    if resume_path is not None:
        try:
            payload = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(resume_path, map_location=device)
        resume_payload = payload
        model.load_state_dict(payload["model_state_dict"])
        start_step = int(payload.get("next_step", 0))
        best_eval_final_error = float(payload.get("best_eval_final_error", float("inf")))
        for batch in payload.get("support_batches", []) or []:
            support_buffer.append(batch.cpu())
        if start_step < stage2_start and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        print(
            f"Resuming staged run at step {start_step}/{total_steps} from {resume_path.name}",
            flush=True,
        )

    last_metrics: Dict[str, float] = {}
    _log_phase(
        run_dir,
        "stage1_start",
        device=device,
        start_step=start_step,
        end_step=stage2_start,
    )
    for step in range(start_step, stage2_start):
        rng = rngs[step % num_batches]
        x_seq = generate_sequence_batch_for_device(
            train_env,
            rng,
            window_length=cfg.TRAIN.SEQUENCE_LENGTH,
            device=device,
        )
        support_buffer.append(x_seq.detach().cpu())
        metrics = train_step(model, optimizer, x_seq, step=step)
        last_metrics = metrics
        logger.log_dict(metrics, step, prefix="stage1_train")

        if step % 100 == 0:
            print(
                f"Stage 1 step {step}/{stage1_steps} "
                f"loss={metrics['loss']:.6g} pred={metrics['prediction_loss']:.6g}",
                flush=True,
            )
        if (step > 0 and step % cfg.TRAIN.EVAL_EVERY == 0) or step == stage2_start - 1:
            if args.save_last_checkpoint:
                _save_checkpoint(
                    last_path,
                    stage="stage1_joint",
                    next_step=step + 1,
                    model=model,
                    optimizer=optimizer,
                    bundle=None,
                    local_optimizer=None,
                    best_eval_final_error=best_eval_final_error,
                    metrics=metrics,
                    cfg=cfg,
                    support_batches=list(support_buffer),
                    include_optimizer_state=True,
                )

    _log_phase(
        run_dir,
        "stage1_end",
        device=device,
        next_step=max(start_step, stage2_start),
    )
    if route_codebook is None:
        _log_phase(
            run_dir,
            "route_construction_start",
            device=device,
            routing_object=args.routing_object,
            support_family_fit_source=args.support_family_fit_source,
            stable_fit_trajectories=int(args.stable_fit_trajectories),
            stable_fit_trajectory_length=int(args.stable_fit_trajectory_length),
        )
        learn_target_centers = _learn_target_centers(args.local_map_parameterization)
        artifact: Optional[Dict[str, object]] = None
        if stage2_artifact_path.exists():
            try:
                artifact = torch.load(stage2_artifact_path, map_location=device, weights_only=False)
            except TypeError:
                artifact = torch.load(stage2_artifact_path, map_location=device)
        elif resume_payload is not None and resume_payload.get("route_codebook") is not None:
            artifact = resume_payload
        if artifact is not None:
            route_codebook = artifact["route_codebook"]
            route_metadata = dict(artifact.get("route_metadata", {}))
            route_metadata.setdefault("routing_object", args.routing_object)
            route_metadata.setdefault("support_family_fit_source", args.support_family_fit_source)
            route_metadata.setdefault("local_map_parameterization", args.local_map_parameterization)
            route_metadata.setdefault("target_center_rule", _target_center_rule(args.local_map_parameterization))
            route_metadata.setdefault("learn_target_centers", bool(learn_target_centers))
            global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
            target_centers = artifact.get("target_centers")
            if target_centers is None:
                target_centers = _target_centers_from_global(
                    route_codebook["centers"],
                    route_codebook["fitted_family_ids"],
                    global_k,
                )
            bundle = SourceTargetLocalMapBundle(
                family_ids=route_codebook["fitted_family_ids"],
                source_centers=route_codebook["centers"],
                target_centers=target_centers,
                global_k=global_k,
                device=device,
                learn_target_centers=learn_target_centers,
            ).to(device)
            if artifact.get("local_bundle_state_dict") is not None:
                bundle.load_state_dict(artifact["local_bundle_state_dict"])
        else:
            if args.routing_object == "support_family":
                if args.support_family_fit_source == "stable_fit_trajectories":
                    support_family_fit_batches = _generate_support_fit_batches(
                        train_env,
                        num_trajectories=int(args.stable_fit_trajectories),
                        trajectory_length=int(args.stable_fit_trajectory_length),
                        seed=int(cfg.SEED) + int(args.stable_fit_seed_offset),
                    )
                    fit_latents = _encode_sequence_batches(model, support_family_fit_batches, device)
                else:
                    fit_latents = _encode_sequence_batches(model, list(support_buffer), device)
                route_codebook = _build_route_codebook(
                    fit_latents,
                    scheme=scheme,
                    value=support_value,
                    min_operator_transitions=max(1, int(args.min_family_transitions)),
                    family_jaccard_threshold=float(args.family_jaccard_threshold),
                )
                route_codebook["routing_object"] = "support_family"
                route_codebook["runtime_routing_kind"] = "support_jaccard"
            else:
                stable_fit_batches = _generate_support_fit_batches(
                    train_env,
                    num_trajectories=int(args.stable_fit_trajectories),
                    trajectory_length=int(args.stable_fit_trajectory_length),
                    seed=int(cfg.SEED) + int(args.stable_fit_seed_offset),
                )
                fit_latents = _encode_sequence_batches(model, stable_fit_batches, device)
                stable_route_codebook: Optional[Dict[str, object]] = None
                if args.routing_object in {
                    "stable_support_component",
                    "latent_kmeans",
                    "random_matched",
                }:
                    stable_route_codebook = _build_stable_route_codebook(
                        fit_latents,
                        scheme=scheme,
                        value=support_value,
                        min_operator_transitions=max(1, int(args.min_family_transitions)),
                        route_jaccard_threshold=float(args.family_jaccard_threshold),
                        stable_base_object=args.stable_base_object,
                        stable_base_family_jaccard=float(args.stable_base_family_jaccard),
                        stable_tail_window=int(args.stable_tail_window),
                        stable_min_edge_count=int(args.stable_min_edge_count),
                        stable_min_edge_probability=float(args.stable_min_edge_probability),
                        stable_max_recurrent_out_probability=float(args.stable_max_recurrent_out_probability),
                        stable_min_tail_count=int(args.stable_min_tail_count),
                        stable_min_absorption_observations=int(args.stable_min_absorption_observations),
                        stable_min_absorption_confidence=float(args.stable_min_absorption_confidence),
                    )
                    stable_route_codebook["runtime_routing_kind"] = "support_jaccard"
                if args.routing_object == "stable_support_component":
                    if stable_route_codebook is None:
                        raise RuntimeError("stable_support_component route construction failed")
                    route_codebook = stable_route_codebook
                elif args.routing_object == "oracle_basin":
                    fit_states = torch.cat(stable_fit_batches, dim=0).detach().cpu().numpy().astype(
                        np.float32,
                        copy=False,
                    )
                    route_codebook = _build_oracle_basin_route_codebook(
                        fit_latents,
                        fit_states,
                        env=base_env,
                        min_operator_transitions=max(1, int(args.min_family_transitions)),
                    )
                elif args.routing_object == "latent_kmeans":
                    if stable_route_codebook is None:
                        raise RuntimeError("latent_kmeans requires a C_stab route count reference")
                    if int(args.baseline_latent_cluster_count) > 0:
                        latent_cluster_count = int(args.baseline_latent_cluster_count)
                        cluster_count_source = "manual"
                    else:
                        latent_cluster_count = int(len(stable_route_codebook["fitted_family_ids"]))
                        cluster_count_source = "stable_support_component_fit_count"
                    route_codebook = _build_latent_kmeans_route_codebook(
                        fit_latents,
                        n_clusters=latent_cluster_count,
                        min_operator_transitions=max(1, int(args.min_family_transitions)),
                        seed=int(cfg.SEED) + int(args.baseline_route_seed_offset),
                        n_init=int(args.baseline_kmeans_n_init),
                        cluster_count_source=cluster_count_source,
                    )
                elif args.routing_object == "latent_tail_fate":
                    route_codebook = _build_latent_tail_fate_route_codebook(
                        fit_latents,
                        min_operator_transitions=max(1, int(args.min_family_transitions)),
                        seed=int(cfg.SEED) + int(args.baseline_route_seed_offset),
                        n_init=int(args.baseline_kmeans_n_init),
                        tail_window=int(args.latent_fate_tail_window),
                        max_clusters=int(args.latent_fate_max_clusters),
                        min_silhouette=float(args.latent_fate_min_silhouette),
                        pca_components=int(args.latent_fate_pca_components),
                    )
                elif args.routing_object == "random_matched":
                    if stable_route_codebook is None:
                        raise RuntimeError("random_matched requires a C_stab route count reference")
                    route_codebook = _build_random_matched_route_codebook(
                        fit_latents,
                        stable_route_codebook,
                        min_operator_transitions=max(1, int(args.min_family_transitions)),
                        seed=int(cfg.SEED) + int(args.baseline_route_seed_offset),
                    )
                else:
                    raise ValueError(f"Unsupported routing_object={args.routing_object!r}")
            if not route_codebook["fitted_family_ids"]:
                raise RuntimeError(
                    f"No trainable {args.routing_object} routes were identified from the stage-1 training batches"
                )
            global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
            target_centers = _target_centers_from_global(
                route_codebook["centers"],
                route_codebook["fitted_family_ids"],
                global_k,
            )
            bundle = SourceTargetLocalMapBundle(
                family_ids=route_codebook["fitted_family_ids"],
                source_centers=route_codebook["centers"],
                target_centers=target_centers,
                global_k=global_k,
                device=device,
                learn_target_centers=learn_target_centers,
            ).to(device)
            route_metadata = _route_codebook_metadata(route_codebook)
            route_metadata["local_map_parameterization"] = args.local_map_parameterization
            route_metadata["target_center_rule"] = _target_center_rule(args.local_map_parameterization)
            route_metadata["learn_target_centers"] = bool(learn_target_centers)
            route_metadata["routing_object"] = args.routing_object
            route_metadata["support_family_fit_source"] = args.support_family_fit_source
            route_metadata["baseline_route_seed_offset"] = int(args.baseline_route_seed_offset)
            route_metadata["baseline_latent_cluster_count"] = int(args.baseline_latent_cluster_count)
            route_metadata["baseline_kmeans_n_init"] = int(args.baseline_kmeans_n_init)
            route_metadata["latent_fate_tail_window"] = int(args.latent_fate_tail_window)
            route_metadata["latent_fate_max_clusters"] = int(args.latent_fate_max_clusters)
            route_metadata["latent_fate_min_silhouette"] = float(args.latent_fate_min_silhouette)
            route_metadata["latent_fate_pca_components"] = int(args.latent_fate_pca_components)
            if args.routing_object in {
                "stable_support_component",
                "oracle_basin",
                "latent_kmeans",
                "latent_tail_fate",
                "random_matched",
            } or (
                args.routing_object == "support_family"
                and args.support_family_fit_source == "stable_fit_trajectories"
            ):
                route_metadata["stable_fit_trajectories"] = int(args.stable_fit_trajectories)
                route_metadata["stable_fit_trajectory_length"] = int(args.stable_fit_trajectory_length)
                route_metadata["stable_fit_seed"] = int(cfg.SEED) + int(args.stable_fit_seed_offset)
            if args.save_stage2_artifacts:
                _write_torch(
                    stage2_artifact_path,
                    {
                        "route_codebook": route_codebook,
                        "route_metadata": route_metadata,
                        "target_centers": target_centers,
                        "local_bundle_state_dict": bundle.state_dict(),
                    },
                )
            _write_json(run_dir / "route_codebook.json", route_metadata)

    assert route_codebook is not None
    assert bundle is not None
    assert target_centers is not None
    _log_phase(
        run_dir,
        "route_construction_end",
        device=device,
        fitted_route_count=len(route_codebook["fitted_family_ids"]),
        routing_object=args.routing_object,
    )
    _freeze_autoencoder(model)
    for name, param in model.named_parameters():
        if "kmat" in name or name.startswith("K_"):
            param.requires_grad_(False)
    local_lr = float(args.local_lr) if args.local_lr is not None else float(cfg.TRAIN.K_MATRIX_LR)
    local_optimizer = torch.optim.AdamW(bundle.parameters(), lr=local_lr, weight_decay=0.0)
    stage2_route_counts_path = run_dir / "stage2_route_counts.json"
    stage2_route_counts = {str(family_id): 0 for family_id in bundle.family_ids}
    stage2_route_total_count = 0
    stage2_route_fallback_count = 0

    if args.resume_from_latest and resume_payload is not None and start_step >= stage2_start:
        payload = resume_payload
        if payload.get("local_bundle_state_dict") is not None:
            bundle.load_state_dict(payload["local_bundle_state_dict"])
        if payload.get("local_optimizer_state_dict") is not None:
            local_optimizer.load_state_dict(payload["local_optimizer_state_dict"])
        if stage2_route_counts_path.exists():
            saved_route_counts = json.loads(stage2_route_counts_path.read_text())
            for family_id, value in saved_route_counts.get("route_counts_by_family", {}).items():
                if family_id in stage2_route_counts:
                    stage2_route_counts[family_id] = int(value)
            stage2_route_total_count = int(saved_route_counts.get("route_total_count", 0))
            stage2_route_fallback_count = int(saved_route_counts.get("route_fallback_count", 0))

    _log_phase(
        run_dir,
        "stage2_start",
        device=device,
        start_step=max(start_step, stage2_start),
        end_step=total_steps,
    )
    for step in range(max(start_step, stage2_start), total_steps):
        rng = rngs[step % num_batches]
        x_seq = generate_sequence_batch_for_device(
            train_env,
            rng,
            window_length=cfg.TRAIN.SEQUENCE_LENGTH,
            device=device,
        )
        metrics = _local_train_step(
            model=model,
            bundle=bundle,
            route_codebook=route_codebook,
            route_env=base_env,
            x_seq=x_seq,
            scheme=scheme,
            support_value=support_value,
            family_jaccard_threshold=float(args.family_jaccard_threshold),
            optimizer=local_optimizer,
            family_cache=local_family_cache,
            step=step,
        )
        last_metrics = metrics
        stage2_route_total_count += int(metrics.get("route_total_count", 0.0))
        stage2_route_fallback_count += int(metrics.get("route_fallback_count", 0.0))
        for family_id in bundle.family_ids:
            stage2_route_counts[str(family_id)] += int(metrics.get(f"route_family_{family_id}_count", 0.0))
        logger.log_dict(metrics, step, prefix="stage2_local_train")

        if step % 100 == 0:
            print(
                f"Stage 2 step {step}/{total_steps} "
                f"loss={metrics['loss']:.6g} coverage={metrics['route_coverage']:.3f}",
                flush=True,
            )
        if (step > stage2_start and step % cfg.TRAIN.EVAL_EVERY == 0) or step == total_steps - 1:
            wrapped = _make_wrapped_model(
                model,
                bundle,
                route_codebook,
                route_env=base_env,
                scheme=scheme,
                support_value=support_value,
                family_jaccard_threshold=float(args.family_jaccard_threshold),
            )
            if args.stage2_selection_metric == "best_periodic_horizon_mse":
                if not stage2_selection_periods:
                    raise ValueError("--stage2_selection_periods must include at least one period")
                selection_score, best_by_horizon = _quick_eval_best_periodic_horizon_mse(
                    wrapped,
                    val_x=val_x,
                    eval_env=eval_env,
                    horizons=stage2_selection_horizons,
                    periods=stage2_selection_periods,
                )
                logger.log_scalar("stage2_eval/best_periodic_horizon_mse", selection_score, step)
                for horizon, (value, best_period) in best_by_horizon.items():
                    logger.log_scalar(f"stage2_eval/h{horizon}_best_periodic_mse", value, step)
                    logger.log_scalar(f"stage2_eval/h{horizon}_best_periodic_period", float(best_period), step)
                detail = ", ".join(
                    f"H{horizon}={value:.6g}@{best_period}"
                    for horizon, (value, best_period) in best_by_horizon.items()
                )
                print(
                    "  Stage 2 quick eval best periodic horizon MSE: "
                    f"{selection_score:.6g} ({detail})",
                    flush=True,
                )
            elif args.stage2_selection_metric == "best_periodic_mse":
                if not stage2_selection_periods:
                    raise ValueError("--stage2_selection_periods must include at least one period")
                selection_score, best_period = _quick_eval_best_periodic_mse(
                    wrapped,
                    val_x=val_x,
                    eval_env=eval_env,
                    num_steps=cfg.TRAIN.EVAL_NUM_STEPS,
                    periods=stage2_selection_periods,
                )
                logger.log_scalar("stage2_eval/best_periodic_mse", selection_score, step)
                logger.log_scalar("stage2_eval/best_periodic_period", float(best_period), step)
                print(
                    "  Stage 2 quick eval best periodic MSE: "
                    f"{selection_score:.6g} (period={best_period})",
                    flush=True,
                )
            else:
                selection_score = _quick_eval_final_error(
                    wrapped,
                    val_x=val_x,
                    eval_env=eval_env,
                    num_steps=cfg.TRAIN.EVAL_NUM_STEPS,
                )
                logger.log_scalar("stage2_eval/final_error", selection_score, step)
                print(f"  Stage 2 quick eval final error: {selection_score:.6g}", flush=True)
            improved_best = selection_score < best_eval_final_error
            if improved_best:
                best_eval_final_error = selection_score
            if args.save_last_checkpoint:
                _save_checkpoint(
                    last_path,
                    stage="stage2_local",
                    next_step=step + 1,
                    model=model,
                    optimizer=None,
                    bundle=bundle,
                    local_optimizer=local_optimizer,
                    best_eval_final_error=best_eval_final_error,
                    metrics=metrics,
                    cfg=cfg,
                    route_metadata=route_metadata,
                    route_codebook=route_codebook,
                    target_centers=target_centers,
                    include_optimizer_state=True,
                )
            _write_json(
                stage2_route_counts_path,
                {
                    "stage2_start_step": int(stage2_start),
                    "last_recorded_step": int(step),
                    "route_total_count": int(stage2_route_total_count),
                    "route_fallback_count": int(stage2_route_fallback_count),
                    "route_counts_by_family": dict(stage2_route_counts),
                    "family_ids": list(bundle.family_ids),
                },
            )
            if improved_best:
                _save_checkpoint(
                    run_dir / "checkpoint.pt",
                    stage="stage2_local",
                    next_step=step + 1,
                    model=model,
                    optimizer=None,
                    bundle=bundle,
                    local_optimizer=local_optimizer,
                    best_eval_final_error=best_eval_final_error,
                    metrics=metrics,
                    cfg=cfg,
                    route_metadata=route_metadata,
                    route_codebook=route_codebook,
                    target_centers=target_centers,
                )
                _write_json(
                    run_dir / "stage2_route_counts_best.json",
                    {
                        "stage2_start_step": int(stage2_start),
                        "best_recorded_step": int(step),
                        "route_total_count": int(stage2_route_total_count),
                        "route_fallback_count": int(stage2_route_fallback_count),
                        "route_counts_by_family": dict(stage2_route_counts),
                        "family_ids": list(bundle.family_ids),
                    },
                )
                print(f"  Saved best staged local-K checkpoint ({selection_score:.6g})", flush=True)

    _log_phase(run_dir, "stage2_end", device=device, next_step=total_steps)
    best_checkpoint_path = run_dir / "checkpoint.pt"
    if not best_checkpoint_path.exists():
        _save_checkpoint(
            best_checkpoint_path,
            stage="stage2_local",
            next_step=total_steps,
            model=model,
            optimizer=None,
            bundle=bundle,
            local_optimizer=local_optimizer,
            best_eval_final_error=best_eval_final_error,
            metrics=last_metrics,
            cfg=cfg,
            route_metadata=route_metadata,
            route_codebook=route_codebook,
            target_centers=target_centers,
        )
        print("  Saved final staged local-K checkpoint to checkpoint.pt", flush=True)
    logger.close()
    _write_json(run_dir / "final_metrics.json", last_metrics)

    try:
        best_payload = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        best_payload = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])
    if best_payload.get("local_bundle_state_dict") is not None:
        bundle.load_state_dict(best_payload["local_bundle_state_dict"])
    wrapped = _make_wrapped_model(
        model,
        bundle,
        route_codebook,
        route_env=base_env,
        scheme=scheme,
        support_value=support_value,
        family_jaccard_threshold=float(args.family_jaccard_threshold),
    )
    eval_settings = _make_eval_settings(
        args.eval_profile,
        cfg,
        periodic_periods_override=eval_periodic_periods,
        save_rollout_artifacts=bool(args.save_eval_rollout_artifacts),
        save_plots=bool(args.save_eval_plots),
        include_per_ic_values=bool(args.save_eval_per_ic_values),
        include_error_curves=bool(args.save_eval_error_curves),
    )
    _log_phase(
        run_dir,
        "final_eval_start",
        device=device,
        eval_profile=args.eval_profile,
    )
    eval_results = evaluate_model(
        model=wrapped,
        cfg=cfg,
        device=device,
        settings=eval_settings,
        output_dir=run_dir / "evaluation_best",
    )
    _log_phase(run_dir, "final_eval_end", device=device)
    _write_json(run_dir / "evaluation_results_best.json", eval_results)
    _write_json(
        run_dir / "evaluation_summary.json",
        {
            "best_checkpoint": True,
            "last_checkpoint": bool(args.save_last_checkpoint and last_path.exists()),
            "staged_local_k": True,
            "support_definition": args.support_definition,
            "family_jaccard_threshold": float(args.family_jaccard_threshold),
            **route_metadata,
        },
    )
    print(f"Training complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
