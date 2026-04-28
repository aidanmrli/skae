#!/usr/bin/env python3
"""Test whether re-encoding refreshes support after basin entry.

This evaluator targets the specific mechanism claim:

Periodic decode/re-encode gives the model a way to refresh latent support from
the current decoded state; once the state is in a new basin, the refreshed
support should become target-like and subsequent Koopman evolution can be routed
through different active coordinates than a stale source support would use.

The experiment uses the same controlled state-space source-to-target transfer
as the switching evaluator, but measures a different object:

1. At measured target entry / post-entry, encode the current target-basin state.
2. Compare no-refresh source-latent continuation, current-state global
   continuation, periodic global decode/re-encode, previous source-support
   gated K, and refreshed-support gated K.
3. Track source/target support dominance, re-encoding support changes, route
   switches, fallback rate, and forecast MSE against the post-entry unforced
   transfer segment.

Basin labels and fixed source/target counts are used only for benchmark
evaluation and controlled trajectory construction.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

EPS = 1e-12
FALLBACK_ROUTE = "__global_fallback__"


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CTRL = _load_module(
    "evaluate_transition_rich_controlled_transfer_switching.py",
    "evaluate_transition_rich_controlled_transfer_switching_periodic_refresh",
)
SELF = _load_module(
    "evaluate_transition_rich_self_routed_forecasting.py",
    "evaluate_transition_rich_self_routed_forecasting_periodic_refresh",
)
REDUCER = CTRL.REDUCER
OPSEL = CTRL.OPSEL


@dataclass
class ObjectContext:
    object_kind: str
    scheme: str
    value: float
    source_ref: object
    target_ref: object
    source_mask: np.ndarray
    target_mask: np.ndarray
    source_center: np.ndarray
    target_center: np.ndarray
    family_prototypes: Optional[Dict[object, np.ndarray]] = None
    family_threshold: float = 0.5

    @property
    def source_target_same_object(self) -> bool:
        return self.source_ref == self.target_ref


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csvs", required=True, help="comma-separated forecasting_rows.csv files")
    parser.add_argument("--output_dir", required=True, help="directory for output artifacts")
    parser.add_argument("--root_labels", required=True, help="comma-separated root labels to include")
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated integer seed filter")
    parser.add_argument(
        "--support_definitions",
        default="absolute:0.001,topk:8",
        help="comma-separated support definitions formatted as scheme:value",
    )
    parser.add_argument("--num_transfers_per_pair", type=int, default=2)
    parser.add_argument("--max_pairs_per_system", type=int, default=0, help="0 means all ordered pairs")
    parser.add_argument("--pre_steps", type=int, default=32)
    parser.add_argument("--bridge_steps", type=int, default=32)
    parser.add_argument("--post_steps", type=int, default=128)
    parser.add_argument("--continuation_horizon", type=int, default=64)
    parser.add_argument("--reference_tail_steps", type=int, default=32)
    parser.add_argument("--reencode_periods", default="1,8", help="comma-separated positive periods")
    parser.add_argument(
        "--start_modes",
        default="target_entry,post_start",
        help="comma-separated start modes from {target_entry,post_start}",
    )
    parser.add_argument("--source_depth_fraction", type=float, default=0.12)
    parser.add_argument("--target_depth_fraction", type=float, default=0.12)
    parser.add_argument("--source_pre_min_fraction", type=float, default=0.80)
    parser.add_argument("--final_target_min_fraction", type=float, default=0.80)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.5)
    parser.add_argument(
        "--use_dynamics_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="warm-start encode_with_prior with the predicted latent during re-encoding",
    )
    parser.add_argument("--max_specs", type=int, default=0, help="0 means no limit")
    parser.add_argument("--progress_every_runs", type=int, default=1)
    parser.add_argument("--flush_every_runs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="override to a tiny evaluator subset")
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_periods(raw: str) -> List[int]:
    periods = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if any(period <= 0 for period in periods):
        raise ValueError("reencode periods must be positive")
    return periods


def _parse_support_definitions(raw: str) -> List[Tuple[str, float]]:
    definitions: List[Tuple[str, float]] = []
    for item in _parse_csv_strings(raw):
        if ":" not in item:
            raise ValueError(f"Support definition must be scheme:value, got '{item}'")
        scheme, raw_value = item.split(":", 1)
        scheme = scheme.strip()
        raw_value = raw_value.strip()
        if scheme == "topk":
            definitions.append((scheme, float(int(raw_value))))
        else:
            definitions.append((scheme, float(raw_value)))
    return definitions


def _stringify_support_definition(scheme: str, value: float) -> str:
    if scheme == "topk":
        return f"{scheme}:{int(value)}"
    return f"{scheme}:{value:.6g}"


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mode(values: Sequence[object]) -> Optional[object]:
    if len(values) == 0:
        return None
    return Counter(values).most_common(1)[0][0]


def _fraction_equal(values: Sequence[object], target: object) -> Optional[float]:
    if target is None or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=object)
    return float(np.mean(arr == target))


def _switch_rate(values: Sequence[object]) -> Optional[float]:
    if len(values) <= 1:
        return None
    arr = np.asarray(values, dtype=object)
    return float(np.mean(arr[1:] != arr[:-1]))


def _first_index(values: Sequence[object], target: object) -> Optional[int]:
    if target is None:
        return None
    for idx, value in enumerate(values):
        if value == target:
            return int(idx)
    return None


def _binary_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool, copy=False)
    bb = b.astype(bool, copy=False)
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum()) / float(union)


def _support_masks(latents: np.ndarray, scheme: str, value: float) -> np.ndarray:
    return REDUCER._support_mask(np.asarray(latents), scheme=scheme, value=value)


def _support_keys(masks: np.ndarray) -> np.ndarray:
    return REDUCER._support_keys(masks).reshape(-1)


def _mask_for_label(labels: np.ndarray, masks: np.ndarray, label: object) -> np.ndarray:
    if label is None:
        return np.ones(masks.shape[-1], dtype=bool)
    select = labels == label
    if not np.any(select):
        return np.ones(masks.shape[-1], dtype=bool)
    keys = _support_keys(masks[select])
    mode_key = _mode(keys.tolist())
    for key, mask in zip(keys.tolist(), masks[select]):
        if key == mode_key:
            return mask.astype(bool, copy=True)
    return masks[select][0].astype(bool, copy=True)


def _center_for_label(latents: np.ndarray, labels: np.ndarray, label: object) -> np.ndarray:
    select = labels == label
    if np.any(select):
        return latents[select].mean(axis=0).astype(np.float32, copy=False)
    return latents.mean(axis=0).astype(np.float32, copy=False)


def _family_context_from_masks(
    source_masks: np.ndarray,
    target_masks: np.ndarray,
    *,
    min_jaccard: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[object, np.ndarray]]:
    joint = np.concatenate([source_masks, target_masks], axis=0)
    labels = REDUCER.support_family_labels(joint[None, :, :], min_jaccard=min_jaccard)[0].astype(object)
    source_labels = labels[: source_masks.shape[0]]
    target_labels = labels[source_masks.shape[0] :]
    prototypes: Dict[object, np.ndarray] = {}
    for family in sorted({item for item in labels.tolist()}):
        family_masks = joint[labels == family]
        prototypes[family] = (family_masks.mean(axis=0) >= 0.5)
    return source_labels, target_labels, prototypes


def _assign_family_labels(
    masks: np.ndarray,
    prototypes: Dict[object, np.ndarray],
    *,
    min_jaccard: float,
) -> np.ndarray:
    labels: List[object] = []
    for mask in masks:
        best_label = None
        best_similarity = -1.0
        for label, prototype in prototypes.items():
            similarity = _binary_jaccard(mask, prototype)
            if similarity > best_similarity:
                best_label = label
                best_similarity = similarity
        labels.append(best_label if best_similarity >= float(min_jaccard) else None)
    return np.asarray(labels, dtype=object)


def _labels_for_context(context: ObjectContext, latents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    masks = _support_masks(latents, context.scheme, context.value)
    if context.object_kind == "support":
        return _support_keys(masks), masks
    if context.family_prototypes is None:
        raise ValueError("family context requires prototypes")
    return (
        _assign_family_labels(
            masks,
            context.family_prototypes,
            min_jaccard=context.family_threshold,
        ),
        masks,
    )


def _build_object_context(
    transfer_latents: np.ndarray,
    target_latents: np.ndarray,
    *,
    source_end_index: int,
    reference_tail_steps: int,
    scheme: str,
    value: float,
    object_kind: str,
    family_jaccard_threshold: float,
) -> Optional[ObjectContext]:
    source_end_index = max(1, min(int(source_end_index), transfer_latents.shape[0]))
    source_latents = transfer_latents[:source_end_index]
    tail = max(1, min(int(reference_tail_steps), target_latents.shape[0]))
    target_tail_latents = target_latents[-tail:]

    source_masks = _support_masks(source_latents, scheme, value)
    target_masks = _support_masks(target_tail_latents, scheme, value)

    if object_kind == "support":
        source_labels = _support_keys(source_masks)
        target_labels = _support_keys(target_masks)
        prototypes = None
    elif object_kind == "family":
        source_labels, target_labels, prototypes = _family_context_from_masks(
            source_masks,
            target_masks,
            min_jaccard=family_jaccard_threshold,
        )
    else:
        raise ValueError(f"Unknown object kind '{object_kind}'")

    source_ref = _mode(source_labels.tolist())
    target_ref = _mode(target_labels.tolist())
    if source_ref is None or target_ref is None:
        return None

    source_mask = _mask_for_label(source_labels, source_masks, source_ref)
    target_mask = _mask_for_label(target_labels, target_masks, target_ref)
    if object_kind == "family" and prototypes is not None:
        source_mask = prototypes.get(source_ref, source_mask).astype(bool, copy=True)
        target_mask = prototypes.get(target_ref, target_mask).astype(bool, copy=True)

    return ObjectContext(
        object_kind=object_kind,
        scheme=scheme,
        value=value,
        source_ref=source_ref,
        target_ref=target_ref,
        source_mask=source_mask.astype(bool, copy=True),
        target_mask=target_mask.astype(bool, copy=True),
        source_center=_center_for_label(source_latents, source_labels, source_ref),
        target_center=_center_for_label(target_tail_latents, target_labels, target_ref),
        family_prototypes=prototypes,
        family_threshold=family_jaccard_threshold,
    )


def _encode_state(model, state: torch.Tensor) -> np.ndarray:
    model_device = next(model.parameters()).device
    with torch.no_grad():
        latent = model.encode(state.reshape(1, -1).to(model_device))
    return latent.detach().cpu().numpy().astype(np.float32, copy=False)[0]


def _decode_latent(model, latent: np.ndarray) -> torch.Tensor:
    model_device = next(model.parameters()).device
    latent_tensor = torch.as_tensor(latent.reshape(1, -1), dtype=torch.float32, device=model_device)
    with torch.no_grad():
        decoded = model.decode(latent_tensor)
    return decoded.detach().cpu().reshape(-1).float()


def _reencode_state(
    model,
    state: torch.Tensor,
    predicted_latent: np.ndarray,
    *,
    use_dynamics_prior: bool,
) -> np.ndarray:
    model_device = next(model.parameters()).device
    state_tensor = state.reshape(1, -1).to(model_device)
    prior = None
    if use_dynamics_prior:
        prior = torch.as_tensor(predicted_latent.reshape(1, -1), dtype=torch.float32, device=model_device)
    with torch.no_grad():
        if hasattr(model, "encode_with_prior"):
            latent = model.encode_with_prior(state_tensor, latent_prior=prior)
        else:
            latent = model.encode(state_tensor)
    return latent.detach().cpu().numpy().astype(np.float32, copy=False)[0]


def _predict_global(latent: np.ndarray, global_k: np.ndarray) -> np.ndarray:
    return (latent.reshape(1, -1) @ global_k).astype(np.float32, copy=False)[0]


def _predict_gated(
    latent: np.ndarray,
    center: np.ndarray,
    global_k: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    out = SELF._predict_gated_k(
        latent.reshape(1, -1).astype(np.float32, copy=False),
        center.astype(np.float32, copy=False),
        global_k,
        mask.reshape(1, -1),
    )
    return out.astype(np.float32, copy=False)[0]


def _route_for_current_label(context: ObjectContext, label: object) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if label == context.target_ref:
        return context.target_mask, context.target_center, "target_ref"
    if label == context.source_ref:
        return context.source_mask, context.source_center, "source_ref"
    return None, None, FALLBACK_ROUTE


def _rollout_continuation(
    model,
    initial_latent: np.ndarray,
    true_future: torch.Tensor,
    *,
    global_k: np.ndarray,
    mode: str,
    context: ObjectContext,
    reencode_period: int,
    use_dynamics_prior: bool,
) -> Dict[str, object]:
    horizon = int(true_future.shape[0])
    latent = initial_latent.astype(np.float32, copy=True)
    predictions: List[np.ndarray] = []
    current_labels: List[object] = []
    pre_reencode_labels: List[object] = []
    post_reencode_labels: List[object] = []
    route_labels: List[str] = []
    refresh_changed: List[bool] = []
    reencode_applied: List[bool] = []

    for step in range(horizon):
        current_label = _labels_for_context(context, latent.reshape(1, -1))[0][0]
        current_labels.append(current_label)

        if mode in {"global_no_reencode", "global_periodic_reencode"}:
            next_latent = _predict_global(latent, global_k)
            route_label = "global_k"
        elif mode == "frozen_source_gated_periodic":
            next_latent = _predict_gated(latent, context.source_center, global_k, context.source_mask)
            route_label = "source_ref"
        elif mode == "current_support_gated_periodic":
            mask, center, route_label = _route_for_current_label(context, current_label)
            if mask is None or center is None:
                next_latent = _predict_global(latent, global_k)
            else:
                next_latent = _predict_gated(latent, center, global_k, mask)
        else:
            raise ValueError(f"Unknown rollout mode '{mode}'")

        pre_label = _labels_for_context(context, next_latent.reshape(1, -1))[0][0]
        decoded = _decode_latent(model, next_latent)
        predictions.append(decoded.detach().cpu().numpy().astype(np.float32, copy=False))

        should_reencode = mode != "global_no_reencode" and reencode_period > 0 and ((step + 1) % reencode_period == 0)
        if should_reencode:
            refreshed_latent = _reencode_state(
                model,
                decoded,
                next_latent,
                use_dynamics_prior=use_dynamics_prior,
            )
            post_label = _labels_for_context(context, refreshed_latent.reshape(1, -1))[0][0]
            latent = refreshed_latent
            refresh_changed.append(post_label != pre_label)
            reencode_applied.append(True)
        else:
            post_label = pre_label
            latent = next_latent
            refresh_changed.append(False)
            reencode_applied.append(False)

        pre_reencode_labels.append(pre_label)
        post_reencode_labels.append(post_label)
        route_labels.append(route_label)

    preds = np.stack(predictions, axis=0) if predictions else np.empty((0, int(true_future.shape[-1])), dtype=np.float32)
    truth = true_future.detach().cpu().numpy().astype(np.float32, copy=False)
    mse = None if preds.size == 0 else float(np.mean((preds - truth) ** 2))
    first_target = _first_index(post_reencode_labels, context.target_ref)
    applied = np.asarray(reencode_applied, dtype=bool)
    changed = np.asarray(refresh_changed, dtype=bool)
    route_arr = np.asarray(route_labels, dtype=object)
    return {
        "forecast_mse": mse,
        "initial_object": str(current_labels[0]) if current_labels else None,
        "source_object": str(context.source_ref),
        "target_object": str(context.target_ref),
        "source_target_same_object": bool(context.source_target_same_object),
        "current_source_dominance": _fraction_equal(current_labels, context.source_ref),
        "current_target_dominance": _fraction_equal(current_labels, context.target_ref),
        "pre_reencode_source_dominance": _fraction_equal(pre_reencode_labels, context.source_ref),
        "pre_reencode_target_dominance": _fraction_equal(pre_reencode_labels, context.target_ref),
        "post_reencode_source_dominance": _fraction_equal(post_reencode_labels, context.source_ref),
        "post_reencode_target_dominance": _fraction_equal(post_reencode_labels, context.target_ref),
        "first_post_reencode_target_step": first_target,
        "target_switch_detected": first_target is not None and not context.source_target_same_object,
        "support_refresh_event_fraction": None if not np.any(applied) else float(np.mean(changed[applied])),
        "support_chatter_switch_rate": _switch_rate(post_reencode_labels),
        "route_source_fraction": float(np.mean(route_arr == "source_ref")) if route_arr.size else None,
        "route_target_fraction": float(np.mean(route_arr == "target_ref")) if route_arr.size else None,
        "route_global_fraction": float(np.mean(route_arr == "global_k")) if route_arr.size else None,
        "route_fallback_fraction": float(np.mean(route_arr == FALLBACK_ROUTE)) if route_arr.size else None,
        "route_switch_rate": _switch_rate(route_labels),
        "reencode_applied_fraction": float(np.mean(applied)) if applied.size else None,
    }


def _skip_rows(
    spec,
    support_definitions: Sequence[Tuple[str, float]],
    *,
    status: str,
    skip_reason: str,
    label_source: str = "",
    center_source: str = "",
) -> List[Dict[str, object]]:
    rows = []
    for scheme, value in support_definitions:
        for object_kind in ("support", "family"):
            rows.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "system_name": spec.system_name,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
                    "support_definition": _stringify_support_definition(scheme, value),
                    "object_kind": object_kind,
                    "status": status,
                    "skip_reason": skip_reason,
                    "label_source": label_source,
                    "center_source": center_source,
                }
            )
    return rows


def _mse_ratio(rows: Sequence[Dict[str, object]], numerator_mode: str, denominator_mode: str) -> Optional[float]:
    numerator = next((row for row in rows if row.get("rollout_mode") == numerator_mode), None)
    denominator = next((row for row in rows if row.get("rollout_mode") == denominator_mode), None)
    if numerator is None or denominator is None:
        return None
    num = _as_float(numerator.get("forecast_mse"))
    den = _as_float(denominator.get("forecast_mse"))
    if num is None or den is None or den <= EPS:
        return None
    return float(num) / float(den)


def evaluate_run(
    spec,
    *,
    support_definitions: Sequence[Tuple[str, float]],
    num_transfers_per_pair: int,
    max_pairs_per_system: int,
    pre_steps: int,
    bridge_steps: int,
    post_steps: int,
    continuation_horizon: int,
    reference_tail_steps: int,
    reencode_periods: Sequence[int],
    start_modes: Sequence[str],
    source_depth_fraction: float,
    target_depth_fraction: float,
    source_pre_min_fraction: float,
    final_target_min_fraction: float,
    endpoint_rollout_steps: int,
    eval_seed: int,
    device: str,
    family_jaccard_threshold: float,
    use_dynamics_prior: bool,
) -> List[Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    _cfg, env, model = REDUCER._load_checkpoint_model(checkpoint_path, spec.system_key, device)
    model.eval()
    global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
    state_dim = int(env.observation_size)
    basin_count = int(REDUCER.get_transition_rich_basin_count(spec.system_key))
    centers, center_source = CTRL._extract_centers(env, state_dim, basin_count)
    if centers is None:
        return _skip_rows(
            spec,
            support_definitions,
            status="skipped",
            skip_reason="missing_transfer_centers",
            center_source=center_source,
        )
    if centers.shape[0] < basin_count:
        return _skip_rows(
            spec,
            support_definitions,
            status="skipped",
            skip_reason=f"insufficient_transfer_centers:{centers.shape[0]}<{basin_count}",
            center_source=center_source,
        )

    rng_np = np.random.default_rng(eval_seed + int(spec.seed))
    pairs = CTRL._ordered_pairs(basin_count, max_pairs_per_system, rng_np)
    rows: List[Dict[str, object]] = []

    for pair_index, (source, target) in enumerate(pairs):
        for transfer_index in range(max(1, int(num_transfers_per_pair))):
            torch_rng = torch.Generator().manual_seed(
                int(eval_seed) + 10_000 * int(spec.seed) + 101 * pair_index + transfer_index
            )
            jitter_scale = 0.01 * float(transfer_index)
            try:
                transfer, _no_transfer, target_reference, metadata = CTRL._build_transfer_trajectory(
                    env,
                    centers,
                    source,
                    target,
                    pre_steps=pre_steps,
                    bridge_steps=bridge_steps,
                    post_steps=post_steps,
                    source_depth_fraction=source_depth_fraction,
                    target_depth_fraction=target_depth_fraction,
                    jitter_scale=jitter_scale,
                    rng=torch_rng,
                )
            except Exception as exc:  # pragma: no cover - runtime environment variability
                rows.append(
                    {
                        "root_label": spec.root_label,
                        "system_key": spec.system_key,
                        "system_name": spec.system_name,
                        "seed": spec.seed,
                        "run_dir": spec.run_dir,
                        "source_basin": source,
                        "target_basin": target,
                        "transfer_index": transfer_index,
                        "status": "skipped",
                        "skip_reason": "trajectory_construction_failed",
                        "error": repr(exc),
                        "center_source": center_source,
                    }
                )
                continue

            labels, label_source, label_error = CTRL._label_states_for_evaluation(
                env,
                transfer,
                centers,
                endpoint_rollout_steps=endpoint_rollout_steps,
            )
            if labels is None:
                rows.extend(
                    _skip_rows(
                        spec,
                        support_definitions,
                        status="skipped",
                        skip_reason=f"labeling_failed:{label_error}",
                        label_source=label_source,
                        center_source=center_source,
                    )
                )
                continue

            (
                success,
                transfer_reason,
                source_exit_index,
                target_entry_index,
                source_pre_fraction,
                final_target_fraction,
            ) = CTRL._status_from_labels(
                labels,
                source,
                target,
                pre_end_index=int(metadata["bridge_start_index"]),
                final_target_min_fraction=final_target_min_fraction,
                source_pre_min_fraction=source_pre_min_fraction,
            )

            base = {
                "root_label": spec.root_label,
                "system_key": spec.system_key,
                "system_name": spec.system_name,
                "seed": spec.seed,
                "run_dir": spec.run_dir,
                "source_basin": source,
                "target_basin": target,
                "transfer_index": transfer_index,
                "center_source": center_source,
                "label_source": label_source,
                "transfer_success": bool(success),
                "transfer_failure_reason": None if success else transfer_reason,
                "source_exit_index": source_exit_index,
                "target_entry_index": target_entry_index,
                "source_pre_fraction": source_pre_fraction,
                "final_target_fraction": final_target_fraction,
                "intervention_admissibility": "state_bridge_not_dynamics_control",
                "mechanism_test": "post_entry_reencode_refresh_and_routing",
                "use_dynamics_prior": bool(use_dynamics_prior),
                **metadata,
            }

            if not success or source_exit_index is None or target_entry_index is None:
                for scheme, value in support_definitions:
                    for object_kind in ("support", "family"):
                        rows.append(
                            {
                                **base,
                                "support_definition": _stringify_support_definition(scheme, value),
                                "object_kind": object_kind,
                                "status": "transfer_failed",
                                "skip_reason": transfer_reason,
                            }
                        )
                continue

            transfer_latents = CTRL._encode_single(model, transfer, device)
            target_latents = CTRL._encode_single(model, target_reference, device)

            source_stale_index = max(0, int(source_exit_index) - 1)
            stale_source_latent = transfer_latents[source_stale_index].astype(np.float32, copy=False)
            start_indices: Dict[str, int] = {}
            if "target_entry" in start_modes:
                start_indices["target_entry"] = int(target_entry_index)
            if "post_start" in start_modes:
                start_indices["post_start"] = max(int(target_entry_index), int(metadata["post_start_index"]))

            for start_mode, start_index in start_indices.items():
                max_start = int(transfer.shape[0]) - int(continuation_horizon) - 1
                if start_index > max_start:
                    for scheme, value in support_definitions:
                        for object_kind in ("support", "family"):
                            rows.append(
                                {
                                    **base,
                                    "support_definition": _stringify_support_definition(scheme, value),
                                    "object_kind": object_kind,
                                    "start_mode": start_mode,
                                    "start_index": start_index,
                                    "status": "skipped",
                                    "skip_reason": "insufficient_post_entry_horizon",
                                }
                            )
                    continue

                current_latent = transfer_latents[start_index].astype(np.float32, copy=False)
                true_future = transfer[start_index + 1 : start_index + 1 + int(continuation_horizon)]
                post_entry_state = transfer[start_index]

                for scheme, value in support_definitions:
                    support_definition = _stringify_support_definition(scheme, value)
                    for object_kind in ("support", "family"):
                        context = _build_object_context(
                            transfer_latents,
                            target_latents,
                            source_end_index=max(1, int(source_exit_index)),
                            reference_tail_steps=reference_tail_steps,
                            scheme=scheme,
                            value=value,
                            object_kind=object_kind,
                            family_jaccard_threshold=family_jaccard_threshold,
                        )
                        if context is None:
                            rows.append(
                                {
                                    **base,
                                    "support_definition": support_definition,
                                    "object_kind": object_kind,
                                    "start_mode": start_mode,
                                    "start_index": start_index,
                                    "status": "skipped",
                                    "skip_reason": "could_not_build_object_context",
                                }
                            )
                            continue

                        init_label = _labels_for_context(context, current_latent.reshape(1, -1))[0][0]
                        stale_label = _labels_for_context(context, stale_source_latent.reshape(1, -1))[0][0]
                        grouped_rows: List[Dict[str, object]] = []
                        rollout_specs: List[Tuple[str, np.ndarray, int]] = [
                            ("stale_source_global_no_reencode", stale_source_latent, 0),
                            ("current_global_no_reencode", current_latent, 0),
                        ]
                        for period in reencode_periods:
                            rollout_specs.extend(
                                [
                                    ("global_periodic_reencode", current_latent, int(period)),
                                    ("frozen_source_gated_periodic", current_latent, int(period)),
                                    ("current_support_gated_periodic", current_latent, int(period)),
                                ]
                            )

                        for rollout_mode, initial_latent, period in rollout_specs:
                            metrics = _rollout_continuation(
                                model,
                                initial_latent,
                                true_future,
                                global_k=global_k,
                                mode=(
                                    "global_no_reencode"
                                    if rollout_mode.endswith("_no_reencode")
                                    else rollout_mode
                                ),
                                context=context,
                                reencode_period=period,
                                use_dynamics_prior=use_dynamics_prior,
                            )
                            row = {
                                **base,
                                "support_definition": support_definition,
                                "object_kind": object_kind,
                                "start_mode": start_mode,
                                "start_index": start_index,
                                "post_entry_state_basin_label": int(labels[start_index]),
                                "source_stale_index": source_stale_index,
                                "source_stale_object": str(stale_label),
                                "post_entry_initial_object": str(init_label),
                                "post_entry_initial_is_target_object": bool(init_label == context.target_ref),
                                "post_entry_initial_is_source_object": bool(init_label == context.source_ref),
                                "rollout_mode": rollout_mode,
                                "reencode_period": period,
                                "status": "ok",
                                "skip_reason": None,
                                **metrics,
                            }
                            grouped_rows.append(row)

                        current_vs_stale = _mse_ratio(
                            grouped_rows,
                            "current_global_no_reencode",
                            "stale_source_global_no_reencode",
                        )
                        for period in reencode_periods:
                            period_rows = [row for row in grouped_rows if int(row.get("reencode_period", -1)) == int(period)]
                            refreshed_vs_frozen = _mse_ratio(
                                period_rows,
                                "current_support_gated_periodic",
                                "frozen_source_gated_periodic",
                            )
                            reencoded_vs_no = _mse_ratio(
                                period_rows + grouped_rows,
                                "global_periodic_reencode",
                                "current_global_no_reencode",
                            )
                            for row in period_rows:
                                row["current_global_mse_vs_stale_source_ratio"] = current_vs_stale
                                row["refreshed_gated_mse_vs_frozen_source_gated_ratio"] = refreshed_vs_frozen
                                row["global_periodic_mse_vs_current_no_reencode_ratio"] = reencoded_vs_no
                        for row in grouped_rows:
                            row.setdefault("current_global_mse_vs_stale_source_ratio", current_vs_stale)
                            rows.append(row)

    return rows


def _mean(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    values = [_as_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def _format(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4g}"


def _write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    status_counts = Counter(str(row.get("status", "")) for row in rows)
    grouped: Dict[Tuple[str, str, str, str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[
                (
                    str(row.get("root_label")),
                    str(row.get("support_definition")),
                    str(row.get("object_kind")),
                    str(row.get("start_mode")),
                    str(row.get("rollout_mode")),
                    int(row.get("reencode_period", 0) or 0),
                )
            ].append(row)

    lines = [
        "# Periodic Support Refresh Summary",
        "",
        "Generated by `tools/evaluate_transition_rich_periodic_support_refresh.py`.",
        "",
        "This directly tests whether post-entry re-encoding refreshes support and whether the refreshed support routes later Koopman evolution differently from the previous source-basin support.",
        "",
        "## Coverage",
        "",
        f"- Total rows: {len(rows)}",
        f"- Status counts: {dict(sorted(status_counts.items()))}",
        "",
        "## Aggregate Metrics",
        "",
        "| root | support_definition | object | start | mode | period | rows | init_target | post_target | refresh | mse | refreshed/previous | reencode/no_reencode | route_target | fallback | chatter |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, group_rows in sorted(grouped.items()):
        root, support_definition, object_kind, start_mode, rollout_mode, period = key
        lines.append(
            (
                f"| `{root}` | `{support_definition}` | `{object_kind}` | `{start_mode}` | "
                f"`{rollout_mode}` | {period} | {len(group_rows)} | "
                f"{_format(_mean(group_rows, 'post_entry_initial_is_target_object'))} | "
                f"{_format(_mean(group_rows, 'post_reencode_target_dominance'))} | "
                f"{_format(_mean(group_rows, 'support_refresh_event_fraction'))} | "
                f"{_format(_mean(group_rows, 'forecast_mse'))} | "
                f"{_format(_mean(group_rows, 'refreshed_gated_mse_vs_frozen_source_gated_ratio'))} | "
                f"{_format(_mean(group_rows, 'global_periodic_mse_vs_current_no_reencode_ratio'))} | "
                f"{_format(_mean(group_rows, 'route_target_fraction'))} | "
                f"{_format(_mean(group_rows, 'route_fallback_fraction'))} | "
                f"{_format(_mean(group_rows, 'support_chatter_switch_rate'))} |"
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    rows_csvs: Sequence[str],
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    support_definitions: Sequence[Tuple[str, float]],
    reencode_periods: Sequence[int],
    start_modes: Sequence[str],
    specs_count: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
) -> None:
    manifest = {
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "rows_csvs": list(rows_csvs),
        "root_labels": list(root_labels),
        "systems": list(systems),
        "seeds": list(seeds),
        "support_definitions": [_stringify_support_definition(*item) for item in support_definitions],
        "reencode_periods": list(reencode_periods),
        "start_modes": list(start_modes),
        "specs_count": specs_count,
        "completed_specs": completed_specs,
        "row_count": len(rows),
        "failure_count": len(failures),
        "status_counts": dict(Counter(str(row.get("status", "")) for row in rows)),
        "rollout_modes": sorted({str(row.get("rollout_mode")) for row in rows if row.get("rollout_mode") is not None}),
        "construction": {
            "method": "state_bridge_linear_intervention",
            "mechanism_test": "post_entry_reencode_refresh_and_routing",
            "pre_steps": args.pre_steps,
            "bridge_steps": args.bridge_steps,
            "post_steps": args.post_steps,
            "continuation_horizon": args.continuation_horizon,
            "source_depth_fraction": args.source_depth_fraction,
            "target_depth_fraction": args.target_depth_fraction,
            "source_pre_min_fraction": args.source_pre_min_fraction,
            "final_target_min_fraction": args.final_target_min_fraction,
            "endpoint_rollout_steps": args.endpoint_rollout_steps,
            "use_dynamics_prior": args.use_dynamics_prior,
        },
        "argv": sys.argv,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _write_progress(
    path: Path,
    *,
    completed_specs: int,
    specs_count: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    elapsed_seconds: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "completed_specs": completed_specs,
                "specs_count": specs_count,
                "row_count": len(rows),
                "failure_count": len(failures),
                "elapsed_seconds": elapsed_seconds,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _flush(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    rows_csvs: Sequence[str],
    root_labels: Sequence[str],
    systems: Sequence[str],
    seeds: Sequence[int],
    support_definitions: Sequence[Tuple[str, float]],
    reencode_periods: Sequence[int],
    start_modes: Sequence[str],
    specs_count: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
) -> None:
    _write_csv(output_dir / "periodic_support_refresh_rows.csv", rows)
    _write_summary(output_dir / "periodic_support_refresh_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(list(failures), indent=2, sort_keys=True))
    _write_manifest(
        output_dir / "manifest.json",
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        reencode_periods=reencode_periods,
        start_modes=start_modes,
        specs_count=specs_count,
        completed_specs=completed_specs,
        rows=rows,
        failures=failures,
        status=status,
        elapsed_seconds=elapsed_seconds,
    )
    _write_progress(
        output_dir / "progress.json",
        completed_specs=completed_specs,
        specs_count=specs_count,
        rows=rows,
        failures=failures,
        elapsed_seconds=elapsed_seconds,
    )


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.num_transfers_per_pair = min(args.num_transfers_per_pair, 1)
        args.max_pairs_per_system = 1 if args.max_pairs_per_system == 0 else min(args.max_pairs_per_system, 1)
        args.pre_steps = min(args.pre_steps, 8)
        args.bridge_steps = min(args.bridge_steps, 8)
        args.post_steps = min(args.post_steps, 24)
        args.continuation_horizon = min(args.continuation_horizon, 12)
        args.endpoint_rollout_steps = min(args.endpoint_rollout_steps, 200)
        args.max_specs = 1 if args.max_specs == 0 else min(args.max_specs, 1)

    rows_csvs = _parse_csv_strings(args.rows_csvs)
    root_labels = _parse_csv_strings(args.root_labels)
    systems = _parse_csv_strings(args.systems)
    seeds = _parse_csv_ints(args.seeds)
    support_definitions = _parse_support_definitions(args.support_definitions)
    reencode_periods = _parse_periods(args.reencode_periods)
    start_modes = _parse_csv_strings(args.start_modes)
    invalid_start_modes = sorted(set(start_modes) - {"target_entry", "post_start"})
    if invalid_start_modes:
        raise ValueError(f"Unknown start modes: {invalid_start_modes}")

    specs = OPSEL._load_latest_specs(
        [Path(item) for item in rows_csvs],
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
    )
    if args.max_specs and args.max_specs > 0:
        specs = specs[: int(args.max_specs)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    start_time = time.time()
    progress_every = max(1, int(args.progress_every_runs))
    flush_every = max(0, int(args.flush_every_runs))
    specs_count = len(specs)

    _flush(
        output_dir,
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        reencode_periods=reencode_periods,
        start_modes=start_modes,
        specs_count=specs_count,
        completed_specs=0,
        rows=rows,
        failures=failures,
        status="running",
        elapsed_seconds=0.0,
    )

    for spec_index, spec in enumerate(specs, start=1):
        try:
            rows.extend(
                evaluate_run(
                    spec,
                    support_definitions=support_definitions,
                    num_transfers_per_pair=args.num_transfers_per_pair,
                    max_pairs_per_system=args.max_pairs_per_system,
                    pre_steps=args.pre_steps,
                    bridge_steps=args.bridge_steps,
                    post_steps=args.post_steps,
                    continuation_horizon=args.continuation_horizon,
                    reference_tail_steps=args.reference_tail_steps,
                    reencode_periods=reencode_periods,
                    start_modes=start_modes,
                    source_depth_fraction=args.source_depth_fraction,
                    target_depth_fraction=args.target_depth_fraction,
                    source_pre_min_fraction=args.source_pre_min_fraction,
                    final_target_min_fraction=args.final_target_min_fraction,
                    endpoint_rollout_steps=args.endpoint_rollout_steps,
                    eval_seed=args.eval_seed,
                    device=args.device,
                    family_jaccard_threshold=args.family_jaccard_threshold,
                    use_dynamics_prior=args.use_dynamics_prior,
                )
            )
            status = "ok"
            error = None
        except Exception as exc:  # pragma: no cover - keep batch moving across bad runs
            status = "failed"
            error = repr(exc)
            failures.append(
                {
                    "root_label": spec.root_label,
                    "system_key": spec.system_key,
                    "seed": spec.seed,
                    "run_dir": spec.run_dir,
                    "error": error,
                }
            )
        elapsed = time.time() - start_time
        if spec_index % progress_every == 0 or spec_index == specs_count:
            print(
                (
                    f"[{spec_index}/{specs_count}] {status} "
                    f"root={spec.root_label} system={spec.system_key} seed={spec.seed} "
                    f"rows={len(rows)} failures={len(failures)} elapsed_s={elapsed:.1f}"
                ),
                flush=True,
            )
        if flush_every > 0 and (spec_index % flush_every == 0 or spec_index == specs_count):
            _flush(
                output_dir,
                args=args,
                rows_csvs=rows_csvs,
                root_labels=root_labels,
                systems=systems,
                seeds=seeds,
                support_definitions=support_definitions,
                reencode_periods=reencode_periods,
                start_modes=start_modes,
                specs_count=specs_count,
                completed_specs=spec_index,
                rows=rows,
                failures=failures,
                status="running" if spec_index < specs_count else "complete",
                elapsed_seconds=elapsed,
            )

    _flush(
        output_dir,
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
        reencode_periods=reencode_periods,
        start_modes=start_modes,
        specs_count=specs_count,
        completed_specs=specs_count,
        rows=rows,
        failures=failures,
        status="complete",
        elapsed_seconds=time.time() - start_time,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "specs_count": specs_count,
                "row_count": len(rows),
                "failure_count": len(failures),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
