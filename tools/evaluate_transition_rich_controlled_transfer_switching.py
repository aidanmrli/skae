#!/usr/bin/env python3
"""Evaluate support switching under controlled source-to-target basin transfer.

This is an evaluation-only first pass. It constructs deterministic state-space
bridge interventions on existing fixed-17 checkpoints, measures the benchmark
basin crossing time, and then asks whether exact supports and support families
switch near that crossing. Basin labels are used only for evaluation.
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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

EPS = 1e-12


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REDUCER = _load_module(
    "reduce_transition_rich_interpretability_metrics.py",
    "reduce_transition_rich_interpretability_metrics_controlled_transfer",
)
OPSEL = _load_module(
    "evaluate_transition_rich_operator_selection.py",
    "evaluate_transition_rich_operator_selection_controlled_transfer",
)


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
    parser.add_argument("--post_steps", type=int, default=96)
    parser.add_argument("--reference_tail_steps", type=int, default=32)
    parser.add_argument("--source_depth_fraction", type=float, default=0.12)
    parser.add_argument("--target_depth_fraction", type=float, default=0.12)
    parser.add_argument("--source_pre_min_fraction", type=float, default=0.80)
    parser.add_argument("--final_target_min_fraction", type=float, default=0.80)
    parser.add_argument("--endpoint_rollout_steps", type=int, default=5000)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.5)
    parser.add_argument("--include_no_transfer_controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_specs", type=int, default=0, help="0 means no limit")
    parser.add_argument("--progress_every_runs", type=int, default=1)
    parser.add_argument("--flush_every_runs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="override to a tiny evaluator subset")
    return parser.parse_args()


def _parse_csv_strings(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_support_definitions(raw: str) -> List[Tuple[str, float]]:
    definitions: List[Tuple[str, float]] = []
    for item in _parse_csv_strings(raw):
        if ":" not in item:
            raise ValueError(f"Support definition must be scheme:value, got '{item}'")
        scheme, raw_value = item.split(":", 1)
        scheme = scheme.strip()
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
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


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


def _tensor_from_centers(raw: object) -> Optional[torch.Tensor]:
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        if raw.ndim == 2 and raw.shape[0] > 0:
            return raw.detach().to(dtype=torch.float32).cpu()
        return None
    try:
        tensor = torch.tensor(raw, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tensor.ndim == 2 and tensor.shape[0] > 0:
        return tensor
    return None


def _centers_from_wells(raw: object) -> Optional[torch.Tensor]:
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        if raw.ndim == 2 and raw.shape[1] >= 2:
            return raw[:, :2].detach().to(dtype=torch.float32).cpu()
        return None
    centers = []
    try:
        iterator = list(raw)
    except TypeError:
        return None
    for item in iterator:
        try:
            centers.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError, IndexError):
            return None
    if not centers:
        return None
    return torch.tensor(centers, dtype=torch.float32)


def _pad_or_trim_centers(centers: torch.Tensor, state_dim: int) -> torch.Tensor:
    if centers.shape[1] == state_dim:
        return centers.to(dtype=torch.float32)
    if centers.shape[1] > state_dim:
        return centers[:, :state_dim].to(dtype=torch.float32)
    padded = torch.zeros((centers.shape[0], state_dim), dtype=torch.float32)
    padded[:, : centers.shape[1]] = centers.to(dtype=torch.float32)
    return padded


def _extract_centers(env, state_dim: int, basin_count: int) -> Tuple[Optional[torch.Tensor], str]:
    direct = _tensor_from_centers(getattr(env, "points", None))
    if direct is not None:
        return _pad_or_trim_centers(direct, state_dim), "env.points"

    base = getattr(env, "unwrapped", env)
    system = getattr(base, "system", None)
    candidates = []
    if system is not None:
        for name in ("well_centers", "centers", "room_centers", "dipoles", "points"):
            tensor = _tensor_from_centers(getattr(system, name, None))
            if tensor is not None:
                candidates.append((tensor, f"system.{name}"))
        for name in ("_wells", "wells"):
            tensor = _centers_from_wells(getattr(system, name, None))
            if tensor is not None:
                candidates.append((tensor, f"system.{name}"))

    for tensor, source in candidates:
        if tensor.shape[0] >= basin_count:
            return _pad_or_trim_centers(tensor[:basin_count], state_dim), source
    if candidates:
        tensor, source = max(candidates, key=lambda item: item[0].shape[0])
        return _pad_or_trim_centers(tensor, state_dim), f"{source}:insufficient_count"
    return None, "missing_centers"


def _label_states_for_evaluation(
    env,
    states: torch.Tensor,
    centers: torch.Tensor,
    *,
    endpoint_rollout_steps: int,
) -> Tuple[Optional[np.ndarray], str, Optional[str]]:
    native_errors = []
    native_candidates = [("env", env)]
    base = getattr(env, "unwrapped", env)
    if base is not env:
        native_candidates.append(("env.unwrapped", base))
    for owner_name, owner in native_candidates:
        if not hasattr(owner, "basin_label"):
            continue
        try:
            labels = owner.basin_label(states)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            labels_np = labels.detach().cpu().to(dtype=torch.long).numpy().reshape(-1)
            if labels_np.shape[0] != int(states.shape[0]):
                native_errors.append(
                    f"{owner_name}.basin_label_shape:{tuple(labels.shape)} expected_first_dim={int(states.shape[0])}"
                )
                continue
            return labels_np, f"{owner_name}.native_basin_label", None
        except Exception as exc:  # pragma: no cover - runtime environment variability
            native_errors.append(f"{owner_name}.basin_label_failed:{repr(exc)}")

    try:
        endpoints = REDUCER._long_rollout(env, states.float(), endpoint_rollout_steps)
        labels = torch.cdist(endpoints.float(), centers.float()).argmin(dim=1)
        source = "endpoint_rollout_nearest_center"
        if native_errors:
            source = "native_failed_then_" + source
        return labels.detach().cpu().to(dtype=torch.long).numpy().reshape(-1), source, None
    except Exception as exc:  # pragma: no cover - runtime environment variability
        errors = ";".join(native_errors + [f"endpoint_rollout_failed:{repr(exc)}"])
        return None, "labeling_failed", errors


def _roll_unforced(env, init_state: torch.Tensor, steps: int) -> torch.Tensor:
    states = [init_state.detach().clone().float()]
    current = states[0]
    for _ in range(max(0, int(steps))):
        current = env.step(current).detach().clone().float()
        states.append(current)
    return torch.stack(states, dim=0)


def _build_transfer_trajectory(
    env,
    centers: torch.Tensor,
    source: int,
    target: int,
    *,
    pre_steps: int,
    bridge_steps: int,
    post_steps: int,
    source_depth_fraction: float,
    target_depth_fraction: float,
    jitter_scale: float,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    source_center = centers[source].float()
    target_center = centers[target].float()
    delta = target_center - source_center
    distance = torch.linalg.vector_norm(delta).clamp_min(1e-8)
    direction = delta / distance
    normal = torch.zeros_like(direction)
    if normal.numel() >= 2:
        normal[0] = -direction[1]
        normal[1] = direction[0]
    if jitter_scale > 0.0 and normal.numel() >= 2:
        jitter = (torch.rand((), generator=rng).item() * 2.0 - 1.0) * float(jitter_scale) * float(distance)
    else:
        jitter = 0.0

    source_start = source_center + float(source_depth_fraction) * distance * direction + jitter * normal
    target_entry = target_center - float(target_depth_fraction) * distance * direction + jitter * normal

    pre = _roll_unforced(env, source_start, pre_steps)
    bridge_states = []
    bridge_origin = pre[-1]
    for step_idx in range(1, max(1, int(bridge_steps)) + 1):
        alpha = float(step_idx) / float(max(1, int(bridge_steps)))
        bridge_states.append((1.0 - alpha) * bridge_origin + alpha * target_entry)
    bridge = torch.stack(bridge_states, dim=0)
    post = _roll_unforced(env, bridge[-1], post_steps)[1:]
    transfer = torch.cat([pre, bridge, post], dim=0).float()

    no_transfer = _roll_unforced(env, source_start, transfer.shape[0] - 1)
    target_reference = _roll_unforced(env, target_entry, post_steps)
    metadata = {
        "construction_method": "state_bridge_linear_intervention",
        "source_start_index": 0,
        "bridge_start_index": int(pre.shape[0]),
        "post_start_index": int(pre.shape[0] + bridge.shape[0]),
        "total_states": int(transfer.shape[0]),
        "source_depth_fraction": float(source_depth_fraction),
        "target_depth_fraction": float(target_depth_fraction),
        "jitter_scale": float(jitter_scale),
    }
    return transfer, no_transfer, target_reference, metadata


def _first_source_exit(labels: np.ndarray, source: int) -> Optional[int]:
    crossings = np.flatnonzero(labels != int(source))
    if crossings.size == 0:
        return None
    return int(crossings[0])


def _first_target_entry(labels: np.ndarray, target: int, *, start_index: int = 0) -> Optional[int]:
    entries = np.flatnonzero(labels[max(0, int(start_index)) :] == int(target))
    if entries.size == 0:
        return None
    return int(entries[0] + max(0, int(start_index)))


def _mode(values: Sequence[object]) -> Optional[object]:
    if len(values) == 0:
        return None
    return Counter(values).most_common(1)[0][0]


def _fraction_equal(values: np.ndarray, target: object) -> Optional[float]:
    if values.size == 0 or target is None:
        return None
    return float(np.mean(values == target))


def _switch_rate(values: np.ndarray) -> Optional[float]:
    if values.size <= 1:
        return None
    return float(np.mean(values[1:] != values[:-1]))


def _entropy(values: Sequence[object]) -> Tuple[Optional[float], Optional[float]]:
    if len(values) == 0:
        return None, None
    counts = Counter(values)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        prob = float(count) / max(total, 1.0)
        entropy -= prob * math.log(prob + EPS)
    norm = entropy / math.log(max(len(counts), 2))
    return float(entropy), float(norm)


def _first_target_index(values: np.ndarray, target_ref: object, start_index: int) -> Optional[int]:
    if target_ref is None:
        return None
    for idx in range(max(0, int(start_index)), int(values.shape[0])):
        if values[idx] == target_ref:
            return idx
    return None


def _object_labels_for_sequences(
    sequence_latents: Dict[str, np.ndarray],
    *,
    scheme: str,
    value: float,
    family_jaccard_threshold: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    names = list(sequence_latents.keys())
    masks_by_name: Dict[str, np.ndarray] = {}
    support_by_name: Dict[str, np.ndarray] = {}
    lengths = []
    for name in names:
        wrapped = sequence_latents[name][None, :, :]
        support_mask = REDUCER._support_mask(wrapped, scheme=scheme, value=value)
        masks_by_name[name] = support_mask[0]
        support_by_name[name] = REDUCER._support_keys(support_mask)[0].astype(object)
        lengths.append(support_mask.shape[1])

    # Family IDs must be assigned in one shared clustering problem. Otherwise
    # family label 0 on the transfer and reference trajectories can denote
    # different support prototypes, making switch timing uninterpretable.
    joint_mask = np.concatenate([masks_by_name[name] for name in names], axis=0)[None, :, :]
    joint_family = REDUCER.support_family_labels(
        joint_mask,
        min_jaccard=family_jaccard_threshold,
    )[0].astype(object)

    output: Dict[str, Dict[str, np.ndarray]] = {}
    cursor = 0
    for name, length in zip(names, lengths):
        output[name] = {
            "support": support_by_name[name],
            "family": joint_family[cursor : cursor + length],
        }
        cursor += length
    return output


def _encode_single(model, states: torch.Tensor, device: str) -> np.ndarray:
    latents = REDUCER._encode_trajectories(model, states.unsqueeze(0).float(), device)
    return latents[0]


def _status_from_labels(
    labels: np.ndarray,
    source: int,
    target: int,
    *,
    pre_end_index: int,
    final_target_min_fraction: float,
    source_pre_min_fraction: float,
) -> Tuple[bool, str, Optional[int], Optional[int], float, float]:
    pre_labels = labels[: max(1, pre_end_index)]
    tail_count = max(1, min(16, labels.shape[0]))
    tail_labels = labels[-tail_count:]
    source_pre_fraction = float(np.mean(pre_labels == int(source)))
    final_target_fraction = float(np.mean(tail_labels == int(target)))
    source_exit_index = _first_source_exit(labels, source)
    target_entry_index = _first_target_entry(
        labels,
        target,
        start_index=0 if source_exit_index is None else source_exit_index,
    )
    if source_pre_fraction < source_pre_min_fraction:
        return (
            False,
            "source_unstable_pre",
            source_exit_index,
            target_entry_index,
            source_pre_fraction,
            final_target_fraction,
        )
    if source_exit_index is None:
        return (
            False,
            "no_measured_source_exit",
            source_exit_index,
            target_entry_index,
            source_pre_fraction,
            final_target_fraction,
        )
    if target_entry_index is None:
        return (
            False,
            "no_measured_target_entry",
            source_exit_index,
            target_entry_index,
            source_pre_fraction,
            final_target_fraction,
        )
    if final_target_fraction < final_target_min_fraction:
        return (
            False,
            "target_not_reached",
            source_exit_index,
            target_entry_index,
            source_pre_fraction,
            final_target_fraction,
        )
    return True, "ok", source_exit_index, target_entry_index, source_pre_fraction, final_target_fraction


def _metrics_for_transfer(
    object_labels: np.ndarray,
    target_reference_labels: np.ndarray,
    *,
    source_exit_index: int,
    target_entry_index: int,
    pre_end_index: int,
    post_start_index: int,
    reference_tail_steps: int,
) -> Dict[str, object]:
    source_window = object_labels[: max(1, int(source_exit_index))]
    source_ref = _mode(source_window.tolist())
    target_tail = target_reference_labels[-max(1, int(reference_tail_steps)) :]
    target_ref = _mode(target_tail.tolist())
    same_object = source_ref == target_ref
    pre_target_window = object_labels[: max(1, int(target_entry_index))]
    post_window = object_labels[int(target_entry_index) :]
    post_phase = object_labels[max(int(post_start_index), int(target_entry_index)) :]
    first_target = None if same_object else _first_target_index(object_labels, target_ref, target_entry_index)
    first_target_post_phase = (
        None
        if same_object
        else _first_target_index(object_labels, target_ref, max(int(post_start_index), int(target_entry_index)))
    )
    entropy, entropy_norm = _entropy(object_labels.tolist())
    pre_target_fraction = _fraction_equal(pre_target_window, target_ref)
    return {
        "source_object": str(source_ref),
        "target_object": str(target_ref),
        "source_target_same_object": bool(same_object),
        "switch_interpretation_status": "source_target_same_object" if same_object else "distinct_source_target_objects",
        "pre_source_dominance": _fraction_equal(source_window, source_ref),
        "pre_target_dominance": pre_target_fraction,
        "post_target_dominance": _fraction_equal(post_window, target_ref),
        "post_source_dominance": _fraction_equal(post_window, source_ref),
        "post_phase_target_dominance": _fraction_equal(post_phase, target_ref),
        "source_exit_to_target_entry_steps": int(target_entry_index - source_exit_index),
        "crossing_lag_steps": None if first_target is None else int(first_target - target_entry_index),
        "post_phase_crossing_lag_steps": (
            None if first_target_post_phase is None else int(first_target_post_phase - max(int(post_start_index), int(target_entry_index)))
        ),
        "target_switch_detected": first_target is not None and not same_object,
        "post_phase_target_switch_detected": first_target_post_phase is not None and not same_object,
        "premature_switch_rate": pre_target_fraction,
        "chatter_switch_rate": _switch_rate(object_labels),
        "post_chatter_switch_rate": _switch_rate(post_window),
        "support_entropy": entropy,
        "support_entropy_normalized": entropy_norm,
        "pre_end_index": int(pre_end_index),
        "post_start_index": int(post_start_index),
    }


def _metrics_for_no_transfer(
    object_labels: np.ndarray,
    *,
    pre_end_index: int,
) -> Dict[str, object]:
    source_ref = _mode(object_labels[: max(1, int(pre_end_index))].tolist())
    post_source = object_labels[int(pre_end_index) :]
    entropy, entropy_norm = _entropy(object_labels.tolist())
    source_dom = _fraction_equal(post_source, source_ref)
    return {
        "source_object": str(source_ref),
        "target_object": None,
        "source_target_same_object": None,
        "switch_interpretation_status": "no_transfer_control",
        "pre_source_dominance": _fraction_equal(object_labels[: max(1, int(pre_end_index))], source_ref),
        "pre_target_dominance": None,
        "post_target_dominance": None,
        "post_source_dominance": source_dom,
        "post_phase_target_dominance": None,
        "crossing_lag_steps": None,
        "source_exit_to_target_entry_steps": None,
        "target_switch_detected": None,
        "premature_switch_rate": None,
        "false_switch_rate": None if source_dom is None else 1.0 - float(source_dom),
        "chatter_switch_rate": _switch_rate(object_labels),
        "post_chatter_switch_rate": _switch_rate(post_source),
        "support_entropy": entropy,
        "support_entropy_normalized": entropy_norm,
        "pre_end_index": int(pre_end_index),
        "post_start_index": int(pre_end_index),
    }


def _ordered_pairs(num_basins: int, max_pairs: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    pairs = [(i, j) for i in range(num_basins) for j in range(num_basins) if i != j]
    if max_pairs and max_pairs > 0 and len(pairs) > max_pairs:
        order = rng.permutation(len(pairs))[:max_pairs]
        pairs = [pairs[int(idx)] for idx in order]
    return pairs


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
                    "control_kind": "transfer",
                    "status": status,
                    "skip_reason": skip_reason,
                    "crossing_index": None,
                    "source_exit_index": None,
                    "target_entry_index": None,
                    "intervention_admissibility": "state_bridge_not_dynamics_control",
                    "label_source": label_source,
                    "center_source": center_source,
                }
            )
    return rows


def evaluate_run(
    spec,
    *,
    support_definitions: Sequence[Tuple[str, float]],
    num_transfers_per_pair: int,
    max_pairs_per_system: int,
    pre_steps: int,
    bridge_steps: int,
    post_steps: int,
    reference_tail_steps: int,
    source_depth_fraction: float,
    target_depth_fraction: float,
    source_pre_min_fraction: float,
    final_target_min_fraction: float,
    endpoint_rollout_steps: int,
    eval_seed: int,
    device: str,
    family_jaccard_threshold: float,
    include_no_transfer_controls: bool,
) -> List[Dict[str, object]]:
    checkpoint_path = Path(spec.run_dir) / "checkpoint.pt"
    _cfg, env, model = REDUCER._load_checkpoint_model(checkpoint_path, spec.system_key, device)
    state_dim = int(env.observation_size)
    basin_count = int(REDUCER.get_transition_rich_basin_count(spec.system_key))
    centers, center_source = _extract_centers(env, state_dim, basin_count)
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
    pairs = _ordered_pairs(basin_count, max_pairs_per_system, rng_np)
    rows: List[Dict[str, object]] = []

    for pair_index, (source, target) in enumerate(pairs):
        for transfer_index in range(max(1, int(num_transfers_per_pair))):
            torch_rng = torch.Generator().manual_seed(
                int(eval_seed) + 10_000 * int(spec.seed) + 101 * pair_index + transfer_index
            )
            jitter_scale = 0.01 * float(transfer_index)
            try:
                transfer, no_transfer, target_reference, metadata = _build_transfer_trajectory(
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
                for scheme, value in support_definitions:
                    for object_kind in ("support", "family"):
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
                                "support_definition": _stringify_support_definition(scheme, value),
                                "object_kind": object_kind,
                                "control_kind": "transfer",
                                "status": "skipped",
                                "skip_reason": "trajectory_construction_failed",
                                "error": repr(exc),
                                "center_source": center_source,
                            }
                        )
                continue

            labels, label_source, label_error = _label_states_for_evaluation(
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
            ) = _status_from_labels(
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
                "crossing_index": target_entry_index,
                "source_exit_index": source_exit_index,
                "target_entry_index": target_entry_index,
                "source_pre_fraction": source_pre_fraction,
                "final_target_fraction": final_target_fraction,
                "intervention_admissibility": "state_bridge_not_dynamics_control",
                **metadata,
            }

            transfer_latents = _encode_single(model, transfer, device)
            target_latents = _encode_single(model, target_reference, device)
            no_transfer_latents = _encode_single(model, no_transfer, device) if include_no_transfer_controls else None

            for scheme, value in support_definitions:
                support_definition = _stringify_support_definition(scheme, value)
                sequence_latents = {
                    "transfer": transfer_latents,
                    "target_reference": target_latents,
                }
                if no_transfer_latents is not None:
                    sequence_latents["no_transfer"] = no_transfer_latents
                sequence_objects = _object_labels_for_sequences(
                    sequence_latents,
                    scheme=scheme,
                    value=value,
                    family_jaccard_threshold=family_jaccard_threshold,
                )

                for object_kind in ("support", "family"):
                    object_labels = sequence_objects["transfer"][object_kind]
                    target_reference_labels = sequence_objects["target_reference"][object_kind]
                    if success and source_exit_index is not None and target_entry_index is not None:
                        metrics = _metrics_for_transfer(
                            object_labels,
                            target_reference_labels,
                            source_exit_index=int(source_exit_index),
                            target_entry_index=int(target_entry_index),
                            pre_end_index=int(metadata["bridge_start_index"]),
                            post_start_index=int(metadata["post_start_index"]),
                            reference_tail_steps=reference_tail_steps,
                        )
                        status = "ok"
                        skip_reason = None
                    else:
                        metrics = {}
                        status = "transfer_failed"
                        skip_reason = transfer_reason
                    rows.append(
                        {
                            **base,
                            "support_definition": support_definition,
                            "object_kind": object_kind,
                            "control_kind": "transfer",
                            "status": status,
                            "skip_reason": skip_reason,
                            **metrics,
                        }
                    )

                    if include_no_transfer_controls and "no_transfer" in sequence_objects:
                        control_labels = sequence_objects["no_transfer"][object_kind]
                        rows.append(
                            {
                                **base,
                                "support_definition": support_definition,
                                "object_kind": object_kind,
                                "control_kind": "no_transfer",
                                "status": "ok",
                                "skip_reason": None,
                                "transfer_success": False,
                                "transfer_failure_reason": None,
                                "crossing_index": None,
                                "source_exit_index": None,
                                "target_entry_index": None,
                                **_metrics_for_no_transfer(
                                    control_labels,
                                    pre_end_index=int(metadata["bridge_start_index"]),
                                ),
                            }
                        )
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
    return f"{float(value):.4f}"


def _write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    status_counts = Counter(str(row.get("status", "")) for row in rows)
    control_counts = Counter(str(row.get("control_kind", "")) for row in rows)
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[
                (
                    str(row.get("root_label")),
                    str(row.get("support_definition")),
                    str(row.get("object_kind")),
                    str(row.get("control_kind")),
                )
            ].append(row)

    lines = [
        "# Controlled Transfer Switching Summary",
        "",
        "This summary is generated by `tools/evaluate_transition_rich_controlled_transfer_switching.py`.",
        "",
        "## Coverage",
        "",
        f"- Total rows: {len(rows)}",
        f"- Status counts: {dict(sorted(status_counts.items()))}",
        f"- Control counts: {dict(sorted(control_counts.items()))}",
        "",
        "## Aggregate Metrics",
        "",
        "For transfer rows, `switch_rows` excludes rows where the source and target reference object are identical.",
        "The `pre_source`, `post_target`, `post_phase_target`, `target_lag`, `post_phase_lag`, `premature`, and `chatter` columns are computed on `switch_rows`.",
        "",
        "| root | support_definition | object | control | rows | switch_rows | pre_source | post_target | post_phase_target | same_obj | target_lag | post_phase_lag | exit_to_entry | premature | chatter | false_switch |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, group_rows in sorted(grouped.items()):
        root, support_definition, object_kind, control_kind = key
        metric_rows = group_rows
        switch_rows: Sequence[Dict[str, object]] = []
        if control_kind == "transfer":
            switch_rows = [row for row in group_rows if row.get("source_target_same_object") is False]
            metric_rows = switch_rows
        lines.append(
            (
                f"| `{root}` | `{support_definition}` | `{object_kind}` | `{control_kind}` | "
                f"{len(group_rows)} | {len(switch_rows) if control_kind == 'transfer' else 'N/A'} | "
                f"{_format(_mean(metric_rows, 'pre_source_dominance'))} | "
                f"{_format(_mean(metric_rows, 'post_target_dominance'))} | "
                f"{_format(_mean(metric_rows, 'post_phase_target_dominance'))} | "
                f"{_format(_mean(group_rows, 'source_target_same_object'))} | "
                f"{_format(_mean(metric_rows, 'crossing_lag_steps'))} | "
                f"{_format(_mean(metric_rows, 'post_phase_crossing_lag_steps'))} | "
                f"{_format(_mean(group_rows, 'source_exit_to_target_entry_steps'))} | "
                f"{_format(_mean(metric_rows, 'premature_switch_rate'))} | "
                f"{_format(_mean(metric_rows, 'post_chatter_switch_rate' if control_kind == 'transfer' else 'chatter_switch_rate'))} | "
                f"{_format(_mean(group_rows, 'false_switch_rate'))} |"
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
        "specs_count": specs_count,
        "completed_specs": completed_specs,
        "row_count": len(rows),
        "failure_count": len(failures),
        "status_counts": dict(Counter(str(row.get("status", "")) for row in rows)),
        "control_counts": dict(Counter(str(row.get("control_kind", "")) for row in rows)),
        "construction": {
            "method": "state_bridge_linear_intervention",
            "pre_steps": args.pre_steps,
            "bridge_steps": args.bridge_steps,
            "post_steps": args.post_steps,
            "source_depth_fraction": args.source_depth_fraction,
            "target_depth_fraction": args.target_depth_fraction,
            "source_pre_min_fraction": args.source_pre_min_fraction,
            "final_target_min_fraction": args.final_target_min_fraction,
            "endpoint_rollout_steps": args.endpoint_rollout_steps,
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
    specs_count: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
) -> None:
    _write_csv(output_dir / "controlled_transfer_switching_rows.csv", rows)
    _write_summary(output_dir / "controlled_transfer_switching_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(list(failures), indent=2, sort_keys=True))
    _write_manifest(
        output_dir / "manifest.json",
        args=args,
        rows_csvs=rows_csvs,
        root_labels=root_labels,
        systems=systems,
        seeds=seeds,
        support_definitions=support_definitions,
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
        args.post_steps = min(args.post_steps, 16)
        args.endpoint_rollout_steps = min(args.endpoint_rollout_steps, 200)
        args.max_specs = 1 if args.max_specs == 0 else min(args.max_specs, 1)

    rows_csvs = _parse_csv_strings(args.rows_csvs)
    root_labels = _parse_csv_strings(args.root_labels)
    systems = _parse_csv_strings(args.systems)
    seeds = _parse_csv_ints(args.seeds)
    support_definitions = _parse_support_definitions(args.support_definitions)
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
                    reference_tail_steps=args.reference_tail_steps,
                    source_depth_fraction=args.source_depth_fraction,
                    target_depth_fraction=args.target_depth_fraction,
                    source_pre_min_fraction=args.source_pre_min_fraction,
                    final_target_min_fraction=args.final_target_min_fraction,
                    endpoint_rollout_steps=args.endpoint_rollout_steps,
                    eval_seed=args.eval_seed,
                    device=args.device,
                    family_jaccard_threshold=args.family_jaccard_threshold,
                    include_no_transfer_controls=args.include_no_transfer_controls,
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
