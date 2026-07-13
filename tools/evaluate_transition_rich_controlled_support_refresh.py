#!/usr/bin/env python3
"""Evaluate controlled-transfer support refresh without support-conditioned rollouts.

This evaluator tests the representation-level mechanism needed by the main
paper after removing support-conditioned local predictors:

Periodic re-encoding should update the latent support when the current state has
been moved from a source basin into a target basin.

The experiment constructs the same evaluation-only source-to-target state-space
bridges used by the controlled switching tools.  A latent is carried forward
under the learned global Koopman matrix between refresh events.  At each
scheduled event, the evaluator compares the stale pre-refresh support
``supp(K z)`` with the refreshed support obtained by encoding the controlled
current state.  No local maps, support-gated transitions, or basin-conditioned
rollout rules are fit or used.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def _load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REFRESH = _load_module(
    "evaluate_transition_rich_periodic_support_refresh.py",
    "evaluate_transition_rich_periodic_support_refresh_controlled_support",
)
CTRL = REFRESH.CTRL
REDUCER = REFRESH.REDUCER
OPSEL = REFRESH.OPSEL


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows_csvs", required=True, help="comma-separated forecasting_rows.csv files")
    parser.add_argument("--output_dir", required=True, help="directory for output artifacts")
    parser.add_argument("--root_labels", required=True, help="comma-separated root labels to include")
    parser.add_argument("--systems", default="", help="optional comma-separated system_key filter")
    parser.add_argument("--seeds", default="", help="optional comma-separated integer seed filter")
    parser.add_argument("--support_definitions", default="topk:8")
    parser.add_argument("--num_transfers_per_pair", type=int, default=2)
    parser.add_argument("--max_pairs_per_system", type=int, default=0, help="0 means all ordered pairs")
    parser.add_argument("--pre_steps", type=int, default=32)
    parser.add_argument("--bridge_steps", type=int, default=32)
    parser.add_argument("--post_steps", type=int, default=128)
    parser.add_argument("--reference_tail_steps", type=int, default=32)
    parser.add_argument("--reencode_periods", default="1,10", help="comma-separated positive periods")
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


def _as_float(value: object) -> Optional[float]:
    return REFRESH._as_float(value)


def _mean(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    values = [_as_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def _mean_bool(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                values.append(1.0)
            elif lowered in {"false", "0", "no"}:
                values.append(0.0)
    if not values:
        return None
    return float(np.mean(values))


def _format(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _fraction_equal(values: Sequence[object], target: object) -> Optional[float]:
    return REFRESH._fraction_equal(values, target)


def _switch_rate(values: Sequence[object]) -> Optional[float]:
    return REFRESH._switch_rate(values)


def _first_index(values: Sequence[object], target: object) -> Optional[int]:
    return REFRESH._first_index(values, target)


def _slice(values: Sequence[object], start_index: int) -> List[object]:
    return list(values[max(0, int(start_index)) :])


def _event_metrics(
    *,
    pre_labels: Sequence[object],
    post_labels: Sequence[object],
    event_steps: Sequence[int],
    source_ref: object,
    target_ref: object,
    analysis_start_index: int,
    prefix: str,
) -> Dict[str, object]:
    selected = [
        (pre, post, step)
        for pre, post, step in zip(pre_labels, post_labels, event_steps)
        if int(step) >= int(analysis_start_index)
    ]
    if not selected:
        return {
            f"{prefix}_event_count": 0,
            f"{prefix}_pre_target_fraction": None,
            f"{prefix}_post_target_fraction": None,
            f"{prefix}_switch_gain": None,
            f"{prefix}_source_to_target_fraction": None,
            f"{prefix}_target_drop_fraction": None,
            f"{prefix}_refresh_changed_fraction": None,
            f"{prefix}_first_post_target_lag": None,
        }
    pre = [item[0] for item in selected]
    post = [item[1] for item in selected]
    steps = [int(item[2]) for item in selected]
    pre_target = _fraction_equal(pre, target_ref)
    post_target = _fraction_equal(post, target_ref)
    first_target_event = _first_index(post, target_ref)
    first_lag = None if first_target_event is None else int(steps[first_target_event]) - int(analysis_start_index)
    source_to_target = [
        pre_item == source_ref and post_item == target_ref
        for pre_item, post_item in zip(pre, post)
    ]
    target_drop = [
        pre_item == target_ref and post_item != target_ref
        for pre_item, post_item in zip(pre, post)
    ]
    changed = [pre_item != post_item for pre_item, post_item in zip(pre, post)]
    return {
        f"{prefix}_event_count": len(selected),
        f"{prefix}_pre_target_fraction": pre_target,
        f"{prefix}_post_target_fraction": post_target,
        f"{prefix}_switch_gain": None if pre_target is None or post_target is None else float(post_target - pre_target),
        f"{prefix}_source_to_target_fraction": float(np.mean(source_to_target)),
        f"{prefix}_target_drop_fraction": float(np.mean(target_drop)),
        f"{prefix}_refresh_changed_fraction": float(np.mean(changed)),
        f"{prefix}_first_post_target_lag": first_lag,
    }


def _controlled_refresh_trace(
    *,
    controlled_latents: np.ndarray,
    global_k: np.ndarray,
    context,
    reencode_period: int,
    target_entry_index: int,
    post_start_index: int,
    control_kind: str,
) -> Dict[str, object]:
    if controlled_latents.shape[0] < 2:
        raise ValueError("controlled_latents must contain at least two states")

    latent = controlled_latents[0].astype(np.float32, copy=True)
    pre_labels_all: List[object] = []
    post_labels_all: List[object] = []
    event_steps: List[int] = []
    event_pre_labels: List[object] = []
    event_post_labels: List[object] = []

    for step in range(controlled_latents.shape[0] - 1):
        predicted_latent = REFRESH._predict_global(latent, global_k)
        pre_label = REFRESH._labels_for_context(context, predicted_latent.reshape(1, -1))[0][0]
        state_index = step + 1
        if state_index % int(reencode_period) == 0:
            refreshed_latent = controlled_latents[state_index].astype(np.float32, copy=False)
            post_label = REFRESH._labels_for_context(context, refreshed_latent.reshape(1, -1))[0][0]
            latent = refreshed_latent
            event_steps.append(state_index)
            event_pre_labels.append(pre_label)
            event_post_labels.append(post_label)
        else:
            post_label = pre_label
            latent = predicted_latent
        pre_labels_all.append(pre_label)
        post_labels_all.append(post_label)

    target_entry = int(target_entry_index)
    post_start = int(post_start_index)
    pre_post_entry = _slice(pre_labels_all, target_entry)
    post_post_entry = _slice(post_labels_all, target_entry)
    pre_post_phase = _slice(pre_labels_all, post_start)
    post_post_phase = _slice(post_labels_all, post_start)
    first_post_entry_target = _first_index(post_post_entry, context.target_ref)
    first_post_phase_target = _first_index(post_post_phase, context.target_ref)

    metrics: Dict[str, object] = {
        "control_kind": control_kind,
        "source_object": str(context.source_ref),
        "target_object": str(context.target_ref),
        "source_target_same_object": bool(context.source_target_same_object),
        "switch_interpretation_status": (
            "source_target_same_object"
            if context.source_target_same_object
            else "distinct_source_target_objects"
        ),
        "event_count": len(event_steps),
        "pre_target_dominance": _fraction_equal(pre_labels_all, context.target_ref),
        "post_target_dominance": _fraction_equal(post_labels_all, context.target_ref),
        "post_entry_pre_target_dominance": _fraction_equal(pre_post_entry, context.target_ref),
        "post_entry_post_target_dominance": _fraction_equal(post_post_entry, context.target_ref),
        "post_phase_pre_target_dominance": _fraction_equal(pre_post_phase, context.target_ref),
        "post_phase_post_target_dominance": _fraction_equal(post_post_phase, context.target_ref),
        "post_entry_target_switch_detected": (
            first_post_entry_target is not None and not context.source_target_same_object
        ),
        "post_phase_target_switch_detected": (
            first_post_phase_target is not None and not context.source_target_same_object
        ),
        "post_entry_first_post_target_lag": first_post_entry_target,
        "post_phase_first_post_target_lag": first_post_phase_target,
        "support_chatter_switch_rate": _switch_rate(post_labels_all),
        "post_entry_chatter_switch_rate": _switch_rate(post_post_entry),
        "post_phase_chatter_switch_rate": _switch_rate(post_post_phase),
    }
    metrics.update(
        _event_metrics(
            pre_labels=event_pre_labels,
            post_labels=event_post_labels,
            event_steps=event_steps,
            source_ref=context.source_ref,
            target_ref=context.target_ref,
            analysis_start_index=target_entry,
            prefix="post_entry",
        )
    )
    metrics.update(
        _event_metrics(
            pre_labels=event_pre_labels,
            post_labels=event_post_labels,
            event_steps=event_steps,
            source_ref=context.source_ref,
            target_ref=context.target_ref,
            analysis_start_index=post_start,
            prefix="post_phase",
        )
    )
    return metrics


def _skip_rows(
    spec,
    support_definitions: Sequence[Tuple[str, float]],
    reencode_periods: Sequence[int],
    *,
    status: str,
    skip_reason: str,
    label_source: str = "",
    center_source: str = "",
) -> List[Dict[str, object]]:
    rows = []
    for scheme, value in support_definitions:
        for object_kind in ("support", "family"):
            for period in reencode_periods:
                rows.append(
                    {
                        "root_label": spec.root_label,
                        "system_key": spec.system_key,
                        "system_name": spec.system_name,
                        "seed": spec.seed,
                        "run_dir": spec.run_dir,
                        "support_definition": REFRESH._stringify_support_definition(scheme, value),
                        "object_kind": object_kind,
                        "reencode_period": period,
                        "status": status,
                        "skip_reason": skip_reason,
                        "intervention_admissibility": "state_bridge_not_dynamics_control",
                        "mechanism_test": "controlled_transfer_global_k_support_refresh",
                        "label_source": label_source,
                        "center_source": center_source,
                    }
                )
    return rows


def evaluate_run(
    spec,
    *,
    support_definitions: Sequence[Tuple[str, float]],
    reencode_periods: Sequence[int],
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
    model.eval()
    global_k = model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
    state_dim = int(env.observation_size)
    basin_count = int(REDUCER.get_transition_rich_basin_count(spec.system_key))
    centers, center_source = CTRL._extract_centers(env, state_dim, basin_count)
    if centers is None:
        return _skip_rows(
            spec,
            support_definitions,
            reencode_periods,
            status="skipped",
            skip_reason="missing_transfer_centers",
            center_source=center_source,
        )
    if centers.shape[0] < basin_count:
        return _skip_rows(
            spec,
            support_definitions,
            reencode_periods,
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
                transfer, no_transfer, target_reference, metadata = CTRL._build_transfer_trajectory(
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
                        for period in reencode_periods:
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
                                    "support_definition": REFRESH._stringify_support_definition(scheme, value),
                                    "object_kind": object_kind,
                                    "reencode_period": period,
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
                        reencode_periods,
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
                "mechanism_test": "controlled_transfer_global_k_support_refresh",
                **metadata,
            }

            if not success or source_exit_index is None or target_entry_index is None:
                for scheme, value in support_definitions:
                    for object_kind in ("support", "family"):
                        for period in reencode_periods:
                            rows.append(
                                {
                                    **base,
                                    "support_definition": REFRESH._stringify_support_definition(scheme, value),
                                    "object_kind": object_kind,
                                    "reencode_period": period,
                                    "control_kind": "transfer",
                                    "status": "transfer_failed",
                                    "skip_reason": transfer_reason,
                                }
                            )
                continue

            transfer_latents = CTRL._encode_single(model, transfer, device)
            target_latents = CTRL._encode_single(model, target_reference, device)
            no_transfer_latents = (
                CTRL._encode_single(model, no_transfer, device) if include_no_transfer_controls else None
            )

            for scheme, value in support_definitions:
                support_definition = REFRESH._stringify_support_definition(scheme, value)
                for object_kind in ("support", "family"):
                    context = REFRESH._build_object_context(
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
                        for period in reencode_periods:
                            rows.append(
                                {
                                    **base,
                                    "support_definition": support_definition,
                                    "object_kind": object_kind,
                                    "reencode_period": period,
                                    "control_kind": "transfer",
                                    "status": "skipped",
                                    "skip_reason": "could_not_build_object_context",
                                }
                            )
                        continue

                    control_sequences = [("transfer", transfer_latents)]
                    if no_transfer_latents is not None:
                        control_sequences.append(("no_transfer", no_transfer_latents))

                    for period in reencode_periods:
                        for control_kind, controlled_latents in control_sequences:
                            if control_kind == "transfer":
                                analysis_target_entry = int(target_entry_index)
                                analysis_post_start = max(
                                    int(target_entry_index),
                                    int(metadata["post_start_index"]),
                                )
                            else:
                                analysis_target_entry = int(metadata["bridge_start_index"])
                                analysis_post_start = int(metadata["post_start_index"])
                            metrics = _controlled_refresh_trace(
                                controlled_latents=controlled_latents,
                                global_k=global_k,
                                context=context,
                                reencode_period=int(period),
                                target_entry_index=analysis_target_entry,
                                post_start_index=analysis_post_start,
                                control_kind=control_kind,
                            )
                            rows.append(
                                {
                                    **base,
                                    "support_definition": support_definition,
                                    "object_kind": object_kind,
                                    "reencode_period": int(period),
                                    "control_kind": control_kind,
                                    "analysis_target_entry_index": analysis_target_entry,
                                    "analysis_post_start_index": analysis_post_start,
                                    "status": "ok",
                                    "skip_reason": None,
                                    **metrics,
                                }
                            )

    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    REFRESH._write_csv(path, rows)


def _write_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    status_counts = Counter(str(row.get("status", "")) for row in rows)
    grouped: Dict[Tuple[str, str, str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[
                (
                    str(row.get("root_label")),
                    str(row.get("support_definition")),
                    str(row.get("object_kind")),
                    str(row.get("control_kind")),
                    int(row.get("reencode_period", 0)),
                )
            ].append(row)

    lines = [
        "# Controlled Support Refresh Summary",
        "",
        "Generated by `tools/evaluate_transition_rich_controlled_support_refresh.py`.",
        "",
        "This tests support switching under controlled source-to-target basin transfer.  The latent is advanced with the learned global Koopman matrix; at scheduled events the current controlled state is encoded and its refreshed support is compared with the stale pre-refresh support.  No support-conditioned local maps or support-gated transitions are used.",
        "",
        f"Status counts: `{dict(status_counts)}`",
        "",
        "| root | support_definition | object | control | period | rows | distinct | post pre-target | post post-target | gain | source->target | target drop | lag | chatter |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(grouped):
        root, support_definition, object_kind, control_kind, period = key
        group_rows = grouped[key]
        same_fraction = _mean_bool(group_rows, "source_target_same_object")
        distinct_fraction = None if same_fraction is None else 1.0 - float(same_fraction)
        lines.append(
            "| "
            f"`{root}` | `{support_definition}` | `{object_kind}` | `{control_kind}` | {period} | "
            f"{len(group_rows)} | {_format(distinct_fraction)} | "
            f"{_format(_mean(group_rows, 'post_phase_pre_target_fraction'))} | "
            f"{_format(_mean(group_rows, 'post_phase_post_target_fraction'))} | "
            f"{_format(_mean(group_rows, 'post_phase_switch_gain'))} | "
            f"{_format(_mean(group_rows, 'post_phase_source_to_target_fraction'))} | "
            f"{_format(_mean(group_rows, 'post_phase_target_drop_fraction'))} | "
            f"{_format(_mean(group_rows, 'post_phase_first_post_target_lag'))} | "
            f"{_format(_mean(group_rows, 'post_phase_chatter_switch_rate'))} |"
        )
    path.write_text("\n".join(lines) + "\n")


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
    specs_count: int,
    completed_specs: int,
    rows: Sequence[Dict[str, object]],
    failures: Sequence[Dict[str, object]],
    status: str,
    elapsed_seconds: float,
) -> None:
    _write_csv(output_dir / "controlled_support_refresh_rows.csv", rows)
    _write_summary(output_dir / "controlled_support_refresh_summary.md", rows)
    (output_dir / "failures.json").write_text(json.dumps(list(failures), indent=2, sort_keys=True))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rows_csvs": list(rows_csvs),
                "root_labels": list(root_labels),
                "systems": list(systems),
                "seeds": list(seeds),
                "support_definitions": [REFRESH._stringify_support_definition(*item) for item in support_definitions],
                "reencode_periods": [int(item) for item in reencode_periods],
                "include_no_transfer_controls": bool(args.include_no_transfer_controls),
                "num_transfers_per_pair": int(args.num_transfers_per_pair),
                "max_pairs_per_system": int(args.max_pairs_per_system),
                "pre_steps": int(args.pre_steps),
                "bridge_steps": int(args.bridge_steps),
                "post_steps": int(args.post_steps),
                "endpoint_rollout_steps": int(args.endpoint_rollout_steps),
                "specs_count": int(specs_count),
                "completed_specs": int(completed_specs),
                "remaining_specs": max(0, int(specs_count) - int(completed_specs)),
                "row_count": len(rows),
                "failure_count": len(failures),
                "status": status,
                "elapsed_seconds": float(elapsed_seconds),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.num_transfers_per_pair = 1
        args.max_pairs_per_system = 1 if args.max_pairs_per_system == 0 else min(args.max_pairs_per_system, 1)
        args.pre_steps = min(args.pre_steps, 8)
        args.bridge_steps = min(args.bridge_steps, 8)
        args.post_steps = min(args.post_steps, 24)
        args.endpoint_rollout_steps = min(args.endpoint_rollout_steps, 200)
        args.max_specs = 1 if args.max_specs == 0 else min(args.max_specs, 1)

    rows_csvs = REFRESH._parse_csv_strings(args.rows_csvs)
    root_labels = REFRESH._parse_csv_strings(args.root_labels)
    systems = REFRESH._parse_csv_strings(args.systems)
    seeds = REFRESH._parse_csv_ints(args.seeds)
    support_definitions = REFRESH._parse_support_definitions(args.support_definitions)
    reencode_periods = REFRESH._parse_periods(args.reencode_periods)

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
                    reencode_periods=reencode_periods,
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
            run_status = "ok"
            error = None
        except Exception as exc:  # pragma: no cover - keep batch moving across bad runs
            run_status = "failed"
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
                    f"[{spec_index}/{specs_count}] {run_status} "
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
