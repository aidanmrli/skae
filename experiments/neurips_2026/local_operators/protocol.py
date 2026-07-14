"""Versioned scientific contract for the published support-routed run.

This module freezes scientific choices.  The CLI intentionally exposes only
execution and artifact-retention controls; changing route fitting or checkpoint
selection requires a new protocol name and new experiment results.
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from skae.config import Config
from skae.data import VectorWrapper, generate_trajectory
from skae.evaluation import EvaluationSettings
from experiments.neurips_2026.local_operators.contract import (
    FINAL_EVALUATION_BATCH_SIZE,
    FINAL_EVALUATION_SEED_OFFSET,
    FAMILY_CLUSTERING_RULE,
    FAMILY_JACCARD_THRESHOLD,
    FAMILY_REPRESENTATIVE_RULE,
    FIT_CONFIGURED_ROWS,
    FIT_DUPLICATION_FACTOR,
    FIT_SEED_OFFSET,
    FIT_SOURCE_TRANSITIONS,
    FIT_STATES,
    FIT_SUPPORTS_CONSIDERED,
    FIT_TRANSITIONS,
    FIT_UNIQUE_SOURCE_TRANSITIONS,
    FIT_UNIQUE_TRAJECTORIES,
    LEGACY_ROUTE_PROTOCOLS,
    LOCAL_MAP_PARAMETERIZATION,
    MIN_FAMILY_TRANSITIONS,
    PAPER_REENCODE_PERIODS,
    REENCODING_ROLE,
    ROUTE_PROTOCOL,
    ROUTE_SCHEMA_VERSION,
    ROUTING_CADENCE,
    STAGE1_TRAINING_STEPS,
    STAGE2_SELECTION_BATCH_SIZE,
    STAGE2_SELECTION_CANDIDATE_STEPS,
    STAGE2_SELECTION_HORIZONS,
    STAGE2_SELECTION_SEED_OFFSET,
    STAGE2_TRAINING_STEPS,
    SUPPORT_DEFINITION,
    SUPPORT_SCHEME,
    SUPPORT_THRESHOLD,
    TARGET_CENTER_RULE,
    TOTAL_TRAINING_STEPS,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the fixed staged F_abs source-protocol model."
    )
    parser.add_argument("--task_tsv", required=True)
    parser.add_argument("--array_index", type=int, default=0)
    parser.add_argument("--array_offset", type=int, default=0)
    parser.add_argument("--base_out", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--eval_profile", default="full", choices=("full", "smoke"))
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--resume_from_latest", action="store_true", default=True)
    parser.add_argument(
        "--no_resume_from_latest",
        dest="resume_from_latest",
        action="store_false",
    )
    parser.add_argument("--save_metrics_history", action="store_true")
    parser.add_argument("--save_last_checkpoint", action="store_true")
    parser.add_argument("--save_stage2_artifacts", action="store_true")
    parser.add_argument("--save_eval_rollout_artifacts", action="store_true")
    parser.add_argument("--save_eval_plots", action="store_true")
    parser.add_argument("--save_eval_per_ic_values", action="store_true")
    parser.add_argument("--save_eval_error_curves", action="store_true")
    return parser


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _support_definition(raw: str) -> Tuple[str, float]:
    if raw != SUPPORT_DEFINITION:
        raise ValueError(f"The staged protocol requires {SUPPORT_DEFINITION}, got {raw!r}.")
    return SUPPORT_SCHEME, SUPPORT_THRESHOLD


def _parse_int_csv(raw: Optional[str]) -> Tuple[int, ...]:
    if raw is None:
        return ()
    values: List[int] = []
    for item in str(raw).split(","):
        if not item.strip():
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Periodic re-encoding periods must be positive, got {value}")
        values.append(value)
    return tuple(dict.fromkeys(values))


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
    settings.batch_size = FINAL_EVALUATION_BATCH_SIZE
    settings.seed_offset = FINAL_EVALUATION_SEED_OFFSET
    settings.save_rollout_artifacts = bool(save_rollout_artifacts)
    settings.save_plots = bool(save_plots)
    settings.include_per_ic_values = bool(include_per_ic_values)
    settings.include_error_curves = bool(include_error_curves)
    if profile == "smoke":
        settings.batch_size = 32
        settings.horizons = STAGE2_SELECTION_HORIZONS
        settings.phase_portrait_samples = 8
        settings.phase_portrait_length = 100
        settings.phase_portrait_batch_size = 64
    settings.periodic_reencode_periods = tuple(
        int(period) for period in (periodic_periods_override or PAPER_REENCODE_PERIODS)
    )
    settings.phase_portrait_reencode_periods = tuple(
        dict.fromkeys((0, 1, *settings.periodic_reencode_periods))
    )
    return settings


def _make_stage2_selection_starts(
    base_env: object,
    *,
    seed: int,
    device: str,
) -> Tuple[VectorWrapper, torch.Tensor]:
    env = VectorWrapper(base_env, STAGE2_SELECTION_BATCH_SIZE)
    rng = torch.Generator().manual_seed(int(seed) + STAGE2_SELECTION_SEED_OFFSET)
    return env, env.reset(rng).to(device)


def _finite_prefix_start_mean(squared_error: torch.Tensor, horizon: int) -> float:
    """Apply the source selector's per-start then finite-start reducer."""
    finite_error = torch.where(
        torch.isfinite(squared_error[: int(horizon)]),
        squared_error[: int(horizon)],
        torch.nan,
    )
    per_start = torch.nanmean(finite_error, dim=0)
    finite_starts = per_start[torch.isfinite(per_start)]
    return (
        float(finite_starts.mean().item())
        if finite_starts.numel()
        else float("inf")
    )


def _strictly_improves(score: float, incumbent: float) -> bool:
    return float(score) < float(incumbent)


def _quick_eval_best_periodic_horizon_mse(
    eval_model: nn.Module,
    *,
    val_x: torch.Tensor,
    eval_env: VectorWrapper,
    horizons: Sequence[int] = STAGE2_SELECTION_HORIZONS,
    periods: Sequence[int] = PAPER_REENCODE_PERIODS,
) -> Tuple[float, Dict[int, Tuple[float, int]]]:
    """Reproduce the source selector's finite-prefix, per-horizon reducer."""
    clean_horizons = sorted({int(horizon) for horizon in horizons if int(horizon) > 0})
    if not clean_horizons:
        raise ValueError("Checkpoint-selection horizons must include a positive value")
    max_horizon = max(clean_horizons)
    device = next(eval_model.parameters()).device
    true_traj = generate_trajectory(eval_env.step, val_x.cpu(), length=max_horizon)
    best = {horizon: (float("inf"), -1) for horizon in clean_horizons}
    eval_model.eval()
    with torch.no_grad():
        for period in periods:
            latent = eval_model.encode(val_x.to(device))
            predictions: List[torch.Tensor] = []
            for step in range(max_horizon):
                latent_pred = eval_model.step_latent(latent)
                prediction = eval_model.decode(latent_pred)
                predictions.append(prediction)
                latent = eval_model.encode(prediction) if (step + 1) % int(period) == 0 else latent_pred
            pred_traj = torch.stack(predictions).detach().cpu()
            squared = torch.sum((pred_traj - true_traj) ** 2, dim=-1)
            for horizon in clean_horizons:
                score = _finite_prefix_start_mean(squared, horizon)
                if math.isfinite(score) and _strictly_improves(score, best[horizon][0]):
                    best[horizon] = (score, int(period))
    finite_positive = [value for value, _ in best.values() if math.isfinite(value) and value > 0.0]
    if len(finite_positive) != len(clean_horizons):
        return float("inf"), best
    return float(sum(finite_positive) / len(finite_positive)), best


def _route_codebook_metadata(route_codebook: Mapping[str, object]) -> Dict[str, object]:
    prototypes = route_codebook.get("family_prototypes", {})
    metadata = {
        "family_counts": {
            str(key): int(value) for key, value in route_codebook["family_counts"].items()
        },
        "fitted_family_ids": [str(item) for item in route_codebook["fitted_family_ids"]],
        "family_class_count_total": len(route_codebook["family_counts"]),
        "family_class_count_fit": len(route_codebook["fitted_family_ids"]),
        "family_prototypes": {
            str(key): np.asarray(value, dtype=bool).astype(int).tolist()
            for key, value in prototypes.items()
        },
    }
    for key in (
        "routing_object",
        "runtime_routing_kind",
        "route_jaccard_threshold",
        "family_representative_rule",
        "family_clustering_rule",
        "clustering_state_count",
        "source_transition_count",
    ):
        if key in route_codebook:
            metadata[key] = route_codebook[key]
    return metadata


def _paper_route_metadata(
    route_codebook: Mapping[str, object],
    *,
    fit_seed: int,
) -> Dict[str, object]:
    support_mask = np.asarray(route_codebook["support_mask"])
    clustering_state_count = int(support_mask.shape[0] * support_mask.shape[1])
    source_transition_count = int(sum(route_codebook["family_counts"].values()))
    if (
        clustering_state_count != FIT_SUPPORTS_CONSIDERED
        or source_transition_count != FIT_SOURCE_TRANSITIONS
    ):
        raise ValueError(
            "Cannot label a route codebook as the source protocol unless it "
            "contains 98,816 clustered states and 98,304 source transitions."
        )
    metadata = _route_codebook_metadata(route_codebook)
    metadata.update(
        {
            "route_schema_version": ROUTE_SCHEMA_VERSION,
            "protocol": ROUTE_PROTOCOL,
            "routing_object": "support_family",
            "runtime_routing_kind": "support_jaccard",
            "routing_cadence": ROUTING_CADENCE,
            "reencoding_role": REENCODING_ROLE,
            "support_definition": SUPPORT_DEFINITION,
            "family_jaccard_threshold": FAMILY_JACCARD_THRESHOLD,
            "family_representative_rule": FAMILY_REPRESENTATIVE_RULE,
            "family_clustering_rule": FAMILY_CLUSTERING_RULE,
            "fit_source": "training_distribution_trajectories",
            "fit_construction": "two_bitwise_identical_copies_of_one_256_row_batch",
            "fit_configured_rows": FIT_CONFIGURED_ROWS,
            "fit_unique_trajectories": FIT_UNIQUE_TRAJECTORIES,
            "fit_duplication_factor": FIT_DUPLICATION_FACTOR,
            "fit_transitions": FIT_TRANSITIONS,
            "fit_states": FIT_STATES,
            "fit_supports_considered": FIT_SUPPORTS_CONSIDERED,
            "fit_source_transitions": FIT_SOURCE_TRANSITIONS,
            "fit_unique_source_transitions": FIT_UNIQUE_SOURCE_TRANSITIONS,
            "clustering_state_count": clustering_state_count,
            "source_transition_count": source_transition_count,
            "fit_seed": int(fit_seed),
            "fit_seed_offset": FIT_SEED_OFFSET,
            # Legacy names remain serialized for existing consumers.
            "fit_num_trajectories": FIT_CONFIGURED_ROWS,
            "fit_trajectory_length": FIT_TRANSITIONS,
            "min_family_transitions": MIN_FAMILY_TRANSITIONS,
            "local_map_parameterization": LOCAL_MAP_PARAMETERIZATION,
            "target_center_rule": TARGET_CENTER_RULE,
            "learn_target_centers": True,
            "total_training_steps": TOTAL_TRAINING_STEPS,
            "stage1_joint_steps": STAGE1_TRAINING_STEPS,
            "stage2_local_steps": STAGE2_TRAINING_STEPS,
            "checkpoint_selection_metric": "best_periodic_horizon_mse",
            "checkpoint_selection_horizons": list(STAGE2_SELECTION_HORIZONS),
            "checkpoint_selection_periods": list(PAPER_REENCODE_PERIODS),
            "checkpoint_selection_batch_size": STAGE2_SELECTION_BATCH_SIZE,
            "checkpoint_selection_seed_offset": STAGE2_SELECTION_SEED_OFFSET,
            "checkpoint_selection_candidate_steps": list(STAGE2_SELECTION_CANDIDATE_STEPS),
            "checkpoint_selection_improvement_rule": "strict_less_than",
            "periodic_reencode_periods": list(PAPER_REENCODE_PERIODS),
        }
    )
    return metadata


def _validate_codebook_structure(route_codebook: Mapping[str, object]) -> None:
    required = {
        "family_counts",
        "fitted_family_ids",
        "centers",
        "family_prototypes",
        "support_key_to_family",
    }
    missing = sorted(required.difference(route_codebook))
    if missing:
        raise ValueError(f"Checkpoint route codebook is missing {missing}.")
    family_ids = list(route_codebook["fitted_family_ids"])
    if not family_ids or len({str(item) for item in family_ids}) != len(family_ids):
        raise ValueError("Checkpoint fitted family IDs are empty or ambiguous.")
    centers = route_codebook["centers"]
    prototypes = route_codebook["family_prototypes"]
    latent_dims = {np.asarray(centers[item]).shape for item in family_ids}
    mask_dims = {np.asarray(prototypes[item]).shape for item in family_ids}
    if len(latent_dims) != 1 or len(mask_dims) != 1 or latent_dims != mask_dims:
        raise ValueError("Checkpoint family center/prototype shapes are inconsistent.")


def _validate_frozen_fabs_artifact(
    route_codebook: Mapping[str, object],
    route_metadata: Mapping[str, object],
    *,
    expected_fit_seed: Optional[int] = None,
) -> None:
    """Validate new artifacts strictly and adapt only the known legacy schema."""
    if str(route_codebook.get("routing_object", "support_family")) != "support_family":
        raise ValueError("Only frozen F_abs support-family checkpoints are supported.")
    _validate_codebook_structure(route_codebook)
    protocol = route_metadata.get("protocol")
    schema = route_metadata.get("route_schema_version")
    if schema is None:
        if protocol not in LEGACY_ROUTE_PROTOCOLS:
            raise ValueError(f"Unknown legacy route protocol {protocol!r}.")
        # Historical checkpoints split scientific fields between the route
        # metadata and codebook.  Read only this known allow-list, reject
        # conflicts, and do not infer unspecified settings from arbitrary keys.
        allowed_keys = {
            "fit_num_trajectories",
            "stable_fit_trajectories",
            "fit_trajectory_length",
            "stable_fit_trajectory_length",
            "fit_seed",
            "stable_fit_seed",
            "support_definition",
            "family_jaccard_threshold",
            "route_jaccard_threshold",
            "min_family_transitions",
            "local_map_parameterization",
            "target_center_rule",
            "learn_target_centers",
            "stage1_joint_steps",
            "stage2_local_steps",
            "support_family_fit_source",
            "fitted_family_ids",
            "family_class_count_fit",
        }
        legacy: Dict[str, object] = {}
        for key in allowed_keys:
            values = [
                source[key]
                for source in (route_codebook, route_metadata)
                if key in source
            ]
            if len(values) == 2:
                if key == "fitted_family_ids":
                    left = [str(item) for item in values[0]]
                    right = [str(item) for item in values[1]]
                    conflict = left != right
                else:
                    conflict = values[0] != values[1]
                if conflict:
                    raise ValueError(f"Conflicting legacy checkpoint field {key}.")
            if values:
                legacy[key] = values[-1]
        count = legacy.get(
            "fit_num_trajectories", legacy.get("stable_fit_trajectories")
        )
        if count is None or int(count) != FIT_CONFIGURED_ROWS:
            raise ValueError("Legacy checkpoint must declare fit_num_trajectories=512.")
        optional = {
            "support_definition": SUPPORT_DEFINITION,
            "family_jaccard_threshold": FAMILY_JACCARD_THRESHOLD,
            "fit_trajectory_length": FIT_TRANSITIONS,
            "stable_fit_trajectory_length": FIT_TRANSITIONS,
            "min_family_transitions": MIN_FAMILY_TRANSITIONS,
            "local_map_parameterization": LOCAL_MAP_PARAMETERIZATION,
            "target_center_rule": TARGET_CENTER_RULE,
            "learn_target_centers": True,
            "stage1_joint_steps": STAGE1_TRAINING_STEPS,
            "stage2_local_steps": STAGE2_TRAINING_STEPS,
            "support_family_fit_source": "stable_fit_trajectories",
        }
        for key, expected in optional.items():
            if key in legacy and legacy[key] != expected:
                raise ValueError(f"Legacy checkpoint {key}={legacy[key]!r}; expected {expected!r}.")
        route_jaccard = legacy.get("route_jaccard_threshold")
        if route_jaccard is not None and float(route_jaccard) != FAMILY_JACCARD_THRESHOLD:
            raise ValueError("Legacy checkpoint route Jaccard threshold is not 0.40.")
        legacy_fit_seed = legacy.get("fit_seed", legacy.get("stable_fit_seed"))
        if expected_fit_seed is not None and legacy_fit_seed is not None:
            if int(legacy_fit_seed) != int(expected_fit_seed):
                raise ValueError("Legacy checkpoint route-fit seed does not match this task.")
        expected_family_ids = [str(item) for item in route_codebook["fitted_family_ids"]]
        if "fitted_family_ids" in legacy and [
            str(item) for item in legacy["fitted_family_ids"]
        ] != expected_family_ids:
            raise ValueError("Legacy checkpoint fitted family IDs do not match its codebook.")
        if "family_class_count_fit" in legacy and int(
            legacy["family_class_count_fit"]
        ) != len(expected_family_ids):
            raise ValueError("Legacy checkpoint fitted family count does not match its codebook.")
        return

    if int(schema) != ROUTE_SCHEMA_VERSION or protocol != ROUTE_PROTOCOL:
        raise ValueError(f"Unsupported route schema/protocol: {schema!r}/{protocol!r}.")
    expected: Dict[str, object] = {
        "support_definition": SUPPORT_DEFINITION,
        "family_jaccard_threshold": FAMILY_JACCARD_THRESHOLD,
        "family_representative_rule": FAMILY_REPRESENTATIVE_RULE,
        "family_clustering_rule": FAMILY_CLUSTERING_RULE,
        "routing_cadence": ROUTING_CADENCE,
        "reencoding_role": REENCODING_ROLE,
        "fit_configured_rows": FIT_CONFIGURED_ROWS,
        "fit_unique_trajectories": FIT_UNIQUE_TRAJECTORIES,
        "fit_duplication_factor": FIT_DUPLICATION_FACTOR,
        "fit_transitions": FIT_TRANSITIONS,
        "fit_states": FIT_STATES,
        "fit_supports_considered": FIT_SUPPORTS_CONSIDERED,
        "fit_source_transitions": FIT_SOURCE_TRANSITIONS,
        "fit_unique_source_transitions": FIT_UNIQUE_SOURCE_TRANSITIONS,
        "fit_seed_offset": FIT_SEED_OFFSET,
        "min_family_transitions": MIN_FAMILY_TRANSITIONS,
        "local_map_parameterization": LOCAL_MAP_PARAMETERIZATION,
        "target_center_rule": TARGET_CENTER_RULE,
        "learn_target_centers": True,
        "total_training_steps": TOTAL_TRAINING_STEPS,
        "stage1_joint_steps": STAGE1_TRAINING_STEPS,
        "stage2_local_steps": STAGE2_TRAINING_STEPS,
        "checkpoint_selection_metric": "best_periodic_horizon_mse",
        "checkpoint_selection_horizons": list(STAGE2_SELECTION_HORIZONS),
        "checkpoint_selection_periods": list(PAPER_REENCODE_PERIODS),
        "checkpoint_selection_batch_size": STAGE2_SELECTION_BATCH_SIZE,
        "checkpoint_selection_seed_offset": STAGE2_SELECTION_SEED_OFFSET,
        "checkpoint_selection_candidate_steps": list(STAGE2_SELECTION_CANDIDATE_STEPS),
        "checkpoint_selection_improvement_rule": "strict_less_than",
        "fitted_family_ids": [
            str(item) for item in route_codebook["fitted_family_ids"]
        ],
        "family_class_count_fit": len(route_codebook["fitted_family_ids"]),
        "clustering_state_count": FIT_SUPPORTS_CONSIDERED,
        "source_transition_count": FIT_SOURCE_TRANSITIONS,
    }
    for key, value in expected.items():
        if route_metadata.get(key) != value:
            raise ValueError(f"Checkpoint {key}={route_metadata.get(key)!r}; expected {value!r}.")
    if expected_fit_seed is not None and int(route_metadata.get("fit_seed", -1)) != int(expected_fit_seed):
        raise ValueError("Checkpoint route-fit seed does not match this task.")


assert len(STAGE2_SELECTION_CANDIDATE_STEPS) == 200
