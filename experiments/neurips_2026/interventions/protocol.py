"""Frozen contract for the paper's one-checkpoint intervention case study."""

from __future__ import annotations

from typing import Mapping


INTERVENTION_PROTOCOL_ID = "support_coordinate_intervention_case_study_v1"
ROOT_LABEL = "lista_dense_signsplit_p256_hardinit_basin_partition"
SYSTEM_KEY = "gated_local_linear"
TRAINING_SEED = 0
NUM_INITIAL_POINTS = 100
NUM_CANDIDATE_TRAJECTORIES = 512
TRAJECTORY_LENGTH = 64
EVALUATION_SEED = 42
ENDPOINT_ROLLOUT_STEPS = 5000
SUPPORT_DEFINITION = "absolute:0.001"
HORIZONS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
MAX_DROP = 10
RANDOM_SUPPORT_REPEATS = 20
RANDOM_SEED = 123
DEPTH_SLICE_MODE = "per_basin"
REQUIRE_STABLE_TRUE_BASIN = True
PLOT_FORMATS = "pdf,png"


def intervention_protocol_metadata() -> dict[str, object]:
    """Return the scientific settings shared by CLI, launcher, and freezer."""

    return {
        "protocol_id": INTERVENTION_PROTOCOL_ID,
        "root_label": ROOT_LABEL,
        "system_key": SYSTEM_KEY,
        "training_seed": TRAINING_SEED,
        "num_initial_points": NUM_INITIAL_POINTS,
        "num_candidate_trajectories": NUM_CANDIDATE_TRAJECTORIES,
        "trajectory_length": TRAJECTORY_LENGTH,
        "evaluation_seed": EVALUATION_SEED,
        "endpoint_rollout_steps": ENDPOINT_ROLLOUT_STEPS,
        "support_definition": SUPPORT_DEFINITION,
        "horizons": list(HORIZONS),
        "max_drop": MAX_DROP,
        "random_support_repeats": RANDOM_SUPPORT_REPEATS,
        "random_seed": RANDOM_SEED,
        "depth_slice_mode": DEPTH_SLICE_MODE,
        "require_stable_true_basin": REQUIRE_STABLE_TRUE_BASIN,
        "plot_formats": PLOT_FORMATS.split(","),
    }


def validate_intervention_protocol_record(record: Mapping[str, object]) -> None:
    """Reject frozen evidence that does not match the case-study contract."""

    expected = {
        "root_label": ROOT_LABEL,
        "system_key": SYSTEM_KEY,
        "system_name": SYSTEM_KEY,
        "seed": TRAINING_SEED,
        "support_definition": SUPPORT_DEFINITION,
        "depth_slice_mode": DEPTH_SLICE_MODE,
        "eval_seed": EVALUATION_SEED,
        "num_candidate_trajectories": NUM_CANDIDATE_TRAJECTORIES,
        "trajectory_length": TRAJECTORY_LENGTH,
        "max_horizon": max(HORIZONS),
        "horizons": list(HORIZONS),
        "num_initial_points": NUM_INITIAL_POINTS,
    }
    drift = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    if drift:
        raise ValueError(f"Intervention evidence protocol drifted: {drift}")


__all__ = [
    "INTERVENTION_PROTOCOL_ID",
    "ROOT_LABEL",
    "SYSTEM_KEY",
    "TRAINING_SEED",
    "NUM_INITIAL_POINTS",
    "NUM_CANDIDATE_TRAJECTORIES",
    "TRAJECTORY_LENGTH",
    "EVALUATION_SEED",
    "ENDPOINT_ROLLOUT_STEPS",
    "SUPPORT_DEFINITION",
    "HORIZONS",
    "MAX_DROP",
    "RANDOM_SUPPORT_REPEATS",
    "RANDOM_SEED",
    "DEPTH_SLICE_MODE",
    "REQUIRE_STABLE_TRUE_BASIN",
    "PLOT_FORMATS",
    "intervention_protocol_metadata",
    "validate_intervention_protocol_record",
]
