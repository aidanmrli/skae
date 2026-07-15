"""Static scientific contract for the staged support-routed operator run."""

from __future__ import annotations

ROUTE_PROTOCOL = "staged_fabs_route_source_v3"
ROUTE_SCHEMA_VERSION = 3
ROUTING_CADENCE = "every_latent_transition_step"
REENCODING_ROLE = "periodic_decode_encode_refreshes_latent_before_next_route"
LEGACY_ROUTE_PROTOCOLS = (None, "staged_fabs_route_v1")
SUPPORT_DEFINITION = "absolute:0.001"
SUPPORT_SCHEME = "absolute"
SUPPORT_THRESHOLD = 1e-3
FAMILY_JACCARD_THRESHOLD = 0.40
# Historical source-route fitting contract.  The nominal 512-row packet was
# produced from two identical copies of one 256-row environment batch.  A
# 512-unique-trajectory fit is a new protocol and cannot reuse this label.
FIT_CONFIGURED_ROWS = 512
FIT_UNIQUE_TRAJECTORIES = 256
FIT_DUPLICATION_FACTOR = 2
FIT_TRANSITIONS = 192
FIT_STATES = FIT_TRANSITIONS + 1
FIT_SEED_OFFSET = 271_828
FIT_SUPPORTS_CONSIDERED = FIT_CONFIGURED_ROWS * FIT_STATES
FIT_SOURCE_TRANSITIONS = FIT_CONFIGURED_ROWS * FIT_TRANSITIONS
FIT_UNIQUE_SOURCE_TRANSITIONS = FIT_UNIQUE_TRAJECTORIES * FIT_TRANSITIONS
FIT_NUM_TRAJECTORIES = FIT_CONFIGURED_ROWS
FIT_TRAJECTORY_LENGTH = FIT_TRANSITIONS
MIN_FAMILY_TRANSITIONS = 1
FAMILY_REPRESENTATIVE_RULE = "modal_source_support"
FAMILY_CLUSTERING_RULE = "all_193_states_then_fit_on_first_192_sources"
TOTAL_TRAINING_STEPS = 200_000
STAGE1_TRAINING_STEPS = 100_000
STAGE2_TRAINING_STEPS = 100_000
LOCAL_MAP_PARAMETERIZATION = "source_target_affine_learned_intercept"
TARGET_CENTER_RULE = "learned target center initialized as source_center @ frozen_global_k"
PAPER_REENCODE_PERIODS = (1, 2, 5, 10, 20, 25, 50, 100)
STAGE2_SELECTION_HORIZONS = (100, 500, 1000)
STAGE2_SELECTION_BATCH_SIZE = 32
STAGE2_SELECTION_SEED_OFFSET = 12_345
STAGE2_SELECTION_CANDIDATE_STEPS = (
    *range(100_500, 200_000, 500),
    199_999,
)
FINAL_EVALUATION_BATCH_SIZE = 100
FINAL_EVALUATION_SEED_OFFSET = 12_345


def route_protocol_metadata() -> dict[str, object]:
    """Return the task-manifest view of the immutable staged protocol."""

    return {
        "protocol": ROUTE_PROTOCOL,
        "route_schema_version": ROUTE_SCHEMA_VERSION,
        "total_steps": TOTAL_TRAINING_STEPS,
        "stage1_joint_steps": STAGE1_TRAINING_STEPS,
        "stage2_local_steps": STAGE2_TRAINING_STEPS,
        "support_definition": SUPPORT_DEFINITION,
        "family_jaccard_threshold": FAMILY_JACCARD_THRESHOLD,
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
        "fit_seed_offset": FIT_SEED_OFFSET,
        "family_clustering": FAMILY_CLUSTERING_RULE,
        "family_representative": FAMILY_REPRESENTATIVE_RULE,
        "min_family_transitions": MIN_FAMILY_TRANSITIONS,
        "routing_cadence": ROUTING_CADENCE,
        "reencoding_role": REENCODING_ROLE,
        "local_map": LOCAL_MAP_PARAMETERIZATION,
        "checkpoint_selection": {
            "candidate_count": len(STAGE2_SELECTION_CANDIDATE_STEPS),
            "first_regular_step": STAGE2_SELECTION_CANDIDATE_STEPS[0],
            "last_regular_step": STAGE2_SELECTION_CANDIDATE_STEPS[-2],
            "final_step": STAGE2_SELECTION_CANDIDATE_STEPS[-1],
            "batch_size": STAGE2_SELECTION_BATCH_SIZE,
            "seed_offset": STAGE2_SELECTION_SEED_OFFSET,
            "horizons": list(STAGE2_SELECTION_HORIZONS),
            "periods": list(PAPER_REENCODE_PERIODS),
            "metric": "finite_prefix_state_summed_squared_error",
            "improvement": "strict_less_than",
        },
        "final_evaluation": {
            "batch_size": FINAL_EVALUATION_BATCH_SIZE,
            "seed_offset": FINAL_EVALUATION_SEED_OFFSET,
            "selector_overlap_count": STAGE2_SELECTION_BATCH_SIZE,
            "selector_overlap_fraction": (
                STAGE2_SELECTION_BATCH_SIZE / FINAL_EVALUATION_BATCH_SIZE
            ),
        },
    }


assert len(STAGE2_SELECTION_CANDIDATE_STEPS) == 200
