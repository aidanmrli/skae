"""Compatibility imports for the frozen controlled-paper protocol.

New code should import :mod:`experiments.neurips_2026.controlled` directly.
"""

from experiments.neurips_2026.controlled import *  # noqa: F401,F403
from experiments.neurips_2026.controlled import __all__ as _CANONICAL_ALL


# Historical names remain available only at this historical import path. The
# maintained paper package uses the concise ``Controlled*`` API.
TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS = CONTROLLED_NUM_STEPS
TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE = CONTROLLED_BATCH_SIZE
TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE = CONTROLLED_TARGET_SIZE
TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH = CONTROLLED_SEQUENCE_LENGTH
TRANSITION_RICH_BASIN_PARTITION_SEEDS = CONTROLLED_SEEDS
TransitionRichBasinPartitionSystem = ControlledSystem
TransitionRichBasinPartitionModel = ControlledModel
TRANSITION_RICH_BASIN_PARTITION_SYSTEMS = CONTROLLED_SYSTEMS
TRANSITION_RICH_BASIN_PARTITION_MODELS = CONTROLLED_MODELS
transition_rich_basin_partition_systems = controlled_systems
transition_rich_basin_partition_models = controlled_models
get_transition_rich_basin_partition_system = get_controlled_system
get_transition_rich_basin_partition_model = get_controlled_model
resolve_transition_rich_default_dt = resolve_controlled_default_dt
get_transition_rich_basin_count = get_controlled_basin_count
transition_rich_basin_partition_manifest_jsonable = controlled_manifest_jsonable

__all__ = [
    *_CANONICAL_ALL,
    "TRANSITION_RICH_BASIN_PARTITION_NUM_STEPS",
    "TRANSITION_RICH_BASIN_PARTITION_BATCH_SIZE",
    "TRANSITION_RICH_BASIN_PARTITION_TARGET_SIZE",
    "TRANSITION_RICH_BASIN_PARTITION_SEQUENCE_LENGTH",
    "TRANSITION_RICH_BASIN_PARTITION_SEEDS",
    "TransitionRichBasinPartitionSystem",
    "TransitionRichBasinPartitionModel",
    "TRANSITION_RICH_BASIN_PARTITION_SYSTEMS",
    "TRANSITION_RICH_BASIN_PARTITION_MODELS",
    "transition_rich_basin_partition_systems",
    "transition_rich_basin_partition_models",
    "get_transition_rich_basin_partition_system",
    "get_transition_rich_basin_partition_model",
    "resolve_transition_rich_default_dt",
    "get_transition_rich_basin_count",
    "transition_rich_basin_partition_manifest_jsonable",
]
