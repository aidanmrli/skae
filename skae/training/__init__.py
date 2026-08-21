"""Reusable training primitives and the standard SKAE training runner."""

from skae.training.runner import (
    MetricsLogger,
    build_optimizer,
    evaluate,
    generate_sequence_batch_for_device,
    get_device,
    parse_optional_bool,
    train,
    train_step,
)
from skae.training.checkpointing import (
    CHECKPOINT_EXIT_CODE,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointManager,
    CheckpointSignalExit,
    SignalStopper,
    capture_rng_state,
    restore_rng_state,
)

__all__ = [
    "MetricsLogger",
    "generate_sequence_batch_for_device",
    "parse_optional_bool",
    "train_step",
    "build_optimizer",
    "evaluate",
    "train",
    "get_device",
    "CHECKPOINT_EXIT_CODE",
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointSignalExit",
    "SignalStopper",
    "capture_rng_state",
    "restore_rng_state",
]
