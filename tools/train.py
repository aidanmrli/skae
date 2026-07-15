"""Compatibility entry point for :mod:`skae.training.runner`."""

from skae.training.runner import (
    DYSTS_CACHE_PROFILES,
    MetricsLogger,
    apply_dysts_cache_profile,
    build_optimizer,
    evaluate,
    generate_sequence_batch_for_device,
    get_device,
    main,
    parse_optional_bool,
    train,
    train_step,
)

__all__ = [
    "DYSTS_CACHE_PROFILES",
    "apply_dysts_cache_profile",
    "MetricsLogger",
    "generate_sequence_batch_for_device",
    "parse_optional_bool",
    "train_step",
    "build_optimizer",
    "evaluate",
    "train",
    "get_device",
    "main",
]


if __name__ == "__main__":
    main()
