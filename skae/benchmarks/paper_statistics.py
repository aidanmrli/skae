"""Compatibility imports for paper statistics.

Paper-only implementations live in :mod:`experiments.neurips_2026` so the
reusable :mod:`skae` package does not own submission-specific aggregation.
"""

from experiments.neurips_2026.evidence.statistics import (
    IQM_CONVENTION,
    IQM_PROPORTION_TO_CUT,
    interquartile_mean,
    rowwise_interquartile_mean,
)

__all__ = [
    "IQM_CONVENTION",
    "IQM_PROPORTION_TO_CUT",
    "interquartile_mean",
    "rowwise_interquartile_mean",
]
