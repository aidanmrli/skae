"""Evidence collection, validation, aggregation, and rendering."""

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
