"""Backward-compatible access to the maintained Dysts panel renderer.

The active paper no longer builds an IQM-over-systems figure.  This adapter is
kept only so historical imports under ``tools/`` continue to resolve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from experiments.neurips_2026.evidence.forecasting_horizon_rendering import (
    HIGH_COLUMN,
    LOW_COLUMN,
    MEAN_COLUMN,
    render_panel,
)


def render_dysts_figure(
    summary: pd.DataFrame,
    methods: Mapping[str, tuple[str, str, str]],
    horizons: Sequence[int],
) -> bytes:
    """Render a historical Dysts summary through the maintained plotter."""

    frame = summary.copy()
    if MEAN_COLUMN not in frame:
        source = (
            "cross_system_mean"
            if "cross_system_mean" in frame
            else "iqm_over_system_seed_iqms"
        )
        frame[MEAN_COLUMN] = frame[source]
    if LOW_COLUMN not in frame:
        frame[LOW_COLUMN] = frame["system_q25"]
    if HIGH_COLUMN not in frame:
        frame[HIGH_COLUMN] = frame["system_q75"]
    return render_panel(
        frame,
        methods,
        horizons,
        title=r"10-system Dysts $dt{\times}30$ forecasting performance",
    )
