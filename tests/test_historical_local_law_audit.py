"""Regression checks for the authenticated historical local-law audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.neurips_2026.evidence.historical_local_law_audit import (
    OUTPUT,
    build_payload,
    ratio_summary,
)


def test_ratio_summary_exposes_selection_and_nonfinite_values() -> None:
    summary = ratio_summary([0.5, 1.0, 2.0, np.inf, np.nan])
    assert summary == {
        "ratio_row_count": 4,
        "finite_ratio_count": 3,
        "infinite_ratio_count": 1,
        "wins_below_one": 1,
        "exact_ties_at_one": 1,
        "finite_median": 1.0,
        "finite_mean": 3.5 / 3.0,
    }


def test_frozen_historical_local_law_audit_recomputes_exactly() -> None:
    payload = build_payload()
    assert payload == json.loads(Path(OUTPUT).read_text())

    centered = payload["centered_chart"]["relative_threshold_results"]
    assert centered["full_k_sparse"]["local_refit_over_learned_k"][
        "finite_median"
    ] == 0.00022897888423802375
    assert centered["dense_mlp"]["same_learned_k_input_mask_over_learned_k"][
        "finite_median"
    ] == 0.00013053370800874734
    assert centered["full_k_sparse"][
        "observed_local_refit_over_latent_kmeans_refit"
    ]["finite_median"] == 1.0

    routed = payload["self_routed_forecasting"]["results"]["full_k_sparse"]
    assert routed["modes"]["support_gated_k"]["h1000_over_global"][
        "finite_ratio_count"
    ] == 75
    assert routed["modes"]["support_gated_k"]["route_coverage"][
        "median"
    ] == 0.5290749967098236

    recipes = payload["checkpoint_recipe_audit"]["recipes"]
    assert recipes["dense_mlp"]["activation_last_relu_counts"] == {
        "relu|last_relu=true": 1,
        "tanh|last_relu=false": 169,
    }
