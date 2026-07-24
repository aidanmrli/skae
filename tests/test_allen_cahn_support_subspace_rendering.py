"""Outcome-blind contracts for the Allen--Cahn mechanism renderer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.neurips_2026.evidence.allen_cahn_support_subspace_rendering import (
    render_support_subspace_mechanism,
    validate_mechanism_display_inputs,
)


def _rows() -> pd.DataFrame:
    records = []
    for offset, seed in enumerate(range(64, 74)):
        row = {
            "seed": seed,
            "family_eligible": offset != 9,
            "signature_observed_over_null": 1.15 + 0.01 * offset,
        }
        for horizon in (160, 200):
            scale = 1.0 + 0.001 * horizon + 0.01 * offset
            for arm, multiplier in (("sparse", 0.35), ("dense", 0.65)):
                row[f"h{horizon}_{arm}_k_leakage"] = multiplier * scale
                row[f"h{horizon}_{arm}_k_null"] = 1.0 * scale
                row[f"h{horizon}_{arm}_kminusI_leakage"] = (
                    multiplier + 0.1
                ) * scale
                row[f"h{horizon}_{arm}_rho"] = multiplier + 0.55
            row[f"h{horizon}_correct_family_rho"] = 0.78 + 0.005 * offset
            row[f"h{horizon}_wrong_family_rho"] = 1.0 + 0.005 * offset
        records.append(row)
    return pd.DataFrame(records)


def _decision() -> dict:
    routing = {
        str(horizon): {
            "correct_over_wrong_restriction_factor_ratio_of_seed_means": 0.78,
            "restriction_factor_ratio_bootstrap": [0.72, 0.84],
        }
        for horizon in (160, 200)
    }
    return {
        "decision": "strong_routed_low_leakage_charts_with_distinct_signatures",
        "validity": {"passed": True},
        "exact_fixed_P0_closure": {
            "passed": True,
            "checks": {
                "dense_activity_specificity": True,
                "dense_matrix_specificity": True,
            },
        },
        "decoded_forecast": {
            "passed": True,
            "projected_vs_dense_full_passed": True,
        },
        "family": {
            "family_passed": True,
            "signature_differentiation_passed": True,
            "routing_specificity_passed": True,
            "signature_ratio_mean": 1.19,
            "signature_ratio_bootstrap": [1.14, 1.24],
            "routing_specificity": routing,
        },
    }


def _negative() -> dict:
    return {
        "model_seed": 61,
        "mean_local_over_global": 1.218,
        "terminal_local_over_global": 1.185,
        "route_coverage": 0.840,
        "coverage_gate": 0.90,
        "all_local_updates_zero": True,
        "different_recipe_from_positive_global_model": True,
    }


def test_renderer_writes_only_explicit_outputs(tmp_path):
    output_pdf = tmp_path / "mechanism.pdf"
    output_png = tmp_path / "mechanism.png"
    render_support_subspace_mechanism(
        _rows(),
        _decision(),
        _negative(),
        output_pdf=output_pdf,
        output_png=output_png,
    )
    assert output_pdf.stat().st_size > 10_000
    assert output_png.stat().st_size > 10_000


def test_renderer_fails_closed_on_missing_paired_seed():
    with pytest.raises(ValueError, match="paired seed 64--73"):
        validate_mechanism_display_inputs(
            _rows().iloc[:-1], _decision(), _negative()
        )


def test_renderer_fails_closed_on_nonpositive_null():
    rows = _rows()
    rows.loc[0, "h160_sparse_k_null"] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        validate_mechanism_display_inputs(rows, _decision(), _negative())


def test_renderer_allows_ineligible_family_rows_to_be_nan():
    rows = _rows()
    ineligible = ~rows["family_eligible"]
    rows.loc[ineligible, "signature_observed_over_null"] = np.nan
    rows.loc[ineligible, "h160_correct_family_rho"] = np.nan
    rows.loc[ineligible, "h160_wrong_family_rho"] = np.nan
    validate_mechanism_display_inputs(rows, _decision(), _negative())
