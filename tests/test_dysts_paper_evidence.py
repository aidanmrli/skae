from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from skae.benchmarks.paper_statistics import rowwise_interquartile_mean
from tools.build_dysts_paper_evidence import (
    DEFAULT_INPUT,
    DEFAULT_PROVENANCE,
    HORIZONS,
    METHODS,
    TABLE_DIR,
    aggregate_tests,
    iqm,
    load_rows,
    render_ratio_table,
    render_table,
    robust_summary,
    summarize_rows,
    verify_provenance,
    write_or_check,
)
from tools.dysts_paper_rendering import render_dysts_figure


def _synthetic_rows() -> pd.DataFrame:
    records = []
    for method_index, root_label in enumerate(METHODS):
        for system_index in range(2):
            for seed in range(4):
                row = {
                    "root_label": root_label,
                    "system_key": f"system_{system_index}",
                    "seed": seed,
                    "status": "complete",
                }
                multiplier = (
                    1.0
                    if root_label == "dense_mlp_tanh"
                    else 0.45 + 0.04 * method_index + 0.01 * system_index
                )
                for horizon in HORIZONS:
                    row[f"h{horizon}_best_periodic_mean"] = (
                        multiplier
                        * (horizon / 100)
                        * (system_index + 1)
                        * (seed + 1)
                    )
                    row[f"h{horizon}_best_periodic_full_finite_fraction"] = 1.0
                records.append(row)
    return pd.DataFrame(records)


def test_frozen_rows_have_the_exact_paper_roster() -> None:
    verify_provenance(DEFAULT_INPUT, DEFAULT_PROVENANCE)
    rows = load_rows(DEFAULT_INPUT)
    assert len(rows) == 900
    assert rows[["root_label", "system_key", "seed"]].duplicated().sum() == 0


def test_summary_and_tests_keep_two_stage_aggregation_semantics() -> None:
    per_system, summary = summarize_rows(_synthetic_rows(), bootstrap_reps=16)
    assert len(per_system) == len(METHODS) * 2 * len(HORIZONS)
    assert len(summary) == len(METHODS) * len(HORIZONS)
    dense_h100 = summary[
        (summary["root_label"] == "dense_mlp_tanh")
        & (summary["horizon"] == 100)
    ].iloc[0]
    assert dense_h100["cross_system_mean"] == 3.75
    assert dense_h100["cross_system_iqm_legacy"] == 3.75
    assert dense_h100["full_finite_mean"] == 1.0

    tests = aggregate_tests(per_system)
    assert len(tests) == (len(METHODS) - 1) * len(HORIZONS)
    assert set(tests["n_systems"]) == {2}
    assert np.all(tests["ratio_mean"] < 1.0)
    assert np.all(tests["p_system_sign_holm_all"] >= tests["p_system_sign_raw"])


def test_active_appendix_display_is_deterministic(tmp_path) -> None:
    _, summary = summarize_rows(_synthetic_rows(), bootstrap_reps=16)
    robust = robust_summary(summary)
    table = render_table(robust)
    first_pdf = render_dysts_figure(robust, METHODS, HORIZONS)
    second_pdf = render_dysts_figure(robust, METHODS, HORIZONS)
    assert table.startswith(b"\\begin{tabular}")
    assert first_pdf.startswith(b"%PDF")
    assert first_pdf == second_pdf

    outputs = {tmp_path / "table.tex": table, tmp_path / "figure.pdf": first_pdf}
    write_or_check(outputs, check=False)
    write_or_check(outputs, check=True)


def test_iqm_matches_scipy_trim_mean_convention() -> None:
    assert iqm([1.0, 2.0, 3.0, 100.0]) == 2.5
    values = [0.0] * 11 + [100.0] * 4
    # n=15 drops floor(15 * .25)=3 values from each tail, retaining 9.
    assert iqm(values) == pytest.approx(100.0 / 9.0)
    rows = np.asarray([values, list(reversed(values))])
    assert rowwise_interquartile_mean(rows).tolist() == pytest.approx(
        [100.0 / 9.0, 100.0 / 9.0]
    )


def test_ratio_table_matches_the_reviewed_inline_values() -> None:
    tests = pd.read_csv(TABLE_DIR / "dysts_dt30_aggregate_tests_vs_dense.csv")
    table = render_ratio_table(tests).decode()
    assert "LISTA & \\(0.491\\) & \\(0.426\\) & \\(0.440\\)" in table
    assert "Sparse MLP-BD & \\(0.614\\) & \\(0.496\\) & \\(0.458\\)" in table
    assert "Dense MLP & " + " & ".join([r"\(1.000\)"] * 8) in table


def test_provenance_rejects_tampered_frozen_rows(tmp_path) -> None:
    tampered = tmp_path / DEFAULT_INPUT.name
    tampered.write_bytes(DEFAULT_INPUT.read_bytes() + b"\n")
    custom_provenance = tmp_path / "provenance.json"
    custom_provenance.write_text(
        DEFAULT_PROVENANCE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Frozen evidence hash mismatch"):
        verify_provenance(tampered, custom_provenance)


def test_provenance_rejects_wrong_manifest_and_implicit_custom_input(
    tmp_path,
) -> None:
    custom_input = tmp_path / DEFAULT_INPUT.name
    custom_input.write_bytes(DEFAULT_INPUT.read_bytes())
    with pytest.raises(ValueError, match="custom --input"):
        verify_provenance(custom_input, DEFAULT_PROVENANCE)

    wrong_provenance = tmp_path / "wrong.json"
    wrong_provenance.write_text(json.dumps({"outputs": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing valid provenance"):
        verify_provenance(custom_input, wrong_provenance)
