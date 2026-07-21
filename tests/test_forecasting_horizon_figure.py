from __future__ import annotations

import io
from collections import OrderedDict

import pandas as pd
import pytest

from experiments.neurips_2026.evidence.forecasting_horizon_rendering import (
    render_panel,
)
from experiments.neurips_2026.evidence.forecasting_horizons import (
    CONTROLLED_HORIZONS,
    DYSTS_HORIZONS,
    DYSTS_METHODS,
    DYSTS_METHODS_NO_LISTA_SB,
    DYSTS_PANEL_NO_LISTA_SB_PATH,
    DYSTS_PANEL_PATH,
    OUTPUT_PATHS,
    build_outputs,
    summarize_benchmark,
    write_or_check,
)


def test_cross_system_mean_keeps_a_hard_system() -> None:
    systems = tuple(f"system_{index}" for index in range(10))
    records = []
    for system_index, system in enumerate(systems):
        for seed in range(4):
            records.append(
                {
                    "root_label": "candidate",
                    "system_key": system,
                    "seed": seed,
                    "h100_best_periodic_mean": (
                        1_000.0 if system_index == 9 else 1.0
                    ),
                }
            )
    styles = OrderedDict(
        [("candidate", ("Candidate", "#0072B2", "-"))]
    )
    summary = summarize_benchmark(
        pd.DataFrame(records),
        benchmark="synthetic",
        system_column="system_key",
        expected_systems=systems,
        method_styles=styles,
        horizons=(100,),
        bootstrap_reps=16,
    )

    row = summary.iloc[0]
    assert row["n_systems"] == 10
    assert row["mean_over_system_seed_iqms"] == pytest.approx(100.9)
    assert row["mean_over_system_seed_iqms"] != pytest.approx(1.0)


def test_rendering_is_deterministic() -> None:
    summary = pd.DataFrame(
        {
            "root_label": ["candidate", "candidate"],
            "horizon": [100, 500],
            "mean_over_system_seed_iqms": [0.1, 0.2],
            "log_relative_seed_bootstrap_ci95_low": [0.08, 0.16],
            "log_relative_seed_bootstrap_ci95_high": [0.12, 0.24],
        }
    )
    styles = OrderedDict(
        [("candidate", ("Candidate", "#0072B2", "-"))]
    )
    first = render_panel(
        summary,
        styles,
        (100, 500),
        title="Synthetic benchmark",
    )
    second = render_panel(
        summary,
        styles,
        (100, 500),
        title="Synthetic benchmark",
    )

    assert first.startswith(b"%PDF")
    assert first == second


def test_dysts_panel_variants_have_truthful_method_rosters() -> None:
    assert DYSTS_PANEL_PATH.name == "fig_dysts_dt30_forecasting_performance.pdf"
    assert (
        DYSTS_PANEL_NO_LISTA_SB_PATH.name
        == "fig_dysts_dt30_forecasting_performance_no_lista_sb.pdf"
    )
    assert [style[0] for style in DYSTS_METHODS.values()].count("LISTA-SB") == 1
    assert "LISTA-SB" not in {
        style[0] for style in DYSTS_METHODS_NO_LISTA_SB.values()
    }
    assert len(DYSTS_METHODS_NO_LISTA_SB) == len(DYSTS_METHODS) - 1


def test_frozen_figure2_packet_has_full_rosters(tmp_path) -> None:
    outputs = build_outputs(bootstrap_reps=16)
    summary = pd.read_csv(io.BytesIO(outputs[OUTPUT_PATHS[0]]))
    controlled = summary[summary["benchmark"] == "controlled"]
    dysts = summary[summary["benchmark"] == "dysts_dt30"]

    assert len(controlled) == 6 * len(CONTROLLED_HORIZONS)
    assert len(dysts) == 6 * len(DYSTS_HORIZONS)
    assert set(controlled["n_systems"]) == {15}
    assert set(dysts["n_systems"]) == {10}
    assert set(summary["min_finite_seeds_per_system"]) == {15}
    assert set(summary["max_finite_seeds_per_system"]) == {15}
    assert all(outputs[path].startswith(b"%PDF") for path in OUTPUT_PATHS[1:])

    temporary_outputs = {
        tmp_path / path.name: content for path, content in outputs.items()
    }
    write_or_check(temporary_outputs, check=False)
    write_or_check(temporary_outputs, check=True)
