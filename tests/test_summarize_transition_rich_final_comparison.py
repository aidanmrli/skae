import csv
from pathlib import Path

import pytest

from tools.summarize_transition_rich_final_comparison import (
    _latest_forecasting_rows,
    _latest_interpretability_rows,
    _read_csv,
    build_summary,
)


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_summary_reports_root_medians_and_pairwise_wins(tmp_path):
    forecast_rows = [
        {
            "root_label": "lista_a",
            "system_key": "sys1",
            "seed_name": "seed_0",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/lista_a/sys1/seed_0/20260409-100000",
            "h100_best_periodic_mean": "0.10",
            "h500_best_periodic_mean": "0.20",
            "h1000_best_periodic_mean": "0.30",
        },
        {
            "root_label": "lista_a",
            "system_key": "sys1",
            "seed_name": "seed_1",
            "seed": "1",
            "run_id": "20260409-100001",
            "run_dir": "/tmp/lista_a/sys1/seed_1/20260409-100001",
            "h100_best_periodic_mean": "0.20",
            "h500_best_periodic_mean": "0.30",
            "h1000_best_periodic_mean": "0.40",
        },
        {
            "root_label": "lista_a",
            "system_key": "sys2",
            "seed_name": "seed_0",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/lista_a/sys2/seed_0/20260409-100000",
            "h100_best_periodic_mean": "0.40",
            "h500_best_periodic_mean": "0.60",
            "h1000_best_periodic_mean": "0.80",
        },
        {
            "root_label": "lista_a",
            "system_key": "sys2",
            "seed_name": "seed_1",
            "seed": "1",
            "run_id": "20260409-100001",
            "run_dir": "/tmp/lista_a/sys2/seed_1/20260409-100001",
            "h100_best_periodic_mean": "0.50",
            "h500_best_periodic_mean": "0.70",
            "h1000_best_periodic_mean": "0.90",
        },
        {
            "root_label": "lista_b",
            "system_key": "sys1",
            "seed_name": "seed_0",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/lista_b/sys1/seed_0/20260409-100000",
            "h100_best_periodic_mean": "0.15",
            "h500_best_periodic_mean": "0.25",
            "h1000_best_periodic_mean": "0.35",
        },
        {
            "root_label": "lista_b",
            "system_key": "sys2",
            "seed_name": "seed_0",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/lista_b/sys2/seed_0/20260409-100000",
            "h100_best_periodic_mean": "0.60",
            "h500_best_periodic_mean": "0.80",
            "h1000_best_periodic_mean": "1.00",
        },
        {
            "root_label": "mlp_sparse",
            "system_key": "sys1",
            "seed_name": "seed_0",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/mlp_sparse/sys1/seed_0/20260409-100000",
            "h100_best_periodic_mean": "0.25",
            "h500_best_periodic_mean": "0.35",
            "h1000_best_periodic_mean": "0.45",
        },
        {
            "root_label": "mlp_sparse",
            "system_key": "sys1",
            "seed_name": "seed_1",
            "seed": "1",
            "run_id": "20260409-100001",
            "run_dir": "/tmp/mlp_sparse/sys1/seed_1/20260409-100001",
            "h100_best_periodic_mean": "0.35",
            "h500_best_periodic_mean": "0.45",
            "h1000_best_periodic_mean": "0.55",
        },
        {
            "root_label": "mlp_sparse",
            "system_key": "sys2",
            "seed_name": "seed_0",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/mlp_sparse/sys2/seed_0/20260409-100000",
            "h100_best_periodic_mean": "0.70",
            "h500_best_periodic_mean": "0.90",
            "h1000_best_periodic_mean": "1.10",
        },
        {
            "root_label": "mlp_sparse",
            "system_key": "sys2",
            "seed_name": "seed_1",
            "seed": "1",
            "run_id": "20260409-100001",
            "run_dir": "/tmp/mlp_sparse/sys2/seed_1/20260409-100001",
            "h100_best_periodic_mean": "0.80",
            "h500_best_periodic_mean": "1.00",
            "h1000_best_periodic_mean": "1.20",
        },
        {
            "root_label": "mlp_zero",
            "system_key": "sys1",
            "seed_name": "seed_0",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/mlp_zero/sys1/seed_0/20260409-100000",
            "h100_best_periodic_mean": "0.12",
            "h500_best_periodic_mean": "0.24",
            "h1000_best_periodic_mean": "0.36",
        },
        {
            "root_label": "mlp_zero",
            "system_key": "sys2",
            "seed_name": "seed_0",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/mlp_zero/sys2/seed_0/20260409-100000",
            "h100_best_periodic_mean": "0.45",
            "h500_best_periodic_mean": "0.65",
            "h1000_best_periodic_mean": "0.85",
        },
    ]
    interp_rows = [
        {
            "root_label": "lista_a",
            "system_key": "sys1",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/lista_a/sys1/seed_0/20260409-100000",
            "support_scheme": "absolute:0.001",
            "subset": "deep",
            "h_basin_given_support": "0.01",
            "h_support_given_basin": "0.10",
            "support_nmi": "0.80",
            "u_exact": "0.90",
            "family_h_family_given_basin": "0.05",
            "support_projection_self_over_base": "0.60",
            "support_freeze_self_over_base_h20": "0.40",
            "support_persistence": "0.85",
            "operator_between_over_within": "2.50",
        },
        {
            "root_label": "lista_a",
            "system_key": "sys2",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/lista_a/sys2/seed_0/20260409-100000",
            "support_scheme": "absolute:0.001",
            "subset": "deep",
            "h_basin_given_support": "0.02",
            "h_support_given_basin": "0.15",
            "support_nmi": "0.75",
            "u_exact": "0.88",
            "family_h_family_given_basin": "0.07",
            "support_projection_self_over_base": "0.70",
            "support_freeze_self_over_base_h20": "0.50",
            "support_persistence": "0.83",
            "operator_between_over_within": "2.00",
        },
        {
            "root_label": "mlp_sparse",
            "system_key": "sys1",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/mlp_sparse/sys1/seed_0/20260409-100000",
            "support_scheme": "absolute:0.001",
            "subset": "deep",
            "h_basin_given_support": "0.05",
            "h_support_given_basin": "0.20",
            "support_nmi": "0.60",
            "u_exact": "0.70",
            "family_h_family_given_basin": "0.10",
            "support_projection_self_over_base": "1.10",
            "support_freeze_self_over_base_h20": "0.90",
            "support_persistence": "0.70",
            "operator_between_over_within": "1.50",
        },
        {
            "root_label": "mlp_sparse",
            "system_key": "sys2",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/mlp_sparse/sys2/seed_0/20260409-100000",
            "support_scheme": "absolute:0.001",
            "subset": "deep",
            "h_basin_given_support": "0.04",
            "h_support_given_basin": "0.18",
            "support_nmi": "0.62",
            "u_exact": "0.72",
            "family_h_family_given_basin": "0.11",
            "support_projection_self_over_base": "0.95",
            "support_freeze_self_over_base_h20": "0.85",
            "support_persistence": "0.72",
            "operator_between_over_within": "1.60",
        },
        {
            "root_label": "mlp_zero",
            "system_key": "sys1",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/mlp_zero/sys1/seed_0/20260409-100000",
            "support_scheme": "absolute:0.001",
            "subset": "deep",
            "h_basin_given_support": "0.015",
            "h_support_given_basin": "0.12",
            "support_nmi": "0.78",
            "u_exact": "0.86",
            "family_h_family_given_basin": "0.06",
            "support_projection_self_over_base": "0.75",
            "support_freeze_self_over_base_h20": "0.55",
            "support_persistence": "0.81",
            "operator_between_over_within": "2.20",
        },
        {
            "root_label": "mlp_zero",
            "system_key": "sys2",
            "seed": "0",
            "run_id": "20260409-100000",
            "run_dir": "/tmp/mlp_zero/sys2/seed_0/20260409-100000",
            "support_scheme": "absolute:0.001",
            "subset": "deep",
            "h_basin_given_support": "0.025",
            "h_support_given_basin": "0.14",
            "support_nmi": "0.76",
            "u_exact": "0.84",
            "family_h_family_given_basin": "0.08",
            "support_projection_self_over_base": "0.80",
            "support_freeze_self_over_base_h20": "0.60",
            "support_persistence": "0.80",
            "operator_between_over_within": "2.10",
        },
    ]

    forecast_csv = tmp_path / "forecasting_rows.csv"
    interp_csv = tmp_path / "interpretability_rows.csv"
    _write_csv(forecast_csv, forecast_rows)
    _write_csv(interp_csv, interp_rows)

    summary = build_summary(
        forecasting_rows=_latest_forecasting_rows(_read_csv(forecast_csv)),
        interpretability_rows=_latest_interpretability_rows(_read_csv(interp_csv)),
        candidate_roots=["lista_a", "lista_b"],
        control_roots=["mlp_sparse", "mlp_zero"],
        support_scheme="absolute:0.001",
        subset="deep",
        good_threshold=0.95,
    )

    assert summary["forecast_summary"]["lista_a"]["H1000 best"] == pytest.approx(0.6)
    assert summary["forecast_summary"]["lista_a"]["good systems (H1000 best)"] == 2
    assert summary["interpretability_summary"]["lista_a"]["H(S|B)"] == pytest.approx(0.125)
    assert summary["interpretability_summary"]["lista_a"]["U_exact"] == pytest.approx(0.89)

    lista_vs_sparse = summary["pairwise"]["lista_a"]["mlp_sparse"]
    assert lista_vs_sparse["H1000 best"]["left_better"] == 2
    assert lista_vs_sparse["H1000 best"]["right_better"] == 0
    assert lista_vs_sparse["H(S|B)"]["left_better"] == 2
    assert lista_vs_sparse["U_exact"]["left_better"] == 2

    lista_vs_zero = summary["pairwise"]["lista_a"]["mlp_zero"]
    assert lista_vs_zero["H1000 best"]["left_better"] == 1
    assert lista_vs_zero["H1000 best"]["right_better"] == 0
    assert lista_vs_zero["H1000 best"]["ties"] == 1
    assert lista_vs_zero["own/base"]["left_better"] == 2
