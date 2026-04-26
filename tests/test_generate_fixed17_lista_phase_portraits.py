from __future__ import annotations

import csv
from pathlib import Path

from tools.generate_fixed17_lista_phase_portraits import (
    _parse_mode_name,
    _system_title,
    collect_candidates,
)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system_key",
        "root_label",
        "run_dir",
        "seed",
        "env_dt",
        "num_steps",
        "h100_best_periodic_mean",
        "h500_best_periodic_mean",
        "h1000_best_periodic_mean",
        "h1000_best_periodic_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_collect_candidates_filters_dedups_and_sorts(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    first_csv = results_dir / "transition_rich_example_a" / "collect_pass0" / "forecasting_rows.csv"
    second_csv = results_dir / "transition_rich_example_b" / "collect_pass0" / "forecasting_rows.csv"

    _write_rows(
        first_csv,
        [
            {
                "system_key": "gated_local_linear",
                "root_label": "lista_alpha",
                "run_dir": "/tmp/run_a",
                "seed": "0",
                "env_dt": "0.04",
                "num_steps": "20000",
                "h100_best_periodic_mean": "0.05",
                "h500_best_periodic_mean": "0.10",
                "h1000_best_periodic_mean": "0.20",
                "h1000_best_periodic_mode": "periodic_10",
            },
            {
                "system_key": "gated_local_linear",
                "root_label": "lista_beta",
                "run_dir": "/tmp/run_b",
                "seed": "1",
                "env_dt": "0.04",
                "num_steps": "20000",
                "h100_best_periodic_mean": "0.06",
                "h500_best_periodic_mean": "0.08",
                "h1000_best_periodic_mean": "0.10",
                "h1000_best_periodic_mode": "periodic_20",
            },
            {
                "system_key": "gated_local_linear",
                "root_label": "generic_sparse_control",
                "run_dir": "/tmp/run_control",
                "seed": "2",
                "env_dt": "0.04",
                "num_steps": "20000",
                "h100_best_periodic_mean": "0.01",
                "h500_best_periodic_mean": "0.01",
                "h1000_best_periodic_mean": "0.01",
                "h1000_best_periodic_mode": "periodic_10",
            },
            {
                "system_key": "not_in_fixed17",
                "root_label": "lista_other",
                "run_dir": "/tmp/run_other",
                "seed": "3",
                "env_dt": "0.04",
                "num_steps": "20000",
                "h100_best_periodic_mean": "0.01",
                "h500_best_periodic_mean": "0.01",
                "h1000_best_periodic_mean": "0.01",
                "h1000_best_periodic_mode": "periodic_10",
            },
        ],
    )
    _write_rows(
        second_csv,
        [
            {
                "system_key": "gated_local_linear",
                "root_label": "lista_beta",
                "run_dir": "/tmp/run_b",
                "seed": "1",
                "env_dt": "0.04",
                "num_steps": "20000",
                "h100_best_periodic_mean": "0.07",
                "h500_best_periodic_mean": "0.09",
                "h1000_best_periodic_mean": "0.10",
                "h1000_best_periodic_mode": "periodic_20",
            },
            {
                "system_key": "gated_local_linear",
                "root_label": "lista_gamma",
                "run_dir": "/tmp/run_c",
                "seed": "4",
                "env_dt": "0.04",
                "num_steps": "10000",
                "h100_best_periodic_mean": "0.07",
                "h500_best_periodic_mean": "0.05",
                "h1000_best_periodic_mean": "0.10",
                "h1000_best_periodic_mode": "periodic_10",
            },
        ],
    )

    grouped = collect_candidates(results_dir, root_label_prefix="lista_")
    candidates = grouped["gated_local_linear"]

    assert [candidate.root_label for candidate in candidates] == [
        "lista_gamma",
        "lista_beta",
        "lista_alpha",
    ]
    assert [str(candidate.run_dir) for candidate in candidates] == [
        "/tmp/run_c",
        "/tmp/run_b",
        "/tmp/run_a",
    ]


def test_system_title_and_mode_parsing() -> None:
    assert _system_title("claude:var_l_shape_5") == "var_l_shape_5"
    assert _system_title("gated_local_linear") == "gated_local_linear"
    assert _parse_mode_name("no_reencode") == 0
    assert _parse_mode_name("every_step") == 1
    assert _parse_mode_name("periodic_10") == 10
