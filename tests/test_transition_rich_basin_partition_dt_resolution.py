"""Tests for transition-rich dt rescue resolution."""

from __future__ import annotations

from tools.resolve_transition_rich_basin_partition_dt import resolve_rows


def _find_arm(rows, *, model_variant: str, system_key: str):
    for row in rows:
        if row["model_variant"] == model_variant and row["system_key"] == system_key:
            return row
    raise AssertionError(f"Missing arm {(model_variant, system_key)}")


def test_transition_rich_dt_resolution_requests_halving_after_bad_default():
    selected_rows, request_rows, _report = resolve_rows(
        [
            {
                "root_label": "lista_dense_basin_partition",
                "system_key": "gated_local_linear",
                "env_dt": "0.04",
                "seed_name": "seed_0",
                "run_id": "20260407_000001",
                "run_dir": "/tmp/default_run",
                "h1000_best_periodic_mean": "75.0",
            }
        ],
        threshold=50.0,
        current_pass=0,
        max_halvings=2,
        min_seeds=1,
    )

    arm = _find_arm(
        selected_rows,
        model_variant="lista_dense_basin_partition",
        system_key="gated_local_linear",
    )
    assert arm["status"] == "pending_halving_1"

    request = _find_arm(
        request_rows,
        model_variant="lista_dense_basin_partition",
        system_key="gated_local_linear",
    )
    assert request["requested_dt"] == 0.02


def test_transition_rich_dt_resolution_accepts_first_good_halving():
    selected_rows, request_rows, _report = resolve_rows(
        [
            {
                "root_label": "lista_dense_basin_partition",
                "system_key": "gated_local_linear",
                "env_dt": "0.04",
                "seed_name": "seed_0",
                "run_id": "20260407_000001",
                "run_dir": "/tmp/default_run",
                "h1000_best_periodic_mean": "75.0",
            },
            {
                "root_label": "lista_dense_basin_partition",
                "system_key": "gated_local_linear",
                "env_dt": "0.02",
                "seed_name": "seed_0",
                "run_id": "20260407_000002",
                "run_dir": "/tmp/halved_run",
                "h1000_best_periodic_mean": "40.0",
            },
        ],
        threshold=50.0,
        current_pass=1,
        max_halvings=2,
        min_seeds=1,
    )

    arm = _find_arm(
        selected_rows,
        model_variant="lista_dense_basin_partition",
        system_key="gated_local_linear",
    )
    assert arm["selected_dt"] == 0.02
    assert arm["status"] == "accepted_halving_1"

    matching_requests = [
        row for row in request_rows
        if row["model_variant"] == "lista_dense_basin_partition"
        and row["system_key"] == "gated_local_linear"
    ]
    assert matching_requests == []
