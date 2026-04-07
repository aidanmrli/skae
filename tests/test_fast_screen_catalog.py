"""Regression tests for the fast Claude-catalog screen."""

from __future__ import annotations

import pytest

from tools.fast_screen_catalog import screen_system


def _screen_signature(name: str) -> dict[str, object]:
    result, *_ = screen_system(
        name,
        n_traj=24,
        traj_len=120,
        extra_steps=300,
        seed=42,
    )
    return {
        "n_basins": int(result["n_basins"]),
        "min_occupancy": float(result["min_occupancy"]),
        "overall_crossing": float(result["overall_crossing"]),
        "per_basin_crossing": {
            int(key): float(value)
            for key, value in (result.get("per_basin_crossing") or {}).items()
        },
    }


@pytest.mark.parametrize("name", ["cal_hexagon_6", "var_depth_gradient_4"])
def test_screen_system_is_repeatable_for_fixed_seed(name: str) -> None:
    first = _screen_signature(name)
    second = _screen_signature(name)

    assert first["n_basins"] == second["n_basins"]
    assert first["min_occupancy"] == pytest.approx(second["min_occupancy"], abs=1e-12)
    assert first["overall_crossing"] == pytest.approx(second["overall_crossing"], abs=1e-12)
    assert first["per_basin_crossing"].keys() == second["per_basin_crossing"].keys()
    for key in first["per_basin_crossing"]:
        assert first["per_basin_crossing"][key] == pytest.approx(
            second["per_basin_crossing"][key],
            abs=1e-12,
        )
