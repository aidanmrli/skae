import json
from pathlib import Path

import numpy as np
import pytest

from experiments.neurips_2026.global_k_distinct_laws_v2 import (
    _authenticate_mechanism_geometry,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_math import (
    antithetic_directions,
    authenticate_local_geometry,
    origin_linear_fit_diagnostics,
    sample_centered_disk,
)
from skae.config import get_config
from skae.data import make_env


ROOT = Path(__file__).resolve().parents[1]
CARD = json.loads((
    ROOT / "experiments/neurips_2026/global_k_distinct_laws_v2_card.json"
).read_text())


def _environment():
    cfg = get_config("generic_no_shrink")
    cfg.ENV.ENV_NAME = "gated_local_linear"
    cfg.ENV.GATED_LOCAL_LINEAR.DT = CARD["benchmark"]["dt"]
    return make_env(cfg)


def _disk_authentication(env, kind):
    protocol = CARD["evaluation_only_family_matching"]
    centers = env.unwrapped.points_2d.detach().cpu().numpy()
    offsets = sample_centered_disk(
        protocol[f"{kind}_points_per_basin"],
        protocol[f"{kind}_disk_radius"],
        protocol[f"{kind}_seed"],
    )
    return authenticate_local_geometry(
        env, centers[:, None, :] + offsets[None, :, :], category=kind,
        dt=CARD["benchmark"]["dt"],
        max_abs_error=CARD["validity"][
            "max_analytic_environment_step_disagreement"
        ],
    )


def test_every_frozen_geometry_construction_has_exact_count_region_and_rk4_map():
    env = _environment()
    calibration = _disk_authentication(env, "calibration")
    verification = _disk_authentication(env, "verification")
    routing = {"geometry_authentication": {
        "calibration": calibration, "verification": verification,
    }}
    centers = env.unwrapped.points_2d.detach().cpu().numpy()
    authentication = _authenticate_mechanism_geometry(
        env, centers, routing, CARD
    )
    expected = CARD["benchmark"]["geometry_authentication"][
        "expected_point_counts"
    ]
    assert authentication["passed"] is True
    assert authentication["point_counts_exact"] is True
    assert authentication["all_points_in_intended_regions"] is True
    assert authentication["observed_total_point_count"] == expected["total"] == 2139
    assert {
        name: row["total_point_count"]
        for name, row in authentication["categories"].items()
    } == {name: expected[name] for name in expected if name != "total"}
    assert authentication["maximum_analytic_rk4_abs_error"] <= 1e-6


def test_geometry_authentication_rejects_cross_basin_and_wrong_step(monkeypatch):
    env = _environment()
    centers = env.unwrapped.points_2d.detach().cpu().numpy()
    points = [centers[basin : basin + 1].copy() for basin in range(3)]
    points[0] = centers[1:2].copy()
    wrong_region = authenticate_local_geometry(
        env, points, category="wrong_region", dt=CARD["benchmark"]["dt"],
        max_abs_error=1e-6,
    )
    assert wrong_region["passed"] is False
    assert wrong_region["all_points_in_intended_region"] is False

    original_step = env.step
    monkeypatch.setattr(env, "step", lambda state: original_step(state) + 1e-3)
    wrong_step = authenticate_local_geometry(
        env, [centers[b : b + 1] for b in range(3)],
        category="wrong_step", dt=CARD["benchmark"]["dt"],
        max_abs_error=1e-6,
    )
    assert wrong_step["passed"] is False
    assert wrong_step["analytic_rk4_max_abs_error"] > 1e-6


def test_residual_detects_even_nonlinearity_that_preserves_average_slope():
    directions = antithetic_directions(64, 20260731)
    target = np.asarray([[0.08, -0.03], [0.02, -0.06]])
    linear_changes = directions @ target.T
    fitted, _residual, _energy, normalized = origin_linear_fit_diagnostics(
        directions, linear_changes
    )
    np.testing.assert_allclose(fitted, target, atol=1e-12)
    assert normalized == pytest.approx(0.0, abs=1e-12)

    even_component = np.column_stack((directions[:, 0] ** 2, np.zeros(64)))
    nonlinear_changes = linear_changes + even_component
    fitted, _residual, _energy, normalized = origin_linear_fit_diagnostics(
        directions, nonlinear_changes
    )
    np.testing.assert_allclose(fitted, target, atol=1e-12)
    assert normalized > 0.25
