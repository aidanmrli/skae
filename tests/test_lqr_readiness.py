"""Tests for LQR-readiness evaluation and decision utilities."""

import numpy as np

from tools.collect_lqr_decision_results import decide_winner
from tools.evaluate_lqr_readiness import (
    aggregate_metrics,
    ridge_fit_row_linear,
    solve_lqr,
)


def test_ridge_fit_row_linear_recovers_matrix():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    a_true = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.7, 0.1],
            [0.1, 0.0, 0.0, 0.6],
        ],
        dtype=np.float64,
    )
    y = x @ a_true + 1e-3 * rng.normal(size=(400, 4))

    a_hat = ridge_fit_row_linear(x, y, l2_reg=1e-6)
    err = np.linalg.norm(a_hat - a_true) / np.linalg.norm(a_true)
    assert err < 1e-2


def test_solve_lqr_stabilizes_unstable_system():
    # Row convention; transpose is the same for this diagonal case.
    a_row = np.array([[1.1, 0.0], [0.0, 0.95]], dtype=np.float64)
    b_col = np.eye(2, dtype=np.float64)

    success, _k, rho, _reason = solve_lqr(
        a_row=a_row,
        b_col=b_col,
        q_weight=1.0,
        r_weight=0.1,
    )

    assert success
    assert rho is not None
    assert rho < 1.0


def test_aggregate_metrics_uses_only_evaluable_regimes():
    regimes = [
        {
            "status": "ok",
            "local_fit_nrmse_1_step": 0.2,
            "local_fit_nrmse_h_step": 0.4,
            "dare_success": True,
            "closed_loop_stable": True,
            "closed_loop_cost_reduction": 0.3,
            "recovery_success_rate": 0.8,
        },
        {
            "status": "insufficient_samples",
            "dare_success": False,
            "closed_loop_stable": False,
        },
        {
            "status": "ok",
            "local_fit_nrmse_1_step": 0.4,
            "local_fit_nrmse_h_step": 0.6,
            "dare_success": False,
            "closed_loop_stable": False,
            "closed_loop_cost_reduction": None,
            "recovery_success_rate": None,
        },
    ]

    agg = aggregate_metrics(regimes)

    assert agg["num_regimes_total"] == 3
    assert agg["num_regimes_evaluable"] == 2
    assert np.isclose(agg["m1_local_fit_nrmse_1_step"], 0.3)
    assert np.isclose(agg["m2_lqr_feasibility_rate"], 0.5)
    assert np.isclose(agg["m3_closed_loop_stability_rate"], 0.5)


def test_decide_winner_lexicographic_m2_with_threshold():
    rows = []
    for seed in [0, 1, 2, 3]:
        for b_proxy in [8, 13, 20]:
            key = {
                "stage": 2,
                "system": "lyapunov",
                "target_size": 256,
                "b_proxy": b_proxy,
                "seed": seed,
            }
            rows.append(
                {
                    **key,
                    "arm": "bd_c1",
                    "m2_lqr_feasibility_rate": 0.85,
                    "m3_closed_loop_stability_rate": 0.82,
                    "m4_closed_loop_cost_reduction": 0.20,
                }
            )
            rows.append(
                {
                    **key,
                    "arm": "ah_prag",
                    "m2_lqr_feasibility_rate": 0.65,
                    "m3_closed_loop_stability_rate": 0.70,
                    "m4_closed_loop_cost_reduction": 0.18,
                }
            )

    decision = decide_winner(
        decision_rows=rows,
        arm_a="bd_c1",
        arm_b="ah_prag",
        num_bootstrap=200,
        bootstrap_seed=0,
        threshold=0.10,
    )

    assert decision["decision"] == "winner"
    assert decision["final_choice"] == "bd_c1"
    assert decision["decided_by"] == "M2"
