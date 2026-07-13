import numpy as np
import torch

from experiments.benchmark_suite.baselines import fit_arx
from experiments.benchmark_suite.data import generate_lorenz96
from experiments.benchmark_suite.metrics import summarize_rows
from experiments.benchmark_suite.models import koopman_diagnostics


def test_lorenz96_generation_is_deterministic_for_small_case():
    kwargs = dict(
        dimension=8,
        forcing=8.0,
        n_trajectories=2,
        time_points=5,
        seed=123,
        warmup_steps=4,
        anchor_gap_saved=1,
        reequilibration_steps=2,
    )
    first = generate_lorenz96(**kwargs)
    second = generate_lorenz96(**kwargs)
    assert first.shape == (2, 5, 8)
    assert first.dtype == np.float32
    np.testing.assert_allclose(first, second)


def test_koopman_diagnostics_reports_effective_density_keys():
    diag = koopman_diagnostics(torch.eye(4))
    assert diag["k_l1"] == 4.0
    assert diag["spectral_radius"] == 1.0
    assert diag["effective_density_1e4"] == 0.25
    assert diag["effective_density_1e3"] == 0.25
    assert diag["effective_density_1e2"] == 0.25


def test_arx_fit_predicts_linear_series():
    n = 80
    u = np.sin(np.linspace(0, 4, n)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + 0.25 * u[t]
    model = fit_arx(u, y, order_y=1, order_u=1, ridge=1e-8)
    pred = model.one_step_series(u, y)
    mask = np.isfinite(pred)
    assert np.mean((pred[mask] - y[mask]) ** 2) < 1e-6


def test_summarize_rows_skips_all_nan_groups():
    rows = [
        {
            "benchmark": "b",
            "condition": "c",
            "model": "m",
            "split": "test",
            "horizon": 1,
            "latent_dim": 1,
            "sparsity_coefficient": 0.0,
            "metric_name": "bad",
            "metric_value": float("nan"),
        },
        {
            "benchmark": "b",
            "condition": "c",
            "model": "m",
            "split": "test",
            "horizon": 1,
            "latent_dim": 1,
            "sparsity_coefficient": 0.0,
            "metric_name": "nrmse",
            "metric_value": 1.0,
        },
    ]
    summary = summarize_rows(rows, n_resamples=10, seed=0)
    assert [row["metric_name"] for row in summary] == ["nrmse"]
