"""Tests for the paper-reachable standalone baseline task matrix."""

from argparse import Namespace

import pytest

from tools.build_paper_baseline_tasks import _build_rows
from tools.summarize_paper_baseline_suite import METHOD_ORDER, _summarize


def _args(baseline_families: str) -> Namespace:
    return Namespace(
        systems="gated_local_linear,claude:arrested_spiral",
        seeds="0,1",
        baseline_families=baseline_families,
        horizons="100,500,1000",
        num_trajectories=256,
        trajectory_length=1000,
        train_fraction=0.6,
        ridge_lambda=1e-6,
        edmd_degree=3,
        kernel_centers=128,
        kernel_gamma=0.0,
        max_train_pairs=0,
        num_components=4,
        component_mode="fixed",
        env_dt=0.0,
        dysts_dt_multiplier=0.0,
        dysts_standardize=False,
        config_name="default",
        torch_threads=1,
    )


def test_baseline_tasks_emit_exact_paper_method_matrix():
    rows = _build_rows(_args("classical_koopman,mixture_local_linear"))

    assert len(rows) == 8
    methods_by_family = {
        row["baseline_family"]: row["methods"]
        for row in rows
    }
    assert methods_by_family == {
        "classical_koopman": "dmd,edmd_poly,rbf_dictionary_edmd",
        "mixture_local_linear": "kmeans_hard,gmm_hard,gmm_soft",
    }
    assert METHOD_ORDER == [
        "dmd",
        "edmd_poly",
        "rbf_dictionary_edmd",
        "kmeans_hard",
        "gmm_hard",
        "gmm_soft",
    ]
    assert all(not any(key.startswith("local_") for key in row) for row in rows)


def test_unknown_baseline_family_is_rejected():
    with pytest.raises(ValueError, match="Unknown baseline family"):
        _build_rows(_args("unsupported_family"))


def test_baseline_summary_ignores_methods_outside_the_paper_roster():
    rows = [
        {
            "status": "ok",
            "method": method,
            "horizon": "100",
            "system": "gated_local_linear",
            "mse": "1.0",
        }
        for method in ("dmd", "retired_local_method")
    ]

    per_system, summary = _summarize(rows, [100])

    assert {row["method"] for row in per_system} == {"dmd"}
    assert {row["method"] for row in summary} == {"dmd"}
