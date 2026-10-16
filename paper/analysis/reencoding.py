"""Re-encoding ablation: what periodic re-encoding contributes (multibasin; LISTA, Sparse MLP, Dense MLP).

Rollouts of the same checkpoint from the same held-out test starts:

* ``none``: ``x_hat_{t+h} = D(E(x_t) K^h)``, no re-encoding;
* ``support_frozen``: every ``m`` steps the decoded prediction is re-encoded
  and projected onto the start's initial active pool
  ``P = {i : |E(x_t)_i| > 1e-3}``, so re-encoding can correct coefficients but
  never activate a new coordinate;
* ``unrestricted``: ordinary periodic re-encoding every ``m`` steps;
* ``support_frozen_own``: ``support_frozen`` at the period selected for it.

One period ``m`` per model and system is selected on validation starts only
(100 starts, seed offset 54321, error at step 500): the candidate minimizing
the mean over seeds of ``ln(MSE + 1e-12)``, ties to the smaller ``m``, selected
on ``unrestricted`` for both re-encoding conditions and on ``support_frozen``
for ``support_frozen_own``. Divergence and errors follow
:mod:`paper.evaluate`: from its first non-finite or out-of-radius prediction a
start is forecast by the training-mean state; a start's error at horizon ``H``
is the squared error summed over state dimensions, averaged over ``h <= H``.
Writes ``reencoding.csv``: one row per run, condition and horizon, with the
mean size of the initial active pool (``pool_size_mean``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from paper import config
from paper.analysis import finish, pinned_threads, producer_parser
from paper.config import SEEDS, SUPPORT_THRESHOLD, find_run, systems_for
from paper.data import test_set, validation_forecast_set
from paper.evaluate import TrainedRun, horizon_scores, load_run_model, rollout_with_fallback
from paper.io import write_csv

MODELS = ("lista", "sparse_mlp", "dense_mlp")
# condition -> (re-encoding period: the shared one or the support-frozen one, frozen support)
CONDITIONS = {"none": (None, False), "support_frozen": ("shared", True), "unrestricted": ("shared", False),
              "support_frozen_own": ("own", True)}
LOG_OFFSET = 1e-12


def validation_errors(run: TrainedRun, device: str) -> Dict[str, Dict[int, float]]:
    """Validation MSE at step 500 of ``unrestricted`` and ``support_frozen`` for every candidate period."""
    starts, truth = validation_forecast_set(run.system, run.cfg)
    horizon = config.SELECTION_HORIZON
    errors: Dict[str, Dict[int, float]] = {}
    for condition, frozen in (("unrestricted", False), ("support_frozen", True)):
        errors[condition] = {}
        for period in run.cfg.eval.periods:
            predictions, diverged_at = rollout_with_fallback(run.model, starts.to(device), horizon, period,
                                                             run.fallback, run.radius, frozen)
            scores = horizon_scores(predictions, diverged_at, truth, (horizon,))
            errors[condition][int(period)] = scores[str(horizon)]["mean"]
    return errors


def period_over_seeds(errors: Sequence[Dict[int, float]]) -> int:
    """The period shared by every seed of a (model, system), from ``[{period: validation MSE}]`` per seed.

    It minimizes the mean over seeds of ``ln(MSE + 1e-12)`` (ties: the smaller;
    none finite: the smallest). :func:`paper.evaluate.select_period` instead
    picks one period per run.
    """
    periods = sorted(errors[0])
    scores = {}
    for m in periods:
        values = [errors_of_seed[m] for errors_of_seed in errors]
        if all(v is not None and math.isfinite(v) for v in values):
            scores[m] = float(np.mean([math.log(v + LOG_OFFSET) for v in values]))
    return min(scores, key=lambda m: (scores[m], m)) if scores else periods[0]


def test_rows(run: TrainedRun, period: int, own_period: int, device: str) -> List[Dict[str, object]]:
    """Rows of the four conditions on the run's test starts at the selected periods."""
    starts, truth = test_set(run.system, run.cfg)
    horizons = tuple(run.cfg.eval.horizons)
    with torch.no_grad():
        pool = run.model.encode(starts.to(device)).abs() > SUPPORT_THRESHOLD
    pool_size = float(pool.sum(dim=-1).to(torch.float64).mean().item())
    periods = {None: None, "shared": period, "own": own_period}
    computed, rows = {}, []
    for name, (cadence, frozen) in CONDITIONS.items():
        setting = (periods[cadence], frozen)
        if setting not in computed:  # support_frozen_own repeats support_frozen when both periods agree
            rollout = rollout_with_fallback(run.model, starts.to(device), max(horizons), setting[0], run.fallback,
                                            run.radius, frozen)
            computed[setting] = horizon_scores(*rollout, truth, horizons)
        for h in horizons:
            stats = computed[setting][str(h)]
            rows.append({"benchmark": "multibasin", "system": run.cfg.system, "model": run.cfg.model_name,
                         "seed": run.cfg.seed, "horizon": h, "condition": name,
                         "period": "" if cadence is None else periods[cadence], "mse": stats["mean"],
                         "std": stats["std"], "num_valid": stats["num_valid"],
                         "divergence_rate": stats["divergence_rate"], "pool_size_mean": pool_size,
                         "selected_step": run.step})
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = producer_parser("reencoding", __doc__).parse_args(argv)
    rows: List[Dict[str, object]] = []
    used = 0
    with pinned_threads():
        for system in systems_for("multibasin"):
            for name in MODELS:
                directories = [find_run(args.runs, "multibasin", system, name, seed, args.include_short)
                               for seed in SEEDS]
                runs = [load_run_model(d, args.device) for d in directories if (d / "final.pt").exists()]
                if not runs:
                    continue
                errors = [validation_errors(run, args.device) for run in runs]
                period = period_over_seeds([e["unrestricted"] for e in errors])
                own_period = period_over_seeds([e["support_frozen"] for e in errors])
                for run in runs:
                    rows += test_rows(run, period, own_period, args.device)
                used += len(runs)
    path = args.evidence / "reencoding.csv"
    if rows:
        rows.sort(key=lambda row: (row["system"], row["model"], row["seed"]))  # stable: conditions, horizons kept
        write_csv(path, rows)
    return finish("reencoding", used, len(systems_for("multibasin")) * len(MODELS) * len(SEEDS),
                  args.allow_missing_runs, path)
