"""Forecast error against time on the 10 Dysts systems (LISTA, Sparse MLP, Dense MLP): the numbers of Figure 3.

Every run is rolled out from all 100 held-out test starts for 5,000 stored
steps at its validation-selected re-encoding period, with the evaluation's
divergence fallback (the reported forecast). The squared error at step ``t``
is ``||x_hat_t - x_t||^2`` summed over state coordinates (float32); per run,
its median over the starts is taken at every step. The figure's line is the
median over the 15 seeds of these per-seed medians, its band their 25th-75th
percentiles (NumPy linear interpolation). Writes ``error_growth.json`` (values
at steps 1-4, then every 5th step, to 6 significant digits).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from paper.analysis import finish, pinned_threads, producer_parser
from paper.config import SEEDS, find_run, systems_for
from paper.io import write_json
from paper.systems import DYSTS_NAMES

MODELS = ("dense_mlp", "sparse_mlp", "lista")  # Figure 3 draws them in this order, LISTA on top
HORIZON = 5000
PLOTTED_STEPS = np.unique(np.r_[1, 2, 3, 4, np.arange(5, HORIZON + 1, 5)])
STYLES = {
    "dense_mlp": {"label": "Dense MLP", "color": "#D62728", "ls": "-", "z": 3},
    "sparse_mlp": {"label": "Sparse MLP", "color": "#009E73", "ls": "(0, (3.2, 1.6))", "z": 4},
    "lista": {"label": "LISTA", "color": "#0072B2", "ls": "-", "z": 5},
}
# Written into error_growth.json, so that the evidence file says what its numbers are.
WHAT = ("Forecast error vs time on the 10 Dysts systems; periodic re-encoding at each run's validation-selected "
        "period (the paper's reported rollout); final checkpoint; 100 held-out test starts x 15 seeds per model; "
        "5,000 stored steps = 5 dominant periods.")
AGGREGATION = ("squared error at step t = ||x_hat_t - x_t||^2 summed over state coordinates (float32, divergence "
               "fallback to the training-mean state as in the forecasting evaluation). Per run: median over the 100 "
               "test starts at each step. Line: median over the 15 seeds of these per-seed medians; band: 25th-75th "
               "percentile over seeds (numpy linear interpolation).")


def median_error(run_dir: Path, data_dir: Path, device: str = "cpu") -> np.ndarray:
    """``[5000]`` float32: median over the test starts of the squared error at steps 1..5000."""
    from paper.data import test_set
    from paper.evaluate import load_run_model, rollout_with_fallback
    from paper.systems.dysts import load_split

    period = int(json.loads((Path(run_dir) / "metrics.json").read_text())["selected_period"])
    model, cfg, system, _, fallback, radius = load_run_model(run_dir, device)
    starts, truth = test_set(system, cfg, load_split(Path(data_dir), cfg.system, "test"))
    forecast, _ = rollout_with_fallback(model, starts.to(device), HORIZON, period, fallback, radius)
    squared = torch.sum((forecast.cpu().float() - truth[:HORIZON].float()) ** 2, dim=-1)
    squared = torch.where(torch.isfinite(squared), squared, torch.nan).numpy()  # [5000, starts]
    return np.nanpercentile(squared, 50, axis=1).astype(np.float32)


def _rounded(values: np.ndarray) -> list:
    return [float(f"{v:.6g}") for v in values]


def summarize(medians: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, object]:
    """The evidence payload from ``{system: {model: [seeds, 5000] per-seed medians}}``."""
    values, bands = {}, []
    for system, by_model in medians.items():
        values[system] = {"title": DYSTS_NAMES[system]}
        for model in MODELS:
            low, line, high = np.percentile(np.asarray(by_model[model], dtype=np.float64), [25, 50, 75], axis=0)
            low, line, high = (array[PLOTTED_STEPS - 1] for array in (low, line, high))
            values[system][model] = {"line": _rounded(line), "band_low": _rounded(low), "band_high": _rounded(high)}
            bands += [low, high]
    bands = np.concatenate(bands)
    limits = [float(10 ** np.floor(np.log10(bands.min()))), float(10 ** np.ceil(np.log10(bands.max())))]
    return {
        "what": WHAT, "aggregation": AGGREGATION, "plotted_steps": PLOTTED_STEPS.tolist(),
        "y_limits": [int(v) if v.is_integer() else v for v in limits], "vertical_lines_at_periods": [2, 4],
        "styles": STYLES, "values": values,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = producer_parser("error-growth", __doc__).parse_args(argv)
    medians: Dict[str, Dict[str, np.ndarray]] = {}
    unevaluated = 0
    with pinned_threads():
        for system in systems_for("dysts"):
            runs = {model: [find_run(args.runs, "dysts", system, model, seed, args.include_short) for seed in SEEDS]
                    for model in MODELS}
            unevaluated += sum((run / "final.pt").exists() and not (run / "metrics.json").exists()
                               for model_runs in runs.values() for run in model_runs)
            if all((run / "metrics.json").exists() for model_runs in runs.values() for run in model_runs):
                medians[system] = {model: np.stack([median_error(run, args.data, args.device) for run in model_runs])
                                   for model, model_runs in runs.items()}
    if unevaluated:
        print(f"error-growth: {unevaluated} runs have no metrics.json (trained with --no-eval); rerun their train "
              f"command without --no-eval to evaluate them")
    path = args.evidence / "error_growth.json"
    if medians:
        write_json(path, summarize(medians), ensure_ascii=False)
    used = len(medians) * len(MODELS) * len(SEEDS)  # only systems with every seed of every model are used
    return finish(f"error-growth ({len(medians)} of {len(systems_for('dysts'))} systems complete)", used,
                  len(systems_for("dysts")) * len(MODELS) * len(SEEDS), args.allow_missing_runs, path)
