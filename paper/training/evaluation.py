"""Score a trained run's held-out forecasts and write ``metrics.json``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from paper import config
from paper.data import HardInitSampler, test_set, validation_forecast_set
from paper.evaluate import PRIMARY_CHECKPOINT, forecast_metrics, load_run_model, select_period
from paper.io import write_json


def run_settings() -> Dict[str, object]:
    """The constants of ``paper.config`` and ``paper.data`` that a run depends on but ``config.json`` does not hold.

    ``run_evaluation`` writes them to ``metrics.json`` (key ``settings``), and ``collect`` warns about runs whose
    settings differ from the current ones.
    """
    hard_init = {name.lower(): value for name, value in vars(HardInitSampler).items() if name.isupper()}
    return {
        "batch_streams": config.BATCH_STREAMS,
        "lr_tail_fraction": config.LR_TAIL_FRACTION,
        "lr_final_factor": config.LR_FINAL_FACTOR,
        "hard_init": {**hard_init, "seed_offset": config.HARD_INIT_SEED_OFFSET,
                      "noise_seed_offset": config.HARD_INIT_NOISE_SEED_OFFSET},
        "test_starts": config.TEST_STARTS,
        "test_seed_offset": config.TEST_SEED_OFFSET,
        "validation_starts": config.VALIDATION_STARTS,
        "validation_seed_offset": config.VALIDATION_SEED_OFFSET,
        "selection_horizon": config.SELECTION_HORIZON,
        "divergence_radius": config.DIVERGENCE_RADIUS,
    }


def run_evaluation(run_dir: Path, device: str = "cpu", data_dir: Path = Path("data")) -> Path:
    """Score ``final.pt`` on the test starts and write ``metrics.json``; return its path.

    The re-encoding period is chosen on the validation starts (lowest MSE at
    ``SELECTION_HORIZON``); its test error is reported as ``selected_periodic``.
    ``metrics.json`` holds the keys of :func:`paper.evaluate.forecast_metrics`
    (``modes``, ``selected_period``, ``selected_periodic``) and
    ``validation_mse_at_selected_period``, ``run`` (the run name), ``checkpoint``,
    ``selected_step`` (the last update) and ``settings`` (:func:`run_settings`).
    """
    run = load_run_model(run_dir, device)
    cfg, test, validation = run.cfg, None, None
    if cfg.benchmark == "dysts":  # Dysts forecasts start from cached trajectories, multibasin ones from resets
        from paper.systems.dysts import load_split

        test, validation = load_split(data_dir, cfg.system, "test"), load_split(data_dir, cfg.system, "val")
    starts, truth = test_set(run.system, cfg, test)
    val_starts, val_truth = validation_forecast_set(run.system, cfg, validation)
    period, validation_mse = select_period(run.model, val_starts, val_truth, cfg.eval, run.fallback,
                                           torch.device(device), run.radius)
    results = forecast_metrics(run.model, starts, truth, cfg.eval, device=torch.device(device), fallback=run.fallback,
                               radius=run.radius, selected_period=period)
    results.update(validation_mse_at_selected_period=validation_mse, run=cfg.run_name,
                   checkpoint=PRIMARY_CHECKPOINT, selected_step=run.step, settings=run_settings())
    write_json(Path(run_dir) / "metrics.json", results, indent=1)
    return Path(run_dir) / "metrics.json"
