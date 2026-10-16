"""Training-window ablation table (appendix ``tab:window_ablation``): 9- and 64-state windows on multibasin.

Means over systems of per-system seed IQMs of the direct rollout and the
reported re-encoded forecast. The 64-state runs (``forecasting_window64.csv``,
models ``<model>_w64``) have fewer seeds; each is paired with the 9-state run
of the same model, system and seed in ``forecasting_multibasin.csv``.
"""

from __future__ import annotations

import math
from pathlib import Path

from paper.config import WINDOW64_SEEDS, systems_for

from .common import HORIZONS, MISSING, Coverage, number, read_forecasts, row, system_mean, write

MODELS = (("lista", "LISTA"), ("sparse_mlp", "Sparse MLP"), ("dense_mlp", "Dense MLP"))
HEADER = [r"\begin{tabular}{@{}ll ccc ccc@{}}", r"\toprule",
          r"& & \multicolumn{3}{c}{Direct rollout} & \multicolumn{3}{c}{Re-encoded (reported)} \\",
          r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
          r"Model & Window & H100 & H500 & H1000 & H100 & H500 & H1000 \\", r"\midrule"]


def short(value: float, missing: bool) -> str:
    """Three significant digits without trailing zeros (``0.098``, ``25``)."""
    if not math.isfinite(value):
        return "--"
    text = number(value)
    if "." in text and r"\times" not in text:
        text = text.rstrip("0").rstrip(".")
    return rf"\({text}\)" + (MISSING if missing else "")


def build(evidence: Path, out: Path, coverage: Coverage) -> Path:
    """The table from ``forecasting_window64.csv`` and the paired runs of ``forecasting_multibasin.csv``."""
    evidence, systems = Path(evidence), systems_for("multibasin")
    window64 = {(label.replace("_w64", ""), system, h): runs
                for (label, system, h), runs in read_forecasts(evidence / "forecasting_window64.csv").items()}
    # The 9-state runs paired with the 64-state ones: the same model, system and seeds.
    window9 = {key: {seed: value for seed, value in runs.items() if seed in window64.get(key, {})}
               for key, runs in read_forecasts(evidence / "forecasting_multibasin.csv").items()}
    lines = list(HEADER)
    for index, (model, name) in enumerate(MODELS):
        if index:
            lines.append(r"\midrule")
        for values, window in ((window9, "9"), (window64, "64")):
            cells = [short(system_mean(values, model + suffix, systems, h),
                           not coverage.complete(values, model + suffix, systems, h, WINDOW64_SEEDS))
                     for suffix in ("_direct", "") for h in HORIZONS["multibasin"]]
            lines.append(row(f"{name if window == '9' else ''} & \\({window}\\) states", cells))
    return write(Path(out), lines + [r"\bottomrule", r"\end{tabular}"])
