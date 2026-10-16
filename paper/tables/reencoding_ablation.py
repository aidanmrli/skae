"""Re-encoding ablation table (appendix ``tab:reencoding_ablation``) at H = 100 from ``reencoding.csv``.

Conditions: (a) no re-encoding (``none``), (b) support-constrained
re-encoding (``support_frozen``; ``support_frozen_own`` at its own period),
(c) unrestricted re-encoding. A count is the number of system--seed cells in
which the first condition has strictly lower test MSE than the second, out of
all cells of the model; the ratio is the geometric mean over cells of the MSE
of (c) over (b).
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper.config import SEEDS, systems_for
from paper.io import read_csv

from .common import MISSING, Coverage, number, row, write

HORIZON = 100
MODELS = (("lista", "LISTA"), ("sparse_mlp", "Sparse MLP"), ("dense_mlp", "Dense MLP"))
# (candidate, reference) of each count column, in table order; the ratio column comes after the third.
COUNTS = (("support_frozen", "none"), ("unrestricted", "support_frozen"), ("unrestricted", "none"),
          ("support_frozen_own", "none"), ("unrestricted", "support_frozen_own"))
HEADER = [r"\begin{tabular}{@{}l ccc c cc@{}}", r"\toprule",
          r"& \multicolumn{3}{c}{Cells with lower test MSE} & Ratio & \multicolumn{2}{c}{(b) at its own period} \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(l){6-7}",
          r"Model & (b) \(<\) (a) & (c) \(<\) (b) & (c) \(<\) (a) & (c)/(b) & (b) \(<\) (a) & (c) \(<\) (b) \\",
          r"\midrule"]


def build(evidence: Path, out: Path, coverage: Coverage) -> Path:
    """The table from the H = 100 rows of ``reencoding.csv``."""
    mse = defaultdict(lambda: defaultdict(lambda: math.nan))  # (model, system, seed) -> condition -> test MSE
    for r in read_csv(Path(evidence) / "reencoding.csv"):
        if int(r["horizon"]) == HORIZON:
            value = float(r["mse"]) if r["mse"] else math.nan
            mse[(r["model"], r["system"], r["seed"])][r["condition"]] = value if math.isfinite(value) else math.nan
    lines = list(HEADER)
    for model, name in MODELS:
        cells = [scores for (label, _, _), scores in mse.items() if label == model]
        flag = "" if coverage.count(model, len(cells), len(systems_for("multibasin")) * len(SEEDS)) else MISSING
        wins = [f"{sum(c[a] < c[b] for c in cells)}/{len(cells)}" for a, b in COUNTS]  # NaN (diverged) never wins
        ratios = [c["unrestricted"] / c["support_frozen"] for c in cells
                  if c["unrestricted"] > 0 and c["support_frozen"] > 0]
        ratio = f"${number(float(np.exp(np.mean(np.log(ratios)))), digits=2)}$" if ratios else "--"
        lines.append(row(name, [text + flag for text in (*wins[:3], ratio, *wins[3:])]))
    return write(Path(out), lines + [r"\bottomrule", r"\end{tabular}"])
