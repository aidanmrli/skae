"""Table 2 (``tab:support_coordinate_h21``): accumulated forecast error at horizon 21 under support interventions.

Mean, SD, median and quartiles over the evaluation points of each rollout in
``interventions.csv.gz`` (written by :mod:`paper.analysis.interventions`).
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from paper.io import read_csv

from .common import Coverage, number, write

HORIZON = 21
DROPS = (1, 2, 3, 5, 10)  # the drop_top_k rows of the table
HEADER = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Rollout & Mean \(\pm\) SD & Median [IQR] \\", r"\midrule"]


def table2_rows(errors: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    """Mean, SD, median and quartiles of the accumulated error at H = 21, from ``{condition: [points]}``.

    The random-support row pools its 20 repeats. ``drop_top_k`` exists only up
    to the smallest active count over the starts; a missing drop count is
    skipped with a warning.
    """
    groups = [("Standard", ["baseline"])]
    for k in DROPS:
        if f"drop_top_{k}" in errors:
            groups.append((f"Drop top-{k}", [f"drop_top_{k}"]))
        else:
            warnings.warn(f"no drop_top_{k} condition: some start has fewer than {k} active coordinates")
    groups.append(("Random support", [k for k in errors if k.startswith("random_support_")]))
    rows = []
    for label, keys in groups:
        values = np.concatenate([errors[k] for k in keys])
        rows.append({"condition": label, "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
                     "median": float(np.median(values)), "q1": float(np.percentile(values, 25)),
                     "q3": float(np.percentile(values, 75))})
    return rows


def build(evidence: Path, out: Path, coverage: Coverage) -> Path:
    """Table 2 from the one intervention run (``coverage`` is unused: the file holds that run or is missing)."""
    errors = defaultdict(dict)  # condition (in file order) -> point -> accumulated error at H = 21
    for record in read_csv(Path(evidence) / "interventions.csv.gz"):
        if int(record["horizon"]) == HORIZON:
            errors[record["condition"]][int(record["point_id"])] = float(record["cumulative_mse_sum"])
    lines = list(HEADER)
    for stats in table2_rows({condition: np.array([points[p] for p in sorted(points)])
                              for condition, points in errors.items()}):
        spread = f"{number(stats['mean'])}{{\\pm}}{number(stats['sd'])}"
        quartiles = f"{number(stats['median'])}\\,[{number(stats['q1'])},{number(stats['q3'])}]"
        if stats["condition"] == "Standard":
            spread, quartiles = rf"\mathbf{{{spread}}}", rf"\mathbf{{{quartiles}}}"
        lines.append(f"{stats['condition']} & ${spread}$ & ${quartiles}$ \\\\")
    return write(Path(out), lines + [r"\bottomrule", r"\end{tabular}"])
