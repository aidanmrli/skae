"""Sensitivity of the support-basin alignment to the two support thresholds (threshold-sensitivity table).

Repeats :mod:`paper.analysis.support_transfer` for LISTA and Sparse MLP on a
grid of absolute support thresholds ``a`` (``|z_i| > a``) and family Jaccard
thresholds ``j``. Basin labels, centers and the high-margin subset do not
depend on either; families are refitted per cell. Writes
``support_sensitivity.json.gz``: per model and cell, on the high-margin test
states and pooled over systems and seeds, the number of runs, the exact
matches and the mean normalized ``H(B|F)`` over informative runs (at least two
basins observed); threshold values appear as text.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from paper.analysis import finish, pinned_threads, producer_parser
from paper.analysis.support_transfer import evaluate
from paper.config import SEEDS, systems_for
from paper.io import write_gzip

ABSOLUTE_THRESHOLDS = (1e-4, 1e-3, 1e-2)
JACCARD_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MODELS = ("lista", "sparse_mlp")


def _true(value) -> bool:
    return str(value) == "True"  # rows may come from support_transfer.csv


def _finite(value) -> Optional[float]:
    value = None if value in (None, "") else float(value)
    return value if value is not None and math.isfinite(value) else None


def summarize(rows: Sequence[Dict]) -> List[Dict[str, object]]:
    """Per model and threshold cell (sorted by their text), on the high-margin states: runs, exact matches and
    the mean normalized ``H(B|F)`` over informative runs."""
    groups = defaultdict(list)
    for row in rows:
        if row["subset"] == "high_margin":
            groups[(str(row["model"]), str(row["absolute_threshold"]), str(row["jaccard_threshold"]))].append(row)
    summary = []
    for (model, absolute, jaccard), group in sorted(groups.items()):
        informative = [row for row in group if _true(row["informative"])]
        normalized = [v for v in (_finite(row["normalized_h_basin_given_family"]) for row in informative)
                      if v is not None]
        summary.append({"model": model, "absolute_threshold": absolute, "jaccard_threshold": jaccard,
                        "num_cells": len(group), "exact_bijections": sum(_true(row["exact_bijection"])
                                                                         for row in informative),
                        "normalized_h_basin_given_family_mean": float(np.mean(normalized)) if normalized else None})
    return summary


def summary_text(pooled: List[Dict[str, object]]) -> str:
    """The evidence file's JSON text: the threshold grid and the pooled summaries."""
    payload = {"settings": {"absolute_thresholds": list(ABSOLUTE_THRESHOLDS),
                            "jaccard_thresholds": list(JACCARD_THRESHOLDS)}, "pooled": pooled}
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = producer_parser("support-sensitivity", __doc__).parse_args(argv)
    with pinned_threads():
        rows = evaluate(args.runs, ABSOLUTE_THRESHOLDS, JACCARD_THRESHOLDS, args.device, args.include_short, MODELS)
    path = args.evidence / "support_sensitivity.json.gz"
    if rows:
        write_gzip(path, summary_text(summarize(rows)))
    used = len({(row["model"], row["system"], row["seed"]) for row in rows})
    return finish("support-sensitivity", used, len(MODELS) * len(systems_for("multibasin")) * len(SEEDS),
                  args.allow_missing_runs, path)
