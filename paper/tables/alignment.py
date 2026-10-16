"""Support--basin alignment tables of the appendix, on the high-margin multibasin test states.

* ``alignment_counts.tex`` (``tab:alignment_counts``): seeds per system whose
  support families match the basins one to one, and the fraction of test
  states kept in the high-margin subset;
* ``normalized_entropy.tex`` (``tab:normalized_entropy``): ``H(B|F)`` and
  ``H(B|F)/H(B)``, means over systems of per-system seed IQMs;
* ``threshold_sensitivity.tex`` (``tab:threshold_sensitivity``): mean
  ``H(B|F)/H(B)`` and exact matches over the threshold grid.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from paper.config import SEEDS, systems_for
from paper.systems import MULTIBASIN_SYSTEMS

from .common import MODELS, MISSING, Coverage, by_model, number, row, support_rows, system_mean, write

STACKED = {"lista": "LISTA", "lista_bd": "LISTA-BD", "lista_sb": "LISTA-SB",
           "sparse_mlp_bd": r"\begin{tabular}[b]{@{}c@{}}Sparse\\ MLP-BD\end{tabular}",
           "sparse_mlp": r"\begin{tabular}[b]{@{}c@{}}Sparse\\ MLP\end{tabular}",
           "dense_mlp": r"\begin{tabular}[b]{@{}c@{}}Dense\\ MLP\end{tabular}"}
MODEL_HEADER = " & ".join(STACKED[model] for model in MODELS)
SENSITIVITY_MODELS = (("lista", "LISTA"), ("sparse_mlp", "Sparse MLP"))


def alignment_counts(rows, coverage: Coverage) -> List[str]:
    """Exact matches per system and model (seeds), and the mean high-margin fraction of each system."""
    systems = systems_for("multibasin")
    exact = Counter((r["model"], r["system"]) for r in rows if r["exact_bijection"] == "True")
    seeds = {(model, system): {r["seed"] for r in rows if (r["model"], r["system"]) == (model, system)}
             for model in MODELS for system in systems}
    lines = [r"\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}l c cccccc@{}}", r"\toprule",
             rf"System & Kept & {MODEL_HEADER} \\", r"\midrule"]
    for system in systems:
        kept = [float(r["retained_fraction"]) for r in rows if r["system"] == system]  # same for every model
        counts = [str(exact[(m, system)]) + ("" if coverage.count(m, len(seeds[(m, system)]), len(SEEDS))
                                             else MISSING) for m in MODELS]
        lines.append(row(MULTIBASIN_SYSTEMS[system].display_name,
                         [rf"\({np.mean(kept):.3f}\)" if kept else "--", *counts]))
    cells = len({(r["system"], r["seed"]) for r in rows})
    totals = [str(sum(exact[(model, system)] for system in systems))
              + ("" if coverage.count(model, sum(len(seeds[(model, s)]) for s in systems), len(systems) * len(SEEDS))
                 else MISSING) for model in MODELS]
    return lines + [r"\midrule", row(rf"Total (of \({cells}\)) &", totals), r"\bottomrule", r"\end{tabular}"]


def normalized_entropy(rows, coverage: Coverage) -> List[str]:
    """``H(B|F)`` and ``H(B|F)/H(B)`` per model: means over systems of per-system seed IQMs."""
    systems = systems_for("multibasin")
    lines = [r"\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}l cccccc@{}}", r"\toprule",
             rf"& {MODEL_HEADER} \\", r"\midrule"]
    for column, label in (("h_basin_given_family", r"\(H(B\mid F_{\rm abs})\) (nats)"),
                          ("normalized_h_basin_given_family", r"\(H(B\mid F_{\rm abs})/H(B)\)")):
        values = by_model(rows, column)
        cells = []
        for model in MODELS:
            mean = system_mean(values, model, systems, column)
            flag = "" if coverage.complete(values, model, systems, column) else MISSING
            cells.append(rf"\({number(mean)}\){flag}" if math.isfinite(mean) else "--")
        lines.append(row(label, cells))
    return lines + [r"\bottomrule", r"\end{tabular}"]


def threshold_sensitivity(evidence: Path, rows, coverage: Coverage) -> List[str]:
    """Mean normalized ``H(B|F)`` (exact matches) per threshold cell, ``--`` without an informative run.

    Bold: the primary cell, the thresholds of the primary rows ``rows`` of ``support_transfer.csv``.
    """
    with gzip.open(Path(evidence) / "support_sensitivity.json.gz", "rt") as handle:
        payload = json.load(handle)
    primary = {(float(r["absolute_threshold"]), float(r["jaccard_threshold"])) for r in rows}
    absolute, jaccard = payload["settings"]["absolute_thresholds"], payload["settings"]["jaccard_thresholds"]
    pooled = {(r["model"], float(r["absolute_threshold"]), float(r["jaccard_threshold"])): r
              for r in payload["pooled"]}
    runs = len(systems_for("multibasin")) * len(SEEDS)
    lines = [rf"\setlength{{\tabcolsep}}{{5pt}}\begin{{tabular}}{{@{{}}l c {'c' * len(jaccard)}@{{}}}}", r"\toprule",
             rf"& & \multicolumn{{{len(jaccard)}}}{{c}}{{Jaccard threshold}} \\",
             rf"\cmidrule(l){{3-{2 + len(jaccard)}}}",
             row("Model & Absolute threshold", [rf"\({j:g}\)" for j in jaccard]), r"\midrule"]
    for index, (model, name) in enumerate(SENSITIVITY_MODELS):
        if index:
            lines.append(r"\midrule")
        for a in absolute:
            cells = []
            for j in jaccard:
                r = pooled.get((model, a, j))
                if r is None:  # no runs of this model in the evidence
                    cells.append("--")
                    continue
                mean = r["normalized_h_basin_given_family_mean"]  # None without an informative run
                text = (rf"\({mean:.3f}\)" if mean is not None and math.isfinite(mean) else "--") \
                    + rf" (\({r['exact_bijections']}\))"
                flag = "" if coverage.count(model, r["num_cells"], runs) else MISSING
                cells.append((rf"\textbf{{\boldmath{text}}}" if (a, j) in primary else text) + flag)
            label = name if a == absolute[0] else ""
            lines.append(row(f"{label} & \\(10^{{{round(math.log10(a))}}}\\)", cells))
    return lines + [r"\bottomrule", r"\end{tabular}"]


def build(evidence: Path, out: Path, coverage: Coverage) -> List[Path]:
    """The three tables, from ``support_transfer.csv`` and ``support_sensitivity.json.gz``."""
    rows, out = support_rows(evidence), Path(out)
    return [write(out / "alignment_counts.tex", alignment_counts(rows, coverage)),
            write(out / "normalized_entropy.tex", normalized_entropy(rows, coverage)),
            write(out / "threshold_sensitivity.tex", threshold_sensitivity(evidence, rows, coverage))]
