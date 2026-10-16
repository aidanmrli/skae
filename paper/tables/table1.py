"""Table 1 (``tab:forecasting_main``): forecasting MSE on both benchmarks and the support diagnostics.

Every cell is the mean over systems of the per-system seed IQMs. Forecasting
columns: the reported (re-encoded) forecast of the neural models and the
rollout named in each classical row. Support columns (multibasin, high-margin
test states): the entropy ``H(B|F)`` of the basin given the support family and
the number of support families. A sparse model's MSE and ``H(B|F)`` cells carry
the two-level marks of its test against Dense MLP; bold is the lowest value of
each MSE and ``H(B|F)`` column.
"""

from __future__ import annotations

import math
from pathlib import Path

from paper.config import systems_for

from .common import (DENSE, HORIZONS, MODELS, SPARSE, Coverage, benchmark_forecasts, by_model, cell, passing, row,
                     support_rows, system_mean, two_level_marks, write)

LABELS = {
    "lista": "LISTA",
    "lista_bd": "LISTA-BD",
    "lista_sb": "LISTA-SB",
    "sparse_mlp_bd": "Sparse MLP-BD",
    "sparse_mlp": r"Sparse MLP \citep{fathi2024course}",
    "dense_mlp": r"Dense MLP \emph{[baseline]}",
    # Re-lifting does not change DMD, so it has one row.
    "dmd": r"DMD \citep{schmid_dmd_2010,tu_dynamic_2014}",
    "edmd_poly_direct": r"Poly.\ EDMD \citep{williams_datadriven_2015}, direct",
    "edmd_poly": r"Poly.\ EDMD \citep{williams_datadriven_2015}, re-lifted",
    "edmd_rbf_direct": r"RBF EDMD \citep{williams_datadriven_2015}, direct",
    "edmd_rbf": r"RBF EDMD \citep{williams_datadriven_2015}, re-lifted",
}
HEADER = [
    r"\begin{tabular}{@{}l ccc ccc cc@{}}",
    r"\toprule",
    r"& \multicolumn{6}{c}{Forecasting MSE\(\downarrow\)} & \multicolumn{2}{c}{Support diagnostics} \\",
    r"\cmidrule(lr){2-7}\cmidrule(l){8-9}",
    r"& \multicolumn{3}{c}{Multibasin, 15 systems} & \multicolumn{3}{c}{Dysts, 10 systems} & "
    r"\multicolumn{2}{c}{Multibasin, 15 systems} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}",
    r"Model & H100 & H500 & H1000 & H100 & H2000 & H4000 & $H(B\!\mid\!F_{\rm abs})\downarrow$ & "
    r"$\overline{|F_{\rm abs}|}$ \\",
    r"\midrule",
]


def column(values, systems, key, labels, coverage: Coverage, raw=False, directional=True, digits=3):
    """Cells of one column: system means of ``labels``; in a lower-is-better column, marks and bold."""
    means = {label: system_mean(values, label, systems, key) for label in labels}
    best = min((m for m in means.values() if math.isfinite(m)), default=None) if directional else None
    cells = {}
    for label, mean in means.items():
        marks = ""
        if directional and label in SPARSE:
            marks = two_level_marks(passing(values, label, DENSE, key, systems, raw), systems)
        cells[label] = cell(mean, marks, bold=mean == best, digits=digits,
                            missing=not coverage.complete(values, label, systems, key))
    return cells


def build(evidence: Path, out: Path, coverage: Coverage) -> Path:
    """Table 1 from the forecasting and baseline files of both benchmarks and ``support_transfer.csv``."""
    evidence = Path(evidence)
    columns = []
    for benchmark in ("multibasin", "dysts"):
        values = benchmark_forecasts(evidence, benchmark)
        columns += [column(values, systems_for(benchmark), h, LABELS, coverage) for h in HORIZONS[benchmark]]
    support, systems = support_rows(evidence), systems_for("multibasin")
    for key, options in (("h_basin_given_family", {"raw": True}),
                         ("observed_family_count", {"directional": False, "digits": 2})):
        columns.append(column(by_model(support, key), systems, key, MODELS, coverage, **options))
    lines = list(HEADER)
    for label in LABELS:
        lines.append(row(LABELS[label], [cells.get(label, "--") for cells in columns]))
        if label in (SPARSE[-1], DENSE):
            lines.append(r"\midrule")
    return write(Path(out), lines + [r"\bottomrule", r"\end{tabular}"])
