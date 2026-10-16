"""Per-system forecasting tables (appendix ``tab:persystem_mb_h*`` and ``tab:persystem_dysts_h*``).

One table per benchmark and horizon: the seed IQM of each system. Top block,
the sparse models, each cell marked per system (after Holm across systems)
when the model is significantly lower than Dense MLP (``\\ast``), DMD
(``\\dagger``) or both polynomial EDMD rollouts (``\\ddagger``); bottom
block, the baselines, unmarked.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from paper.config import systems_for
from paper.systems import DYSTS_NAMES, MULTIBASIN_SYSTEMS

from .common import DENSE, HORIZONS, SPARSE, Coverage, benchmark_forecasts, cell, passing, row, seed_iqm, write

BASELINES = (DENSE, "dmd", "edmd_poly_direct", "edmd_poly", "edmd_rbf_direct", "edmd_rbf")
TOP = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l ccccc@{}}", r"\toprule",
       r"System & LISTA & LISTA-BD & LISTA-SB & Sparse MLP-BD & Sparse MLP \\", r"\midrule", r"\addlinespace[1pt]"]
BOTTOM = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l cccccc@{}}", r"\toprule",
          r" & & & \multicolumn{2}{c}{Poly.\ EDMD} & \multicolumn{2}{c}{RBF EDMD} \\",
          r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
          r"System & Dense MLP & DMD & direct & re-lifted & direct & re-lifted \\", r"\midrule"]
END = [r"\bottomrule", r"\end{tabular*}"]


def system_name(system: str) -> str:
    """The display name in LaTeX (an en dash as ``--``)."""
    spec = MULTIBASIN_SYSTEMS.get(system)
    return spec.display_name if spec else DYSTS_NAMES[system].replace("–", "--")


def table(values, systems, horizon, coverage: Coverage) -> List[str]:
    """The LaTeX lines of one benchmark and horizon: the marked sparse models, then the baselines."""
    marks = {}
    for model in SPARSE:
        dense = passing(values, model, DENSE, horizon, systems)
        dmd = passing(values, model, "dmd", horizon, systems)
        edmd = (passing(values, model, "edmd_poly_direct", horizon, systems)
                & passing(values, model, "edmd_poly", horizon, systems))
        for system in systems:
            marks[(model, system)] = ((r"\ast" if system in dense else "") + (r"\dagger" if system in dmd else "")
                                      + (r"\ddagger" if system in edmd else ""))
    def value(label, system, marks=""):
        return cell(seed_iqm(values, label, system, horizon), marks,
                    missing=not coverage.complete(values, label, [system], horizon))

    top = [row(system_name(s), [value(m, s, marks[(m, s)]) for m in SPARSE]) for s in systems]
    bottom = [row(system_name(s), [value(label, s) for label in BASELINES]) for s in systems]
    return TOP + top + END + ["", r"\smallskip", ""] + BOTTOM + bottom + END


def build(evidence: Path, out: Path, coverage: Coverage, benchmark: str) -> List[Path]:
    """The three per-system tables of one benchmark from its forecasting and baseline files."""
    values = benchmark_forecasts(evidence, benchmark)
    return [write(Path(out) / f"per_system_{benchmark}_h{horizon}.tex",
                  table(values, systems_for(benchmark), horizon, coverage)) for horizon in HORIZONS[benchmark]]
