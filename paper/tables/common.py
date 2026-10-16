"""Shared helpers of the paper tables: evidence loading, seed IQMs, coverage, significance marks and LaTeX cells.

Per-run values are kept as ``{(label, system, key): {seed: value}}``, where
``key`` is a forecast horizon or a metric name. A table cell is the mean over
systems of the per-system seed IQMs. A cell computed without some of the
paper's runs (:class:`Coverage`) is marked ``(missing runs)``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from paper.analysis.stats import holm, interquartile_mean, paired_wilcoxon_less, paired_wilcoxon_less_raw
from paper.config import seeds_for
from paper.io import read_csv

Values = Dict[Tuple[str, str, object], Dict[int, float]]

MODELS = ("lista", "lista_bd", "lista_sb", "sparse_mlp_bd", "sparse_mlp", "dense_mlp")  # table order
SPARSE = MODELS[:-1]
DENSE = "dense_mlp"
HORIZONS = {"multibasin": (100, 500, 1000), "dysts": (100, 2000, 4000)}
ALPHA = 0.05
WIDE_COLUMN = re.compile(r"h(\d+)(_direct)?")
ROLLOUT_SUFFIX = {"selected_periodic": "", "direct": "_direct"}
MISSING = " (missing runs)"


def read_forecasts(*paths: Path) -> Values:
    """Test MSE of every run in forecasting or baseline evidence files.

    Wide files (``forecasting_*.csv``: ``model, system, seed, h{H}, h{H}_direct, ...``)
    give the label ``<model>`` to the reported, re-encoded forecast and
    ``<model>_direct`` to the direct rollout. Long files have one row per
    horizon and name the label in ``model``, or in ``method`` plus ``rollout``
    (``selected_periodic`` or ``direct``, which adds ``_direct``). Rows with a
    status other than ``ok`` and non-finite or non-positive values are left out.
    """
    values: Values = defaultdict(dict)
    for path in paths:
        for row in read_csv(path):
            if row.get("status") not in (None, "", "ok"):
                continue
            if "horizon" in row:
                label = row.get("model") or row["method"] + ROLLOUT_SUFFIX[row["rollout"]]
                items = [(label, int(row["horizon"]), row["mse"])]
            else:
                items = [(row["model"] + (match[2] or ""), int(match[1]), text) for column, text in row.items()
                         if (match := WIDE_COLUMN.fullmatch(column))]
            for label, horizon, text in items:
                runs, seed = values[(label, row["system"], horizon)], int(row["seed"])
                if seed in runs:
                    raise ValueError(f"{path}: two values for {label}, {row['system']}, seed {seed}, H={horizon}")
                value = float(text) if text else math.nan
                if math.isfinite(value) and value > 0:
                    runs[seed] = value
    return values


def benchmark_forecasts(evidence: Path, benchmark: str) -> Values:
    """Test MSE of the neural runs and classical baselines of one benchmark (``forecasting_<benchmark>.csv``,
    ``baselines_<benchmark>.csv``).

    A missing file adds no values: its cells show ``--`` and count as missing runs, so a reproduction of one
    benchmark still gets Table 1 under ``--allow-missing-runs``.
    """
    paths = [Path(evidence) / f"{kind}_{benchmark}.csv" for kind in ("forecasting", "baselines")]
    return read_forecasts(*(path for path in paths if path.is_file()))


def support_rows(evidence: Path) -> List[dict]:
    """Rows of ``support_transfer.csv`` behind the paper's alignment results.

    The high-margin test states at the paper's thresholds (``primary``), for
    runs with at least two observed basins (``informative``).
    """
    return [row for row in read_csv(Path(evidence) / "support_transfer.csv")
            if row["primary"] == "True" and row["subset"] == "high_margin" and row["informative"] == "True"]


def by_model(rows: Iterable[dict], column: str) -> Values:
    """``{(model, system, column): {seed: value}}`` of the finite values of one column."""
    values: Values = defaultdict(dict)
    for row in rows:
        value = float(row[column]) if row[column] else math.nan
        if math.isfinite(value):
            values[(row["model"], row["system"], column)][int(row["seed"])] = value
    return values


def seed_iqm(values: Values, label: str, system: str, key) -> float:
    """Interquartile mean over the seeds of one system (NaN without values)."""
    return interquartile_mean(values.get((label, system, key), {}).values())


def system_mean(values: Values, label: str, systems: Sequence[str], key) -> float:
    """Mean over systems of the per-system seed IQMs; systems without values are left out."""
    iqms = [v for v in (seed_iqm(values, label, system, key) for system in systems) if math.isfinite(v)]
    return float(np.mean(iqms)) if iqms else math.nan


class Coverage:
    """The runs of the paper's roster (``paper.config.paper_runs``) that one table lacks. A cell computed without
    some of its runs is marked ``(missing runs)``."""

    def __init__(self, name: str):
        self.name = name
        self.missing: Dict[str, Tuple[int, int]] = {}  # label -> (missing, expected) runs of its least complete cell

    def count(self, label: str, found: int, expected: int) -> bool:
        """Record a cell made from ``found`` of ``expected`` runs; return whether none is missing."""
        if found < expected:
            self.missing[label] = max(self.missing.get(label, (0, expected)), (expected - found, expected))
        return found >= expected

    def complete(self, values: Values, label: str, systems: Sequence[str], key, seeds=None) -> bool:
        """Whether every system has a value of every paper seed of ``label`` (a model, ``<model>_direct`` or a
        classical baseline, seeds 0-14), or of ``seeds``."""
        seeds = set(seeds_for(label.removesuffix("_direct")) if seeds is None else seeds)
        found = sum(len(seeds & values.get((label, system, key), {}).keys()) for system in systems)
        return self.count(label, found, len(seeds) * len(systems))

    def summary(self) -> Optional[str]:
        """One line naming the labels with missing runs, or None when no run is missing."""
        if not self.missing:
            return None
        shown = [f"{label} {missing} of {expected}" for label, (missing, expected) in list(self.missing.items())[:4]]
        more = f" and {len(self.missing) - 4} more" if len(self.missing) > 4 else ""
        return f"{self.name}: runs missing ({', '.join(shown)}{more})"


def passing(values: Values, candidate: str, reference: str, key, systems: Sequence[str],
            raw: bool = False) -> Set[str]:
    """Systems in which ``candidate`` is significantly lower than ``reference``.

    Within each system, a one-sided paired Wilcoxon signed-rank test over the
    seeds both have: on log10 values (MSE), or on raw differences with
    ``raw=True`` (the support entropy, which can be exactly zero). Holm's
    correction runs across all ``systems`` at ``ALPHA``. A system with fewer
    than four paired seeds, or only tied pairs, cannot be tested: it enters the
    correction with p = 1, so it never passes and does not weaken the
    correction of the others.
    """
    test = paired_wilcoxon_less_raw if raw else paired_wilcoxon_less
    p_values = []
    for system in systems:
        a, b = values.get((candidate, system, key), {}), values.get((reference, system, key), {})
        seeds = sorted(a.keys() & b.keys())
        x, y = np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])
        p = test(x, y) if len(seeds) >= 4 and not np.all(x == y) else math.nan
        p_values.append(p if math.isfinite(p) else 1.0)
    return {system for system, p in zip(systems, holm(p_values)) if p < ALPHA}


def two_level_marks(passed: Set[str], systems: Sequence[str]) -> str:
    r"""``\ast\ast`` when the test passes in every system, ``\ast`` in all but one.

    The count runs over ALL systems of the benchmark, not only the tested ones:
    a system with missing runs or too few paired seeds counts as not passing,
    so it prevents ``\ast\ast`` (and a second such system prevents ``\ast``).
    """
    missed = len(systems) - len(passed)
    return r"\ast\ast" if missed == 0 else r"\ast" if missed == 1 and passed else ""


def number(value: float, digits: int = 3) -> str:
    r"""``digits`` significant digits; ``a.bc{\times}10^{e}`` outside [1e-3, 1e4)."""
    if value == 0:
        return "0"
    rounded = float(f"{value:.{digits}g}")
    if abs(rounded) < 1e-3 or abs(rounded) >= 1e4:
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return f"{mantissa}{{\\times}}10^{{{int(exponent)}}}"
    if abs(rounded) >= 10 ** (digits - 1):  # an integer once rounded, e.g. 482 or 2060
        return f"{rounded:.0f}"
    return f"{value:#.{digits}g}"


def cell(value: float, marks: str = "", bold: bool = False, digits: int = 3, missing: bool = False) -> str:
    r"""``$0.654$``, ``${0.0292}^{\ast\ast}$`` or ``${\mathbf{0.0269}}^{\ast\ast}$``; ``--`` without a value.

    A value computed without some of the paper's runs is followed by ``(missing runs)``.
    """
    if not math.isfinite(value):
        return "--"
    text = number(value, digits)
    body = rf"\mathbf{{{text}}}" if bold else text
    flag = MISSING if missing else ""
    if not marks:
        return f"${body}${flag}"
    superscript = r"\,{}^" if r"\times" in text else "^"  # keeps the marks clear of a power of ten
    return f"${{{body}}}{superscript}{{{marks}}}${flag}"


def row(label: str, cells: Iterable[str]) -> str:
    return " & ".join([label, *cells]) + r" \\"


def write(path: Path, lines: Sequence[str]) -> Path:
    """Write ``lines`` (one per line) to ``path``, creating its directory; return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
