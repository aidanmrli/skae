"""The paper's tables, built from an evidence directory (``python -m paper tables``).

:func:`build_tables` reads an evidence directory with the layout of
``paper/evidence`` (the bundled one or one written by ``python -m paper
evidence``) and writes each paper table as a plain LaTeX ``tabular`` with
literal values:

====================================  ===============================================
``table1_forecasting.tex``            Table 1, ``tab:forecasting_main``
``table2_interventions.tex``          Table 2, ``tab:support_coordinate_h21``
``per_system_multibasin_h{H}.tex``    ``tab:persystem_mb_h{H}``, H = 100, 500, 1000
``per_system_dysts_h{H}.tex``         ``tab:persystem_dysts_h{H}``, H = 100, 2000, 4000
``alignment_counts.tex``              ``tab:alignment_counts``
``normalized_entropy.tex``            ``tab:normalized_entropy``
``threshold_sensitivity.tex``         ``tab:threshold_sensitivity``
``sparsity_mechanisms.tex``           ``tab:sparsity_mechanisms``
``reencoding_ablation.tex``           ``tab:reencoding_ablation``
``window_ablation.tex``               ``tab:window_ablation``
====================================  ===============================================

Each table module has ``build(evidence, out, coverage)``: it reads its
evidence files from the directory ``evidence``, writes its table to the path
``out`` (or its tables into the directory ``out``), returns the written
path(s) and records in ``coverage`` (:class:`common.Coverage`) every cell
computed without some of the paper's runs. ``per_system.build`` also takes the
benchmark.
"""

from __future__ import annotations

import tempfile
from functools import partial
from pathlib import Path
from typing import List, Tuple

from . import alignment, per_system, reencoding_ablation, sparsity_mechanisms, table1, table2, window_ablation
from .common import Coverage

# Written by ``python -m paper collect`` when shortened runs (``--include-short``) stood in for paper runs.
SHORTENED_RUNS = "shortened_runs.txt"


def build_tables(evidence: Path, out: Path, allow_missing_runs: bool = False) -> Tuple[List[Path], List[str]]:
    """Write every paper table from ``evidence`` into ``out``; return the written files and the problems.

    A table whose evidence files are missing is not written. A table whose evidence lacks some of the paper's
    runs, or comes from shortened runs, is written only with ``allow_missing_runs`` (a test run), with the
    cells that lack runs marked ``(missing runs)``. Each problem is one line, which is also printed. A builder
    that fails does not stop the others: its error is raised once every other table is written.
    """
    evidence, out = Path(evidence), Path(out)
    written: List[Path] = []
    problems: List[str] = []
    failures: List[Exception] = []
    shortened = (evidence / SHORTENED_RUNS).read_text().splitlines() if (evidence / SHORTENED_RUNS).is_file() else []
    # (name, builder, output path in the scratch directory)
    builders = (("Table 1", table1.build, "table1_forecasting.tex"),
                ("Table 2", table2.build, "table2_interventions.tex"),
                ("Multibasin per-system tables", partial(per_system.build, benchmark="multibasin"), ""),
                ("Dysts per-system tables", partial(per_system.build, benchmark="dysts"), ""),
                ("Alignment tables", alignment.build, ""),
                ("Sparsity-mechanisms table", sparsity_mechanisms.build, "sparsity_mechanisms.tex"),
                ("Re-encoding table", reencoding_ablation.build, "reencoding_ablation.tex"),
                ("Window-ablation table", window_ablation.build, "window_ablation.tex"))
    not_written = "" if allow_missing_runs else "; not written (use --allow-missing-runs for a test run)"
    with tempfile.TemporaryDirectory() as scratch:  # a table is copied to ``out`` only once it may be written
        for name, build, target in builders:
            coverage = Coverage(name)
            try:
                tables = build(evidence, Path(scratch) / target, coverage)
            except FileNotFoundError as missing:
                problems.append(f"{name}: not written, no {missing.filename}")
                continue
            except Exception as error:  # build the other tables first
                problems.append(f"{name}: not written, {type(error).__name__}: {error}")
                failures.append(error)
                continue
            found = [coverage.summary()] if coverage.summary() else []
            if shortened:  # the analyses of ``evidence --include-short`` read the same shortened runs
                found.append(f"{name}: built from {len(shortened)} shortened runs listed in "
                             f"{evidence / SHORTENED_RUNS}, not the paper's runs")
            problems += [problem + not_written for problem in found]
            if found and not allow_missing_runs:
                continue
            out.mkdir(parents=True, exist_ok=True)
            for table in tables if isinstance(tables, list) else [tables]:
                written.append(out / table.name)
                written[-1].write_bytes(table.read_bytes())
    for problem in problems:
        print(problem)
    if failures:
        raise RuntimeError(f"{len(failures)} table builder(s) failed; the other tables are in {out}") from failures[0]
    return written, problems
