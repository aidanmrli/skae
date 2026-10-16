"""Evidence producers that analyse trained runs (``python -m paper analyze <name>``).

Every producer module has ``main(argv) -> int``: it parses :func:`producer_parser`'s
flags, reads the runs it needs under ``--runs`` (with ``paper.config.find_run``),
writes one evidence file into ``--evidence`` with the bundled name and schema
(nothing when none of its runs exists) and returns :func:`finish`'s exit status.
Every producer, and ``python -m paper baselines``, runs inside
:func:`pinned_threads`.

This module stays free of heavy imports, so that ``--help`` of every producer is fast.
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def producer_parser(name: str, description: str) -> argparse.ArgumentParser:
    """The standard producer flags: ``--runs``, ``--data``, ``--evidence``, ``--device``, ``--include-short``
    and ``--allow-missing-runs``."""
    parser = argparse.ArgumentParser(prog=f"python -m paper analyze {name}", description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="trained runs (default: runs)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Dysts caches (default: data)")
    parser.add_argument("--evidence", type=Path, default=Path("evidence"),
                        help="evidence directory to write (default: evidence)")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--include-short", action="store_true",
                        help="use a shortened run (seed_<n>_steps<N>) where the full-length run is missing")
    parser.add_argument("--allow-missing-runs", action="store_true",
                        help="exit 0 even when some of the paper's runs are missing (a test run)")
    return parser


def finish(name: str, used: int, expected: int, allow_missing_runs: bool, output: Path) -> int:
    """Print a producer's one-line summary; return 1 when runs are missing, unless that is allowed."""
    if not used:
        needed = "the run it needs is" if expected == 1 else f"all {expected} runs it needs are"
        print(f"{name}: {needed} missing; nothing written")
        return 1
    if used == expected:
        print(f"{name}: {used} run{'s' * (used != 1)} -> {output}")
        return 0
    print(f"{name}: {expected - used} of {expected} runs missing -> {output}"
          + ("" if allow_missing_runs else " (use --allow-missing-runs for a test run)"))
    return 0 if allow_missing_runs else 1


@contextmanager
def pinned_threads(count: int = 1) -> Iterator[None]:
    """Run with ``count`` threads, then restore the previous settings.

    Torch in this process uses ``count`` threads, and so do the processes started meanwhile (the classical
    baselines' task processes), which read the OpenMP/BLAS variables. A producer then gives the same numbers
    on any core count, alone and inside ``python -m paper evidence``.
    """
    import torch

    previous_threads = torch.get_num_threads()
    previous_variables = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    torch.set_num_threads(count)
    os.environ.update(dict.fromkeys(THREAD_VARIABLES, str(count)))
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
        for name, value in previous_variables.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
