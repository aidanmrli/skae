"""Classical baselines -> evidence: ``python -m paper baselines [COMMAND]``.

Commands::

    all         (default) multibasin, then dysts, then collect
    multibasin  DMD, polynomial EDMD and RBF EDMD on the 15 multibasin systems
    dysts       the five Dysts rows, with the per-system settings of --settings
    collect     <evidence>/baselines_multibasin.csv and baselines_dysts.csv from the saved results
    select      re-choose the Dysts settings on validation starts and compare them with --settings

Each (system, seed) result is saved to ``<runs>/baselines/<benchmark>/<system>/seed_<k>.json``
and skipped when present, so a sweep can be split over jobs with --systems and --seeds and
resumed; every fitting command ends with ``collect``. ``select`` writes
``<runs>/baselines/dysts_selection/<system>.json``. The Dysts commands read the caches of
``python -m paper data`` under --data. Everything runs on CPU, one process per task
(--workers in parallel). The command exits with status 1 when a task fails or is skipped,
or when there is nothing to collect.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
import traceback
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from paper.analysis import pinned_threads

# Heavy modules (torch, SciPy) are imported inside the functions, so that --help is fast.
# Threads per task process, as in the paper's runs: the RBF EDMD fits are ill-conditioned, so
# other thread counts (or CPU types) change a few of their errors in the fourth digit.
THREADS = {"multibasin": 1, "dysts": 2, "select": 2}


def result_path(runs: Path, benchmark: str, system: str, seed: int) -> Path:
    return Path(runs) / "baselines" / benchmark / system / f"seed_{seed}.json"


def selection_path(runs: Path, system: str) -> Path:
    return Path(runs) / "baselines" / "dysts_selection" / f"{system}.json"


def run_task(task: tuple) -> Tuple[bool, str]:
    """Compute and save one (command, system, seed) result; runs in its own process.

    Returns whether it succeeded and a one-line message (with the traceback of a failure).
    """
    from paper.baselines import dysts, multibasin
    from paper.io import write_json

    command, system, seed, runs, data, settings = task
    task_name = f"{command} {system}" if seed is None else f"{command} {system} seed {seed}"
    started = time.perf_counter()
    try:
        if command == "multibasin":
            write_json(result_path(runs, command, system, seed), multibasin.evaluate(system, seed), indent=1)
        elif command == "dysts":
            result = dysts.evaluate(system, seed, data, dysts.load_settings(settings)[system])
            write_json(result_path(runs, command, system, seed), result, indent=1)
        else:
            write_json(selection_path(runs, system), dysts.select(system, data), indent=1)
    except Exception:  # reported by the parent, which runs the remaining tasks
        return False, f"{task_name}: FAILED\n{traceback.format_exc()}"
    return True, f"{task_name}: {time.perf_counter() - started:.0f} s"


def collect(runs: Path, evidence: Path) -> bool:
    """Write the evidence CSVs from every saved result, in the bundled row order; return whether any was written."""
    from paper.baselines import dysts, multibasin
    from paper.config import SEEDS, systems_for
    from paper.io import write_csv

    written = False
    for name, module in (("multibasin", multibasin), ("dysts", dysts)):
        paths = sorted((Path(runs) / "baselines" / name).glob("*/seed_*.json"))
        rows = [row for path in paths for row in json.loads(path.read_text())["rows"]]
        expected = len(systems_for(name)) * len(SEEDS)
        if not rows:
            print(f"{name}: no results under {Path(runs) / 'baselines' / name}")
            continue
        out = Path(evidence) / f"baselines_{name}.csv"
        write_csv(out, [{field: row[field] for field in module.FIELDS} for row in sorted(rows, key=module.sort_key)])
        print(f"{name}: {len(paths)} of {expected} (system, seed) results -> {out}")
        written = True
    return written


def compare_selection(runs: Path, systems: Sequence[str], settings: Path) -> None:
    """Print whether each re-selected choice equals the one in ``settings``."""
    from paper.baselines import dysts

    frozen = dysts.load_settings(settings)
    for system in systems:
        if not selection_path(runs, system).is_file():
            print(f"select {system}: no selection")
            continue
        chosen = json.loads(selection_path(runs, system).read_text())["choices"]
        for label, choice in chosen.items():
            same = {k: v for k, v in choice.items() if k != "validation_mse"} == \
                {k: v for k, v in frozen[system][label].items() if k != "validation_mse"}
            print(f"select {system} {label}: {'same as' if same else 'DIFFERS from'} --settings "
                  f"({choice} vs {frozen[system][label]})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m paper baselines", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all",
                        choices=("all", "multibasin", "dysts", "collect", "select"))
    parser.add_argument("--runs", type=Path, default=Path("runs"),
                        help="results go to <runs>/baselines (default: runs)")
    parser.add_argument("--data", type=Path, default=Path("data"), help="parent of the Dysts caches (default: data)")
    parser.add_argument("--evidence", type=Path, default=Path("evidence"), help="output directory (default: evidence)")
    parser.add_argument("--systems", nargs="+", metavar="SYSTEM", help="default: every system of each benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, help="default: 0-14")
    parser.add_argument("--workers", type=int, default=1, help="tasks in parallel (default: 1)")
    parser.add_argument("--settings", type=Path, default=Path(__file__).with_name("dysts_settings.json"),
                        help="Dysts per-system settings (default: the frozen paper/baselines/dysts_settings.json)")
    args = parser.parse_args(argv)
    from paper.config import SEEDS, systems_for
    from paper.systems.dysts import SPLIT_SEEDS, cache_path

    args.seeds = list(SEEDS) if args.seeds is None else args.seeds
    every = systems_for("multibasin") + systems_for("dysts")
    unknown = sorted(set(args.systems or ()) - set(every))
    if unknown:
        parser.error(f"unknown systems {unknown}")

    def systems(benchmark: str) -> List[str]:
        return [system for system in systems_for(benchmark) if not args.systems or system in args.systems]

    commands = {"all": ("multibasin", "dysts"), "collect": ()}.get(args.command, (args.command,))
    skipped = failed = 0
    for command in commands:
        tasks = []
        for system in systems("dysts" if command == "select" else command):
            missing = [split for split in SPLIT_SEEDS if not cache_path(args.data, system, split).exists()] \
                if command != "multibasin" else []
            if missing:
                print(f"{command} {system}: skipped (no {'/'.join(missing)} cache under {args.data}; "
                      f"see python -m paper data)")
                skipped += 1
                continue
            for seed in (None,) if command == "select" else args.seeds:
                done = selection_path(args.runs, system) if command == "select" else \
                    result_path(args.runs, command, system, seed)
                if not done.exists():
                    tasks.append((command, system, seed, args.runs, args.data, args.settings))
        if tasks:
            print(f"{command}: {len(tasks)} tasks to run", flush=True)
            # Task processes are spawned, so they start with these thread counts.
            with pinned_threads(THREADS[command]), \
                    multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
                for succeeded, message in pool.imap_unordered(run_task, tasks):
                    print(message, flush=True)
                    failed += not succeeded
    collected = True
    if args.command == "select":
        compare_selection(args.runs, systems("dysts"), args.settings)
    else:
        collected = collect(args.runs, args.evidence)
    if failed or skipped or not collected:
        print(f"baselines: {failed} tasks failed, {skipped} systems skipped"
              + ("" if collected else ", nothing collected"))
        return 1
    return 0
