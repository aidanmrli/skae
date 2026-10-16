"""Reproduce the paper: ``python -m paper <command>``, run from the repository root.

Commands (``python -m paper <command> --help`` for options)::

    data       build the Dysts trajectory caches (data/dysts)
    tasks      list the paper's runs, one per line: "<benchmark> <system> <model> <seed>"
    train      train one run, or one line of a task file (runs/)
    collect    finished runs -> forecasting evidence CSVs
    baselines  classical baselines -> evidence
    analyze    runs -> support, sparsity, re-encoding, intervention and error-growth evidence
    evidence   run every producer: runs -> a complete evidence directory (evidence/)
    tables     evidence -> paper tables (LaTeX)
    rebuild    verify the bundled evidence checksums, then build the tables

``tables`` and ``rebuild`` read the bundled ``paper/evidence/`` unless
``--evidence DIR`` is given, and write to ``results/``. The commands that turn
runs or evidence into outputs exit with status 1 when some of the paper's runs
are missing, unless ``--allow-missing-runs`` is given (for a test run). Usage
errors exit with status 2.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import List, NoReturn, Optional, Sequence

BUNDLED_EVIDENCE = Path(__file__).resolve().parent / "evidence"
# Commands run by a module's main(argv) -> int. The evidence producers (collect, baselines and every
# analysis) take --runs DIR --evidence DIR; baselines and the analyses also --data DIR, the analyses --device D.
MODULES = {
    "data": "paper.systems.cli",
    "collect": "paper.training.collect",
    "baselines": "paper.baselines.cli",
}
ANALYSES = {  # in dependency order
    "support-transfer": "paper.analysis.support_transfer",
    "support-sensitivity": "paper.analysis.support_sensitivity",
    "reencoding": "paper.analysis.reencoding",
    "interventions": "paper.analysis.interventions",
    "error-growth": "paper.analysis.error_growth",
    "sparsity-mechanisms": "paper.analysis.sparsity_mechanisms",
}


def run_module(module: str, argv: Sequence[str]) -> int:
    """Import ``module`` and return ``module.main(argv)``."""
    return importlib.import_module(module).main(list(argv)) or 0


def parser(command: str, description: str) -> argparse.ArgumentParser:
    """The argument parser of a command."""
    return argparse.ArgumentParser(prog=f"python -m paper {command}", description=description)


def usage_error(message: str) -> NoReturn:
    """Print a usage error and exit with status 2, as argparse does."""
    print(f"python -m paper: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def outcome(problems: List[str], allow_missing_runs: bool) -> int:
    """Exit status of a command whose problems were printed: 1 when runs are missing, unless that is allowed."""
    return 1 if problems and not allow_missing_runs else 0


# -- commands ---------------------------------------------------------------------------
def tasks(argv: Sequence[str]) -> int:
    """List the paper's runs, one task line each."""
    from paper.config import BENCHMARKS, models_for, paper_runs, systems_for  # no torch

    p = parser("tasks", "List the paper's runs, one per line: '<benchmark> <system> <model> <seed>'.")
    p.add_argument("--benchmark", choices=(*BENCHMARKS, "all"), default="all", help="default: all")
    p.add_argument("--systems", nargs="+", metavar="SYSTEM", choices=systems_for("multibasin") + systems_for("dysts"),
                   help="default: every system")
    p.add_argument("--models", nargs="+", metavar="MODEL", choices=models_for("multibasin"),
                   help="default: every model and multibasin ablation")
    p.add_argument("--seeds", nargs="+", type=int,
                   help="default: the paper's seeds (0-14; 0-4 for the 64-state-window ablation)")
    p.add_argument("--no-ablations", action="store_true", help="only the six models of Table 1")
    p.add_argument("--out", type=Path, help="task file to write (default: print the tasks)")
    args = p.parse_args(argv)

    benchmarks = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    lines = [f"{benchmark} {system} {model} {seed}\n"
             for benchmark, system, model, seed in paper_runs(benchmarks, ablations=not args.no_ablations)
             if (args.systems is None or system in args.systems)
             and (args.models is None or model in args.models)
             and (args.seeds is None or seed in args.seeds)]
    if not lines:
        p.error("no paper run matches these options")
    if args.out is None:
        sys.stdout.write("".join(lines))
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(lines))
        print(f"wrote {len(lines)} tasks to {args.out}")
    return 0


def train(argv: Sequence[str]) -> int:
    """Train and evaluate one run, or the lines of a task file."""
    p = parser("train", "Train one paper run (or one line of a task file), then score it on held-out data "
                        "(metrics.json).")
    p.add_argument("--benchmark", choices=("multibasin", "dysts"))
    p.add_argument("--system")
    p.add_argument("--model")
    p.add_argument("--seed", type=int, default=0, help="default: 0")
    p.add_argument("--tasks", type=Path, help="task file written by 'tasks' (every line unless --index is given)")
    p.add_argument("--index", type=int, help="0-based line of the task file, e.g. $SLURM_ARRAY_TASK_ID")
    p.add_argument("--runs", type=Path, default=Path("runs"), help="output root (default: runs)")
    p.add_argument("--data", type=Path, default=Path("data"), help="parent of the Dysts caches (default: data)")
    p.add_argument("--device", default="cuda", help="cuda (default; falls back to cpu with a warning), cuda:N for GPU N, or cpu")
    p.add_argument("--steps", type=int, help="shorten training (writes seed_<n>_steps<N>; default: the paper's)")
    p.add_argument("--no-eval", action="store_true", help="skip the held-out evaluation")
    p.add_argument("--print-config", action="store_true", help="print the run configuration(s) and exit")
    args = p.parse_args(argv)
    from paper.config import paper_config, run_dir
    from paper.training.train import train as train_run

    if args.tasks is not None:
        lines = [line.split() for line in args.tasks.read_text().splitlines() if line.strip()]
        if args.index is not None:
            if not 0 <= args.index < len(lines):
                p.error(f"--index {args.index} is outside the {len(lines)} lines of {args.tasks}")
            lines = [lines[args.index]]
    elif None in (args.benchmark, args.system, args.model):
        p.error("pass --tasks FILE [--index I], or --benchmark, --system and --model")
    else:
        lines = [[args.benchmark, args.system, args.model, str(args.seed)]]
    for fields in lines:
        if len(fields) != 4:
            p.error(f"a task line is '<benchmark> <system> <model> <seed>', got {' '.join(fields)!r}")
        benchmark, system, model, seed = fields
        try:
            cfg = paper_config(benchmark, system, model, int(seed))
        except ValueError as error:
            p.error(str(error))
        if args.steps is not None:
            cfg.train.steps = args.steps
        if args.print_config:
            print(json.dumps(cfg.to_dict(), indent=2))
            continue
        directory = run_dir(args.runs, cfg)
        print(f"[skae] {cfg.run_name} -> {directory}", flush=True)
        train_run(cfg, directory, device=args.device, data_dir=args.data, evaluate=not args.no_eval)
    return 0


def analyze(argv: Sequence[str]) -> int:
    """``analyze <name> [options]``: one analysis producer. Its help comes from the module's docstring without
    importing it, since importing loads torch."""
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: python -m paper analyze {{{','.join(ANALYSES)}}} [--help] [options]")
        return 0 if argv else 2
    if argv[0] not in ANALYSES:
        usage_error(f"unknown analysis {argv[0]!r}; choose from {', '.join(ANALYSES)}")
    name, rest = argv[0], list(argv[1:])
    if "-h" in rest or "--help" in rest:
        from paper.analysis import producer_parser

        source = Path(__file__).parent.joinpath(*ANALYSES[name].split(".")[1:]).with_suffix(".py").read_text()
        producer_parser(name, ast.get_docstring(ast.parse(source))).parse_args(["--help"])
    return run_module(ANALYSES[name], rest)


def trained_roster(runs: Path, include_short: bool) -> dict:
    """``{benchmark: (systems, seeds)}`` of the trained runs under ``runs`` (with shortened ones if asked)."""
    from paper.config import BENCHMARKS

    roster = {}
    for benchmark in BENCHMARKS:
        found = [final.parent for final in Path(runs).glob(f"{benchmark}/*/*/seed_*/final.pt")
                 if include_short or "_steps" not in final.parent.name]
        if found:
            roster[benchmark] = (sorted({run.parent.parent.name for run in found}),
                                 sorted({int(run.name.split("_")[1]) for run in found}))
    return roster


def evidence(argv: Sequence[str]) -> int:
    """Run every evidence producer into one new evidence directory, then write its SHA256SUMS.

    The order is ``collect``, ``baselines`` (only for the systems and seeds that have trained runs), then the
    analyses in dependency order (``ANALYSES``). Every producer accepts missing runs; this command then checks
    the paper's roster once.
    """
    p = parser("evidence", "Run every evidence producer (collect, baselines, each analysis): trained runs -> an "
                           "evidence directory with the layout of paper/evidence.")
    p.add_argument("--runs", type=Path, default=Path("runs"), help="trained runs (default: runs)")
    p.add_argument("--data", type=Path, default=Path("data"), help="parent of the Dysts caches (default: data)")
    p.add_argument("--evidence", type=Path, default=Path("evidence"), help="output directory (default: evidence)")
    p.add_argument("--device", default="cpu", help="torch device of the analyses (default: cpu)")
    p.add_argument("--workers", type=int, default=1,
                   help="classical-baseline tasks in parallel, one CPU process each (default: 1)")
    p.add_argument("--include-short", action="store_true",
                   help="use shortened runs (seed_<n>_steps<N>) where full-length runs are missing (for a quick check)")
    p.add_argument("--allow-missing-runs", action="store_true",
                   help="exit 0 even when some of the paper's runs are missing (a test run)")
    p.add_argument("--force", action="store_true", help="delete a non-empty output directory first")
    args = p.parse_args(argv)
    from paper.config import find_run, paper_runs

    # SHA256SUMS certifies every file of the directory, so it must hold only what this command writes.
    if args.evidence.is_dir() and any(args.evidence.iterdir()):
        if not args.force:
            p.error(f"{args.evidence} is not empty; pass --force to delete it first")
        if args.evidence.resolve() == BUNDLED_EVIDENCE.resolve():
            p.error("refusing to delete the bundled evidence")
        shutil.rmtree(args.evidence)

    # The producers always accept missing runs here; this command checks the roster once, below.
    paths = ["--runs", str(args.runs), "--data", str(args.data), "--evidence", str(args.evidence)]
    flags = ["--include-short"] * args.include_short + ["--allow-missing-runs"]
    steps = [("collect", MODULES["collect"], ["--runs", str(args.runs), "--evidence", str(args.evidence), *flags])]
    for benchmark, (systems, seeds) in trained_roster(args.runs, args.include_short).items():
        steps.append((f"baselines {benchmark}", MODULES["baselines"],
                      [benchmark, *paths, "--workers", str(args.workers), "--systems", *systems,
                       "--seeds", *map(str, seeds)]))
    steps += [(f"analyze {name}", module, [*paths, "--device", args.device, *flags])
              for name, module in ANALYSES.items()]
    empty = []
    for name, module, step_argv in steps:
        print(f"== python -m paper {name}", flush=True)
        if run_module(module, step_argv):
            empty.append(name)
    print(f"evidence: wrote {args.evidence}/SHA256SUMS ({write_checksums(args.evidence)} files)")
    if empty:
        print(f"evidence: nothing from {', '.join(empty)} (their runs are missing)")
    runs = list(paper_runs())
    evaluated = sum((find_run(args.runs, *run, include_short=args.include_short) / "metrics.json").exists()
                    for run in runs)
    if evaluated == len(runs):
        return 0
    print(f"evidence: {len(runs) - evaluated} of the paper's {len(runs)} runs missing or not evaluated"
          + ("" if args.allow_missing_runs else " (use --allow-missing-runs for a test run)"))
    return outcome(["missing runs"], args.allow_missing_runs)


def output_parser(command: str, description: str) -> argparse.ArgumentParser:
    """The parser of the commands that read an evidence directory and write outputs."""
    p = parser(command, description)
    out = Path("results") / ("tables" if command == "tables" else "")
    p.add_argument("--evidence", type=Path, default=BUNDLED_EVIDENCE,
                   help="evidence directory (default: the bundled paper/evidence)")
    p.add_argument("--out", type=Path, default=out, help=f"output directory (default: {out})")
    p.add_argument("--allow-missing-runs", action="store_true",
                   help="write tables even when their evidence misses some of the paper's runs (a test run)")
    return p


def tables(argv: Sequence[str]) -> int:
    """Build the paper's tables from an evidence directory."""
    args = output_parser("tables", "Build the paper's tables (LaTeX) from an evidence directory.").parse_args(argv)
    from paper.tables import build_tables

    written, problems = build_tables(args.evidence, args.out, args.allow_missing_runs)
    print(f"wrote {len(written)} tables to {args.out}")
    return outcome(problems, args.allow_missing_runs)


def rebuild(argv: Sequence[str]) -> int:
    """Verify an evidence directory's checksums, then build the tables."""
    args = output_parser("rebuild", "Verify the evidence checksums (SHA256SUMS), then build the tables into "
                                    "<out>/tables.").parse_args(argv)
    print(f"verified {verify_checksums(args.evidence)} evidence files in {args.evidence}", flush=True)
    return tables(["--evidence", str(args.evidence), "--out", str(args.out / "tables")]
                  + ["--allow-missing-runs"] * args.allow_missing_runs)


def write_checksums(evidence_dir: Path) -> int:
    """Write ``<evidence>/SHA256SUMS`` for every file of an evidence directory; return the count."""
    from paper.io import atomic_path, sha256_file

    files = sorted(path for path in Path(evidence_dir).rglob("*")
                   if path.is_file() and path.name not in ("SHA256SUMS", "README.md"))
    lines = [f"{sha256_file(path)}  {path.relative_to(evidence_dir).as_posix()}\n" for path in files]
    with atomic_path(Path(evidence_dir) / "SHA256SUMS") as temporary:
        temporary.write_text("".join(lines))
    return len(lines)


def verify_checksums(evidence_dir: Path) -> int:
    """Check every file listed in ``<evidence>/SHA256SUMS``; return the number checked."""
    from paper.io import sha256_file

    sums = Path(evidence_dir) / "SHA256SUMS"
    if not sums.is_file():
        raise SystemExit(f"missing {sums}")
    entries = [line.split(maxsplit=1) for line in sums.read_text().splitlines() if line.strip()]
    failed = [name for expected, name in entries
              if not (Path(evidence_dir) / name).is_file() or sha256_file(Path(evidence_dir) / name) != expected]
    if failed:
        raise SystemExit(f"evidence checksum mismatch or missing file: {failed}")
    return len(entries)


COMMANDS = {"tasks": tasks, "train": train, "analyze": analyze, "evidence": evidence, "tables": tables,
            "rebuild": rebuild}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch ``python -m paper <command> [options]``; return the exit status."""
    argv = list(sys.argv[1:] if argv is None else argv)
    command, rest = (argv[0], argv[1:]) if argv else ("", [])
    if command in MODULES:
        return run_module(MODULES[command], rest)
    if command in COMMANDS:
        return COMMANDS[command](rest)
    if command not in ("", "-h", "--help"):
        usage_error(f"unknown command {command!r}; 'python -m paper --help' lists the commands")
    print(__doc__.strip())
    return 0 if command else 2


if __name__ == "__main__":
    sys.exit(main())
