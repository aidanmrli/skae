"""Collect the held-out forecasts of finished runs into the forecasting evidence files.

    python -m paper collect --runs runs --evidence evidence

Reads ``metrics.json`` of every paper run ``<runs>/<benchmark>/<system>/<model>/seed_<n>`` and writes

* ``forecasting_multibasin.csv``: the six models and Sparse MLP with L1 = 0 (Table 1, Figure 2a);
* ``forecasting_window64.csv``: the 64-state-window ablation;
* ``forecasting_dysts.csv``: the six models on Dysts;
* ``forecasting_sparsity_ablation.csv``: the encoders of the sparsity-mechanisms table (Dense MLP,
  Dense MLP with ReLU hidden layers, Sparse MLP with and without L1, LISTA), one row per run and horizon.

Per horizon ``H``, ``h{H}`` is the test MSE at the validation-selected re-encoding period
and ``h{H}_direct`` that of the rollout without re-encoding, each with its divergence rate.
Rows follow the sorted run directories. With ``--include-short``, the longest shortened run
(``seed_<n>_steps<N>``) stands in for a missing full-length run, as in every analysis; the
directories of the shortened runs used are then listed in ``shortened_runs.txt`` (relative to
``--runs``, one per line), which marks the evidence as built from shortened runs. Without shortened runs that
file is removed. ``collect`` also warns about runs whose ``metrics.json`` ``settings`` differ from
the current ones (:func:`paper.training.evaluation.run_settings`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from paper.io import atomic_path, write_csv

SHORTENED_RUNS = "shortened_runs.txt"  # evidence built from shortened runs lists them here
HORIZONS = {"multibasin": (100, 500, 1000), "dysts": (100, 500, 1000, 1500, 2000, 3000, 4000, 5000)}
# The sparsity-ablation encoders, then the Dense MLP with ReLU hidden layers in a block of its own.
SPARSITY_ABLATION = ("dense_mlp", "lista", "sparse_mlp", "sparse_mlp_l1_0")
RELU_CONTROL = "dense_mlp_relu"


def _finite_or_nan(value) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else float("nan")


def forecast_row(metrics: Dict[str, object], horizons: Sequence[int]) -> Dict[str, object]:
    """The evidence columns of one run's ``metrics.json``."""
    row: Dict[str, object] = {}
    for h in horizons:
        chosen, direct = metrics["selected_periodic"][str(h)], metrics["modes"]["direct"][str(h)]
        row[f"h{h}"] = _finite_or_nan(chosen["mean"])
        row[f"h{h}_divergence"] = chosen["divergence_rate"]
        row[f"h{h}_direct"] = _finite_or_nan(direct["mean"])
        row[f"h{h}_direct_divergence"] = direct["divergence_rate"]
    return row


def finished_runs(runs: Path, benchmark: str, include_short: bool) -> Tuple[List[dict], List[Path], int]:
    """The evaluated paper runs of a benchmark (``{system, model, seed, dir, metrics}``, sorted like the run
    directories), the run directories without ``metrics.json``, and the number of paper runs."""
    from paper.config import find_run, paper_runs

    found, unevaluated, expected = [], [], 0
    for _, system, model, seed in paper_runs((benchmark,)):
        expected += 1
        run_dir = find_run(runs, benchmark, system, model, seed, include_short)
        if (run_dir / "metrics.json").is_file():
            found.append({"system": system, "model": model, "seed": seed, "dir": run_dir,
                          "metrics": json.loads((run_dir / "metrics.json").read_text())})
        elif (run_dir / "final.pt").is_file():
            unevaluated.append(run_dir)
    found.sort(key=lambda run: (run["system"], run["model"], f"seed_{run['seed']}"))
    return found, unevaluated, expected


def collect(runs: Path, evidence: Path, include_short: bool = False) -> bool:
    """Write the forecasting evidence files to ``evidence``; return whether every paper run was found."""
    from paper.config import MODELS, WINDOW64_ABLATIONS
    from paper.training.evaluation import run_settings

    settings = json.loads(json.dumps(run_settings()))  # as metrics.json stores them
    files = {"forecasting_multibasin.csv": ("multibasin", MODELS + ("sparse_mlp_l1_0",)),
             "forecasting_window64.csv": ("multibasin", WINDOW64_ABLATIONS),
             "forecasting_dysts.csv": ("dysts", MODELS)}
    found, complete = {}, True
    for benchmark in HORIZONS:
        found[benchmark], unevaluated, expected = finished_runs(runs, benchmark, include_short)
        complete &= len(found[benchmark]) == expected
        missing = expected - len(found[benchmark])
        print(f"collect {benchmark}: "
              + (f"{missing} of {expected} runs missing" if missing else f"all {expected} runs")
              + (f", {len(unevaluated)} of them without metrics.json (trained with --no-eval; rerun their train "
                 f"command without it), e.g. {unevaluated[0]}" if unevaluated else ""))
        other = [run["dir"] for run in found[benchmark] if run["metrics"].get("settings", settings) != settings]
        if other:
            print(f"collect {benchmark}: warning: {len(other)} runs were evaluated with other settings than "
                  f"paper.config and paper.data now hold (metrics.json \"settings\"), e.g. {other[0]}")
    tables = {}
    for name, (benchmark, models) in files.items():
        tables[name] = [{"model": run["model"], "system": run["system"], "seed": run["seed"],
                         **forecast_row(run["metrics"], HORIZONS[benchmark])}
                        for run in found[benchmark] if run["model"] in models]
    ablation = [run for run in found["multibasin"] if run["model"] in SPARSITY_ABLATION]
    ablation += [run for run in found["multibasin"] if run["model"] == RELU_CONTROL]
    tables["forecasting_sparsity_ablation.csv"] = [
        {"benchmark": "multibasin", "system": run["system"], "model": run["model"], "seed": run["seed"],
         "horizon": h, "mse": _finite_or_nan(run["metrics"]["selected_periodic"][str(h)]["mean"]),
         "divergence_rate": run["metrics"]["selected_periodic"][str(h)]["divergence_rate"], "status": "ok"}
        for run in ablation for h in HORIZONS["multibasin"]]
    for name, rows in tables.items():
        if rows:
            write_csv(Path(evidence) / name, rows)
    print(f"collect: wrote {', '.join(name for name, rows in tables.items() if rows) or 'nothing'} to {evidence}")
    shortened = [run["dir"].relative_to(runs).as_posix() for runs_found in found.values() for run in runs_found
                 if run["dir"].name != f"seed_{run['seed']}"]
    marker = Path(evidence) / SHORTENED_RUNS
    if shortened:
        with atomic_path(marker) as temporary:
            temporary.write_text("".join(f"{directory}\n" for directory in shortened))
        print(f"collect: warning: {len(shortened)} shortened runs stand in for full-length runs (listed in {marker}); "
              "the evidence comes from shortened runs, not the paper's")
    else:
        marker.unlink(missing_ok=True)
    return complete


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m paper collect", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="trained runs (default: runs)")
    parser.add_argument("--evidence", type=Path, default=Path("evidence"), help="output directory (default: evidence)")
    parser.add_argument("--include-short", action="store_true",
                        help="use a shortened run (seed_<n>_steps<N>) where the full-length run is missing")
    parser.add_argument("--allow-missing-runs", action="store_true",
                        help="exit 0 even when some of the paper's runs are missing (a test run)")
    args = parser.parse_args(argv)
    complete = collect(args.runs, args.evidence, args.include_short)
    if not complete and not args.allow_missing_runs:
        print("collect: runs missing (use --allow-missing-runs for a test run)")
        return 1
    return 0
