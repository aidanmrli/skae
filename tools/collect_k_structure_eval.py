#!/usr/bin/env python
"""Collect and summarize evaluation results from K-structure sweep.

Walks the sweep directory tree and aggregates:
  - Prediction MSE at key horizons for each rollout mode
  - Best periodic reencoding period per horizon
  - Per-block spectral radius
  - Basin-block correlation strength

Usage:
    python tools/collect_k_structure_eval.py
    python tools/collect_k_structure_eval.py --base_dir /path/to/sweep --output results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

BASE_DIR = "/network/scratch/l/lia/skae/lyapunov_k_structure_sweep"

# Experiment naming: dim8_nb13_ts{TS}_{K_STRUCT}
EXP_NAME_RE = re.compile(
    r"dim(?P<dim>\d+)_nb(?P<nb>\d+)_ts(?P<ts>\d+)_(?P<kstruct>.+)"
)

KEY_HORIZONS = [100, 500, 1000]
ROLLOUT_MODES = ["no_reencode", "every_step", "periodic_10", "periodic_25", "periodic_50", "periodic_100"]


def _find_latest_timestamp_dir(exp_dir: Path) -> Optional[Path]:
    """Return the most recent timestamp subdirectory."""
    subdirs = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: d.name,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


def _load_eval_metrics(run_dir: Path) -> Optional[Dict]:
    """Load evaluation metrics from the checkpoint evaluation output."""
    # Look for evaluation results from evaluate_checkpoints.py
    for pattern in [
        "evaluation_checkpoint/metrics.json",
        "evaluation_checkpoint/lyapunov/../../metrics.json",  # fallback
        "evaluation_results_checkpoint.json",
    ]:
        path = run_dir / pattern
        if path.exists():
            with open(path) as f:
                return json.load(f)

    # Also check for the top-level evaluation results file
    for candidate in run_dir.glob("evaluation_*/metrics.json"):
        with open(candidate) as f:
            return json.load(f)

    # Check for evaluation_results_checkpoint.json (from evaluate_checkpoints.py)
    results_file = run_dir / "evaluation_results_checkpoint.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    return None


def _load_eigenvalue_analysis(run_dir: Path) -> Optional[Dict]:
    """Load eigenvalue analysis results."""
    path = run_dir / "eigenvalue_analysis" / "eigenvalue_analysis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _extract_prediction_metrics(eval_data: Dict, system: str = "lyapunov") -> Dict:
    """Extract prediction MSE at key horizons for each rollout mode."""
    sys_data = eval_data.get(system, {})
    modes = sys_data.get("modes", {})
    best_periodic = sys_data.get("best_periodic", {})

    result: Dict[str, Any] = {}

    for horizon in KEY_HORIZONS:
        hkey = str(horizon)
        horizon_result: Dict[str, Any] = {}

        for mode in ROLLOUT_MODES:
            mode_data = modes.get(mode, {}).get("horizons", {}).get(hkey)
            if mode_data:
                horizon_result[mode] = {
                    "mean": mode_data["mean"],
                    "std": mode_data.get("std", 0.0),
                    "num_valid": mode_data.get("num_valid", 0),
                }

        bp = best_periodic.get(hkey)
        if bp:
            horizon_result["best_periodic_mode"] = bp["mode"]
            horizon_result["best_periodic_mse"] = bp["mean"]

        result[hkey] = horizon_result

    return result


def _extract_eigenvalue_summary(eigen_data: Dict) -> Dict:
    """Extract eigenvalue summary stats."""
    blocks = eigen_data.get("blocks", [])
    spectral_radii = [b["spectral_radius"] for b in blocks]

    summary: Dict[str, Any] = {
        "k_structure": eigen_data.get("k_structure", "unknown"),
        "num_blocks": len(blocks),
        "max_spectral_radius": max(spectral_radii) if spectral_radii else 0.0,
        "min_spectral_radius": min(spectral_radii) if spectral_radii else 0.0,
        "mean_spectral_radius": float(np.mean(spectral_radii)) if spectral_radii else 0.0,
        "std_spectral_radius": float(np.std(spectral_radii)) if len(spectral_radii) > 1 else 0.0,
        "per_block": [
            {"name": b["name"], "spectral_radius": b["spectral_radius"],
             "num_stable": b["num_stable"], "num_unstable": b["num_unstable"]}
            for b in blocks
        ],
    }

    # Basin-block correlation strength (if available)
    corr = eigen_data.get("basin_block_correlation")
    if corr:
        heatmap = np.array(corr["heatmap"])
        if heatmap.size > 0:
            # Row-normalize, then measure how peaked the distribution is per basin
            row_sums = heatmap.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            normed = heatmap / row_sums
            # Entropy per basin row (lower = more concentrated)
            eps = 1e-10
            entropies = -np.sum(normed * np.log(normed + eps), axis=1)
            max_entropy = np.log(heatmap.shape[1] + eps)
            summary["basin_block_mean_entropy"] = float(np.mean(entropies))
            summary["basin_block_max_entropy"] = float(max_entropy)
            summary["basin_block_concentration"] = float(1.0 - np.mean(entropies) / max_entropy) if max_entropy > 0 else 0.0

    return summary


def collect_all(base_dir: str) -> List[Dict]:
    """Walk the sweep directory and aggregate results."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory not found: {base_dir}")
        return []

    results = []

    for exp_dir in sorted(base_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        match = EXP_NAME_RE.match(exp_dir.name)
        if not match:
            continue

        dim = int(match.group("dim"))
        nb = int(match.group("nb"))
        ts = int(match.group("ts"))
        kstruct = match.group("kstruct")

        run_dir = _find_latest_timestamp_dir(exp_dir)
        if run_dir is None:
            print(f"  {exp_dir.name}: no timestamp dir found, skipping")
            continue

        entry: Dict[str, Any] = {
            "experiment": exp_dir.name,
            "dim": dim,
            "num_basins": nb,
            "target_size": ts,
            "k_structure": kstruct,
            "run_dir": str(run_dir),
        }

        # Load prediction metrics
        eval_data = _load_eval_metrics(run_dir)
        if eval_data:
            entry["prediction"] = _extract_prediction_metrics(eval_data)
        else:
            print(f"  {exp_dir.name}: no evaluation metrics found")

        # Load eigenvalue analysis
        eigen_data = _load_eigenvalue_analysis(run_dir)
        if eigen_data:
            entry["eigenvalues"] = _extract_eigenvalue_summary(eigen_data)
        else:
            print(f"  {exp_dir.name}: no eigenvalue analysis found")

        results.append(entry)

    return results


def print_summary_table(results: List[Dict]) -> None:
    """Print a summary table to stdout."""
    if not results:
        print("No results to display.")
        return

    # Sort by k_structure then target_size
    results.sort(key=lambda r: (r["k_structure"], r["target_size"]))

    # Header
    header = (
        f"{'K Structure':<20} {'TS':>5} "
        f"{'NoRE@100':>10} {'NoRE@500':>10} {'NoRE@1000':>10} "
        f"{'EvStep@100':>11} {'EvStep@1000':>12} "
        f"{'BestPR@1000':>12} "
        f"{'SpecRad':>8} {'#Blocks':>7}"
    )
    print("\n" + "=" * len(header))
    print("K-STRUCTURE EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        pred = r.get("prediction", {})
        eigen = r.get("eigenvalues", {})

        def _get_mse(horizon: int, mode: str) -> str:
            h_data = pred.get(str(horizon), {}).get(mode, {})
            val = h_data.get("mean")
            if val is None:
                return "N/A"
            return f"{val:.4e}"

        bp = pred.get("1000", {})
        bp_str = "N/A"
        if "best_periodic_mse" in bp:
            bp_str = f"{bp['best_periodic_mse']:.4e}"

        sr = eigen.get("max_spectral_radius", 0.0)
        nb = eigen.get("num_blocks", 0)

        print(
            f"{r['k_structure']:<20} {r['target_size']:>5} "
            f"{_get_mse(100, 'no_reencode'):>10} "
            f"{_get_mse(500, 'no_reencode'):>10} "
            f"{_get_mse(1000, 'no_reencode'):>10} "
            f"{_get_mse(100, 'every_step'):>11} "
            f"{_get_mse(1000, 'every_step'):>12} "
            f"{bp_str:>12} "
            f"{sr:>8.4f} {nb:>7}"
        )

    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Collect K-structure evaluation results")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR,
                        help="Base directory of K-structure sweep")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: base_dir/k_structure_comparison.json)")
    args = parser.parse_args()

    results = collect_all(args.base_dir)

    if not results:
        print("No results collected.")
        return

    print_summary_table(results)

    output_path = Path(args.output) if args.output else Path(args.base_dir) / "k_structure_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
