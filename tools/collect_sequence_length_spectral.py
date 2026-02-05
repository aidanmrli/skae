#!/usr/bin/env python
"""Collect sequence-length spectral-stability sweep results.

Aggregates:
  - max Koopman spectral radius from eigenvalue analysis
  - H1000 no-reencode and best periodic MSE from checkpoint evaluation

Expected experiment naming:
  dim{dim}_nb{nb}_L{L}_ts{ts}_{kstruct}
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


EXP_NAME_RE = re.compile(
    r"dim(?P<dim>\d+)_nb(?P<nb>\d+)_L(?P<L>\d+)_ts(?P<ts>\d+)_(?P<kstruct>.+)"
)

DEFAULT_BASE_DIR = "/network/scratch/l/lia/skae/sequence_length_spectral_sweep"


def _latest_timestamp_dir(exp_dir: Path) -> Optional[Path]:
    candidates = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name[:1].isdigit()],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_sr(eigen_data: Optional[Dict[str, Any]]) -> Optional[float]:
    if not eigen_data:
        return None
    blocks = eigen_data.get("blocks", [])
    if not blocks:
        return None
    return max(float(b.get("spectral_radius", 0.0)) for b in blocks)


def _extract_h1000(eval_data: Optional[Dict[str, Any]], system: str) -> Dict[str, Optional[float]]:
    if not eval_data:
        return {"no_reencode": None, "best_periodic": None}

    sys_data = eval_data.get(system, {})
    modes = sys_data.get("modes", {})

    no_re = (
        modes.get("no_reencode", {})
        .get("horizons", {})
        .get("1000", {})
        .get("mean")
    )
    best_periodic = (
        sys_data.get("best_periodic", {})
        .get("1000", {})
        .get("mean")
    )
    return {
        "no_reencode": float(no_re) if no_re is not None else None,
        "best_periodic": float(best_periodic) if best_periodic is not None else None,
    }


def collect(base_dir: Path, system: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not base_dir.exists():
        return rows

    for exp_dir in sorted(base_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        m = EXP_NAME_RE.match(exp_dir.name)
        if not m:
            continue

        run_dir = _latest_timestamp_dir(exp_dir)
        if run_dir is None:
            continue

        eigen_data = _safe_load_json(run_dir / "eigenvalue_analysis" / "eigenvalue_analysis.json")
        eval_data = _safe_load_json(run_dir / "evaluation_results_checkpoint.json")

        h1000 = _extract_h1000(eval_data, system=system)
        row = {
            "experiment": exp_dir.name,
            "run_dir": str(run_dir),
            "dim": int(m.group("dim")),
            "num_basins": int(m.group("nb")),
            "sequence_length": int(m.group("L")),
            "target_size": int(m.group("ts")),
            "k_structure": m.group("kstruct"),
            "max_spectral_radius": _extract_sr(eigen_data),
            "h1000_no_reencode": h1000["no_reencode"],
            "h1000_best_periodic": h1000["best_periodic"],
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["k_structure"], r["target_size"], r["sequence_length"]))
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No results found.")
        return

    header = (
        f"{'K':<16} {'TS':>5} {'L':>4} {'MaxSR':>8} "
        f"{'H1000 no-re':>13} {'H1000 best-PR':>14}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        sr = "N/A" if r["max_spectral_radius"] is None else f"{r['max_spectral_radius']:.4f}"
        h_nr = "N/A" if r["h1000_no_reencode"] is None else f"{r['h1000_no_reencode']:.3e}"
        h_bp = "N/A" if r["h1000_best_periodic"] is None else f"{r['h1000_best_periodic']:.3e}"
        print(
            f"{r['k_structure']:<16} {r['target_size']:>5} {r['sequence_length']:>4} "
            f"{sr:>8} {h_nr:>13} {h_bp:>14}"
        )


def main():
    parser = argparse.ArgumentParser(description="Collect sequence-length spectral sweep results")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR)
    parser.add_argument("--system", type=str, default="lyapunov")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <base_dir>/sequence_length_spectral_summary.json)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rows = collect(base_dir=base_dir, system=args.system)
    print_table(rows)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = base_dir / "sequence_length_spectral_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
