#!/usr/bin/env python
"""Collect and summarize LISTA final-op experiment outputs.

This script scans phase directories produced by:
  - scripts/sweep_lista_final_op_phase0_smoke.sh
  - scripts/sweep_lista_final_op_phase1_core.sh
  - scripts/sweep_lista_final_op_phase2_sparsity_match.sh
  - scripts/sweep_lista_final_op_phase3_structured_transfer.sh

It aggregates:
  - Cosine separation metrics (support_eval/cosine_metrics.json)
  - Thresholded support metrics at tau=1e-3 (support_eval/threshold_sweep.json)
  - Horizon-1000 prediction metrics (evaluation_results_checkpoint.json)
  - Max spectral radius (eigenvalue_analysis/eigenvalue_analysis.json)

It also computes paired relu-vs-shrink deltas for matched seeds/configs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PHASE_DIRS = (
    "phase0_smoke",
    "phase1_core",
    "phase2_sparsity_match",
    "phase3_structured_transfer",
)


@dataclass
class RunMeta:
    phase: int
    phase_dir: str
    system: str
    k_structure: str
    target_size: int
    final_op: str
    sparsity_coeff: float
    seed: int
    exp_dir: Path
    run_dir: Path


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _find_latest_run_dir(exp_dir: Path) -> Optional[Path]:
    candidates = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name[:1].isdigit()],
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _parse_meta_from_name(exp_dir_name: str, phase_dir: str, exp_dir: Path, run_dir: Path) -> Optional[RunMeta]:
    parts = exp_dir_name.split("__")
    if len(parts) < 2 or not parts[0].startswith("phase"):
        return None
    try:
        phase = int(parts[0].replace("phase", ""))
    except ValueError:
        return None

    kv: Dict[str, str] = {}
    for part in parts[1:]:
        if "-" not in part:
            continue
        key, val = part.split("-", 1)
        kv[key] = val

    required = ("sys", "k", "ts", "op", "sp", "seed")
    if not all(k in kv for k in required):
        return None

    try:
        system = kv["sys"]
        k_structure = kv["k"].replace("-", "_")
        target_size = int(kv["ts"])
        final_op = kv["op"]
        sparsity_coeff = float(kv["sp"].replace("p", "."))
        seed = int(kv["seed"])
    except ValueError:
        return None

    return RunMeta(
        phase=phase,
        phase_dir=phase_dir,
        system=system,
        k_structure=k_structure,
        target_size=target_size,
        final_op=final_op,
        sparsity_coeff=sparsity_coeff,
        seed=seed,
        exp_dir=exp_dir,
        run_dir=run_dir,
    )


def _extract_tau_1e3(threshold_sweep: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not threshold_sweep:
        return {}

    best = None
    best_diff = None
    for row in threshold_sweep:
        t = row.get("support_threshold")
        if t is None:
            continue
        diff = abs(float(t) - 1e-3)
        if best is None or diff < (best_diff or 1e9):
            best = row
            best_diff = diff

    if best is None:
        return {}

    return {
        "tau_support_threshold": best.get("support_threshold"),
        "tau_unique_mode_supports": best.get("unique_mode_supports"),
        "tau_num_basins": best.get("num_basins"),
        "tau_mode_uniqueness_rate": best.get("mode_uniqueness_rate"),
        "tau_mean_basin_consistency": best.get("mean_basin_consistency"),
        "tau_mean_pairwise_jaccard": best.get("mean_pairwise_jaccard"),
        "tau_mean_mode_support_size": best.get("mean_mode_support_size"),
    }


def _extract_eval_metrics(eval_data: Optional[Dict[str, Any]], system: str) -> Dict[str, Any]:
    if not eval_data:
        return {}
    sys_data = eval_data.get(system, {})
    if not sys_data:
        return {}

    out: Dict[str, Any] = {}

    def _get_horizon(mode: str, horizon: int) -> Optional[Dict[str, Any]]:
        return (
            sys_data.get("modes", {})
            .get(mode, {})
            .get("horizons", {})
            .get(str(horizon))
        )

    no_re_1000 = _get_horizon("no_reencode", 1000)
    every_1000 = _get_horizon("every_step", 1000)
    best_1000 = sys_data.get("best_periodic", {}).get("1000")

    if no_re_1000:
        out["h1000_no_reencode_mean"] = no_re_1000.get("mean")
        out["h1000_no_reencode_std"] = no_re_1000.get("std")
    if every_1000:
        out["h1000_every_step_mean"] = every_1000.get("mean")
        out["h1000_every_step_std"] = every_1000.get("std")
    if best_1000:
        out["h1000_best_periodic_mean"] = best_1000.get("mean")
        out["h1000_best_periodic_mode"] = best_1000.get("mode")

    return out


def _extract_spectral_metrics(eigen_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not eigen_data:
        return {}
    blocks = eigen_data.get("blocks", [])
    if not blocks:
        return {}
    max_sr = max(float(b.get("spectral_radius", 0.0)) for b in blocks)
    return {
        "max_spectral_radius": max_sr,
        "is_unstable_sr_gt_1": max_sr > 1.0,
    }


def collect_rows(base_dir: Path, phase_dirs: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for phase_dir_name in phase_dirs:
        phase_dir = base_dir / phase_dir_name
        if not phase_dir.exists():
            continue
        for exp_dir in sorted([p for p in phase_dir.iterdir() if p.is_dir()]):
            run_dir = _find_latest_run_dir(exp_dir)
            if run_dir is None:
                continue

            meta = _parse_meta_from_name(exp_dir.name, phase_dir_name, exp_dir, run_dir)
            if meta is None:
                continue

            cosine = _load_json(run_dir / "support_eval" / "cosine_metrics.json")
            threshold_sweep = _load_json(run_dir / "support_eval" / "threshold_sweep.json")
            eval_data = _load_json(run_dir / "evaluation_results_checkpoint.json")
            eigen_data = _load_json(run_dir / "eigenvalue_analysis" / "eigenvalue_analysis.json")

            row: Dict[str, Any] = {
                "phase": meta.phase,
                "phase_dir": meta.phase_dir,
                "system": meta.system,
                "k_structure": meta.k_structure,
                "target_size": meta.target_size,
                "final_op": meta.final_op,
                "sparsity_coeff": meta.sparsity_coeff,
                "seed": meta.seed,
                "exp_dir": str(meta.exp_dir),
                "run_dir": str(meta.run_dir),
            }

            if cosine:
                row.update(
                    {
                        "cosine_sep": cosine.get("cosine_separation_score"),
                        "cosine_intra": cosine.get("mean_intra_basin_cosine"),
                        "cosine_inter": cosine.get("mean_inter_basin_cosine"),
                    }
                )

            sweep_rows = threshold_sweep if isinstance(threshold_sweep, list) else None
            row.update(_extract_tau_1e3(sweep_rows))
            row.update(_extract_eval_metrics(eval_data, meta.system))
            row.update(_extract_spectral_metrics(eigen_data))
            rows.append(row)

    return rows


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_by_arm(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (
            r["phase"],
            r["phase_dir"],
            r["system"],
            r["k_structure"],
            r["target_size"],
            r["sparsity_coeff"],
            r["final_op"],
        )
        grouped[key].append(r)

    summary: List[Dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        phase, phase_dir, system, k_structure, target_size, sparsity_coeff, final_op = key

        def gather(field: str) -> List[float]:
            vals = [x.get(field) for x in items if x.get(field) is not None]
            return [float(v) for v in vals]

        unstable = [bool(x.get("is_unstable_sr_gt_1", False)) for x in items if x.get("is_unstable_sr_gt_1") is not None]
        summary.append(
            {
                "phase": phase,
                "phase_dir": phase_dir,
                "system": system,
                "k_structure": k_structure,
                "target_size": target_size,
                "sparsity_coeff": sparsity_coeff,
                "final_op": final_op,
                "n_runs": len(items),
                "mean_cosine_sep": _mean(gather("cosine_sep")),
                "mean_h1000_no_reencode": _mean(gather("h1000_no_reencode_mean")),
                "mean_h1000_best_periodic": _mean(gather("h1000_best_periodic_mean")),
                "mean_max_spectral_radius": _mean(gather("max_spectral_radius")),
                "unstable_rate_sr_gt_1": (_mean([1.0 if x else 0.0 for x in unstable]) if unstable else None),
                "mean_tau_mode_uniqueness_rate": _mean(gather("tau_mode_uniqueness_rate")),
                "mean_tau_mean_basin_consistency": _mean(gather("tau_mean_basin_consistency")),
                "mean_tau_mean_mode_support_size": _mean(gather("tau_mean_mode_support_size")),
            }
        )
    return summary


def summarize_paired_deltas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    paired: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        key = (
            r["phase"],
            r["phase_dir"],
            r["system"],
            r["k_structure"],
            r["target_size"],
            r["sparsity_coeff"],
            r["seed"],
        )
        paired[key][r["final_op"]] = r

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for key, ops in paired.items():
        if "relu" not in ops or "shrink" not in ops:
            continue
        phase, phase_dir, system, k_structure, target_size, sparsity_coeff, _seed = key
        grouped[(phase, phase_dir, system, k_structure, target_size, sparsity_coeff)].append(
            {
                "delta_cosine_sep": _delta(ops["relu"].get("cosine_sep"), ops["shrink"].get("cosine_sep")),
                "delta_h1000_no_reencode": _delta(
                    ops["relu"].get("h1000_no_reencode_mean"),
                    ops["shrink"].get("h1000_no_reencode_mean"),
                ),
                "delta_h1000_best_periodic": _delta(
                    ops["relu"].get("h1000_best_periodic_mean"),
                    ops["shrink"].get("h1000_best_periodic_mean"),
                ),
                "delta_max_spectral_radius": _delta(
                    ops["relu"].get("max_spectral_radius"),
                    ops["shrink"].get("max_spectral_radius"),
                ),
                "delta_unstable_rate": _delta(
                    1.0 if ops["relu"].get("is_unstable_sr_gt_1") else 0.0,
                    1.0 if ops["shrink"].get("is_unstable_sr_gt_1") else 0.0,
                ),
            }
        )

    summary: List[Dict[str, Any]] = []
    for key, diffs in sorted(grouped.items()):
        phase, phase_dir, system, k_structure, target_size, sparsity_coeff = key

        def gather(field: str) -> List[float]:
            vals = [x.get(field) for x in diffs if x.get(field) is not None]
            return [float(v) for v in vals]

        summary.append(
            {
                "phase": phase,
                "phase_dir": phase_dir,
                "system": system,
                "k_structure": k_structure,
                "target_size": target_size,
                "sparsity_coeff": sparsity_coeff,
                "n_pairs": len(diffs),
                "mean_delta_cosine_sep_relu_minus_shrink": _mean(gather("delta_cosine_sep")),
                "mean_delta_h1000_no_reencode_relu_minus_shrink": _mean(gather("delta_h1000_no_reencode")),
                "mean_delta_h1000_best_periodic_relu_minus_shrink": _mean(gather("delta_h1000_best_periodic")),
                "mean_delta_max_spectral_radius_relu_minus_shrink": _mean(gather("delta_max_spectral_radius")),
                "mean_delta_unstable_rate_relu_minus_shrink": _mean(gather("delta_unstable_rate")),
            }
        )

    return summary


def _delta(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        return float(a) - float(b)
    except Exception:
        return None


def render_markdown(summary_by_arm: List[Dict[str, Any]], summary_paired: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# LISTA Final-Op Experiment Summary")
    lines.append("")
    lines.append("## By Arm")
    lines.append("")
    lines.append("| phase | system | k | ts | sparsity | op | n | cossep | H1000 no-re | max SR | unstable |")
    lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in summary_by_arm:
        lines.append(
            f"| {row['phase']} | {row['system']} | {row['k_structure']} | {row['target_size']} | "
            f"{row['sparsity_coeff']:.3f} | {row['final_op']} | {row['n_runs']} | "
            f"{_fmt(row.get('mean_cosine_sep'))} | {_fmt(row.get('mean_h1000_no_reencode'))} | "
            f"{_fmt(row.get('mean_max_spectral_radius'))} | {_fmt(row.get('unstable_rate_sr_gt_1'))} |"
        )
    lines.append("")
    lines.append("## Paired ReLU - Shrink Deltas")
    lines.append("")
    lines.append("| phase | system | k | ts | sparsity | n pairs | d CosSep | d H1000 no-re | d max SR | d unstable |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_paired:
        lines.append(
            f"| {row['phase']} | {row['system']} | {row['k_structure']} | {row['target_size']} | "
            f"{row['sparsity_coeff']:.3f} | {row['n_pairs']} | "
            f"{_fmt(row.get('mean_delta_cosine_sep_relu_minus_shrink'))} | "
            f"{_fmt(row.get('mean_delta_h1000_no_reencode_relu_minus_shrink'))} | "
            f"{_fmt(row.get('mean_delta_max_spectral_radius_relu_minus_shrink'))} | "
            f"{_fmt(row.get('mean_delta_unstable_rate_relu_minus_shrink'))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    if abs(value) >= 1e3 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LISTA final-op experiment results")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/network/scratch/l/lia/skae/lista_final_op_experiment",
        help="Base directory containing phase subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <base_dir>/results)",
    )
    parser.add_argument(
        "--phase_dirs",
        nargs="*",
        default=list(PHASE_DIRS),
        help="Phase subdirectories to scan",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (base_dir / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(base_dir, args.phase_dirs)
    summary_by_arm = summarize_by_arm(rows)
    summary_paired = summarize_paired_deltas(rows)
    md = render_markdown(summary_by_arm, summary_paired)

    raw_path = output_dir / "lista_final_op_rows.json"
    by_arm_path = output_dir / "lista_final_op_summary_by_arm.json"
    paired_path = output_dir / "lista_final_op_summary_paired.json"
    md_path = output_dir / "lista_final_op_summary.md"

    with open(raw_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(by_arm_path, "w") as f:
        json.dump(summary_by_arm, f, indent=2)
    with open(paired_path, "w") as f:
        json.dump(summary_paired, f, indent=2)
    with open(md_path, "w") as f:
        f.write(md + "\n")

    print(f"Collected runs: {len(rows)}")
    print(f"Saved raw rows: {raw_path}")
    print(f"Saved arm summary: {by_arm_path}")
    print(f"Saved paired summary: {paired_path}")
    print(f"Saved markdown summary: {md_path}")


if __name__ == "__main__":
    main()
