#!/usr/bin/env python3
"""Collect long-horizon forecasting metrics from mixed-system run roots.

This generalizes dysts-only collection to support any system key present in
`evaluation_results_best.json`, including built-in environments (e.g. multiwell).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            out = float(raw)
        except ValueError:
            return None
        if math.isfinite(out):
            return out
    return None


def _read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _horizon_mean(system_data: Dict, mode_name: str, horizon: int) -> Optional[float]:
    mean = (
        system_data.get("modes", {})
        .get(mode_name, {})
        .get("horizons", {})
        .get(str(horizon), {})
        .get("mean")
    )
    return _safe_float(mean)


def _is_run_dir(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "evaluation_results_best.json").exists()


def _discover_run_dirs(system_dir: Path) -> List[Tuple[Optional[str], Path]]:
    """Discover run directories and associated seed labels.

    Supports layouts:
    - system/run_id
    - system/seed_x/run_id
    - system/<nested>/run_id
    """
    run_dirs: List[Tuple[Optional[str], Path]] = []
    for child in sorted([p for p in system_dir.iterdir() if p.is_dir()]):
        if _is_run_dir(child):
            run_dirs.append((None, child))
            continue

        nested = sorted([p for p in child.iterdir() if p.is_dir() and _is_run_dir(p)])
        if child.name.startswith("seed_"):
            run_dirs.extend([(child.name, p) for p in nested])
        else:
            run_dirs.extend([(None, p) for p in nested])

    dedup: Dict[str, Tuple[Optional[str], Path]] = {}
    for seed_name, run_dir in run_dirs:
        dedup[str(run_dir)] = (seed_name, run_dir)
    return sorted(dedup.values(), key=lambda t: (t[0] or "", t[1].name, str(t[1])))


def _select_run_dirs(system_dir: Path, select: str) -> List[Tuple[Optional[str], Path]]:
    run_dirs = _discover_run_dirs(system_dir)
    if not run_dirs:
        return []
    if select == "all":
        return run_dirs

    # "latest": choose latest per seed group if seed directories exist.
    grouped: Dict[str, List[Tuple[Optional[str], Path]]] = defaultdict(list)
    for seed_name, run_dir in run_dirs:
        key = seed_name if seed_name else "__root__"
        grouped[key].append((seed_name, run_dir))

    selected: List[Tuple[Optional[str], Path]] = []
    for _, items in sorted(grouped.items()):
        selected.append(sorted(items, key=lambda t: (t[1].name, str(t[1])))[-1])
    return selected


def _infer_seed_name(seed_name: Optional[str], run_dir: Path) -> Optional[str]:
    if seed_name:
        return seed_name
    parent = run_dir.parent.name
    if parent.startswith("seed_"):
        return parent
    match = re.search(r"/(seed_\d+)(?:/|$)", str(run_dir))
    if match:
        return match.group(1)
    return None


def _seed_to_int(seed_name: Optional[str]) -> Optional[int]:
    if not seed_name:
        return None
    match = re.match(r"seed_(\d+)$", seed_name)
    if not match:
        return None
    return int(match.group(1))


def _candidate_system_keys(eval_data: Dict) -> List[str]:
    keys: List[str] = []
    for key, value in eval_data.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        if "modes" in value or "best_periodic" in value:
            keys.append(key)
    return sorted(keys)


def _select_system_key(
    eval_data: Dict,
    train_env_name: Optional[str],
    system_dir_name: str,
) -> Optional[str]:
    keys = _candidate_system_keys(eval_data)
    if not keys:
        return None

    if train_env_name and train_env_name in keys:
        return train_env_name
    if system_dir_name in keys:
        return system_dir_name
    if f"dysts:{system_dir_name}" in keys:
        return f"dysts:{system_dir_name}"
    if system_dir_name.startswith("dysts_"):
        raw = system_dir_name.split("dysts_", 1)[1]
        if f"dysts:{raw}" in keys:
            return f"dysts:{raw}"
    if train_env_name:
        train_base = train_env_name.split(":", 1)[-1]
        for key in keys:
            if key.split(":", 1)[-1] == train_base:
                return key
    if len(keys) == 1:
        return keys[0]
    return keys[0]


def _extract_row(
    root_label: str,
    system_dir: Path,
    run_dir: Path,
    horizon: int,
    eval_file_name: str,
    seed_name: Optional[str],
) -> Optional[Dict[str, object]]:
    eval_path = run_dir / eval_file_name
    if not eval_path.exists():
        return None
    eval_data = _read_json(eval_path)
    if not isinstance(eval_data, dict):
        return None

    cfg_data = _read_json(run_dir / "config.json") or {}
    train_env_name = cfg_data.get("ENV", {}).get("ENV_NAME")
    system_key = _select_system_key(
        eval_data=eval_data,
        train_env_name=train_env_name if isinstance(train_env_name, str) else None,
        system_dir_name=system_dir.name,
    )
    if system_key is None:
        return None

    system_data = eval_data.get(system_key, {})
    if not isinstance(system_data, dict):
        return None

    no_re = _horizon_mean(system_data, "no_reencode", horizon)
    every_step = _horizon_mean(system_data, "every_step", horizon)
    best = system_data.get("best_periodic", {}).get(str(horizon), {})
    best_periodic = _safe_float(best.get("mean"))
    best_mode = best.get("mode")

    model_cfg = cfg_data.get("MODEL", {})
    train_cfg = cfg_data.get("TRAIN", {})
    lista_cfg = model_cfg.get("ENCODER", {}).get("LISTA", {})

    canonical_seed_name = _infer_seed_name(seed_name=seed_name, run_dir=run_dir)
    row: Dict[str, object] = {
        "root_label": root_label,
        "root_path": str(system_dir.parent),
        "system_name": system_dir.name,
        "system_key": system_key,
        "train_env_name": train_env_name,
        "seed_name": canonical_seed_name,
        "seed": _seed_to_int(canonical_seed_name),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": model_cfg.get("MODEL_NAME"),
        "target_size": model_cfg.get("TARGET_SIZE"),
        "sparsity_coeff": model_cfg.get("SPARSITY_COEFF"),
        "reconst_coeff": model_cfg.get("RECONST_COEFF"),
        "pred_coeff": model_cfg.get("PRED_COEFF"),
        "lista_alpha": lista_cfg.get("ALPHA"),
        "lista_num_loops": lista_cfg.get("NUM_LOOPS"),
        "lista_final_op": lista_cfg.get("FINAL_OP"),
        "use_sequence_loss": train_cfg.get("USE_SEQUENCE_LOSS"),
        "num_steps": train_cfg.get("NUM_STEPS"),
        f"h{horizon}_no_reencode_mean": no_re,
        f"h{horizon}_every_step_mean": every_step,
        f"h{horizon}_best_periodic_mean": best_periodic,
        f"h{horizon}_best_periodic_mode": best_mode,
    }

    if best_periodic is not None and no_re is not None and no_re > 0.0:
        row[f"h{horizon}_ratio_best_over_no_re"] = best_periodic / no_re
    else:
        row[f"h{horizon}_ratio_best_over_no_re"] = None
    if best_periodic is not None and every_step is not None and every_step > 0.0:
        row[f"h{horizon}_ratio_best_over_every_step"] = best_periodic / every_step
    else:
        row[f"h{horizon}_ratio_best_over_every_step"] = None

    return row


def _collect_rows(
    root_specs: Iterable[Tuple[str, Path]],
    horizon: int,
    eval_file_name: str,
    select: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for root_label, root_dir in root_specs:
        if not root_dir.exists():
            continue
        for system_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
            for seed_name, run_dir in _select_run_dirs(system_dir, select):
                row = _extract_row(
                    root_label=root_label,
                    system_dir=system_dir,
                    run_dir=run_dir,
                    horizon=horizon,
                    eval_file_name=eval_file_name,
                    seed_name=seed_name,
                )
                if row is not None:
                    rows.append(row)
    return rows


def _system_id(row: Dict[str, object]) -> str:
    system_key = str(row.get("system_key", "")).strip()
    if system_key and system_key != "None":
        return system_key
    return str(row.get("system_name", "")).strip()


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _summarize(
    rows: List[Dict[str, object]],
    horizon: int,
    good_threshold: float,
    essential_factor: float,
) -> Dict[str, Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["root_label"])].append(row)

    summary: Dict[str, Dict[str, object]] = {}
    for root_label, grp in groups.items():
        bp_vals = [_safe_float(r.get(f"h{horizon}_best_periodic_mean")) for r in grp]
        nr_vals = [_safe_float(r.get(f"h{horizon}_no_reencode_mean")) for r in grp]
        es_vals = [_safe_float(r.get(f"h{horizon}_every_step_mean")) for r in grp]
        bp_vals_f = [v for v in bp_vals if v is not None]

        improved_nr = 0
        improved_es = 0
        essential = 0
        per_system_bp: Dict[str, List[float]] = defaultdict(list)
        per_system_mode: Dict[str, List[str]] = defaultdict(list)

        for r in grp:
            sid = _system_id(r)
            bp = _safe_float(r.get(f"h{horizon}_best_periodic_mean"))
            nr = _safe_float(r.get(f"h{horizon}_no_reencode_mean"))
            es = _safe_float(r.get(f"h{horizon}_every_step_mean"))
            mode = r.get(f"h{horizon}_best_periodic_mode")

            if bp is not None:
                per_system_bp[sid].append(bp)
            if isinstance(mode, str):
                per_system_mode[sid].append(mode)

            if bp is not None and nr is not None:
                if bp < nr:
                    improved_nr += 1
                if bp > 0 and nr / bp >= essential_factor:
                    essential += 1
            if bp is not None and es is not None and bp < es:
                improved_es += 1

        mode_counts = Counter(
            str(r.get(f"h{horizon}_best_periodic_mode"))
            for r in grp
            if r.get(f"h{horizon}_best_periodic_mode") is not None
        )
        system_medians = [median(vals) for vals in per_system_bp.values() if vals]
        system_mode_counts = Counter()
        for modes in per_system_mode.values():
            if modes:
                system_mode_counts[Counter(modes).most_common(1)[0][0]] += 1

        top_modes = [{"mode": mode, "count": count} for mode, count in mode_counts.most_common(5)]
        top_system_modes = [
            {"mode": mode, "count": count} for mode, count in system_mode_counts.most_common(5)
        ]

        summary[root_label] = {
            "n_rows": len(grp),
            "n_systems": len(per_system_bp),
            f"h{horizon}_median_no_reencode_mean": median([v for v in nr_vals if v is not None]) if any(v is not None for v in nr_vals) else None,
            f"h{horizon}_median_every_step_mean": median([v for v in es_vals if v is not None]) if any(v is not None for v in es_vals) else None,
            f"h{horizon}_median_best_periodic_mean_rows": median(bp_vals_f) if bp_vals_f else None,
            f"h{horizon}_median_best_periodic_mean_systems": median(system_medians) if system_medians else None,
            f"h{horizon}_systems_below_{good_threshold}": sum(1 for v in system_medians if v < good_threshold),
            f"h{horizon}_improved_vs_no_reencode": improved_nr,
            f"h{horizon}_improved_vs_every_step": improved_es,
            f"h{horizon}_essential_improvement_ge_{essential_factor}x_vs_no_reencode": essential,
            f"h{horizon}_top_best_periodic_modes": top_modes,
            f"h{horizon}_top_system_consensus_modes": top_system_modes,
        }

    return summary


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    path: Path,
    summary: Dict[str, Dict[str, object]],
    rows: List[Dict[str, object]],
    horizon: int,
    good_threshold: float,
    essential_factor: float,
) -> None:
    lines: List[str] = []
    lines.append("# Forecasting Summary")
    lines.append("")
    lines.append(f"- Horizon: H{horizon}")
    lines.append(f"- Good-forecast threshold (system median best-periodic): H{horizon} < {good_threshold}")
    lines.append(
        f"- Essential reencoding threshold (row-level): no-reencode / best-periodic >= {essential_factor}x"
    )
    lines.append("")

    lines.append(
        f"| root | n_rows | n_systems | median H{horizon} best (rows) | median H{horizon} best (systems) | good systems | improved vs no-re | improved vs every-step | essential (>= {essential_factor}x) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for root_label, stats in sorted(summary.items()):
        lines.append(
            f"| {root_label} | {stats['n_rows']} | {stats['n_systems']} | "
            f"{_fmt(_safe_float(stats.get(f'h{horizon}_median_best_periodic_mean_rows')))} | "
            f"{_fmt(_safe_float(stats.get(f'h{horizon}_median_best_periodic_mean_systems')))} | "
            f"{stats.get(f'h{horizon}_systems_below_{good_threshold}', 0)} | "
            f"{stats.get(f'h{horizon}_improved_vs_no_reencode', 0)} | "
            f"{stats.get(f'h{horizon}_improved_vs_every_step', 0)} | "
            f"{stats.get(f'h{horizon}_essential_improvement_ge_{essential_factor}x_vs_no_reencode', 0)} |"
        )

    lines.append("")
    lines.append("## Per-Run Rows")
    lines.append("")
    lines.append(
        f"| root | system_key | seed | run_id | H{horizon} no-re | H{horizon} every-step | H{horizon} best-periodic | best mode | ratio bp/no-re | ratio bp/every-step |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---|---:|---:|")
    sort_key = lambda r: (
        str(r["root_label"]),
        str(r.get("system_key") or r.get("system_name")),
        str(r.get("seed_name") or ""),
        str(r["run_id"]),
    )
    for row in sorted(rows, key=sort_key):
        lines.append(
            f"| {row['root_label']} | {row.get('system_key') or row.get('system_name')} | {row.get('seed_name') or 'N/A'} | {row['run_id']} | "
            f"{_fmt(_safe_float(row.get(f'h{horizon}_no_reencode_mean')))} | "
            f"{_fmt(_safe_float(row.get(f'h{horizon}_every_step_mean')))} | "
            f"{_fmt(_safe_float(row.get(f'h{horizon}_best_periodic_mean')))} | "
            f"{row.get(f'h{horizon}_best_periodic_mode') or 'N/A'} | "
            f"{_fmt(_safe_float(row.get(f'h{horizon}_ratio_best_over_no_re')))} | "
            f"{_fmt(_safe_float(row.get(f'h{horizon}_ratio_best_over_every_step')))} |"
        )

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect mixed-system forecasting metrics from run directories."
    )
    parser.add_argument(
        "--run_roots",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Run roots to scan. Supports label via LABEL=PATH, e.g. "
            "depth_3=/scratch/lista_depth/phase1/depth_3"
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for summary artifacts.")
    parser.add_argument("--horizon", type=int, default=1000, help="Horizon to summarize.")
    parser.add_argument(
        "--good_threshold",
        type=float,
        default=10.0,
        help="Threshold for counting systems with good best-periodic forecasting.",
    )
    parser.add_argument(
        "--essential_factor",
        type=float,
        default=10.0,
        help="Threshold for counting essential reencoding improvements (no-re / best-periodic).",
    )
    parser.add_argument(
        "--select",
        type=str,
        choices=["latest", "all"],
        default="latest",
        help=(
            "Selection mode. 'latest' keeps latest run per system-seed when seed folders exist; "
            "'all' keeps all discovered runs."
        ),
    )
    parser.add_argument(
        "--eval_file_name",
        type=str,
        default="evaluation_results_best.json",
        help="Evaluation artifact filename expected under each run directory.",
    )
    return parser.parse_args()


def _parse_root_spec(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        return label.strip(), Path(raw_path).expanduser()
    path = Path(spec).expanduser()
    return path.name, path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root_specs = [_parse_root_spec(spec) for spec in args.run_roots]
    rows = _collect_rows(
        root_specs=root_specs,
        horizon=args.horizon,
        eval_file_name=args.eval_file_name,
        select=args.select,
    )

    summary = _summarize(
        rows=rows,
        horizon=args.horizon,
        good_threshold=args.good_threshold,
        essential_factor=args.essential_factor,
    )

    rows_json = output_dir / "forecasting_rows.json"
    summary_json = output_dir / "forecasting_summary.json"
    rows_csv = output_dir / "forecasting_rows.csv"
    summary_md = output_dir / "forecasting_summary.md"

    rows_json.write_text(json.dumps(rows, indent=2))
    summary_json.write_text(json.dumps(summary, indent=2))
    _write_csv(rows_csv, rows)
    _write_markdown(
        path=summary_md,
        summary=summary,
        rows=rows,
        horizon=args.horizon,
        good_threshold=args.good_threshold,
        essential_factor=args.essential_factor,
    )

    print(f"Collected {len(rows)} rows from {len(root_specs)} roots.")
    print(f"Wrote: {rows_json}")
    print(f"Wrote: {rows_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_md}")


if __name__ == "__main__":
    main()
