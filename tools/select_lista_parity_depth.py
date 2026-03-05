#!/usr/bin/env python3
"""Select depth_star_parity for LISTA-vs-generic_sparse parity experiments."""

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
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            out = float(raw)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def _safe_int(value: object) -> Optional[int]:
    out = _safe_float(value)
    if out is None:
        return None
    return int(round(out))


def _rel_abs(candidate: Optional[float], anchor: Optional[float]) -> Optional[float]:
    if candidate is None or anchor is None:
        return None
    denom = max(abs(anchor), 1.0)
    return abs(candidate - anchor) / denom


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not usable:
        return None
    return sum(usable) / float(len(usable))


def _parse_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _system_id(row: Dict[str, object]) -> str:
    system_key = str(row.get("system_key", "")).strip()
    if system_key and system_key != "None":
        return system_key
    return str(row.get("system_name", "")).strip()


def _seed_id(row: Dict[str, object]) -> str:
    seed_name = str(row.get("seed_name", "")).strip()
    if seed_name and seed_name != "None":
        return seed_name
    seed = _safe_int(row.get("seed"))
    if seed is not None:
        return f"seed_{seed}"
    run_dir = str(row.get("run_dir", ""))
    match = re.search(r"/(seed_\d+)(?:/|$)", run_dir)
    if match:
        return match.group(1)
    return "seed_0"


def _run_sort_key(row: Dict[str, object]) -> Tuple[str, str]:
    return (str(row.get("run_id", "")), str(row.get("run_dir", "")))


def _latest_rows_by_system_seed(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    latest: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        system = _system_id(row)
        if not system:
            continue
        seed = _seed_id(row)
        key = (system, seed)
        prev = latest.get(key)
        if prev is None or _run_sort_key(row) > _run_sort_key(prev):
            latest[key] = row
    return list(latest.values())


def _group_rows_by_root(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        root = str(row.get("root_label", "")).strip()
        if root:
            grouped[root].append(row)
    return grouped


def _mode_distribution(consensus_modes: List[str]) -> Dict[str, float]:
    if not consensus_modes:
        return {}
    counts = Counter(consensus_modes)
    total = float(sum(counts.values()))
    return {mode: count / total for mode, count in counts.items()}


def _distribution_l1_half(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a.keys()) | set(b.keys())
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _root_forecast_stats(
    rows: List[Dict[str, object]],
    horizon: int,
    good_threshold: float,
    catastrophic_threshold: float,
) -> Dict[str, object]:
    best_key = f"h{horizon}_best_periodic_mean"
    no_re_key = f"h{horizon}_no_reencode_mean"
    every_key = f"h{horizon}_every_step_mean"
    mode_key = f"h{horizon}_best_periodic_mode"

    latest = _latest_rows_by_system_seed(rows)
    by_system: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in latest:
        by_system[_system_id(row)].append(row)

    system_medians: Dict[str, float] = {}
    system_worst: Dict[str, float] = {}
    system_consensus_mode: Dict[str, str] = {}
    system_no_re_median: Dict[str, float] = {}
    system_every_median: Dict[str, float] = {}

    for system, items in by_system.items():
        best_vals = [_safe_float(i.get(best_key)) for i in items]
        best_vals = [v for v in best_vals if v is not None]
        if not best_vals:
            continue
        no_re_vals = [_safe_float(i.get(no_re_key)) for i in items]
        no_re_vals = [v for v in no_re_vals if v is not None]
        every_vals = [_safe_float(i.get(every_key)) for i in items]
        every_vals = [v for v in every_vals if v is not None]

        system_medians[system] = float(median(best_vals))
        system_worst[system] = float(max(best_vals))
        if no_re_vals:
            system_no_re_median[system] = float(median(no_re_vals))
        if every_vals:
            system_every_median[system] = float(median(every_vals))

        modes = [str(i.get(mode_key)) for i in items if i.get(mode_key)]
        if modes:
            system_consensus_mode[system] = Counter(modes).most_common(1)[0][0]

    medians = list(system_medians.values())
    worst_vals = list(system_worst.values())
    consensus_modes = list(system_consensus_mode.values())

    good_systems = sum(1 for v in medians if v < good_threshold)
    catastrophic_systems = sum(1 for v in medians if v >= catastrophic_threshold)
    all_seeds_good = sum(1 for _, v in system_worst.items() if v < good_threshold)
    any_seed_catastrophic = sum(1 for _, v in system_worst.items() if v >= catastrophic_threshold)
    improved_vs_no_re = sum(
        1
        for system, bp in system_medians.items()
        if system in system_no_re_median and bp < system_no_re_median[system]
    )
    improved_vs_every = sum(
        1
        for system, bp in system_medians.items()
        if system in system_every_median and bp < system_every_median[system]
    )

    return {
        "n_systems": len(system_medians),
        "n_rows_collected": len(latest),
        "system_medians": system_medians,
        "median_h_best": float(median(medians)) if medians else None,
        "median_worst_seed_h_best": float(median(worst_vals)) if worst_vals else None,
        "good_systems": good_systems,
        "catastrophic_systems": catastrophic_systems,
        "all_seeds_good_systems": all_seeds_good,
        "any_seed_catastrophic_systems": any_seed_catastrophic,
        "improved_vs_no_reencode": improved_vs_no_re,
        "improved_vs_every_step": improved_vs_every,
        "mode_distribution": _mode_distribution(consensus_modes),
    }


def _is_run_dir(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "metrics_history.jsonl").exists()


def _discover_run_dirs(system_dir: Path) -> List[Tuple[Optional[str], Path]]:
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


def _latest_run_dirs(system_dir: Path) -> List[Tuple[Optional[str], Path]]:
    run_dirs = _discover_run_dirs(system_dir)
    if not run_dirs:
        return []
    grouped: Dict[str, List[Tuple[Optional[str], Path]]] = defaultdict(list)
    for seed_name, run_dir in run_dirs:
        key = seed_name if seed_name else "__root__"
        grouped[key].append((seed_name, run_dir))
    selected: List[Tuple[Optional[str], Path]] = []
    for _, items in sorted(grouped.items()):
        selected.append(sorted(items, key=lambda t: (t[1].name, str(t[1])))[-1])
    return selected


def _parse_metrics_history(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    target_names = {
        "train/loss",
        "train/alignment_loss",
        "train/residual_loss",
        "train/reconst_loss",
        "train/sparsity_loss",
    }
    out: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            name = str(row.get("name", ""))
            if name not in target_names:
                continue
            step = _safe_int(row.get("step"))
            if step is None:
                continue
            try:
                value = float(row.get("value"))
            except (TypeError, ValueError):
                continue
            out[name].append((step, value))
    for key in out:
        out[key] = sorted(out[key], key=lambda x: x[0])
    return out


def _run_train_features(metrics_series: Dict[str, List[Tuple[int, float]]], spike_factor: float) -> Dict[str, float]:
    def last(name: str) -> Optional[float]:
        series = metrics_series.get(name, [])
        if not series:
            return None
        return float(series[-1][1])

    loss_series = [v for _, v in metrics_series.get("train/loss", [])]
    spikes = 0
    for idx in range(1, len(loss_series)):
        prev = max(abs(loss_series[idx - 1]), 1e-8)
        if loss_series[idx] > spike_factor * prev:
            spikes += 1
    has_non_finite = 0.0
    for entries in metrics_series.values():
        for _, value in entries:
            if not math.isfinite(float(value)):
                has_non_finite = 1.0
                break
        if has_non_finite > 0.0:
            break

    alignment_last = last("train/alignment_loss")
    if alignment_last is None:
        alignment_last = last("train/residual_loss")

    return {
        "final_loss": last("train/loss"),
        "final_alignment": alignment_last,
        "final_reconst": last("train/reconst_loss"),
        "final_sparsity": last("train/sparsity_loss"),
        "loss_spike_count": float(spikes),
        "has_non_finite": has_non_finite,
    }


def _root_train_stats(root_path: Path, spike_factor: float) -> Dict[str, object]:
    run_features: List[Dict[str, float]] = []
    if not root_path.exists():
        return {"n_runs": 0}
    for system_dir in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        for _, run_dir in _latest_run_dirs(system_dir):
            metrics_path = run_dir / "metrics_history.jsonl"
            if not metrics_path.exists():
                continue
            series = _parse_metrics_history(metrics_path)
            if not series:
                continue
            run_features.append(_run_train_features(series, spike_factor=spike_factor))

    def med(key: str) -> Optional[float]:
        vals = [_safe_float(row.get(key)) for row in run_features]
        vals = [v for v in vals if v is not None]
        return float(median(vals)) if vals else None

    def mean(key: str) -> Optional[float]:
        vals = [_safe_float(row.get(key)) for row in run_features]
        vals = [v for v in vals if v is not None]
        return (sum(vals) / float(len(vals))) if vals else None

    return {
        "n_runs": len(run_features),
        "final_loss_median": med("final_loss"),
        "final_alignment_median": med("final_alignment"),
        "final_reconst_median": med("final_reconst"),
        "final_sparsity_median": med("final_sparsity"),
        "loss_spike_count_mean": mean("loss_spike_count"),
        "non_finite_fraction": mean("has_non_finite"),
    }


def _parse_root_specs(specs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid run root spec '{spec}'. Expected LABEL=PATH.")
        label, raw = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid run root spec '{spec}': empty label.")
        out[label] = Path(raw).expanduser()
    return out


def _train_distance(candidate: Dict[str, object], anchor: Dict[str, object]) -> Optional[float]:
    features = [
        _rel_abs(_safe_float(candidate.get("final_loss_median")), _safe_float(anchor.get("final_loss_median"))),
        _rel_abs(
            _safe_float(candidate.get("final_alignment_median")),
            _safe_float(anchor.get("final_alignment_median")),
        ),
        _rel_abs(
            _safe_float(candidate.get("final_reconst_median")),
            _safe_float(anchor.get("final_reconst_median")),
        ),
        _rel_abs(
            _safe_float(candidate.get("final_sparsity_median")),
            _safe_float(anchor.get("final_sparsity_median")),
        ),
        _rel_abs(
            _safe_float(candidate.get("loss_spike_count_mean")),
            _safe_float(anchor.get("loss_spike_count_mean")),
        ),
    ]
    non_finite_c = _safe_float(candidate.get("non_finite_fraction"))
    non_finite_a = _safe_float(anchor.get("non_finite_fraction"))
    if non_finite_c is not None and non_finite_a is not None:
        features.append(abs(non_finite_c - non_finite_a))
    return _mean(features)


def _forecast_distance(candidate: Dict[str, object], anchor: Dict[str, object]) -> Optional[float]:
    n_norm = max(int(anchor.get("n_systems", 0)), 1)
    d_mode = _distribution_l1_half(
        candidate.get("mode_distribution", {}),
        anchor.get("mode_distribution", {}),
    )
    parts = [
        _rel_abs(_safe_float(candidate.get("median_h_best")), _safe_float(anchor.get("median_h_best"))),
        abs(int(candidate.get("good_systems", 0)) - int(anchor.get("good_systems", 0))) / float(n_norm),
        abs(int(candidate.get("catastrophic_systems", 0)) - int(anchor.get("catastrophic_systems", 0)))
        / float(n_norm),
        d_mode,
    ]
    return _mean(parts)


def _robust_distance(candidate: Dict[str, object], anchor: Dict[str, object]) -> Optional[float]:
    n_norm = max(int(anchor.get("n_systems", 0)), 1)
    parts = [
        abs(int(candidate.get("all_seeds_good_systems", 0)) - int(anchor.get("all_seeds_good_systems", 0)))
        / float(n_norm),
        abs(
            int(candidate.get("any_seed_catastrophic_systems", 0))
            - int(anchor.get("any_seed_catastrophic_systems", 0))
        )
        / float(n_norm),
        _rel_abs(
            _safe_float(candidate.get("median_worst_seed_h_best")),
            _safe_float(anchor.get("median_worst_seed_h_best")),
        ),
    ]
    return _mean(parts)


def _depth_from_label(label: str) -> int:
    match = re.match(r"depth_(\d+)$", label)
    if match:
        return int(match.group(1))
    return 10**9


def _score_candidate(
    d_train: Optional[float],
    d_forecast: Optional[float],
    d_robust: Optional[float],
    w_train: float,
    w_forecast: float,
    w_robust: float,
) -> Tuple[Optional[float], str]:
    available: List[Tuple[float, float]] = []
    if d_train is not None:
        available.append((w_train, d_train))
    if d_forecast is not None:
        available.append((w_forecast, d_forecast))
    if d_robust is not None:
        available.append((w_robust, d_robust))

    if not available:
        return None, "no_components_available"
    weight_sum = sum(w for w, _ in available)
    if weight_sum <= 0:
        return None, "invalid_weights"
    score = sum(w * d for w, d in available) / weight_sum
    if d_train is None:
        return score, "train_distance_missing_reweighted"
    return score, "full_weights"


def _write_markdown(
    out_path: Path,
    anchor_root: str,
    selected: Optional[Dict[str, object]],
    candidate_rows: List[Dict[str, object]],
    notes: List[str],
) -> None:
    def _fmt(value: object) -> str:
        fval = _safe_float(value)
        if fval is None:
            return "N/A"
        return f"{fval:.6f}"

    lines: List[str] = []
    lines.append("# LISTA Parity Depth Selection")
    lines.append("")
    lines.append(f"- Anchor root: `{anchor_root}`")
    if selected is None:
        lines.append("- Selected depth: `N/A` (no candidate passed hard gates)")
    else:
        lines.append(
            f"- Selected depth: `{selected['candidate_root']}` "
            f"(depth={selected['depth']}, score={selected['score']:.6f})"
        )
    lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Candidate Table")
    lines.append("")
    lines.append(
        "| candidate | depth | pass_hard_gates | score | D_train | D_forecast | D_robust | catastrophic | all_seeds_good | any_seed_catastrophic |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in candidate_rows:
        stats = row.get("forecast_stats", {})
        lines.append(
            f"| {row['candidate_root']} | {row['depth']} | {row['pass_hard_gates']} | "
            f"{_fmt(row.get('score'))} | "
            f"{_fmt(row.get('d_train'))} | "
            f"{_fmt(row.get('d_forecast'))} | "
            f"{_fmt(row.get('d_robust'))} | "
            f"{int(stats.get('catastrophic_systems', 0))} | "
            f"{int(stats.get('all_seeds_good_systems', 0))} | "
            f"{int(stats.get('any_seed_catastrophic_systems', 0))} |"
        )

    out_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select depth_star_parity from collected forecasting rows."
    )
    parser.add_argument("--rows_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--anchor_root", type=str, default="generic_sparse")
    parser.add_argument(
        "--candidate_roots",
        type=str,
        nargs="+",
        default=["depth_0", "depth_1", "depth_2", "depth_3"],
    )
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--good_threshold", type=float, default=10.0)
    parser.add_argument("--catastrophic_threshold", type=float, default=1000.0)
    parser.add_argument("--catastrophic_slack", type=int, default=1)
    parser.add_argument("--seed_catastrophic_slack", type=int, default=1)
    parser.add_argument("--all_seeds_good_drop", type=int, default=2)
    parser.add_argument("--weight_train", type=float, default=0.45)
    parser.add_argument("--weight_forecast", type=float, default=0.35)
    parser.add_argument("--weight_robust", type=float, default=0.20)
    parser.add_argument("--run_roots", type=str, nargs="*", default=[])
    parser.add_argument(
        "--spike_factor",
        type=float,
        default=3.0,
        help="A loss spike is counted when loss_t > spike_factor * abs(loss_{t-1}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _parse_rows(Path(args.rows_csv))
    grouped = _group_rows_by_root(rows)
    if args.anchor_root not in grouped:
        raise ValueError(f"Anchor root '{args.anchor_root}' not found in {args.rows_csv}")

    forecast_stats = {
        root: _root_forecast_stats(
            root_rows,
            horizon=args.horizon,
            good_threshold=args.good_threshold,
            catastrophic_threshold=args.catastrophic_threshold,
        )
        for root, root_rows in grouped.items()
    }
    anchor_forecast = forecast_stats[args.anchor_root]

    train_stats: Dict[str, Dict[str, object]] = {}
    notes: List[str] = []
    if args.run_roots:
        run_roots = _parse_root_specs(args.run_roots)
        for label, root_path in run_roots.items():
            train_stats[label] = _root_train_stats(root_path, spike_factor=args.spike_factor)
    else:
        notes.append("No --run_roots were provided; D_train is unavailable and weights are re-normalized.")

    candidate_rows: List[Dict[str, object]] = []
    for candidate_root in args.candidate_roots:
        if candidate_root not in grouped:
            candidate_rows.append(
                {
                    "candidate_root": candidate_root,
                    "depth": _depth_from_label(candidate_root),
                    "present": False,
                    "pass_hard_gates": False,
                    "hard_gate_reasons": ["missing_from_rows_csv"],
                    "score": None,
                    "score_mode": "missing",
                    "d_train": None,
                    "d_forecast": None,
                    "d_robust": None,
                    "forecast_stats": {},
                    "train_stats": {},
                }
            )
            continue

        cand_forecast = forecast_stats[candidate_root]
        cand_train = train_stats.get(candidate_root, {})
        anchor_train = train_stats.get(args.anchor_root, {})

        d_forecast = _forecast_distance(cand_forecast, anchor_forecast)
        d_robust = _robust_distance(cand_forecast, anchor_forecast)
        d_train = None
        if cand_train and anchor_train and int(cand_train.get("n_runs", 0)) > 0 and int(anchor_train.get("n_runs", 0)) > 0:
            d_train = _train_distance(cand_train, anchor_train)

        score, score_mode = _score_candidate(
            d_train=d_train,
            d_forecast=d_forecast,
            d_robust=d_robust,
            w_train=args.weight_train,
            w_forecast=args.weight_forecast,
            w_robust=args.weight_robust,
        )

        hard_gate_reasons: List[str] = []
        cand_cat = int(cand_forecast.get("catastrophic_systems", 0))
        anchor_cat = int(anchor_forecast.get("catastrophic_systems", 0))
        if cand_cat > anchor_cat + int(args.catastrophic_slack):
            hard_gate_reasons.append(
                "catastrophic_system_gate_failed"
            )

        cand_any_seed_cat = int(cand_forecast.get("any_seed_catastrophic_systems", 0))
        anchor_any_seed_cat = int(anchor_forecast.get("any_seed_catastrophic_systems", 0))
        if cand_any_seed_cat > anchor_any_seed_cat + int(args.seed_catastrophic_slack):
            hard_gate_reasons.append("seed_catastrophic_gate_failed")

        cand_all_good = int(cand_forecast.get("all_seeds_good_systems", 0))
        anchor_all_good = int(anchor_forecast.get("all_seeds_good_systems", 0))
        if cand_all_good < anchor_all_good - int(args.all_seeds_good_drop):
            hard_gate_reasons.append("all_seeds_good_gate_failed")

        if score is None:
            hard_gate_reasons.append("score_unavailable")

        candidate_rows.append(
            {
                "candidate_root": candidate_root,
                "depth": _depth_from_label(candidate_root),
                "present": True,
                "pass_hard_gates": len(hard_gate_reasons) == 0,
                "hard_gate_reasons": hard_gate_reasons,
                "score": score,
                "score_mode": score_mode,
                "d_train": d_train,
                "d_forecast": d_forecast,
                "d_robust": d_robust,
                "forecast_stats": cand_forecast,
                "train_stats": cand_train,
            }
        )

    passing = [row for row in candidate_rows if row.get("pass_hard_gates")]
    passing_sorted = sorted(
        passing,
        key=lambda row: (
            float(row["score"]),
            int(row["forecast_stats"].get("catastrophic_systems", 0)),
            int(row["depth"]),
            float(row["forecast_stats"].get("median_h_best") or math.inf),
        ),
    )
    selected = passing_sorted[0] if passing_sorted else None
    candidate_rows_sorted = sorted(
        candidate_rows,
        key=lambda row: (
            0 if row.get("pass_hard_gates") else 1,
            float(row["score"]) if isinstance(row.get("score"), float) else math.inf,
            int(row.get("depth", 10**9)),
        ),
    )

    output = {
        "rows_csv": args.rows_csv,
        "anchor_root": args.anchor_root,
        "candidate_roots": args.candidate_roots,
        "weights": {
            "train": args.weight_train,
            "forecast": args.weight_forecast,
            "robust": args.weight_robust,
        },
        "hard_gates": {
            "catastrophic_slack": args.catastrophic_slack,
            "seed_catastrophic_slack": args.seed_catastrophic_slack,
            "all_seeds_good_drop": args.all_seeds_good_drop,
        },
        "anchor_forecast_stats": anchor_forecast,
        "anchor_train_stats": train_stats.get(args.anchor_root, {}),
        "candidates": candidate_rows_sorted,
        "selected_candidate": selected,
        "depth_star_parity": selected["depth"] if selected is not None else None,
        "selection_status": "ok" if selected is not None else "no_candidate_passed_hard_gates",
        "notes": notes,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "depth_star_parity_selection.json"
    out_md = out_dir / "depth_star_parity_selection.md"
    out_json.write_text(json.dumps(output, indent=2))
    _write_markdown(
        out_path=out_md,
        anchor_root=args.anchor_root,
        selected=selected,
        candidate_rows=candidate_rows_sorted,
        notes=notes,
    )

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")
    if selected is None:
        print("No candidate passed hard gates.")
    else:
        print(
            f"Selected depth_star_parity={selected['depth']} "
            f"({selected['candidate_root']}, score={selected['score']:.6f})"
        )


if __name__ == "__main__":
    main()
