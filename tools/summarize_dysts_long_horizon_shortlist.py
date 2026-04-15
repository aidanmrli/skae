#!/usr/bin/env python3
"""Rescore shortlisted Dysts checkpoints at ultra-long horizons.

Protocol:
1. Scan the current matched Dysts LISTA roots.
2. Rank checkpoints per system by saved H3000 best-periodic mean.
3. Keep the top-k checkpoints per system.
4. Re-evaluate that shortlist on one shared rollout batch out to max(horizons).
5. Read off final-state MSE at each requested horizon from the same rollout.

Outputs:
- candidate-level CSV/JSON for every rescored checkpoint
- selected-winner CSV/JSON for each system/horizon
- combined JSON/Markdown summary with winner counts and switch diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Optional, Sequence

import torch

from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, generate_trajectory, make_env
from skae.evaluation import _make_km_env_n_step
from skae.model import make_model


DYSTS_SYSTEMS: tuple[str, ...] = (
    "dysts:Chua",
    "dysts:Dadras",
    "dysts:DequanLi",
    "dysts:Duffing",
    "dysts:Hadley",
    "dysts:LorenzCoupled",
    "dysts:LuChenCheng",
    "dysts:MultiChua",
    "dysts:QiChen",
    "dysts:RikitakeDynamo",
    "dysts:Sakarya",
    "dysts:SanUmSrisuchinwong",
    "dysts:ShimizuMorioka",
    "dysts:SprottTorus",
    "dysts:WangSun",
)


@dataclass(frozen=True)
class RootSpec:
    label: str
    display_name: str
    root_dir: Path


ROOT_SPECS: tuple[RootSpec, ...] = (
    RootSpec(
        label="dense_lista",
        display_name="dense LISTA",
        root_dir=Path(
            "/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/"
            "paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3"
        ),
    ),
    RootSpec(
        label="blockdiag_lista_sc3em3",
        display_name="blockdiag LISTA (sc=3e-3)",
        root_dir=Path(
            "/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/"
            "paper_followup_recipes/lista_blockdiag_ns200k_denseopt_sc3em3"
        ),
    ),
    RootSpec(
        label="blockdiag_lista_sc6em3",
        display_name="blockdiag LISTA (sc=6e-3)",
        root_dir=Path(
            "/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/"
            "paper_followup_recipes/lista_blockdiag_ns200k_denseopt_sc6em3"
        ),
    ),
)


@dataclass(frozen=True)
class Candidate:
    system: str
    root_label: str
    root_display_name: str
    run_dir: Path
    checkpoint: Path
    eval_json: Path
    h3000_best_periodic_mean: float
    h3000_best_periodic_mode: Optional[str]
    seed: Optional[int]


def _system_dir_name(system: str) -> str:
    if system.startswith("dysts:"):
        return f"dysts_{system.split(':', 1)[1]}"
    return system


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _seed_from_run_dir(run_dir: Path) -> Optional[int]:
    for part in run_dir.parts:
        if part.startswith("seed_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return None
    return None


def _parse_mode_name(mode_name: Optional[str]) -> int:
    if mode_name is None:
        raise ValueError("Missing periodic mode")
    if mode_name == "no_reencode":
        return 0
    if mode_name == "every_step":
        return 1
    if mode_name.startswith("periodic_"):
        return int(mode_name.split("_", 1)[1])
    raise ValueError(f"Unsupported periodic mode name: {mode_name}")


def _load_eval_json(eval_path: Path) -> Dict[str, Any]:
    with eval_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_root_candidates(system: str, root_spec: RootSpec, shortlist_horizon: int) -> list[Candidate]:
    system_dir = root_spec.root_dir / _system_dir_name(system)
    if not system_dir.exists():
        return []

    candidates: list[Candidate] = []
    for eval_json in sorted(system_dir.glob("**/evaluation_results_best.json")):
        payload = _load_eval_json(eval_json)
        system_metrics = payload.get(system)
        if not isinstance(system_metrics, dict):
            continue
        best_periodic = system_metrics.get("best_periodic", {})
        if not isinstance(best_periodic, dict):
            continue
        horizon_metrics = best_periodic.get(str(shortlist_horizon))
        if not isinstance(horizon_metrics, dict):
            continue
        mean = _safe_float(horizon_metrics.get("mean"))
        if mean is None:
            continue

        run_dir = eval_json.parent
        candidates.append(
            Candidate(
                system=system,
                root_label=root_spec.label,
                root_display_name=root_spec.display_name,
                run_dir=run_dir,
                checkpoint=run_dir / "checkpoint.pt",
                eval_json=eval_json,
                h3000_best_periodic_mean=mean,
                h3000_best_periodic_mode=horizon_metrics.get("mode"),
                seed=_seed_from_run_dir(run_dir),
            )
        )
    return candidates


def _load_model(run_dir: Path, system: str, device: str):
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg_dict = checkpoint["config"]
    env_dict = cfg_dict.get("ENV", {})
    competitive_lv_dict = env_dict.get("COMPETITIVE_LV")
    if isinstance(competitive_lv_dict, dict):
        competitive_lv_dict.pop("SYSTEM_SEED", None)

    cfg = Config.from_dict(cfg_dict)
    cfg.ENV.ENV_NAME = system

    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    load_model_state_dict_compat(model, checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    model.dt = getattr(env.unwrapped, "dt", model.dt)
    return cfg, env, model


def _make_shared_batch(
    *,
    env,
    batch_size: int,
    horizon: int,
    rng_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(int(rng_seed))
    vec_env = VectorWrapper(env, batch_size)
    init_states = vec_env.reset(rng)
    true_future = generate_trajectory(vec_env.step, init_states, length=horizon)
    return init_states, true_future


def _metric_from_rollout(
    pred_future: torch.Tensor,
    true_future: torch.Tensor,
    horizon: int,
) -> tuple[Optional[float], Optional[float], int]:
    final_sq_error = (pred_future[horizon - 1] - true_future[horizon - 1]).pow(2).mean(dim=-1)
    finite_mask = torch.isfinite(final_sq_error)
    valid = final_sq_error[finite_mask]
    if valid.numel() == 0:
        return None, None, 0
    mean = float(valid.mean().item())
    std = float(valid.std(unbiased=False).item()) if valid.numel() > 1 else 0.0
    return mean, std, int(valid.numel())


def _selection_key(row: Dict[str, Any], horizon: int) -> float:
    value = _safe_float(row.get(f"shared_batch_h{horizon}_best_periodic_mean"))
    return value if value is not None else float("inf")


def _evaluate_candidate(
    candidate: Candidate,
    *,
    init_states: torch.Tensor,
    true_future: torch.Tensor,
    horizons: Sequence[int],
    device: str,
    shortlist_rank_h3000: int,
) -> Dict[str, Any]:
    _, _, model = _load_model(candidate.run_dir, candidate.system, device)
    max_horizon = max(int(h) for h in horizons)
    periodic_mode = _parse_mode_name(candidate.h3000_best_periodic_mode)
    pred_future = _make_km_env_n_step(model, init_states, max_horizon, periodic_mode)

    row: Dict[str, Any] = {
        "system": candidate.system,
        "root_label": candidate.root_label,
        "root_display_name": candidate.root_display_name,
        "run_dir": str(candidate.run_dir),
        "checkpoint": str(candidate.checkpoint),
        "seed": candidate.seed,
        "periodic_mode": periodic_mode,
        "shortlist_rank_h3000": shortlist_rank_h3000,
        "h3000_best_periodic_mean": candidate.h3000_best_periodic_mean,
        "h3000_best_periodic_mode": candidate.h3000_best_periodic_mode,
    }
    for horizon in sorted(int(h) for h in horizons):
        mean, std, num_valid = _metric_from_rollout(pred_future, true_future, horizon)
        row[f"shared_batch_h{horizon}_best_periodic_mean"] = mean
        row[f"shared_batch_h{horizon}_best_periodic_std"] = std
        row[f"shared_batch_h{horizon}_num_valid"] = num_valid
    return row


def _rank_candidates(candidate_rows: list[Dict[str, Any]], horizons: Sequence[int]) -> None:
    for horizon in sorted(int(h) for h in horizons):
        ordered = sorted(candidate_rows, key=lambda row: _selection_key(row, horizon))
        for rank, row in enumerate(ordered, start=1):
            row[f"rank_h{horizon}"] = rank


def _family_best_by_horizon(
    candidate_rows: Sequence[Dict[str, Any]],
    horizons: Sequence[int],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    family_best: Dict[str, Dict[str, Dict[str, Any]]] = {}
    roots = sorted({str(row["root_label"]) for row in candidate_rows})
    for horizon in sorted(int(h) for h in horizons):
        family_best[str(horizon)] = {}
        for root_label in roots:
            root_rows = [row for row in candidate_rows if str(row["root_label"]) == root_label]
            if not root_rows:
                continue
            family_best[str(horizon)][root_label] = min(
                root_rows,
                key=lambda row: _selection_key(row, horizon),
            )
    return family_best


def _collect_system_result(
    system: str,
    *,
    shortlist_horizon: int,
    shortlist_k: int,
    batch_size: int,
    horizons: Sequence[int],
    rng_seed: int,
    device: str,
) -> Dict[str, Any]:
    all_candidates = [
        candidate
        for root_spec in ROOT_SPECS
        for candidate in _discover_root_candidates(system, root_spec, shortlist_horizon)
    ]
    all_candidates = [c for c in all_candidates if c.h3000_best_periodic_mode is not None]
    if not all_candidates:
        raise RuntimeError(f"No shortlist candidates discovered for {system}")

    shortlist_candidates = sorted(all_candidates, key=lambda item: item.h3000_best_periodic_mean)[:shortlist_k]
    _, shared_env, _ = _load_model(shortlist_candidates[0].run_dir, system, device)
    init_states, true_future = _make_shared_batch(
        env=shared_env,
        batch_size=batch_size,
        horizon=max(int(h) for h in horizons),
        rng_seed=rng_seed,
    )

    candidate_rows = [
        _evaluate_candidate(
            candidate,
            init_states=init_states,
            true_future=true_future,
            horizons=horizons,
            device=device,
            shortlist_rank_h3000=rank,
        )
        for rank, candidate in enumerate(shortlist_candidates, start=1)
    ]
    _rank_candidates(candidate_rows, horizons)

    winners: Dict[str, Dict[str, Any]] = {}
    for horizon in sorted(int(h) for h in horizons):
        winners[str(horizon)] = min(candidate_rows, key=lambda row: _selection_key(row, horizon))

    return {
        "system": system,
        "shortlist_size": len(shortlist_candidates),
        "candidate_rows": candidate_rows,
        "winners": winners,
        "family_best_by_horizon": _family_best_by_horizon(candidate_rows, horizons),
    }


def _flatten_candidate_rows(system_results: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for system_result in system_results:
        rows.extend(system_result["candidate_rows"])
    return rows


def _selected_rows(system_results: Sequence[Dict[str, Any]], horizons: Sequence[int]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for system_result in system_results:
        system = str(system_result["system"])
        for horizon in sorted(int(h) for h in horizons):
            winner = dict(system_result["winners"][str(horizon)])
            winner["selection_horizon"] = horizon
            winner["selection_system"] = system
            rows.append(winner)
    return rows


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return float(median(finite))


def _summarize_horizon(system_results: Sequence[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    winner_rows = [result["winners"][str(horizon)] for result in system_results]
    winner_metrics = [_safe_float(row.get(f"shared_batch_h{horizon}_best_periodic_mean")) for row in winner_rows]
    winner_root_counts = Counter(str(row["root_label"]) for row in winner_rows)
    winner_mode_counts = Counter(str(row["periodic_mode"]) for row in winner_rows)

    family_rows: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for result in system_results:
        family_best = result["family_best_by_horizon"][str(horizon)]
        for root_label, row in family_best.items():
            family_rows[root_label].append(row)

    family_summary: Dict[str, Any] = {}
    for root_label, rows in sorted(family_rows.items()):
        metrics = [_safe_float(row.get(f"shared_batch_h{horizon}_best_periodic_mean")) for row in rows]
        family_summary[root_label] = {
            "systems_covered": len(rows),
            "systems_won": int(winner_root_counts.get(root_label, 0)),
            "median_family_best_mean": _median(metrics),
            "max_family_best_mean": max((m for m in metrics if m is not None), default=None),
            "min_family_best_mean": min((m for m in metrics if m is not None), default=None),
        }

    return {
        "n_systems": len(winner_rows),
        "selected_median_mean": _median(winner_metrics),
        "selected_min_mean": min((m for m in winner_metrics if m is not None), default=None),
        "selected_max_mean": max((m for m in winner_metrics if m is not None), default=None),
        "winner_root_counts": dict(sorted(winner_root_counts.items())),
        "winner_mode_counts": dict(sorted(winner_mode_counts.items(), key=lambda item: int(item[0]))),
        "family_summary": family_summary,
        "winner_rows": [
            {
                "system": row["system"],
                "root_label": row["root_label"],
                "root_display_name": row["root_display_name"],
                "seed": row["seed"],
                "periodic_mode": row["periodic_mode"],
                "h3000_best_periodic_mean": row["h3000_best_periodic_mean"],
                f"shared_batch_h{horizon}_best_periodic_mean": row[f"shared_batch_h{horizon}_best_periodic_mean"],
                f"shared_batch_h{horizon}_best_periodic_std": row[f"shared_batch_h{horizon}_best_periodic_std"],
                "run_dir": row["run_dir"],
            }
            for row in sorted(
                winner_rows,
                key=lambda item: _selection_key(item, horizon),
            )
        ],
    }


def _summarize_switches(system_results: Sequence[Dict[str, Any]], horizons: Sequence[int]) -> Dict[str, Any]:
    if len(horizons) != 2:
        return {}
    h1, h2 = sorted(int(h) for h in horizons)
    same_root = 0
    same_run = 0
    switched_root_rows: list[Dict[str, Any]] = []
    switched_run_rows: list[Dict[str, Any]] = []

    for result in system_results:
        winner1 = result["winners"][str(h1)]
        winner2 = result["winners"][str(h2)]
        same_root_flag = str(winner1["root_label"]) == str(winner2["root_label"])
        same_run_flag = str(winner1["run_dir"]) == str(winner2["run_dir"])
        if same_root_flag:
            same_root += 1
        else:
            switched_root_rows.append(
                {
                    "system": result["system"],
                    "horizon_a_root": winner1["root_label"],
                    "horizon_b_root": winner2["root_label"],
                    f"h{h1}_mean": winner1[f"shared_batch_h{h1}_best_periodic_mean"],
                    f"h{h2}_mean": winner2[f"shared_batch_h{h2}_best_periodic_mean"],
                }
            )
        if same_run_flag:
            same_run += 1
        else:
            switched_run_rows.append(
                {
                    "system": result["system"],
                    "horizon_a_root": winner1["root_label"],
                    "horizon_a_seed": winner1["seed"],
                    "horizon_b_root": winner2["root_label"],
                    "horizon_b_seed": winner2["seed"],
                    "horizon_a_run_dir": winner1["run_dir"],
                    "horizon_b_run_dir": winner2["run_dir"],
                }
            )

    return {
        "pair": [h1, h2],
        "same_root_count": same_root,
        "same_run_count": same_run,
        "switched_root_rows": switched_root_rows,
        "switched_run_rows": switched_run_rows,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(
    path: Path,
    *,
    summary: Dict[str, Any],
    horizons: Sequence[int],
) -> None:
    lines: list[str] = []
    protocol = summary["protocol"]

    lines.append("# Dysts Long-Horizon Shortlist Summary")
    lines.append("")
    lines.append(
        f"- Systems: {protocol['n_systems']} Dysts systems"
    )
    lines.append(
        f"- Roots: {', '.join(protocol['root_labels'])}"
    )
    lines.append(
        f"- Shortlist protocol: top {protocol['shortlist_k']} checkpoints per system ranked by saved "
        f"H{protocol['shortlist_horizon']} best-periodic mean"
    )
    lines.append(
        f"- Rescored checkpoints: {protocol['n_candidate_rows']} "
        f"({protocol['n_systems']} systems x shortlist size up to {protocol['shortlist_k']})"
    )
    lines.append(
        f"- Shared rollout batch: {protocol['batch_size']} initial conditions per system, rng seed {protocol['rng_seed']}"
    )
    lines.append(
        f"- Horizons extracted from one shared rollout to H{max(int(h) for h in horizons)}: "
        + ", ".join(f"H{int(h)}" for h in sorted(int(h) for h in horizons))
    )
    lines.append("")

    for horizon in sorted(int(h) for h in horizons):
        horizon_summary = summary["horizon_summary"][str(horizon)]
        lines.append(f"## H{horizon}")
        lines.append("")
        lines.append(
            f"- Selected-winner median shared-batch H{horizon}: "
            f"{_fmt(_safe_float(horizon_summary['selected_median_mean']))}"
        )
        lines.append(
            f"- Selected-winner range: "
            f"{_fmt(_safe_float(horizon_summary['selected_min_mean']))} to "
            f"{_fmt(_safe_float(horizon_summary['selected_max_mean']))}"
        )
        lines.append(
            "- Winner counts by root: "
            + ", ".join(
                f"{root}={count}"
                for root, count in horizon_summary["winner_root_counts"].items()
            )
        )
        lines.append(
            "- Winner periodic modes: "
            + ", ".join(
                f"{mode}={count}"
                for mode, count in horizon_summary["winner_mode_counts"].items()
            )
        )
        lines.append("")
        lines.append(
            f"| root | systems won | median family-best H{horizon} | min family-best | max family-best |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for root_label, root_summary in horizon_summary["family_summary"].items():
            lines.append(
                f"| {root_label} | {root_summary['systems_won']} | "
                f"{_fmt(_safe_float(root_summary['median_family_best_mean']))} | "
                f"{_fmt(_safe_float(root_summary['min_family_best_mean']))} | "
                f"{_fmt(_safe_float(root_summary['max_family_best_mean']))} |"
            )
        lines.append("")
        lines.append(
            f"| system | selected root | seed | periodic mode | H{horizon} | H3000 shortlist score |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in horizon_summary["winner_rows"]:
            lines.append(
                f"| {row['system']} | {row['root_label']} | {row['seed']} | {row['periodic_mode']} | "
                f"{_fmt(_safe_float(row.get(f'shared_batch_h{horizon}_best_periodic_mean')))} | "
                f"{_fmt(_safe_float(row.get('h3000_best_periodic_mean')))} |"
            )
        lines.append("")

    switch_summary = summary.get("switch_summary", {})
    if switch_summary:
        h1, h2 = switch_summary["pair"]
        lines.append(f"## H{h1} To H{h2} Switches")
        lines.append("")
        lines.append(
            f"- Same winning root on both horizons: {switch_summary['same_root_count']}/{protocol['n_systems']}"
        )
        lines.append(
            f"- Same exact run on both horizons: {switch_summary['same_run_count']}/{protocol['n_systems']}"
        )
        lines.append("")
        if switch_summary["switched_root_rows"]:
            lines.append(f"| system | H{h1} root | H{h2} root | H{h1} | H{h2} |")
            lines.append("|---|---|---|---:|---:|")
            for row in switch_summary["switched_root_rows"]:
                lines.append(
                    f"| {row['system']} | {row['horizon_a_root']} | {row['horizon_b_root']} | "
                    f"{_fmt(_safe_float(row.get(f'h{h1}_mean')))} | "
                    f"{_fmt(_safe_float(row.get(f'h{h2}_mean')))} |"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for summary artifacts.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DYSTS_SYSTEMS),
        help="Systems to evaluate.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[10000, 20000],
        help="Forecast horizons to summarize.",
    )
    parser.add_argument("--shortlist-horizon", type=int, default=3000)
    parser.add_argument("--shortlist-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=20260414)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _build_summary(
    *,
    system_results: Sequence[Dict[str, Any]],
    systems: Sequence[str],
    horizons: Sequence[int],
    shortlist_horizon: int,
    shortlist_k: int,
    batch_size: int,
    rng_seed: int,
) -> Dict[str, Any]:
    candidate_rows = _flatten_candidate_rows(system_results)
    selected_rows = _selected_rows(system_results, horizons)
    return {
        "protocol": {
            "systems": list(systems),
            "n_systems": len(systems),
            "root_labels": [root.label for root in ROOT_SPECS],
            "root_display_names": {root.label: root.display_name for root in ROOT_SPECS},
            "shortlist_horizon": int(shortlist_horizon),
            "shortlist_k": int(shortlist_k),
            "batch_size": int(batch_size),
            "rng_seed": int(rng_seed),
            "horizons": horizons,
            "n_candidate_rows": len(candidate_rows),
        },
        "horizon_summary": {
            str(horizon): _summarize_horizon(system_results, horizon)
            for horizon in horizons
        },
        "switch_summary": _summarize_switches(system_results, horizons),
        "system_results": system_results,
    }


def _write_outputs(
    *,
    output_dir: Path,
    summary: Dict[str, Any],
    horizons: Sequence[int],
) -> None:
    candidate_rows = _flatten_candidate_rows(summary["system_results"])
    selected_rows = _selected_rows(summary["system_results"], horizons)

    candidate_json = output_dir / "dysts_long_horizon_candidate_rows.json"
    candidate_csv = output_dir / "dysts_long_horizon_candidate_rows.csv"
    selected_json = output_dir / "dysts_long_horizon_selected_rows.json"
    selected_csv = output_dir / "dysts_long_horizon_selected_rows.csv"
    summary_json = output_dir / "dysts_long_horizon_summary.json"
    summary_md = output_dir / "dysts_long_horizon_summary.md"

    candidate_json.write_text(json.dumps(candidate_rows, indent=2) + "\n", encoding="utf-8")
    selected_json.write_text(json.dumps(selected_rows, indent=2) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(candidate_csv, candidate_rows)
    _write_csv(selected_csv, selected_rows)
    _write_markdown(summary_md, summary=summary, horizons=horizons)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizons = sorted({int(h) for h in args.horizons})

    system_results: list[Dict[str, Any]] = []
    total_systems = len(args.systems)
    for idx, system in enumerate(args.systems, start=1):
        system_results.append(
            _collect_system_result(
                system=system,
                shortlist_horizon=args.shortlist_horizon,
                shortlist_k=args.shortlist_k,
                batch_size=args.batch_size,
                horizons=horizons,
                rng_seed=args.rng_seed,
                device=args.device,
            )
        )
        summary = _build_summary(
            system_results=system_results,
            systems=args.systems,
            horizons=horizons,
            shortlist_horizon=args.shortlist_horizon,
            shortlist_k=args.shortlist_k,
            batch_size=args.batch_size,
            rng_seed=args.rng_seed,
        )
        _write_outputs(output_dir=args.output_dir, summary=summary, horizons=horizons)
        print(f"[{idx}/{total_systems}] completed {system}", flush=True)

    candidate_rows = _flatten_candidate_rows(system_results)
    selected_rows = _selected_rows(system_results, horizons)
    summary = _build_summary(
        system_results=system_results,
        systems=args.systems,
        horizons=horizons,
        shortlist_horizon=args.shortlist_horizon,
        shortlist_k=args.shortlist_k,
        batch_size=args.batch_size,
        rng_seed=args.rng_seed,
    )
    _write_outputs(output_dir=args.output_dir, summary=summary, horizons=horizons)

    print(f"Evaluated {len(candidate_rows)} shortlisted checkpoints across {len(args.systems)} systems.")
    print(f"Wrote: {args.output_dir / 'dysts_long_horizon_candidate_rows.json'}")
    print(f"Wrote: {args.output_dir / 'dysts_long_horizon_candidate_rows.csv'}")
    print(f"Wrote: {args.output_dir / 'dysts_long_horizon_selected_rows.json'}")
    print(f"Wrote: {args.output_dir / 'dysts_long_horizon_selected_rows.csv'}")
    print(f"Wrote: {args.output_dir / 'dysts_long_horizon_summary.json'}")
    print(f"Wrote: {args.output_dir / 'dysts_long_horizon_summary.md'}")


if __name__ == "__main__":
    main()
