#!/usr/bin/env python3
"""Generate Dysts phase portraits from the true best long-horizon roots.

Selection protocol:
1. Read one or more Dysts long-horizon collector CSVs.
2. Keep complete rows with an available compact selected-rollout artifact.
3. For each system, choose the row with the lowest H{horizon} best-periodic MSE.
4. Load that row's stored rollout artifact and render a phase portrait using the
   exact periodic mode that achieved the recorded best-periodic score.

Outputs:
- one PNG and one PDF per Dysts system
- one metadata JSON per Dysts system
- one manifest JSON summarizing all per-system selections
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch


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


DEFAULT_FORECASTING_CSVS: tuple[Path, ...] = (
    Path("results/dysts_long_horizon_eval_20260414/collect/forecasting_rows.csv"),
    Path("results/dysts_long_horizon_eval_mlp_blockdiag_20260415/collect/forecasting_rows.csv"),
)


def _safe_float(value: Any) -> Optional[float]:
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


def _system_slug(system: str) -> str:
    name = system.split(":", 1)[1] if system.startswith("dysts:") else system
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.replace("-", "_").lower()


def _horizon_tag(horizon: int) -> str:
    return f"h{int(horizon)}"


def _read_rows(path: Path) -> list[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metric_key(row: Dict[str, str], horizon: int) -> Optional[float]:
    return _safe_float(row.get(f"h{int(horizon)}_best_periodic_mean"))


def _mode_key(row: Dict[str, str], horizon: int) -> Optional[str]:
    mode = row.get(f"h{int(horizon)}_best_periodic_mode")
    if not isinstance(mode, str):
        return None
    mode = mode.strip()
    return mode or None


def _selected_rollout_path(row: Dict[str, str]) -> Optional[Path]:
    raw = row.get("selected_rollout_artifacts")
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    return Path(raw)


def _valid_candidate(row: Dict[str, str], horizon: int) -> bool:
    if row.get("status") != "complete":
        return False
    if not row.get("system_key"):
        return False
    if _metric_key(row, horizon) is None:
        return False
    if _mode_key(row, horizon) is None:
        return False
    selected_rollout = _selected_rollout_path(row)
    return selected_rollout is not None and selected_rollout.exists()


def _selection_sort_key(row: Dict[str, str], horizon: int) -> tuple[float, str, int, str]:
    metric = _metric_key(row, horizon)
    if metric is None:
        metric = float("inf")
    seed_raw = row.get("seed")
    try:
        seed = int(seed_raw) if seed_raw is not None else 10**9
    except ValueError:
        seed = 10**9
    return (
        metric,
        row.get("root_label", ""),
        seed,
        row.get("run_dir", ""),
    )


def _load_rollout_payload(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Rollout payload is not a dict: {path}")
    return payload


def _extract_prediction(payload: Dict[str, Any], mode_name: str, horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = payload.get("predictions", {})
    if not isinstance(predictions, dict) or mode_name not in predictions:
        raise KeyError(f"Mode '{mode_name}' missing from rollout payload")

    init_states = payload.get("init_states")
    true_future = payload.get("true_future")
    pred_future = predictions[mode_name]
    if not isinstance(init_states, torch.Tensor):
        raise TypeError("Missing tensor init_states in rollout payload")
    if not isinstance(true_future, torch.Tensor):
        raise TypeError("Missing tensor true_future in rollout payload")
    if not isinstance(pred_future, torch.Tensor):
        raise TypeError(f"Prediction for mode '{mode_name}' is not a tensor")
    if true_future.shape[0] < horizon or pred_future.shape[0] < horizon:
        raise ValueError(
            f"Rollout payload is too short for H{horizon}: "
            f"true={true_future.shape[0]}, pred={pred_future.shape[0]}"
        )
    return init_states, true_future[:horizon], pred_future[:horizon]


def _axis_limits(true_future: torch.Tensor, pred_future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    true_xy = true_future[:, :, :2].reshape(-1, 2)
    true_xy = true_xy[torch.isfinite(true_xy).all(dim=1)]
    pred_xy = pred_future[:, :, :2].reshape(-1, 2)
    pred_xy = pred_xy[torch.isfinite(pred_xy).all(dim=1)]
    xy = torch.cat([true_xy, pred_xy], dim=0) if pred_xy.numel() else true_xy
    if xy.numel() == 0:
        raise RuntimeError("No finite XY points available for axis-limit estimation")
    xy = xy.to(torch.float64)
    mins = torch.quantile(xy, 0.005, dim=0)
    maxs = torch.quantile(xy, 0.995, dim=0)
    if not torch.isfinite(mins).all() or not torch.isfinite(maxs).all():
        mins = true_xy.to(torch.float64).min(dim=0).values
        maxs = true_xy.to(torch.float64).max(dim=0).values
    span = (maxs - mins).clamp_min(1e-6)
    pad = 0.06 * span
    lo = mins - pad
    hi = maxs + pad
    if not torch.isfinite(lo).all() or not torch.isfinite(hi).all():
        raise RuntimeError("Axis-limit estimation produced non-finite bounds")
    return lo, hi


def _trace_valid_lengths(pred_future: torch.Tensor) -> torch.Tensor:
    pred_finite = torch.isfinite(pred_future[:, :, :2]).all(dim=-1)
    return pred_finite.sum(dim=0)


def _render_phase_portrait(
    *,
    system: str,
    root_display_name: str,
    seed: str,
    periodic_mode: str,
    init_states: torch.Tensor,
    true_future: torch.Tensor,
    pred_future: torch.Tensor,
    horizon: int,
    output_png: Path,
    max_traces: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_png.parent.mkdir(parents=True, exist_ok=True)

    valid_lengths = _trace_valid_lengths(pred_future)
    valid_indices = torch.nonzero(valid_lengths > 0, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise RuntimeError(f"No finite traces available for {system}")
    order = torch.argsort(valid_lengths[valid_indices], descending=True)
    valid_indices = valid_indices[order]
    if valid_indices.numel() > max_traces:
        valid_indices = valid_indices[:max_traces]

    lo, hi = _axis_limits(true_future, pred_future)

    fig, ax = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    for idx in valid_indices.tolist():
        valid_len = int(valid_lengths[idx].item())
        if valid_len <= 0:
            continue
        gt_xy = torch.cat([init_states[idx : idx + 1, :2], true_future[:, idx, :2]], dim=0).cpu().numpy()
        pred_xy = torch.cat(
            [init_states[idx : idx + 1, :2], pred_future[:valid_len, idx, :2]],
            dim=0,
        ).cpu().numpy()
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#9a9a9a", alpha=0.12, linewidth=1.1, zorder=2)
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#d55e00", alpha=0.12, linewidth=1.1, zorder=3)

    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"{system.split(':', 1)[1]} ({int(horizon)}-step forecast)")
    ax.text(
        0.03,
        0.97,
        f"{root_display_name}, seed {seed}, {periodic_mode}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2.8},
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#9a9a9a", linewidth=1.8, label="Ground truth"),
            Line2D([0], [0], color="#d55e00", linewidth=1.8, label="Forecast"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.92,
    )

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _selection_basis(horizon: int) -> str:
    return (
        f"lowest H{int(horizon)} best-periodic forecasting MSE across all completed "
        "checked-in Dysts long-horizon roots, seeds, and periodic re-encoding cadences"
    )


def _top_candidates(rows: Iterable[Dict[str, str]], horizon: int, limit: int) -> list[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: _selection_sort_key(row, horizon))
    out = []
    for row in ranked[:limit]:
        out.append(
            {
                "root_label": row.get("root_label"),
                "root_display_name": row.get("root_display_name"),
                "model_family": row.get("model_family"),
                "seed": int(row["seed"]) if row.get("seed") not in (None, "") else None,
                f"h{int(horizon)}_best_periodic_mean": _metric_key(row, horizon),
                f"h{int(horizon)}_best_periodic_mode": _mode_key(row, horizon),
                "run_dir": row.get("run_dir"),
            }
        )
    return out


def _write_metadata(
    *,
    row: Dict[str, str],
    horizon: int,
    collector_csvs: Sequence[Path],
    system_rows: Sequence[Dict[str, str]],
    top_candidate_limit: int,
    output_json: Path,
    output_png: Path,
) -> None:
    horizon_tag = _horizon_tag(horizon)
    payload = {
        "system": row["system_key"],
        "selection_basis": _selection_basis(horizon),
        "collector_csvs": [str(path) for path in collector_csvs],
        "root_label": row["root_label"],
        "root_display_name": row["root_display_name"],
        "model_family": row["model_family"],
        "seed": int(row["seed"]),
        "run_dir": row["run_dir"],
        "reeval_results_json": row["reeval_results_json"],
        "selected_rollout_artifacts": row["selected_rollout_artifacts"],
        "status": row["status"],
        "horizon": int(horizon),
        f"{horizon_tag}_best_periodic_mean": _metric_key(row, horizon),
        f"{horizon_tag}_best_periodic_mode": _mode_key(row, horizon),
        "num_candidates_considered": len(system_rows),
        "top_candidates": _top_candidates(system_rows, horizon, top_candidate_limit),
        "outputs": [
            str(output_png),
            str(output_png.with_suffix(".pdf")),
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasting-csv",
        nargs="+",
        type=Path,
        default=list(DEFAULT_FORECASTING_CSVS),
        help="Collector forecasting_rows.csv files to merge before selection.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures/dysts_phase_portraits"),
        help="Directory for per-system figure packets.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DYSTS_SYSTEMS),
        help="Systems to render (default: all 15 Dysts systems).",
    )
    parser.add_argument("--horizon", type=int, default=30000)
    parser.add_argument("--max-traces", type=int, default=100)
    parser.add_argument(
        "--top-candidate-limit",
        type=int,
        default=10,
        help="Number of top-ranked candidates to store in each metadata JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(args.horizon)
    horizon_tag = _horizon_tag(horizon)

    merged_rows: list[Dict[str, str]] = []
    for csv_path in args.forecasting_csv:
        merged_rows.extend(_read_rows(csv_path))

    valid_rows = [row for row in merged_rows if _valid_candidate(row, horizon)]
    rows_by_system: dict[str, list[Dict[str, str]]] = defaultdict(list)
    for row in valid_rows:
        rows_by_system[row["system_key"]].append(row)

    manifest_rows = []
    winner_counts: Counter[str] = Counter()
    for system in args.systems:
        system_rows = rows_by_system.get(system, [])
        if not system_rows:
            raise RuntimeError(f"No valid H{horizon} candidates found for {system}")

        selected_row = min(system_rows, key=lambda row: _selection_sort_key(row, horizon))
        winner_counts[selected_row["root_label"]] += 1

        rollout_path = _selected_rollout_path(selected_row)
        if rollout_path is None:
            raise RuntimeError(f"Missing rollout artifact path for {system}")
        payload = _load_rollout_payload(rollout_path)
        periodic_mode = _mode_key(selected_row, horizon)
        if periodic_mode is None:
            raise RuntimeError(f"Missing H{horizon} periodic mode for {system}")

        init_states, true_future, pred_future = _extract_prediction(payload, periodic_mode, horizon)
        slug = _system_slug(system)
        output_png = args.out_dir / f"{slug}_{horizon_tag}_best_root_phase_portrait.png"
        output_json = args.out_dir / f"{slug}_{horizon_tag}_best_root_phase_portrait.json"
        _render_phase_portrait(
            system=system,
            root_display_name=selected_row["root_display_name"],
            seed=selected_row["seed"],
            periodic_mode=periodic_mode,
            init_states=init_states,
            true_future=true_future,
            pred_future=pred_future,
            horizon=horizon,
            output_png=output_png,
            max_traces=int(args.max_traces),
        )
        _write_metadata(
            row=selected_row,
            horizon=horizon,
            collector_csvs=args.forecasting_csv,
            system_rows=system_rows,
            top_candidate_limit=int(args.top_candidate_limit),
            output_json=output_json,
            output_png=output_png,
        )

        row_summary = {
            "system": system,
            "root_label": selected_row["root_label"],
            "root_display_name": selected_row["root_display_name"],
            "model_family": selected_row["model_family"],
            "seed": int(selected_row["seed"]),
            "run_dir": selected_row["run_dir"],
            "reeval_results_json": selected_row["reeval_results_json"],
            "selected_rollout_artifacts": selected_row["selected_rollout_artifacts"],
            "periodic_mode": periodic_mode,
            f"{horizon_tag}_best_periodic_mean": _metric_key(selected_row, horizon),
            "png": str(output_png),
            "pdf": str(output_png.with_suffix(".pdf")),
            "json": str(output_json),
        }
        manifest_rows.append(row_summary)
        print(
            f"{system}: {selected_row['root_display_name']} seed={selected_row['seed']} "
            f"{periodic_mode} H{horizon}={_metric_key(selected_row, horizon):.6g}"
        )

    manifest = {
        "selection_basis": _selection_basis(horizon),
        "horizon": horizon,
        "collector_csvs": [str(path) for path in args.forecasting_csv],
        "winner_counts_by_root_label": dict(sorted(winner_counts.items())),
        "rows": manifest_rows,
    }
    manifest_path = args.out_dir / f"dysts_{horizon_tag}_best_root_phase_portraits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
