#!/usr/bin/env python3
"""Render Dysts long-horizon phase portraits for LISTA-BD vs Dense MLP.

This uses the compact rollout artifacts produced by
``tools/evaluate_dysts_long_horizon_run.py``. Each output figure contains one
system, with rows for H30000/H40000/H50000/H60000 and columns for the selected
model roots.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch


DEFAULT_CSV = Path(
    "results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428/"
    "collect/forecasting_rows.csv"
)

DEFAULT_SYSTEMS: tuple[str, ...] = (
    "dysts:Chua",
    "dysts:Dadras",
    "dysts:DequanLi",
    "dysts:Hadley",
    "dysts:LorenzCoupled",
    "dysts:LuChenCheng",
    "dysts:MultiChua",
    "dysts:QiChen",
    "dysts:Sakarya",
    "dysts:SanUmSrisuchinwong",
    "dysts:ShimizuMorioka",
    "dysts:WangSun",
)

DEFAULT_ROOTS: tuple[str, ...] = (
    "lista_blockdiag_ns200k_denseopt_sc6em3",
    "generic_sparse_sc0_ns200k_best",
)

DISPLAY_NAMES: dict[str, str] = {
    "lista_blockdiag_ns200k_denseopt_sc6em3": "LISTA-BD",
    "generic_sparse_sc0_ns200k_best": "Dense MLP",
    "lista_dense_promoted_stage4": "LISTA-D",
    "lista": "LISTA",
    "lista_bd": "LISTA-BD",
    "lista_sb": "LISTA-SB",
    "sparse_mlp": "Sparse MLP",
    "sparse_mlp_bd": "Sparse MLP-BD",
    "dense_mlp_tanh": "Dense MLP",
}

COLORS: dict[str, str] = {
    "lista_blockdiag_ns200k_denseopt_sc6em3": "#cc79a7",
    "generic_sparse_sc0_ns200k_best": "#e69f00",
    "lista_dense_promoted_stage4": "#009e73",
    "lista": "#0072b2",
    "lista_bd": "#cc79a7",
    "lista_sb": "#d55e00",
    "sparse_mlp": "#009e73",
    "sparse_mlp_bd": "#56b4e9",
    "dense_mlp_tanh": "#000000",
}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, int)):
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


def _read_rows(path: Path) -> list[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _row_key(row: Dict[str, str]) -> tuple[str, str, int]:
    return (row["system_key"], row["root_label"], int(row["seed"]))


def _metric(row: Dict[str, str], horizon: int) -> Optional[float]:
    return _safe_float(row.get(f"h{int(horizon)}_best_periodic_mean"))


def _mode(row: Dict[str, str], horizon: int) -> str:
    value = str(row.get(f"h{int(horizon)}_best_periodic_mode") or "").strip()
    if not value:
        raise RuntimeError(
            f"Missing best-periodic mode for {row.get('system_key')} "
            f"{row.get('root_label')} H{horizon}"
        )
    return value


def _artifact_path(row: Dict[str, str]) -> Path:
    raw = str(row.get("selected_rollout_artifacts") or "").strip()
    if not raw:
        raise RuntimeError(f"Missing selected_rollout_artifacts for row: {row}")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_payload(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}")
    return payload


def _truth_xy(payload: Dict[str, Any], horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    init_states = payload.get("init_states")
    true_future = payload.get("true_future")
    if not isinstance(init_states, torch.Tensor) or not isinstance(true_future, torch.Tensor):
        raise TypeError("Rollout payload is missing init_states or true_future tensors")
    if true_future.shape[0] < horizon:
        raise ValueError(f"true_future is too short for H{horizon}: {tuple(true_future.shape)}")
    return init_states[:, :2].float(), true_future[:horizon, :, :2].float()


def _prediction_xy(payload: Dict[str, Any], mode_name: str, horizon: int) -> torch.Tensor:
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict) or mode_name not in predictions:
        raise KeyError(f"Prediction mode '{mode_name}' missing from artifact")
    pred = predictions[mode_name]
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"Prediction mode '{mode_name}' is not a tensor")
    if pred.shape[0] < horizon:
        raise ValueError(f"Prediction mode '{mode_name}' too short for H{horizon}: {tuple(pred.shape)}")
    return pred[:horizon, :, :2].float()


def _valid_lengths(pred_xy: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(pred_xy).all(dim=-1)
    first_bad = (~finite).float().argmax(dim=0)
    all_good = finite.all(dim=0)
    lengths = torch.where(all_good, torch.full_like(first_bad, pred_xy.shape[0]), first_bad)
    return lengths.to(torch.long)


def _axis_limits(
    *,
    init_xy: torch.Tensor,
    true_xy: torch.Tensor,
    pred_xys: Sequence[torch.Tensor],
    max_points: int = 250_000,
) -> tuple[float, float, float, float]:
    points = [init_xy, true_xy.reshape(-1, 2)]
    for pred_xy in pred_xys:
        finite = pred_xy[torch.isfinite(pred_xy).all(dim=-1)]
        if finite.numel() > 0:
            points.append(finite.reshape(-1, 2))
    xy = torch.cat(points, dim=0).to(torch.float64)
    xy = xy[torch.isfinite(xy).all(dim=-1)]
    if xy.numel() == 0:
        raise RuntimeError("No finite points available for axis limits")
    if xy.shape[0] > max_points:
        step = max(1, xy.shape[0] // max_points)
        xy = xy[::step][:max_points]
    lo = torch.quantile(xy, 0.005, dim=0)
    hi = torch.quantile(xy, 0.995, dim=0)
    truth = torch.cat([init_xy, true_xy.reshape(-1, 2)], dim=0).to(torch.float64)
    truth = truth[torch.isfinite(truth).all(dim=-1)]
    if not torch.isfinite(lo).all() or not torch.isfinite(hi).all():
        lo = truth.min(dim=0).values
        hi = truth.max(dim=0).values
    span = (hi - lo).clamp_min(1e-6)
    pad = 0.06 * span
    lo = lo - pad
    hi = hi + pad
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _thin_trace(xy: torch.Tensor, max_points: int = 4000) -> torch.Tensor:
    if xy.shape[0] <= max_points:
        return xy
    step = max(1, int(math.ceil(xy.shape[0] / float(max_points))))
    thinned = xy[::step]
    if not torch.equal(thinned[-1], xy[-1]):
        thinned = torch.cat([thinned, xy[-1:]], dim=0)
    return thinned


def _plot_panel(
    ax,
    *,
    init_xy: torch.Tensor,
    true_xy: torch.Tensor,
    pred_xy: torch.Tensor,
    root_label: str,
    mode_name: str,
    mse: Optional[float],
    horizon: int,
    max_traces: int,
    axis_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    valid_lengths = _valid_lengths(pred_xy)
    valid_indices = torch.nonzero(valid_lengths > 1, as_tuple=False).flatten()
    if valid_indices.numel() > 0:
        order = torch.argsort(valid_lengths[valid_indices], descending=True)
        valid_indices = valid_indices[order][:max_traces]

    trace_count = int(valid_indices.numel())
    full_trace_fraction = float((valid_lengths >= horizon).float().mean().item())
    color = COLORS.get(root_label, "#d55e00")

    for idx in valid_indices.tolist():
        length = int(valid_lengths[idx].item())
        gt = _thin_trace(torch.cat([init_xy[idx : idx + 1], true_xy[:, idx]], dim=0)).cpu().numpy()
        pred = _thin_trace(torch.cat([init_xy[idx : idx + 1], pred_xy[:length, idx]], dim=0)).cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], color="#7f7f7f", alpha=0.11, linewidth=0.85)
        ax.plot(pred[:, 0], pred[:, 1], color=color, alpha=0.17, linewidth=0.9)

    x0, x1, y0, y1 = axis_bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.32)
    ax.text(
        0.03,
        0.97,
        f"{mode_name}\nMSE {mse:.3g}" if mse is not None else f"{mode_name}\nMSE NA",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 2.0},
    )
    ax.text(
        0.03,
        0.04,
        f"full finite {full_trace_fraction:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
    )
    return {
        "mode": mode_name,
        "mse": mse,
        "trace_count_plotted": trace_count,
        "full_trace_fraction": full_trace_fraction,
    }


def _render_system(
    *,
    system: str,
    rows: Sequence[Dict[str, str]],
    roots: Sequence[str],
    horizons: Sequence[int],
    out_dir: Path,
    max_traces: int,
    file_stem_suffix: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    max_horizon = max(int(h) for h in horizons)
    loaded: dict[str, dict[str, Any]] = {}
    pred_for_limits: list[torch.Tensor] = []
    init_xy: Optional[torch.Tensor] = None
    true_xy_full: Optional[torch.Tensor] = None

    for row in rows:
        root = row["root_label"]
        payload = _load_payload(_artifact_path(row))
        row_init_xy, row_true_xy = _truth_xy(payload, max_horizon)
        if init_xy is None:
            init_xy = row_init_xy
            true_xy_full = row_true_xy
        for horizon in horizons:
            mode_name = _mode(row, int(horizon))
            pred_for_limits.append(_prediction_xy(payload, mode_name, int(horizon)))
        loaded[root] = {"row": row, "payload": payload}

    if init_xy is None or true_xy_full is None:
        raise RuntimeError(f"No rollout payloads loaded for {system}")
    axis_bounds = _axis_limits(
        init_xy=init_xy,
        true_xy=true_xy_full,
        pred_xys=pred_for_limits,
    )

    fig, axes = plt.subplots(
        len(horizons),
        len(roots),
        figsize=(3.4 * len(roots), 3.0 * len(horizons)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{system} phase portraits, seed {rows[0]['seed']}", fontsize=13)

    panel_meta: dict[str, dict[str, Any]] = {}
    for row_idx, horizon in enumerate(horizons):
        for col_idx, root in enumerate(roots):
            ax = axes[row_idx, col_idx]
            data = loaded[root]
            row = data["row"]
            mode_name = _mode(row, int(horizon))
            pred_xy = _prediction_xy(data["payload"], mode_name, int(horizon))
            true_xy = true_xy_full[: int(horizon)]
            panel = _plot_panel(
                ax,
                init_xy=init_xy,
                true_xy=true_xy,
                pred_xy=pred_xy,
                root_label=root,
                mode_name=mode_name,
                mse=_metric(row, int(horizon)),
                horizon=int(horizon),
                max_traces=max_traces,
                axis_bounds=axis_bounds,
            )
            if row_idx == 0:
                ax.set_title(DISPLAY_NAMES.get(root, row["root_display_name"]), fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"H{int(horizon)}\n$x_2$")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            if row_idx == len(horizons) - 1:
                ax.set_xlabel("$x_1$")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            panel_meta[f"{root}:H{int(horizon)}"] = panel

    handles = [
        Line2D([0], [0], color="#7f7f7f", linewidth=2.0, label="Ground truth"),
        Line2D([0], [0], color="#d55e00", linewidth=2.0, label="Forecast"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)

    slug = _system_slug(system)
    horizon_tag = f"h{min(int(h) for h in horizons)}_h{max(int(h) for h in horizons)}"
    suffix = re.sub(r"[^a-zA-Z0-9_+-]+", "_", file_stem_suffix).strip("_")
    out_png = out_dir / f"{slug}_seed{rows[0]['seed']}_{horizon_tag}_{suffix}.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    root_meta = {}
    for root, data in loaded.items():
        row = data["row"]
        root_meta[root] = {
            "display": DISPLAY_NAMES.get(root, row["root_display_name"]),
            "run_dir": row["run_dir"],
            "selected_rollout_artifacts": row["selected_rollout_artifacts"],
            "status": row["status"],
        }
        del data["payload"]
    gc.collect()

    return {
        "system": system,
        "seed": int(rows[0]["seed"]),
        "png": str(out_png),
        "pdf": str(out_pdf),
        "axis_bounds": axis_bounds,
        "roots": root_meta,
        "panels": panel_meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasting-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures/dysts_phase_portraits_seed0_h30k_h60k_lista_bd_vs_dense_mlp_20260430"),
    )
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS))
    parser.add_argument("--root-labels", nargs="+", default=list(DEFAULT_ROOTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizons", nargs="+", type=int, default=[30000, 40000, 50000, 60000])
    parser.add_argument("--max-traces", type=int, default=80)
    parser.add_argument("--file-stem-suffix", default="lista_bd_vs_dense_mlp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = tuple(str(root) for root in args.root_labels)
    horizons = tuple(sorted({int(h) for h in args.horizons}))
    seed = int(args.seed)

    rows = _read_rows(args.forecasting_csv)
    by_key = {_row_key(row): row for row in rows}
    manifest_rows = []
    for system in args.systems:
        system_rows = []
        for root in roots:
            key = (system, root, seed)
            if key not in by_key:
                raise RuntimeError(f"No collector row for {key}")
            row = by_key[key]
            if row.get("status") != "complete":
                raise RuntimeError(f"Collector row is not complete for {key}: {row.get('status')}")
            _artifact_path(row)
            for horizon in horizons:
                _mode(row, horizon)
                if _metric(row, horizon) is None:
                    raise RuntimeError(f"Missing H{horizon} metric for {key}")
            system_rows.append(row)

        result = _render_system(
            system=system,
            rows=system_rows,
            roots=roots,
            horizons=horizons,
            out_dir=args.out_dir,
            max_traces=int(args.max_traces),
            file_stem_suffix=str(args.file_stem_suffix),
        )
        manifest_rows.append(result)
        print(f"{system}: saved {result['png']}")

    manifest = {
        "forecasting_csv": str(args.forecasting_csv),
        "seed": seed,
        "horizons": list(horizons),
        "root_labels": list(roots),
        "file_stem_suffix": str(args.file_stem_suffix),
        "rows": manifest_rows,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
