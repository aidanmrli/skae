#!/usr/bin/env python3
"""Build the benchmark support-map and Dysts forecast composite figure.

The top row overlays LISTA support-family assignments on evaluation-only true
basin maps and vector fields for representative fixed-17 systems. The bottom
row shows H5000 dt x30 Dysts phase portraits from the best seed-0 primary row
among the stored all-model rollout artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    get_transition_rich_basin_count,
)
from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import make_env
from skae.model import make_model
from tools.reduce_transition_rich_interpretability_metrics import support_family_labels


ROOT_LISTA_SUPPORT = "lista_dense_softblock_signsplit_p64_hardinit_basin_partition"
DEFAULT_INTERPRETABILITY_CSV = Path(
    "results/transition_rich_basin_partition_final_seed10_20260409/"
    "interpretability_final_pass1/interpretability_rows.csv"
)
DEFAULT_DYSTS_MANIFEST = Path(
    "docs/figures/dysts_dt30_phase_portraits_seed0_h1000_h5000_all_models_20260501/"
    "manifest.json"
)
DEFAULT_OUTPUT = Path(
    "docs/figures/neurips_paper_2026/fig_benchmark_support_dysts_composite.png"
)

OKABE_ITO = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

DYSTS_FORECAST_COLOR = "#0072B2"

DYSTS_COLORS = {
    "lista": "#0072B2",
    "lista_bd": "#CC79A7",
    "lista_sb": "#D55E00",
    "sparse_mlp": "#009E73",
    "sparse_mlp_bd": "#56B4E9",
    "dense_mlp_tanh": "#000000",
}

DYSTS_DISPLAY = {
    "lista": "LISTA",
    "lista_bd": "LISTA-BD",
    "lista_sb": "LISTA-SB",
    "sparse_mlp": "Sparse MLP",
    "sparse_mlp_bd": "Sparse MLP-BD",
    "dense_mlp_tanh": "Dense MLP",
}

DYSTS_AXIS_ZOOM = {
    "dysts:Dadras": 0.93,
    "dysts:SanUmSrisuchinwong": 0.90,
}

DYSTS_AXIS_XSHIFT_FRAC = {
    "dysts:Dadras": 0.025,
}


@dataclass(frozen=True)
class SupportPanelSpec:
    system: str
    title: str
    seed: int
    xlim: tuple[float, float]
    ylim: tuple[float, float]


@dataclass(frozen=True)
class DystsPanelSpec:
    system: str
    title: str


SUPPORT_PANELS = (
    SupportPanelSpec(
        system="gated_local_linear",
        title="Local-linear gates",
        seed=0,
        xlim=(-4.0, 4.0),
        ylim=(-4.0, 4.0),
    ),
    SupportPanelSpec(
        system="claude:transition_routes_4",
        title="Transition routes",
        seed=0,
        xlim=(-3.5, 3.5),
        ylim=(-3.5, 3.5),
    ),
    SupportPanelSpec(
        system="claude:cal_square_4",
        title="Square wells",
        seed=0,
        xlim=(-3.3, 3.3),
        ylim=(-3.3, 3.3),
    ),
    SupportPanelSpec(
        system="claude:cal_high_cross_3",
        title="High cross",
        seed=0,
        xlim=(-3.3, 3.3),
        ylim=(-3.3, 3.3),
    ),
)

DYSTS_PANELS = (
    DystsPanelSpec(system="dysts:Chua", title="Chua"),
    DystsPanelSpec(system="dysts:Dadras", title="Dadras"),
    DystsPanelSpec(system="dysts:ShimizuMorioka", title="ShimizuMorioka"),
    DystsPanelSpec(system="dysts:SanUmSrisuchinwong", title="SanUmSrisuchinwong"),
)

PRIMARY_DYSTS_ROOTS = (
    "lista",
    "lista_bd",
    "sparse_mlp",
    "sparse_mlp_bd",
    "dense_mlp_tanh",
)


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _find_run_dir(
    rows: Iterable[dict[str, str]],
    *,
    root_label: str,
    system: str,
    seed: int,
) -> Path:
    for row in rows:
        if (
            row.get("root_label") == root_label
            and row.get("system_key") == system
            and int(row.get("seed", "-1")) == int(seed)
        ):
            run_dir = str(row.get("run_dir") or "").strip()
            if run_dir:
                return Path(run_dir)
    raise RuntimeError(f"No run_dir for {root_label}, {system}, seed {seed}")


def _load_model_and_env(run_dir: Path, system: str, device: str):
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
    return model, env


def _grid_states(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    points: int,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    xs = np.linspace(xlim[0], xlim[1], points)
    ys = np.linspace(ylim[0], ylim[1], points)
    xx, yy = np.meshgrid(xs, ys)
    states = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    return xx, yy, states


def _long_rollout(env, states: torch.Tensor, steps: int, chunk_size: int = 8192) -> torch.Tensor:
    outputs = []
    for chunk in states.split(chunk_size, dim=0):
        current = chunk
        for _ in range(int(steps)):
            current = env.step(current)
            current = torch.clamp(current, -1e6, 1e6)
        outputs.append(current.detach().cpu())
    return torch.cat(outputs, dim=0)


def _kmeans_centers(points: torch.Tensor, num_centers: int, num_iters: int = 30) -> torch.Tensor:
    if points.ndim != 2:
        raise ValueError("points must be two-dimensional")
    finite = torch.isfinite(points).all(dim=1)
    points = points[finite]
    if points.shape[0] < num_centers:
        raise ValueError("Need at least as many finite points as centers")
    centers = [points[0]]
    while len(centers) < num_centers:
        current = torch.stack(centers, dim=0)
        min_dists = torch.cdist(points, current).min(dim=1).values
        centers.append(points[min_dists.argmax()])
    centers_t = torch.stack(centers, dim=0).clone()
    for _ in range(num_iters):
        assignments = torch.cdist(points, centers_t).argmin(dim=1)
        updated = []
        for idx in range(num_centers):
            mask = assignments == idx
            updated.append(points[mask].mean(dim=0) if bool(mask.any()) else centers_t[idx])
        updated_t = torch.stack(updated, dim=0)
        if torch.allclose(updated_t, centers_t):
            break
        centers_t = updated_t
    return centers_t


def _known_attractor_centers(env, states: torch.Tensor, expected_count: int) -> Optional[torch.Tensor]:
    if hasattr(env, "points_2d"):
        centers = getattr(env, "points_2d")
    elif hasattr(env, "system") and (hasattr(env.system, "wells") or hasattr(env.system, "_wells")):
        wells = getattr(env.system, "wells") if hasattr(env.system, "wells") else getattr(env.system, "_wells")
        centers = torch.tensor(
            [[float(well[0]), float(well[1])] for well in wells],
            dtype=states.dtype,
        )
    elif hasattr(env, "system") and hasattr(env.system, "basins"):
        centers = torch.tensor(
            [[float(center[0]), float(center[1])] for center in getattr(env.system, "basins")],
            dtype=states.dtype,
        )
    elif hasattr(env, "system") and any(
        hasattr(env.system, attr) for attr in ("well_centers", "room_centers", "centers")
    ):
        attr = next(
            name
            for name in ("well_centers", "room_centers", "centers")
            if hasattr(env.system, name)
        )
        raw_centers = getattr(env.system, attr)
        if isinstance(raw_centers, torch.Tensor):
            centers = raw_centers.detach().cpu().to(dtype=states.dtype)
        else:
            centers = torch.tensor(
                [[float(center[0]), float(center[1])] for center in raw_centers],
                dtype=states.dtype,
            )
    else:
        return None
    if not isinstance(centers, torch.Tensor):
        centers = torch.tensor(centers, dtype=states.dtype)
    centers = centers.detach().cpu().to(dtype=states.dtype)
    if centers.ndim != 2 or centers.shape[0] < expected_count or centers.shape[1] < 2:
        return None
    return centers[:expected_count, : states.shape[1]]


def _basin_labels_for_states(
    env,
    system: str,
    states: torch.Tensor,
    *,
    endpoint_rollout_steps: int,
) -> tuple[torch.Tensor, str]:
    if hasattr(env, "basin_label"):
        labels = env.basin_label(states)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)
        return labels.detach().cpu().to(dtype=torch.long), "env.basin_label"
    basin_count = int(get_transition_rich_basin_count(system))
    centers = _known_attractor_centers(env, states, basin_count)
    if centers is not None:
        labels = torch.cdist(states.detach().cpu(), centers).argmin(dim=1).to(dtype=torch.long)
        return labels, "benchmark_attractor_centers"
    endpoints = _long_rollout(env, states, endpoint_rollout_steps)
    centers = _kmeans_centers(endpoints, basin_count)
    labels = torch.cdist(endpoints, centers).argmin(dim=1).to(dtype=torch.long)
    return labels, f"endpoint_rollout_{endpoint_rollout_steps}"


def _dynamics_for_states(env, states: torch.Tensor) -> torch.Tensor:
    if hasattr(env, "dynamics"):
        return env.dynamics(states).detach().cpu().to(dtype=torch.float32)
    if hasattr(env, "system") and hasattr(env.system, "dynamics"):
        states64 = states.to(dtype=torch.float64)
        try:
            dyn = torch.vmap(env.system.dynamics)(states64)
        except RuntimeError:
            dyn = torch.stack([env.system.dynamics(state) for state in states64], dim=0)
        return dyn.detach().cpu().to(dtype=torch.float32)
    raise RuntimeError("Environment does not expose a vector field")


def _encode_latents(model, states: torch.Tensor, device: str, chunk_size: int = 4096) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for chunk in states.split(chunk_size, dim=0):
            z = model.encode(chunk.to(device))
            chunks.append(z.detach().cpu())
    return torch.cat(chunks, dim=0).numpy()


def _topk_support_mask(latents: np.ndarray, k: int) -> np.ndarray:
    abs_latents = np.abs(latents)
    k = min(int(k), abs_latents.shape[1])
    mask = np.zeros_like(abs_latents, dtype=bool)
    indices = np.argpartition(abs_latents, -k, axis=1)[:, -k:]
    rows = np.arange(abs_latents.shape[0])[:, None]
    mask[rows, indices] = True
    return mask


def _family_to_dominant_basin(
    families: np.ndarray,
    basin_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    mapping: dict[Any, int] = {}
    for family in list(dict.fromkeys(families.tolist())):
        mask = families == family
        counts = Counter(int(item) for item in basin_labels[mask].tolist() if int(item) >= 0)
        mapping[family] = counts.most_common(1)[0][0] if counts else -1
    mapped = np.asarray([mapping[item] for item in families.tolist()], dtype=np.int64)
    serializable = {str(key): int(value) for key, value in mapping.items()}
    return mapped, serializable


def _render_support_panel(
    ax,
    *,
    spec: SupportPanelSpec,
    rows: list[dict[str, str]],
    device: str,
    grid_points: int,
    vector_points: int,
    endpoint_rollout_steps: int,
    family_jaccard: float,
    topk: int,
) -> dict[str, Any]:
    print(f"Rendering support panel: {spec.system}", flush=True)
    run_dir = _find_run_dir(
        rows,
        root_label=ROOT_LISTA_SUPPORT,
        system=spec.system,
        seed=spec.seed,
    )
    model, env = _load_model_and_env(run_dir, spec.system, device)
    xx, yy, states = _grid_states(spec.xlim, spec.ylim, grid_points)
    basin_labels_t, basin_label_source = _basin_labels_for_states(
        env,
        spec.system,
        states,
        endpoint_rollout_steps=endpoint_rollout_steps,
    )
    basin_labels = basin_labels_t.numpy()
    basin_count = int(max(get_transition_rich_basin_count(spec.system), int(basin_labels.max()) + 1))

    latents = _encode_latents(model, states, device)
    support_mask = _topk_support_mask(latents, topk)
    families = support_family_labels(
        support_mask[:, None, :],
        min_jaccard=family_jaccard,
    ).reshape(-1)
    support_basin, family_map = _family_to_dominant_basin(families, basin_labels)
    agreement = float(np.mean(support_basin == basin_labels))

    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = OKABE_ITO[:basin_count]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, basin_count + 0.5, 1.0), basin_count)

    basin_grid = basin_labels.reshape(grid_points, grid_points)
    support_grid = support_basin.reshape(grid_points, grid_points)
    ax.pcolormesh(
        xx,
        yy,
        basin_grid,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        alpha=0.18,
        rasterized=True,
    )
    ax.scatter(
        states[:, 0].numpy(),
        states[:, 1].numpy(),
        c=support_grid.reshape(-1),
        cmap=cmap,
        norm=norm,
        s=2.0,
        alpha=0.62,
        linewidths=0.0,
        rasterized=True,
    )
    mismatch = support_basin != basin_labels
    if bool(np.any(mismatch)):
        ax.scatter(
            states[mismatch, 0].numpy(),
            states[mismatch, 1].numpy(),
            c="#111111",
            s=0.9,
            alpha=0.45,
            linewidths=0.0,
            rasterized=True,
        )

    vx, vy, vstates = _grid_states(spec.xlim, spec.ylim, vector_points)
    velocity = _dynamics_for_states(env, vstates)
    u = velocity[:, 0].numpy().reshape(vector_points, vector_points)
    v = velocity[:, 1].numpy().reshape(vector_points, vector_points)
    speed = np.sqrt(u**2 + v**2)
    linewidth = 0.35 + 0.45 * speed / max(float(np.nanpercentile(speed, 95)), 1e-8)
    ax.streamplot(
        vx[0],
        vy[:, 0],
        u,
        v,
        color="#404040",
        linewidth=linewidth,
        density=1.05,
        arrowsize=0.55,
        zorder=4,
    )
    ax.set_title(spec.title)
    ax.set_xlim(spec.xlim)
    ax.set_ylim(spec.ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return {
        "system": spec.system,
        "title": spec.title,
        "root_label": ROOT_LISTA_SUPPORT,
        "seed": int(spec.seed),
        "run_dir": str(run_dir),
        "support_definition": f"topk:{topk}",
        "family_jaccard": float(family_jaccard),
        "grid_points": int(grid_points),
        "basin_label_source": basin_label_source,
        "basin_count": int(basin_count),
        "support_family_count": int(len(set(families.tolist()))),
        "dominant_basin_agreement_on_grid": agreement,
        "family_to_dominant_basin": family_map,
    }


def _system_slug(system: str) -> str:
    name = system.split(":", 1)[1] if ":" in system else system
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.replace("-", "_").lower()


def _valid_lengths(pred_xy: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(pred_xy).all(dim=-1)
    first_bad = (~finite).float().argmax(dim=0)
    all_good = finite.all(dim=0)
    return torch.where(all_good, torch.full_like(first_bad, pred_xy.shape[0]), first_bad).to(torch.long)


def _thin_trace(xy: torch.Tensor, max_points: int = 1200) -> torch.Tensor:
    if xy.shape[0] <= max_points:
        return xy
    step = max(1, int(math.ceil(xy.shape[0] / float(max_points))))
    out = xy[::step]
    if not torch.equal(out[-1], xy[-1]):
        out = torch.cat([out, xy[-1:]], dim=0)
    return out


def _axis_limits(init_xy: torch.Tensor, true_xy: torch.Tensor, pred_xy: torch.Tensor) -> tuple[float, float, float, float]:
    points = [init_xy, true_xy.reshape(-1, 2)]
    finite_pred = pred_xy[torch.isfinite(pred_xy).all(dim=-1)]
    if finite_pred.numel() > 0:
        points.append(finite_pred.reshape(-1, 2))
    xy = torch.cat(points, dim=0).to(torch.float64)
    xy = xy[torch.isfinite(xy).all(dim=-1)]
    if xy.shape[0] > 200_000:
        xy = xy[:: max(1, xy.shape[0] // 200_000)]
    lo = xy.min(dim=0).values
    hi = xy.max(dim=0).values
    span = (hi - lo).clamp_min(1e-6)
    pad = 0.15 * span
    lo = lo - pad
    hi = hi + pad
    center = 0.5 * (lo + hi)
    half_span = 0.5 * torch.max(hi - lo)
    lo = center - half_span
    hi = center + half_span
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _zoom_bounds(
    bounds: tuple[float, float, float, float],
    factor: float,
) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = bounds
    factor = float(factor)
    if factor <= 0.0 or factor >= 1.0:
        return bounds
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.5 * (x1 - x0) * factor
    hy = 0.5 * (y1 - y0) * factor
    return cx - hx, cx + hx, cy - hy, cy + hy


def _shift_x_bounds(
    bounds: tuple[float, float, float, float],
    fraction: float,
) -> tuple[float, float, float, float]:
    x0, x1, y0, y1 = bounds
    shift = float(fraction) * (x1 - x0)
    return x0 + shift, x1 + shift, y0, y1


def _load_rollout_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict rollout payload in {path}")
    return payload


def _prediction_xy(payload: dict[str, Any], mode_name: str, horizon: int) -> torch.Tensor:
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict) or mode_name not in predictions:
        raise KeyError(f"Prediction mode {mode_name!r} is missing")
    pred = predictions[mode_name]
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"Prediction mode {mode_name!r} is not a tensor")
    return pred[:horizon, :, :2].float()


def _select_dysts_root(row: dict[str, Any], horizon: int) -> tuple[str, dict[str, Any]]:
    panels = row["panels"]
    candidates = []
    for root in PRIMARY_DYSTS_ROOTS:
        key = f"{root}:H{horizon}"
        panel = panels.get(key)
        if not isinstance(panel, dict):
            continue
        mse = _safe_float(panel.get("mse"))
        if mse is not None:
            candidates.append((mse, root, panel))
    if not candidates:
        raise RuntimeError(f"No primary Dysts panels found for {row['system']} H{horizon}")
    _, root, panel = min(candidates, key=lambda item: item[0])
    return root, panel


def _render_dysts_panel(
    ax,
    *,
    spec: DystsPanelSpec,
    manifest_rows: list[dict[str, Any]],
    horizon: int,
    max_traces: int,
) -> dict[str, Any]:
    print(f"Rendering Dysts panel: {spec.system}", flush=True)
    row = next((item for item in manifest_rows if item.get("system") == spec.system), None)
    if row is None:
        raise RuntimeError(f"No Dysts manifest row for {spec.system}")
    root, panel = _select_dysts_root(row, horizon)
    root_meta = row["roots"][root]
    payload = _load_rollout_payload(Path(root_meta["selected_rollout_artifacts"]))
    init_states = payload["init_states"][:, :2].float()
    true_xy = payload["true_future"][:horizon, :, :2].float()
    pred_xy = _prediction_xy(payload, str(panel["mode"]), horizon)
    bounds = _axis_limits(init_states, true_xy, pred_xy)
    zoom_factor = float(DYSTS_AXIS_ZOOM.get(spec.system, 1.0))
    bounds = _zoom_bounds(bounds, zoom_factor)
    xshift_fraction = float(DYSTS_AXIS_XSHIFT_FRAC.get(spec.system, 0.0))
    bounds = _shift_x_bounds(bounds, xshift_fraction)
    valid_lengths = _valid_lengths(pred_xy)
    valid = torch.nonzero(valid_lengths > 1, as_tuple=False).flatten()
    if valid.numel() > 0:
        valid = valid[torch.argsort(valid_lengths[valid], descending=True)][:max_traces]
    for idx in valid.tolist():
        length = int(valid_lengths[idx].item())
        truth = _thin_trace(torch.cat([init_states[idx : idx + 1], true_xy[:, idx]], dim=0)).numpy()
        pred = _thin_trace(torch.cat([init_states[idx : idx + 1], pred_xy[:length, idx]], dim=0)).numpy()
        ax.plot(truth[:, 0], truth[:, 1], color="#7F7F7F", alpha=0.14, linewidth=0.65)
        ax.plot(pred[:, 0], pred[:, 1], color=DYSTS_FORECAST_COLOR, alpha=0.22, linewidth=0.78)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.45, alpha=0.28)
    ax.set_xticks([])
    ax.set_yticks([])
    display = DYSTS_DISPLAY.get(root, root)
    mse = float(panel["mse"])
    title = f"{spec.title}"
    ax.set_title(title)
    return {
        "system": spec.system,
        "title": spec.title,
        "selected_root": root,
        "selected_root_display": display,
        "horizon": int(horizon),
        "mode": str(panel["mode"]),
        "mse": mse,
        "axis_zoom_factor": zoom_factor,
        "axis_xshift_fraction": xshift_fraction,
        "selected_rollout_artifacts": root_meta["selected_rollout_artifacts"],
        "trace_count_plotted": int(valid.numel()),
    }


def render_composite(args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    support_rows = _read_rows(args.interpretability_csv)
    dysts_manifest = json.loads(args.dysts_manifest.read_text(encoding="utf-8"))
    dysts_rows = list(dysts_manifest["rows"])

    ncols = max(len(SUPPORT_PANELS), len(DYSTS_PANELS))
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(7.4, 4.15),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)

    support_meta = []
    for col, spec in enumerate(SUPPORT_PANELS):
        meta = _render_support_panel(
            axes[0, col],
            spec=spec,
            rows=support_rows,
            device=args.device,
            grid_points=args.support_grid_points,
            vector_points=args.vector_grid_points,
            endpoint_rollout_steps=args.endpoint_rollout_steps,
            family_jaccard=args.family_jaccard,
            topk=args.topk,
        )
        support_meta.append(meta)

    dysts_meta = []
    for col, spec in enumerate(DYSTS_PANELS):
        meta = _render_dysts_panel(
            axes[1, col],
            spec=spec,
            manifest_rows=dysts_rows,
            horizon=args.dysts_horizon,
            max_traces=args.dysts_max_traces,
        )
        dysts_meta.append(meta)

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for label, ax in zip(labels, axes.reshape(-1)):
        ax.text(
            -0.10,
            1.025,
            label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
        )

    fig.text(0.008, 0.74, "Support-basin alignment", rotation=90, va="center")
    fig.text(0.008, 0.25, "Dysts phase portraits", rotation=90, va="center")

    match_handle = tuple(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=3.0,
        )
        for color in OKABE_ITO[:4]
    )
    handles = [
        Line2D([0], [0], color="#7F7F7F", linewidth=1.5),
        Line2D([0], [0], color=DYSTS_FORECAST_COLOR, linewidth=1.5),
        match_handle,
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#111111",
            markeredgecolor="none",
            markersize=3.0,
        ),
    ]
    fig.legend(
        handles=handles,
        labels=[
            "Ground truth trajectory",
            "Model forecast",
            "Support/basin match",
            "Support/basin mismatch",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=4,
        frameon=False,
        fontsize=7.8,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "output_png": str(args.output),
        "output_pdf": str(pdf_path),
        "support_row": {
            "description": (
                "Evaluation-only basin-map backgrounds use the benchmark basin-label "
                "function when exposed, otherwise the catalog attractor-center map used "
                "by the validation figures; streamlines show the true vector field; "
                "colored grid points are the learned LISTA support-family partition "
                "mapped post hoc to each family's dominant basin."
            ),
            "panels": support_meta,
        },
        "dysts_row": {
            "description": "H5000 dt x30 seed-0 portraits using the best primary root in the stored all-model manifest.",
            "panels": dysts_meta,
        },
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpretability-csv", type=Path, default=DEFAULT_INTERPRETABILITY_CSV)
    parser.add_argument("--dysts-manifest", type=Path, default=DEFAULT_DYSTS_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--support-grid-points", type=int, default=84)
    parser.add_argument("--vector-grid-points", type=int, default=25)
    parser.add_argument("--endpoint-rollout-steps", type=int, default=360)
    parser.add_argument("--family-jaccard", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dysts-horizon", type=int, default=5000)
    parser.add_argument("--dysts-max-traces", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    manifest = render_composite(parse_args())
    print(json.dumps({
        "output_png": manifest["output_png"],
        "output_pdf": manifest["output_pdf"],
        "support_systems": [panel["system"] for panel in manifest["support_row"]["panels"]],
        "dysts_systems": [panel["system"] for panel in manifest["dysts_row"]["panels"]],
    }, indent=2))


if __name__ == "__main__":
    main()
