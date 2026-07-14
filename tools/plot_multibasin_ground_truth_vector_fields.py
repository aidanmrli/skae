#!/usr/bin/env python3
"""Plot ground-truth vector fields for the retained 15 multibasin systems."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import torch

from skae.config import Config, get_env_dt
from skae.data import make_env


DEFAULT_OUTPUT_DIR = Path("docs/figures/neurips_paper_2026/ground_truth_vector_fields")
DEFAULT_FORMATS = ("png", "pdf", "svg")
CMAP = "viridis"


@dataclass(frozen=True)
class SystemSpec:
    system_key: str
    title: str
    basin_count: int
    xlim: tuple[float, float]
    ylim: tuple[float, float]

    @property
    def slug(self) -> str:
        return self.system_key.removeprefix("claude:").replace(":", "_")


RETAINED_15_SYSTEMS: tuple[SystemSpec, ...] = (
    SystemSpec("gated_local_linear", "Local-linear gates", 3, (-4.0, 4.0), (-4.0, 4.0)),
    SystemSpec("gated_transfer_linear", "Transfer-gated local-linear", 3, (-2.8, 2.8), (-2.8, 2.8)),
    SystemSpec("claude:arrested_spiral", "Arrested spiral", 5, (-3.0, 3.0), (-3.0, 3.0)),
    SystemSpec("claude:cal_asymmetric_3", "Asymmetric three-well", 3, (-3.0, 3.0), (-2.5, 3.0)),
    SystemSpec("claude:cal_high_cross_3", "High-cross three-well", 3, (-3.3, 3.3), (-3.3, 3.3)),
    SystemSpec("claude:cal_hexagon_6", "Hexagonal six-well", 6, (-3.3, 3.3), (-3.3, 3.3)),
    SystemSpec("claude:cal_octagon_8", "Octagonal eight-well", 8, (-4.0, 4.0), (-4.0, 4.0)),
    SystemSpec("claude:cal_pentagon_5", "Pentagonal five-well", 5, (-3.3, 3.3), (-3.3, 3.3)),
    SystemSpec("claude:cal_square_4", "Square four-well", 4, (-3.3, 3.3), (-3.3, 3.3)),
    SystemSpec("claude:duffing_triple_well", "Triple-well Duffing", 3, (-2.0, 2.0), (-2.0, 2.0)),
    SystemSpec("claude:snic_multi", "SNIC multi-attractor", 3, (-2.0, 2.0), (-2.0, 2.0)),
    SystemSpec("claude:transition_routes_4", "Transition-routes four-well", 4, (-3.5, 3.5), (-3.5, 3.5)),
    SystemSpec("claude:var_depth_gradient_4", "Depth-gradient four-well", 4, (-3.0, 3.0), (-3.0, 3.0)),
    SystemSpec("claude:var_diamond_4", "Diamond four-well", 4, (-3.5, 3.5), (-3.5, 3.5)),
    SystemSpec("claude:var_l_shape_5", "L-shaped five-well", 5, (-3.5, 3.5), (-3.5, 3.5)),
)

SYSTEMS_BY_KEY = {spec.system_key: spec for spec in RETAINED_15_SYSTEMS}


@dataclass
class FieldData:
    spec: SystemSpec
    dt: float
    xs: np.ndarray
    ys: np.ndarray
    u: np.ndarray
    v: np.ndarray
    log_speed: np.ndarray
    centers: np.ndarray | None
    log_speed_vmin: float
    log_speed_vmax: float


def parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(item.strip().lower().lstrip(".") for item in value.split(",") if item.strip())
    allowed = {"png", "pdf", "svg"}
    unknown = sorted(set(formats) - allowed)
    if unknown:
        raise ValueError(f"Unsupported output format(s): {unknown}. Expected subset of {sorted(allowed)}")
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def parse_systems(value: str) -> list[SystemSpec]:
    if value.strip().lower() in {"", "all", "retained15"}:
        return list(RETAINED_15_SYSTEMS)
    specs: list[SystemSpec] = []
    seen: set[str] = set()
    for system_key in [item.strip() for item in value.split(",") if item.strip()]:
        if system_key in seen:
            continue
        if system_key not in SYSTEMS_BY_KEY:
            known = ", ".join(SYSTEMS_BY_KEY)
            raise ValueError(f"Unknown retained-15 system {system_key!r}; known systems: {known}")
        specs.append(SYSTEMS_BY_KEY[system_key])
        seen.add(system_key)
    if not specs:
        raise ValueError("At least one system is required")
    return specs


def make_system_env(system_key: str):
    cfg = Config()
    cfg.ENV.ENV_NAME = system_key
    return make_env(cfg)


def vector_field_callable(env):
    base = env.unwrapped
    if hasattr(base, "dynamics"):
        return base.dynamics, "env.dynamics", False
    system = getattr(base, "system", None)
    if system is not None and hasattr(system, "dynamics"):
        return system.dynamics, "env.system.dynamics", True
    raise RuntimeError("Environment does not expose a ground-truth vector field")


def evaluate_dynamics(env, states: torch.Tensor, chunk_size: int) -> torch.Tensor:
    dynamics_fn, _source, needs_vmap = vector_field_callable(env)
    chunks = []
    for chunk in states.split(int(chunk_size), dim=0):
        if not needs_vmap:
            out = dynamics_fn(chunk)
        else:
            chunk64 = chunk.to(dtype=torch.float64)
            try:
                out = torch.vmap(dynamics_fn)(chunk64)
            except RuntimeError:
                out = torch.stack([dynamics_fn(row) for row in chunk64], dim=0)
        out = out.detach().cpu().to(dtype=torch.float32)
        if out.shape != chunk.shape:
            raise RuntimeError(f"Unexpected vector-field shape {tuple(out.shape)} for chunk {tuple(chunk.shape)}")
        chunks.append(out)
    return torch.cat(chunks, dim=0)


def grid_states(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_points: int,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    xs = np.linspace(float(xlim[0]), float(xlim[1]), int(grid_points))
    ys = np.linspace(float(ylim[0]), float(ylim[1]), int(grid_points))
    xx, yy = np.meshgrid(xs, ys)
    states = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    return xs, ys, states


def _centers_from_sequence(raw: Iterable[object]) -> np.ndarray | None:
    rows: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, torch.Tensor):
            values = item.detach().cpu().flatten().tolist()
        else:
            values = list(item)  # type: ignore[arg-type]
        if len(values) < 2:
            continue
        rows.append((float(values[0]), float(values[1])))
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def _snic_centers(system) -> np.ndarray:
    phi = math.acos(1.0 / float(system.a))
    roots = np.roots([1.0, 0.0, -1.0, float(system.eps) / float(system.a)])
    radius = float(max(root.real for root in roots if abs(root.imag) < 1e-8 and root.real > 0.0))
    centers = []
    for k in range(3):
        theta = (-phi + 2.0 * math.pi * k) / 3.0
        centers.append((radius * math.cos(theta), radius * math.sin(theta)))
    return np.asarray(centers, dtype=float)


def attractor_centers(env, system_key: str) -> np.ndarray | None:
    base = env.unwrapped
    if hasattr(base, "points_2d"):
        raw = getattr(base, "points_2d")
        return raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else _centers_from_sequence(raw)

    system = getattr(base, "system", None)
    if system is None:
        return None

    if system_key == "claude:arrested_spiral" and hasattr(system, "well_centers"):
        wells = getattr(system, "well_centers").detach().cpu().numpy()
        return np.vstack([wells, np.zeros((1, 2), dtype=float)])

    if system_key == "claude:duffing_triple_well":
        a = float(getattr(system, "a", 0.3))
        outer = math.sqrt(1.0 + math.sqrt(max(0.0, 1.0 - 2.0 * a)))
        return np.asarray([(-outer, 0.0), (0.0, 0.0), (outer, 0.0)], dtype=float)

    if system_key == "claude:snic_multi":
        return _snic_centers(system)

    for attr in ("_wells", "wells", "well_centers", "basins", "centers", "room_centers"):
        if hasattr(system, attr):
            raw = getattr(system, attr)
            if isinstance(raw, torch.Tensor):
                arr = raw.detach().cpu().numpy()
                return arr[:, :2] if arr.ndim == 2 and arr.shape[1] >= 2 else None
            return _centers_from_sequence(raw)
    return None


def compute_field(spec: SystemSpec, grid_points: int, chunk_size: int) -> FieldData:
    env = make_system_env(spec.system_key)
    xs, ys, states = grid_states(spec.xlim, spec.ylim, grid_points)
    vectors = evaluate_dynamics(env, states, chunk_size=chunk_size).numpy()
    u = vectors[:, 0].reshape((grid_points, grid_points))
    v = vectors[:, 1].reshape((grid_points, grid_points))
    speed = np.sqrt(u * u + v * v)
    log_speed = np.log10(np.maximum(speed, 1e-10))
    finite = log_speed[np.isfinite(log_speed)]
    if finite.size == 0:
        vmin, vmax = -10.0, 0.0
    else:
        vmin, vmax = np.percentile(finite, [2.0, 98.0])
        if math.isclose(float(vmin), float(vmax)):
            vmin = float(vmin) - 1.0
            vmax = float(vmax) + 1.0
    return FieldData(
        spec=spec,
        dt=float(get_env_dt(Config(), spec.system_key)),
        xs=xs,
        ys=ys,
        u=u,
        v=v,
        log_speed=log_speed,
        centers=attractor_centers(env, spec.system_key),
        log_speed_vmin=float(vmin),
        log_speed_vmax=float(vmax),
    )


def draw_field(
    ax,
    field: FieldData,
    *,
    title: str,
    title_size: float,
    label_axes: bool,
    stream_density: float,
) -> object:
    speed = 10.0 ** field.log_speed
    denom = np.maximum(speed, 1e-8)
    u_dir = np.nan_to_num(field.u / denom)
    v_dir = np.nan_to_num(field.v / denom)
    log_speed = np.nan_to_num(field.log_speed, nan=field.log_speed_vmin)
    norm = Normalize(vmin=field.log_speed_vmin, vmax=field.log_speed_vmax)
    stream = ax.streamplot(
        field.xs,
        field.ys,
        u_dir,
        v_dir,
        color=log_speed,
        cmap=CMAP,
        norm=norm,
        density=stream_density,
        linewidth=0.8,
        arrowsize=0.75,
        minlength=0.08,
        maxlength=3.0,
    )
    if field.centers is not None and field.centers.size:
        ax.scatter(
            field.centers[:, 0],
            field.centers[:, 1],
            marker="x",
            s=36,
            color="#111827",
            linewidths=1.5,
            zorder=4,
        )
    ax.set_title(title, fontsize=title_size, pad=5)
    ax.set_xlim(field.spec.xlim)
    ax.set_ylim(field.spec.ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f8fafc")
    if label_axes:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    else:
        ax.tick_params(labelbottom=False, labelleft=False)
    ax.tick_params(labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.6)
    return stream.lines


def save_figure(fig, output_base: Path, formats: Iterable[str], dpi: int) -> list[str]:
    paths: list[str] = []
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(str(path))
    return paths


def render_individual(
    field: FieldData,
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    stream_density: float,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(4.0, 3.8), constrained_layout=True)
    mappable = draw_field(
        ax,
        field,
        title=f"{field.spec.title} ({field.spec.basin_count} basins)",
        title_size=10.5,
        label_axes=True,
        stream_density=stream_density,
    )
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\log_{10}\|f(x)\|_2$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    paths = save_figure(fig, output_dir / f"ground_truth_vector_field_{field.spec.slug}", formats, dpi)
    plt.close(fig)
    return paths


def render_composite(
    fields: list[FieldData],
    output_dir: Path,
    formats: Iterable[str],
    dpi: int,
    stream_density: float,
) -> list[str]:
    ncols = 5
    nrows = math.ceil(len(fields) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 7.8), constrained_layout=True)
    axes_flat = np.ravel(axes)
    for idx, field in enumerate(fields):
        ax = axes_flat[idx]
        draw_field(
            ax,
            field,
            title=f"{idx + 1}. {field.spec.title}",
            title_size=8.6,
            label_axes=False,
            stream_density=stream_density,
        )
    for ax in axes_flat[len(fields) :]:
        ax.axis("off")
    return save_figure(fig, output_dir / "ground_truth_vector_fields_retained15", formats, dpi)


def write_manifest(
    output_dir: Path,
    fields: list[FieldData],
    *,
    grid_points: int,
    individual_paths: dict[str, list[str]],
    composite_paths: list[str],
) -> None:
    composite_pdf = next(
        (Path(path).name for path in composite_paths if Path(path).suffix == ".pdf"),
        None,
    )
    payload = {
        "schema_version": 1,
        "description": (
            "PDF vector-field displays for the 15 controlled multibasin "
            "systems retained by the paper."
        ),
        "generated_by": "tools/plot_multibasin_ground_truth_vector_fields.py",
        "compute_allocation": "salloc --mem=8G -c 4 --partition=long",
        "reproduction_command_inside_allocation": (
            "uv run python tools/plot_multibasin_ground_truth_vector_fields.py "
            "--formats pdf"
        ),
        "grid_points": int(grid_points),
        "composite": composite_pdf,
        "systems": [
            {
                "system_key": field.spec.system_key,
                "display_name": field.spec.title,
                "basins": field.spec.basin_count,
                "pdf": next(
                    (
                        Path(path).name
                        for path in individual_paths.get(field.spec.system_key, [])
                        if Path(path).suffix == ".pdf"
                    ),
                    None,
                ),
            }
            for field in fields
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default="retained15", help="retained15/all or comma-separated retained-15 system keys")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid_points", type=int, default=80)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--stream_density", type=float, default=1.05)
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip_individual", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_systems(args.systems)
    formats = parse_formats(args.formats)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fields = []
    for spec in specs:
        print(f"Computing vector field: {spec.system_key}", flush=True)
        fields.append(compute_field(spec, grid_points=args.grid_points, chunk_size=args.chunk_size))

    individual_paths: dict[str, list[str]] = {}
    if not args.skip_individual:
        for field in fields:
            print(f"Rendering individual plot: {field.spec.system_key}", flush=True)
            individual_paths[field.spec.system_key] = render_individual(
                field,
                args.output_dir,
                formats=formats,
                dpi=args.dpi,
                stream_density=args.stream_density,
            )

    print("Rendering retained-15 composite", flush=True)
    composite_paths = render_composite(
        fields,
        args.output_dir,
        formats=formats,
        dpi=args.dpi,
        stream_density=max(0.75, args.stream_density * 0.82),
    )
    write_manifest(
        args.output_dir,
        fields,
        grid_points=args.grid_points,
        individual_paths=individual_paths,
        composite_paths=composite_paths,
    )
    print(f"Wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
