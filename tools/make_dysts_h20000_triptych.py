"""Generate a cached 3-panel H20000 Dysts phase-portrait figure.

This script reproduces the existing H20000 Dadras / QiChen /
ShimizuMorioka phase-portrait style, but places the three systems in a single
multi-panel figure. It saves the rollout tensors needed for plotting so the
figure can be regenerated quickly without rerunning long-horizon inference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, generate_trajectory, make_env
from skae.evaluation import _make_km_env_n_step
from skae.model import make_model


@dataclass(frozen=True)
class TriptychSpec:
    metadata_path: Path


SPECS = (
    TriptychSpec(metadata_path=Path("docs/figures/dadras_h20000_lista_phase_portrait.json")),
    TriptychSpec(metadata_path=Path("docs/figures/qi_chen_h20000_lista_phase_portrait.json")),
    TriptychSpec(metadata_path=Path("docs/figures/shimizu_morioka_h20000_lista_phase_portrait.json")),
)


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return env, model


def _make_rollouts(
    *,
    env,
    model,
    horizon: int,
    batch_size: int,
    periodic_mode: int,
    rng_seed: int,
):
    rng = torch.Generator().manual_seed(int(rng_seed))
    vec_env = VectorWrapper(env, batch_size)
    init_states = vec_env.reset(rng)
    true_future = generate_trajectory(vec_env.step, init_states, length=horizon)
    pred_future = _make_km_env_n_step(model, init_states, horizon, periodic_mode)
    return init_states, true_future, pred_future


def _axis_limits(true_xy: torch.Tensor, pred_xy: torch.Tensor):
    xy = torch.cat(
        [
            true_xy.reshape(-1, 2),
            pred_xy.reshape(-1, 2),
        ],
        dim=0,
    )
    finite = torch.isfinite(xy).all(dim=1)
    xy = xy[finite]
    mins = xy.min(dim=0).values
    maxs = xy.max(dim=0).values
    span = (maxs - mins).clamp_min(1e-6)
    pad = 0.06 * span
    return (mins - pad).cpu().numpy(), (maxs + pad).cpu().numpy()


def _valid_indices(true_xy: torch.Tensor, pred_xy: torch.Tensor) -> torch.Tensor:
    true_finite = torch.isfinite(true_xy).all(dim=(0, 2))
    pred_finite = torch.isfinite(pred_xy).all(dim=(0, 2))
    return torch.nonzero(true_finite & pred_finite, as_tuple=False).flatten()


def build_cache(cache_path: Path, device: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    systems: list[dict[str, Any]] = []
    for spec in SPECS:
        metadata = _load_metadata(spec.metadata_path)
        env, model = _load_model(Path(metadata["run_dir"]), metadata["system"], device)
        init_states, true_future, pred_future = _make_rollouts(
            env=env,
            model=model,
            horizon=int(metadata["horizon"]),
            batch_size=int(metadata["batch_size"]),
            periodic_mode=int(metadata["periodic_mode"]),
            rng_seed=int(metadata["rng_seed"]),
        )

        systems.append(
            {
                "label": metadata["system"].split(":", 1)[1],
                "system": metadata["system"],
                "run_dir": metadata["run_dir"],
                "root_display_name": metadata["root_display_name"],
                "seed": int(metadata["seed"]),
                "periodic_mode": int(metadata["periodic_mode"]),
                "horizon": int(metadata["horizon"]),
                "init_xy": init_states[:, :2].cpu().to(torch.float32),
                "true_xy": true_future[:, :, :2].cpu().to(torch.float32),
                "pred_xy": pred_future[:, :, :2].cpu().to(torch.float32),
            }
        )

    torch.save({"systems": systems}, cache_path)


def render_figure(cache_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    payload = torch.load(cache_path, map_location="cpu")
    systems = payload["systems"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.3), constrained_layout=True)

    for ax, item in zip(axes, systems):
        init_xy = item["init_xy"]
        true_xy = item["true_xy"]
        pred_xy = item["pred_xy"]
        valid_indices = _valid_indices(true_xy, pred_xy)
        if valid_indices.numel() == 0:
            raise RuntimeError(f"No finite traces available for {item['system']}")

        for idx in valid_indices.tolist():
            gt_xy = torch.cat([init_xy[idx : idx + 1], true_xy[:, idx]], dim=0).numpy()
            pred_panel_xy = torch.cat([init_xy[idx : idx + 1], pred_xy[:, idx]], dim=0).numpy()
            ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#9a9a9a", alpha=0.12, linewidth=1.1, zorder=2)
            ax.plot(pred_panel_xy[:, 0], pred_panel_xy[:, 1], color="#d55e00", alpha=0.12, linewidth=1.1, zorder=3)

        lo, hi = _axis_limits(true_xy, pred_xy)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"{item['label']} ({item['horizon']}-step forecast)")
        ax.text(
            0.03,
            0.97,
            f"{item['root_display_name']}, seed {item['seed']}, periodic {item['periodic_mode']}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2.6},
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

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/dysts_dadras_qichen_shimizumorioka_h20000_subfigures.png"),
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("docs/figures/dysts_dadras_qichen_shimizumorioka_h20000_rollouts.pt"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Regenerate cached rollout tensors before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_cache or not args.cache_path.exists():
        build_cache(args.cache_path, args.device)
        print(f"Saved rollout cache to {args.cache_path}")
    render_figure(args.cache_path, args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
