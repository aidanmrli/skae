"""Generate a 2-panel 3000-step Dysts forecast figure for slides.

This script uses the current best paper-facing checkpoints for:
- dysts:Duffing
- dysts:LorenzCoupled

Selection criterion: lowest saved H1000 best-periodic mean among the current
paper-facing run families inspected on 2026-04-14.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, generate_trajectory, make_env
from skae.evaluation import _make_km_env_n_step
from skae.model import make_model


@dataclass(frozen=True)
class ForecastSpec:
    system: str
    label: str
    run_dir: str
    periodic_mode: int


SPECS = (
    ForecastSpec(
        system="dysts:Duffing",
        label="Duffing",
        run_dir=(
            "/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/"
            "paper_followup_recipes/lista_blockdiag_ns200k_denseopt_sc6em3/"
            "dysts_Duffing/dt_0p0003725420903484178/seed_0/20260309-123530"
        ),
        periodic_mode=5,
    ),
    ForecastSpec(
        system="dysts:LorenzCoupled",
        label="LorenzCoupled",
        run_dir=(
            "/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/"
            "paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3/"
            "dysts_LorenzCoupled/dt_8p104850808468456em05/seed_5/20260325-010235"
        ),
        periodic_mode=200,
    ),
)


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


def _make_rollouts(
    *,
    cfg: Config,
    env,
    model,
    length: int,
    batch_size: int,
    periodic_mode: int,
):
    rng = torch.Generator().manual_seed(cfg.SEED + 12345 + 999)
    vec_env = VectorWrapper(env, batch_size)
    init_states = vec_env.reset(rng)
    true_future = generate_trajectory(vec_env.step, init_states, length=length)
    pred_future = _make_km_env_n_step(model, init_states, length, periodic_mode)
    return init_states, true_future, pred_future


def _axis_limits(true_future: torch.Tensor, pred_future: torch.Tensor):
    xy = torch.cat(
        [
            true_future[:, :, :2].reshape(-1, 2),
            pred_future[:, :, :2].reshape(-1, 2),
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


def _best_indices(true_future: torch.Tensor, pred_future: torch.Tensor, count: int) -> list[int]:
    diff = pred_future - true_future
    mse = diff.pow(2).mean(dim=(0, 2))
    finite = torch.isfinite(mse)
    valid = torch.arange(mse.numel())[finite]
    if valid.numel() == 0:
        return list(range(min(count, mse.numel())))
    order = torch.argsort(mse[finite])[:count]
    return valid[order].tolist()


def render_figure(output_path: Path, length: int, batch_size: int, plot_count: int, device: str) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    for ax, spec in zip(axes, SPECS):
        cfg, env, model = _load_model(Path(spec.run_dir), spec.system, device)
        init_states, true_future, pred_future = _make_rollouts(
            cfg=cfg,
            env=env,
            model=model,
            length=length,
            batch_size=batch_size,
            periodic_mode=spec.periodic_mode,
        )

        indices = _best_indices(true_future, pred_future, plot_count)

        # Plot forecast first, then the ground truth on top.
        for idx in indices:
            pred_xy = torch.cat(
                [init_states[idx : idx + 1, :2], pred_future[:, idx, :2]],
                dim=0,
            ).cpu().numpy()
            ax.plot(
                pred_xy[:, 0],
                pred_xy[:, 1],
                color="#d55e00",
                alpha=0.75,
                linewidth=1.2,
                label="Forecast" if idx == indices[0] else None,
                zorder=2,
            )

        for idx in indices:
            gt_xy = torch.cat(
                [init_states[idx : idx + 1, :2], true_future[:, idx, :2]],
                dim=0,
            ).cpu().numpy()
            ax.plot(
                gt_xy[:, 0],
                gt_xy[:, 1],
                color="#4d4d4d",
                alpha=0.35,
                linewidth=1.0,
                label="Ground truth" if idx == indices[0] else None,
                zorder=3,
            )

        lo, hi = _axis_limits(true_future, pred_future)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"{spec.label} ({length}-step forecast)")
        ax.text(
            0.02,
            0.98,
            f"best saved mode: periodic {spec.periodic_mode}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.5},
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/dysts_duffing_lorenzcoupled_h3000_forecasts.png"),
    )
    parser.add_argument("--length", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--plot-count", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_figure(
        output_path=args.output,
        length=args.length,
        batch_size=args.batch_size,
        plot_count=args.plot_count,
        device=args.device,
    )
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
