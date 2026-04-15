"""Generate per-system long-horizon forecast phase-portrait figures for Dysts.

Selection protocol:
1. Scan the current matched paper-facing Dysts LISTA roots.
2. For each system and root family, keep the checkpoint with the lowest saved
   H3000 best-periodic mean from ``evaluation_results_best.json``.
3. Build a short checkpoint shortlist by saved H3000 best-periodic mean.
4. Re-evaluate that shortlist at the requested target horizon using each run's
   saved H3000 best-periodic mode and keep the best target-horizon run.

Outputs:
- one PNG and one PDF per Dysts system
- one metadata JSON per Dysts system
- one manifest JSON summarizing all selections
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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


@dataclass
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


@dataclass
class EvaluatedCandidate:
    candidate: Candidate
    target_best_periodic_mean: float
    target_best_periodic_std: float
    target_num_valid: int
    periodic_mode: int
    true_future: torch.Tensor
    pred_future: torch.Tensor
    init_states: torch.Tensor


def _system_dir_name(system: str) -> str:
    if system.startswith("dysts:"):
        return f"dysts_{system.split(':', 1)[1]}"
    return system


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _system_slug(system: str) -> str:
    name = system.split(":", 1)[1] if system.startswith("dysts:") else system
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.replace("-", "_").lower()


def _horizon_tag(horizon: int) -> str:
    return f"h{int(horizon)}"


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
        candidates.append(Candidate(
            system=system,
            root_label=root_spec.label,
            root_display_name=root_spec.display_name,
            run_dir=run_dir,
            checkpoint=run_dir / "checkpoint.pt",
            eval_json=eval_json,
            h3000_best_periodic_mean=mean,
            h3000_best_periodic_mode=horizon_metrics.get("mode"),
            seed=_seed_from_run_dir(run_dir),
        ))
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
    cfg: Config,
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


def _evaluate_periodic_mode(
    model,
    init_states: torch.Tensor,
    true_future: torch.Tensor,
    horizon: int,
    periodic_mode: int,
) -> tuple[float, float, int, torch.Tensor]:
    pred_future = _make_km_env_n_step(model, init_states, horizon, periodic_mode)
    final_sq_error = (pred_future[-1] - true_future[-1]).pow(2).mean(dim=-1)
    finite_mask = torch.isfinite(final_sq_error)
    valid = final_sq_error[finite_mask]
    if valid.numel() == 0:
        return float("inf"), float("nan"), 0, pred_future
    return (
        float(valid.mean().item()),
        float(valid.std(unbiased=False).item()) if valid.numel() > 1 else 0.0,
        int(valid.numel()),
        pred_future,
    )


def _evaluate_candidate(
    candidate: Candidate,
    *,
    init_states: torch.Tensor,
    true_future: torch.Tensor,
    horizon: int,
    periodic_mode: int,
    device: str,
) -> EvaluatedCandidate:
    _, _, model = _load_model(candidate.run_dir, candidate.system, device)
    mean, std, num_valid, pred_future = _evaluate_periodic_mode(
        model,
        init_states=init_states,
        true_future=true_future,
        horizon=horizon,
        periodic_mode=periodic_mode,
    )
    return EvaluatedCandidate(
        candidate=candidate,
        target_best_periodic_mean=mean,
        target_best_periodic_std=std,
        target_num_valid=num_valid,
        periodic_mode=periodic_mode,
        true_future=true_future,
        pred_future=pred_future,
        init_states=init_states,
    )


def _axis_limits(true_future: torch.Tensor, pred_future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    return mins - pad, maxs + pad


def _valid_trace_indices(true_future: torch.Tensor, pred_future: torch.Tensor) -> torch.Tensor:
    true_finite = torch.isfinite(true_future).all(dim=(0, 2))
    pred_finite = torch.isfinite(pred_future).all(dim=(0, 2))
    return torch.nonzero(true_finite & pred_finite, as_tuple=False).flatten()


def _render_phase_portrait(
    selection: EvaluatedCandidate,
    *,
    output_png: Path,
    max_traces: int,
    horizon: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_png.parent.mkdir(parents=True, exist_ok=True)

    true_future = selection.true_future
    pred_future = selection.pred_future
    init_states = selection.init_states

    valid_indices = _valid_trace_indices(true_future, pred_future)
    if valid_indices.numel() == 0:
        raise RuntimeError(f"No finite traces available for {selection.candidate.system}")
    if valid_indices.numel() > max_traces:
        valid_indices = valid_indices[:max_traces]

    lo, hi = _axis_limits(true_future, pred_future)

    fig, ax = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)

    for idx in valid_indices.tolist():
        gt_xy = torch.cat([init_states[idx : idx + 1, :2], true_future[:, idx, :2]], dim=0).cpu().numpy()
        pred_xy = torch.cat([init_states[idx : idx + 1, :2], pred_future[:, idx, :2]], dim=0).cpu().numpy()
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#9a9a9a", alpha=0.12, linewidth=1.1, zorder=2)
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], color="#d55e00", alpha=0.12, linewidth=1.1, zorder=3)

    ax.set_xlim(float(lo[0]), float(hi[0]))
    ax.set_ylim(float(lo[1]), float(hi[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    system_name = selection.candidate.system.split(":", 1)[1]
    ax.set_title(f"{system_name} ({int(horizon)}-step forecast)")
    seed_text = "unknown" if selection.candidate.seed is None else str(selection.candidate.seed)
    ax.text(
        0.03,
        0.97,
        f"{selection.candidate.root_display_name}, seed {seed_text}, periodic {selection.periodic_mode}",
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


def _selection_basis(shortlist_horizon: int, target_horizon: int) -> str:
    return (
        f"best H{int(target_horizon)} checkpoint among a shortlist of current matched Dysts LISTA runs, "
        f"where the shortlist is ranked by saved H{shortlist_horizon} best-periodic mean "
        f"and each shortlisted run is rescored at H{int(target_horizon)} using its saved H{shortlist_horizon} best-periodic mode"
    )


def _write_metadata(
    selection: EvaluatedCandidate,
    *,
    output_json: Path,
    output_png: Path,
    shortlist_horizon: int,
    horizon: int,
    rng_seed: int,
    batch_size: int,
    shortlist_candidates: Sequence[Candidate],
) -> None:
    horizon_tag = _horizon_tag(horizon)
    payload = {
        "system": selection.candidate.system,
        "selection_basis": _selection_basis(shortlist_horizon, horizon),
        "run_dir": str(selection.candidate.run_dir),
        "checkpoint": str(selection.candidate.checkpoint),
        "root_label": selection.candidate.root_label,
        "root_display_name": selection.candidate.root_display_name,
        "seed": selection.candidate.seed,
        "periodic_mode": selection.periodic_mode,
        "horizon": int(horizon),
        "batch_size": batch_size,
        "rng_seed": rng_seed,
        "h3000_best_periodic_mean": selection.candidate.h3000_best_periodic_mean,
        "h3000_best_periodic_mode": selection.candidate.h3000_best_periodic_mode,
        f"shared_batch_{horizon_tag}_best_periodic_mean": selection.target_best_periodic_mean,
        f"shared_batch_{horizon_tag}_best_periodic_std": selection.target_best_periodic_std,
        "num_valid": selection.target_num_valid,
        "shortlist_candidates": [
            {
                "root_label": finalist.root_label,
                "root_display_name": finalist.root_display_name,
                "run_dir": str(finalist.run_dir),
                "seed": finalist.seed,
                "h3000_best_periodic_mean": finalist.h3000_best_periodic_mean,
                "h3000_best_periodic_mode": finalist.h3000_best_periodic_mode,
            }
            for finalist in shortlist_candidates
        ],
        "outputs": [
            str(output_png),
            str(output_png.with_suffix(".pdf")),
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _generate_one_system(
    system: str,
    *,
    out_dir: Path,
    shortlist_horizon: int,
    shortlist_k: int,
    batch_size: int,
    horizon: int,
    rng_seed: int,
    device: str,
    max_traces: int,
) -> Dict[str, Any]:
    horizon_tag = _horizon_tag(horizon)
    all_candidates = [
        candidate
        for root_spec in ROOT_SPECS
        for candidate in _discover_root_candidates(system, root_spec, shortlist_horizon)
    ]
    all_candidates = [c for c in all_candidates if c.h3000_best_periodic_mode is not None]
    if not all_candidates:
        raise RuntimeError(f"No shortlist candidates discovered for {system}")
    shortlist_candidates = sorted(all_candidates, key=lambda item: item.h3000_best_periodic_mean)[:shortlist_k]

    shared_cfg, shared_env, _ = _load_model(shortlist_candidates[0].run_dir, system, device)
    init_states, true_future = _make_shared_batch(
        cfg=shared_cfg,
        env=shared_env,
        batch_size=batch_size,
        horizon=horizon,
        rng_seed=rng_seed,
    )

    selection = min(
        (
            _evaluate_candidate(
                candidate,
                init_states=init_states,
                true_future=true_future,
                horizon=horizon,
                periodic_mode=_parse_mode_name(candidate.h3000_best_periodic_mode),
                device=device,
            )
            for candidate in shortlist_candidates
        ),
        key=lambda item: item.target_best_periodic_mean,
    )

    slug = _system_slug(system)
    output_png = out_dir / f"{slug}_{horizon_tag}_lista_phase_portrait.png"
    output_json = out_dir / f"{slug}_{horizon_tag}_lista_phase_portrait.json"
    _render_phase_portrait(selection, output_png=output_png, max_traces=max_traces, horizon=horizon)
    _write_metadata(
        selection,
        output_json=output_json,
        output_png=output_png,
        shortlist_horizon=shortlist_horizon,
        horizon=horizon,
        rng_seed=rng_seed,
        batch_size=batch_size,
        shortlist_candidates=shortlist_candidates,
    )

    return {
        "system": system,
        "root_label": selection.candidate.root_label,
        "root_display_name": selection.candidate.root_display_name,
        "run_dir": str(selection.candidate.run_dir),
        "seed": selection.candidate.seed,
        "periodic_mode": selection.periodic_mode,
        "h3000_best_periodic_mean": selection.candidate.h3000_best_periodic_mean,
        f"shared_batch_{horizon_tag}_best_periodic_mean": selection.target_best_periodic_mean,
        f"shared_batch_{horizon_tag}_best_periodic_std": selection.target_best_periodic_std,
        "num_valid": selection.target_num_valid,
        "png": str(output_png),
        "pdf": str(output_png.with_suffix(".pdf")),
        "json": str(output_json),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Directory for per-system figure packets.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DYSTS_SYSTEMS),
        help="Systems to render (default: all 15 Dysts systems).",
    )
    parser.add_argument("--shortlist-horizon", type=int, default=3000)
    parser.add_argument(
        "--shortlist-k",
        type=int,
        default=8,
        help="Number of H3000-ranked checkpoints to rescore at the target horizon per system.",
    )
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=20260414)
    parser.add_argument("--max-traces", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizon_tag = _horizon_tag(args.horizon)

    manifest_rows = []
    for system in args.systems:
        row = _generate_one_system(
            system,
            out_dir=args.out_dir,
            shortlist_horizon=args.shortlist_horizon,
            shortlist_k=args.shortlist_k,
            batch_size=args.batch_size,
            horizon=args.horizon,
            rng_seed=args.rng_seed,
            device=args.device,
            max_traces=args.max_traces,
        )
        manifest_rows.append(row)
        print(
            f"{system}: {row['root_display_name']} seed={row['seed']} periodic={row['periodic_mode']} "
            f"H{int(args.horizon)}={row[f'shared_batch_{horizon_tag}_best_periodic_mean']:.6g}"
        )

    manifest = {
        "selection_basis": _selection_basis(args.shortlist_horizon, args.horizon),
        "horizon": args.horizon,
        "shortlist_horizon": args.shortlist_horizon,
        "shortlist_k": args.shortlist_k,
        "batch_size": args.batch_size,
        "rng_seed": args.rng_seed,
        "rows": manifest_rows,
    }
    manifest_path = args.out_dir / f"dysts_{horizon_tag}_lista_phase_portraits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
