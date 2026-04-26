"""Generate fixed-17 transition-rich LISTA phase portraits at requested horizons.

Selection protocol:
1. Scan collected ``forecasting_rows.csv`` files under ``results/``.
2. Keep only rows on the fixed 17-system transition-rich shortlist whose
   ``root_label`` starts with ``lista_``.
3. Deduplicate by ``run_dir`` and rank runs within each system by saved
   ``H1000`` best-periodic mean (tie-breaks: ``H500``, ``H100``, then
   ``num_steps`` descending).
4. Pick the best loadable checkpoint per system and reuse its saved
   ``H1000`` best-periodic mode for rollouts at all requested horizons.

Outputs:
- one PNG and one PDF per system per horizon
- one JSON metadata file per system
- one top-level manifest JSON summarizing the packet
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from skae.benchmarks.transition_rich_basin_partition_manifest import (
    transition_rich_basin_partition_systems,
)
from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.data import VectorWrapper, generate_trajectory, make_env
from skae.evaluation import _make_km_env_n_step
from skae.model import make_model


FIXED17_SYSTEMS: tuple[str, ...] = tuple(
    spec.system_key for spec in transition_rich_basin_partition_systems()
)


@dataclass(frozen=True)
class Candidate:
    system: str
    root_label: str
    run_dir: Path
    checkpoint: Path
    seed: Optional[int]
    env_dt: Optional[float]
    num_steps: Optional[int]
    h100_best_periodic_mean: float
    h500_best_periodic_mean: float
    h1000_best_periodic_mean: float
    h1000_best_periodic_mode: str
    source_csv: Path


@dataclass(frozen=True)
class Selection:
    candidate: Candidate
    periodic_mode: int


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _system_slug(system: str) -> str:
    name = system.split(":", 1)[1] if ":" in system else system
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.replace("-", "_").lower()


def _system_title(system: str) -> str:
    return system.split(":", 1)[1] if ":" in system else system


def _horizon_tag(horizon: int) -> str:
    return f"h{int(horizon)}"


def _parse_mode_name(mode_name: str) -> int:
    if mode_name == "no_reencode":
        return 0
    if mode_name == "every_step":
        return 1
    if mode_name.startswith("periodic_"):
        return int(mode_name.split("_", 1)[1])
    raise ValueError(f"Unsupported periodic mode name: {mode_name}")


def _candidate_sort_key(candidate: Candidate) -> tuple[float, float, float, int, str, str]:
    num_steps = candidate.num_steps if candidate.num_steps is not None else -1
    seed = candidate.seed if candidate.seed is not None else 10**9
    return (
        candidate.h1000_best_periodic_mean,
        candidate.h500_best_periodic_mean,
        candidate.h100_best_periodic_mean,
        -num_steps,
        candidate.root_label,
        f"{seed:09d}",
    )


def _discover_forecasting_csvs(results_dir: Path) -> List[Path]:
    csvs = sorted(results_dir.glob("**/forecasting_rows.csv"))
    return [path for path in csvs if "transition_rich" in str(path)]


def _row_to_candidate(row: Dict[str, str], source_csv: Path, root_label_prefix: str) -> Optional[Candidate]:
    system = str(row.get("system_key") or "").strip()
    if system not in FIXED17_SYSTEMS:
        return None

    root_label = str(row.get("root_label") or "").strip()
    if not root_label.startswith(root_label_prefix):
        return None

    run_dir_raw = str(row.get("run_dir") or "").strip()
    if not run_dir_raw:
        return None
    run_dir = Path(run_dir_raw)
    checkpoint = run_dir / "checkpoint.pt"

    h1000 = _safe_float(row.get("h1000_best_periodic_mean"))
    h500 = _safe_float(row.get("h500_best_periodic_mean"))
    h100 = _safe_float(row.get("h100_best_periodic_mean"))
    mode_name = str(row.get("h1000_best_periodic_mode") or "").strip()
    if h1000 is None or h500 is None or h100 is None or not mode_name:
        return None

    return Candidate(
        system=system,
        root_label=root_label,
        run_dir=run_dir,
        checkpoint=checkpoint,
        seed=_safe_int(row.get("seed")),
        env_dt=_safe_float(row.get("env_dt")),
        num_steps=_safe_int(row.get("num_steps")),
        h100_best_periodic_mean=h100,
        h500_best_periodic_mean=h500,
        h1000_best_periodic_mean=h1000,
        h1000_best_periodic_mode=mode_name,
        source_csv=source_csv,
    )


def collect_candidates(results_dir: Path, root_label_prefix: str = "lista_") -> Dict[str, List[Candidate]]:
    dedup: Dict[str, Candidate] = {}
    for source_csv in _discover_forecasting_csvs(results_dir):
        with source_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                candidate = _row_to_candidate(row, source_csv, root_label_prefix)
                if candidate is None:
                    continue
                key = str(candidate.run_dir)
                previous = dedup.get(key)
                if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
                    dedup[key] = candidate

    grouped: Dict[str, List[Candidate]] = {system: [] for system in FIXED17_SYSTEMS}
    for candidate in dedup.values():
        grouped[candidate.system].append(candidate)
    for system in grouped:
        grouped[system] = sorted(grouped[system], key=_candidate_sort_key)
    return grouped


def _load_model(selection: Selection, device: str):
    checkpoint = torch.load(selection.candidate.checkpoint, map_location=device)
    cfg = Config.from_dict(checkpoint["config"])
    cfg.ENV.ENV_NAME = selection.candidate.system

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


def _compute_horizon_metric(
    pred_future: torch.Tensor,
    true_future: torch.Tensor,
    horizon: int,
) -> tuple[float, float, int]:
    pred_slice = pred_future[:horizon]
    true_slice = true_future[:horizon]
    final_sq_error = (pred_slice[-1] - true_slice[-1]).pow(2).mean(dim=-1)
    finite_mask = torch.isfinite(final_sq_error)
    valid = final_sq_error[finite_mask]
    if valid.numel() == 0:
        return float("inf"), float("nan"), 0
    std = float(valid.std(unbiased=False).item()) if valid.numel() > 1 else 0.0
    return float(valid.mean().item()), std, int(valid.numel())


def _trace_prefix_lengths(pred_future: torch.Tensor) -> List[tuple[int, int]]:
    finite = torch.isfinite(pred_future[:, :, :2]).all(dim=2)
    lengths: List[tuple[int, int]] = []
    for trace_idx in range(finite.shape[1]):
        mask = finite[:, trace_idx]
        first_bad = torch.nonzero(~mask, as_tuple=False)
        if first_bad.numel() == 0:
            prefix_len = int(mask.shape[0])
        else:
            prefix_len = int(first_bad[0].item())
        lengths.append((trace_idx, prefix_len))
    return sorted(lengths, key=lambda item: item[1], reverse=True)


def _axis_limits(
    pred_future: torch.Tensor,
    ranked_prefixes: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    points: List[torch.Tensor] = []
    for trace_idx, prefix_len in ranked_prefixes:
        if prefix_len < 2:
            continue
        points.append(pred_future[:prefix_len, trace_idx, :2])
    if not points:
        raise RuntimeError("No finite points available for axis limits")
    xy = torch.cat(points, dim=0)
    if xy.numel() == 0:
        raise RuntimeError("No finite points available for axis limits")
    if xy.shape[0] >= 10:
        mins = torch.quantile(xy, 0.01, dim=0)
        maxs = torch.quantile(xy, 0.99, dim=0)
    else:
        mins = xy.min(dim=0).values
        maxs = xy.max(dim=0).values
    span = (maxs - mins).clamp_min(1e-6)
    pad = 0.06 * span
    return mins - pad, maxs + pad


def _render_phase_portrait(
    *,
    system: str,
    init_states: torch.Tensor,
    pred_future: torch.Tensor,
    output_png: Path,
    max_traces: int,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)

    ranked_prefixes = _trace_prefix_lengths(pred_future)
    ranked_prefixes = [item for item in ranked_prefixes if item[1] >= 2]
    if not ranked_prefixes:
        raise RuntimeError(f"No finite forecast trace prefixes available for {system}")
    ranked_prefixes = ranked_prefixes[:max_traces]

    lo, hi = _axis_limits(pred_future, ranked_prefixes)
    x_bounds = xlim if xlim is not None else (float(lo[0]), float(hi[0]))
    y_bounds = ylim if ylim is not None else (float(lo[1]), float(hi[1]))

    fig, ax = plt.subplots(figsize=(6.4, 6.4), constrained_layout=True)
    for idx, prefix_len in ranked_prefixes:
        xy = torch.cat(
            [init_states[idx : idx + 1, :2], pred_future[:prefix_len, idx, :2]],
            dim=0,
        ).cpu().numpy()
        ax.plot(xy[:, 0], xy[:, 1], color="#d55e00", alpha=0.16, linewidth=1.0)

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(_system_title(system))

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _select_best_loadable_candidate(candidates: Sequence[Candidate]) -> Selection:
    errors: List[str] = []
    for candidate in candidates:
        if not candidate.checkpoint.exists():
            errors.append(f"missing checkpoint: {candidate.checkpoint}")
            continue
        try:
            periodic_mode = _parse_mode_name(candidate.h1000_best_periodic_mode)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{candidate.run_dir}: {exc}")
            continue
        return Selection(candidate=candidate, periodic_mode=periodic_mode)
    raise RuntimeError("No loadable candidate found:\n" + "\n".join(errors))


def _selection_basis() -> str:
    return (
        "best fixed-17 LISTA checkpoint per system by saved H1000 best-periodic mean "
        "across collected transition-rich forecasting rows; each selected checkpoint is "
        "then rolled out at the requested horizons using its saved H1000 best-periodic mode"
    )


def _generate_one_system(
    *,
    system: str,
    candidates: Sequence[Candidate],
    out_dir: Path,
    horizons: Sequence[int],
    batch_size: int,
    rng_seed: int,
    device: str,
    max_traces: int,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> Dict[str, Any]:
    selection = _select_best_loadable_candidate(candidates)
    cfg, env, model = _load_model(selection, device)

    max_horizon = max(int(h) for h in horizons)
    init_states, true_future = _make_shared_batch(
        cfg=cfg,
        env=env,
        batch_size=batch_size,
        horizon=max_horizon,
        rng_seed=rng_seed,
    )
    pred_future = _make_km_env_n_step(model, init_states, max_horizon, selection.periodic_mode)

    outputs: Dict[str, Dict[str, str]] = {}
    shared_metrics: Dict[str, Dict[str, Any]] = {}
    slug = _system_slug(system)
    for horizon in horizons:
        horizon = int(horizon)
        horizon_tag = _horizon_tag(horizon)
        pred_slice = pred_future[:horizon]
        output_png = out_dir / f"{slug}_{horizon_tag}_lista_phase_portrait.png"
        _render_phase_portrait(
            system=system,
            init_states=init_states,
            pred_future=pred_slice,
            output_png=output_png,
            max_traces=max_traces,
            xlim=xlim,
            ylim=ylim,
        )
        mean, std, num_valid = _compute_horizon_metric(pred_future, true_future, horizon)
        outputs[horizon_tag] = {
            "png": str(output_png),
            "pdf": str(output_png.with_suffix(".pdf")),
        }
        shared_metrics[horizon_tag] = {
            "shared_batch_best_periodic_mean": mean,
            "shared_batch_best_periodic_std": std,
            "num_valid": num_valid,
        }

    metadata = {
        "system": system,
        "selection_basis": _selection_basis(),
        "selected_candidate": {
            "root_label": selection.candidate.root_label,
            "seed": selection.candidate.seed,
            "run_dir": str(selection.candidate.run_dir),
            "checkpoint": str(selection.candidate.checkpoint),
            "env_dt": selection.candidate.env_dt,
            "num_steps": selection.candidate.num_steps,
            "h100_best_periodic_mean": selection.candidate.h100_best_periodic_mean,
            "h500_best_periodic_mean": selection.candidate.h500_best_periodic_mean,
            "h1000_best_periodic_mean": selection.candidate.h1000_best_periodic_mean,
            "h1000_best_periodic_mode": selection.candidate.h1000_best_periodic_mode,
            "periodic_mode": selection.periodic_mode,
            "source_csv": str(selection.candidate.source_csv),
        },
        "batch_size": batch_size,
        "rng_seed": rng_seed,
        "xlim": list(xlim) if xlim is not None else None,
        "ylim": list(ylim) if ylim is not None else None,
        "shared_batch_metrics": shared_metrics,
        "outputs": outputs,
        "top_ranked_candidates": [
            {
                "root_label": candidate.root_label,
                "seed": candidate.seed,
                "run_dir": str(candidate.run_dir),
                "env_dt": candidate.env_dt,
                "num_steps": candidate.num_steps,
                "h100_best_periodic_mean": candidate.h100_best_periodic_mean,
                "h500_best_periodic_mean": candidate.h500_best_periodic_mean,
                "h1000_best_periodic_mean": candidate.h1000_best_periodic_mean,
                "h1000_best_periodic_mode": candidate.h1000_best_periodic_mode,
            }
            for candidate in candidates[:5]
        ],
    }
    metadata_path = out_dir / f"{slug}_lista_phase_portrait_selection.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "system": system,
        "root_label": selection.candidate.root_label,
        "seed": selection.candidate.seed,
        "periodic_mode": selection.periodic_mode,
        "run_dir": str(selection.candidate.run_dir),
        "env_dt": selection.candidate.env_dt,
        "num_steps": selection.candidate.num_steps,
        "h100_best_periodic_mean": selection.candidate.h100_best_periodic_mean,
        "h500_best_periodic_mean": selection.candidate.h500_best_periodic_mean,
        "h1000_best_periodic_mean": selection.candidate.h1000_best_periodic_mean,
        "shared_batch_metrics": shared_metrics,
        "metadata_json": str(metadata_path),
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to scan for collected forecasting_rows.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures/fixed17_lista_phase_portraits_20260414"),
        help="Output directory for figure files and manifests.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(FIXED17_SYSTEMS),
        help="Optional subset of fixed-17 systems to render.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1000, 3000, 5000],
        help="Requested rollout horizons.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=20260414)
    parser.add_argument("--max-traces", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--xlim",
        nargs=2,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX"),
        help="Optional fixed x-axis limits for every portrait.",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Optional fixed y-axis limits for every portrait.",
    )
    parser.add_argument(
        "--root-label-prefix",
        default="lista_",
        help="Root-label prefix used to identify standard LISTA runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates_by_system = collect_candidates(
        results_dir=args.results_dir,
        root_label_prefix=args.root_label_prefix,
    )

    manifest_rows: List[Dict[str, Any]] = []
    for system in args.systems:
        candidates = candidates_by_system.get(system, [])
        if not candidates:
            raise RuntimeError(f"No LISTA forecasting rows found for {system}")
        row = _generate_one_system(
            system=system,
            candidates=candidates,
            out_dir=args.out_dir,
            horizons=args.horizons,
            batch_size=args.batch_size,
            rng_seed=args.rng_seed,
            device=args.device,
            max_traces=args.max_traces,
            xlim=tuple(args.xlim) if args.xlim is not None else None,
            ylim=tuple(args.ylim) if args.ylim is not None else None,
        )
        manifest_rows.append(row)
        print(
            f"{system}: {row['root_label']} seed={row['seed']} "
            f"H1000={row['h1000_best_periodic_mean']:.6g}"
        )

    horizons_tag = "_".join(_horizon_tag(h) for h in args.horizons)
    manifest = {
        "selection_basis": _selection_basis(),
        "systems": list(args.systems),
        "horizons": [int(h) for h in args.horizons],
        "batch_size": int(args.batch_size),
        "rng_seed": int(args.rng_seed),
        "xlim": list(args.xlim) if args.xlim is not None else None,
        "ylim": list(args.ylim) if args.ylim is not None else None,
        "root_label_prefix": str(args.root_label_prefix),
        "rows": manifest_rows,
    }
    manifest_path = args.out_dir / f"fixed17_{horizons_tag}_lista_phase_portraits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
