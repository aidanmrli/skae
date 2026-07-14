"""Reevaluate one Dysts training run at long horizons without plot generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from skae.checkpoint_compat import load_model_state_dict_compat
from skae.config import Config
from skae.dysts_cache_profiles import apply_dysts_cache_profile, default_dysts_cache_dir
from skae.data import make_env
from skae.evaluation import EvaluationSettings, evaluate_model
from skae.model import make_model


DEFAULT_HORIZONS: Tuple[int, ...] = (
    100,
    500,
    1000,
    1500,
    2000,
    3000,
    4000,
    5000,
)
DEFAULT_PERIODIC_REENCODE_PERIODS: Tuple[int, ...] = (10, 25, 50, 100, 150, 200)


def _get_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _torch_load(path: Path, *, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _output_root(run_dir: Path, output_tag: str) -> Path:
    return run_dir / f"reeval_{output_tag}"


def _results_json_path(run_dir: Path, output_tag: str, checkpoint_name: str) -> Path:
    return _output_root(run_dir, output_tag) / f"evaluation_results_{checkpoint_name}.json"


def _evaluation_dir(run_dir: Path, output_tag: str, checkpoint_name: str) -> Path:
    return _output_root(run_dir, output_tag) / f"evaluation_{checkpoint_name}"


def _rollout_paths(
    run_dir: Path,
    output_tag: str,
    checkpoint_name: str,
    system: str,
) -> tuple[Path, Path]:
    system_dir = _evaluation_dir(run_dir, output_tag, checkpoint_name) / system
    return (
        system_dir / "rollout_artifacts.pt",
        system_dir / "selected_rollout_artifacts.pt",
    )


def _has_required_horizons(system_data: Dict[str, Any], horizons: Sequence[int]) -> bool:
    for horizon in horizons:
        horizon_key = str(horizon)
        best = system_data.get("best_periodic", {}).get(horizon_key)
        if not isinstance(best, dict):
            return False
        if best.get("mean") is None:
            return False
    return True


def _is_complete(
    *,
    run_dir: Path,
    output_tag: str,
    checkpoint_name: str,
    system: str,
    horizons: Sequence[int],
    require_selected_rollouts: bool = False,
) -> bool:
    results_json = _results_json_path(run_dir, output_tag, checkpoint_name)
    compact_rollouts = _rollout_paths(run_dir, output_tag, checkpoint_name, system)[1]
    payload = _safe_json_load(results_json)
    if not isinstance(payload, dict):
        return False
    system_data = payload.get(system)
    if not isinstance(system_data, dict):
        return False
    if not _has_required_horizons(system_data, horizons):
        return False
    if not require_selected_rollouts:
        return True
    files = system_data.get("files", {})
    selected_path = files.get("selected_rollout_artifacts")
    if isinstance(selected_path, str) and Path(selected_path).exists():
        return True
    return compact_rollouts.exists()


def _load_checkpoint_model(
    *,
    run_dir: Path,
    checkpoint_name: str,
    system: str,
    device: str,
    dysts_cache_profile: str,
    dysts_cache_split: str,
    dysts_cache_dir: Optional[str],
    dysts_cache_num_workers: Optional[int],
):
    checkpoint_file = "checkpoint.pt" if checkpoint_name == "checkpoint" else f"{checkpoint_name}.pt"
    checkpoint_path = run_dir / checkpoint_file
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = _torch_load(checkpoint_path, map_location=device)
    if "config" not in checkpoint:
        raise KeyError(f"Checkpoint missing config payload: {checkpoint_path}")

    model_cfg = Config.from_dict(checkpoint["config"])
    model_env = make_env(model_cfg)
    model = make_model(model_cfg, model_env.observation_size)
    load_model_state_dict_compat(model, checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    eval_cfg = Config.from_dict(model_cfg.to_dict())
    eval_cfg.ENV.ENV_NAME = system
    eval_cfg.ENV.DYSTS.USE_NATIVE_CACHE = True
    eval_cfg.ENV.DYSTS.CACHE_REUSE = True
    eval_cfg.ENV.DYSTS.CACHE_SPLIT = dysts_cache_split
    apply_dysts_cache_profile(eval_cfg, dysts_cache_profile)
    if dysts_cache_dir:
        eval_cfg.ENV.DYSTS.CACHE_DIR = str(dysts_cache_dir)
    elif not eval_cfg.ENV.DYSTS.CACHE_DIR:
        eval_cfg.ENV.DYSTS.CACHE_DIR = default_dysts_cache_dir()
    if dysts_cache_num_workers is not None:
        eval_cfg.ENV.DYSTS.CACHE_NUM_WORKERS = int(dysts_cache_num_workers)

    eval_env = make_env(eval_cfg)
    model.dt = getattr(eval_env.unwrapped, "dt", getattr(model, "dt", None))
    if eval_env.observation_size != model.observation_size:
        raise ValueError(
            f"Observation mismatch for {run_dir}: system {system} has obs dim "
            f"{eval_env.observation_size}, model expects {model.observation_size}"
        )

    return model, eval_cfg, checkpoint_path


def _selected_modes_from_results(
    system_results: Dict[str, Any],
    horizons: Sequence[int],
) -> tuple[List[str], Dict[str, Dict[str, str]]]:
    selected: List[str] = []
    mappings: Dict[str, Dict[str, str]] = {"best_periodic": {}, "best_reset": {}}
    for summary_key in ("best_periodic", "best_reset"):
        summary = system_results.get(summary_key, {})
        for horizon in horizons:
            horizon_key = str(horizon)
            mode = summary.get(horizon_key, {}).get("mode")
            if not isinstance(mode, str):
                continue
            mappings[summary_key][horizon_key] = mode
            if mode not in selected:
                selected.append(mode)
    return selected, mappings


def _compact_rollouts(
    *,
    run_dir: Path,
    output_tag: str,
    checkpoint_name: str,
    system: str,
    horizons: Sequence[int],
    keep_full_rollouts: bool,
) -> List[str]:
    full_rollouts, compact_rollouts = _rollout_paths(run_dir, output_tag, checkpoint_name, system)
    results_json = _results_json_path(run_dir, output_tag, checkpoint_name)
    if not full_rollouts.exists():
        raise FileNotFoundError(f"Missing rollout artifacts: {full_rollouts}")
    payload = torch.load(full_rollouts, map_location="cpu")
    results = _safe_json_load(results_json)
    if not isinstance(results, dict):
        raise RuntimeError(f"Unable to load reevaluation JSON: {results_json}")
    system_results = results.get(system)
    if not isinstance(system_results, dict):
        raise RuntimeError(f"Missing system '{system}' in {results_json}")

    selected_modes, mappings = _selected_modes_from_results(system_results, horizons)
    if not selected_modes:
        raise RuntimeError(
            f"No selected rollout modes were found for {run_dir} / {system} in {results_json}"
        )
    predictions = payload.get("predictions", {})
    diagnostics = payload.get("mode_diagnostics", {})
    compact_payload = {
        "system": payload.get("system"),
        "seed": payload.get("seed"),
        "run_dir": str(run_dir),
        "checkpoint_name": checkpoint_name,
        "horizons": [int(h) for h in horizons],
        "selected_modes": list(selected_modes),
        "best_periodic_modes": mappings["best_periodic"],
        "best_reset_modes": mappings["best_reset"],
        "init_states": payload["init_states"],
        "true_future": payload["true_future"],
        "true_sequences": payload["true_sequences"],
        "predictions": {
            mode_name: predictions[mode_name]
            for mode_name in selected_modes
            if mode_name in predictions
        },
        "mode_diagnostics": {
            mode_name: diagnostics.get(mode_name, {})
            for mode_name in selected_modes
        },
        "source_rollout_artifacts": str(full_rollouts),
    }
    torch.save(compact_payload, compact_rollouts)

    files = system_results.setdefault("files", {})
    files["selected_rollout_artifacts"] = str(compact_rollouts)
    if keep_full_rollouts:
        files["rollout_artifacts"] = str(full_rollouts)
    else:
        files["rollout_artifacts"] = None
        full_rollouts.unlink(missing_ok=True)
    system_results["selected_rollout_modes"] = list(selected_modes)
    system_results["selected_rollout_mode_map"] = mappings
    results[system] = system_results
    results_json.write_text(json.dumps(results, indent=2))
    return selected_modes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reevaluate one Dysts run at long horizons.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--system", required=True, help="Dysts system key, e.g. dysts:Dadras.")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Evaluation horizons.",
    )
    parser.add_argument(
        "--dysts-periodic-reencode-periods",
        nargs="+",
        type=int,
        default=None,
        help="Periodic reencoding periods to test for Dysts systems.",
    )
    parser.add_argument("--checkpoint-name", default="checkpoint", help="Checkpoint stem to evaluate.")
    parser.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps.")
    parser.add_argument(
        "--output-tag",
        default="dysts_dt30_h100_to_h5000_paper",
        help="Reevaluation output tag.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Held-out rollout batch size.")
    parser.add_argument("--save-plots", action="store_true", help="Also render qualitative plots.")
    parser.add_argument(
        "--save-selected-rollouts",
        action="store_true",
        help="Save compact rollout tensors for the selected best modes. Off by default.",
    )
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help="Exit early when requested horizons already exist; also requires selected rollouts if requested.",
    )
    parser.add_argument(
        "--keep-full-rollouts",
        action="store_true",
        help="Keep the large all-modes rollout artifact in addition to the compact selected artifact.",
    )
    parser.add_argument("--dysts-cache-profile", default="full", help="Shared Dysts cache profile.")
    parser.add_argument("--dysts-cache-split", default="test", help="Dysts cache split.")
    parser.add_argument("--dysts-cache-dir", default=None, help="Optional shared Dysts cache directory.")
    parser.add_argument("--dysts-cache-num-workers", type=int, default=None, help="Optional cache worker override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    system = str(args.system)
    horizons = tuple(sorted({int(h) for h in args.horizons}))
    dysts_periodic_reencode_periods = (
        tuple(sorted({int(p) for p in args.dysts_periodic_reencode_periods}))
        if args.dysts_periodic_reencode_periods is not None
        else DEFAULT_PERIODIC_REENCODE_PERIODS
    )
    checkpoint_name = str(args.checkpoint_name)
    output_tag = str(args.output_tag)
    save_selected_rollouts = bool(args.save_selected_rollouts or args.keep_full_rollouts)

    if args.skip_if_complete and _is_complete(
        run_dir=run_dir,
        output_tag=output_tag,
        checkpoint_name=checkpoint_name,
        system=system,
        horizons=horizons,
        require_selected_rollouts=save_selected_rollouts,
    ):
        print(
            f"Skip complete reevaluation: run_dir={run_dir} system={system} "
            f"checkpoint={checkpoint_name} tag={output_tag}"
        )
        return

    device = _get_device(str(args.device))
    model, eval_cfg, checkpoint_path = _load_checkpoint_model(
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        system=system,
        device=device,
        dysts_cache_profile=str(args.dysts_cache_profile),
        dysts_cache_split=str(args.dysts_cache_split),
        dysts_cache_dir=args.dysts_cache_dir,
        dysts_cache_num_workers=args.dysts_cache_num_workers,
    )
    print(
        f"Reevaluating {checkpoint_path} on {system} with device={device}, "
        f"horizons={horizons}, output_tag={output_tag}",
        flush=True,
    )

    settings = EvaluationSettings(
        systems=(system,),
        horizons=horizons,
        dysts_periodic_reencode_periods=dysts_periodic_reencode_periods,
        batch_size=int(args.batch_size),
        save_rollout_artifacts=save_selected_rollouts,
        save_plots=bool(args.save_plots),
    )
    output_dir = _output_root(run_dir, output_tag)
    eval_results = evaluate_model(
        model=model,
        cfg=eval_cfg,
        device=device,
        settings=settings,
        output_dir=_evaluation_dir(run_dir, output_tag, checkpoint_name),
    )
    system_results = eval_results.get(system)
    if not isinstance(system_results, dict) or not _has_required_horizons(system_results, horizons):
        raise RuntimeError(
            f"Long-horizon reevaluation did not produce all requested horizons for "
            f"{run_dir} / {system}"
        )

    results_json = _results_json_path(run_dir, output_tag, checkpoint_name)
    results_json.parent.mkdir(parents=True, exist_ok=True)
    eval_results["reeval_metadata"] = {
        "run_dir": str(run_dir),
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "system": system,
        "device": device,
        "output_tag": output_tag,
        "horizons": list(horizons),
        "dysts_periodic_reencode_periods": list(dysts_periodic_reencode_periods),
        "batch_size": int(args.batch_size),
        "dysts_cache_profile": str(args.dysts_cache_profile),
        "dysts_cache_split": str(args.dysts_cache_split),
        "dysts_cache_dir": eval_cfg.ENV.DYSTS.CACHE_DIR,
        "save_selected_rollouts": save_selected_rollouts,
        "keep_full_rollouts": bool(args.keep_full_rollouts),
    }
    results_json.write_text(json.dumps(eval_results, indent=2))

    if save_selected_rollouts:
        selected_modes = _compact_rollouts(
            run_dir=run_dir,
            output_tag=output_tag,
            checkpoint_name=checkpoint_name,
            system=system,
            horizons=horizons,
            keep_full_rollouts=bool(args.keep_full_rollouts),
        )
    else:
        selected_modes = []
    print(
        f"Completed reevaluation for {run_dir} ({system}); "
        f"selected rollout modes={selected_modes if save_selected_rollouts else 'not saved'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
