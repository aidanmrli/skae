#!/usr/bin/env python3
"""Re-evaluate staged local-K runs against matched global-K runs.

This collector is intentionally separate from the historical Table 1 collector
because staged local-K checkpoints need a wrapper around the frozen base model
and learned local-map bundle. It evaluates both roots on the same horizon and
period grid, then writes paired rows and a compact markdown summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skae.config import Config
from skae.data import make_env
from skae.evaluation import EvaluationSettings, evaluate_model
from skae.model import make_model
from tools.train_staged_support_family_local_k import (
    SourceTargetLocalMapBundle,
    _make_wrapped_model,
    _parse_int_csv,
    _support_definition,
)


RunKey = Tuple[str, int]


def _safe_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str) and value.strip():
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def _seed_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        if not part.startswith("seed_"):
            continue
        raw = part.split("seed_", 1)[1]
        if raw.isdigit():
            return int(raw)
    return None


def _system_from_run(root: Path, run_dir: Path) -> Optional[str]:
    try:
        rel = run_dir.relative_to(root).parts
    except ValueError:
        rel = run_dir.parts
    seed_index = next((idx for idx, part in enumerate(rel) if part.startswith("seed_")), None)
    if seed_index is None:
        return None
    system_index = seed_index - 1
    if system_index >= 0 and rel[system_index].startswith("dt_"):
        system_index -= 1
    if system_index < 0:
        return None
    slug = rel[system_index]
    if slug.startswith("claude_"):
        return "claude:" + slug[len("claude_") :]
    return slug


def _discover_runs(root: Path) -> Dict[RunKey, Path]:
    candidates: Dict[RunKey, List[Path]] = defaultdict(list)
    for checkpoint in root.rglob("checkpoint.pt"):
        run_dir = checkpoint.parent
        seed = _seed_from_path(run_dir)
        system = _system_from_run(root, run_dir)
        if seed is None or system is None:
            continue
        candidates[(system, seed)].append(run_dir)
    return {key: sorted(paths, key=lambda path: (path.name, str(path)))[-1] for key, paths in candidates.items()}


def _evaluation_settings(
    *,
    cfg: Config,
    horizons: Sequence[int],
    periods: Sequence[int],
    batch_size: int,
) -> EvaluationSettings:
    settings = EvaluationSettings()
    settings.systems = [cfg.ENV.ENV_NAME]
    settings.horizons = tuple(int(horizon) for horizon in horizons)
    settings.periodic_reencode_periods = tuple(int(period) for period in periods)
    settings.phase_portrait_reencode_periods = tuple(dict.fromkeys((0, 1, *settings.periodic_reencode_periods)))
    settings.batch_size = int(batch_size)
    settings.save_rollout_artifacts = False
    settings.save_plots = False
    return settings


def _load_global(run_dir: Path, device: str) -> Tuple[Config, torch.nn.Module]:
    payload = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    cfg = Config.from_dict(payload["config"])
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return cfg, model


def _load_staged(
    run_dir: Path,
    *,
    device: str,
    support_definition: str,
    family_jaccard_threshold: float,
) -> Tuple[Config, torch.nn.Module]:
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=False)
    artifact_path = run_dir / "stage2_artifacts.pt"
    if checkpoint.get("route_codebook") is not None:
        artifact = checkpoint
    elif artifact_path.exists():
        artifact = torch.load(artifact_path, map_location=device, weights_only=False)
    else:
        raise FileNotFoundError(
            "Missing staged route metadata in checkpoint and missing staged "
            f"artifact: {artifact_path}"
        )

    cfg = Config.from_dict(checkpoint["config"])
    env = make_env(cfg)
    base_model = make_model(cfg, env.observation_size).to(device)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    route_codebook = artifact["route_codebook"]
    global_k = base_model.kmatrix().detach().cpu().numpy().astype(np.float32, copy=False)
    bundle = SourceTargetLocalMapBundle(
        family_ids=route_codebook["fitted_family_ids"],
        source_centers=route_codebook["centers"],
        target_centers=artifact["target_centers"],
        global_k=global_k,
        device=device,
    ).to(device)
    bundle.load_state_dict(checkpoint["local_bundle_state_dict"])

    scheme, support_value = _support_definition(support_definition)
    wrapped = _make_wrapped_model(
        base_model,
        bundle,
        route_codebook,
        route_env=env,
        scheme=scheme,
        support_value=support_value,
        family_jaccard_threshold=float(family_jaccard_threshold),
    )
    wrapped.eval()
    return cfg, wrapped


def _best_periodic_metrics(result: Dict, system: str, horizons: Sequence[int]) -> Dict[str, object]:
    if system in result:
        data = result[system]
    elif len(result) == 1:
        data = next(iter(result.values()))
    else:
        raise KeyError(f"Could not find system {system!r} in evaluation result keys {sorted(result)}")
    out: Dict[str, object] = {}
    for horizon in horizons:
        best = data.get("best_periodic", {}).get(str(horizon), {})
        out[f"h{horizon}_best_periodic_mean"] = _safe_float(best.get("mean"))
        out[f"h{horizon}_best_periodic_per_dim_mean"] = _safe_float(best.get("per_dim_mean"))
        out[f"h{horizon}_best_periodic_mode"] = best.get("mode")
    return out


def _evaluate_one(
    *,
    kind: str,
    system: str,
    seed: int,
    run_dir: Path,
    output_dir: Path,
    device: str,
    horizons: Sequence[int],
    periods: Sequence[int],
    batch_size: int,
    support_definition: str,
    family_jaccard_threshold: float,
    force: bool,
) -> Tuple[Config, Dict]:
    cache_path = output_dir / "eval_json" / f"{kind}__{system.replace(':', '_')}__seed{seed}.json"
    if cache_path.exists() and not force:
        payload = json.loads(cache_path.read_text())
        return Config.from_dict(payload["config"]), payload["result"]

    if kind == "global":
        cfg, model = _load_global(run_dir, device)
    elif kind == "staged":
        cfg, model = _load_staged(
            run_dir,
            device=device,
            support_definition=support_definition,
            family_jaccard_threshold=family_jaccard_threshold,
        )
    else:
        raise ValueError(f"Unknown run kind: {kind}")

    settings = _evaluation_settings(cfg=cfg, horizons=horizons, periods=periods, batch_size=batch_size)
    eval_dir = output_dir / "eval_artifacts" / f"{kind}__{cfg.ENV.ENV_NAME.replace(':', '_')}__seed{seed}"
    result = evaluate_model(model=model, cfg=cfg, device=device, settings=settings, output_dir=eval_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"config": cfg.to_dict(), "result": result}, indent=2))
    return cfg, result


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: List[Dict[str, object]], horizons: Sequence[int], periods: Sequence[int]) -> None:
    by_system: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_system[str(row["system_key"])].append(row)

    lines = [
        "# Staged Local-K vs Global-K Wide Periodic Re-evaluation",
        "",
        f"- Period grid: `{','.join(str(period) for period in periods)}`.",
        f"- Horizons: `{','.join('H' + str(horizon) for horizon in horizons)}`.",
        "- Metric: best-periodic raw cumulative MSE.",
        "",
    ]
    for system, system_rows in sorted(by_system.items()):
        lines.append(f"## {system}")
        header = "| seed | all horizons win | " + " | ".join(
            f"H{horizon} staged/global" for horizon in horizons
        ) + " |"
        lines.append(header)
        lines.append("|---:|:---:|" + "---:|" * len(horizons))
        for row in sorted(system_rows, key=lambda item: int(item["seed"])):
            values = []
            for horizon in horizons:
                staged = _safe_float(row.get(f"h{horizon}_staged_best_periodic_mean"))
                global_ = _safe_float(row.get(f"h{horizon}_global_best_periodic_mean"))
                ratio = _safe_float(row.get(f"h{horizon}_staged_over_global"))
                if staged is None or global_ is None or ratio is None:
                    values.append("N/A")
                else:
                    values.append(f"{staged:.4g}/{global_:.4g} ({ratio:.2f}x)")
            lines.append(
                f"| {row['seed']} | {'yes' if row['wins_all_horizons'] else 'no'} | "
                + " | ".join(values)
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged_root", required=True)
    parser.add_argument("--global_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--horizons", default="100,500,1000")
    parser.add_argument("--periods", default="1,2,5,10,20,25,50,100")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--support_definition", default="absolute:0.001")
    parser.add_argument("--family_jaccard_threshold", type=float, default=0.4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_pairs", type=int, default=0, help="0 evaluates all discovered pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    staged_root = Path(args.staged_root).expanduser()
    global_root = Path(args.global_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    horizons = _parse_int_csv(args.horizons)
    periods = _parse_int_csv(args.periods)
    staged_runs = _discover_runs(staged_root)
    global_runs = _discover_runs(global_root)
    pairs = sorted(set(staged_runs) & set(global_runs))
    if args.max_pairs and args.max_pairs > 0:
        pairs = pairs[: int(args.max_pairs)]

    manifest = {
        "staged_root": str(staged_root),
        "global_root": str(global_root),
        "output_dir": str(output_dir),
        "horizons": list(horizons),
        "periods": list(periods),
        "batch_size": int(args.batch_size),
        "support_definition": args.support_definition,
        "family_jaccard_threshold": float(args.family_jaccard_threshold),
        "num_staged_runs": len(staged_runs),
        "num_global_runs": len(global_runs),
        "num_pairs": len(pairs),
        "pairs": [{"system_key": system, "seed": seed} for system, seed in pairs],
    }
    (output_dir / "wide_periodic_reeval_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in ("num_staged_runs", "num_global_runs", "num_pairs")}, indent=2))
    if args.dry_run:
        return

    rows: List[Dict[str, object]] = []
    for idx, (system, seed) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] {system} seed={seed}", flush=True)
        staged_dir = staged_runs[(system, seed)]
        global_dir = global_runs[(system, seed)]
        try:
            global_cfg, global_result = _evaluate_one(
                kind="global",
                system=system,
                seed=seed,
                run_dir=global_dir,
                output_dir=output_dir,
                device=args.device,
                horizons=horizons,
                periods=periods,
                batch_size=int(args.batch_size),
                support_definition=args.support_definition,
                family_jaccard_threshold=float(args.family_jaccard_threshold),
                force=bool(args.force),
            )
            staged_cfg, staged_result = _evaluate_one(
                kind="staged",
                system=system,
                seed=seed,
                run_dir=staged_dir,
                output_dir=output_dir,
                device=args.device,
                horizons=horizons,
                periods=periods,
                batch_size=int(args.batch_size),
                support_definition=args.support_definition,
                family_jaccard_threshold=float(args.family_jaccard_threshold),
                force=bool(args.force),
            )
            row: Dict[str, object] = {
                "system_key": system,
                "seed": seed,
                "status": "ok",
                "staged_run_dir": str(staged_dir),
                "global_run_dir": str(global_dir),
                "staged_train_env_name": staged_cfg.ENV.ENV_NAME,
                "global_train_env_name": global_cfg.ENV.ENV_NAME,
            }
            staged_metrics = _best_periodic_metrics(staged_result, staged_cfg.ENV.ENV_NAME, horizons)
            global_metrics = _best_periodic_metrics(global_result, global_cfg.ENV.ENV_NAME, horizons)
            wins = []
            for horizon in horizons:
                staged_value = _safe_float(staged_metrics.get(f"h{horizon}_best_periodic_mean"))
                global_value = _safe_float(global_metrics.get(f"h{horizon}_best_periodic_mean"))
                row[f"h{horizon}_staged_best_periodic_mean"] = staged_value
                row[f"h{horizon}_global_best_periodic_mean"] = global_value
                row[f"h{horizon}_staged_best_periodic_mode"] = staged_metrics.get(
                    f"h{horizon}_best_periodic_mode"
                )
                row[f"h{horizon}_global_best_periodic_mode"] = global_metrics.get(
                    f"h{horizon}_best_periodic_mode"
                )
                if staged_value is not None and global_value is not None and global_value > 0.0:
                    row[f"h{horizon}_staged_over_global"] = staged_value / global_value
                    wins.append(staged_value < global_value)
                else:
                    row[f"h{horizon}_staged_over_global"] = None
                    wins.append(False)
            row["wins_all_horizons"] = bool(all(wins))
        except Exception as exc:  # noqa: BLE001 - keep long collections moving.
            row = {
                "system_key": system,
                "seed": seed,
                "status": "error",
                "error": repr(exc),
                "staged_run_dir": str(staged_dir),
                "global_run_dir": str(global_dir),
            }
            print(f"  ERROR: {exc!r}", flush=True)
        rows.append(row)
        _write_csv(output_dir / "wide_periodic_reeval_rows.csv", rows)

    _write_summary(output_dir / "wide_periodic_reeval_summary.md", rows, horizons, periods)
    (output_dir / "wide_periodic_reeval_rows.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {output_dir / 'wide_periodic_reeval_rows.csv'}")
    print(f"Wrote {output_dir / 'wide_periodic_reeval_summary.md'}")


if __name__ == "__main__":
    main()
