"""Prebuild deterministic Dysts caches for the paper workflows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from skae.config import apply_env_dt_override, get_config
from skae.data import DystsTrajectoryCache, make_env
from skae.dysts_cache_profiles import (
    DYSTS_CACHE_PROFILES,
    apply_dysts_cache_profile,
    default_dysts_cache_dir,
)
from experiments.neurips_2026.workflows.dysts_tasks import DYSTS_SYSTEM_SPECS


def _normalize_system(name: str) -> str:
    raw = name.strip()
    if not raw:
        raise ValueError("Empty system name")
    if raw.startswith("dysts:"):
        return raw
    return f"dysts:{raw}"


def _load_systems(args: argparse.Namespace) -> List[str]:
    systems: List[str] = []
    if args.systems:
        systems.extend(args.systems)
    if args.systems_file:
        for line in Path(args.systems_file).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            systems.append(line)
    if not systems:
        raise ValueError("No systems provided. Use --systems and/or --systems_file.")
    # Preserve order while removing duplicates.
    seen = set()
    deduped: List[str] = []
    for system in systems:
        normalized = _normalize_system(system)
        if normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped


def _apply_dysts_dt_multiplier(cfg, multiplier: float) -> float:
    """Apply a multiplier to the intrinsic Dysts timestep and return the dt used."""

    multiplier = float(multiplier)
    if multiplier <= 0.0:
        raise ValueError("--dt_multiplier must be positive")

    paper_spec = DYSTS_SYSTEM_SPECS.get(str(cfg.ENV.ENV_NAME))
    if paper_spec is not None:
        # Use the exact frozen value used by the training task builder. Even a
        # last-bit float difference changes the deterministic cache key and
        # would make GPU jobs rebuild a supposedly prebuilt CPU cache.
        base_dt = float(paper_spec["base_dt"])
    else:
        base_cfg = get_config("lista_nonlinear")
        base_cfg.ENV.ENV_NAME = cfg.ENV.ENV_NAME
        base_cfg.ENV.DYSTS.STANDARDIZE = bool(cfg.ENV.DYSTS.STANDARDIZE)
        base_cfg.ENV.DYSTS.IC_NOISE_SCALE = float(cfg.ENV.DYSTS.IC_NOISE_SCALE)
        base_env = make_env(base_cfg)
        base_dt = float(getattr(base_env.unwrapped, "dt"))
    dt_used = base_dt * multiplier
    apply_env_dt_override(cfg, dt_used, env_name=cfg.ENV.ENV_NAME)
    return dt_used


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prebuild dysts train/val/test caches for reuse across experiments"
    )
    parser.add_argument("--systems", nargs="+", default=None, help="System names (with or without dysts: prefix)")
    parser.add_argument("--systems_file", type=str, default=None, help="File with one system name per line")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["full"],
        choices=sorted(DYSTS_CACHE_PROFILES.keys()),
        help="Cache profiles to build",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "policy", "test"],
        help="Cache splits to build",
    )
    parser.add_argument("--config", type=str, default="lista_nonlinear", help="Base config preset")
    parser.add_argument("--cache_dir", type=str, default=default_dysts_cache_dir(), help="Cache output directory")
    parser.add_argument("--cache_num_workers", type=int, default=2, help="Workers for cache build fallback")
    parser.add_argument("--primary_method", default="Radau", help="Primary scipy solve_ivp method")
    parser.add_argument("--trajectory_timeout_seconds", type=float, default=0.0)
    parser.add_argument("--timeout_fallback_method", default="")
    parser.add_argument("--fallback_timeout_seconds", type=float, default=0.0)
    parser.add_argument("--ic_noise_scale", type=float, default=0.2, help="IC perturbation scale")
    parser.add_argument(
        "--dt_multiplier",
        type=float,
        default=30.0,
        help="Multiplier applied to each Dysts system's intrinsic timestep (paper default: 30).",
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use standardized coordinates",
    )
    args = parser.parse_args()

    standardize = bool(args.standardize)
    systems = _load_systems(args)
    total = len(systems) * len(args.profiles) * len(args.splits)

    print("=" * 80)
    print("Prebuild Dysts Cache")
    print(f"Systems: {len(systems)}")
    print(f"Profiles: {args.profiles}")
    print(f"Splits: {args.splits}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Standardize: {standardize}")
    print(f"DT multiplier: {args.dt_multiplier}")
    print(f"Cache workers: {args.cache_num_workers}")
    print(
        "Integration policy: "
        f"{args.primary_method} timeout={args.trajectory_timeout_seconds}s "
        f"fallback={args.timeout_fallback_method or '<none>'} "
        f"fallback_timeout={args.fallback_timeout_seconds}s"
    )
    print(f"Total cache jobs: {total}")
    print("=" * 80)

    done = 0
    failures: List[str] = []
    for system in systems:
        for profile in args.profiles:
            for split in args.splits:
                done += 1
                print(
                    f"[{done}/{total}] system={system} profile={profile} split={split}",
                    flush=True,
                )
                try:
                    cfg = get_config(args.config)
                    cfg.SEED = 0
                    cfg.ENV.ENV_NAME = system
                    cfg.ENV.DYSTS.STANDARDIZE = standardize
                    cfg.ENV.DYSTS.IC_NOISE_SCALE = float(args.ic_noise_scale)
                    cfg.ENV.DYSTS.USE_NATIVE_CACHE = True
                    cfg.ENV.DYSTS.CACHE_DIR = args.cache_dir
                    cfg.ENV.DYSTS.CACHE_REUSE = True
                    cfg.ENV.DYSTS.CACHE_SPLIT = split
                    cfg.ENV.DYSTS.CACHE_NUM_WORKERS = int(args.cache_num_workers)
                    cfg.ENV.DYSTS.CACHE_PRIMARY_METHOD = str(args.primary_method)
                    cfg.ENV.DYSTS.CACHE_TRAJECTORY_TIMEOUT_SECONDS = float(
                        args.trajectory_timeout_seconds
                    )
                    cfg.ENV.DYSTS.CACHE_TIMEOUT_FALLBACK_METHOD = str(
                        args.timeout_fallback_method
                    )
                    cfg.ENV.DYSTS.CACHE_FALLBACK_TIMEOUT_SECONDS = float(
                        args.fallback_timeout_seconds
                    )
                    apply_dysts_cache_profile(cfg, profile)
                    dt_used = _apply_dysts_dt_multiplier(cfg, float(args.dt_multiplier))

                    env = make_env(cfg)
                    t0 = time.perf_counter()
                    cache = DystsTrajectoryCache(env.unwrapped, cfg)
                    dt = time.perf_counter() - t0
                    print(
                        f"  -> ok dt={dt_used:.8g} shape={tuple(cache.trajectories.shape)} elapsed={dt:.1f}s",
                        flush=True,
                    )
                except Exception as exc:
                    key = f"{system} profile={profile} split={split}"
                    failures.append(f"{key}: {exc}")
                    print(f"  -> FAIL: {exc}", flush=True)

    print("=" * 80)
    print(f"Completed {done}/{total}")
    if failures:
        print(f"Failures: {len(failures)}")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)
    print("All cache builds completed.")


if __name__ == "__main__":
    main()
