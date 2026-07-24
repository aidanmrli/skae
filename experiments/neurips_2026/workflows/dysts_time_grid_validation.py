"""Validate the corrected Dysts dt-x30 sampling contract before GPU work."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

from experiments.neurips_2026.workflows.dysts_tasks import DYSTS_SYSTEM_SPECS
from skae.benchmarks.dysts_adapter import DystsEnv


HORIZONS = (100, 2000, 4000, 5000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_system(system_key: str, multiplier: int) -> dict[str, object]:
    spec = DYSTS_SYSTEM_SPECS[system_key]
    name = system_key.split(":", 1)[1]
    expected_native_dt = float(spec["base_dt"])
    native = DystsEnv(name, standardize=False)
    observed_native_dt = float(native.system.dt)
    if not math.isclose(
        observed_native_dt,
        expected_native_dt,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(
            f"{name}: installed native dt {observed_native_dt} != {expected_native_dt}"
        )

    requested_dt = expected_native_dt * multiplier
    corrected = DystsEnv(name, dt_override=requested_dt, standardize=False)
    if not math.isclose(float(corrected.dt), requested_dt, rel_tol=0, abs_tol=1e-15):
        raise RuntimeError(f"{name}: wrapper dt override failed")
    if not math.isclose(
        float(corrected.system.dt), requested_dt, rel_tol=0, abs_tol=1e-15
    ):
        raise RuntimeError(f"{name}: native integrator dt override failed")

    corrected_result = corrected.system.make_trajectory(
        n=9,
        dt=requested_dt,
        resample=False,
        standardize=False,
        return_times=True,
    )
    if not isinstance(corrected_result, tuple) or len(corrected_result) != 2:
        raise RuntimeError(f"{name}: return_times did not return a pair")
    first = np.asarray(corrected_result[0])
    second = np.asarray(corrected_result[1])
    expected_trajectory_shape = (9, corrected.observation_size)
    if first.shape == expected_trajectory_shape:
        corrected_traj = first
        corrected_times = second.reshape(-1)
    elif second.shape == expected_trajectory_shape:
        corrected_traj = second
        corrected_times = first.reshape(-1)
    else:
        raise RuntimeError(
            f"{name}: cannot identify trajectory in shapes "
            f"{first.shape}, {second.shape}"
        )
    if corrected_traj.shape != (9, corrected.observation_size):
        raise RuntimeError(f"{name}: corrected trajectory shape {corrected_traj.shape}")
    time_diffs = np.diff(corrected_times)
    if not np.allclose(time_diffs, requested_dt, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{name}: returned time grid does not use requested dt")

    native_traj = native.system.make_trajectory(
        n=8 * multiplier + 1,
        dt=expected_native_dt,
        resample=False,
        standardize=False,
    )
    native_subsampled = np.asarray(native_traj)[::multiplier]
    scale = max(1.0, float(np.max(np.abs(native_subsampled))))
    normalized_max_difference = float(
        np.max(np.abs(corrected_traj - native_subsampled)) / scale
    )
    if normalized_max_difference > 1e-5:
        raise RuntimeError(
            f"{name}: corrected/native-grid mismatch {normalized_max_difference}"
        )
    if not np.isfinite(corrected_traj).all() or not np.isfinite(native_subsampled).all():
        raise RuntimeError(f"{name}: nonfinite time-grid probe")

    period = float(corrected.period)
    periods_by_horizon = {
        str(horizon): horizon * requested_dt / period for horizon in HORIZONS
    }
    return {
        "system": name,
        "native_dt": expected_native_dt,
        "requested_dt": requested_dt,
        "realized_multiplier": requested_dt / expected_native_dt,
        "period": period,
        "time_grid_diffs": time_diffs.tolist(),
        "normalized_max_difference_vs_native_subsample": normalized_max_difference,
        "periods_by_horizon": periods_by_horizon,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multiplier", type=int, default=30)
    args = parser.parse_args()
    if args.multiplier != 30:
        raise ValueError("paper protocol requires multiplier 30")
    rows = [
        validate_system(system_key, args.multiplier)
        for system_key in DYSTS_SYSTEM_SPECS
    ]
    payload = {
        "schema_version": 1,
        "status": "passed",
        "dysts_version": importlib.metadata.version("dysts"),
        "uv_lock_sha256": _sha256(Path("uv.lock")),
        "dt_multiplier": args.multiplier,
        "systems": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
