"""Data generation and loading utilities for the SKAE benchmark suite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TrajectoryDataset:
    observations: np.ndarray
    clean: np.ndarray
    split: np.ndarray
    trajectory_ids: np.ndarray
    metadata: Dict[str, object]
    controls: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None

    def indices(self, split_name: str) -> np.ndarray:
        return np.nonzero(self.split.astype(str) == str(split_name))[0].astype(np.int64)


@dataclass
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_jsonable(self) -> Dict[str, object]:
        return {"mean": self.mean.astype(float).tolist(), "std": self.std.astype(float).tolist()}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_split(
    n_items: int,
    *,
    seed: int,
    train_count: int,
    val_count: int,
    test_count: int,
) -> np.ndarray:
    if train_count + val_count + test_count > n_items:
        raise ValueError("Requested split counts exceed number of trajectories.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    split = np.full(n_items, "unused", dtype=object)
    split[order[:train_count]] = "train"
    split[order[train_count : train_count + val_count]] = "val"
    split[order[train_count + val_count : train_count + val_count + test_count]] = "test"
    return split


def compute_normalization(values: np.ndarray, train_indices: Sequence[int]) -> Normalization:
    train = values[np.asarray(train_indices, dtype=np.int64)]
    mean = train.reshape(-1, train.shape[-1]).mean(axis=0).astype(np.float32)
    std = train.reshape(-1, train.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return Normalization(mean=mean, std=std)


def lorenz96_rhs(x: np.ndarray, forcing: float) -> np.ndarray:
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + forcing


def rk4_step_lorenz96(x: np.ndarray, *, forcing: float, dt: float) -> np.ndarray:
    k1 = lorenz96_rhs(x, forcing)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1, forcing)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2, forcing)
    k4 = lorenz96_rhs(x + dt * k3, forcing)
    return (x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).astype(np.float32)


def generate_lorenz96(
    *,
    dimension: int,
    forcing: float,
    n_trajectories: int,
    time_points: int,
    seed: int,
    dt: float = 0.005,
    sample_every: int = 10,
    warmup_steps: int = 2000,
    anchor_gap_saved: int = 25,
    reequilibration_steps: int = 200,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (forcing + 0.01 * rng.standard_normal(dimension)).astype(np.float32)
    for _ in range(int(warmup_steps)):
        x = rk4_step_lorenz96(x, forcing=forcing, dt=dt)

    anchors: List[np.ndarray] = []
    gap_steps = int(anchor_gap_saved) * int(sample_every)
    for _ in range(int(n_trajectories)):
        for _ in range(gap_steps):
            x = rk4_step_lorenz96(x, forcing=forcing, dt=dt)
        anchors.append(x.copy())

    trajectories = np.empty((int(n_trajectories), int(time_points), int(dimension)), dtype=np.float32)
    for traj_idx, anchor in enumerate(anchors):
        state = anchor + 1e-3 * rng.standard_normal(dimension).astype(np.float32)
        for _ in range(int(reequilibration_steps)):
            state = rk4_step_lorenz96(state, forcing=forcing, dt=dt)
        for t in range(int(time_points)):
            trajectories[traj_idx, t] = state
            for _ in range(int(sample_every)):
                state = rk4_step_lorenz96(state, forcing=forcing, dt=dt)
    return trajectories


def build_lorenz96_dataset(
    *,
    dimension: int,
    forcing: float,
    n_train: int,
    n_val: int,
    n_test: int,
    time_points: int,
    seed: int,
    observed_fraction: float = 1.0,
    noise_fraction: float = 0.0,
    output_dir: Path,
) -> TrajectoryDataset:
    n_total = int(n_train + n_val + n_test)
    clean = generate_lorenz96(
        dimension=dimension,
        forcing=forcing,
        n_trajectories=n_total,
        time_points=time_points,
        seed=seed,
    )
    split = trajectory_split(n_total, seed=seed + 17, train_count=n_train, val_count=n_val, test_count=n_test)
    train_clean = clean[np.nonzero(split == "train")[0]]
    channel_std = train_clean.reshape(-1, dimension).std(axis=0).astype(np.float32)
    channel_std = np.where(channel_std < 1e-6, 1.0, channel_std).astype(np.float32)

    rng = np.random.default_rng(seed + 101)
    observed_count = max(1, int(round(float(observed_fraction) * dimension)))
    sensors = np.sort(rng.choice(dimension, size=observed_count, replace=False)).astype(np.int64)
    obs = clean[..., sensors].copy()
    if float(noise_fraction) > 0:
        obs = obs + rng.normal(
            0.0,
            float(noise_fraction) * channel_std[sensors].reshape(1, 1, -1),
            size=obs.shape,
        ).astype(np.float32)
    mask = np.ones_like(obs, dtype=np.float32)
    trajectory_ids = np.asarray([f"l96_F{forcing:g}_D{dimension}_traj{i:04d}" for i in range(n_total)], dtype=object)
    metadata: Dict[str, object] = {
        "dataset_version": "generated_lorenz96_rk4_v1",
        "dimension": int(dimension),
        "forcing": float(forcing),
        "dt_integrator": 0.005,
        "sample_every": 10,
        "dt_saved": 0.05,
        "observed_fraction": float(observed_fraction),
        "sensor_indices": sensors.astype(int).tolist(),
        "noise_fraction": float(noise_fraction),
        "seed": int(seed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"lorenz96_D{dimension}_F{forcing:g}_seed{seed}.npz", clean=clean, observations=obs, split=split, sensor_indices=sensors)
    write_json(output_dir / f"lorenz96_D{dimension}_F{forcing:g}_seed{seed}_manifest.json", metadata | {"split": split.astype(str).tolist(), "trajectory_ids": trajectory_ids.astype(str).tolist()})
    return TrajectoryDataset(obs.astype(np.float32), clean.astype(np.float32), split, trajectory_ids, metadata, mask=mask)


def load_pdebench_h5(path: Path, *, max_trajectories: int = 0) -> Tuple[np.ndarray, Dict[str, object]]:
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("h5py is required to read official PDEBench HDF5 files.") from exc

    datasets: List[Tuple[str, Tuple[int, ...]]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: object) -> None:
            if hasattr(obj, "shape") and len(obj.shape) >= 4:
                datasets.append((name, tuple(int(v) for v in obj.shape)))

        handle.visititems(visitor)
        if not datasets:
            raise ValueError(f"No array-like trajectory dataset found in {path}.")
        key, shape = max(datasets, key=lambda item: len(item[1]))
        data = handle[key]
        count = int(max_trajectories) if int(max_trajectories) > 0 else int(shape[0])
        arr = np.asarray(data[:count], dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., None]
    metadata = {
        "dataset_version": "PDEBench DaRUS DOI 10.18419/DARUS-2986",
        "source_path": str(path),
        "source_sha256": file_sha256(path),
        "hdf5_dataset_key": key,
        "original_shape": list(shape),
        "loaded_shape": list(arr.shape),
    }
    return arr, metadata


def generate_pde_smoke_fixture(*, seed: int, n_trajectories: int, time_points: int, grid_size: int) -> Tuple[np.ndarray, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    fields = np.empty((n_trajectories, time_points, grid_size, grid_size, 1), dtype=np.float32)
    for i in range(n_trajectories):
        radius = 0.15 + 0.15 * rng.random()
        cx, cy = rng.uniform(-0.3, 0.3, size=2)
        height = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / radius).astype(np.float32)
        vx, vy = rng.uniform(-0.03, 0.03, size=2)
        for t in range(time_points):
            fields[i, t, ..., 0] = height
            height = 0.98 * height + 0.01 * (
                np.roll(height, 1, 0) + np.roll(height, -1, 0) + np.roll(height, 1, 1) + np.roll(height, -1, 1)
            )
            height = np.roll(np.roll(height, int(np.sign(vx)), axis=0), int(np.sign(vy)), axis=1)
    return fields, {
        "dataset_version": "synthetic_pde_smoke_fixture_not_official_pdebench",
        "warning": "This fixture only exercises the PDE pipeline; it is not a PDEBench result.",
        "grid_size": int(grid_size),
        "channels": 1,
    }


def pde_fields_to_dataset(fields: np.ndarray, metadata: Dict[str, object], *, seed: int) -> TrajectoryDataset:
    n_total = int(fields.shape[0])
    n_train = max(1, int(round(0.7 * n_total)))
    n_val = max(1, int(round(0.15 * n_total)))
    n_test = max(1, n_total - n_train - n_val)
    if n_train + n_val + n_test > n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
        n_test = 1
    split = trajectory_split(n_total, seed=seed + 23, train_count=n_train, val_count=n_val, test_count=n_test)
    flat = fields.reshape(fields.shape[0], fields.shape[1], -1).astype(np.float32)
    ids = np.asarray([f"pde_traj{i:04d}" for i in range(n_total)], dtype=object)
    return TrajectoryDataset(flat, flat.copy(), split, ids, metadata)


def load_or_fixture_silverbox(*, allow_fixture: bool, seed: int) -> TrajectoryDataset:
    try:
        import nonlinear_benchmarks  # type: ignore

        train_val, tests = nonlinear_benchmarks.Silverbox(atleast_2d=True)
        u_train = np.asarray(train_val.u, dtype=np.float32).reshape(-1, 1)
        y_train = np.asarray(train_val.y, dtype=np.float32).reshape(-1, 1)
        test_items = tests if isinstance(tests, (list, tuple)) else [tests]
        obs_parts = [np.concatenate([u_train, y_train], axis=1)]
        splits = ["estimation"]
        ids = ["silverbox_estimation"]
        for idx, item in enumerate(test_items):
            obs_parts.append(np.concatenate([np.asarray(item.u, dtype=np.float32).reshape(-1, 1), np.asarray(item.y, dtype=np.float32).reshape(-1, 1)], axis=1))
            splits.append("test")
            ids.append(f"silverbox_official_test_{idx}")
        max_len = max(part.shape[0] for part in obs_parts)
        padded = np.full((len(obs_parts), max_len, 2), np.nan, dtype=np.float32)
        mask = np.zeros_like(padded, dtype=np.float32)
        for i, part in enumerate(obs_parts):
            padded[i, : part.shape[0]] = part
            mask[i, : part.shape[0]] = 1.0
        split = np.asarray(splits, dtype=object)
        metadata = {
            "dataset_version": "official_nonlinearbenchmark_silverbox_loader",
            "source": "https://www.nonlinearbenchmark.org/benchmarks/silverbox",
            "channels": ["input_u", "output_y"],
            "loader": "nonlinear_benchmarks.Silverbox",
        }
        return TrajectoryDataset(padded, padded.copy(), split, np.asarray(ids, dtype=object), metadata, mask=mask)
    except Exception as exc:
        if not allow_fixture:
            raise RuntimeError("Official Silverbox loader failed and fixture use is disabled.") from exc

    rng = np.random.default_rng(seed)
    n = 400
    u = rng.normal(0.0, 0.5, size=n).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    for t in range(2, n):
        y[t] = 0.85 * y[t - 1] - 0.2 * y[t - 2] + 0.25 * u[t - 1] + 0.1 * y[t - 1] ** 3
    y += 0.02 * rng.standard_normal(n).astype(np.float32)
    obs = np.concatenate([u[:, None], y[:, None]], axis=1)[None]
    split = np.asarray(["estimation"], dtype=object)
    metadata = {
        "dataset_version": "synthetic_silverbox_smoke_fixture_not_official",
        "warning": "This fixture only exercises the Silverbox pipeline; it is not an official benchmark result.",
        "channels": ["input_u", "output_y"],
    }
    return TrajectoryDataset(obs, obs.copy(), split, np.asarray(["silverbox_fixture"], dtype=object), metadata, mask=np.ones_like(obs, dtype=np.float32))
