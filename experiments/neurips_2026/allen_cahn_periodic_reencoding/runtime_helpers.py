"""GPU-run serialization and forecast-cross helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from experiments.neurips_2026.allen_cahn_periodic_reencoding.core import (
    evaluate_model_packed,
    validate_period_candidates,
)
from experiments.neurips_2026.allen_cahn_periodic_reencoding.io import (
    field_payload,
    sha256_path,
    torch_save_once,
    write_json_once,
)


def save_datasets(
    fields: torch.Tensor,
    card: dict[str, Any],
    *,
    role: str,
    seeds: list[int],
    root: Path,
) -> tuple[Path, str]:
    if fields.shape[0] != 3:
        raise ValueError("Expected three packed datasets")
    records = []
    for dataset_index, seed in enumerate(seeds):
        path = root / "data" / f"{role}_seed{seed}_fields.pt"
        payload = field_payload(
            fields[dataset_index],
            card,
            role=role,
            dataset_index=dataset_index,
            seed=seed,
        )
        torch_save_once(path, payload)
        records.append(
            {
                "role": role,
                "dataset_index": dataset_index,
                "dataset_seed": seed,
                "path": str(path),
                "sha256": sha256_path(path),
                "shape": list(payload["fields"].shape),
                "storage_bytes": int(payload["fields"].untyped_storage().nbytes()),
            }
        )
    manifest_path = root / f"{role}_data_manifest.json"
    write_json_once(
        manifest_path,
        {
            "schema_version": 1,
            "protocol_id": card["protocol_id"],
            "role": role,
            "datasets": records,
        },
    )
    return manifest_path, sha256_path(manifest_path)


def cadence_grid(card: dict[str, Any]) -> list[str | int]:
    grid = list(card["cadence_selection"]["cadence_grid"])
    if not grid or grid[0] != "direct":
        raise RuntimeError("Frozen cadence grid must begin with direct")
    periods = validate_period_candidates(
        [int(value) for value in grid[1:]],
        horizon=int(card["system"]["validation_horizon_steps"]),
    )
    if grid != ["direct", *periods]:
        raise RuntimeError("Cadence grid serialization drifted")
    return grid


def _row(
    record: dict[str, Any],
    *,
    arm: str,
    model_seed: int,
    dataset_seed: int,
    cadence: str | int,
) -> dict[str, Any]:
    if int(record["horizon"]) not in {200, 400}:
        raise RuntimeError("Unexpected scoring horizon")
    return {
        "arm": arm,
        "model_seed": int(model_seed),
        "dataset_seed": int(dataset_seed),
        "cadence": cadence,
        "horizon_steps": int(record["horizon"]),
        "trajectory_count": int(record["trajectory_count"]),
        "state_size": int(record["state_size"]),
        "instantaneous_field_mse": record["instantaneous_field_mse"],
        "cumulative_field_mse": record["cumulative_field_mse"],
        "instantaneous_persistence_mse": record["instantaneous_persistence_mse"],
        "cumulative_persistence_mse": record["cumulative_persistence_mse"],
        "instantaneous_model_over_persistence": record[
            "instantaneous_model_over_persistence"
        ],
        "cumulative_model_over_persistence": record[
            "cumulative_model_over_persistence"
        ],
    }


def evaluate_cross(
    specs_and_models: Iterable[tuple[Any, torch.nn.Module]],
    fields: torch.Tensor,
    *,
    dataset_seeds: list[int],
    cadences: list[str | int],
    horizon: int,
    batch_size: int,
    max_decode_segment: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec, model in specs_and_models:
        for cadence in cadences:
            records = evaluate_model_packed(
                model,
                fields,
                horizon=horizon,
                period=None if cadence == "direct" else int(cadence),
                batch_size=batch_size,
                max_decode_segment=max_decode_segment,
            )
            if len(records) != len(dataset_seeds):
                raise RuntimeError("Scorer did not preserve the dataset cross")
            for dataset_index, record in enumerate(records):
                rows.append(
                    _row(
                        record,
                        arm=str(spec.arm),
                        model_seed=int(spec.seed),
                        dataset_seed=int(dataset_seeds[dataset_index]),
                        cadence=cadence,
                    )
                )
    return rows
