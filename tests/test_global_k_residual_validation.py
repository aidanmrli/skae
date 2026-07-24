from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.neurips_2026.global_k_residual_forecast.shard_validation import (
    _validate_provenance,
)
from experiments.neurips_2026.global_k_residual_forecast.summarize import (
    _curve_summary,
)
from experiments.neurips_2026.global_k_residual_forecast.validation import (
    DENSE_METHODS,
    H500_METHODS,
    SPARSE_METHODS,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "neurips_2026" / "global_k_residual_forecast"


def _tasks() -> dict:
    return json.loads((PACKAGE / "task_manifest.json").read_text())


def test_checkpoint_provenance_requires_exact_arm_counts_and_bindings(
    tmp_path: Path,
) -> None:
    tasks = _tasks()
    task = tasks["tasks"][0]
    expected_data = tmp_path / "outcome_free_data" / "manifest.json"
    evaluator_hash = "e" * 64
    provenance = {
        "sparse": {
            "checkpoint_path": task["sparse_checkpoint"]["path"],
            "checkpoint_sha256": task["sparse_checkpoint"]["sha256"],
            "v2_exact_checkpoint_audit_passed": True,
            "checkpoint_step": 100,
            "trainable_parameter_counts": tasks["provenance_contract"]
            ["sparse_trainable_parameter_counts"],
        },
        "dense": {
            "checkpoint_path": task["dense_checkpoint"]["path"],
            "checkpoint_sha256": task["dense_checkpoint"]["sha256"],
            "v2_exact_checkpoint_audit_passed": True,
            "checkpoint_step": 100,
            "trainable_parameter_counts": tasks["provenance_contract"]
            ["dense_trainable_parameter_counts"],
        },
        "data_manifest_path": str(expected_data),
        "data_manifest_sha256": "d" * 64,
        "evaluator_sha256": evaluator_hash,
        "git_commit": "a" * 40,
        "gpu": {
            "name": "NVIDIA A100-SXM4-80GB",
            "total_memory_bytes": 80 * 1024**3,
        },
    }
    assert _validate_provenance(
        {"provenance": provenance},
        task,
        tasks,
        expected_data_path=expected_data,
        expected_evaluator=evaluator_hash,
    ) == "d" * 64
    broken = copy.deepcopy(provenance)
    broken["sparse"]["trainable_parameter_counts"]["total"] = 95103
    with pytest.raises(RuntimeError, match="checkpoint provenance"):
        _validate_provenance(
            {"provenance": broken},
            task,
            tasks,
            expected_data_path=expected_data,
            expected_evaluator=evaluator_hash,
        )
    forged_tasks = copy.deepcopy(tasks)
    forged_tasks["provenance_contract"]["sparse_trainable_parameter_counts"][
        "total"
    ] = 95103
    with pytest.raises(RuntimeError, match="parameter-count contract"):
        _validate_provenance(
            {"provenance": broken},
            task,
            forged_tasks,
            expected_data_path=expected_data,
            expected_evaluator=evaluator_hash,
        )


def _curve_row(name: str) -> dict[str, list[float | None]]:
    length = 500 if name in H500_METHODS else 200
    curve: list[float | None] = [1.0] * length
    if name == "sparse_routed_residual":
        curve[300] = None
    return {"mean_mse_curve": curve}


def test_curve_summary_preserves_valid_h200_prefix_when_h500_fails() -> None:
    sparse_methods = {name: _curve_row(name) for name in SPARSE_METHODS}
    dense_methods = {name: _curve_row(name) for name in DENSE_METHODS}
    dataset = {
        "sparse": {"methods": sparse_methods},
        "dense": {"methods": dense_methods},
    }
    shards = [{"dataset_rows": [dataset, dataset, dataset]} for _ in range(10)]
    summary = _curve_summary(shards)["sparse_routed_residual"]
    assert summary["curve_length"] == 500
    assert summary["finite_h200_model_seed_count"] == 10
    assert summary["finite_full_curve_model_seed_count"] == 0
    assert summary["mean_over_model_seeds_and_datasets"][:200] == [1.0] * 200
    assert summary["mean_over_model_seeds_and_datasets"][300] is None
    assert all(
        curve[:200] == [1.0] * 200
        for curve in summary["model_seed_curves_after_dataset_averaging"]
    )
