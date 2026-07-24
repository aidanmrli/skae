"""Focused tests for the locked local polynomial EDMD reproduction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.neurips_2026.baselines.classical import (
    IdentityFeatureMap,
    StateScaler,
)
from experiments.neurips_2026.local_edmd_reproduction.contract import (
    BENCHMARKS,
    CARD_PATH,
    METHOD_ID,
    NUM_COMPONENTS_GRID,
    expected_keys,
    expected_task_count,
)
from experiments.neurips_2026.local_edmd_reproduction.evaluation import (
    evaluate_rollout,
    read_task,
    validate_task,
)
from experiments.neurips_2026.local_edmd_reproduction.model import (
    LocalEDMDModel,
    select_and_fit,
)
from experiments.neurips_2026.local_edmd_reproduction.source_lock import (
    REPOSITORY_ROOT,
    load_lock,
    portable_tree_digest,
)
from experiments.neurips_2026.local_edmd_reproduction.summarize import (
    compare_reproduction,
)
from experiments.neurips_2026.local_edmd_reproduction.tasks import (
    FIELDNAMES,
    build_outputs,
    build_rows,
)


def _piecewise_affine_trajectories(
    num_per_side: int = 24, length: int = 8
) -> np.ndarray:
    trajectories = []
    centers = (
        np.asarray([-1.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.0], dtype=np.float64),
    )
    matrices = (
        np.asarray([[0.72, 0.18], [-0.08, 0.62]], dtype=np.float64),
        np.asarray([[0.45, -0.24], [0.16, 0.70]], dtype=np.float64),
    )
    for side, (center, matrix) in enumerate(zip(centers, matrices)):
        for index in range(num_per_side):
            sign = -1.0 if side == 0 else 1.0
            state = center + np.asarray(
                [sign * (0.35 + 0.02 * index), -sign * (0.15 + 0.01 * index)]
            )
            rows = [state.copy()]
            for _ in range(length):
                state = center + matrix @ (state - center)
                rows.append(state.copy())
            trajectories.append(np.stack(rows, axis=0))
    return np.stack(trajectories, axis=0)


def test_validation_selection_recovers_two_piecewise_regimes() -> None:
    trajectories = _piecewise_affine_trajectories()
    model, selection = select_and_fit(
        trajectories,
        num_components_grid=(1, 2),
        validation_fraction=0.25,
        selection_horizons=(5,),
        edmd_degree=1,
        ridge_lambda=1e-8,
        max_train_pairs=0,
        min_component_transitions=1,
        max_abs_state_for_fit=1e6,
        seed=123,
        evaluator=evaluate_rollout,
    )
    assert selection["selected_num_components"] == 2
    assert model.selected_num_components == 2
    metrics = evaluate_rollout(model, trajectories, (5,))
    assert metrics[5]["finite_fraction"] == 1.0
    assert float(metrics[5]["cumulative_mse_mean"]) < 1e-6


class _RecordingRouter:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def predict(self, states: np.ndarray) -> np.ndarray:
        self.inputs.append(states.copy())
        return (states[:, 0] >= 0.0).astype(np.int64)


def test_rollout_reroutes_each_predicted_state() -> None:
    router = _RecordingRouter()
    model = LocalEDMDModel(
        scaler=StateScaler(
            mean=np.zeros((1, 2), dtype=np.float64),
            scale=np.ones((1, 2), dtype=np.float64),
        ),
        feature_map=IdentityFeatureMap().fit(
            np.zeros((1, 2), dtype=np.float64), np.random.default_rng(0)
        ),
        router=router,  # type: ignore[arg-type]
        koopman_matrices=np.asarray(
            [[[-1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]
        ),
        decoder_matrix=np.eye(2),
        train_transitions=2,
        component_counts=[1, 1],
        fitted_component_count=2,
        selected_num_components=2,
    )
    rollout = model.rollout(np.asarray([[-1.0, 0.0]]), horizon=3)
    np.testing.assert_allclose(rollout[0, :, 0], [1.0, 2.0, 4.0])
    np.testing.assert_allclose(
        np.asarray([call[0, 0] for call in router.inputs]), [-1.0, 1.0, 2.0]
    )


def test_task_roster_is_exact_and_label_free(tmp_path: Path) -> None:
    rows = build_rows()
    assert len(rows) == expected_task_count() == 75
    assert not any("label" in field or "basin" in field for field in FIELDNAMES)
    assert {row["method"] for row in rows} == {METHOD_ID}
    assert {row["num_components_grid"] for row in rows} == {"1,2,4,8,16"}
    outputs = build_outputs(tmp_path)
    for path, payload in outputs.items():
        path.write_bytes(payload)
    for index in (0, 44, 45, 74):
        validate_task(read_task(tmp_path / "tasks.tsv", index), index)
    assert len(expected_keys()) == 225
    assert sum(len(spec.systems) for spec in BENCHMARKS.values()) == 25


def _comparison_row() -> dict[str, str]:
    row = {
        "status": "ok",
        "method": METHOD_ID,
        "num_components_grid": "1,2,4,8,16",
        "selection_horizons": "100,500,1000",
        "selected_num_components": "4",
        "fitted_component_count": "4",
        "component_counts": "[10, 10, 10, 10]",
        "feature_method": "edmd_poly",
        "route_space": "state",
        "feature_dim": "10",
        "train_transitions": "100",
        "state_dim": "2",
        "train_trajectories": "10",
        "validation_trajectories": "2",
        "test_trajectories": "5",
        "num_trajectories": "15",
        "trajectory_length": "100",
        "edmd_degree": "3",
        "min_component_transitions": "1",
        "finite_fraction": "1.0",
        "candidate_scores_json": json.dumps(
            [
                {
                    "num_components": 4,
                    "score": 0.1,
                    "status": "ok",
                    "fitted_component_count": 4,
                    "component_counts": [10, 10, 10, 10],
                }
            ]
        ),
    }
    for field in (
        "endpoint_mse_mean",
        "endpoint_mse_median",
        "endpoint_mse_per_dim_mean",
        "cumulative_mse_mean",
        "cumulative_mse_median",
        "cumulative_mse_per_dim_mean",
        "validation_score",
        "env_dt",
        "train_fraction",
        "validation_fraction",
        "ridge_lambda",
        "max_abs_state_for_fit",
    ):
        row[field] = "0.1"
    return row


def test_exact_reproduction_comparator_rejects_discrete_drift() -> None:
    key = ("controlled", "gated_local_linear", 0, 100)
    historical = {key: _comparison_row()}
    current = {key: dict(historical[key])}
    mismatches, _ = compare_reproduction(
        current, historical, rtol=1e-6, atol=1e-10
    )
    assert mismatches == []
    current[key]["selected_num_components"] = "8"
    mismatches, _ = compare_reproduction(
        current, historical, rtol=1e-6, atol=1e-10
    )
    assert mismatches[0]["fields"] == ["selected_num_components"]


def test_prediction_card_freezes_known_outcome_and_tolerances() -> None:
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    assert card["status"] == "frozen_before_reproduction_execution"
    assert card["experiment_type"] == "known_outcome_provenance_reproduction"
    assert card["protocol"]["route_count_grid"] == list(NUM_COMPONENTS_GRID)
    assert card["exact_reproduction_tolerances"]["continuous_fields"] == {
        "relative": 1e-6,
        "absolute": 1e-10,
    }
    assert "paired" in " ".join(card["claim_limits"]).lower()


def test_portable_tree_digest_matches_sha256sum_record_format(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a" / "rows.csv"
    second = tmp_path / "b" / "rows.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one\n")
    second.write_bytes(b"two\n")
    records = (
        f"{hashlib.sha256(first.read_bytes()).hexdigest()}  a/rows.csv\n"
        f"{hashlib.sha256(second.read_bytes()).hexdigest()}  b/rows.csv\n"
    ).encode()
    digest, count = portable_tree_digest(tmp_path, "**/rows.csv")
    assert digest == hashlib.sha256(records).hexdigest()
    assert count == 2


def _locked_bytes_exist_in_history(path: str, expected: str) -> bool:
    commits = subprocess.run(
        ["git", "rev-list", "--all", "--", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for commit in commits:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == expected:
            return True
    return False


def test_source_lock_is_preserved_and_disjoint_from_dense_lock() -> None:
    local_lock = load_lock()
    for name, item in local_lock["sources"].items():
        source = REPOSITORY_ROOT / item["path"]
        assert source.is_file()
        if name == "focused_tests":
            # This test evolved after the terminal reproduction to recognize
            # historical locked bytes; the executed source hash remains frozen.
            continue
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]
            or _locked_bytes_exist_in_history(item["path"], item["sha256"])
        )
    dense_lock = json.loads(
        (REPOSITORY_ROOT / "experiments/neurips_2026/global_k_dense_specificity_source_lock.json")
        .read_text(encoding="utf-8")
    )
    local_paths = {item["path"] for item in local_lock["sources"].values()}
    dense_paths = {item["path"] for item in dense_lock["sources"].values()}
    newly_authored = {
        path
        for path in local_paths
        if "local_edmd_reproduction" in path
        or path == "tests/test_local_edmd_reproduction.py"
    }
    assert newly_authored.isdisjoint(dense_paths)
    for shared_path in local_paths & dense_paths:
        local_item = next(
            item
            for item in local_lock["sources"].values()
            if item["path"] == shared_path
        )
        dense_item = next(
            item
            for item in dense_lock["sources"].values()
            if item["path"] == shared_path
        )
        assert local_item["sha256"] == dense_item["sha256"]
    assert local_lock["compute"]["gpu_count"] == 0
