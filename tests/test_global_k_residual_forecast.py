from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.neurips_2026.global_k_residual_forecast import protocol
from experiments.neurips_2026.global_k_residual_forecast.evaluate import (
    _load_data_manifest,
)
from experiments.neurips_2026.global_k_residual_forecast.rollout import (
    _method_payload,
    projected_decode_components,
)
from experiments.neurips_2026.global_k_residual_forecast.routing import (
    evenly_spaced_indices,
    fit_codebook,
)
from experiments.neurips_2026.global_k_residual_forecast.telemetry import (
    _assess_one,
    _minimum_time_rolling,
    _read_trace,
    _validate_smoke_shard,
)
from experiments.neurips_2026.global_k_residual_forecast.validation import (
    validate_method,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "neurips_2026" / "global_k_residual_forecast"
SCRIPTS = ROOT / "scripts" / "neurips_2026" / "global_k_residual_forecast"


def _card() -> dict:
    return json.loads((PACKAGE / "prediction_card.json").read_text())


def _tasks() -> dict:
    return json.loads((PACKAGE / "task_manifest.json").read_text())


def test_card_freezes_long_physical_horizons_and_no_label_routing() -> None:
    card = _card()
    assert card["benchmark"]["dt"] == 0.04
    assert card["forecast_protocol"]["primary_horizon_steps"] == 200
    assert card["forecast_protocol"]["stress_horizon_steps"] == 500
    assert card["benchmark"]["primary_physical_time"] == 8.0
    assert card["benchmark"]["stress_physical_time"] == 20.0
    assert card["benchmark"]["training_or_deployment_uses_basin_labels_or_count"] is False
    assert card["forecast_protocol"]["unchanged_global_k"] is True
    assert card["forecast_protocol"]["local_operator_fit"] is False
    assert "autonomous nonlinear" in card["forecast_protocol"]["semantic_boundary"]
    assert "pure K^h" in card["claim_boundary"]


def test_task_roster_matches_authenticated_v2_audit_summary() -> None:
    tasks = _tasks()["tasks"]
    assert [row["task_id"] for row in tasks] == list(range(10))
    assert [row["model_seed"] for row in tasks] == list(range(100, 110))
    v2 = _card()["authenticated_v2_inputs"]
    audit_path = Path(v2["audit_summary_path"])
    audit = protocol.load_verified_json(
        audit_path, v2["audit_summary_sha256"], "V2 audit summary"
    )
    expected = {
        (row["arm"], int(row["seed"])): row["checkpoint_sha256"]
        for row in audit["rows"]
    }
    for row in tasks:
        seed = row["model_seed"]
        assert row["sparse_checkpoint"]["sha256"] == expected[("sparse", seed)]
        assert row["dense_checkpoint"]["sha256"] == expected[("dense", seed)]
        assert len(row["sparse_checkpoint"]["sha256"]) == 64
        assert len(row["dense_checkpoint"]["sha256"]) == 64


def test_all_prospective_whole_trajectory_seeds_are_unique() -> None:
    corpora = _card()["outcome_free_trajectory_corpora"]
    seeds = [
        corpora["route_fit"]["seed"],
        corpora["route_audit"]["seed"],
        *corpora["evaluation"]["seeds"],
        corpora["smoke_evaluation"]["seed"],
    ]
    assert len(seeds) == len(set(seeds)) == 6
    assert not set(seeds) & set(range(100, 110))
    assert not set(seeds) & {20260726, 20260727, 20260728, 20260729, 20260730}


def test_sign_pair_nulls_are_unique_nonidentity_and_preserve_geometry() -> None:
    permutations = protocol.stable_sign_pair_permutations(12, 32, 17)
    assert permutations.shape == (32, 12)
    assert len({row.tobytes() for row in permutations}) == 32
    identity = np.arange(12)
    assert all(not np.array_equal(row, identity) for row in permutations)
    assert all(np.array_equal(row[6:] - 6, row[:6]) for row in permutations)
    supports = np.asarray(
        [[1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
         [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0]],
        dtype=bool,
    )
    base_cardinality = supports.sum(axis=1)
    base_intersection = np.logical_and(supports[0], supports[1]).sum()
    for permutation in permutations:
        permuted = supports[:, permutation]
        assert np.array_equal(permuted.sum(axis=1), base_cardinality)
        assert np.logical_and(permuted[0], permuted[1]).sum() == base_intersection


def test_nearest_family_is_total_deterministic_and_has_no_label_argument() -> None:
    latent = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    representatives = torch.tensor(
        [[True, False, False, False], [False, True, False, False]]
    )
    assignment, similarity = protocol.nearest_family(latent, representatives, 1e-3)
    assert assignment.tolist() == [0, 1]
    assert torch.equal(similarity, torch.ones(2))
    assert set(inspect.signature(fit_codebook).parameters) == {
        "model", "fit_trajectories", "card"
    }


def test_null_scale_match_rows_are_evenly_spaced_over_full_route_fit_corpus() -> None:
    indices = evenly_spaced_indices(131072, 8192)
    assert indices.shape == (8192,)
    assert int(indices[0]) == 0
    assert int(indices[-1]) == 131071
    assert int(torch.unique(indices).numel()) == 8192
    assert int(indices[4096]) > 65000


class _LinearDummy:
    def __init__(self) -> None:
        self._k = torch.tensor([[1.0, 2.0], [0.0, 1.0]])
        self._decoder = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

    def kmatrix(self) -> torch.Tensor:
        return self._k

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent @ self._decoder


def test_projected_components_are_exact_frozen_predictor_formula() -> None:
    model = _LinearDummy()
    latent = torch.tensor([[[1.0, 4.0], [2.0, 5.0]]])
    projector = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    decoded_step, decoded_source = projected_decode_components(
        model, latent, projector
    )
    source = latent * projector
    expected_step = ((source @ model.kmatrix()) * projector) @ model._decoder
    expected_source = source @ model._decoder
    assert torch.equal(decoded_step, expected_step)
    assert torch.equal(decoded_source, expected_source)


def test_nonfinite_policy_suppresses_complete_endpoint_not_prefix() -> None:
    curve = torch.ones(500)
    curve[300] = torch.nan
    payload, _ = _method_payload(
        curve,
        torch.ones(4) * 200,
        torch.ones(4) * 500,
        finite_h200=True,
        finite_h500=False,
    )
    assert payload["through_h200_mse"] == 1.0
    assert payload["through_h500_mse"] is None
    assert payload["mean_mse_curve"][300] is None
    validate_method("sparse_routed_residual", payload)


def test_h500_publication_requires_h200_mechanism_support() -> None:
    assert protocol.publish_h500_extension(True, True) is True
    assert protocol.publish_h500_extension(True, False) is False
    assert protocol.publish_h500_extension(False, True) is False
    assert protocol.publish_h500_extension(False, False) is False


def test_exact_statistics_and_holm_are_deterministic() -> None:
    assert protocol.exact_sign_flip_pvalue([1.0, 2.0, 3.0]) == 1 / 8
    adjusted = protocol.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.2}
    first = protocol.paired_bootstrap_reduction_interval(
        np.ones(10), np.ones(10) * 2, replicates=1000, seed=9
    )
    second = protocol.paired_bootstrap_reduction_interval(
        np.ones(10), np.ones(10) * 2, replicates=1000, seed=9
    )
    assert first == second == (0.5, 0.5)


def test_gpu_scripts_request_long_a100_and_dependency_chain() -> None:
    forecast = (SCRIPTS / "run_forecast.sh").read_text()
    queue = (SCRIPTS / "queue.sh").read_text()
    assert "#SBATCH --partition=long" in forecast
    assert "#SBATCH --gres=gpu:a100l:1" in forecast
    assert (
        "nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total"
        in forecast
    )
    assert "--array=0-9%10" in queue
    assert "afterok:${SMOKE_GATE_JOB_ID}" in queue
    assert "afterok:${SCIENCE_GATE_JOB_ID}" in queue
    assert "global_k_residual_forecast.preflight" in queue
    assert "sha256sum --check --strict --status" in queue
    for worker in (
        "run_prepare.sh", "run_forecast.sh", "run_telemetry.sh", "run_summary.sh"
    ):
        text = (SCRIPTS / worker).read_text()
        assert 'ROOT_DIR="${PROJECT_DIR}"\nsource scripts/common/cluster_env.sh' in text
    assert "GPU telemetry monitor exited unexpectedly" in forecast
    assert "An explicit post-compute sample brackets" in forecast
    gpu = _card()["gpu_utilization_gate"]
    assert gpu["minimum_compute_window_p10_utilization_percent"] >= 80
    assert gpu["minimum_rolling_utilization_percent"] >= 80


def test_launch_authorization_is_one_field_lifecycle_transition(tmp_path: Path) -> None:
    preauthorization = copy.deepcopy(_card())
    preauthorization["freeze"]["launch_authorized"] = False
    authorized = copy.deepcopy(preauthorization)
    authorized["freeze"]["launch_authorized"] = True
    protocol.validate_launch_authorization_transition(preauthorization, authorized)
    changed = copy.deepcopy(authorized)
    changed["hypothesis"] += " changed"
    with pytest.raises(RuntimeError, match="only"):
        protocol.validate_launch_authorization_transition(preauthorization, changed)

    authorized_path = tmp_path / "authorized_card.json"
    source_path = tmp_path / "source_manifest.sha256"
    stable_source = Path("pyproject.toml")
    source_path.write_text(
        f"{protocol.sha256_path(stable_source)}  {stable_source.as_posix()}\n"
    )
    authorized["freeze"]["source_manifest_sha256"] = protocol.sha256_path(source_path)
    authorized_path.write_text(json.dumps(authorized))
    loaded, _, freeze = protocol.load_frozen_protocol(
        card_path=authorized_path,
        task_path=PACKAGE / "task_manifest.json",
        source_manifest_path=source_path,
        expected_card_sha256=protocol.sha256_path(authorized_path),
        expected_task_sha256=protocol.sha256_path(PACKAGE / "task_manifest.json"),
        expected_source_manifest_sha256=protocol.sha256_path(source_path),
    )
    assert loaded["freeze"]["launch_authorized"] is True
    assert freeze["source_manifest_sha256"] == authorized["freeze"][
        "source_manifest_sha256"
    ]


def test_telemetry_rolling_gate_and_source_freeze() -> None:
    epoch = np.arange(61, dtype=np.float64)
    values = np.full(61, 80.0)
    assert _minimum_time_rolling(epoch, values, 30.0) == 80.0
    assert np.isnan(_minimum_time_rolling(epoch[:20], values[:20], 30.0))
    card = _card()
    assert card["freeze"]["source_manifest_sha256"] != "PENDING_FINAL_SOURCE_FREEZE"
    assert card["freeze"]["task_manifest_sha256"] == protocol.sha256_path(
        PACKAGE / "task_manifest.json"
    )
    assert card["freeze"]["source_manifest_sha256"] == protocol.sha256_path(
        PACKAGE / "source_manifest.sha256"
    )
    source_rows = [
        line.split(maxsplit=1)[1]
        for line in (PACKAGE / "source_manifest.sha256").read_text().splitlines()
        if line.strip()
    ]
    locked = set(source_rows)
    assert {
        "experiments/__init__.py",
        "experiments/neurips_2026/__init__.py",
        "experiments/neurips_2026/global_k_residual_forecast/__init__.py",
        "experiments/neurips_2026/global_k_distinct_laws_v2_tasks.py",
        "experiments/neurips_2026/global_k_distinct_laws_v2_math.py",
        "skae/__init__.py",
        "skae/evaluation.py",
        "skae/training/__init__.py",
        "skae/training/runner.py",
        "skae/dysts_cache_profiles.py",
        "skae/runtime_paths.py",
        "scripts/common/cluster_env.sh",
        "pyproject.toml",
        "uv.lock",
    } <= locked
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        protocol.verify_source_manifest(PACKAGE / "source_manifest.sha256")


def test_telemetry_schema_identity_ranges_and_monotonicity(tmp_path: Path) -> None:
    path = tmp_path / "trace.csv"
    path.write_text(
        "epoch_seconds,gpu_uuid,gpu_name,utilization_gpu,memory_used_mib,memory_total_mib\n"
        "1.0,GPU-a,NVIDIA A100,90,100,81920\n"
        "2.0,GPU-a,NVIDIA A100,95,110,81920\n"
    )
    trace = _read_trace(path)
    assert trace["gpu_uuid"] == "GPU-a"
    assert trace["gpu_name"] == "NVIDIA A100"
    path.write_text(
        "epoch_seconds,gpu_uuid,gpu_name,utilization_gpu,memory_used_mib,memory_total_mib\n"
        "2.0,GPU-a,NVIDIA A100,90,100,81920\n"
        "1.0,GPU-b,NVIDIA A100,101,100,81920\n"
    )
    try:
        _read_trace(path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Malformed telemetry must fail closed")


def test_telemetry_marker_identity_and_full_interval_bracketing(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.csv"
    rows = [
        "epoch_seconds,gpu_uuid,gpu_name,utilization_gpu,memory_used_mib,memory_total_mib"
    ]
    rows.extend(
        f"{epoch},GPU-a,NVIDIA A100-SXM4-80GB,90,1000,81920"
        for epoch in range(99, 142)
    )
    trace_path.write_text("\n".join(rows) + "\n")
    window_path = tmp_path / "window.json"
    window = {
        "schema_version": 1,
        "protocol_id": _card()["protocol_id"],
        "artifact_role": "forecast_compute_window",
        "mode": "scientific",
        "task_id": 3,
        "start_epoch_seconds": 100.0,
        "end_epoch_seconds": 140.0,
        "elapsed_seconds": 40.0,
    }
    window_path.write_text(json.dumps(window))
    assessed = _assess_one(
        trace_path,
        window_path,
        _card()["gpu_utilization_gate"],
        protocol_id=_card()["protocol_id"],
        mode="scientific",
        task_id=3,
        freeze={"card_sha256": "a"},
    )
    assert assessed["passed"] is True
    assert assessed["checks"]["trace_brackets_compute_start"] is True
    assert assessed["checks"]["trace_brackets_compute_end"] is True
    assert assessed["task_id"] == 3
    window["mode"] = "smoke"
    window_path.write_text(json.dumps(window))
    with pytest.raises(RuntimeError, match="identity"):
        _assess_one(
            trace_path,
            window_path,
            _card()["gpu_utilization_gate"],
            protocol_id=_card()["protocol_id"],
            mode="scientific",
            task_id=3,
            freeze={"card_sha256": "a"},
        )


def test_smoke_validation_checks_every_field_and_freeze(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json"
    freeze = {"card_sha256": "a", "task_manifest_sha256": "b"}
    smoke = {
        "schema_version": 1,
        "protocol_id": _card()["protocol_id"],
        "artifact_role": "outcome_blind_gpu_smoke",
        "task_id": 0,
        "all_required_predictions_finite": True,
        "exact_method_count": 41,
        "route_fit_completed": True,
        "route_audit_completed": True,
        "null_scale_matching_completed": True,
        "forecast_metrics_labels_and_alignment_values_persisted": False,
        "outcomes_inspected": False,
        "elapsed_seconds": 40.0,
        "freeze": freeze,
    }
    path.write_text(json.dumps(smoke))
    assert all(
        _validate_smoke_shard(
            path, protocol_id=_card()["protocol_id"], freeze=freeze
        ).values()
    )
    smoke["freeze"] = {"card_sha256": "wrong"}
    path.write_text(json.dumps(smoke))
    assert not all(
        _validate_smoke_shard(
            path, protocol_id=_card()["protocol_id"], freeze=freeze
        ).values()
    )


def test_data_manifest_semantics_are_bound_to_card(tmp_path: Path) -> None:
    card = _card()
    card["outcome_free_trajectory_corpora"] = {
        "route_fit": {"seed": 11, "trajectory_count": 2, "horizon_steps": 1},
        "route_audit": {"seed": 12, "trajectory_count": 2, "horizon_steps": 1},
        "evaluation": {"seeds": [13, 14, 15], "trajectory_count_each": 2, "horizon_steps": 1},
        "smoke_evaluation": {"seed": 16, "dataset_index": 0, "trajectory_count": 2, "horizon_steps": 1},
    }
    freeze = {"card_sha256": "a", "task_manifest_sha256": "b"}
    data_dir = tmp_path / "outcome_free_data"
    data_dir.mkdir()
    specs = [
        ("route_fit", None, 11, "route_fit.pt"),
        ("route_audit", None, 12, "route_audit.pt"),
        ("evaluation", 0, 13, "evaluation_0.pt"),
        ("evaluation", 1, 14, "evaluation_1.pt"),
        ("evaluation", 2, 15, "evaluation_2.pt"),
        ("smoke_evaluation", 0, 16, "smoke_evaluation_0.pt"),
    ]
    rows = []
    for role, index, seed, filename in specs:
        path = data_dir / filename
        metadata = {
            "protocol_id": card["protocol_id"],
            "role": role,
            "seed": seed,
            "shape": [2, 2, 2],
            "contains_forecast_or_representation_outcomes": False,
        }
        row = {"role": role, "seed": seed, "path": str(path)}
        if index is not None:
            metadata["dataset_index"] = index
            row["dataset_index"] = index
        torch.save({"trajectories": torch.zeros(2, 2, 2), "metadata": metadata}, path)
        row["sha256"] = protocol.sha256_path(path)
        rows.append(row)
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": card["protocol_id"],
                "artifact_role": "outcome_free_physical_trajectory_manifest",
                "freeze": freeze,
                "rows": rows,
            }
        )
    )
    assert len(_load_data_manifest(tmp_path, card, freeze)["rows"]) == 6
    rows[0]["seed"] = 999
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": card["protocol_id"],
                "artifact_role": "outcome_free_physical_trajectory_manifest",
                "freeze": freeze,
                "rows": rows,
            }
        )
    )
    try:
        _load_data_manifest(tmp_path, card, freeze)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Semantic seed drift must fail closed")
