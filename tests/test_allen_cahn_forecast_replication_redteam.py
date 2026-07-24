from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments.neurips_2026.allen_cahn_forecast_replication.core import (
    HORIZON,
    curve_metrics,
    direct_rollout,
    evaluate_model_packed,
    evaluate_model_sequential,
    realized_rng_streams,
    validate_crossed_rows,
    validate_curve_record,
)
from experiments.neurips_2026.allen_cahn_forecast_replication.io import (
    CARD_PATH,
    duplicate_safe_json,
    field_payload,
    load_card,
    load_fields_only,
    load_pinned_module,
    pinned_source,
    sha256_path,
    torch_save_once,
    validate_field_payload,
    write_json_once,
)
from experiments.neurips_2026.allen_cahn_forecast_replication.statistics import (
    descriptive_arm_curves,
    exact_paired_sign_flip,
    paired_seed_bootstrap,
)
from experiments.neurips_2026.allen_cahn_forecast_replication.run import (
    configure_precision,
)
from experiments.neurips_2026.allen_cahn_forecast_replication.summarize import (
    verify_outcome_guard_receipt,
)


CARD, _CARD_HASH = load_card(CARD_PATH)


class ToyKoopman(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kmat = torch.nn.Parameter(
            torch.tensor([[0.9, 0.4], [-0.2, 1.1]], dtype=torch.float32)
        )
        self.encode_calls = 0

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        self.encode_calls += 1
        return values

    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values


def synthetic_fields() -> torch.Tensor:
    initial = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)[:, None]
    initial = torch.cat((initial, 0.5 * initial + 0.25), dim=1)
    initial = torch.stack([initial + 0.1 * index for index in range(3)])
    times = torch.arange(HORIZON + 1, dtype=torch.float32).view(1, 1, -1, 1)
    velocity = torch.tensor([0.002, -0.001], dtype=torch.float32).view(1, 1, 1, 2)
    return initial[:, :, None, :] + times * velocity


def test_duplicate_json_keys_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        duplicate_safe_json(duplicate)


def test_field_only_serialization_exact_schema_and_recursive_firewall(tmp_path: Path) -> None:
    card = deepcopy(CARD)
    card["field_only_firewall"]["saved_field_shape"] = [2, 3, 2, 2, 1]
    fields = torch.zeros((2, 3, 2, 2, 1), dtype=torch.float32)
    payload = field_payload(fields, card, dataset_index=0, seed=1775404171)
    assert set(payload) == {"fields", "split_indices", "metadata"}
    assert set(payload["split_indices"]) == {"val"}
    assert set(payload["metadata"]) == set(card["field_only_firewall"]["exact_metadata_keys"])
    output = tmp_path / "field_only.pt"
    torch_save_once(output, payload)
    loaded = load_fields_only(
        output,
        card,
        expected_sha256=sha256_path(output),
        dataset_index=0,
        seed=1775404171,
    )
    assert loaded.shape == (2, 3, 4)

    bad_metadata = deepcopy(payload)
    bad_metadata["metadata"]["basin_hint"] = 3
    with pytest.raises(AssertionError):
        validate_field_payload(
            bad_metadata, card, dataset_index=0, seed=1775404171
        )
    bad_top_level = deepcopy(payload)
    bad_top_level["selected_center_indices"] = torch.zeros(1)
    with pytest.raises(AssertionError):
        validate_field_payload(
            bad_top_level, card, dataset_index=0, seed=1775404171
        )


def test_field_and_curve_nonfinite_values_invalidate() -> None:
    card = deepcopy(CARD)
    card["field_only_firewall"]["saved_field_shape"] = [2, 3, 2, 2, 1]
    fields = torch.zeros((2, 3, 2, 2, 1), dtype=torch.float32)
    fields[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError):
        field_payload(fields, card, dataset_index=0, seed=1775404171)

    truth = synthetic_fields()[0, :, 1:]
    x0 = synthetic_fields()[0, :, 0]
    prediction = truth.clone()
    prediction[0, 0, 0] = torch.inf
    with pytest.raises(FloatingPointError):
        curve_metrics(prediction, truth, x0)


def test_exact_f_linear_orientation_and_one_encode() -> None:
    model = ToyKoopman()
    initial = torch.tensor([[1.0, 2.0], [-0.5, 0.25]], dtype=torch.float32)
    predicted = direct_rollout(model, initial, horizon=3)
    expected_steps = []
    latent = initial
    for _ in range(3):
        latent = latent @ model.kmat.T
        expected_steps.append(latent)
    expected = torch.stack(expected_steps, dim=1)
    torch.testing.assert_close(predicted, expected, rtol=0, atol=0)
    assert model.encode_calls == 1
    wrong_first_step = initial @ model.kmat
    assert not torch.allclose(predicted[:, 0], wrong_first_step)


def test_pinned_import_root_isolation_and_direct_reference_equivalence() -> None:
    before = tuple(sys.path)
    module = load_pinned_module(pinned_source(CARD, "checkpoint_model"))
    assert tuple(sys.path) == before
    assert Path(module.__file__).resolve() == Path(
        pinned_source(CARD, "checkpoint_model")["path"]
    ).resolve()
    torch.manual_seed(7)
    config = module.SpatialConvKoopmanConfig(
        grid_size=4,
        channels=2,
        z_dim=8,
        hidden_channels=4,
        num_blocks=1,
        encoder_kind="dense",
        lista_loops=1,
        lista_alpha=0.0,
        decoder_kind="upsample",
        k_init_scale=0.0,
        dense_activation="tanh",
        conv_activation="tanh",
        padding_mode="circular",
    )
    model = module.SpatialConvKoopman(config).float().eval()
    with torch.no_grad():
        model.kmat.copy_(0.05 * torch.randn_like(model.kmat))
    initial = torch.randn(3, 32, dtype=torch.float32)
    ours = direct_rollout(model, initial, horizon=4)
    _reference_latents, reference = model.rollout_observation_discrete(initial, horizon=4)
    torch.testing.assert_close(ours, reference, rtol=1e-6, atol=1e-7)


def test_synthetic_packed_and_sequential_scoring_are_equivalent() -> None:
    fields = synthetic_fields()
    packed_model = ToyKoopman()
    packed = evaluate_model_packed(packed_model, fields, batch_size=256)
    assert packed_model.encode_calls == 3
    sequential_model = ToyKoopman()
    sequential_model.load_state_dict(packed_model.state_dict())
    sequential = evaluate_model_sequential(sequential_model, fields)
    assert sequential_model.encode_calls == 3
    for packed_dataset, sequential_dataset in zip(packed, sequential, strict=True):
        for key in packed_dataset:
            torch.testing.assert_close(
                torch.tensor(packed_dataset[key]),
                torch.tensor(sequential_dataset[key]),
                rtol=1e-6,
                atol=1e-7,
            )


def test_full_curve_identities_and_unclipped_ratios() -> None:
    fields = synthetic_fields()[0]
    truth = fields[:, 1:]
    prediction = 0.95 * truth
    record = curve_metrics(prediction, truth, fields[:, 0])
    validate_curve_record(record)
    instantaneous = np.asarray(record["instantaneous_field_mse"])
    cumulative = np.asarray(record["cumulative_field_mse"])
    np.testing.assert_allclose(
        cumulative,
        np.cumsum(instantaneous) / np.arange(1, HORIZON + 1),
        rtol=1e-12,
        atol=1e-14,
    )
    corrupted = deepcopy(record)
    corrupted["cumulative_field_mse"][50] += 1e-3
    with pytest.raises(AssertionError):
        validate_curve_record(corrupted)


def test_exact_test_and_paired_bootstrap_are_frozen_and_deterministic() -> None:
    exact = exact_paired_sign_flip(np.ones(10))
    assert exact["enumerated_sign_vectors"] == 1024
    assert exact["one_sided_exact_p"] == pytest.approx(1.0 / 1024.0)
    assert exact["comparison"] == "T_perm >= T_observed_literal_no_tolerance"
    first = paired_seed_bootstrap([0.9] * 10, [1.0] * 10)
    second = paired_seed_bootstrap([0.9] * 10, [1.0] * 10)
    assert first == second
    assert first["replicates"] == 100_000
    assert first["seed"] == 20_260_720
    assert first["relative_reduction_of_arm_means"] == pytest.approx(0.1)
    assert first["ci95_lower"] == pytest.approx(0.1)
    assert first["ci95_upper"] == pytest.approx(0.1)


def test_pinned_float32_matmul_precision_is_set_and_asserted() -> None:
    previous = torch.get_float32_matmul_precision()
    try:
        observed = configure_precision(CARD)
        assert observed == {
            "float32_matmul_precision": "high",
            "cuda_matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
        }
    finally:
        torch.set_float32_matmul_precision(previous)


def test_descriptive_bands_reduce_datasets_before_paired_model_seeds() -> None:
    rows = []
    horizons = np.arange(1, HORIZON + 1, dtype=np.float64)
    curve_names = (
        "instantaneous_field_mse",
        "cumulative_field_mse",
        "instantaneous_persistence_mse",
        "cumulative_persistence_mse",
        "instantaneous_model_over_persistence",
        "cumulative_model_over_persistence",
    )
    for arm_index, arm in enumerate(("dense", "sparse")):
        for model_seed in CARD["checkpoint_roster"]["model_seeds"]:
            for dataset_index, dataset_seed in enumerate(
                CARD["prospective_datasets"]["seeds"]
            ):
                value = (
                    1.0
                    + 0.20 * arm_index
                    + 0.01 * (int(model_seed) - 64)
                    + 0.001 * dataset_index
                )
                curve = (value * horizons).tolist()
                rows.append(
                    {
                        "arm": arm,
                        "model_seed": int(model_seed),
                        "dataset_seed": int(dataset_seed),
                        **{name: curve for name in curve_names},
                    }
                )
    first = descriptive_arm_curves(
        rows,
        CARD,
        bootstrap_replicates=200,
        bootstrap_chunk_size=37,
    )
    second = descriptive_arm_curves(
        rows,
        CARD,
        bootstrap_replicates=200,
        bootstrap_chunk_size=37,
    )
    assert first == second
    seed_curves = first["paired_model_seed_curves_after_three_dataset_average"]
    assert np.asarray(seed_curves["dense"]["cumulative_field_mse"]).shape == (
        10,
        HORIZON,
    )
    assert seed_curves["dense"]["cumulative_field_mse"][0][0] == pytest.approx(
        np.mean([1.0, 1.001, 1.002])
    )
    bands = first["pointwise_paired_seed_bootstrap"]
    assert bands["replicates"] == 200
    assert bands["seed"] == 20_260_721
    assert bands["resampling_unit"] == (
        "paired_model_seed_after_three_dataset_curve_average"
    )
    assert bands["coverage_policy"] == (
        "pointwise_descriptive_not_simultaneous_no_test_no_rescue"
    )
    assert first["aggregation_order"].startswith("average_three_datasets")


def test_exact_20_by_3_crossed_roster_and_duplicate_rejection() -> None:
    fields = synthetic_fields()[0]
    curves = curve_metrics(0.95 * fields[:, 1:], fields[:, 1:], fields[:, 0])
    rows = []
    for arm in CARD["checkpoint_roster"]["arms"]:
        for model_seed in CARD["checkpoint_roster"]["model_seeds"]:
            for dataset_index, dataset_seed in enumerate(CARD["prospective_datasets"]["seeds"]):
                rows.append(
                    {
                        "arm": arm,
                        "model_seed": model_seed,
                        "dataset_index": dataset_index,
                        "dataset_seed": dataset_seed,
                        **deepcopy(curves),
                    }
                )
    validate_crossed_rows(rows, CARD)
    with pytest.raises(AssertionError):
        validate_crossed_rows(rows + [deepcopy(rows[0])], CARD)


def test_all_realized_rng_streams_are_disjoint_from_excluded_streams() -> None:
    proof = realized_rng_streams(CARD)
    assert proof["new_stream_cardinality"] == 768
    assert proof["excluded_intersection_empty"] is True
    assert proof["modular_residue_proof_passed"] is True
    assert proof["new_stream_maximum"] < 2**31


def test_outcome_guard_receipt_is_required_and_scientific_payload_is_hash_only(
    tmp_path: Path,
) -> None:
    checkpoint_roster = [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "checkpoint_step": int(row["checkpoint_step"]),
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in CARD["checkpoint_roster"]["runs"]
    ]
    roster_encoded = json.dumps(
        checkpoint_roster, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    roster_hash = hashlib.sha256(roster_encoded).hexdigest()
    scientific = tmp_path / "scientific_curves.json"
    scientific.write_bytes(b"this is intentionally not deserialized JSON")
    dataset = tmp_path / "dataset_manifest.json"
    write_json_once(dataset, {"schema_version": 1})
    runtime = tmp_path / "runtime_lineage.json"
    runtime_payload = {
        "status": "scientific_payload_written_but_not_authorized_for_summary",
        "scientific_payload_sha256": sha256_path(scientific),
        "checkpoint_roster_sha256": roster_hash,
        "checkpoint_roster": checkpoint_roster,
        "environment": {"slurm_job_id": "not_recorded"},
        "scientific_metrics_printed": False,
    }
    write_json_once(runtime, runtime_payload)
    telemetry = tmp_path / "telemetry_audit.json"
    write_json_once(
        telemetry,
        {
            "status": "passed",
            "scientific_payload_opened": False,
            "evaluation_checks": {"mean": True, "p10": True},
            "gpu_uuid": "GPU-synthetic-test",
            "slurm_job_id": "not_recorded",
        },
    )
    receipt = tmp_path / "outcome_guard_receipt.json"
    write_json_once(
        receipt,
        {
            "status": "authorized_for_dependent_cpu_summary",
            "card_sha256": _CARD_HASH,
            "source_manifest_sha256": "b" * 64,
            "runtime_lineage_path": str(runtime),
            "runtime_lineage_sha256": sha256_path(runtime),
            "dataset_manifest_path": str(dataset),
            "dataset_manifest_sha256": sha256_path(dataset),
            "telemetry_audit_path": str(telemetry),
            "telemetry_audit_sha256": sha256_path(telemetry),
            "scientific_payload_path": str(scientific),
            "scientific_payload_sha256": sha256_path(scientific),
            "checkpoint_roster_sha256": roster_hash,
            "checkpoint_roster": checkpoint_roster,
            "gpu_uuid": "GPU-synthetic-test",
            "slurm_job_id": "not_recorded",
            "crossed_cells": 60,
            "scientific_payload_opened": False,
        },
    )
    with pytest.raises(RuntimeError, match="receipt hash mismatch"):
        verify_outcome_guard_receipt(
            receipt,
            expected_sha256="0" * 64,
            card_hash=_CARD_HASH,
            source_hash="b" * 64,
        )
    verified = verify_outcome_guard_receipt(
        receipt,
        expected_sha256=sha256_path(receipt),
        card_hash=_CARD_HASH,
        source_hash="b" * 64,
    )
    assert verified["scientific_payload_sha256"] == sha256_path(scientific)
