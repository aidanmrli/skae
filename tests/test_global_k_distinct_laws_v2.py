import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import experiments.neurips_2026.global_k_distinct_laws_v2 as v2_evaluator

from experiments.neurips_2026.assess_global_k_distinct_laws_v2_smoke import assess
from experiments.neurips_2026.assess_global_k_distinct_laws_v2_scientific_gpu import (
    assess_scientific_gpu,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_math import (
    autograd_jacobian,
    central_difference_jacobian,
    direct_latent_closure,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_checkpoint_audit import (
    TrainedRun,
    _schema_changes,
    assert_exact_checkpoint,
    load_trained_model,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_routing import (
    dense_center_projectors,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_source_lock import (
    REPOSITORY_ROOT,
    SOURCE_PATHS,
)
from experiments.neurips_2026.global_k_distinct_laws_v2_tasks import (
    APPROVED_CARD_SHA256,
    build_rows,
    load_card,
)
from skae.config import Config, get_config
from skae.data import make_env
from skae.model import make_model
from skae.training import runner


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "experiments/neurips_2026/global_k_distinct_laws_v2_card.json"


class LinearToy(nn.Module):
    """Toy where G is correct but reconstruction makes H wrong."""

    def __init__(self, target: np.ndarray):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "operator",
            torch.eye(2) + 0.5 * torch.as_tensor(target, dtype=torch.float32).T,
        )

    def encode(self, x):
        return 2.0 * x + 0.0 * self.anchor

    def decode(self, z):
        return z

    def kmatrix(self):
        return self.operator


class ClosureToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        operator = torch.eye(4)
        operator[0, 0] = 1.5
        operator[1, 2] = 8.0
        self.register_buffer("operator", operator)

    def encode(self, x):
        zeros = torch.zeros((*x.shape[:-1], 3), dtype=x.dtype, device=x.device)
        return torch.cat((x[..., :1], zeros), dim=-1) + 0.0 * self.anchor

    def kmatrix(self):
        return self.operator


class TiedDenseToy(nn.Module):
    def __init__(self, latent_dim=6):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.latent_dim = latent_dim

    def encode(self, x):
        return torch.ones((*x.shape[:-1], self.latent_dim), device=x.device) + 0.0 * self.anchor


class NonfiniteToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def encode(self, x):
        return x * torch.tensor(float("nan"), device=x.device) + 0.0 * self.anchor

    def decode(self, z):
        return z

    def kmatrix(self):
        return torch.eye(2, device=self.anchor.device)


def test_h_block_exposes_reconstruction_derivative_loophole_that_g_misses():
    target = np.asarray([[-0.10, -0.03], [0.04, -0.08]], dtype=np.float64)
    model = LinearToy(target).eval()
    center = np.asarray([1.0, -0.5], dtype=np.float32)
    mask = np.ones(2, dtype=bool)
    g = autograd_jacobian(model, center, mask, "g_block")
    h = autograd_jacobian(model, center, mask, "h_block")
    np.testing.assert_allclose(g, target, atol=1e-7)
    np.testing.assert_allclose(h, target + np.eye(2), atol=1e-7)
    assert np.linalg.norm(h - target) > 1.0


def test_both_h_and_g_have_independent_kink_finite_difference_checks():
    target = np.asarray([[-0.10, 0.02], [-0.03, -0.07]], dtype=np.float64)
    model = LinearToy(target).eval()
    center = np.asarray([0.2, -0.3], dtype=np.float32)
    mask = np.ones(2, dtype=bool)
    for estimand in ("h_block", "g_block"):
        automatic = autograd_jacobian(model, center, mask, estimand)
        finite = central_difference_jacobian(
            model, center, mask, estimand, epsilon=0.0015
        )
        np.testing.assert_allclose(automatic, finite, atol=5e-5)


def test_task_tables_are_exact_mixed_twenty_with_frozen_recipes():
    card, digest = load_card(CARD_PATH)
    assert digest == APPROVED_CARD_SHA256
    smoke = build_rows(card, "smoke")
    full = build_rows(card, "full")
    assert len(smoke) == len(full) == 20
    assert card["task_table_contract"]["smoke_num_steps"] == 5000
    assert [(row["arm"], row["seed"]) for row in full] == (
        [("sparse", seed) for seed in range(100, 110)]
        + [("dense", seed) for seed in range(100, 110)]
    )
    assert [row["num_steps"] for row in smoke] == [5000] * 20
    assert [row["num_steps"] for row in full] == [200000] * 20
    sparse, dense = full[0], full[10]
    assert sparse["lista_final_op"] == "sign_split"
    assert sparse["sparsity_coeff"] == 0.003
    assert sparse["weight_decay"] == 0.0001
    assert dense["config_name"] == "generic_no_shrink"
    assert dense["sparsity_coeff"] == dense["weight_decay"] == 0.0
    assert dense["encoder_group_shrinkage"] == "false"


def test_exact_sparse_task_cli_materializes_normalized_historical_recipe(
    monkeypatch, tmp_path,
):
    card, _digest = load_card(CARD_PATH)
    row = build_rows(card, "full")[0]
    captured = {}
    monkeypatch.setattr(runner, "train", lambda cfg, **kwargs: captured.setdefault("cfg", cfg))
    arguments = [
        "skae-train", "--config", str(row["config_name"]), "--env", str(row["env_name"]),
        "--env_dt", str(row["env_dt"]), "--num_steps", str(row["num_steps"]),
        "--batch_size", str(row["batch_size"]), "--target_size", str(row["target_size"]),
        "--res_coeff", str(row["res_coeff"]), "--reconst_coeff", str(row["reconst_coeff"]),
        "--pred_coeff", str(row["pred_coeff"]), "--sparsity_coeff", str(row["sparsity_coeff"]),
        "--sequence_length", str(row["sequence_length"]), "--eval_profile", str(row["eval_profile"]),
        "--seed", str(row["seed"]), "--device", "cpu", "--log_dir", str(tmp_path),
        "--eval_every", str(row["eval_every"]), "--eval_num_steps", str(row["eval_num_steps"]),
        "--lista_alpha", str(row["lista_alpha"]), "--lista_num_loops", str(row["lista_num_loops"]),
        "--lista_linear_encoder", str(row["lista_linear_encoder"]),
        "--lista_final_op", str(row["lista_final_op"]),
        "--hard_init_oversample", str(row["hard_init_oversample"]),
        "--hard_init_fraction", str(row["hard_init_fraction"]),
        "--hard_init_pool_size", str(row["hard_init_pool_size"]),
        "--hard_init_num_candidates", str(row["hard_init_num_candidates"]),
        "--hard_init_probe_steps", str(row["hard_init_probe_steps"]),
        "--hard_init_num_perturbations", str(row["hard_init_num_perturbations"]),
        "--hard_init_perturb_scale", str(row["hard_init_perturb_scale"]),
        "--hard_init_transient_window", str(row["hard_init_transient_window"]),
        "--hard_init_transient_weight", str(row["hard_init_transient_weight"]),
        "--hard_init_jitter_scale", str(row["hard_init_jitter_scale"]),
        "--encoder_group_shrinkage", str(row["encoder_group_shrinkage"]),
        "--encoder_topk_groups", str(row["encoder_topk_groups"]),
        "--decoder_coherence_weight", str(row["decoder_coherence_weight"]),
        "--normalize_decoder_atoms", str(row["normalize_decoder_atoms"]),
        "--k_structure", str(row["k_structure"]), "--lr", str(row["lr"]),
        "--k_matrix_lr", str(row["k_matrix_lr"]), "--weight_decay", str(row["weight_decay"]),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    runner.main()
    reference = json.loads(Path(card["training_arms"]["sparse"]["representative_frozen_config"]).read_text())
    expected = Config.from_dict(reference).to_dict()
    expected["SEED"] = int(row["seed"])
    assert captured["cfg"].to_dict() == expected
    changes = _schema_changes(reference, Config.from_dict(reference).to_dict())
    frozen_allowlist = card["training_arms"]["sparse"][
        "representative_schema_compatibility_normalization"
    ]["exact_allowlist"]
    later_unrelated_dysts_defaults = {
        "insert ENV.DYSTS.CACHE_FALLBACK_TIMEOUT_SECONDS=0.0",
        "insert ENV.DYSTS.CACHE_PRIMARY_METHOD=Radau",
        "insert ENV.DYSTS.CACHE_TIMEOUT_FALLBACK_METHOD=",
        "insert ENV.DYSTS.CACHE_TRAJECTORY_TIMEOUT_SECONDS=0.0",
    }
    assert set(changes) == set(frozen_allowlist) | later_unrelated_dysts_defaults
    sparse_cfg = captured["cfg"]
    sparse_model = make_model(sparse_cfg, make_env(sparse_cfg).observation_size)
    sparse_spec = TrainedRun(0, "sparse", 100, "gated_local_linear", tmp_path, 1, 0)
    # The frozen audit is intentionally exact. Later Dysts-only schema
    # defaults therefore make the historical packet non-rerunnable without
    # checking out its locked source state.
    with pytest.raises(
        AssertionError,
        match="historical_schema_normalization_exact_allowlist",
    ):
        assert_exact_checkpoint(
            sparse_cfg,
            sparse_model,
            {"config": sparse_cfg.to_dict(), "step": 500, "optimizer_state_dict": None},
            card,
            sparse_spec,
        )


def test_dense_fake_checkpoint_has_exact_tanh_zero_sparsity_capacity(tmp_path):
    card, _digest = load_card(CARD_PATH)
    recipe = card["training_arms"]["dense"]
    hard = card["training_arms"]["matched_hard_initial_condition_oversampling"]
    cfg = get_config(recipe["config_name"])
    cfg.SEED = 100
    cfg.ENV.ENV_NAME = "gated_local_linear"
    cfg.ENV.GATED_LOCAL_LINEAR.DT = card["benchmark"]["dt"]
    cfg.TRAIN.NUM_STEPS = recipe["num_steps"]
    cfg.TRAIN.BATCH_SIZE = recipe["batch_size"]
    cfg.TRAIN.DATA_SIZE = recipe["data_size"]
    cfg.TRAIN.SEQUENCE_LENGTH = recipe["sequence_length"]
    cfg.TRAIN.EVAL_EVERY = recipe["eval_every"]
    cfg.TRAIN.EVAL_NUM_STEPS = recipe["eval_num_steps"]
    cfg.TRAIN.LR = recipe["learning_rate"]
    cfg.TRAIN.K_MATRIX_LR = recipe["koopman_learning_rate"]
    cfg.TRAIN.WEIGHT_DECAY = recipe["weight_decay"]
    cfg.MODEL.TARGET_SIZE = recipe["latent_dim"]
    cfg.MODEL.RES_COEFF = recipe["loss_weights"]["residual"]
    cfg.MODEL.RECONST_COEFF = recipe["loss_weights"]["reconstruction"]
    cfg.MODEL.PRED_COEFF = recipe["loss_weights"]["prediction"]
    cfg.MODEL.SPARSITY_COEFF = recipe["loss_weights"]["sparsity"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.ENABLED = hard["enabled"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.FRACTION = hard["fraction"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.POOL_SIZE = hard["pool_size"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.NUM_CANDIDATES = hard["num_candidates"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.PROBE_STEPS = hard["probe_steps"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.NUM_PERTURBATIONS = hard["num_perturbations"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.PERTURB_SCALE = hard["perturb_scale"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.TRANSIENT_WINDOW = hard["transient_window"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.TRANSIENT_WEIGHT = hard["transient_weight"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.JITTER_SCALE = hard["jitter_scale"]
    cfg.TRAIN.HARD_INIT_OVERSAMPLE.BUILD_CHUNK_SIZE = hard["build_chunk_size"]
    model = make_model(cfg, make_env(cfg).observation_size)
    spec = TrainedRun(10, "dense", 100, "gated_local_linear", tmp_path, 1, 0)
    audit = assert_exact_checkpoint(
        cfg, model, {"config": cfg.to_dict(), "step": 500, "optimizer_state_dict": None},
        card, spec,
    )
    assert audit["checks"]["tanh_hidden_activation"] is True
    assert audit["checks"]["zero_sparsity"] is True


def test_checkpoint_loader_does_not_overwrite_wrong_serialized_environment(tmp_path):
    cfg = get_config("generic_no_shrink")
    cfg.SEED = 100
    cfg.ENV.ENV_NAME = "duffing"
    model = make_model(cfg, make_env(cfg).observation_size)
    torch.save(
        {"config": cfg.to_dict(), "model_state_dict": model.state_dict(), "step": 500},
        tmp_path / "checkpoint.pt",
    )
    spec = TrainedRun(10, "dense", 100, "gated_local_linear", tmp_path, 1, 0)
    loaded_cfg, _env, loaded_model, checkpoint, _path = load_trained_model(spec)
    assert loaded_cfg.ENV.ENV_NAME == "duffing"
    card, _digest = load_card(CARD_PATH)
    with pytest.raises(AssertionError, match="system"):
        assert_exact_checkpoint(loaded_cfg, loaded_model, checkpoint, card, spec)


def test_smoke_assessor_requires_all_frozen_utilization_gates():
    card = json.loads(CARD_PATH.read_text())
    statuses = [
        {"task_id": index, "start": 0.0, "end": 400.0, "exit_code": 0}
        for index in range(20)
    ]
    starts = {index: 10.0 for index in range(20)}
    ends = {index: (390.0, 0) for index in range(20)}
    timing = {"pack_start": 0.0, "pack_end": 400.0}
    telemetry = [
        {
            "epoch": 10.0 + 2.0 * sample,
            "uuid": "GPU-one",
            "name": "NVIDIA A100-SXM4-80GB",
            "utilization": 95.0,
            "memory_used": 40000.0,
            "memory_total": 81920.0,
        }
        for sample in range(190)
    ]
    payload = assess(card, statuses, starts, ends, timing, telemetry)
    assert payload["passed"] is True
    assert payload["outcomes_inspected"] is False
    assert payload["active_sample_count"] >= 100
    failed = [dict(item) for item in telemetry]
    for item in failed:
        item["utilization"] = 70.0
    assert assess(card, statuses, starts, ends, timing, failed)["passed"] is False


def test_smoke_assessor_retains_low_utilization_tail_through_latest_end():
    card = json.loads(CARD_PATH.read_text())
    statuses = [
        {"task_id": index, "start": 0.0, "end": 400.0, "exit_code": 0}
        for index in range(20)
    ]
    starts = {index: 10.0 for index in range(20)}
    ends = {index: (210.0 if index < 19 else 390.0, 0) for index in range(20)}
    timing = {"pack_start": 0.0, "pack_end": 400.0}
    telemetry = [
        {
            "epoch": 10.0 + 2.0 * sample, "uuid": "GPU-one",
            "name": "NVIDIA A100-SXM4-80GB",
            "utilization": 95.0 if 10.0 + 2.0 * sample <= 210.0 else 20.0,
            "memory_used": 40000.0, "memory_total": 81920.0,
        }
        for sample in range(191)
    ]
    payload = assess(card, statuses, starts, ends, timing, telemetry)
    assert payload["active_sample_count"] == 191
    assert payload["passed"] is False


def test_telemetry_assessors_fail_closed_on_missing_tail_or_internal_gap():
    card = json.loads(CARD_PATH.read_text())
    statuses = [
        {"task_id": index, "start": 0.0, "end": 900.0, "exit_code": 0}
        for index in range(20)
    ]
    starts = {index: 0.0 for index in range(20)}
    ends = {index: (900.0, 0) for index in range(20)}
    timing = {"pack_start": 0.0, "pack_end": 900.0}
    scientific = [
        {
            "epoch": float(epoch), "uuid": "GPU-one",
            "name": "NVIDIA A100-SXM4-80GB", "utilization": 90.0,
            "memory_used": 40000.0, "memory_total": 81920.0,
        }
        for epoch in range(0, 901, 15)
    ]
    assert assess_scientific_gpu(
        card, statuses, starts, ends, timing, scientific
    )["assessment_complete"] is True
    with pytest.raises(RuntimeError, match="densely cover"):
        assess_scientific_gpu(
            card, statuses, starts, ends, timing,
            [row for row in scientific if not 300 <= row["epoch"] <= 390],
        )
    smoke_status = [dict(row, end=390.0) for row in statuses]
    smoke_ends = {index: (390.0, 0) for index in range(20)}
    smoke_timing = {"pack_start": 0.0, "pack_end": 400.0}
    truncated = [
        dict(row, epoch=row["epoch"] * 2.0 / 15.0)
        for row in scientific if row["epoch"] <= 210.0
    ]
    assert assess(
        card, smoke_status, starts, smoke_ends, smoke_timing, truncated
    )["passed"] is False


def test_direct_closure_uses_point_vector_rms_and_not_static_subspace_gate():
    model = ClosureToy().eval()
    points = np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    result = direct_latent_closure(
        model, points, np.asarray([True, True, False, False])
    )
    expected_change_rms = np.sqrt((0.5**2 + 1.5**2) / 2.0)
    np.testing.assert_allclose(result["change_denominator_rms"], expected_change_rms)
    assert result["change_normalized_leakage"] == 0.0
    assert result["matrix_raw_leakage"] > 0.9


def test_dense_center_projector_has_exact_sparse_k_and_index_tie_break():
    centers = np.zeros((3, 2), dtype=np.float32)
    sparse = np.zeros((3, 6), dtype=bool)
    sparse[0, 5] = True
    sparse[1, 4:] = True
    sparse[2, 3:] = True
    masks, diagnostics = dense_center_projectors(
        TiedDenseToy(), centers, sparse, batch_size=8
    )
    assert diagnostics["center_projectors_valid"] is True
    assert masks.sum(axis=1).tolist() == [1, 2, 3]
    assert np.flatnonzero(masks[0]).tolist() == [0]
    assert np.flatnonzero(masks[1]).tolist() == [0, 1]
    assert np.flatnonzero(masks[2]).tolist() == [0, 1, 2]


def test_nonfinite_jacobian_fails_before_nan_can_enter_a_shard(monkeypatch):
    monkeypatch.setattr(
        v2_evaluator, "_geometry",
        lambda *_args: (np.zeros((3, 2, 2)), [1.0, 1.0, 1.0]),
    )
    with pytest.raises(FloatingPointError, match="nonfinite autograd"):
        v2_evaluator.score_mechanism(
            NonfiniteToy(), np.ones((3, 2), dtype=bool),
            np.zeros((3, 2), dtype=np.float32), np.zeros((4, 2), dtype=np.float32),
            [np.zeros((2, 2), dtype=np.float32) for _ in range(3)],
            None, json.loads(CARD_PATH.read_text()), "sparse", {"passed": True},
        )
    failure = v2_evaluator._numerical_failure(
        FloatingPointError("nonfinite downstream mechanism metric")
    )
    assert failure["status"] == "ineligible_numerical"
    json.dumps(failure, allow_nan=False)


def test_source_lock_roster_exists_and_covers_every_execution_boundary():
    assert REPOSITORY_ROOT == ROOT
    assert all((REPOSITORY_ROOT / relative).is_file() for relative in SOURCE_PATHS)
    roster = set(SOURCE_PATHS)
    for required in (
        "experiments/neurips_2026/global_k_distinct_laws_v2_math.py",
        "experiments/neurips_2026/global_k_distinct_laws_v2_routing.py",
        "experiments/neurips_2026/assess_global_k_distinct_laws_v2_smoke.py",
        "experiments/neurips_2026/assess_global_k_distinct_laws_v2_scientific_gpu.py",
        "experiments/neurips_2026/build_global_k_distinct_laws_v2_packet.py",
        "pyproject.toml",
        "uv.lock",
        "skae/checkpoint_compat.py",
        "skae/evaluation.py",
        "scripts/common/cluster_env.sh",
        "scripts/common/run_benchmark_task.sh",
        "scripts/neurips_2026/global_k_distinct_laws_v2/run_mixed_pack.sh",
        "scripts/neurips_2026/global_k_distinct_laws_v2/queue_scientific_chain.sh",
    ):
        assert required in roster


def test_slurm_boundaries_pack_twenty_on_one_a100_and_evaluate_cpu_only():
    pack = (
        ROOT / "scripts/neurips_2026/global_k_distinct_laws_v2/run_mixed_pack.sh"
    ).read_text()
    evaluation = (
        ROOT / "scripts/neurips_2026/global_k_distinct_laws_v2/run_evaluation.sh"
    ).read_text()
    queue = (
        ROOT
        / "scripts/neurips_2026/global_k_distinct_laws_v2/queue_scientific_chain.sh"
    ).read_text()
    assert "#SBATCH --partition=long" in pack
    assert "#SBATCH --gres=gpu:a100l:1" in pack
    assert "#SBATCH --cpus-per-task=24" in pack
    assert "#SBATCH --mem=64G" in pack
    assert 'PACK_SIZE:-20' in pack and 'PACK_CONCURRENCY:-20' in pack
    assert "OMP_NUM_THREADS=1" in pack
    assert "GPU_TELEMETRY=0" in pack
    assert "quarantined_task_logs" in pack
    assert "--gres" not in evaluation
    assert 'export CUDA_VISIBLE_DEVICES=""' in evaluation
    assert (
        queue.index("\nTELEMETRY_JOB_ID=$(" )
        < queue.index("\nAUDIT_JOB_ID=$(" )
        < queue.index("\nEVAL_JOB_ID=$(" )
        < queue.index("\nSUMMARY_JOB_ID=$(" )
        < queue.index("\nPACKET_JOB_ID=$(" )
    )
    assert "afterok:${TELEMETRY_JOB_ID}" in queue
