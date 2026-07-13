import numpy as np
import pytest
import torch
import torch.nn as nn

from skae.benchmarks.control_world_model import (
    ControlTrajectoryDataset,
    ControlWorldModel,
    ControlWorldModelConfig,
    LossWeights,
    WindowSampler,
    compute_normalization,
    evaluate_world_model,
    load_control_dataset,
    normalize_actions,
    normalize_observations,
    normalize_rewards,
    resolve_world_model_activation,
    save_control_dataset,
    train_step,
)


def make_tiny_dataset() -> ControlTrajectoryDataset:
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(4, 7, 5)).astype(np.float32)
    actions = rng.normal(size=(4, 6, 2)).astype(np.float32)
    rewards = rng.normal(size=(4, 6)).astype(np.float32)
    continuations = np.ones((4, 6), dtype=np.float32)
    continuations[:, -1] = 0.0
    valid = np.ones((4, 6), dtype=bool)
    split = np.asarray(["train", "train", "val", "test"])
    return ControlTrajectoryDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        continuations=continuations,
        valid=valid,
        split=split,
        episode_ids=np.arange(4),
        feature_names=tuple(f"obs/{idx}" for idx in range(5)),
        action_names=("a/0", "a/1"),
        metadata={"task": "tiny"},
    )


def test_dense_world_model_default_resolves_to_tanh():
    assert resolve_world_model_activation("additive", False, "auto") == "tanh"
    assert resolve_world_model_activation("mlp", False, "auto") == "tanh"
    assert resolve_world_model_activation("bilinear", True, "auto") == "relu"


def test_dense_world_model_rejects_relu_baseline():
    with pytest.raises(ValueError, match="Dense no-sparsity"):
        resolve_world_model_activation("additive", False, "relu")


def test_additive_step_matches_formula():
    cfg = ControlWorldModelConfig(
        obs_dim=4,
        action_dim=2,
        z_dim=3,
        hidden_dim=8,
        num_hidden_layers=1,
        transition_kind="additive",
        sparse_matrices=True,
        activation="relu",
    )
    model = ControlWorldModel(cfg)
    with torch.no_grad():
        model.k0.copy_(2.0 * torch.eye(3))
        model.action_linear.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 3.0], [-1.0, 2.0]]))
    z = torch.tensor([[1.0, 2.0, 3.0]])
    action = torch.tensor([[4.0, 5.0]])

    out = model.step_latent(z, action)

    expected = 2.0 * z + torch.tensor([[4.0, 15.0, 6.0]])
    assert torch.allclose(out, expected)


def test_bilinear_step_matches_formula():
    cfg = ControlWorldModelConfig(
        obs_dim=4,
        action_dim=2,
        z_dim=3,
        hidden_dim=8,
        num_hidden_layers=1,
        transition_kind="bilinear",
        sparse_matrices=True,
        activation="relu",
    )
    model = ControlWorldModel(cfg)
    with torch.no_grad():
        model.k0.zero_()
        model.action_linear.weight.zero_()
        model.k_action.zero_()
        model.k_action[0].copy_(torch.eye(3))
        model.k_action[1].copy_(2.0 * torch.eye(3))
    z = torch.tensor([[1.0, -2.0, 3.0]])
    action = torch.tensor([[4.0, 0.5]])

    out = model.step_latent(z, action)

    assert torch.allclose(out, 5.0 * z)
    densities = model.matrix_density(threshold=1e-6)
    assert densities["density/K0"] == 0.0
    assert densities["density/K_action_0"] == pytest.approx(1.0 / 3.0)


def test_dataset_roundtrip_and_sampler(tmp_path):
    dataset = make_tiny_dataset()
    path = tmp_path / "tiny_control.npz"
    save_control_dataset(path, dataset)

    loaded = load_control_dataset(path)
    sampler = WindowSampler(
        loaded.observations,
        loaded.actions,
        loaded.rewards,
        loaded.continuations,
        loaded.valid,
        loaded.indices_for_split("train"),
        sequence_length=3,
        rng=np.random.default_rng(1),
    )
    x, a, r, c = sampler.sample(2, device=torch.device("cpu"))

    assert loaded.obs_dim == 5
    assert x.shape == (2, 4, 5)
    assert a.shape == (2, 3, 2)
    assert r.shape == (2, 3)
    assert c.shape == (2, 3)


def test_train_step_and_evaluate_world_model_smoke():
    dataset = make_tiny_dataset()
    train_indices = dataset.indices_for_split("train")
    stats = compute_normalization(dataset, train_indices)
    observations = normalize_observations(dataset.observations, stats)
    actions = normalize_actions(dataset.actions, stats)
    rewards = normalize_rewards(dataset.rewards, stats)
    sampler = WindowSampler(
        observations,
        actions,
        rewards,
        dataset.continuations,
        dataset.valid,
        train_indices,
        sequence_length=3,
        rng=np.random.default_rng(2),
    )
    cfg = ControlWorldModelConfig(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        z_dim=6,
        hidden_dim=10,
        num_hidden_layers=1,
        transition_kind="mlp",
        sparse_matrices=False,
    )
    model = ControlWorldModel(cfg)
    assert isinstance(model.encoder.network[1], nn.Tanh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, a, r, c = sampler.sample(2, device=torch.device("cpu"))

    metrics = train_step(model, optimizer, x, a, r, c, LossWeights())
    eval_metrics = evaluate_world_model(
        model,
        sampler,
        batch_size=2,
        device=torch.device("cpu"),
        horizons=(1, 3),
        timing_repeats=1,
        planning_candidates=4,
    )

    assert metrics["loss"] > 0.0
    assert "open_loop_mse_h3" in eval_metrics
    assert eval_metrics["rollouts_per_second"] > 0.0
