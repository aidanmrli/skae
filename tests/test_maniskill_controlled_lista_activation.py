import pytest
import torch.nn as nn

from skae.benchmarks.maniskill_controlled_lista import (
    ControlledLISTAConfig,
    ControlledLISTAKoopman,
    controlled_lista_config_from_mapping,
    resolve_controlled_training_activation,
)


def test_dense_config_default_resolves_to_tanh():
    cfg = ControlledLISTAConfig(
        obs_dim=4,
        action_dim=2,
        z_dim=8,
        hidden_dim=6,
        num_hidden_layers=1,
        encoder_kind="dense",
    )

    model = ControlledLISTAKoopman(cfg)

    assert model.activation == "tanh"
    assert isinstance(model.pre_code.network[1], nn.Tanh)


def test_lista_config_default_resolves_to_relu():
    cfg = ControlledLISTAConfig(
        obs_dim=4,
        action_dim=2,
        z_dim=8,
        hidden_dim=6,
        num_hidden_layers=1,
        encoder_kind="lista",
    )

    model = ControlledLISTAKoopman(cfg)

    assert model.activation == "relu"
    assert isinstance(model.pre_code.network[1], nn.ReLU)


def test_training_dense_no_sparsity_baseline_defaults_to_tanh():
    assert resolve_controlled_training_activation("dense", "auto", 0.0) == "tanh"
    assert resolve_controlled_training_activation("dense", "auto", 0.01) == "relu"
    assert resolve_controlled_training_activation("lista", "auto", 0.0) == "relu"


def test_training_rejects_dense_no_sparsity_relu_baseline():
    with pytest.raises(ValueError, match="Dense no-sparsity baselines must use tanh"):
        resolve_controlled_training_activation("dense", "relu", 0.0)


def test_legacy_checkpoint_config_without_activation_stays_relu():
    cfg = controlled_lista_config_from_mapping(
        {
            "obs_dim": 4,
            "action_dim": 2,
            "z_dim": 8,
            "hidden_dim": 6,
            "num_hidden_layers": 1,
            "encoder_kind": "dense",
        }
    )

    model = ControlledLISTAKoopman(cfg)

    assert cfg.activation == "relu"
    assert model.activation == "relu"
    assert isinstance(model.pre_code.network[1], nn.ReLU)
