"""Tests for the spatialized multibasin reaction-diffusion benchmark helpers."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from skae.benchmarks.spatialized_conv_koopman import SpatialConvKoopman, SpatialConvKoopmanConfig
from skae.benchmarks.spatialized_reaction_diffusion import (
    SpatialReactionDiffusionConfig,
    flatten_fields,
    generate_dataset,
    get_source_system,
    reshape_flat_fields,
    spatial_gradient,
)
from tools.build_spatialized_reaction_diffusion_tasks import _build_rows
from tools.evaluate_spatialized_reaction_diffusion import (
    paper_support_family_labels,
    summarize_paper_support_collection,
)


def test_flatten_reshape_fields_roundtrip():
    fields = torch.randn(3, 5, 8, 8, 2)

    flat = flatten_fields(fields)
    restored = reshape_flat_fields(flat, grid_size=8)

    assert flat.shape == (3, 5, 128)
    assert torch.allclose(restored, fields)


def test_spatial_gradient_shape():
    fields = torch.randn(3, 5, 8, 8, 2)

    grad_x, grad_y = spatial_gradient(fields)

    assert grad_x.shape == fields.shape
    assert grad_y.shape == fields.shape


def test_spatial_config_defaults_include_label_horizon_and_laplacian_mode():
    cfg = SpatialReactionDiffusionConfig()

    assert cfg.label_extra_observations > 0
    assert cfg.laplacian_scaling in {"continuum", "graph"}
    assert cfg.spatial_extent > 0


def test_multiwell_allen_cahn_source_exposes_well_centers():
    system = get_source_system("allen_cahn_4")
    state = torch.zeros(2, 5, 6)

    rhs = system.dynamics(state)

    assert system.centers.shape == (4, 2)
    assert rhs.shape == state.shape
    assert torch.isfinite(rhs).all()


def test_multiwell_allen_cahn_tiny_dataset_has_evaluation_labels():
    cfg = SpatialReactionDiffusionConfig(
        source_system="allen_cahn_3",
        grid_size=6,
        diffusion=0.001,
        rk4_dt=0.005,
        substeps_per_observation=1,
        trajectory_length=2,
        label_extra_observations=1,
        train_trajectories=2,
        val_trajectories=1,
        test_trajectories=1,
        laplacian_scaling="graph",
        seed=11,
    )

    bundle = generate_dataset(cfg)

    assert bundle["fields"].shape == (4, 3, 6, 6, 2)
    assert bundle["basin_maps"].shape == (4, 6, 6)
    assert bundle["attractor_centers"].shape == (3, 2)
    assert bundle["metadata"]["labels_are_evaluation_only"] is True
    assert "No basin labels" in bundle["metadata"]["training_label_policy"]


def test_conv_koopman_shape_contracts():
    cfg = SpatialConvKoopmanConfig(
        grid_size=8,
        channels=2,
        z_dim=16,
        hidden_channels=8,
        num_blocks=2,
        encoder_kind="lista",
    )
    model = SpatialConvKoopman(cfg)
    x = torch.randn(4, 8 * 8 * 2)

    z = model.encode(x)
    recon = model.decode(z)
    z_roll, pred = model.rollout_observation_discrete(x, horizon=3)

    assert z.shape == (4, 16)
    assert recon.shape == x.shape
    assert z_roll.shape == (4, 3, 16)
    assert pred.shape == (4, 3, x.shape[-1])


def test_conv_dense_uses_tanh_hidden_activations():
    dense_model = SpatialConvKoopman(
        SpatialConvKoopmanConfig(
            grid_size=8,
            channels=2,
            z_dim=16,
            hidden_channels=8,
            num_blocks=2,
            encoder_kind="dense",
        )
    )
    lista_model = SpatialConvKoopman(
        SpatialConvKoopmanConfig(
            grid_size=8,
            channels=2,
            z_dim=16,
            hidden_channels=8,
            num_blocks=2,
            encoder_kind="lista",
        )
    )

    assert any(isinstance(module, torch.nn.Tanh) for module in dense_model.encoder_conv.modules())
    assert any(isinstance(module, torch.nn.Tanh) for module in dense_model.decoder_conv.modules())
    assert not any(isinstance(module, torch.nn.ReLU) for module in dense_model.encoder_conv.modules())
    assert any(isinstance(module, torch.nn.GELU) for module in lista_model.encoder_conv.modules())


def test_sparse_conv_activation_can_match_dense_tanh():
    model = SpatialConvKoopman(
        SpatialConvKoopmanConfig(
            grid_size=8,
            channels=2,
            z_dim=16,
            hidden_channels=8,
            num_blocks=2,
            encoder_kind="lista",
            conv_activation="tanh",
        )
    )

    assert any(isinstance(module, torch.nn.Tanh) for module in model.encoder_conv.modules())
    assert any(isinstance(module, torch.nn.Tanh) for module in model.decoder_conv.modules())
    assert not any(isinstance(module, torch.nn.GELU) for module in model.encoder_conv.modules())


def test_spatialized_task_builder_matrix():
    args = Namespace(
        output_tsv="unused.tsv",
        output_manifest_json=None,
        base_out="/tmp/spatial-rd",
        systems_csv="cal_square_4,transition_routes_4",
        model_variants_csv="conv_lista,conv_dense",
        seeds_csv="0,1",
        grid_size=16,
        diffusion=0.01,
        rk4_dt=0.005,
        substeps_per_observation=5,
        trajectory_length=12,
        label_extra_observations=12,
        train_trajectories=16,
        val_trajectories=4,
        test_trajectories=4,
        laplacian_scaling="continuum",
        target_size=0,
        min_latent_state_ratio=4.0,
        hidden_channels=16,
        num_blocks=2,
        conv_activation="",
        num_steps=100,
        batch_size=8,
        sequence_length=4,
        train_observation_limit=0,
        lista_num_loops=2,
        lista_alpha=1e-3,
        sparsity_coeff=0.0,
        support_threshold=1e-4,
        family_jaccard=0.7,
        max_validation_reps=128,
        deep_threshold=0.7,
        eval_horizons="1,4",
        eval_horizon=4,
    )

    rows = _build_rows(args)

    assert len(rows) == 8
    assert {row["trainer"] for row in rows} == {"conv"}
    assert {row["model_variant"] for row in rows} == {"conv_lista", "conv_dense"}
    assert {row["state_dim"] for row in rows} == {512}
    assert {row["target_size"] for row in rows} == {2048}
    assert all(str(row["dataset_path"]).endswith("dataset.pt") for row in rows)


def test_spatialized_task_builder_rejects_undercomplete_latent():
    args = Namespace(
        output_tsv="unused.tsv",
        output_manifest_json=None,
        base_out="/tmp/spatial-rd",
        systems_csv="cal_square_4",
        model_variants_csv="conv_lista",
        seeds_csv="0",
        grid_size=16,
        diffusion=0.01,
        rk4_dt=0.005,
        substeps_per_observation=5,
        trajectory_length=12,
        label_extra_observations=12,
        train_trajectories=16,
        val_trajectories=4,
        test_trajectories=4,
        laplacian_scaling="continuum",
        target_size=128,
        min_latent_state_ratio=4.0,
        hidden_channels=16,
        num_blocks=2,
        num_steps=100,
        batch_size=8,
        sequence_length=4,
        train_observation_limit=0,
        lista_num_loops=2,
        lista_alpha=1e-3,
        lista_alpha_csv=None,
        sparsity_coeff=0.0,
        sparsity_coeff_csv=None,
        support_threshold=1e-4,
        support_threshold_csv=None,
        family_jaccard=0.7,
        family_jaccard_csv=None,
        max_validation_reps=128,
        deep_threshold=0.7,
        eval_horizons="1,4",
        eval_horizon=4,
    )

    with pytest.raises(ValueError, match="overcomplete"):
        _build_rows(args)


def test_paper_support_family_labels_use_frequency_ordered_jaccard_merge():
    masks = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [False, False, True, True],
        ],
        dtype=torch.bool,
    ).numpy()

    labels, prototypes, counts = paper_support_family_labels(masks, min_jaccard=0.5)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]
    assert len(prototypes) == 2
    assert counts.tolist() == [3, 1]


def test_paper_support_collection_reports_s_abs_and_f_abs_entropies():
    labels = torch.tensor([0, 0, 1, 1]).numpy()
    masks = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
            [False, False, True, True],
            [False, False, True, True],
        ],
        dtype=torch.bool,
    ).numpy()

    metrics = summarize_paper_support_collection(labels, masks)

    assert metrics["s_abs"]["exact_support_count"] == 3
    assert metrics["f_abs"]["family_count"] == 2
    assert metrics["f_abs"]["h_basin_given_f_abs"] == 0.0
    assert metrics["f_abs"]["h_f_abs_given_basin"] == 0.0
