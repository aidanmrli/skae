"""Unit tests for the evaluation utilities."""

import numpy as np
import pytest
import torch

from skae.config import get_config
from skae.data import make_env
from skae.evaluation import (
    EvaluationSettings,
    _rollout_event_trigger_reencode_with_diagnostics,
    evaluate_model,
    rollout_every_step_reencode,
    rollout_event_trigger_reencode,
    rollout_no_reencode,
    rollout_periodic_reencode,
    _estimate_learned_attractors,
    _save_lyapunov_phase_portrait_comparison,
)
from skae.model import make_model


def _build_model_and_states(system: str, batch_size: int = 4) -> tuple:
    cfg = get_config("generic")
    cfg.ENV.ENV_NAME = system
    env = make_env(cfg)
    model = make_model(cfg, env.observation_size)
    model.eval()

    rng = torch.Generator().manual_seed(0)
    states = torch.randn(batch_size, env.observation_size, generator=rng)
    return model, states, cfg


def test_rollout_modes_shape():
    model, states, _ = _build_model_and_states("pendulum")

    horizon = 8
    no_re = rollout_no_reencode(model, states, horizon)
    every = rollout_every_step_reencode(model, states, horizon)
    periodic = rollout_periodic_reencode(model, states, horizon, period=2)

    assert no_re.shape == (horizon, states.shape[0], states.shape[1])
    assert every.shape == (horizon, states.shape[0], states.shape[1])
    assert periodic.shape == (horizon, states.shape[0], states.shape[1])


def test_rollout_event_trigger_reencode_can_reset_every_step():
    class AlwaysResetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("bias", torch.nn.Parameter(torch.zeros(1)))

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

        def encode_with_prior(
            self,
            x: torch.Tensor,
            latent_prior: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del x, latent_prior
            return torch.zeros(1, 1)

        def step_latent(self, latent: torch.Tensor) -> torch.Tensor:
            return latent + 1.0

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return latent

    model = AlwaysResetModel()
    x0 = torch.zeros(1, 1)
    pred = rollout_event_trigger_reencode(
        model,
        x0,
        horizon=4,
        proj_threshold=0.1,
        min_dwell=0,
        max_interval=0,
    )

    assert pred.shape == (4, 1, 1)
    assert torch.allclose(pred.squeeze(-1).squeeze(-1), torch.ones(4))


def test_hybrid_event_trigger_records_group_trigger_masks():
    class HybridTriggerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("bias", torch.nn.Parameter(torch.zeros(1)))
            self._k_structure = "block_diagonal"
            self._k_block_sizes = [1, 1]

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def encode_with_prior(
            self,
            x: torch.Tensor,
            latent_prior: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del latent_prior
            return x

        def step_latent(self, latent: torch.Tensor) -> torch.Tensor:
            dominant = latent[..., :1]
            return torch.cat([0.2 * dominant, 0.9 * dominant], dim=-1)

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return latent

    model = HybridTriggerModel()
    x0 = torch.tensor([[1.0, 0.0]])
    predictions, diagnostics = _rollout_event_trigger_reencode_with_diagnostics(
        model,
        x0,
        horizon=1,
        proj_threshold=None,
        ambiguity_threshold=0.15,
        spillover_threshold=0.70,
        min_dwell=0,
        max_interval=0,
    )

    assert predictions.shape == (1, 1, 2)
    assert torch.allclose(diagnostics["projection_gap"][0], torch.tensor([0.0]))
    assert diagnostics["ambiguity_score"][0, 0].item() > 0.15
    assert diagnostics["spillover_score"][0, 0].item() > 0.70
    assert bool(diagnostics["ambiguity_trigger_mask"][0, 0].item()) is True
    assert bool(diagnostics["spillover_trigger_mask"][0, 0].item()) is True
    assert bool(diagnostics["threshold_trigger_mask"][0, 0].item()) is True
    assert bool(diagnostics["reset_mask"][0, 0].item()) is True


def test_support_margin_event_trigger_uses_min_ratio_threshold():
    class MarginTriggerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("bias", torch.nn.Parameter(torch.zeros(1)))

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def encode_with_prior(
            self,
            x: torch.Tensor,
            latent_prior: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del latent_prior
            return x

        def step_latent(self, latent: torch.Tensor) -> torch.Tensor:
            return latent * 0.6

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return latent

    model = MarginTriggerModel()
    x0 = torch.tensor([[0.2, 0.0]])
    _, diagnostics = _rollout_event_trigger_reencode_with_diagnostics(
        model,
        x0,
        horizon=1,
        proj_threshold=None,
        support_margin_min_ratio=1.5,
        support_threshold=0.1,
        min_dwell=0,
        max_interval=0,
    )

    assert diagnostics["support_margin_ratio"][0, 0].item() == pytest.approx(1.2, rel=1e-6)
    assert bool(diagnostics["support_margin_trigger_mask"][0, 0].item()) is True
    assert bool(diagnostics["threshold_trigger_mask"][0, 0].item()) is True


def test_evaluate_model_generates_outputs(tmp_path):
    model, _, cfg = _build_model_and_states("duffing")

    settings = EvaluationSettings(
        systems=("duffing",),
        horizons=(10,),
        periodic_reencode_periods=(2,),
        event_trigger_proj_threshold=0.05,
        event_trigger_min_dwell=2,
        event_trigger_max_interval=5,
        use_dynamics_prior=True,
        batch_size=4,
        phase_portrait_samples=2,
        save_rollout_artifacts=True,
    )

    results = evaluate_model(
        model=model,
        cfg=cfg,
        device="cpu",
        settings=settings,
        output_dir=tmp_path,
    )

    assert "duffing" in results
    duffing_metrics = results["duffing"]
    assert "modes" in duffing_metrics
    assert "no_reencode" in duffing_metrics["modes"]
    assert "best_periodic" in duffing_metrics
    assert "best_reset" in duffing_metrics
    assert "10" in duffing_metrics["best_periodic"]
    horizon_metrics = duffing_metrics["modes"]["no_reencode"]["horizons"]["10"]
    assert "per_dim_mean" in horizon_metrics
    assert "rmse_per_dim_mean" in horizon_metrics
    assert "per_dim_mean" in duffing_metrics["best_periodic"]["10"]
    assert "per_dim_mean" in duffing_metrics["best_reset"]["10"]
    assert any(mode.startswith("event_proj_") for mode in duffing_metrics["modes"])

    metrics_json = tmp_path / "metrics.json"
    assert metrics_json.exists(), "Evaluation should write metrics.json"

    curve_png = tmp_path / "duffing" / "mse_vs_horizon.png"
    assert curve_png.exists(), "Evaluation should write MSE curve plot"

    artifact_path = tmp_path / "duffing" / "rollout_artifacts.pt"
    assert artifact_path.exists(), "Evaluation should write rollout_artifacts.pt when enabled"
    payload = torch.load(artifact_path, map_location="cpu")
    assert payload["system"] == "duffing"
    assert payload["true_sequences"].shape[0] == settings.batch_size
    assert "no_reencode" in payload["predictions"]
    assert payload["evaluation_settings"]["use_dynamics_prior"] is True
    assert payload["evaluation_settings"]["event_trigger_support_threshold"] == pytest.approx(1e-3)
    assert any(mode.startswith("event_proj_") for mode in payload["predictions"])
    assert "mode_diagnostics" in payload


def test_estimate_learned_attractors_contracts_to_origin():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("bias", torch.nn.Parameter(torch.zeros(1)))

        def step_env(self, state: torch.Tensor) -> torch.Tensor:
            return state * 0.5

    model = DummyModel()
    attractors = _estimate_learned_attractors(
        model=model,
        grid_lim=1.0,
        num_samples=16,
        num_steps=6,
        tolerance=1e-3,
        device=torch.device("cpu"),
    )

    assert attractors.shape[0] == 1
    assert np.allclose(attractors[0], 0.0, atol=1e-2)


def test_lyapunov_phase_portrait_outputs(tmp_path):
    model, _, cfg = _build_model_and_states("lyapunov")
    env = make_env(cfg)

    comp_path = tmp_path / "comparison.png"
    files = _save_lyapunov_phase_portrait_comparison(
        model=model,
        env=env,
        path=comp_path,
        num_trajectories=1,
        grid_lim=0.75,
        grid_n=5,
    )

    assert comp_path.exists()
    true_hist = tmp_path / "phase_portrait_vector_hist_true.png"
    learned_hist = tmp_path / "phase_portrait_vector_hist_learned.png"
    assert true_hist.exists()
    assert learned_hist.exists()
    assert "phase_portrait_comparison" in files
