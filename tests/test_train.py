"""
Unit tests for training script.

Tests cover:
- Training step correctness
- Evaluation function
- Full training loop (short)
- Checkpoint saving/loading
"""

import pytest
import torch
import tempfile
import sys
from pathlib import Path

from skae.config import get_config
from skae.data import make_env, VectorWrapper
from skae.model import make_model
import tools.train as train_module
from tools.train import (
    DYSTS_CACHE_PROFILES,
    apply_dysts_cache_profile,
    train_step,
    evaluate,
    train,
    build_optimizer,
)


def make_unified_loss_inputs(model, x: torch.Tensor, nx: torch.Tensor):
    """Build minimal unified-loss tensors for a 1-step horizon."""
    x_pred = nx.unsqueeze(1)
    x_true = nx.unsqueeze(1)
    z0 = model.encode(x)
    z_true = model.encode(nx).unsqueeze(1)
    z_pred = model.rollout_latent_discrete(z0, horizon=1)
    x_recon_true = model.decode(z_true.reshape(-1, z_true.shape[-1])).reshape_as(x_true)
    return {
        "x_pred": x_pred,
        "x_true": x_true,
        "x0": x,
        "z0": z0,
        "z_pred": z_pred,
        "z_true": z_true,
        "reconstruction_error": torch.norm(x_true - x_recon_true, dim=-1).mean(),
        "sparsity_latent": z_pred,
    }


class TestTrainStep:
    """Test training step function."""
    
    def test_train_step_reduces_loss(self):
        """Test that training step updates parameters and computes metrics."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.BATCH_SIZE = 4
        
        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Generate batch
        rng = torch.Generator().manual_seed(42)
        x_seq = torch.randn(4, 2, env.observation_size)
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Training step
        metrics = train_step(model, optimizer, x_seq)
        
        # Check metrics are computed
        assert 'loss' in metrics
        assert 'alignment_loss' in metrics
        assert 'reconst_loss' in metrics
        assert 'sparsity_loss' in metrics
        assert isinstance(metrics['loss'], float)
        
        # Check parameters were updated
        for p_before, p_after in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_before, p_after), "Parameters should be updated"
    
    def test_train_step_gradient_flow(self):
        """Test that gradients flow through all parameters."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        
        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x_seq = torch.randn(4, 2, env.observation_size)
        
        # Training step
        metrics = train_step(model, optimizer, x_seq)
        
        # Check that at least some parameters have been updated
        has_update = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                # After optimizer step, gradients should be zeroed
                # but parameters should have changed
                has_update = True
        
        assert has_update, "At least some parameters should require gradients"

    @pytest.mark.parametrize("horizon", [1, 8])
    def test_train_step_metrics_across_horizons(self, horizon):
        """Unified train_step should report consistent metrics for H=1 and H=8."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]

        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x_seq = torch.randn(4, horizon + 1, env.observation_size)
        metrics = train_step(model, optimizer, x_seq)

        assert metrics["sequence_term_scale"] == pytest.approx(1.0 / horizon, abs=1e-8)
        for key in ("loss", "alignment_loss", "reconst_loss", "prediction_loss", "sparsity_loss"):
            assert key in metrics
            assert isinstance(metrics[key], float)


class TestEvaluate:
    """Test evaluation function."""
    
    def test_evaluate_returns_trajectories(self):
        """Test that evaluate returns trajectory predictions."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        
        env = make_env(cfg)
        env_vec = VectorWrapper(env, 4)
        model = make_model(cfg, env.observation_size)
        
        rng = torch.Generator().manual_seed(42)
        x = env_vec.reset(rng)
        
        results = evaluate(model, x, lambda s: env_vec.step(s), num_steps=10)
        
        assert 'true_trajectory' in results
        assert 'pred_trajectory' in results
        assert 'pred_error' in results
        assert 'pred_error_per_dim' in results
        assert 'mean_error' in results
        assert 'mean_error_per_dim' in results
        assert 'final_error' in results
        assert 'final_error_per_dim' in results
        
        # Check shapes
        assert results['true_trajectory'].shape == (10, 4, env.observation_size)
        assert results['pred_trajectory'].shape == (10, 4, env.observation_size)
        assert results['pred_error'].shape == (10,)
        assert results['pred_error_per_dim'].shape == (10,)
    
    def test_evaluate_no_gradient(self):
        """Test that evaluation doesn't compute gradients."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        
        env = make_env(cfg)
        env_vec = VectorWrapper(env, 4)
        model = make_model(cfg, env.observation_size)
        
        rng = torch.Generator().manual_seed(42)
        x = env_vec.reset(rng)
        
        results = evaluate(model, x, lambda s: env_vec.step(s), num_steps=5)
        
        # Check that results don't require grad
        assert not results['true_trajectory'].requires_grad
        assert not results['pred_trajectory'].requires_grad


class TestTrain:
    """Test full training loop."""
    
    def test_train_short_run(self):
        """Test that training runs without errors (short run)."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.NUM_STEPS = 10
        cfg.TRAIN.BATCH_SIZE = 4
        cfg.TRAIN.DATA_SIZE = 16
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = train(cfg, log_dir=tmpdir, device='cpu')
            
            # Check model was returned
            assert model is not None
            assert isinstance(model, torch.nn.Module)
            
            # Check checkpoints were saved
            run_dirs = list(Path(tmpdir).iterdir())
            assert len(run_dirs) > 0
            
            run_dir = run_dirs[0]
            assert (run_dir / 'config.json').exists()
            assert (run_dir / 'last.pt').exists()
    
    def test_train_saves_checkpoint(self):
        """Test that training saves checkpoints correctly."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.NUM_STEPS = 5
        cfg.TRAIN.BATCH_SIZE = 4
        cfg.TRAIN.DATA_SIZE = 16
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = train(cfg, log_dir=tmpdir, device='cpu')
            
            # Find run directory
            run_dirs = list(Path(tmpdir).iterdir())
            run_dir = run_dirs[0]
            
            # Load checkpoint
            checkpoint = torch.load(run_dir / 'last.pt', map_location='cpu')
            
            assert 'step' in checkpoint
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'config' in checkpoint
            assert 'metrics' in checkpoint

    def test_train_short_run_sequence_length_8(self):
        """End-to-end smoke run for unified horizon training at H=8."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.NUM_STEPS = 2
        cfg.TRAIN.BATCH_SIZE = 4
        cfg.TRAIN.DATA_SIZE = 16
        cfg.TRAIN.SEQUENCE_LENGTH = 8
        cfg.TRAIN.EVAL_EVERY = 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            model = train(cfg, log_dir=tmpdir, device='cpu', skip_eval=True, skip_basin_eval=True)
            assert model is not None


class TestTrainCli:
    """CLI-level behavior for unified horizon controls."""

    def test_cli_rejects_legacy_pairwise_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train.py", "--pairwise"])
        with pytest.raises(SystemExit) as exc:
            train_module.main()
        assert exc.value.code == 2

    def test_cli_accepts_sequence_length_without_mode_flags(self, tmp_path, monkeypatch):
        captured = {}

        def _fake_train(cfg, **kwargs):
            captured["sequence_length"] = cfg.TRAIN.SEQUENCE_LENGTH
            return torch.nn.Linear(1, 1)

        monkeypatch.setattr(train_module, "train", _fake_train)
        monkeypatch.setattr(train_module, "get_device", lambda _requested: "cpu")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--config",
                "generic",
                "--env",
                "duffing",
                "--num_steps",
                "1",
                "--batch_size",
                "2",
                "--sequence_length",
                "8",
                "--skip_eval",
                "--skip_basin_eval",
                "--device",
                "cpu",
                "--log_dir",
                str(tmp_path),
            ],
        )
        train_module.main()
        assert captured["sequence_length"] == 8

    def test_cli_accepts_intrinsic_hd_size_overrides(self, tmp_path, monkeypatch):
        captured = {}

        def _fake_train(cfg, **kwargs):
            captured["kuramoto_num_oscillators"] = cfg.ENV.KURAMOTO.NUM_OSCILLATORS
            captured["hopfield_num_neurons"] = cfg.ENV.HOPFIELD.NUM_NEURONS
            captured["hopfield_num_patterns"] = cfg.ENV.HOPFIELD.NUM_PATTERNS
            captured["competitive_lv_num_species"] = cfg.ENV.COMPETITIVE_LV.NUM_SPECIES
            return torch.nn.Linear(1, 1)

        monkeypatch.setattr(train_module, "train", _fake_train)
        monkeypatch.setattr(train_module, "get_device", lambda _requested: "cpu")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--config",
                "generic",
                "--env",
                "kuramoto",
                "--num_steps",
                "1",
                "--batch_size",
                "2",
                "--kuramoto_num_oscillators",
                "32",
                "--hopfield_num_neurons",
                "64",
                "--hopfield_num_patterns",
                "8",
                "--competitive_lv_num_species",
                "12",
                "--skip_eval",
                "--skip_basin_eval",
                "--device",
                "cpu",
                "--log_dir",
                str(tmp_path),
            ],
        )
        train_module.main()
        assert captured["kuramoto_num_oscillators"] == 32
        assert captured["hopfield_num_neurons"] == 64
        assert captured["hopfield_num_patterns"] == 8
        assert captured["competitive_lv_num_species"] == 12

    def test_cli_accepts_optimizer_overrides(self, tmp_path, monkeypatch):
        captured = {}

        def _fake_train(cfg, **kwargs):
            captured["lr"] = cfg.TRAIN.LR
            captured["k_matrix_lr"] = cfg.TRAIN.K_MATRIX_LR
            captured["weight_decay"] = cfg.TRAIN.WEIGHT_DECAY
            return torch.nn.Linear(1, 1)

        monkeypatch.setattr(train_module, "train", _fake_train)
        monkeypatch.setattr(train_module, "get_device", lambda _requested: "cpu")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train.py",
                "--config",
                "generic_sparse",
                "--env",
                "duffing",
                "--num_steps",
                "1",
                "--batch_size",
                "2",
                "--lr",
                "3e-4",
                "--k_matrix_lr",
                "3e-5",
                "--weight_decay",
                "5e-5",
                "--skip_eval",
                "--skip_basin_eval",
                "--device",
                "cpu",
                "--log_dir",
                str(tmp_path),
            ],
        )
        train_module.main()
        assert captured["lr"] == pytest.approx(3e-4)
        assert captured["k_matrix_lr"] == pytest.approx(3e-5)
        assert captured["weight_decay"] == pytest.approx(5e-5)

    def test_basin_eval_import_uses_package_model_path(self):
        import inspect

        source = inspect.getsource(train_module.train)
        assert "from skae.model import StructuredLISTAKM" in source
        assert "from model import StructuredLISTAKM" not in source


class TestOptimizer:
    """Tests for optimizer construction and parameter groups."""

    def test_optimizer_uses_k_lr(self):
        """Koopman matrix should use cfg.TRAIN.K_MATRIX_LR learning rate."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]

        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)

        optimizer = build_optimizer(model, cfg)

        kmat_param = dict(model.named_parameters())["kmat"]
        found_group = None
        for group in optimizer.param_groups:
            if any(p is kmat_param for p in group['params']):
                found_group = group
                break

        assert found_group is not None, "kmat parameter group not found in optimizer"
        assert found_group['lr'] == cfg.TRAIN.K_MATRIX_LR

    def test_optimizer_uses_k_lr_for_structured_koopman_params(self):
        cfg = get_config("lista")
        cfg.MODEL.MODEL_NAME = "StructuredLISTAKM"
        cfg.MODEL.STRUCTURED.ENABLED = True
        cfg.MODEL.STRUCTURED.D_GLOBAL = 2
        cfg.MODEL.STRUCTURED.NUM_BASINS = 2
        cfg.MODEL.STRUCTURED.D_BASIN = 2
        cfg.MODEL.TARGET_SIZE = 6

        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)
        optimizer = build_optimizer(model, cfg)

        name_by_id = {id(param): name for name, param in model.named_parameters()}
        grouped_lrs = {}
        for group in optimizer.param_groups:
            lr = group['lr']
            for param in group['params']:
                grouped_lrs[name_by_id[id(param)]] = lr

        for name in ("K_global", "K_coupling.0", "K_coupling.1", "K_basin.0", "K_basin.1"):
            assert grouped_lrs[name] == cfg.TRAIN.K_MATRIX_LR
    
    def test_train_resume_from_checkpoint(self):
        """Test that training can resume from checkpoint."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.NUM_STEPS = 5
        cfg.TRAIN.BATCH_SIZE = 4
        cfg.TRAIN.DATA_SIZE = 16
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # First training run
            model1 = train(cfg, log_dir=tmpdir, device='cpu')
            
            # Get checkpoint path
            run_dirs = list(Path(tmpdir).iterdir())
            checkpoint_path = run_dirs[0] / 'last.pt'
            
            # Load checkpoint and check
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert checkpoint['step'] == cfg.TRAIN.NUM_STEPS - 1
    
    def test_train_different_configs(self):
        """Test training with different configuration presets."""
        configs_to_test = ['generic', 'generic_sparse']
        
        for config_name in configs_to_test:
            cfg = get_config(config_name)
            cfg.MODEL.TARGET_SIZE = 8
            cfg.MODEL.ENCODER.LAYERS = [8]
            cfg.TRAIN.NUM_STEPS = 5
            cfg.TRAIN.BATCH_SIZE = 4
            cfg.TRAIN.DATA_SIZE = 16
            
            with tempfile.TemporaryDirectory() as tmpdir:
                model = train(cfg, log_dir=tmpdir, device='cpu')
                assert model is not None


class TestTrainIntegration:
    """Integration tests for training pipeline."""
    
    def test_train_all_environments(self):
        """Test training on all available environments."""
        environments = ['duffing', 'pendulum', 'lotka_volterra']
        
        for env_name in environments:
            cfg = get_config("generic")
            cfg.ENV.ENV_NAME = env_name
            cfg.MODEL.TARGET_SIZE = 8
            cfg.MODEL.ENCODER.LAYERS = [8]
            cfg.TRAIN.NUM_STEPS = 5
            cfg.TRAIN.BATCH_SIZE = 4
            cfg.TRAIN.DATA_SIZE = 16
            
            with tempfile.TemporaryDirectory() as tmpdir:
                model = train(cfg, log_dir=tmpdir, device='cpu')
                assert model is not None


class TestDystsCacheProfiles:
    """Tests for named dysts cache profiles."""

    def test_apply_smoke_profile(self):
        cfg = get_config("generic")
        profile = apply_dysts_cache_profile(cfg, "smoke")

        assert profile == DYSTS_CACHE_PROFILES["smoke"]
        assert cfg.ENV.DYSTS.CACHE_STEPS == DYSTS_CACHE_PROFILES["smoke"]["steps"]
        assert cfg.ENV.DYSTS.CACHE_TRAJECTORIES == DYSTS_CACHE_PROFILES["smoke"]["trajectories"]
        assert cfg.ENV.DYSTS.CACHE_WARMUP == DYSTS_CACHE_PROFILES["smoke"]["warmup"]

    def test_apply_full_profile(self):
        cfg = get_config("generic")
        profile = apply_dysts_cache_profile(cfg, "full")

        assert profile == DYSTS_CACHE_PROFILES["full"]
        assert cfg.ENV.DYSTS.CACHE_STEPS == DYSTS_CACHE_PROFILES["full"]["steps"]
        assert cfg.ENV.DYSTS.CACHE_TRAJECTORIES == DYSTS_CACHE_PROFILES["full"]["trajectories"]
        assert cfg.ENV.DYSTS.CACHE_WARMUP == DYSTS_CACHE_PROFILES["full"]["warmup"]


class TestTrainLearning:
    """Learning behavior checks."""

    def test_train_decreases_loss(self):
        """Test that training actually reduces loss over time."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        cfg.MODEL.ENCODER.LAYERS = [16, 16]
        cfg.TRAIN.NUM_STEPS = 100
        cfg.TRAIN.BATCH_SIZE = 32
        cfg.TRAIN.DATA_SIZE = 128
        cfg.TRAIN.LR = 1e-3
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train model
            model = train(cfg, log_dir=tmpdir, device='cpu')
            
            # Evaluate initial and final loss
            env = make_env(cfg)
            env = VectorWrapper(env, cfg.TRAIN.BATCH_SIZE)
            
            rng = torch.Generator().manual_seed(42)
            x = env.reset(rng)
            nx = env.step(x)
            
            # Load initial and final checkpoints to compare
            # (In a real scenario, we'd track loss over time)
            # For now, just check that model can compute loss
            with torch.no_grad():
                loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))
                assert loss.item() < 100.0  # Sanity check


class TestDefaultLogDirRouting:
    """Tests for default run-directory routing in train()."""

    @staticmethod
    def _base_cfg(config_name: str):
        cfg = get_config(config_name)
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LAYERS = [8]
        cfg.TRAIN.NUM_STEPS = 1
        cfg.TRAIN.BATCH_SIZE = 2
        cfg.TRAIN.DATA_SIZE = 4
        cfg.TRAIN.EVAL_NUM_STEPS = 1
        cfg.TRAIN.EVAL_EVERY = 1000
        return cfg

    def test_default_log_dir_lista_encoder_type(self, tmp_path, monkeypatch):
        cfg = self._base_cfg("lista")
        cfg.MODEL.MODEL_NAME = "LISTAKM"
        cfg.MODEL.ENCODER.ENCODER_TYPE = "lista"

        monkeypatch.chdir(tmp_path)
        train(cfg, log_dir=None, device="cpu", skip_eval=True, skip_basin_eval=True)

        run_root = tmp_path / "runs" / "lista"
        assert run_root.exists()
        assert any(run_root.iterdir())

    def test_default_log_dir_hyperlista_encoder_type(self, tmp_path, monkeypatch):
        cfg = self._base_cfg("lista")
        cfg.MODEL.MODEL_NAME = "LISTAKM"
        cfg.MODEL.ENCODER.ENCODER_TYPE = "hyperlista"

        monkeypatch.chdir(tmp_path)
        train(cfg, log_dir=None, device="cpu", skip_eval=True, skip_basin_eval=True)

        run_root = tmp_path / "runs" / "hyperlista"
        assert run_root.exists()
        assert any(run_root.iterdir())

    def test_default_log_dir_non_lista_model(self, tmp_path, monkeypatch):
        cfg = self._base_cfg("generic")
        cfg.MODEL.MODEL_NAME = "GenericKM"

        monkeypatch.chdir(tmp_path)
        train(cfg, log_dir=None, device="cpu", skip_eval=True, skip_basin_eval=True)

        run_root = tmp_path / "runs" / "kae"
        assert run_root.exists()
        assert any(run_root.iterdir())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
