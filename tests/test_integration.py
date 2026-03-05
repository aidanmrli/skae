"""Integration tests for model + data compatibility."""

import torch
import pytest
from skae.config import get_config
from skae.data import make_env, VectorWrapper, generate_trajectory
from skae.model import make_model


def make_unified_loss_inputs(model, x: torch.Tensor, nx: torch.Tensor):
    """Build minimal unified-loss tensors for a 1-step horizon."""
    x_pred = nx.unsqueeze(1)
    x_true = nx.unsqueeze(1)
    z0 = model.encode(x)
    z_true = model.encode(nx).unsqueeze(1)
    z_pred = model.rollout_latent_discrete(z0, horizon=1)
    x_recon_true = model.decode(z_true.reshape(-1, z_true.shape[-1])).reshape_as(x_true)
    homogeneous_loss = None
    if getattr(model, "use_homogeneous", False) and hasattr(model, "get_homogeneous_coord"):
        c_hat = model.get_homogeneous_coord(z_pred.reshape(-1, z_pred.shape[-1]))
        homogeneous_loss = torch.mean((c_hat - 1.0) ** 2)
    return {
        "x_pred": x_pred,
        "x_true": x_true,
        "x0": x,
        "z0": z0,
        "z_pred": z_pred,
        "z_true": z_true,
        "reconstruction_error": torch.norm(x_true - x_recon_true, dim=-1).mean(),
        "sparsity_latent": z_pred,
        "homogeneous_loss": homogeneous_loss,
    }


class TestModelDataIntegration:
    """Test integration between model and data modules."""
    
    def test_generic_km_with_pendulum(self):
        """Test GenericKM model with Pendulum environment."""
        cfg = get_config("generic")
        cfg.ENV.ENV_NAME = "pendulum"
        cfg.MODEL.TARGET_SIZE = 32
        
        # Create environment
        env = make_env(cfg)
        obs_size = env.observation_size
        
        # Create model
        model = make_model(cfg, obs_size)
        
        # Generate some data
        rng = torch.Generator()
        rng.manual_seed(42)
        x0 = env.reset(rng)
        trajectory = generate_trajectory(
            env_step=lambda s: env.step(s, None),
            init_state=x0,
            length=10
        )
        
        # Test forward pass
        x = trajectory[:-1]
        nx = trajectory[1:]
        
        loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))
        
        assert loss.ndim == 0
        assert loss >= 0
        assert 'loss' in metrics
    
    def test_listakm_with_duffing(self):
        """Test LISTAKM model with Duffing environment."""
        cfg = get_config("lista")
        cfg.ENV.ENV_NAME = "duffing"
        cfg.MODEL.TARGET_SIZE = 128
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 5
        
        # Create environment
        env = make_env(cfg)
        obs_size = env.observation_size
        
        # Create model
        model = make_model(cfg, obs_size)
        
        # Generate some data
        rng = torch.Generator()
        rng.manual_seed(123)
        x0 = env.reset(rng)
        trajectory = generate_trajectory(
            env_step=lambda s: env.step(s, None),
            init_state=x0,
            length=10
        )
        
        # Test forward pass
        x = trajectory[:-1]
        nx = trajectory[1:]
        
        loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))
        
        assert loss.ndim == 0
        assert loss >= 0
        assert 'sparsity_ratio' in metrics
    
    def test_generic_km_with_vectorized_env(self):
        """Test GenericKM with vectorized environment."""
        cfg = get_config("generic")
        cfg.ENV.ENV_NAME = "lotka_volterra"
        cfg.MODEL.TARGET_SIZE = 16
        
        # Create vectorized environment
        env = make_env(cfg)
        batch_size = 8
        vec_env = VectorWrapper(env, batch_size)
        obs_size = vec_env.observation_size
        
        # Create model
        model = make_model(cfg, obs_size)
        
        # Generate batch of initial states
        rng = torch.Generator()
        rng.manual_seed(42)
        x = vec_env.reset(rng)
        nx = vec_env.step(x)
        
        # Test batch forward pass
        loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))
        
        assert loss.ndim == 0
        assert x.shape == (batch_size, obs_size)
        assert nx.shape == (batch_size, obs_size)
    
    def test_prediction_through_model(self):
        """Test multi-step prediction through the model."""
        cfg = get_config("generic")
        cfg.ENV.ENV_NAME = "duffing"
        cfg.MODEL.TARGET_SIZE = 32
        
        # Create environment
        env = make_env(cfg)
        obs_size = env.observation_size
        
        # Create model
        model = make_model(cfg, obs_size)
        
        # Generate ground truth trajectory
        rng = torch.Generator()
        rng.manual_seed(42)
        x0 = env.reset(rng)
        gt_trajectory = generate_trajectory(
            env_step=lambda s: env.step(s, None),
            init_state=x0,
            length=5
        )
        
        # Generate model predictions
        pred_states = []
        x = x0
        for _ in range(5):
            x = model.step_env(x)
            pred_states.append(x)
        
        pred_trajectory = torch.stack(pred_states)
        
        # Check shapes match
        assert pred_trajectory.shape == gt_trajectory.shape
        # Initially untrained model, so predictions may be poor, but structure should work
        assert pred_trajectory.shape[0] == 5  # 5 steps
    
    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip with real environment data."""
        cfg = get_config("generic")
        cfg.ENV.ENV_NAME = "pendulum"
        cfg.MODEL.TARGET_SIZE = 64
        
        # Create environment and model
        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)
        
        # Get some observations
        rng = torch.Generator()
        rng.manual_seed(42)
        states = []
        for _ in range(10):
            states.append(env.reset(rng))
        x = torch.stack(states)
        
        # Encode-decode roundtrip
        z = model.encode(x)
        x_recon = model.decode(z)
        
        assert z.shape == (10, cfg.MODEL.TARGET_SIZE)
        assert x_recon.shape == x.shape
    
    def test_training_step_simulation(self):
        """Test a simulated training step."""
        cfg = get_config("generic_sparse")
        cfg.ENV.ENV_NAME = "duffing"
        cfg.MODEL.TARGET_SIZE = 32
        
        # Create environment and model
        env = make_env(cfg)
        vec_env = VectorWrapper(env, batch_size=16)
        model = make_model(cfg, env.observation_size)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Generate training data
        rng = torch.Generator()
        rng.manual_seed(42)
        x = vec_env.reset(rng)
        nx = vec_env.step(x)
        
        # Training step
        optimizer.zero_grad()
        loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))
        loss.backward()
        optimizer.step()
        
        # Check everything worked
        assert loss.ndim == 0
        assert 'loss' in metrics
        # Check gradients were applied
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_all_environments_with_generic_km(self):
        """Test GenericKM works with all environment types."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        
        env_names = ["pendulum", "duffing", "lotka_volterra", "lorenz63", "parabolic"]
        
        for env_name in env_names:
            cfg.ENV.ENV_NAME = env_name
            
            # Create environment and model
            env = make_env(cfg)
            model = make_model(cfg, env.observation_size)
            
            # Generate data
            rng = torch.Generator()
            rng.manual_seed(42)
            x0 = env.reset(rng)
            trajectory = generate_trajectory(
                env_step=lambda s: env.step(s, None),
                init_state=x0,
                length=5
            )
            
            x = trajectory[:-1]
            nx = trajectory[1:]
            
            # Test loss computation
            loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))
            
            assert loss.ndim == 0, f"Failed for {env_name}"
            assert loss >= 0, f"Failed for {env_name}"

    def test_hyperlista_config_end_to_end_through_make_model_and_loss(self):
        """HyperLISTA preset should run end-to-end through unified LISTAKM factory path."""
        cfg = get_config("hyperlista")
        cfg.ENV.ENV_NAME = "duffing"
        cfg.MODEL.TARGET_SIZE = 64

        env = make_env(cfg)
        model = make_model(cfg, env.observation_size)

        rng = torch.Generator().manual_seed(42)
        x = env.reset(rng)
        nx = env.step(x, None)

        loss, metrics = model.loss(**make_unified_loss_inputs(model, x.unsqueeze(0), nx.unsqueeze(0)))

        assert loss.ndim == 0
        assert "loss" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
