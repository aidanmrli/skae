"""Unit tests for HyperLISTA encoder and HyperLISTAKM model."""

import torch
import pytest
from skae.config import get_config, Config
from skae.model import (
    HyperLISTA,
    HyperLISTAKM,
    make_model,
)


class TestHyperLISTA:
    """Test HyperLISTA encoder module."""
    
    def test_initialization(self):
        """Test HyperLISTA can be initialized."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        xdim = 10
        zdim = cfg.MODEL.TARGET_SIZE
        
        # Create a mock dictionary parameter
        dict_param = torch.nn.Parameter(torch.randn(zdim, xdim))
        
        hyperlista = HyperLISTA(cfg, xdim, dict_param)
        assert hyperlista.xdim == xdim
        assert hyperlista.zdim == zdim
        assert hyperlista.num_loops == cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS
    
    def test_forward_shape(self):
        """Test HyperLISTA forward pass output shape."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 5
        xdim = 10
        zdim = cfg.MODEL.TARGET_SIZE
        
        dict_param = torch.nn.Parameter(torch.randn(zdim, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)
        
        x = torch.randn(16, xdim)
        z = hyperlista(x)
        assert z.shape == (16, zdim)
    
    def test_sparsity(self):
        """Test HyperLISTA produces sparse codes."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 128
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 10
        cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = 0.01
        xdim = 10
        zdim = cfg.MODEL.TARGET_SIZE
        
        dict_param = torch.nn.Parameter(torch.randn(zdim, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)
        
        x = torch.randn(8, xdim)
        z = hyperlista(x)
        
        # Count nonzero elements (with tolerance)
        nonzero_ratio = (z.abs() > 1e-6).float().mean().item()
        # Should have some sparsity (less than 100% active)
        assert nonzero_ratio < 1.0
    
    def test_learnable_hyperparams(self):
        """Test hyperparameters are learnable when configured."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = True
        xdim = 5
        
        dict_param = torch.nn.Parameter(torch.randn(32, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)
        
        # Check hyperparams are nn.Parameters
        assert isinstance(hyperlista.c_theta, torch.nn.Parameter)
        assert isinstance(hyperlista.c_beta, torch.nn.Parameter)
        assert isinstance(hyperlista.c_ss, torch.nn.Parameter)
    
    def test_fixed_hyperparams(self):
        """Test hyperparameters are fixed buffers when configured."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = False
        xdim = 5
        
        dict_param = torch.nn.Parameter(torch.randn(32, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)
        
        # Check hyperparams are buffers, not parameters
        param_names = [n for n, _ in hyperlista.named_parameters()]
        assert 'c_theta' not in param_names
        assert 'c_beta' not in param_names
        assert 'c_ss' not in param_names
    
    def test_no_momentum(self):
        """Test HyperLISTA works without momentum."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.HYPERLISTA.USE_MOMENTUM = False
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 3
        xdim = 5
        
        dict_param = torch.nn.Parameter(torch.randn(32, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)
        
        x = torch.randn(4, xdim)
        z = hyperlista(x)
        assert z.shape == (4, 32)
    
    def test_no_support_selection(self):
        """Test HyperLISTA works without support selection."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.HYPERLISTA.USE_SUPPORT_SELECTION = False
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 3
        xdim = 5
        
        dict_param = torch.nn.Parameter(torch.randn(32, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)
        
        x = torch.randn(4, xdim)
        z = hyperlista(x)
        assert z.shape == (4, 32)


class TestHyperLISTAKM:
    """Test HyperLISTAKM model."""
    
    def test_initialization(self):
        """Test HyperLISTAKM can be initialized."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        obs_size = 2
        model = HyperLISTAKM(cfg, obs_size)
        assert model.observation_size == obs_size
        assert model.target_size == cfg.MODEL.TARGET_SIZE
    
    def test_encode_decode_shape(self):
        """Test encode and decode output shapes."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 128
        obs_size = 3
        batch_size = 16
        
        model = HyperLISTAKM(cfg, obs_size)
        x = torch.randn(batch_size, obs_size)
        
        # Test encode
        z = model.encode(x)
        assert z.shape == (batch_size, cfg.MODEL.TARGET_SIZE)
        
        # Test decode
        x_recon = model.decode(z)
        assert x_recon.shape == (batch_size, obs_size)
    
    def test_sparse_encoding(self):
        """Test HyperLISTAKM produces sparse encodings."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 256
        cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = 0.01
        obs_size = 2
        
        model = HyperLISTAKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        z = model.encode(x)
        
        # Check sparsity
        nonzero = (z.abs() > 1e-6).float().sum(dim=-1).mean()
        sparsity_ratio = 1.0 - nonzero / cfg.MODEL.TARGET_SIZE
        assert sparsity_ratio > 0.0  # Should have some sparsity
    
    def test_kmatrix_shape(self):
        """Test Koopman matrix has correct shape."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        obs_size = 2
        
        model = HyperLISTAKM(cfg, obs_size)
        kmat = model.kmatrix()
        assert kmat.shape == (64, 64)
    
    def test_loss_computation(self):
        """Test full loss computation."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 128
        obs_size = 2
        
        model = HyperLISTAKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        
        loss, metrics = model.loss(x, nx)
        
        # Check loss is scalar
        assert loss.ndim == 0
        
        # Check metrics
        assert 'loss' in metrics
        assert 'residual_loss' in metrics
        assert 'reconst_loss' in metrics
        assert 'sparsity_loss' in metrics
        assert 'sparsity_ratio' in metrics
    
    def test_homogeneous_coordinates(self):
        """Test HyperLISTAKM with homogeneous coordinates."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.USE_HOMOGENEOUS = True
        obs_size = 2
        
        model = HyperLISTAKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        
        # Encode and decode
        z = model.encode(x)
        x_recon = model.decode(z)
        
        # Output shape should be obs_size (not internal size)
        assert x_recon.shape == (4, obs_size)
        
        # Check homogeneous coordinate
        c_hat = model.get_homogeneous_coord(z)
        assert c_hat.shape == (4,)
    
    def test_homogeneous_loss(self):
        """Test homogeneous loss computation."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.USE_HOMOGENEOUS = True
        obs_size = 2
        
        model = HyperLISTAKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        nx = torch.randn(4, obs_size)
        
        loss, metrics = model.loss(x, nx)
        
        # Should have homogeneous loss in metrics
        assert 'homogeneous_loss' in metrics


class TestHyperLISTAGradientFlow:
    """Test gradient flow through HyperLISTA."""
    
    def test_hyperlista_gradient_to_dict(self):
        """Verify gradients flow from loss to dictionary."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 3
        obs_size = 3
        
        model = HyperLISTAKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        
        loss, _ = model.loss(x, nx)
        loss.backward()
        
        # Check gradients exist on dictionary
        assert model.dict.grad is not None
        assert torch.any(model.dict.grad != 0), "Dictionary gradient should be non-zero"
    
    def test_hyperlista_gradient_to_kmat(self):
        """Verify gradients flow to Koopman matrix."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 3
        obs_size = 3
        
        model = HyperLISTAKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        
        loss, _ = model.loss(x, nx)
        loss.backward()
        
        # Check gradients exist on Koopman matrix
        assert model.kmat.grad is not None
        assert torch.any(model.kmat.grad != 0), "Koopman matrix gradient should be non-zero"
    
    def test_learnable_hyperparams_gradient(self):
        """Verify gradients flow to learnable hyperparameters.
        
        Note: c_ss does not receive gradients because the support selection
        computation uses torch.no_grad() for the non-differentiable quantile
        operation. This is expected behavior per the HyperLISTA paper.
        """
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 3
        cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = True
        obs_size = 3
        
        model = HyperLISTAKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        
        loss, _ = model.loss(x, nx)
        loss.backward()
        
        # c_theta and c_beta should have gradients
        assert model.hyperlista.c_theta.grad is not None
        assert model.hyperlista.c_beta.grad is not None
        
        # c_ss does NOT receive gradients because support selection uses
        # torch.no_grad() for the non-differentiable quantile operation
        # This is expected behavior - c_ss is tuned via grid search instead
        # (see tune_hyperlista.py)


class TestHyperLISTAModelFactory:
    """Test model factory with HyperLISTA."""
    
    def test_make_model_hyperlista(self):
        """Test creating HyperLISTAKM via factory."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        obs_size = 2
        model = make_model(cfg, obs_size)
        assert isinstance(model, HyperLISTAKM)
    
    def test_config_hyperlista(self):
        """Test 'hyperlista' config preset."""
        cfg = get_config("hyperlista")
        assert cfg.MODEL.MODEL_NAME == "HyperLISTAKM"
        assert cfg.MODEL.USE_HOMOGENEOUS == True
        assert cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS == True


class TestHyperLISTAShrinkOperators:
    """Test soft-thresholding operators."""
    
    def test_shrink_basic(self):
        """Test basic soft thresholding."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 16
        
        dict_param = torch.nn.Parameter(torch.randn(16, 5))
        hyperlista = HyperLISTA(cfg, 5, dict_param)
        
        x = torch.tensor([2.0, 1.5, 0.5, -2.0, -1.5, -0.5])
        theta = torch.tensor([1.0])
        
        result = hyperlista._shrink(x, theta)
        expected = torch.tensor([1.0, 0.5, 0.0, -1.0, -0.5, 0.0])
        assert torch.allclose(result, expected)
    
    def test_shrink_ss_bypasses_large_entries(self):
        """Test support selection bypasses large magnitude entries."""
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 8
        
        dict_param = torch.nn.Parameter(torch.randn(8, 3))
        hyperlista = HyperLISTA(cfg, 3, dict_param)
        
        # Create input where some entries should be bypassed
        x = torch.tensor([[10.0, 5.0, 0.1, 0.05, -10.0, -5.0, -0.1, -0.05]])
        theta = torch.tensor([[1.0]])
        p = torch.tensor([[0.5]])  # Top 50% should be bypassed
        
        result = hyperlista._shrink_ss(x, theta, p)
        
        # Large entries (10, 5, -10, -5) should be bypassed (kept as-is)
        assert result[0, 0] == 10.0  # Bypassed
        assert result[0, 1] == 5.0   # Bypassed
        assert result[0, 4] == -10.0 # Bypassed
        assert result[0, 5] == -5.0  # Bypassed
        
        # Small entries should be shrunk to zero
        assert result[0, 2] == 0.0   # Shrunk
        assert result[0, 3] == 0.0   # Shrunk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
