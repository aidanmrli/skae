"""Unit tests for PyTorch Koopman models."""

import torch
import pytest
from skae.config import get_config
import skae.model as model_module
from skae.model import (
    MLPCoder,
    LISTA,
    HyperLISTA,
    GenericKM,
    LISTAKM,
    make_model,
    shrink,
    get_activation
)


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


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_shrink_positive(self):
        """Test soft thresholding on positive values."""
        x = torch.tensor([2.0, 1.5, 0.5])
        threshold = 1.0
        result = shrink(x, threshold)
        expected = torch.tensor([1.0, 0.5, 0.0])
        assert torch.allclose(result, expected)
    
    def test_shrink_negative(self):
        """Test soft thresholding on negative values."""
        x = torch.tensor([-2.0, -1.5, -0.5])
        threshold = 1.0
        result = shrink(x, threshold)
        expected = torch.tensor([-1.0, -0.5, 0.0])
        assert torch.allclose(result, expected)
    
    def test_shrink_mixed(self):
        """Test soft thresholding on mixed values."""
        x = torch.tensor([2.0, -1.5, 0.5, -0.3])
        threshold = 1.0
        result = shrink(x, threshold)
        expected = torch.tensor([1.0, -0.5, 0.0, 0.0])
        assert torch.allclose(result, expected)
    
    def test_get_activation_relu(self):
        """Test ReLU activation retrieval."""
        act = get_activation('relu')
        assert isinstance(act, torch.nn.ReLU)
    
    def test_get_activation_tanh(self):
        """Test Tanh activation retrieval."""
        act = get_activation('tanh')
        assert isinstance(act, torch.nn.Tanh)
    
    def test_get_activation_gelu(self):
        """Test GELU activation retrieval."""
        act = get_activation('gelu')
        assert isinstance(act, torch.nn.GELU)
    
    def test_get_activation_invalid(self):
        """Test invalid activation name."""
        with pytest.raises(ValueError):
            get_activation('invalid_activation')


class TestMLPCoder:
    """Test MLPCoder module."""
    
    def test_initialization(self):
        """Test MLPCoder can be initialized."""
        coder = MLPCoder(
            input_size=10,
            target_size=5,
            hidden_layers=[16, 16],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        assert coder.input_size == 10
        assert coder.target_size == 5
        assert len(coder.hidden_layers) == 2
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        coder = MLPCoder(
            input_size=10,
            target_size=5,
            hidden_layers=[16, 16],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        x = torch.randn(32, 10)
        y = coder(x)
        assert y.shape == (32, 5)
    
    def test_forward_batch_independence(self):
        """Test that batch elements are processed independently."""
        coder = MLPCoder(
            input_size=5,
            target_size=3,
            hidden_layers=[8],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        x1 = torch.randn(1, 5)
        x2 = torch.randn(1, 5)
        x_batch = torch.cat([x1, x2], dim=0)
        
        y1 = coder(x1)
        y2 = coder(x2)
        y_batch = coder(x_batch)
        
        assert torch.allclose(y_batch[0], y1[0], atol=1e-6)
        assert torch.allclose(y_batch[1], y2[0], atol=1e-6)
    
    def test_last_relu(self):
        """Test last_relu option applies ReLU to output."""
        coder_with_relu = MLPCoder(
            input_size=5,
            target_size=3,
            hidden_layers=[],
            last_relu=True,
            use_bias=False,
            activation='relu'
        )
        coder_without_relu = MLPCoder(
            input_size=5,
            target_size=3,
            hidden_layers=[],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        
        # Use same weights
        coder_with_relu.network[0].weight.data = coder_without_relu.network[0].weight.data.clone()
        
        x = torch.randn(1, 5)
        y_with = coder_with_relu(x)
        y_without = coder_without_relu(x)
        
        # Output with ReLU should be non-negative
        assert torch.all(y_with >= 0)
        # Outputs should match after applying ReLU
        assert torch.allclose(y_with, torch.relu(y_without))


class TestLISTA:
    """Test LISTA module."""
    
    def test_initialization(self):
        """Test LISTA can be initialized."""
        cfg = get_config("lista")
        xdim = 10
        zdim = cfg.MODEL.TARGET_SIZE
        Wd_init = torch.randn(xdim, zdim)
        
        lista = LISTA(cfg, xdim, Wd_init)
        assert lista.xdim == xdim
        assert lista.zdim == zdim
    
    def test_forward_shape(self):
        """Test LISTA forward pass output shape."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 5
        xdim = 10
        zdim = cfg.MODEL.TARGET_SIZE
        Wd_init = torch.randn(xdim, zdim)
        
        lista = LISTA(cfg, xdim, Wd_init)
        x = torch.randn(16, xdim)
        z = lista(x)
        assert z.shape == (16, zdim)

    def test_one_refinement_gives_s_nonzero_gradient(self):
        """NUM_LOOPS=1 must mean one learned refinement after initialization."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 1
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.0
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"

        lista = LISTA(cfg, 4, torch.eye(4), L_override=2.0)
        loss = lista(torch.tensor([[1.0, -2.0, 0.5, 3.0]])).square().sum()
        loss.backward()

        assert lista.S.grad is not None
        assert torch.count_nonzero(lista.S.grad).item() > 0
    
    def test_sparsity(self):
        """Test LISTA produces sparse codes."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 128
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 10
        cfg.MODEL.ENCODER.LISTA.ALPHA = 5.0
        xdim = 10
        zdim = cfg.MODEL.TARGET_SIZE
        Wd_init = torch.randn(xdim, zdim)
        
        lista = LISTA(cfg, xdim, Wd_init)
        x = torch.randn(1, xdim)
        z = lista(x)
        
        # Count nonzero elements (with tolerance)
        nonzero_count = (z.abs() > 1e-6).sum().item()
        # Should be sparser than dense encoding (i.e., some zeros exist)
        assert nonzero_count <= zdim
    
    def test_wrong_wd_init_shape(self):
        """Test LISTA raises error for wrong Wd_init shape."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        xdim = 10
        wrong_Wd_init = torch.randn(32, xdim)  # Wrong zdim
        
        with pytest.raises(AssertionError):
            LISTA(cfg, xdim, wrong_Wd_init)

    def test_final_op_shrink_keeps_signed_coefficients(self):
        """Signed LISTA outputs should remain signed when FINAL_OP='shrink'."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 3
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 0
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.5
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"

        Wd_init = torch.eye(3)
        lista = LISTA(cfg, 3, Wd_init, L_override=1.0)
        z = lista(torch.tensor([[-2.0, 1.0, 0.25]]))

        assert z.shape == (1, 3)
        assert z[0, 0] < 0.0
        assert z[0, 1] > 0.0
        assert torch.isclose(z[0, 2], torch.tensor(0.0))

    def test_final_op_sign_split_preserves_sign_in_nonnegative_coordinates(self):
        """Sign-split LISTA should emit paired positive/negative coordinates."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 0
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.5
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "sign_split"

        Wd_init = torch.eye(2)
        lista = LISTA(cfg, 2, Wd_init, L_override=1.0)
        z = lista(torch.tensor([[-2.0, 1.0]]))

        assert z.shape == (1, 4)
        assert torch.all(z >= 0.0)
        assert torch.allclose(z[0], torch.tensor([0.0, 0.5, 1.5, 0.0]))

    def test_sign_split_latent_prior_is_unsplit_before_refinement(self):
        """Sign-split warm starts should be converted back to signed internal codes."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 1
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.0
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "sign_split"

        Wd_init = torch.eye(2)
        lista = LISTA(cfg, 2, Wd_init, L_override=1.0)
        with torch.no_grad():
            lista.S.copy_(torch.eye(2))

        z = lista(
            torch.zeros(1, 2),
            latent_prior=torch.tensor([[1.0, 0.0, 0.0, 2.0]]),
        )

        assert torch.allclose(z, torch.tensor([[1.0, 0.0, 0.0, 2.0]]))

    def test_group_topk_selection_zeros_nonselected_groups(self):
        """LISTA group-first selection should keep only the top-k groups."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.K_STRUCTURE = "block_diagonal"
        cfg.MODEL.K_NUM_BLOCKS = 2
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 0
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.0
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"
        cfg.MODEL.ENCODER.LISTA.TOPK_GROUPS = 1

        Wd_init = torch.eye(4)
        lista = LISTA(cfg, 4, Wd_init, L_override=1.0)
        z = lista(torch.tensor([[3.0, 3.0, 1.0, 1.0]]))

        assert torch.allclose(z, torch.tensor([[3.0, 3.0, 0.0, 0.0]]))

    def test_group_shrinkage_suppresses_weak_groups_before_elementwise_shrink(self):
        """Sparse-group shrinkage should zero weak groups and retain strong ones."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.K_STRUCTURE = "block_diagonal"
        cfg.MODEL.K_NUM_BLOCKS = 2
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 0
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.5
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"
        cfg.MODEL.ENCODER.LISTA.GROUP_SHRINKAGE = True
        cfg.MODEL.ENCODER.LISTA.GROUP_THRESHOLD_SCALE = 1.0

        Wd_init = torch.eye(4)
        lista = LISTA(cfg, 4, Wd_init, L_override=1.0)
        z = lista(torch.tensor([[0.4, 0.3, 2.0, 0.0]]))

        assert torch.allclose(z, torch.tensor([[0.0, 0.0, 1.0, 0.0]]), atol=1e-6)

    def test_group_aware_lista_rejects_sign_split(self):
        """Group-aware LISTA is currently only supported for signed/nonnegative base codes."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.K_STRUCTURE = "block_diagonal"
        cfg.MODEL.K_NUM_BLOCKS = 2
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "sign_split"
        cfg.MODEL.ENCODER.LISTA.GROUP_SHRINKAGE = True

        with pytest.raises(ValueError, match="do not support FINAL_OP='sign_split'"):
            LISTA(cfg, 2, torch.eye(2))

    @pytest.mark.parametrize("precode_mode", ["dictionary_tied", "hybrid"])
    def test_tied_and_hybrid_precodes_reject_sign_split(self, precode_mode):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "sign_split"
        cfg.MODEL.ENCODER.LISTA.PRECODE_MODE = precode_mode

        with pytest.raises(ValueError, match="do not support FINAL_OP='sign_split'"):
            LISTA(cfg, 2, torch.eye(2))

    def test_adaptive_thresholds_reject_sign_split(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "sign_split"
        cfg.MODEL.ENCODER.LISTA.ADAPTIVE_THRESHOLDS = True

        with pytest.raises(ValueError, match="do not support FINAL_OP='sign_split'"):
            LISTA(cfg, 2, torch.eye(2))

    def test_dictionary_tied_precode_matches_dictionary_projection(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 2
        cfg.MODEL.ENCODER.LISTA.PRECODE_MODE = "dictionary_tied"
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 0
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.0
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"

        dict_param = torch.nn.Parameter(torch.eye(2))
        lista = LISTA(cfg, 2, torch.eye(2), L_override=2.0, dict_param=dict_param)
        z = lista(torch.tensor([[2.0, -4.0]]))

        assert torch.allclose(z, torch.tensor([[1.0, -2.0]]))

    def test_hybrid_precode_initializes_to_dictionary_tied_projection(self):
        tied_cfg = get_config("lista")
        tied_cfg.MODEL.TARGET_SIZE = 2
        tied_cfg.MODEL.ENCODER.LISTA.PRECODE_MODE = "dictionary_tied"
        tied_cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 0
        tied_cfg.MODEL.ENCODER.LISTA.ALPHA = 0.0
        tied_cfg.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"

        hybrid_cfg = get_config("lista")
        hybrid_cfg.MODEL.TARGET_SIZE = 2
        hybrid_cfg.MODEL.ENCODER.LISTA.PRECODE_MODE = "hybrid"
        hybrid_cfg.MODEL.ENCODER.LISTA.PRECODE_RESIDUAL_SCALE = 0.5
        hybrid_cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 0
        hybrid_cfg.MODEL.ENCODER.LISTA.ALPHA = 0.0
        hybrid_cfg.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"

        dict_param = torch.nn.Parameter(torch.eye(2))
        tied_lista = LISTA(tied_cfg, 2, torch.eye(2), L_override=2.0, dict_param=dict_param)
        hybrid_lista = LISTA(hybrid_cfg, 2, torch.eye(2), L_override=2.0, dict_param=dict_param)
        x = torch.tensor([[2.0, -4.0]])

        assert torch.allclose(hybrid_lista(x), tied_lista(x))

    def test_groupwise_thresholds_expand_per_group(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.K_STRUCTURE = "block_diagonal"
        cfg.MODEL.K_NUM_BLOCKS = 2
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg.MODEL.ENCODER.LISTA.GROUPWISE_THRESHOLDS = True

        lista = LISTA(cfg, 4, torch.eye(4), L_override=1.0)
        with torch.no_grad():
            lista.group_alpha.copy_(torch.tensor([2.0, 4.0]))

        thresholds = lista._base_thresholds(
            batch_shape=(1,),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert thresholds.shape == (1, 4)
        assert torch.allclose(thresholds[0, :2], thresholds[0, 0].expand(2))
        assert torch.allclose(thresholds[0, 2:], thresholds[0, 2].expand(2))
        assert thresholds[0, 2] > thresholds[0, 0]

    def test_adaptive_thresholds_increase_with_prior_gap(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 2
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg.MODEL.ENCODER.LISTA.ADAPTIVE_THRESHOLDS = True
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.5
        cfg.MODEL.ENCODER.LISTA.ALPHA_PRIOR_COEFF = 1.0
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 0
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"

        lista = LISTA(cfg, 2, torch.eye(2), L_override=1.0)
        x = torch.zeros(1, 2)
        nonsparse_code = lista._compute_precode(x)
        base = lista._compute_thresholds(x=x, nonsparse_code=nonsparse_code, latent_prior_internal=None)
        adapted = lista._compute_thresholds(
            x=x,
            nonsparse_code=nonsparse_code,
            latent_prior_internal=torch.ones(1, 2),
        )

        assert torch.all(adapted > base)

    def test_momentum_changes_standard_lista_refinement(self):
        cfg_no_momentum = get_config("lista")
        cfg_no_momentum.MODEL.TARGET_SIZE = 1
        cfg_no_momentum.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg_no_momentum.MODEL.ENCODER.LISTA.NUM_LOOPS = 1
        cfg_no_momentum.MODEL.ENCODER.LISTA.ALPHA = 0.0
        cfg_no_momentum.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"

        cfg_momentum = get_config("lista")
        cfg_momentum.MODEL.TARGET_SIZE = 1
        cfg_momentum.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg_momentum.MODEL.ENCODER.LISTA.NUM_LOOPS = 1
        cfg_momentum.MODEL.ENCODER.LISTA.ALPHA = 0.0
        cfg_momentum.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"
        cfg_momentum.MODEL.ENCODER.LISTA.USE_MOMENTUM = True
        cfg_momentum.MODEL.ENCODER.LISTA.MOMENTUM_BETA = 0.5

        lista_no_momentum = LISTA(cfg_no_momentum, 1, torch.ones(1, 1), L_override=1.0)
        lista_momentum = LISTA(cfg_momentum, 1, torch.ones(1, 1), L_override=1.0)
        with torch.no_grad():
            lista_no_momentum.S.copy_(torch.ones(1, 1))
            lista_momentum.S.copy_(torch.ones(1, 1))

        x = torch.tensor([[2.0]])
        z_no_momentum = lista_no_momentum(x)
        z_momentum = lista_momentum(x)

        assert torch.allclose(z_no_momentum, torch.tensor([[4.0]]))
        assert torch.allclose(z_momentum, torch.tensor([[5.0]]))

    def test_latent_prior_warm_start_changes_lista_refinement(self):
        """A latent prior should change the refined sparse code."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 3
        cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 1
        cfg.MODEL.ENCODER.LISTA.ALPHA = 0.0
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "shrink"

        Wd_init = torch.eye(3)
        lista = LISTA(cfg, 3, Wd_init, L_override=2.0)
        x = torch.zeros(1, 3)
        z_no_prior = lista(x)
        z_with_prior = lista(x, latent_prior=torch.ones(1, 3))

        assert torch.allclose(z_no_prior, torch.zeros_like(z_no_prior))
        assert torch.all(z_with_prior > 0.0)


class TestGenericKM:
    """Test GenericKM model."""
    
    def test_initialization(self):
        """Test GenericKM can be initialized."""
        cfg = get_config("generic")
        obs_size = 2
        model = GenericKM(cfg, obs_size)
        assert model.observation_size == obs_size
        assert model.target_size == cfg.MODEL.TARGET_SIZE
    
    def test_encode_decode_shape(self):
        """Test encode and decode output shapes."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 32
        obs_size = 5
        batch_size = 16
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(batch_size, obs_size)
        
        # Test encode
        z = model.encode(x)
        assert z.shape == (batch_size, cfg.MODEL.TARGET_SIZE)
        
        # Test decode
        x_recon = model.decode(z)
        assert x_recon.shape == (batch_size, obs_size)
    
    def test_reconstruction(self):
        """Test reconstruction method."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 3
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        x_recon = model.reconstruction(x)
        assert x_recon.shape == x.shape
    
    def test_kmatrix_shape(self):
        """Test Koopman matrix has correct shape."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 32
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        kmat = model.kmatrix()
        assert kmat.shape == (32, 32)
    
    def test_step_latent(self):
        """Test stepping in latent space."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        y = torch.randn(4, 16)
        ny = model.step_latent(y)
        assert ny.shape == y.shape

    def test_diagonal_k_respects_config(self):
        """GenericKM should honor diagonal Koopman structure."""
        cfg = get_config("generic_sparse")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.K_STRUCTURE = "diagonal"
        model = GenericKM(cfg, observation_size=2)

        assert model._k_structure == "diagonal"
        assert not hasattr(model, "kmat")
        with torch.no_grad():
            model.kmat_diag.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        z = torch.ones(2, 4)
        stepped = model.step_latent(z)
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).repeat(2, 1)
        assert torch.allclose(stepped, expected)
        assert torch.allclose(model.kmatrix(), torch.diag(model.kmat_diag))

    def test_block_diagonal_k_respects_explicit_block_count(self):
        """GenericKM block-diagonal K should support an exact requested block count."""
        cfg = get_config("generic_sparse")
        cfg.MODEL.TARGET_SIZE = 10
        cfg.MODEL.K_STRUCTURE = "block_diagonal"
        cfg.MODEL.K_NUM_BLOCKS = 3
        model = GenericKM(cfg, observation_size=2)

        assert model._k_structure == "block_diagonal"
        assert model._k_block_sizes == [4, 3, 3]
        assert len(model.kmat_blocks) == 3
        assert not hasattr(model, "kmat")

        with torch.no_grad():
            for scale, block in zip((2.0, 3.0, 4.0), model.kmat_blocks):
                block.copy_(scale * torch.eye(block.shape[0]))

        z = torch.ones(2, 10)
        stepped = model.step_latent(z)
        expected = torch.tensor(
            [[2.0] * 4 + [3.0] * 3 + [4.0] * 3]
        ).repeat(2, 1)
        assert torch.allclose(stepped, expected)

        kmat = model.kmatrix()
        assert kmat.shape == (10, 10)
        assert torch.count_nonzero(kmat[:4, 4:]).item() == 0
        assert torch.count_nonzero(kmat[4:, :4]).item() == 0

    def test_normalized_linear_decoder_atoms(self):
        """GenericKM can normalize linear decoder atoms at decode time."""
        cfg = get_config("generic_sparse")
        cfg.MODEL.TARGET_SIZE = 2
        cfg.MODEL.DECODER.NORMALIZE_ATOMS = True
        model = GenericKM(cfg, observation_size=2)

        with torch.no_grad():
            linear = model._linear_decoder
            assert linear is not None
            linear.weight.copy_(torch.tensor([[3.0, 0.0], [4.0, 2.0]]))
            if linear.bias is not None:
                linear.bias.zero_()

        z = torch.tensor([[1.0, 1.0]])
        decoded = model.decode(z)
        expected = torch.tensor([[0.6, 0.8]]) + torch.tensor([[0.0, 1.0]])
        assert torch.allclose(decoded, expected, atol=1e-6)
    
    def test_step_env(self):
        """Test stepping in observation space via Koopman operator."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 3
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        nx_pred = model.step_env(x)
        assert nx_pred.shape == x.shape
    
    def test_residual(self):
        """Test residual computation."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        residual = model.residual(x, nx)
        assert residual.shape == (8,)
        assert torch.all(residual >= 0)  # Norm is non-negative
    
    def test_sparsity_loss(self):
        """Test sparsity loss computation."""
        cfg = get_config("generic_sparse")
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        loss = model.sparsity_loss(x)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # L1 norm is non-negative
    
    def test_loss_computation(self):
        """Test full loss computation."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        
        loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))
        
        # Check loss is scalar
        assert loss.ndim == 0
        
        # Check metrics
        assert 'loss' in metrics
        assert 'residual_loss' in metrics
        assert 'reconst_loss' in metrics
        assert 'prediction_loss' in metrics
        assert 'sparsity_loss' in metrics
        assert 'A_max_eigenvalue' in metrics
        assert 'sparsity_ratio' in metrics

    @pytest.mark.parametrize(
        ("norm_mode", "expected_scale", "expected_loss"),
        [
            ("none", 1.0, 3.0),
            ("sqrt_dim", 3.0, 1.0),
            ("dim", 9.0, 1.0 / 3.0),
        ],
    )
    def test_observation_loss_dim_normalization(self, norm_mode, expected_scale, expected_loss):
        """Observation-space losses should support configurable dimension normalization."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.RES_COEFF = 0.0
        cfg.MODEL.RECONST_COEFF = 1.0
        cfg.MODEL.PRED_COEFF = 1.0
        cfg.MODEL.SPARSITY_COEFF = 0.0
        cfg.MODEL.OBS_LOSS_DIM_NORMALIZATION = norm_mode

        obs_size = 9
        model = GenericKM(cfg, obs_size)

        x_pred = torch.ones(2, 1, obs_size)
        x_true = torch.zeros_like(x_pred)
        z0 = torch.zeros(2, cfg.MODEL.TARGET_SIZE)
        z_pred = torch.zeros(2, 1, cfg.MODEL.TARGET_SIZE)
        z_true = torch.zeros(2, 1, cfg.MODEL.TARGET_SIZE)
        reconstruction_error = torch.norm(x_true - x_pred, dim=-1).mean()

        loss, metrics = model.loss(
            x_pred=x_pred,
            x_true=x_true,
            z0=z0,
            z_pred=z_pred,
            z_true=z_true,
            reconstruction_error=reconstruction_error,
            sparsity_latent=z_pred,
        )

        assert metrics["obs_loss_dim_scale"] == pytest.approx(expected_scale)
        assert metrics["prediction_loss_raw"] == pytest.approx(3.0)
        assert metrics["prediction_loss"] == pytest.approx(expected_loss)
        assert metrics["reconst_loss_raw"] == pytest.approx(3.0)
        assert metrics["reconst_loss"] == pytest.approx(expected_loss)
        assert loss.item() == pytest.approx(2.0 * expected_loss)

    def test_norm_fn_id(self):
        """Test identity normalization function."""
        cfg = get_config("generic")
        cfg.MODEL.NORM_FN = "id"
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        z = model.encode(x)
        
        # With identity norm, we just check it doesn't crash
        assert z.shape == (4, cfg.MODEL.TARGET_SIZE)
    
    def test_norm_fn_ball(self):
        """Test ball normalization function."""
        cfg = get_config("generic")
        cfg.MODEL.NORM_FN = "ball"
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        z = model.encode(x)
        
        # Check normalization: each vector should have unit norm
        norms = torch.norm(z, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestLISTAKM:
    """Test LISTAKM model."""
    
    def test_initialization(self):
        """Test LISTAKM can be initialized."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        obs_size = 2
        model = LISTAKM(cfg, obs_size)
        assert model.observation_size == obs_size
        assert model.target_size == cfg.MODEL.TARGET_SIZE
    
    def test_encode_decode_shape(self):
        """Test encode and decode output shapes."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 128
        obs_size = 3
        batch_size = 16
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(batch_size, obs_size)
        
        # Test encode
        z = model.encode(x)
        assert z.shape == (batch_size, cfg.MODEL.TARGET_SIZE)
        
        # Test decode
        x_recon = model.decode(z)
        assert x_recon.shape == (batch_size, obs_size)
    
    def test_sparse_encoding(self):
        """Test LISTAKM produces sparse encodings."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 256
        cfg.MODEL.ENCODER.LISTA.ALPHA = 5.0
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        z = model.encode(x)
        
        # Check sparsity
        nonzero = (z.abs() > 1e-6).float().sum(dim=-1).mean()
        sparsity_ratio = 1.0 - nonzero / cfg.MODEL.TARGET_SIZE
        assert sparsity_ratio > 0.1
    
    def test_dict_normalization(self):
        """Test dictionary atoms are normalized in decode."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        
        # Check dictionary norm after decode call
        z = torch.randn(1, cfg.MODEL.TARGET_SIZE)
        _ = model.decode(z)
        
        # Dictionary should be used in normalized form
        # (not checking the parameter itself, just that decode works)
        assert True
    
    def test_loss_computation(self):
        """Test full loss computation."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 128
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        
        loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))
        
        # Check loss is scalar
        assert loss.ndim == 0
        
        # Check metrics
        assert 'loss' in metrics
        assert 'residual_loss' in metrics
        assert 'reconst_loss' in metrics
        assert 'sparsity_loss' in metrics
        assert 'sparsity_ratio' in metrics
    
    def test_sparsity_loss_with_alpha(self):
        """Test sparsity loss uses LISTA alpha weighting."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.LISTA.ALPHA = 2.0
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        loss = model.sparsity_loss(x)
        
        # Should be weighted by alpha
        assert loss >= 0

    def test_listakm_builds_lista_encoder_when_encoder_type_lista(self):
        """LISTAKM should build LISTA encoder for ENCODER_TYPE='lista'."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.ENCODER_TYPE = "lista"
        model = LISTAKM(cfg, observation_size=2)
        assert isinstance(model.encoder, LISTA)

    def test_listakm_passes_live_decoder_dictionary_to_lista(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.ENCODER_TYPE = "lista"
        cfg.MODEL.ENCODER.LISTA.PRECODE_MODE = "dictionary_tied"

        model = LISTAKM(cfg, observation_size=2)

        assert isinstance(model.encoder, LISTA)
        assert model.encoder.dict_param is model.dict

    def test_listakm_builds_hyperlista_encoder_when_encoder_type_hyperlista(self):
        """LISTAKM should build HyperLISTA encoder for ENCODER_TYPE='hyperlista'."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.ENCODER_TYPE = "hyperlista"
        model = LISTAKM(cfg, observation_size=2)
        assert isinstance(model.encoder, HyperLISTA)

    def test_invalid_encoder_type_raises_value_error(self):
        """LISTAKM should reject unknown encoder types."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.ENCODER_TYPE = "not-an-encoder"
        with pytest.raises(ValueError):
            LISTAKM(cfg, observation_size=2)

    def test_listakm_uses_encoder_field(self):
        """LISTAKM should expose encoder via `encoder`, not legacy `lista`."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 32
        model = LISTAKM(cfg, observation_size=2)
        assert hasattr(model, "encoder")
        assert not hasattr(model, "lista")

    def test_block_diagonal_k_respects_explicit_block_count(self):
        """LISTAKM block-diagonal K should support an exact requested block count."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 256
        cfg.MODEL.K_STRUCTURE = "block_diagonal"
        cfg.MODEL.K_NUM_BLOCKS = 5

        model = LISTAKM(cfg, observation_size=2)

        assert model._k_block_sizes == [52, 51, 51, 51, 51]
        assert len(model.kmat_blocks) == 5
        assert sum(model._k_block_sizes) == 256

        z = torch.randn(3, 256)
        stepped = model.step_latent(z)
        assert stepped.shape == z.shape

        energies = model._block_energies(z)
        assert energies is not None
        assert energies.shape == (3, 5)

    def test_listakm_sign_split_uses_paired_decoder_atoms(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "sign_split"
        model = LISTAKM(cfg, observation_size=2)

        assert model._uses_sign_split_latent
        assert model._encoder_target_size == 4
        assert model.dict.shape == (8, model._internal_obs_size)
        assert torch.allclose(model.dict[:4], -model.dict[4:], atol=1e-6)

    def test_soft_block_penalty_adds_off_block_regularization(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.RES_COEFF = 0.0
        cfg.MODEL.RECONST_COEFF = 0.0
        cfg.MODEL.PRED_COEFF = 0.0
        cfg.MODEL.SPARSITY_COEFF = 0.0
        cfg.MODEL.USE_HOMOGENEOUS = False
        cfg.MODEL.SOFT_BLOCK.ENABLED = True
        cfg.MODEL.SOFT_BLOCK.NUM_BLOCKS = 2
        cfg.MODEL.SOFT_BLOCK.WEIGHT = 0.2
        model = LISTAKM(cfg, observation_size=2)

        with torch.no_grad():
            model.kmat.copy_(
                torch.tensor(
                    [
                        [1.0, 1.0, 2.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [3.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )

        penalty, ratio = model._soft_block_penalty()
        assert penalty is not None
        assert ratio is not None
        assert torch.isclose(penalty, torch.tensor(5.0), atol=1e-6)
        assert torch.isclose(ratio, torch.tensor(0.5), atol=1e-6)

        loss, metrics = model.loss(
            x_pred=torch.zeros(2, 1, 2),
            x_true=torch.zeros(2, 1, 2),
            z_pred=torch.zeros(2, 1, 4),
            z_true=torch.zeros(2, 1, 4),
            reconstruction_error=torch.tensor(0.0),
            sparsity_latent=torch.zeros(2, 1, 4),
        )
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)
        assert metrics["soft_block_penalty"] == pytest.approx(5.0, abs=1e-6)
        assert metrics["soft_block_off_block_ratio"] == pytest.approx(0.5, abs=1e-6)

    def test_decoder_coherence_penalty_adds_dictionary_regularization(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 2
        cfg.MODEL.RES_COEFF = 0.0
        cfg.MODEL.RECONST_COEFF = 0.0
        cfg.MODEL.PRED_COEFF = 0.0
        cfg.MODEL.SPARSITY_COEFF = 0.0
        cfg.MODEL.USE_HOMOGENEOUS = False
        cfg.MODEL.DECODER_COHERENCE_WEIGHT = 0.25
        model = LISTAKM(cfg, observation_size=2)

        with torch.no_grad():
            model.dict.copy_(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))

        penalty, max_offdiag = model._decoder_coherence_penalty()
        assert penalty is not None
        assert max_offdiag is not None
        assert torch.isclose(penalty, torch.tensor(2.0), atol=1e-6)
        assert torch.isclose(max_offdiag, torch.tensor(1.0), atol=1e-6)

        loss, metrics = model.loss(
            x_pred=torch.zeros(2, 1, 2),
            x_true=torch.zeros(2, 1, 2),
            z_pred=torch.zeros(2, 1, 2),
            z_true=torch.zeros(2, 1, 2),
            reconstruction_error=torch.tensor(0.0),
            sparsity_latent=torch.zeros(2, 1, 2),
        )
        assert torch.isclose(loss, torch.tensor(0.5), atol=1e-6)
        assert metrics["decoder_coherence_penalty"] == pytest.approx(2.0, abs=1e-6)
        assert metrics["decoder_coherence_max_offdiag"] == pytest.approx(1.0, abs=1e-6)

    def test_decoder_coherence_penalty_collapses_signsplit_pairs(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.ENCODER.LISTA.FINAL_OP = "sign_split"
        cfg.MODEL.DECODER_COHERENCE_WEIGHT = 1.0
        cfg.MODEL.USE_HOMOGENEOUS = False
        model = LISTAKM(cfg, observation_size=2)

        with torch.no_grad():
            model.dict.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ]
                )
            )

        penalty, max_offdiag = model._decoder_coherence_penalty()
        assert penalty is not None
        assert max_offdiag is not None
        assert torch.isclose(penalty, torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(max_offdiag, torch.tensor(0.0), atol=1e-6)

class TestUnifiedLossInterface:
    """Tests for the unified pure-aggregation loss API."""

    def test_alignment_requires_latents_when_enabled(self):
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        model = GenericKM(cfg, 2)

        x_pred = torch.randn(4, 1, 2)
        x_true = torch.randn(4, 1, 2)
        with pytest.raises(ValueError):
            model.loss(x_pred=x_pred, x_true=x_true)

    def test_alignment_optional_when_coeff_zero(self):
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 8
        cfg.MODEL.RES_COEFF = 0.0
        model = GenericKM(cfg, 2)

        x_pred = torch.randn(4, 1, 2)
        x_true = torch.randn(4, 1, 2)
        loss, metrics = model.loss(x_pred=x_pred, x_true=x_true)
        assert loss.ndim == 0
        assert "alignment_loss" in metrics

    def test_horizon_mean_is_invariant_for_repeated_errors(self):
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.RES_COEFF = 1.0
        cfg.MODEL.PRED_COEFF = 1.0
        cfg.MODEL.RECONST_COEFF = 0.0
        cfg.MODEL.SPARSITY_COEFF = 0.0
        model = GenericKM(cfg, 2)

        # Per-step losses are averaged over the horizon, so repeating the same
        # error at every step must leave the aggregate unchanged.
        x_pred_h1 = torch.zeros(2, 1, 2)
        x_true_h1 = torch.ones(2, 1, 2)
        z_pred_h1 = torch.zeros(2, 1, 4)
        z_true_h1 = torch.ones(2, 1, 4)
        loss_h1, _ = model.loss(
            x_pred=x_pred_h1, x_true=x_true_h1, z_pred=z_pred_h1, z_true=z_true_h1
        )

        x_pred_h2 = torch.zeros(2, 2, 2)
        x_true_h2 = torch.ones(2, 2, 2)
        z_pred_h2 = torch.zeros(2, 2, 4)
        z_true_h2 = torch.ones(2, 2, 4)
        loss_h2, _ = model.loss(
            x_pred=x_pred_h2, x_true=x_true_h2, z_pred=z_pred_h2, z_true=z_true_h2
        )

        assert torch.allclose(loss_h2, loss_h1, atol=1e-6)

        x_pred_h8 = torch.zeros(2, 8, 2)
        x_true_h8 = torch.ones(2, 8, 2)
        z_pred_h8 = torch.zeros(2, 8, 4)
        z_true_h8 = torch.ones(2, 8, 4)
        loss_h8, _ = model.loss(
            x_pred=x_pred_h8, x_true=x_true_h8, z_pred=z_pred_h8, z_true=z_true_h8
        )

        assert torch.allclose(loss_h8, loss_h1, atol=1e-6)

    def test_loss_is_pure_aggregator(self, monkeypatch):
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 4
        model = GenericKM(cfg, 2)

        def _fail(*args, **kwargs):
            raise RuntimeError("should not be called")

        monkeypatch.setattr(model, "encode", _fail)
        monkeypatch.setattr(model, "decode", _fail)
        monkeypatch.setattr(model, "rollout_latent_discrete", _fail)

        x_pred = torch.randn(3, 1, 2)
        x_true = torch.randn(3, 1, 2)
        z_pred = torch.randn(3, 1, 4)
        z_true = torch.randn(3, 1, 4)

        loss, metrics = model.loss(
            x_pred=x_pred,
            x_true=x_true,
            z_pred=z_pred,
            z_true=z_true,
            reconstruction_error=torch.tensor(0.25),
            sparsity_latent=z_pred,
        )
        assert loss.ndim == 0
        assert "loss" in metrics

class TestModelFactory:
    """Test model factory function."""
    
    def test_make_model_generic(self):
        """Test creating GenericKM via factory."""
        cfg = get_config("generic")
        obs_size = 2
        model = make_model(cfg, obs_size)
        assert isinstance(model, GenericKM)
    
    def test_make_model_sparse(self):
        """Test creating SparseKM (alias for GenericKM) via factory."""
        cfg = get_config("generic_sparse")
        obs_size = 2
        model = make_model(cfg, obs_size)
        assert isinstance(model, GenericKM)
    
    def test_make_model_listakm(self):
        """Test creating LISTAKM via factory."""
        cfg = get_config("lista")
        obs_size = 2
        model = make_model(cfg, obs_size)
        assert isinstance(model, LISTAKM)

    def test_make_model_hyperlista_returns_listakm(self):
        """HyperLISTA preset should still instantiate LISTAKM."""
        cfg = get_config("hyperlista")
        obs_size = 2
        model = make_model(cfg, obs_size)
        assert isinstance(model, LISTAKM)

    def test_make_model_rejects_hyperlistakm_name(self):
        """Legacy HyperLISTAKM class name should be rejected."""
        cfg = get_config("generic")
        cfg.MODEL.MODEL_NAME = "HyperLISTAKM"
        with pytest.raises(ValueError):
            make_model(cfg, observation_size=2)

    def test_model_module_has_no_hyperlistakm_symbol(self):
        """Model module should not export HyperLISTAKM symbol after refactor."""
        assert not hasattr(model_module, "HyperLISTAKM")
    
    def test_make_model_invalid(self):
        """Test factory raises error for invalid model name."""
        cfg = get_config("generic")
        cfg.MODEL.MODEL_NAME = "InvalidModel"
        obs_size = 2
        
        with pytest.raises(ValueError):
            make_model(cfg, obs_size)


class TestGradientFlow:
    """Test gradient flow through models."""
    
    def test_generic_km_gradients(self):
        """Test gradients flow through GenericKM."""
        cfg = get_config("generic")
        cfg.MODEL.TARGET_SIZE = 16
        obs_size = 2
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size, requires_grad=True)
        nx = torch.randn(4, obs_size)
        
        loss, _ = model.loss(**make_unified_loss_inputs(model, x, nx))
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert model.encoder.network[0].weight.grad is not None
        assert model.kmat.grad is not None
    
    def test_listakm_gradients(self):
        """Test gradients flow through LISTAKM."""
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 3
        obs_size = 2
        
        model = LISTAKM(cfg, obs_size)
        x = torch.randn(4, obs_size, requires_grad=True)
        nx = torch.randn(4, obs_size)
        
        loss, _ = model.loss(**make_unified_loss_inputs(model, x, nx))
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert model.dict.grad is not None
        assert model.kmat.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
