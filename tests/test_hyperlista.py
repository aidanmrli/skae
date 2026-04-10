"""Unit tests for HyperLISTA encoder and unified LISTAKM HyperLISTA mode."""

import torch
import pytest
from skae.config import get_config
from skae.model import HyperLISTA, LISTA, LISTAKM


def make_unified_loss_inputs(model, x: torch.Tensor, nx: torch.Tensor):
    """Build minimal unified-loss tensors for a 1-step horizon."""
    x_pred = nx.unsqueeze(1)
    x_true = nx.unsqueeze(1)
    z0 = model.encode(x)
    z_true = model.encode(nx).unsqueeze(1)
    z_pred = model.rollout_latent_discrete(z0, horizon=1)
    x_recon_true = model.decode(z_true.reshape(-1, z_true.shape[-1])).reshape_as(x_true)
    homogeneous_loss = None
    if getattr(model, "use_homogeneous", False):
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


class TestHyperLISTA:
    """Test HyperLISTA encoder module."""

    def test_initialization(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        xdim = 10
        dict_param = torch.nn.Parameter(torch.randn(cfg.MODEL.TARGET_SIZE, xdim))

        hyperlista = HyperLISTA(cfg, xdim, dict_param)
        assert hyperlista.xdim == xdim
        assert hyperlista.zdim == cfg.MODEL.TARGET_SIZE

    def test_forward_shape(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 3

        xdim = 10
        dict_param = torch.nn.Parameter(torch.randn(cfg.MODEL.TARGET_SIZE, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)

        x = torch.randn(16, xdim)
        z = hyperlista(x)
        assert z.shape == (16, cfg.MODEL.TARGET_SIZE)

    def test_sparsity(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 128
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 5
        cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = 0.01

        xdim = 10
        dict_param = torch.nn.Parameter(torch.randn(cfg.MODEL.TARGET_SIZE, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)

        x = torch.randn(8, xdim)
        z = hyperlista(x)
        nonzero_ratio = (z.abs() > 1e-6).float().mean().item()
        assert nonzero_ratio < 1.0

    def test_learnable_hyperparams(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = True

        xdim = 5
        dict_param = torch.nn.Parameter(torch.randn(32, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)

        assert isinstance(hyperlista.c_theta_raw, torch.nn.Parameter)
        assert isinstance(hyperlista.c_beta, torch.nn.Parameter)
        assert isinstance(hyperlista.c_ss, torch.nn.Parameter)
        assert hyperlista.c_theta.item() > 0.0

    def test_fixed_hyperparams(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = False

        xdim = 5
        dict_param = torch.nn.Parameter(torch.randn(32, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)

        param_names = [n for n, _ in hyperlista.named_parameters()]
        assert "c_theta_raw" not in param_names
        assert "c_beta" not in param_names
        assert "c_ss" not in param_names

    def test_c_theta_stays_positive_when_constrained(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = 5e-4
        cfg.MODEL.ENCODER.HYPERLISTA.C_THETA_MIN = 1e-6
        cfg.MODEL.ENCODER.HYPERLISTA.CONSTRAIN_C_THETA = True

        dict_param = torch.nn.Parameter(torch.randn(32, 5))
        hyperlista = HyperLISTA(cfg, 5, dict_param)
        optimizer = torch.optim.SGD(hyperlista.parameters(), lr=10.0)

        loss = -hyperlista.c_theta.sum()
        loss.backward()
        optimizer.step()

        assert hyperlista.c_theta.item() >= cfg.MODEL.ENCODER.HYPERLISTA.C_THETA_MIN

    def test_no_momentum(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.HYPERLISTA.USE_MOMENTUM = False

        xdim = 5
        dict_param = torch.nn.Parameter(torch.randn(32, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)

        z = hyperlista(torch.randn(4, xdim))
        assert z.shape == (4, 32)

    def test_no_support_selection(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 32
        cfg.MODEL.ENCODER.HYPERLISTA.USE_SUPPORT_SELECTION = False

        xdim = 5
        dict_param = torch.nn.Parameter(torch.randn(32, xdim))
        hyperlista = HyperLISTA(cfg, xdim, dict_param)

        z = hyperlista(torch.randn(4, xdim))
        assert z.shape == (4, 32)


class TestUnifiedHyperLISTAModel:
    """Test unified LISTAKM behavior in hyperlista mode."""

    def _build_model(self, *, obs_size: int = 3, homogeneous: bool = False) -> LISTAKM:
        cfg = get_config("hyperlista")
        cfg.MODEL.MODEL_NAME = "LISTAKM"
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 3
        cfg.MODEL.ENCODER.ENCODER_TYPE = "hyperlista"
        cfg.MODEL.USE_HOMOGENEOUS = homogeneous
        return LISTAKM(cfg, obs_size)

    def test_unified_hyperlista_encode_decode_shapes(self):
        model = self._build_model(obs_size=3)
        x = torch.randn(8, 3)

        z = model.encode(x)
        x_recon = model.decode(z)

        assert z.shape == (8, model.target_size)
        assert x_recon.shape == (8, 3)

    def test_unified_hyperlista_loss_runs(self):
        model = self._build_model(obs_size=2)
        x = torch.randn(8, 2)
        nx = torch.randn(8, 2)

        loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))

        assert loss.ndim == 0
        assert "loss" in metrics
        assert "residual_loss" in metrics
        assert "reconst_loss" in metrics
        assert "sparsity_loss" in metrics

    def test_unified_hyperlista_homogeneous_mode_runs(self):
        model = self._build_model(obs_size=2, homogeneous=True)
        x = torch.randn(4, 2)
        nx = torch.randn(4, 2)

        loss, metrics = model.loss(**make_unified_loss_inputs(model, x, nx))

        assert loss.ndim == 0
        assert "homogeneous_loss" in metrics


class TestUnifiedHyperLISTAGradientFlow:
    """Gradient tests for unified LISTAKM encoder dispatch."""

    def test_encode_only_hyperlista_backprops_to_dict(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.ENCODER_TYPE = "hyperlista"

        model = LISTAKM(cfg, observation_size=3)
        x = torch.randn(8, 3)

        model.zero_grad(set_to_none=True)
        model.encode(x).sum().backward()

        assert model.dict.grad is not None
        assert torch.any(model.dict.grad.abs() > 0)

    def test_encode_only_lista_does_not_backprop_to_dict(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.ENCODER_TYPE = "lista"

        model = LISTAKM(cfg, observation_size=3)
        x = torch.randn(8, 3)

        model.zero_grad(set_to_none=True)
        model.encode(x).sum().backward()

        if model.dict.grad is None:
            assert True
        else:
            assert torch.allclose(model.dict.grad, torch.zeros_like(model.dict.grad))

    def test_learnable_hyperparams_gradient_uses_encoder_field(self):
        cfg = get_config("lista")
        cfg.MODEL.TARGET_SIZE = 64
        cfg.MODEL.ENCODER.ENCODER_TYPE = "hyperlista"
        cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 3
        cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = True

        model = LISTAKM(cfg, observation_size=3)
        assert isinstance(model.encoder, HyperLISTA)

        x = torch.randn(8, 3)
        nx = torch.randn(8, 3)

        loss, _ = model.loss(**make_unified_loss_inputs(model, x, nx))
        loss.backward()

        assert model.encoder.c_theta_raw.grad is not None
        assert model.encoder.c_beta.grad is not None


class TestHyperLISTAShrinkOperators:
    """Test soft-thresholding operators."""

    def test_shrink_basic(self):
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
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 8

        dict_param = torch.nn.Parameter(torch.randn(8, 3))
        hyperlista = HyperLISTA(cfg, 3, dict_param)

        x = torch.tensor([[10.0, 5.0, 0.1, 0.05, -10.0, -5.0, -0.1, -0.05]])
        theta = torch.tensor([[1.0]])
        p = torch.tensor([[0.5]])

        result = hyperlista._shrink_ss(x, theta, p)

        assert result[0, 0] == 10.0
        assert result[0, 1] == 5.0
        assert result[0, 4] == -10.0
        assert result[0, 5] == -5.0
        assert result[0, 2] == 0.0
        assert result[0, 3] == 0.0

    def test_group_structure_masks_to_topk_groups(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.K_STRUCTURE = "block_diagonal"
        cfg.MODEL.K_NUM_BLOCKS = 2
        cfg.MODEL.ENCODER.HYPERLISTA.GROUP_SHRINKAGE = True
        cfg.MODEL.ENCODER.HYPERLISTA.GROUP_THRESHOLD_SCALE = 1.0
        cfg.MODEL.ENCODER.HYPERLISTA.TOPK_GROUPS = 1

        dict_param = torch.nn.Parameter(torch.randn(4, 3))
        hyperlista = HyperLISTA(cfg, 3, dict_param)

        x = torch.tensor([[3.0, 3.0, 2.0, 2.0]])
        theta = torch.tensor([[1.0]])
        result = hyperlista._apply_group_structure(x, theta)

        assert torch.all(result[..., 2:] == 0.0)
        assert torch.all(result[..., :2] > 0.0)

    def test_group_aware_hyperlista_requires_group_structure(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 4
        cfg.MODEL.K_STRUCTURE = "dense"
        cfg.MODEL.ENCODER.HYPERLISTA.GROUP_SHRINKAGE = True

        dict_param = torch.nn.Parameter(torch.randn(4, 3))
        with pytest.raises(ValueError, match="requires structured, block-diagonal, or soft-block"):
            HyperLISTA(cfg, 3, dict_param)

    def test_pinv_refresh_uses_current_dictionary_values(self):
        cfg = get_config("hyperlista")
        cfg.MODEL.TARGET_SIZE = 4

        dict_param = torch.nn.Parameter(torch.randn(4, 3))
        hyperlista = HyperLISTA(cfg, 3, dict_param)

        D = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
            ]
        )
        pinv_1 = hyperlista._get_D_pinv(D)

        D.mul_(2.0)
        pinv_2 = hyperlista._get_D_pinv(D)

        assert not torch.allclose(pinv_1, pinv_2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
