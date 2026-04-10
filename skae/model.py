"""
PyTorch implementation of Koopman Autoencoder models.

This module provides:
- MLPCoder: Multi-layer perceptron for encoding/decoding
- LISTA: Learned Iterative Soft-Thresholding Algorithm for sparse coding
- KoopmanMachine: Abstract base class for Koopman operator learning
- GenericKM: Standard Koopman autoencoder with MLP encoder
- LISTAKM: Koopman machine with pluggable LISTA-family sparse encoder
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from skae.config import Config


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def shrink(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft thresholding operator (shrinkage). Used in LISTA.
    
    Args:
        x: Input tensor
        threshold: Threshold value for soft thresholding
        
    Returns:
        Shrunk tensor
    """
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.
    
    Args:
        name: Activation name ('relu', 'tanh', 'gelu')
        
    Returns:
        Activation module
    """
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(activations.keys())}")
    return activations[name]


def _split_group_sizes(zdim: int, num_groups: int) -> List[int]:
    """Split a latent dimension into near-equal positive group sizes."""

    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_groups > zdim:
        raise ValueError(f"num_groups={num_groups} exceeds latent size zdim={zdim}")
    base = zdim // num_groups
    remainder = zdim % num_groups
    return [base + (1 if group_index < remainder else 0) for group_index in range(num_groups)]


def _infer_encoder_group_slices(cfg: Config, zdim: int) -> Tuple[slice, ...]:
    """Infer latent group slices for group-aware encoder shrinkage."""

    sizes: List[int] = []
    if cfg.MODEL.STRUCTURED.ENABLED:
        d_global = int(cfg.MODEL.STRUCTURED.D_GLOBAL)
        num_basins = int(cfg.MODEL.STRUCTURED.NUM_BASINS)
        d_basin = int(cfg.MODEL.STRUCTURED.D_BASIN)
        if d_global > 0:
            sizes.append(d_global)
        sizes.extend([d_basin] * num_basins)
    elif cfg.MODEL.K_STRUCTURE == "block_diagonal":
        requested_num_blocks = int(getattr(cfg.MODEL, "K_NUM_BLOCKS", 0))
        if requested_num_blocks > 0:
            sizes = _split_group_sizes(zdim, requested_num_blocks)
        else:
            block_size = int(getattr(cfg.MODEL, "K_BLOCK_SIZE", 0))
            if block_size <= 0:
                block_size = max(1, zdim // 13)
            full_blocks = zdim // block_size
            remainder = zdim - full_blocks * block_size
            sizes = [block_size] * full_blocks
            if remainder > 0:
                sizes.append(remainder)
    elif cfg.MODEL.SOFT_BLOCK.ENABLED and int(cfg.MODEL.SOFT_BLOCK.NUM_BLOCKS) > 0:
        sizes = _split_group_sizes(zdim, int(cfg.MODEL.SOFT_BLOCK.NUM_BLOCKS))

    if sum(sizes) != zdim:
        return tuple()

    offset = 0
    group_slices: List[slice] = []
    for group_size in sizes:
        group_slices.append(slice(offset, offset + group_size))
        offset += group_size
    return tuple(group_slices)


def _group_norms_from_slices(x: torch.Tensor, group_slices: Tuple[slice, ...]) -> torch.Tensor:
    """Compute per-group L2 norms for the provided latent slices."""

    return torch.cat(
        [torch.norm(x[..., group_slice], p=2, dim=-1, keepdim=True) for group_slice in group_slices],
        dim=-1,
    )


def _mask_to_topk_groups(
    x: torch.Tensor,
    group_slices: Tuple[slice, ...],
    topk_groups: int,
) -> torch.Tensor:
    """Zero out all but the top-k groups (by group L2 norm) per sample."""

    if topk_groups <= 0 or not group_slices:
        return x

    group_norms = _group_norms_from_slices(x, group_slices)
    flat_norms = group_norms.reshape(-1, group_norms.shape[-1])
    k = min(int(topk_groups), flat_norms.shape[-1])
    if k <= 0:
        return x

    topk_idx = torch.topk(flat_norms, k=k, dim=-1, largest=True, sorted=False).indices
    keep_mask = torch.zeros_like(flat_norms, dtype=torch.bool)
    keep_mask.scatter_(1, topk_idx, True)
    keep_mask = keep_mask.view(*group_norms.shape[:-1], group_norms.shape[-1])

    masked_parts: List[torch.Tensor] = []
    for group_index, group_slice in enumerate(group_slices):
        group_keep = keep_mask[..., group_index].unsqueeze(-1)
        group_values = x[..., group_slice]
        masked_parts.append(torch.where(group_keep, group_values, torch.zeros_like(group_values)))
    return torch.cat(masked_parts, dim=-1)


def _sparse_group_shrink(
    x: torch.Tensor,
    theta_group: torch.Tensor | float,
    group_slices: Tuple[slice, ...],
) -> torch.Tensor:
    """Apply group-lasso shrinkage independently on inferred latent groups."""

    if not group_slices:
        return x

    shrunk_parts: List[torch.Tensor] = []
    for group_slice in group_slices:
        group_values = x[..., group_slice]
        group_norm = torch.norm(group_values, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        shrink_factor = torch.relu(1.0 - theta_group / group_norm)
        shrunk_parts.append(shrink_factor * group_values)
    return torch.cat(shrunk_parts, dim=-1)


def _expand_group_thresholds(
    group_thresholds: torch.Tensor,
    group_slices: Tuple[slice, ...],
) -> torch.Tensor:
    """Expand per-group thresholds to one threshold per latent coordinate."""

    expanded_parts: List[torch.Tensor] = []
    for group_index, group_slice in enumerate(group_slices):
        group_threshold = group_thresholds[..., group_index:group_index + 1]
        group_width = group_slice.stop - group_slice.start
        expanded_parts.append(group_threshold.expand(*group_threshold.shape[:-1], group_width))
    return torch.cat(expanded_parts, dim=-1)


def _reduce_thresholds_to_groups(
    thresholds: torch.Tensor | float,
    group_slices: Tuple[slice, ...],
    *,
    reference: torch.Tensor,
) -> torch.Tensor | float:
    """Reduce elementwise thresholds to one scalar per inferred group."""

    if isinstance(thresholds, float):
        return thresholds
    if thresholds.ndim == 0 or thresholds.shape[-1] == 1:
        return thresholds

    group_thresholds: List[torch.Tensor] = []
    for group_slice in group_slices:
        group_thresholds.append(thresholds[..., group_slice].mean(dim=-1, keepdim=True))
    reduced = torch.cat(group_thresholds, dim=-1)
    return reduced.to(device=reference.device, dtype=reference.dtype)


# ---------------------------------------------------------------------------
# Network Components
# ---------------------------------------------------------------------------


class MLPCoder(nn.Module):
    """Multi-layer perceptron for encoding or decoding.
    
    Args:
        input_size: Input dimension
        target_size: Output dimension
        hidden_layers: List of hidden layer sizes
        last_relu: Whether to apply ReLU to the output
        use_bias: Whether to use bias in linear layers
        activation: Activation function name
    """
    
    def __init__(
        self,
        input_size: int,
        target_size: int,
        hidden_layers: List[int],
        last_relu: bool = False,
        use_bias: bool = False,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.hidden_layers = hidden_layers
        self.last_relu = last_relu
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size, bias=use_bias))
            layers.append(get_activation(activation))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, target_size, bias=use_bias))
        if last_relu:
            layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [..., input_size]
            
        Returns:
            Output tensor of shape [..., target_size]
        """
        return self.network(x)


class LISTA(nn.Module):
    """Learned Iterative Soft-Thresholding Algorithm (LISTA) encoder.
    
    This module implements a LISTA-style encoder: an unrolled, fixed-depth
    approximation to sparse coding built from alternating affine transforms
    and an elementwise soft-thresholding nonlinearity.

    Canonical LISTA (Gregor & LeCun, 2010) uses a linear pre-activation
    z-affine map W_e x and shared "mutual-inhibition" matrix S, with the
    nonlinearity given by the soft-thresholding (shrinkage) operator
    T_λ(v)_i = sign(v_i) * max(|v_i| - λ, 0). The overall encoder is
    therefore nonlinear due to T_λ.
    
    Shapes (standard convention):
        x ∈ ℝ^{xdim},  z ∈ ℝ^{zdim}
        Dictionary W_d ∈ ℝ^{xdim × zdim}  (columns are atoms)
        Linear encoder W_e = (1/L) W_dᵀ ∈ ℝ^{zdim × xdim}
        Inhibition S = I - (1/L) W_dᵀ W_d ∈ ℝ^{zdim × zdim}

    Iterations:
        c = W_e x
        z^(0) = T_{α/L}(c)
        for k = 0..K-1:
            z^(k+1) = T_{α/L}(S z^(k) + c)
        return z^(K)

    Notes:
        • If `use_linear_encode=True`, the module uses the canonical linear
          pre-activation W_e x. If `False`, an MLP can be used to produce c;
          this yields a LISTA-style unrolled network rather than canonical LISTA.
        • L is a Lipschitz constant estimate (e.g., ≥ spectral norm of W_dᵀ W_d).
        • α controls sparsity; K is the number of unrolled iterations.

    Args:
        cfg: Configuration object.
        xdim: Input dimension.
        Wd_init: Initial dictionary matrix with shape [xdim, zdim].
    """
    
    def __init__(
        self,
        cfg: Config,
        xdim: int,
        Wd_init: torch.Tensor,
        L_override: Optional[float] = None,
        dict_param: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.xdim = xdim
        self.zdim = cfg.MODEL.TARGET_SIZE
        self.num_loops = cfg.MODEL.ENCODER.LISTA.NUM_LOOPS
        self.use_momentum = bool(cfg.MODEL.ENCODER.LISTA.USE_MOMENTUM)
        self.momentum_beta = float(cfg.MODEL.ENCODER.LISTA.MOMENTUM_BETA)
        self.alpha = cfg.MODEL.ENCODER.LISTA.ALPHA
        self.L = L_override if L_override is not None else cfg.MODEL.ENCODER.LISTA.L
        self._legacy_linear_encoder = bool(cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER)
        self.precode_mode = str(cfg.MODEL.ENCODER.LISTA.PRECODE_MODE).lower()
        if self.precode_mode not in {"auto", "free_mlp", "linear", "dictionary_tied", "hybrid"}:
            raise ValueError(
                f"Unknown LISTA PRECODE_MODE '{self.precode_mode}'. "
                "Expected one of ['auto', 'free_mlp', 'linear', 'dictionary_tied', 'hybrid']."
            )
        if self.precode_mode == "auto":
            self.precode_mode = "linear" if self._legacy_linear_encoder else "free_mlp"
        self.precode_residual_scale = float(cfg.MODEL.ENCODER.LISTA.PRECODE_RESIDUAL_SCALE)
        self.use_adaptive_thresholds = bool(cfg.MODEL.ENCODER.LISTA.ADAPTIVE_THRESHOLDS)
        self.alpha_residual_coeff = float(cfg.MODEL.ENCODER.LISTA.ALPHA_RESIDUAL_COEFF)
        self.alpha_prior_coeff = float(cfg.MODEL.ENCODER.LISTA.ALPHA_PRIOR_COEFF)
        self.use_groupwise_thresholds = bool(cfg.MODEL.ENCODER.LISTA.GROUPWISE_THRESHOLDS)
        requested_final_op = cfg.MODEL.ENCODER.LISTA.FINAL_OP.lower()
        if requested_final_op not in {"shrink", "relu", "sign_split"}:
            raise ValueError(
                f"Unknown LISTA FINAL_OP '{requested_final_op}'. "
                "Expected one of ['shrink', 'relu', 'sign_split']."
            )
        self.final_op = requested_final_op
        self.base_zdim = self.zdim
        if self.final_op == "sign_split":
            if self.zdim % 2 != 0:
                raise ValueError(
                    "LISTA FINAL_OP='sign_split' requires an even TARGET_SIZE "
                    f"(got {self.zdim})."
                )
            self.base_zdim = self.zdim // 2

        self.use_group_shrinkage = bool(cfg.MODEL.ENCODER.LISTA.GROUP_SHRINKAGE)
        self.group_threshold_scale = float(cfg.MODEL.ENCODER.LISTA.GROUP_THRESHOLD_SCALE)
        self.topk_groups = int(cfg.MODEL.ENCODER.LISTA.TOPK_GROUPS)
        if self.final_op == "sign_split" and (self.use_group_shrinkage or self.topk_groups > 0):
            raise ValueError("LISTA group-aware shrinkage and top-k groups do not support FINAL_OP='sign_split'.")
        if self.final_op == "sign_split" and self.precode_mode in {"dictionary_tied", "hybrid"}:
            raise ValueError("LISTA dictionary-tied and hybrid pre-code modes do not support FINAL_OP='sign_split'.")
        if self.final_op == "sign_split" and self.use_adaptive_thresholds:
            raise ValueError("LISTA adaptive thresholds do not support FINAL_OP='sign_split'.")
        self.group_slices = _infer_encoder_group_slices(cfg, self.base_zdim)
        if (self.use_group_shrinkage or self.topk_groups > 0) and not self.group_slices:
            raise ValueError(
                "LISTA group-aware shrinkage requires structured, block-diagonal, or soft-block latent groups."
            )
        if self.use_groupwise_thresholds and not self.group_slices:
            raise ValueError(
                "LISTA groupwise thresholds require structured, block-diagonal, or soft-block latent groups."
            )
        if self.use_groupwise_thresholds and self.final_op == "sign_split":
            raise ValueError("LISTA groupwise thresholds do not support FINAL_OP='sign_split'.")
        self.dict_param = dict_param
        
        assert Wd_init.shape == (xdim, self.base_zdim), \
            f"Wd_init shape {Wd_init.shape} doesn't match expected ({xdim}, {self.base_zdim})"

        self.register_buffer("encoder_dict_init", Wd_init.clone())
        self.precode_module: Optional[nn.Module]
        self.precode_residual: Optional[nn.Module] = None

        if self.precode_mode == "linear":
            use_bias = cfg.MODEL.ENCODER.USE_BIAS
            self.precode_module = nn.Linear(xdim, self.base_zdim, bias=use_bias)
            with torch.no_grad():
                self.precode_module.weight.copy_((1.0 / self.L) * Wd_init.T)
                if use_bias:
                    self.precode_module.bias.zero_()
        elif self.precode_mode == "free_mlp":
            self.precode_module = MLPCoder(
                input_size=xdim,
                target_size=self.base_zdim,
                hidden_layers=cfg.MODEL.ENCODER.LAYERS,
                use_bias=cfg.MODEL.ENCODER.USE_BIAS,
                last_relu=cfg.MODEL.ENCODER.LAST_RELU,
                activation=cfg.MODEL.ENCODER.ACTIVATION,
            )
        elif self.precode_mode == "dictionary_tied":
            self.precode_module = None
        elif self.precode_mode == "hybrid":
            self.precode_module = None
            self.precode_residual = MLPCoder(
                input_size=xdim,
                target_size=self.base_zdim,
                hidden_layers=cfg.MODEL.ENCODER.LAYERS,
                use_bias=cfg.MODEL.ENCODER.USE_BIAS,
                last_relu=False,
                activation=cfg.MODEL.ENCODER.ACTIVATION,
            )
            final_linear = None
            for module in reversed(self.precode_residual.network):
                if isinstance(module, nn.Linear):
                    final_linear = module
                    break
            if final_linear is not None:
                with torch.no_grad():
                    final_linear.weight.zero_()
                    if final_linear.bias is not None:
                        final_linear.bias.zero_()
        else:
            raise RuntimeError(f"Unhandled LISTA precode mode '{self.precode_mode}'.")

        S_init = torch.eye(self.base_zdim) - (1.0 / self.L) * (Wd_init.T @ Wd_init)
        self.S = nn.Parameter(S_init)
        if self.use_groupwise_thresholds:
            self.group_alpha = nn.Parameter(torch.full((len(self.group_slices),), float(self.alpha)))
            group_lipschitz = []
            for group_slice in self.group_slices:
                group_dict = Wd_init[:, group_slice]
                gram = group_dict.T @ group_dict
                eigvals = torch.linalg.eigvalsh(gram)
                group_lipschitz.append(float(eigvals.max().item()) * 1.05)
            self.register_buffer(
                "group_lipschitz",
                torch.tensor(group_lipschitz, dtype=Wd_init.dtype).clamp(min=1e-8),
            )
        if self.use_group_shrinkage or self.topk_groups > 0:
            print(
                "  LISTA group-aware encoder:"
                f" groups={len(self.group_slices)},"
                f" group_shrinkage={self.use_group_shrinkage},"
                f" threshold_scale={self.group_threshold_scale:.3f},"
                f" topk_groups={self.topk_groups}"
            )
        if self.precode_mode in {"dictionary_tied", "hybrid"}:
            print(
                "  LISTA pre-code:"
                f" mode={self.precode_mode}, residual_scale={self.precode_residual_scale:.3f}"
            )
        if self.use_adaptive_thresholds or self.use_groupwise_thresholds:
            print(
                "  LISTA thresholds:"
                f" adaptive={self.use_adaptive_thresholds},"
                f" groupwise={self.use_groupwise_thresholds},"
                f" alpha_residual_coeff={self.alpha_residual_coeff:.3f},"
                f" alpha_prior_coeff={self.alpha_prior_coeff:.3f}"
            )
        if self.use_momentum:
            print(f"  LISTA momentum: beta={self.momentum_beta:.3f}")

    def _split_sign(self, x: torch.Tensor) -> torch.Tensor:
        """Map signed coefficients to nonnegative sign-split coordinates."""
        return torch.cat([F.relu(x), F.relu(-x)], dim=-1)

    def _unsplit_sign(self, x: torch.Tensor) -> torch.Tensor:
        """Recover signed coefficients from sign-split coordinates."""
        if x.shape[-1] != self.zdim:
            raise ValueError(
                "sign-split latent_prior must match the output latent shape: "
                f"got {x.shape[-1]}, expected {self.zdim}"
            )
        positive, negative = torch.chunk(x, 2, dim=-1)
        return positive - negative

    def _current_dictionary(self) -> torch.Tensor:
        """Return the current normalized encoder dictionary."""

        if self.dict_param is not None:
            dict_matrix = self.dict_param.T
            if dict_matrix.shape[1] != self.base_zdim:
                raise ValueError(
                    "LISTA dictionary-tied pre-code requires decoder dictionary width "
                    f"{self.base_zdim}, got {dict_matrix.shape[1]}"
                )
        else:
            dict_matrix = self.encoder_dict_init
        return dict_matrix / torch.norm(dict_matrix, dim=0, keepdim=True).clamp(min=1e-6)

    def _compute_precode(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the encoder pre-code according to the configured mode."""

        if self.precode_mode == "linear":
            if self.precode_module is None:
                raise RuntimeError("LISTA linear pre-code module is missing")
            return self.precode_module(x)
        if self.precode_mode == "free_mlp":
            if self.precode_module is None:
                raise RuntimeError("LISTA MLP pre-code module is missing")
            return self.precode_module(x)

        tied_precode = x @ (self._current_dictionary() / self.L)
        if self.precode_mode == "dictionary_tied":
            return tied_precode
        if self.precode_mode == "hybrid":
            if self.precode_residual is None:
                raise RuntimeError("LISTA hybrid residual pre-code module is missing")
            return tied_precode + self.precode_residual_scale * self.precode_residual(x)
        raise RuntimeError(f"Unhandled LISTA pre-code mode '{self.precode_mode}'")

    def _base_thresholds(
        self,
        *,
        batch_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the base per-sample threshold tensor before adaptive terms."""

        if self.use_groupwise_thresholds:
            group_alpha = torch.relu(self.group_alpha).to(device=device, dtype=dtype)
            group_thresholds = group_alpha / self.group_lipschitz.to(device=device, dtype=dtype)
            group_thresholds = group_thresholds.view(*([1] * len(batch_shape)), -1)
            return _expand_group_thresholds(group_thresholds, self.group_slices)
        return torch.full((*batch_shape, 1), float(self.alpha / self.L), device=device, dtype=dtype)

    def _decode_from_dictionary(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct observations from the current normalized encoder dictionary."""

        return z @ self._current_dictionary().T

    def _compute_thresholds(
        self,
        *,
        x: torch.Tensor,
        nonsparse_code: torch.Tensor,
        latent_prior_internal: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-sample thresholds, optionally with adaptive terms."""

        thresholds = self._base_thresholds(
            batch_shape=tuple(nonsparse_code.shape[:-1]),
            device=nonsparse_code.device,
            dtype=nonsparse_code.dtype,
        )
        if not self.use_adaptive_thresholds:
            return thresholds

        u0 = shrink(nonsparse_code, thresholds)
        reconstruction_residual = torch.norm(x - self._decode_from_dictionary(u0), dim=-1, keepdim=True)
        if latent_prior_internal is None:
            prior_gap = torch.zeros_like(reconstruction_residual)
        else:
            prior_gap = torch.norm(u0 - latent_prior_internal, dim=-1, keepdim=True)
        adaptive_addition = (
            self.alpha_residual_coeff * reconstruction_residual
            + self.alpha_prior_coeff * prior_gap
        ) / self.L
        return thresholds + adaptive_addition
    
    def forward(
        self,
        x: torch.Tensor,
        latent_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: iterative soft-thresholding.
        
        Args:
            x: Input tensor of shape [..., xdim]
            latent_prior: Optional warm-start latent for dynamics-aware reencoding.
            
        Returns:
            Sparse codes of shape [..., zdim]
        """
        latent_prior_internal: Optional[torch.Tensor] = None
        if latent_prior is not None:
            latent_prior = latent_prior.to(device=x.device, dtype=x.dtype)
            expected_shape = (*x.shape[:-1], self.zdim)
            if latent_prior.shape != expected_shape:
                raise ValueError(
                    "latent_prior must match the encoded shape for LISTA warm starts: "
                    f"got {latent_prior.shape}, expected {expected_shape}"
                )
            latent_prior_internal = (
                self._unsplit_sign(latent_prior) if self.final_op == "sign_split" else latent_prior
            )

        nonsparse_code = self._compute_precode(x)
        threshold = self._compute_thresholds(
            x=x,
            nonsparse_code=nonsparse_code,
            latent_prior_internal=latent_prior_internal,
        )

        def apply_step(pre_act: torch.Tensor, is_final_step: bool) -> torch.Tensor:
            structured_pre_act = pre_act
            if self.topk_groups > 0:
                structured_pre_act = _mask_to_topk_groups(
                    structured_pre_act,
                    self.group_slices,
                    self.topk_groups,
                )
            if self.use_group_shrinkage:
                structured_pre_act = _sparse_group_shrink(
                    structured_pre_act,
                    _reduce_thresholds_to_groups(
                        self.group_threshold_scale * threshold,
                        self.group_slices,
                        reference=structured_pre_act,
                    ),
                    self.group_slices,
                )
            z = shrink(structured_pre_act, threshold)
            if is_final_step:
                if self.final_op == "relu":
                    return F.relu(z)
                if self.final_op == "sign_split":
                    return self._split_sign(z)
            return z

        if latent_prior_internal is not None:
            z = latent_prior_internal
            z_prev = latent_prior_internal
            num_refinement_steps = max(1, self.num_loops)
        else:
            # If no loops, the initialization is also the final step.
            z = apply_step(nonsparse_code, is_final_step=(self.num_loops == 0))
            z_prev = torch.zeros_like(z)
            num_refinement_steps = self.num_loops

        # Iterative refinement
        for loop_idx in range(num_refinement_steps):
            is_final_step = loop_idx == (num_refinement_steps - 1)
            if self.use_momentum:
                momentum = self.momentum_beta * (z - z_prev)
            else:
                momentum = 0.0
            z_next = apply_step(z @ self.S + nonsparse_code + momentum, is_final_step=is_final_step)
            z_prev = z
            z = z_next
        
        return z


class HyperLISTA(nn.Module):
    """HyperLISTA encoder with analytically-derived, instance-adaptive parameters.
    
    Unlike standard LISTA which learns W_e and S matrices, HyperLISTA:
    1. Derives W_e = (1/L) * D.T from the decoder dictionary
    2. Computes S = I - (1/L) * D.T @ D on the fly
    3. Uses instance-adaptive threshold, momentum, and support selection
    4. Has only 3 learnable scalar hyperparameters (c_theta, c_beta, c_ss)
    
    This enables gradient flow from the Koopman loss back to D.
    
    Reference: "Hyperparameter Tuning is All You Need for LISTA" (Chen et al., NeurIPS 2021)
    
    Args:
        cfg: Configuration object with HYPERLISTA settings
        xdim: Input dimension (internal, includes homogeneous coord if used)
        dict_param: Reference to decoder dictionary parameter [zdim, xdim]
    """
    
    def __init__(self, cfg: Config, xdim: int, dict_param: nn.Parameter):
        super().__init__()
        self.cfg = cfg
        self.xdim = xdim
        self.zdim = cfg.MODEL.TARGET_SIZE
        self.num_loops = cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS
        
        # Reference to decoder dictionary (shared, not copied)
        # dict_param has shape [zdim, xdim] (transposed from decoder perspective)
        self.dict_param = dict_param
        
        # Learnable hyperparameters (or fixed if LEARN_HYPERPARAMS=False)
        hypercfg = cfg.MODEL.ENCODER.HYPERLISTA
        self.constrain_c_theta = hypercfg.CONSTRAIN_C_THETA
        self.c_theta_min = hypercfg.C_THETA_MIN
        self.pinv_cache_mode = hypercfg.PINV_CACHE_MODE
        c_theta_init = self._c_theta_to_storage(hypercfg.C_THETA)
        if hypercfg.LEARN_HYPERPARAMS:
            self.c_theta_raw = nn.Parameter(torch.tensor([c_theta_init]))
            self.c_beta = nn.Parameter(torch.tensor([hypercfg.C_BETA]))
            self.c_ss = nn.Parameter(torch.tensor([hypercfg.C_SS]))
        else:
            self.register_buffer('c_theta_raw', torch.tensor([c_theta_init]))
            self.register_buffer('c_beta', torch.tensor([hypercfg.C_BETA]))
            self.register_buffer('c_ss', torch.tensor([hypercfg.C_SS]))
        
        self.use_ss = hypercfg.USE_SUPPORT_SELECTION
        self.use_momentum = hypercfg.USE_MOMENTUM
        self.mag_ratio = hypercfg.MAG_RATIO
        self.use_group_shrinkage = bool(hypercfg.GROUP_SHRINKAGE)
        self.group_threshold_scale = float(hypercfg.GROUP_THRESHOLD_SCALE)
        self.topk_groups = int(hypercfg.TOPK_GROUPS)
        self.group_slices = _infer_encoder_group_slices(cfg, self.zdim)
        if (self.use_group_shrinkage or self.topk_groups > 0) and not self.group_slices:
            raise ValueError(
                "HyperLISTA group-aware shrinkage requires structured, block-diagonal, or soft-block latent groups."
            )
        if self.use_group_shrinkage or self.topk_groups > 0:
            print(
                "  HyperLISTA group-aware encoder:"
                f" groups={len(self.group_slices)},"
                f" group_shrinkage={self.use_group_shrinkage},"
                f" threshold_scale={self.group_threshold_scale:.3f},"
                f" topk_groups={self.topk_groups}"
            )

    def _c_theta_to_storage(self, c_theta: float) -> float:
        """Map user-facing c_theta to the stored parameter domain."""
        if not self.constrain_c_theta:
            return c_theta
        shifted = max(c_theta - self.c_theta_min, 1e-12)
        return math.log(math.expm1(shifted))

    @property
    def c_theta(self) -> torch.Tensor:
        """User-facing threshold scale."""
        if self.constrain_c_theta:
            return F.softplus(self.c_theta_raw) + self.c_theta_min
        return self.c_theta_raw

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        legacy_key = f"{prefix}c_theta"
        raw_key = f"{prefix}c_theta_raw"
        if raw_key not in state_dict and legacy_key in state_dict:
            legacy_tensor = state_dict.pop(legacy_key)
            legacy_value = float(legacy_tensor.reshape(-1)[0].item())
            converted = self._c_theta_to_storage(legacy_value)
            state_dict[raw_key] = legacy_tensor.new_tensor([converted])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _get_D_pinv(self, D: torch.Tensor) -> torch.Tensor:
        """Get pseudo-inverse of D for error approximation."""
        if self.pinv_cache_mode != "none":
            raise ValueError(
                f"Unsupported HyperLISTA PINV_CACHE_MODE='{self.pinv_cache_mode}'. "
                "Only 'none' is implemented."
            )
        with torch.no_grad():
            return torch.linalg.pinv(D)
    
    def _compute_L(self, D: torch.Tensor) -> torch.Tensor:
        """Compute Lipschitz constant L = spectral_norm(D.T @ D).
        
        Uses power iteration for efficiency (5 iterations typically sufficient).
        
        Args:
            D: Dictionary matrix [xdim, zdim]
            
        Returns:
            Lipschitz constant with small safety margin
        """
        # Use power iteration for efficiency
        gram = D.T @ D  # [zdim, zdim]
        v = torch.randn(self.zdim, 1, device=D.device, dtype=D.dtype)
        v = v / torch.norm(v)
        for _ in range(5):  # 5 iterations is usually sufficient
            v = gram @ v
            v = v / torch.norm(v)
        L = torch.norm(gram @ v) / torch.norm(v)
        return L * 1.05  # Safety margin

    def _apply_group_structure(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Apply optional group-first selection and sparse-group shrinkage."""

        structured_x = x
        if self.topk_groups > 0:
            structured_x = _mask_to_topk_groups(structured_x, self.group_slices, self.topk_groups)
        if self.use_group_shrinkage:
            structured_x = _sparse_group_shrink(
                structured_x,
                self.group_threshold_scale * theta,
                self.group_slices,
            )
        return structured_x
    
    def forward(
        self,
        x: torch.Tensor,
        latent_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with instance-adaptive parameters.
        
        The HyperLISTA iteration:
        1. Compute residual: r = D @ z - x
        2. Gradient step: z_tilde = z - γ * D.T @ r + momentum
        3. Adaptive threshold based on current error estimate
        4. Soft-threshold with support selection: z_next = η_θ^SS(z_tilde)
        
        Args:
            x: Input tensor [..., xdim]
            
        Returns:
            Sparse codes [..., zdim]
        """
        # Get normalized dictionary (decoder shares this)
        D = self.dict_param.T  # [xdim, zdim]
        D = D / torch.norm(D, dim=0, keepdim=True).clamp(min=1e-6)
        
        # Compute derived quantities
        L = self._compute_L(D)
        gamma = 1.0 / L
        W_e = gamma * D.T  # [zdim, xdim]
        
        # Initial encoding
        c = x @ W_e.T  # [..., zdim]
        theta_init = self.c_theta * gamma
        c_structured = self._apply_group_structure(c, theta_init)
        z = self._shrink(c_structured, theta_init)  # Initial thresholding
        z_prev = torch.zeros_like(z)
        if latent_prior is not None:
            latent_prior = latent_prior.to(device=z.device, dtype=z.dtype)
            if latent_prior.shape != z.shape:
                raise ValueError(
                    "latent_prior must match the encoded shape for HyperLISTA warm starts: "
                    f"got {latent_prior.shape}, expected {z.shape}"
                )
            z = latent_prior
            z_prev = latent_prior
        
        # Get pseudo-inverse for error approximation
        D_pinv = self._get_D_pinv(D)
        
        # Initial error estimate for support selection
        if self.use_ss:
            initial_error = torch.norm(x @ D_pinv.T, p=1, dim=-1, keepdim=True) + 1e-8
        
        # Warm-started reencoding should still perform at least one correction step.
        num_refinement_steps = self.num_loops if latent_prior is None else max(1, self.num_loops)

        # Unrolled iterations
        for k in range(num_refinement_steps):
            # Compute residual and gradient step
            residual = z @ D.T - x  # [..., xdim]
            grad = residual @ W_e.T  # [..., zdim]
            
            # Momentum term
            if self.use_momentum:
                # Estimate support size from current estimate
                z_abs = z.abs()
                max_mag = z_abs.max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
                support = (z_abs > self.mag_ratio * max_mag).float().sum(dim=-1, keepdim=True)
                beta = self.c_beta * support
                momentum = beta * (z - z_prev)
            else:
                momentum = 0.0
            
            # Pre-threshold state
            z_tilde = z - gamma * grad + momentum
            
            # Adaptive threshold based on error approximation
            approx_error = torch.norm(residual @ D_pinv.T, p=1, dim=-1, keepdim=True) + 1e-8
            theta = self.c_theta * gamma * approx_error
            z_tilde = self._apply_group_structure(z_tilde, theta)
            
            # Support selection
            if self.use_ss:
                log_ratio = torch.log(initial_error / approx_error)
                p = (self.c_ss * log_ratio).clamp(0.0, 1.0)
                z_next = self._shrink_ss(z_tilde, theta, p)
            else:
                z_next = self._shrink(z_tilde, theta)
            
            z_prev = z
            z = z_next
        
        return z
    
    def _shrink(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Soft thresholding operator.
        
        η_θ(x)_i = sign(x_i) * max(|x_i| - θ, 0)
        
        Args:
            x: Input tensor
            theta: Threshold (scalar or broadcastable)
            
        Returns:
            Thresholded tensor
        """
        return torch.sign(x) * torch.relu(x.abs() - theta)
    
    def _shrink_ss(self, x: torch.Tensor, theta: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Soft thresholding with support selection.
        
        Support selection bypasses thresholding for the top-p% largest magnitude entries,
        helping to preserve the support structure during convergence.
        
        NOTE: This uses a vectorized top-k approximation for speed. It selects the
        top-k magnitudes per sample (k = ceil(p * zdim)) instead of computing the
        exact per-sample quantile threshold.
        
        Args:
            x: Input tensor [..., zdim]
            theta: Threshold [..., 1]
            p: Percentage to bypass (0 to 1) [..., 1]
            
        Returns:
            Thresholded tensor with support selection
        """
        x_abs = x.abs()
        batch_shape = x.shape[:-1]
        zdim = x.shape[-1]
        
        # Flatten batch dims for vectorized top-k selection
        flat_x_abs = x_abs.reshape(-1, zdim)
        flat_theta = theta.reshape(-1, 1)
        flat_p = p.reshape(-1).clamp(0.0, 1.0)
        
        if flat_x_abs.numel() == 0:
            return self._shrink(x, theta)
        
        # Select top-k per sample (approximate support selection)
        k = torch.ceil(flat_p * zdim).to(torch.long).clamp(0, zdim)
        k_max = int(k.max().item()) if k.numel() > 0 else 0
        
        if k_max == 0:
            bypass_flat = torch.zeros_like(flat_x_abs, dtype=torch.bool)
        else:
            topk_idx = torch.topk(flat_x_abs, k_max, dim=-1, largest=True, sorted=True).indices
            selector = torch.arange(k_max, device=x.device).unsqueeze(0) < k.unsqueeze(1)
            bypass_flat = torch.zeros_like(flat_x_abs, dtype=torch.bool)
            bypass_flat.scatter_(1, topk_idx, selector)
        
        # Bypass if: (1) selected in top-k AND (2) above soft-threshold theta
        bypass_flat = bypass_flat & (flat_x_abs >= flat_theta)
        bypass = bypass_flat.view(*batch_shape, zdim).detach()
        
        # Apply shrinkage only to non-bypassed entries
        return torch.where(bypass, x, self._shrink(x, theta))


# ---------------------------------------------------------------------------
# Koopman Machine Base Class
# ---------------------------------------------------------------------------

class KoopmanMachine(ABC, nn.Module):
    """Abstract base class for Koopman operator learning.
    
    The Koopman operator is a linear operator that provides a mathematical 
    framework for representing the dynamics of a nonlinear dynamical system (NLDS) 
    in terms of an infinite-dimensional linear operator. 
    Formally, the Koopman operator advances a measurement function forward in time 
    through the underlying system dynamics.
    
    This class provides the interface for learning Koopman representations.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__()
        self.cfg = cfg
        self.observation_size = observation_size
        self.target_size = cfg.MODEL.TARGET_SIZE
        self.dt = None  # Will be set from environment config if needed
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Latent codes of shape [..., target_size]
        """
        pass

    def encode_with_prior(
        self,
        x: torch.Tensor,
        latent_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations with an optional latent warm start.

        Models without dynamics-aware sparse inference simply ignore the prior.
        """
        del latent_prior
        return self.encode(x)
    
    @abstractmethod
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to observation space.
        
        Args:
            y: Latent representations of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        pass
    
    @abstractmethod
    def kmatrix(self) -> torch.Tensor:
        """Extract the learned Koopman matrix from parameters.
        
        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        pass
    
    def residual(self, x: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss between consecutive states in latent space.
        Determines how linearly aligned x & nx are in the latent space.
        
        Args:
            x: Current states of shape [..., observation_size]
            nx: Next states of shape [..., observation_size]
            
        Returns:
            Residual norms of shape [...]
        """
        y = self.encode(x)
        ny = self.encode(nx)
        kmat = self.kmatrix()
        return torch.norm(y @ kmat - ny, dim=-1)
    
    def reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction via encode-decode.
        
        Args:
            x: shape [..., observation_size]
            
        Returns:
            Reconstructions of shape [..., observation_size]
        """
        return self.decode(self.encode(x))
    
    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity loss on latent codes.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Scalar sparsity loss
        """
        z = self.encode(x)
        return torch.norm(z, p=1, dim=-1).mean()
    
    def step_latent(self, y: torch.Tensor) -> torch.Tensor:
        """Step forward in latent space using Koopman matrix.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Next latent codes of shape [..., target_size]
        """
        kmat = self.kmatrix()
        return y @ kmat
    
    def step_env(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next observation using Koopman dynamics.
        
        Args:
            x: Current observations of shape [..., observation_size]
            
        Returns:
            Predicted next observations of shape [..., observation_size]
        """
        y = self.encode(x)
        ny = self.step_latent(y)
        nx = self.decode(ny)
        return nx
    
    def rollout_latent_discrete(
        self,
        z0: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Roll out latent states with repeated discrete Koopman applications."""
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        z_hat = []
        z_cur = z0
        for _ in range(horizon):
            z_cur = self.step_latent(z_cur)
            z_hat.append(z_cur)
        return torch.stack(z_hat, dim=1)

    def rollout_observation_discrete(
        self,
        x0: torch.Tensor,
        horizon: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roll out latents and decoded observations from initial observation."""
        z0 = self.encode(x0)
        z_pred = self.rollout_latent_discrete(z0, horizon=horizon)
        batch_size = z_pred.shape[0]
        x_pred = self.decode(z_pred.reshape(batch_size * horizon, self.target_size))
        x_pred = x_pred.reshape(batch_size, horizon, self.observation_size)
        return z_pred, x_pred

    @staticmethod
    def _to_scalar_tensor(value: Any, *, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
        """Normalize scalar-like loss inputs into a scalar tensor."""
        if isinstance(value, torch.Tensor):
            out = value.to(device=device, dtype=dtype)
            if out.ndim == 0:
                return out
            return out.mean()
        if isinstance(value, (float, int)):
            return torch.tensor(float(value), device=device, dtype=dtype)
        raise ValueError(f"Expected scalar-like value for '{name}', got {type(value).__name__}")

    def _prediction_loss_from_tensors(self, x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
        """Raw prediction loss: ||x_pred - x_true|| averaged over batch/time."""
        return torch.norm(x_pred - x_true, dim=-1).mean()

    def _observation_loss_dim_scale(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the observation-space dimension normalization factor."""
        mode = str(getattr(self.cfg.MODEL, "OBS_LOSS_DIM_NORMALIZATION", "sqrt_dim")).strip().lower()
        obs_dim = max(1, int(self.observation_size))
        if mode == "none":
            scale = 1.0
        elif mode == "sqrt_dim":
            scale = math.sqrt(float(obs_dim))
        elif mode == "dim":
            scale = float(obs_dim)
        else:
            raise ValueError(
                f"Unknown OBS_LOSS_DIM_NORMALIZATION='{self.cfg.MODEL.OBS_LOSS_DIM_NORMALIZATION}'. "
                "Expected one of ['none', 'sqrt_dim', 'dim']."
            )
        return torch.tensor(scale, device=device, dtype=dtype)

    def _alignment_loss_from_tensors(
        self,
        z_pred: Optional[torch.Tensor],
        z_true: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Alignment loss: ||z_pred - z_true|| averaged over batch/time."""
        if z_pred is None or z_true is None:
            if self.cfg.MODEL.RES_COEFF != 0.0:
                raise ValueError("Alignment is enabled (RES_COEFF != 0) but z_pred/z_true were not provided.")
            return torch.zeros((), device=device, dtype=dtype)
        return torch.norm(z_pred - z_true, dim=-1).mean()

    def _reconstruction_loss(
        self,
        reconstruction_error: Optional[Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reconstruction loss from a precomputed scalar term."""
        if reconstruction_error is None:
            return torch.zeros((), device=device, dtype=dtype)
        return self._to_scalar_tensor(
            reconstruction_error, device=device, dtype=dtype, name="reconstruction_error"
        )

    def _sparsity_loss_from_inputs(
        self,
        sparsity_error: Optional[Any],
        sparsity_latent: Optional[torch.Tensor],
        z_pred: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sparsity loss from an explicit scalar term or latent tensor."""
        if sparsity_error is not None:
            return self._to_scalar_tensor(
                sparsity_error, device=device, dtype=dtype, name="sparsity_error"
            )

        if sparsity_latent is None:
            sparsity_latent = z_pred
        if sparsity_latent is None:
            return torch.zeros((), device=device, dtype=dtype)

        sparsity_latent = sparsity_latent.to(device=device, dtype=dtype)
        return torch.norm(sparsity_latent, p=1, dim=-1).mean()

    def _sparsity_ratio_from_latent(
        self,
        z_pred: Optional[torch.Tensor],
        z_true: Optional[torch.Tensor],
        z0: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Monitoring metric: ratio of inactive latent dimensions."""
        z_for_monitor = z_true if z_true is not None else z_pred
        if z_for_monitor is None:
            z_for_monitor = z0
        if z_for_monitor is None:
            return torch.tensor(0.0, device=device, dtype=dtype)

        z_for_monitor = z_for_monitor.to(device=device, dtype=dtype)
        num_nonzero_codes = (z_for_monitor.abs() > 1e-6).float().sum(dim=-1).mean()
        return 1.0 - num_nonzero_codes / self.target_size

    def _k_eigen_metrics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (max_real_eigenvalue, spectral_radius) for monitoring."""
        kmat = self.kmatrix()
        kmat_for_eig = kmat.cpu() if kmat.device.type == 'mps' else kmat
        eigvals = torch.linalg.eigvals(kmat_for_eig)
        return torch.max(eigvals.real), torch.max(eigvals.abs())

    def _aggregate_losses_from_tensors(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
        z_pred: Optional[torch.Tensor] = None,
        z_true: Optional[torch.Tensor] = None,
        reconstruction_error: Optional[Any] = None,
        sparsity_error: Optional[Any] = None,
        sparsity_latent: Optional[torch.Tensor] = None,
        homogeneous_loss: Optional[Any] = None,
        block_losses: Optional[Dict[str, Any]] = None,
        structured_latent: Optional[torch.Tensor] = None,
        temporal_latent_sequence: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Pure tensor aggregation for unified loss."""
        del x0, step, homogeneous_loss, block_losses, structured_latent, temporal_latent_sequence, loss_weights
        # Base aggregation intentionally ignores structured-specific inputs.
        if x_pred.ndim < 3 or x_true.ndim < 3:
            raise ValueError("x_pred/x_true must have shape [B, H, ...]")
        if x_pred.shape != x_true.shape:
            raise ValueError(f"x_pred shape {tuple(x_pred.shape)} must match x_true shape {tuple(x_true.shape)}")
        horizon = x_pred.shape[1]
        if horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {horizon}")

        if z_pred is not None and z_pred.shape[:2] != x_pred.shape[:2]:
            raise ValueError("z_pred must match [B, H] of x_pred")
        if z_true is not None and z_true.shape[:2] != x_true.shape[:2]:
            raise ValueError("z_true must match [B, H] of x_true")

        device = x_pred.device
        dtype = x_pred.dtype

        obs_loss_dim_scale = self._observation_loss_dim_scale(device=device, dtype=dtype)
        prediction_loss_raw = self._prediction_loss_from_tensors(x_pred, x_true)
        alignment_loss = self._alignment_loss_from_tensors(z_pred, z_true, device=device, dtype=dtype)
        reconst_loss_raw = self._reconstruction_loss(reconstruction_error, device=device, dtype=dtype)
        sparsity_loss = self._sparsity_loss_from_inputs(
            sparsity_error, sparsity_latent, z_pred, device=device, dtype=dtype
        )
        prediction_loss = prediction_loss_raw / obs_loss_dim_scale
        reconst_loss = reconst_loss_raw / obs_loss_dim_scale

        sequence_term_scale = 1.0 / float(horizon)
        prediction_loss_raw = prediction_loss_raw * sequence_term_scale
        prediction_loss = prediction_loss * sequence_term_scale
        alignment_loss = alignment_loss * sequence_term_scale
        reconst_loss_raw = reconst_loss_raw * sequence_term_scale
        reconst_loss = reconst_loss * sequence_term_scale
        sparsity_loss = sparsity_loss * sequence_term_scale

        total_loss = (
            self.cfg.MODEL.RES_COEFF * alignment_loss +
            self.cfg.MODEL.RECONST_COEFF * reconst_loss +
            self.cfg.MODEL.PRED_COEFF * prediction_loss +
            self.cfg.MODEL.SPARSITY_COEFF * sparsity_loss
        )

        with torch.no_grad():
            max_eigenvalue, spectral_radius = self._k_eigen_metrics()
            sparsity_ratio = self._sparsity_ratio_from_latent(
                z_pred, z_true, z0, device=device, dtype=dtype
            )

        metrics = {
            'loss': total_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'residual_loss': alignment_loss.item(),
            'reconst_loss_raw': reconst_loss_raw.item(),
            'reconst_loss': reconst_loss.item(),
            'prediction_loss_raw': prediction_loss_raw.item(),
            'prediction_loss': prediction_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'sequence_term_scale': sequence_term_scale,
            'obs_loss_dim_scale': float(obs_loss_dim_scale.item()),
            'obs_loss_dim_normalization': str(
                getattr(self.cfg.MODEL, "OBS_LOSS_DIM_NORMALIZATION", "sqrt_dim")
            ),
            'A_max_eigenvalue': max_eigenvalue.item(),
            'spectral_radius': spectral_radius.item(),
            'sparsity_ratio': sparsity_ratio.item(),
        }
        return total_loss, metrics

    def loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
        z_pred: Optional[torch.Tensor] = None,
        z_true: Optional[torch.Tensor] = None,
        reconstruction_error: Optional[Any] = None,
        sparsity_error: Optional[Any] = None,
        sparsity_latent: Optional[torch.Tensor] = None,
        homogeneous_loss: Optional[Any] = None,
        block_losses: Optional[Dict[str, Any]] = None,
        structured_latent: Optional[torch.Tensor] = None,
        temporal_latent_sequence: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Canonical unified loss API (pure aggregation)."""
        return self._aggregate_losses_from_tensors(
            x_pred=x_pred,
            x_true=x_true,
            x0=x0,
            z0=z0,
            z_pred=z_pred,
            z_true=z_true,
            reconstruction_error=reconstruction_error,
            sparsity_error=sparsity_error,
            sparsity_latent=sparsity_latent,
            homogeneous_loss=homogeneous_loss,
            block_losses=block_losses,
            structured_latent=structured_latent,
            temporal_latent_sequence=temporal_latent_sequence,
            loss_weights=loss_weights,
            step=step,
        )


# ---------------------------------------------------------------------------
# Concrete Implementations
# ---------------------------------------------------------------------------


class GenericKM(KoopmanMachine):
    """Generic Koopman Machine with MLP encoder and decoder.
    
    This is the standard Koopman autoencoder with configurable MLP architectures.
    Optionally supports normalization of latent codes.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__(cfg, observation_size)
        
        # Encoder
        self.encoder = MLPCoder(
            input_size=observation_size,
            target_size=cfg.MODEL.TARGET_SIZE,
            hidden_layers=cfg.MODEL.ENCODER.LAYERS,
            use_bias=cfg.MODEL.ENCODER.USE_BIAS,
            last_relu=cfg.MODEL.ENCODER.LAST_RELU,
            activation=cfg.MODEL.ENCODER.ACTIVATION,
        )
        
        # Decoder
        self.decoder = MLPCoder(
            input_size=cfg.MODEL.TARGET_SIZE,
            target_size=observation_size,
            hidden_layers=cfg.MODEL.DECODER.LAYERS,
            use_bias=cfg.MODEL.DECODER.USE_BIAS,
            last_relu=False,
            activation=cfg.MODEL.DECODER.ACTIVATION,
        )
        
        # Koopman matrix (learnable)
        self.kmat = nn.Parameter(torch.eye(cfg.MODEL.TARGET_SIZE))

        self.norm_fn_name = cfg.MODEL.NORM_FN
    
    def _norm_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to latent codes.
        
        Args:
            x: Latent codes
            
        Returns:
            Normalized latent codes
        """
        if self.norm_fn_name == 'id':
            return x
        elif self.norm_fn_name == 'ball':
            return x / torch.norm(x, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown norm function '{self.norm_fn_name}'")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Latent codes of shape [..., target_size]
        """
        y = self.encoder(x)
        return self._norm_fn(y)

    def encode_with_prior(
        self,
        x: torch.Tensor,
        latent_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations to latent space.

        The generic MLP encoder does not consume latent priors.
        """
        del latent_prior
        return self.encode(x)
    
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to observation space.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        return self.decoder(y)
    
    def kmatrix(self) -> torch.Tensor:
        """Get the Koopman matrix.
        
        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        return self.kmat
    
    def step_latent(self, y: torch.Tensor) -> torch.Tensor:
        """Step forward in latent space with normalization.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Next latent codes of shape [..., target_size]
        """
        ny = y @ self.kmatrix()
        return self._norm_fn(ny)

# TODO: test this class with experiments. Sweep over the sparsity coefficient values.
# TODO: test this on the Lyapunov environment
class LISTAKM(KoopmanMachine):
    """Koopman Machine with a sparse LISTA-family encoder.
    
    Supports both LISTA and HyperLISTA encoders through
    ``cfg.MODEL.ENCODER.ENCODER_TYPE``. The decoder uses a normalized dictionary.
    
    Supports homogeneous coordinates: when enabled, input x is augmented to [x, 1],
    allowing the dictionary to learn an implicit bias through an extra dimension.
    The decoder outputs [x̂, ĉ] internally, with ĉ penalized to stay close to 1.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space (physical, without homogeneous)
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__(cfg, observation_size)
        encoder_type = cfg.MODEL.ENCODER.ENCODER_TYPE.lower()
        lista_final_op = cfg.MODEL.ENCODER.LISTA.FINAL_OP.lower()
        self._uses_sign_split_latent = encoder_type == "lista" and lista_final_op == "sign_split"
        self._encoder_target_size = self.target_size
        if self._uses_sign_split_latent:
            if self.target_size % 2 != 0:
                raise ValueError(
                    "LISTAKM sign-split latents require an even TARGET_SIZE "
                    f"(got {self.target_size})."
                )
            self._encoder_target_size = self.target_size // 2
        
        # Homogeneous coordinates: augment input with constant 1
        self.use_homogeneous = cfg.MODEL.USE_HOMOGENEOUS
        self._internal_obs_size = observation_size + 1 if self.use_homogeneous else observation_size

        # Initialize dictionary with unit-norm columns.
        encoder_Wd_init = self._init_dictionary(self._encoder_target_size)
        Wd_init = self._expand_sign_split_dictionary(encoder_Wd_init)

        # Register as buffer so it's saved/loaded but not updated by optimizer
        self.register_buffer('dict_init', Wd_init.clone())
        
        # Decoder dictionary parameter [zdim, internal_obs_size]
        self.dict = nn.Parameter(Wd_init.T)
        
        # Optional decoder bias [observation_size] - helps capture system mean/center
        # Note: If using homogeneous coordinates, bias is learned implicitly via the extra dimension
        self.use_decoder_bias = cfg.MODEL.DECODER.AFFINE_BIAS and not self.use_homogeneous
        if self.use_decoder_bias:
            bound = 1.0 / math.sqrt(cfg.MODEL.TARGET_SIZE)
            bias_init = torch.empty(observation_size).uniform_(-bound, bound)
            self.decoder_bias = nn.Parameter(bias_init)
        else:
            self.register_buffer('decoder_bias', torch.zeros(observation_size))

        # Encoder (LISTA / HyperLISTA)
        self.encoder = self._build_encoder(cfg, self._internal_obs_size, encoder_Wd_init)
        if self.use_homogeneous:
            print(f"  Using homogeneous coordinates: input {observation_size} -> internal {self._internal_obs_size}")
        if self._uses_sign_split_latent:
            print(
                "  Using sign-split LISTA latents: "
                f"base {self._encoder_target_size} -> output {self.target_size}"
            )

        # Koopman matrix (learnable) — structure depends on cfg.MODEL.K_STRUCTURE
        self._k_structure = cfg.MODEL.K_STRUCTURE
        zdim = cfg.MODEL.TARGET_SIZE
        self._soft_block_cfg = cfg.MODEL.SOFT_BLOCK
        self._soft_block_sizes: List[int] = []
        self.register_buffer("_soft_block_mask", torch.empty(0, 0, dtype=torch.bool), persistent=False)
        if self._k_structure == "diagonal":
            self.kmat_diag = nn.Parameter(torch.ones(zdim))
            print(f"  Diagonal K: {zdim} parameters")
        elif self._k_structure == "block_diagonal":
            requested_num_blocks = int(getattr(cfg.MODEL, "K_NUM_BLOCKS", 0))
            self._k_block_sizes: List[int] = []

            if requested_num_blocks > 0:
                self._k_block_size = 0
                self._k_num_blocks = requested_num_blocks
                self._k_remainder = 0
                self._k_block_sizes = self._split_block_sizes(zdim, requested_num_blocks)
                self.kmat_blocks = nn.ParameterList([
                    nn.Parameter(torch.eye(block_size))
                    for block_size in self._k_block_sizes
                ])
                print(
                    "  Block-diagonal K: "
                    f"{requested_num_blocks} blocks with sizes {self._k_block_sizes}"
                )
            else:
                block_size = cfg.MODEL.K_BLOCK_SIZE
                if block_size <= 0:
                    block_size = max(1, zdim // 13)
                self._k_block_size = block_size
                self._k_num_blocks = zdim // block_size
                self._k_remainder = zdim - self._k_num_blocks * block_size
                self._k_block_sizes = [block_size] * self._k_num_blocks
                self.kmat_blocks = nn.ParameterList([
                    nn.Parameter(torch.eye(block_size))
                    for _ in range(self._k_num_blocks)
                ])
                if self._k_remainder > 0:
                    self.kmat_remainder = nn.Parameter(torch.eye(self._k_remainder))
                    self._k_block_sizes.append(self._k_remainder)
                print(f"  Block-diagonal K: {self._k_num_blocks} blocks of size "
                      f"{block_size}" + (f" + remainder {self._k_remainder}" if self._k_remainder > 0 else ""))
        else:
            # Dense (default)
            self.kmat = nn.Parameter(torch.eye(zdim))

        # Block activation losses (only valid for block_diagonal K)
        self._block_loss_cfg = cfg.MODEL.BLOCK_LOSS
        if self._block_loss_cfg.ENABLED and self._k_structure != "block_diagonal":
            print("  Block losses enabled but K is not block_diagonal; disabling block losses.")
            self._block_loss_cfg.ENABLED = False
        if self._block_loss_cfg.ENABLED:
            print("  Block losses enabled:")
            print(f"    one_block={self._block_loss_cfg.ONE_BLOCK_LOSS}"
                  f" (w={self._block_loss_cfg.ONE_BLOCK_WEIGHT})")
            print(f"    balance={self._block_loss_cfg.BALANCE_LOSS}"
                  f" (w={self._block_loss_cfg.BALANCE_WEIGHT})")
            if self._block_loss_cfg.ONE_BLOCK_LOSS == "top1_margin":
                print(f"    top1_margin={self._block_loss_cfg.TOP1_MARGIN}")
            print(f"    energy_norm={self._block_loss_cfg.ENERGY_NORM}")

        if self._soft_block_cfg.ENABLED:
            if self._k_structure != "dense":
                raise ValueError("Soft block penalty requires K_STRUCTURE='dense'.")
            if int(self._soft_block_cfg.NUM_BLOCKS) <= 0:
                raise ValueError("SOFT_BLOCK.NUM_BLOCKS must be positive when soft block penalty is enabled.")
            self._soft_block_sizes = self._split_block_sizes(zdim, int(self._soft_block_cfg.NUM_BLOCKS))
            mask = torch.zeros(zdim, zdim, dtype=torch.bool)
            offset = 0
            for block_size in self._soft_block_sizes:
                mask[offset:offset + block_size, offset:offset + block_size] = True
                offset += block_size
            self._soft_block_mask = mask
            print(
                "  Soft block penalty enabled: "
                f"{len(self._soft_block_sizes)} blocks with sizes {self._soft_block_sizes}, "
                f"norm={self._soft_block_cfg.NORM}, w={self._soft_block_cfg.WEIGHT}"
            )

    @staticmethod
    def _split_block_sizes(zdim: int, num_blocks: int) -> List[int]:
        """Split a latent dimension into near-equal positive block sizes."""
        if num_blocks <= 0:
            raise ValueError("K_NUM_BLOCKS must be positive when enabled.")
        if num_blocks > zdim:
            raise ValueError(
                f"K_NUM_BLOCKS={num_blocks} exceeds latent size TARGET_SIZE={zdim}."
            )
        base = zdim // num_blocks
        remainder = zdim % num_blocks
        return [base + (1 if block_index < remainder else 0) for block_index in range(num_blocks)]

    def _block_diagonal_matrices(self) -> List[torch.Tensor]:
        """Return all block matrices in latent order."""
        blocks = [block for block in self.kmat_blocks]
        if getattr(self, "_k_remainder", 0) > 0 and hasattr(self, "kmat_remainder"):
            blocks.append(self.kmat_remainder)
        return blocks

    def _soft_block_penalty(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return off-block penalty and off-block ratio for dense soft block-sparse K."""
        if not self._soft_block_cfg.ENABLED or self._k_structure != "dense" or not self._soft_block_sizes:
            return None, None
        off_mask = ~self._soft_block_mask
        off_entries = self.kmat.masked_select(off_mask)
        if self._soft_block_cfg.NORM == "fro":
            penalty = off_entries.square().sum()
            total = self.kmat.square().sum()
        else:
            penalty = off_entries.abs().sum()
            total = self.kmat.abs().sum()
        ratio = penalty / total.clamp(min=1e-8)
        return penalty, ratio

    def _decoder_atoms_for_coherence(self) -> torch.Tensor:
        """Return normalized decoder atoms for coherence diagnostics.

        Sign-split decoders carry paired positive/negative atoms that should not
        be penalized against each other. For those models, collapse each pair
        back to one effective atom before computing coherence.
        """
        atoms = self.dict
        if self._uses_sign_split_latent:
            half = atoms.shape[0] // 2
            atoms = 0.5 * (atoms[:half] - atoms[half:])
        return atoms / torch.norm(atoms, dim=1, keepdim=True).clamp(min=1e-8)

    def _decoder_coherence_penalty(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return off-diagonal decoder coherence penalty and max correlation."""
        weight = float(getattr(self.cfg.MODEL, "DECODER_COHERENCE_WEIGHT", 0.0))
        if weight <= 0.0:
            return None, None
        atoms = self._decoder_atoms_for_coherence()
        if atoms.shape[0] <= 1:
            zero = atoms.new_tensor(0.0)
            return zero, zero
        gram = atoms @ atoms.T
        off_diag = gram - torch.diag_embed(torch.diagonal(gram))
        penalty = off_diag.square().sum()
        max_off_diag = off_diag.abs().max()
        return penalty, max_off_diag

    def _init_dictionary(self, zdim: int) -> torch.Tensor:
        """Initialize dictionary as a union of orthogonal bases."""
        wd = torch.empty(self._internal_obs_size, zdim)
        curr = 0
        while curr < zdim:
            remaining = zdim - curr
            if self._internal_obs_size <= remaining:
                mat = torch.randn(self._internal_obs_size, self._internal_obs_size)
                q, _ = torch.linalg.qr(mat)
                wd[:, curr:curr + self._internal_obs_size] = q
                curr += self._internal_obs_size
            else:
                mat = torch.randn(self._internal_obs_size, self._internal_obs_size)
                q, _ = torch.linalg.qr(mat)
                wd[:, curr:] = q[:, :remaining]
                curr += remaining
        return wd / torch.norm(wd, dim=0, keepdim=True).clamp(min=1e-8)

    def _expand_sign_split_dictionary(self, encoder_dict: torch.Tensor) -> torch.Tensor:
        """Expand encoder atoms into paired positive/negative decoder atoms."""
        if not self._uses_sign_split_latent:
            return encoder_dict
        return torch.cat([encoder_dict, -encoder_dict], dim=1)

    def _compute_lipschitz_constant(self, wd_init: torch.Tensor) -> float:
        """Estimate L >= spectral_norm(WdᵀWd) with power iteration."""
        with torch.no_grad():
            gram = wd_init.T @ wd_init
            v = torch.randn(gram.shape[0], 1, device=gram.device, dtype=gram.dtype)
            v = v / torch.norm(v).clamp(min=1e-8)
            for _ in range(10):
                v = gram @ v
                v = v / torch.norm(v).clamp(min=1e-8)
            l_computed = torch.norm(gram @ v) / torch.norm(v).clamp(min=1e-8)
            return float(l_computed.item() * 1.05)

    def _build_encoder(self, cfg: Config, internal_obs_size: int, wd_init: torch.Tensor) -> nn.Module:
        """Create encoder according to cfg.MODEL.ENCODER.ENCODER_TYPE."""
        encoder_type = cfg.MODEL.ENCODER.ENCODER_TYPE.lower()
        if encoder_type == "lista":
            l_const = self._compute_lipschitz_constant(wd_init)
            print(f"Initialized LISTA encoder with computed Lipschitz constant L={l_const:.4f}")
            return LISTA(
                cfg,
                internal_obs_size,
                wd_init,
                L_override=l_const,
                dict_param=self.dict,
            )

        if encoder_type == "hyperlista":
            print(f"Initialized HyperLISTA encoder with {cfg.MODEL.TARGET_SIZE} latent dims")
            print(f"  HyperLISTA hyperparameters: c_theta={cfg.MODEL.ENCODER.HYPERLISTA.C_THETA:.4f}, "
                  f"c_beta={cfg.MODEL.ENCODER.HYPERLISTA.C_BETA:.4f}, "
                  f"c_ss={cfg.MODEL.ENCODER.HYPERLISTA.C_SS:.4f}")
            print(f"  Learnable hyperparams: {cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS}")
            return HyperLISTA(cfg, internal_obs_size, self.dict)

        raise ValueError(
            f"Unknown ENCODER_TYPE '{cfg.MODEL.ENCODER.ENCODER_TYPE}'. "
            "Expected one of ['lista', 'hyperlista']."
        )
    
    def _augment_homogeneous(self, x: torch.Tensor) -> torch.Tensor:
        """Augment input with homogeneous coordinate [x, 1]."""
        ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=-1)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations using the configured sparse encoder.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Sparse latent codes of shape [..., target_size]
        """
        if self.use_homogeneous:
            x = self._augment_homogeneous(x)
        return self.encoder(x)

    def encode_with_prior(
        self,
        x: torch.Tensor,
        latent_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode observations with an optional latent warm start."""
        if self.use_homogeneous:
            x = self._augment_homogeneous(x)
        if latent_prior is None:
            return self.encoder(x)
        return self.encoder(x, latent_prior=latent_prior)
    
    def _decode_full(self, y: torch.Tensor) -> torch.Tensor:
        """Decode to full internal representation (includes homogeneous coord if enabled).
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Full decoded output of shape [..., internal_obs_size]
        """
        wd = self.dict / torch.norm(self.dict, dim=1, keepdim=True).clamp(min=1e-4)
        return y @ wd
    
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Decode using normalized dictionary.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        full_output = self._decode_full(y)
        
        if self.use_homogeneous:
            # Strip the homogeneous coordinate, return only physical dimensions
            physical_output = full_output[..., :-1]
        else:
            physical_output = full_output
        
        if self.use_decoder_bias:
            return physical_output + self.decoder_bias
        else:
            return physical_output
    
    def get_homogeneous_coord(self, y: torch.Tensor) -> torch.Tensor:
        """Get the reconstructed homogeneous coordinate ĉ (should be close to 1).
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Homogeneous coordinate of shape [...] (scalar per sample)
        """
        if not self.use_homogeneous:
            raise ValueError("Model not using homogeneous coordinates")
        full_output = self._decode_full(y)
        return full_output[..., -1]
    
    def kmatrix(self) -> torch.Tensor:
        """Get the Koopman matrix.

        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        if self._k_structure == "diagonal":
            return torch.diag(self.kmat_diag)
        elif self._k_structure == "block_diagonal":
            return torch.block_diag(*self._block_diagonal_matrices())
        else:
            return self.kmat

    def step_latent(self, y: torch.Tensor) -> torch.Tensor:
        """Step forward in latent space using Koopman matrix.

        Uses efficient computation for structured K:
          - diagonal: element-wise multiply O(n)
          - block_diagonal: per-block matmul O(n * block_size)
          - dense: full matmul O(n^2)
        """
        if self._k_structure == "diagonal":
            return y * self.kmat_diag
        elif self._k_structure == "block_diagonal":
            parts = []
            offset = 0
            for block_size, block in zip(self._k_block_sizes, self._block_diagonal_matrices()):
                yi = y[..., offset:offset + block_size]
                parts.append(yi @ block)
                offset += block_size
            return torch.cat(parts, dim=-1)
        else:
            return y @ self.kmat

    def residual(self, x: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        """Alignment loss using efficient step_latent (avoids materializing full K)."""
        y = self.encode(x)
        ny = self.encode(nx)
        return torch.norm(self.step_latent(y) - ny, dim=-1)

    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute explicit L1 sparsity loss on latent codes.

        Args:
            x: Observations of shape [..., observation_size]

        Returns:
            Scalar sparsity loss
        """
        z = self.encode(x)
        return torch.norm(z, p=1, dim=-1).mean()

    def _block_energies(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute per-block energies for block_diagonal K.

        Args:
            z: Latents of shape [..., target_size]

        Returns:
            Tensor of shape [..., num_blocks_total] with per-block energies,
            or None if K is not block_diagonal.
        """
        if self._k_structure != "block_diagonal":
            return None
        if not self._k_block_sizes:
            return None
        energies = []
        offset = 0
        for block_size in self._k_block_sizes:
            block_latents = z[..., offset:offset + block_size]
            if self._block_loss_cfg.ENERGY_NORM == "l1":
                block_energy = block_latents.abs().sum(dim=-1, keepdim=True)
            else:
                block_energy = torch.norm(block_latents, p=2, dim=-1, keepdim=True)
            energies.append(block_energy)
            offset += block_size
        return torch.cat(energies, dim=-1)

    def _block_losses_from_z(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute block activation losses from pre-encoded latents."""
        cfg = self._block_loss_cfg
        energies = self._block_energies(z)
        if energies is None:
            zero = torch.tensor(0.0, device=z.device)
            return zero, zero, {
                'block_entropy': zero,
                'block_usage_entropy': zero,
                'block_top1_gap': zero,
            }

        eps = cfg.EPS
        denom = energies.sum(dim=-1, keepdim=True) + eps
        probs = energies / denom

        # Per-sample entropy (lower = more exclusive)
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)

        one_block_loss = torch.tensor(0.0, device=z.device)
        if cfg.ONE_BLOCK_LOSS == "low_entropy":
            one_block_loss = entropy.mean()
        elif cfg.ONE_BLOCK_LOSS == "pairwise_overlap":
            sum_e = energies.sum(dim=-1)
            sum_sq = (energies ** 2).sum(dim=-1)
            denom_pair = (sum_e + eps) ** 2
            one_block_loss = ((sum_e ** 2 - sum_sq) / denom_pair).mean()
        elif cfg.ONE_BLOCK_LOSS == "top1_margin":
            if energies.shape[-1] >= 2:
                top2 = torch.topk(energies, k=2, dim=-1).values
                gap = top2[..., 0] - top2[..., 1]
                one_block_loss = torch.relu(cfg.TOP1_MARGIN - gap).mean()

        # Batch-level usage balance
        q = probs.mean(dim=0)  # [num_blocks]
        usage_entropy = -torch.sum(q * torch.log(q + eps))
        balance_loss = torch.tensor(0.0, device=z.device)
        if cfg.BALANCE_LOSS == "usage_entropy":
            balance_loss = -usage_entropy  # minimize negative entropy => maximize usage entropy
        elif cfg.BALANCE_LOSS == "kl_uniform":
            num_blocks = q.shape[0]
            log_uniform = math.log(1.0 / num_blocks)
            balance_loss = torch.sum(q * (torch.log(q + eps) - log_uniform))

        # Top-1 gap metric for monitoring
        if energies.shape[-1] >= 2:
            top2 = torch.topk(energies, k=2, dim=-1).values
            top1_gap = (top2[..., 0] - top2[..., 1]).mean()
        else:
            top1_gap = torch.tensor(0.0, device=z.device)

        metrics = {
            'block_entropy': entropy.mean(),
            'block_usage_entropy': usage_entropy,
            'block_top1_gap': top1_gap,
        }
        return one_block_loss, balance_loss, metrics

    def homogeneous_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute homogeneous coordinate consistency loss: penalize ĉ ≠ 1.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Scalar loss (mean squared deviation of ĉ from 1)
        """
        if not self.use_homogeneous:
            return torch.tensor(0.0, device=x.device)
        z = self.encode(x)
        c_hat = self.get_homogeneous_coord(z)
        return torch.mean((c_hat - 1.0) ** 2)
    
    def loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
        z_pred: Optional[torch.Tensor] = None,
        z_true: Optional[torch.Tensor] = None,
        reconstruction_error: Optional[Any] = None,
        sparsity_error: Optional[Any] = None,
        sparsity_latent: Optional[torch.Tensor] = None,
        homogeneous_loss: Optional[Any] = None,
        block_losses: Optional[Dict[str, Any]] = None,
        structured_latent: Optional[torch.Tensor] = None,
        temporal_latent_sequence: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Unified loss with LISTA-specific homogeneous/block additions."""
        del structured_latent, temporal_latent_sequence, loss_weights
        total_loss, metrics = super().loss(
            x_pred=x_pred,
            x_true=x_true,
            x0=x0,
            z0=z0,
            z_pred=z_pred,
            z_true=z_true,
            reconstruction_error=reconstruction_error,
            sparsity_error=sparsity_error,
            sparsity_latent=sparsity_latent,
            step=step,
        )
        block_losses = block_losses or {}

        device = x_pred.device
        dtype = x_pred.dtype
        horizon = x_pred.shape[1]
        sequence_term_scale = 1.0 / float(horizon)

        if self._block_loss_cfg.ENABLED:
            one_block_loss = self._to_scalar_tensor(
                block_losses.get("one_block_loss", 0.0),
                device=device,
                dtype=dtype,
                name="block_losses.one_block_loss",
            )
            balance_loss = self._to_scalar_tensor(
                block_losses.get("balance_loss", 0.0),
                device=device,
                dtype=dtype,
                name="block_losses.balance_loss",
            )
            one_block_loss = one_block_loss * sequence_term_scale
            balance_loss = balance_loss * sequence_term_scale
            total_loss = (
                total_loss +
                self._block_loss_cfg.ONE_BLOCK_WEIGHT * one_block_loss +
                self._block_loss_cfg.BALANCE_WEIGHT * balance_loss
            )
            metrics.update({
                'block_one_block_loss': one_block_loss.item(),
                'block_balance_loss': balance_loss.item(),
                'block_one_block_weight': self._block_loss_cfg.ONE_BLOCK_WEIGHT,
                'block_balance_weight': self._block_loss_cfg.BALANCE_WEIGHT,
                'block_entropy': self._to_scalar_tensor(
                    block_losses.get("entropy", 0.0),
                    device=device,
                    dtype=dtype,
                    name="block_losses.entropy",
                ).item(),
                'block_usage_entropy': self._to_scalar_tensor(
                    block_losses.get("usage_entropy", 0.0),
                    device=device,
                    dtype=dtype,
                    name="block_losses.usage_entropy",
                ).item(),
                'block_top1_gap': self._to_scalar_tensor(
                    block_losses.get("top1_gap", 0.0),
                    device=device,
                    dtype=dtype,
                    name="block_losses.top1_gap",
                ).item(),
            })

        if self.use_homogeneous:
            if homogeneous_loss is not None:
                homog_loss = self._to_scalar_tensor(
                    homogeneous_loss,
                    device=device,
                    dtype=dtype,
                    name="homogeneous_loss",
                ) * sequence_term_scale
                total_loss = total_loss + self.cfg.MODEL.HOMOGENEOUS_COEFF * homog_loss
                metrics['homogeneous_loss'] = homog_loss.item()
            elif self.cfg.MODEL.HOMOGENEOUS_COEFF != 0.0:
                raise ValueError("homogeneous_loss must be provided when homogeneous coordinates are enabled.")

        soft_block_penalty, soft_block_ratio = self._soft_block_penalty()
        if soft_block_penalty is not None:
            total_loss = total_loss + self._soft_block_cfg.WEIGHT * soft_block_penalty
            metrics.update({
                'soft_block_penalty': float(soft_block_penalty.item()),
                'soft_block_weight': float(self._soft_block_cfg.WEIGHT),
                'soft_block_off_block_ratio': float(soft_block_ratio.item()) if soft_block_ratio is not None else 0.0,
            })

        decoder_coherence_penalty, decoder_coherence_max_offdiag = self._decoder_coherence_penalty()
        if decoder_coherence_penalty is not None:
            decoder_coherence_weight = float(self.cfg.MODEL.DECODER_COHERENCE_WEIGHT)
            total_loss = total_loss + decoder_coherence_weight * decoder_coherence_penalty
            metrics.update({
                'decoder_coherence_penalty': float(decoder_coherence_penalty.item()),
                'decoder_coherence_weight': decoder_coherence_weight,
                'decoder_coherence_max_offdiag': (
                    float(decoder_coherence_max_offdiag.item())
                    if decoder_coherence_max_offdiag is not None
                    else 0.0
                ),
            })

        metrics['loss'] = total_loss.item()
        return total_loss, metrics


class StructuredLISTAKM(LISTAKM):
    """LISTA Koopman Machine with structured latent space for multi-basin dynamics.

    Key differences from LISTAKM:
    - Latent z is partitioned: z^(g) [d_g] + z^(1)...z^(B) [d_b each]
    - Koopman uses separate nn.Parameters per block (no masked single matrix)
    - Block-weighted sparsity loss with near-zero global penalty
    - Exclusivity loss with linear warmup schedule

    The arrowhead Koopman structure enforces:
    - Global dynamics evolve independently
    - Global variables can drive basin variables (forcing)
    - Basin variables evolve within their own block (local linearity)
    - No basin-to-basin interaction

    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space
    """

    def __init__(self, cfg: Config, observation_size: int):
        # Read structured config before calling parent init
        struct_cfg = cfg.MODEL.STRUCTURED
        if cfg.MODEL.SOFT_BLOCK.ENABLED:
            raise ValueError("StructuredLISTAKM does not support soft block dense-K penalties.")
        self.d_global = struct_cfg.D_GLOBAL
        self.num_basins = struct_cfg.NUM_BASINS
        self.d_basin = struct_cfg.D_BASIN
        self.lambda_global = struct_cfg.LAMBDA_GLOBAL
        self.lambda_local = struct_cfg.LAMBDA_LOCAL
        self.lambda_exclusivity = struct_cfg.LAMBDA_EXCLUSIVITY
        self.lambda_entropy = struct_cfg.LAMBDA_ENTROPY
        self.lambda_dominance = struct_cfg.LAMBDA_DOMINANCE
        self.lambda_sparsity = struct_cfg.LAMBDA_SPARSITY
        self.lambda_temporal = struct_cfg.LAMBDA_TEMPORAL
        self.excl_warmup_steps = struct_cfg.EXCL_WARMUP_STEPS

        # Compute and set TARGET_SIZE to match structured dimensions
        # Note: This modifies cfg, which is intentional to ensure consistency
        expected_target_size = self.d_global + self.num_basins * self.d_basin
        if cfg.MODEL.TARGET_SIZE != expected_target_size:
            print(f"[StructuredLISTAKM] Setting TARGET_SIZE to {expected_target_size} "
                  f"(d_g={self.d_global} + {self.num_basins}*d_b={self.d_basin})")
            cfg.MODEL.TARGET_SIZE = expected_target_size

        # Initialize parent (creates self.encoder, self.dict, and K parameters)
        super().__init__(cfg, observation_size)

        # Replace single kmat with block-wise parameters
        del self.kmat  # Remove parent's flat Koopman matrix

        # Block-wise Koopman parameters (no unused entries, optimal memory)
        self.K_global = nn.Parameter(torch.eye(self.d_global))  # [d_g, d_g]
        self.K_coupling = nn.ParameterList([
            nn.Parameter(torch.zeros(self.d_basin, self.d_global))
            for _ in range(self.num_basins)
        ])  # B x [d_b, d_g]
        self.K_basin = nn.ParameterList([
            nn.Parameter(torch.eye(self.d_basin))
            for _ in range(self.num_basins)
        ])  # B x [d_b, d_b]

        # Compute memory savings from block-wise storage
        dense_elements = self.target_size ** 2
        block_elements = (self.d_global ** 2 +
                          self.num_basins * self.d_basin * self.d_global +
                          self.num_basins * self.d_basin ** 2)
        sparsity_pct = 100.0 * (1.0 - block_elements / dense_elements)

        print(f"[StructuredLISTAKM] Initialized with:")
        print(f"  Global block: {self.d_global} dims")
        print(f"  Basin blocks: {self.num_basins} x {self.d_basin} dims")
        print(f"  Total latent: {self.target_size} dims")
        print(f"  Koopman storage: {block_elements:,} params (vs {dense_elements:,} dense, {sparsity_pct:.1f}% sparse)")
        print(f"  λ_global={self.lambda_global}, λ_local={self.lambda_local}, λ_excl={self.lambda_exclusivity}")
        print(f"  λ_entropy={self.lambda_entropy}, λ_dominance={self.lambda_dominance}, λ_sparsity={self.lambda_sparsity}")
        print(f"  λ_temporal={self.lambda_temporal}")
        print(f"  Exclusivity/sparsity warmup: {self.excl_warmup_steps} steps")

    def kmatrix(self) -> torch.Tensor:
        """Assemble full Koopman matrix from block parameters.

        NOTE: This creates a dense N×N matrix. Use only for eigenvalue monitoring
        (in torch.no_grad() blocks). For dynamics computation, use step_latent()
        which operates directly on blocks without assembling the full matrix.

        Returns:
            Koopman matrix of shape [target_size, target_size] with arrowhead structure
        """
        N = self.d_global + self.num_basins * self.d_basin
        device = self.K_global.device
        dtype = self.K_global.dtype
        K = torch.zeros(N, N, device=device, dtype=dtype)

        # Global block (top-left)
        K[:self.d_global, :self.d_global] = self.K_global

        # Basin blocks (diagonal + global coupling)
        for k in range(self.num_basins):
            start = self.d_global + k * self.d_basin
            end = start + self.d_basin
            K[start:end, :self.d_global] = self.K_coupling[k]  # Global-to-basin coupling
            K[start:end, start:end] = self.K_basin[k]  # Basin self-dynamics

        return K

    def _stack_koopman_blocks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack Koopman block parameters for efficient batched computation.

        Returns:
            Tuple of (K_coupling_T [B, d_g, d_b], K_basin_stack [B, d_b, d_b])
        """
        # Stack and transpose coupling: [B, d_b, d_g] -> [B, d_g, d_b]
        K_coupling_T = torch.stack([k.T for k in self.K_coupling])
        # Stack basin blocks: [B, d_b, d_b]
        K_basin_stack = torch.stack(list(self.K_basin))
        return K_coupling_T, K_basin_stack

    def _step_latent_with_blocks(
        self,
        z: torch.Tensor,
        K_coupling_T: torch.Tensor,
        K_basin_stack: torch.Tensor,
    ) -> torch.Tensor:
        """Compute z @ K using pre-stacked block parameters (no dense matrix).

        This is the core block-wise computation:
        - z_g' = z_g @ K_global
        - z_k' = z_g @ K_coupling[k].T + z_k @ K_basin[k]  for each basin k

        Args:
            z: Latent codes [..., target_size]
            K_coupling_T: Pre-stacked coupling matrices [B, d_g, d_b]
            K_basin_stack: Pre-stacked basin matrices [B, d_b, d_b]

        Returns:
            Next latent codes [..., target_size]
        """
        z_g, z_basins = self._partition_latent(z)  # [..., d_g], [..., B, d_b]

        # Global dynamics: z_g' = z_g @ K_global
        z_g_next = z_g @ self.K_global  # [..., d_g]

        # Basin dynamics (vectorized over all basins):
        # Coupling term: z_g @ K_coupling[k].T for each k
        # [..., d_g] @ [B, d_g, d_b] -> [..., B, d_b]
        coupling_term = torch.einsum('...g,bgd->...bd', z_g, K_coupling_T)

        # Self-dynamics term: z_k @ K_basin[k] for each k
        # [..., B, d_b] @ [B, d_b, d_b] -> [..., B, d_b]
        basin_term = torch.einsum('...bd,bde->...be', z_basins, K_basin_stack)

        z_basins_next = coupling_term + basin_term  # [..., B, d_b]

        # Reassemble: flatten basins and concatenate with global
        batch_shape = z.shape[:-1]
        z_basins_flat = z_basins_next.reshape(*batch_shape, self.num_basins * self.d_basin)

        return torch.cat([z_g_next, z_basins_flat], dim=-1)

    def step_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Step forward in latent space using block-wise Koopman computation.

        Avoids assembling the full N×N Koopman matrix by computing directly
        on the block parameters. Memory usage is O(d_g² + B*d_b*d_g + B*d_b²)
        instead of O((d_g + B*d_b)²).

        Args:
            z: Latent codes of shape [..., target_size]

        Returns:
            Next latent codes of shape [..., target_size]
        """
        K_coupling_T, K_basin_stack = self._stack_koopman_blocks()
        return self._step_latent_with_blocks(z, K_coupling_T, K_basin_stack)

    def _partition_latent(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Partition latent codes into global and basin blocks (vectorized).

        Args:
            z: Latent codes of shape [..., target_size]

        Returns:
            Tuple of (z_global [..., d_g], z_basins [..., B, d_b])
            z_basins is a single tensor with basin dimension, not a list.
        """
        z_global = z[..., :self.d_global]  # [..., d_g]
        # Reshape basin portion into [..., num_basins, d_basin] - single tensor view
        z_basins = z[..., self.d_global:].view(*z.shape[:-1], self.num_basins, self.d_basin)
        return z_global, z_basins

    def _structured_sparsity_from_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute block-weighted sparsity loss from pre-encoded latents (vectorized).

        Args:
            z: Latent codes of shape [..., target_size]

        Returns:
            Tuple of (global_sparsity_loss, local_sparsity_loss)
        """
        z_global, z_basins = self._partition_latent(z)  # z_basins: [..., B, d_b]

        # Global: near-zero penalty allows dense activation
        global_loss = torch.norm(z_global, p=1, dim=-1).mean()

        # Local: L1 norm over all basin dimensions (flatten B and d_b)
        # z_basins has shape [..., B, d_b], sum absolute values over last two dims
        local_loss = z_basins.abs().sum(dim=(-2, -1)).mean()

        return global_loss, local_loss

    def _exclusivity_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """Compute mutual exclusivity penalty from pre-encoded latents (vectorized).

        Penalty = (1/(B-1)) * sum_{i<j} ||z^(i)||_2 * ||z^(j)||_2

        Normalized by 1/(B-1) for scale-independence across different num_basins.
        Uses efficient O(B) computation instead of O(B^2) pairwise loop.

        Args:
            z: Latent codes of shape [..., target_size]

        Returns:
            Scalar exclusivity loss
        """
        _, z_basins = self._partition_latent(z)  # z_basins: [..., B, d_b]
        B = self.num_basins

        # Compute L2 norms per basin: [..., B] - fully vectorized
        norms = torch.norm(z_basins, p=2, dim=-1)

        # Efficient pairwise: sum_{i<j} = 0.5 * ((sum norms)^2 - sum(norms^2))
        sum_norms = norms.sum(dim=-1)  # [...]
        sum_sq_norms = (norms ** 2).sum(dim=-1)  # [...]
        pairwise_sum = 0.5 * (sum_norms ** 2 - sum_sq_norms)

        # Normalize by 1/(B-1)
        return pairwise_sum.mean() / max(B - 1, 1)

    def _entropy_exclusivity_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """Compute entropy-based exclusivity penalty from pre-encoded latents.

        Penalizes high entropy of the normalized basin norm distribution.
        Low entropy = one basin dominates. High entropy = multiple basins active.

        Uses softmax over basin norms to get a probability distribution, then
        computes the entropy. Minimizing this encourages single-basin dominance.

        Args:
            z: Latent codes of shape [..., target_size]

        Returns:
            Mean entropy over batch (scalar)
        """
        _, z_basins = self._partition_latent(z)  # z_basins: [..., B, d_b]

        # Compute L2 norms per basin: [..., B]
        norms = torch.norm(z_basins, p=2, dim=-1)

        # Softmax to get probability distribution over basins
        probs = F.softmax(norms, dim=-1)  # [..., B]

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [...]

        return entropy.mean()

    def _dominance_loss_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """Compute top-1 dominance loss from pre-encoded latents.

        Encourages the maximum-norm basin to be significantly larger than others.
        Penalizes the ratio of (sum of non-max norms) / (max norm).

        When ratio is 0, only one basin is active (ideal).
        When ratio is high, multiple basins have comparable activations.

        Args:
            z: Latent codes of shape [..., target_size]

        Returns:
            Mean dominance ratio over batch (scalar)
        """
        _, z_basins = self._partition_latent(z)  # z_basins: [..., B, d_b]

        # Compute L2 norms per basin: [..., B]
        norms = torch.norm(z_basins, p=2, dim=-1)

        # Get max norm and sum of other norms
        max_norm, _ = norms.max(dim=-1, keepdim=True)  # [..., 1]
        other_norms_sum = norms.sum(dim=-1, keepdim=True) - max_norm  # [..., 1]

        # Compute ratio (with epsilon for numerical stability)
        dominance_ratio = other_norms_sum / (max_norm + 1e-8)

        return dominance_ratio.mean()

    def _temporal_consistency_from_z_seq(
        self, z_seq: torch.Tensor
    ) -> torch.Tensor:
        """Compute temporal consistency loss for sequence training.

        Penalizes changes in basin activation pattern within a trajectory window.
        Points in the same trajectory are (almost always) in the same basin, so
        the active basin should remain consistent throughout the sequence.

        Mathematical formulation:
        Given sequence z_0, ..., z_T, compute basin norms n_t^(k) = ||z_t^(k)||_2
        L_temporal = (1/T) Σ_t Σ_k (n_t^(k) - n_0^(k))²

        Args:
            z_seq: Latent codes of shape [batch_size, seq_len, target_size]

        Returns:
            Scalar temporal consistency loss
        """
        batch_size, seq_len, _ = z_seq.shape

        # Partition into basins: [batch, seq_len, B, d_basin]
        z_flat = z_seq.reshape(batch_size * seq_len, -1)
        _, z_basins_flat = self._partition_latent(z_flat)  # [batch*seq, B, d_b]
        z_basins = z_basins_flat.reshape(batch_size, seq_len, self.num_basins, self.d_basin)

        # Compute basin norms: [batch, seq_len, B]
        basin_norms = torch.norm(z_basins, p=2, dim=-1)

        # Reference norms from t=0: [batch, 1, B]
        ref_norms = basin_norms[:, 0:1, :]

        # Temporal consistency: penalize deviation from initial basin pattern
        # [batch, seq_len, B] -> scalar
        temporal_loss = torch.mean((basin_norms - ref_norms) ** 2)

        return temporal_loss

    def get_temporal_weight(self, step: int) -> float:
        """Get current temporal consistency weight based on linear warmup schedule.

        Uses the same warmup schedule as exclusivity (excl_warmup_steps).

        Args:
            step: Current training step

        Returns:
            Current temporal coefficient (0 to lambda_temporal)
        """
        if self.excl_warmup_steps <= 0:
            return self.lambda_temporal
        progress = min(1.0, step / self.excl_warmup_steps)
        return progress * self.lambda_temporal

    def structured_sparsity_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute block-weighted sparsity loss (convenience wrapper).

        Args:
            x: Observations of shape [..., observation_size]

        Returns:
            Tuple of (global_sparsity_loss, local_sparsity_loss)
        """
        return self._structured_sparsity_from_z(self.encode(x))

    def exclusivity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mutual exclusivity penalty (convenience wrapper).

        Args:
            x: Observations of shape [..., observation_size]

        Returns:
            Scalar exclusivity loss
        """
        return self._exclusivity_from_z(self.encode(x))

    def get_exclusivity_weight(self, step: int) -> float:
        """Get current exclusivity weight based on linear warmup schedule.

        Args:
            step: Current training step

        Returns:
            Current exclusivity coefficient (0 to lambda_exclusivity)
        """
        if self.excl_warmup_steps <= 0:
            return self.lambda_exclusivity
        progress = min(1.0, step / self.excl_warmup_steps)
        return progress * self.lambda_exclusivity

    def get_sparsity_weight(self, step: int) -> float:
        """Get current sparsity weight based on linear warmup schedule.

        Uses the same warmup schedule as exclusivity (excl_warmup_steps).

        Args:
            step: Current training step

        Returns:
            Current sparsity coefficient (0 to lambda_sparsity)
        """
        if self.excl_warmup_steps <= 0:
            return self.lambda_sparsity
        progress = min(1.0, step / self.excl_warmup_steps)
        return progress * self.lambda_sparsity

    def get_entropy_weight(self, step: int) -> float:
        """Get current entropy exclusivity weight based on linear warmup schedule.

        Uses the same warmup schedule as exclusivity (excl_warmup_steps).

        Args:
            step: Current training step

        Returns:
            Current entropy coefficient (0 to lambda_entropy)
        """
        if self.excl_warmup_steps <= 0:
            return self.lambda_entropy
        progress = min(1.0, step / self.excl_warmup_steps)
        return progress * self.lambda_entropy

    def get_dominance_weight(self, step: int) -> float:
        """Get current dominance loss weight based on linear warmup schedule.

        Uses the same warmup schedule as exclusivity (excl_warmup_steps).

        Args:
            step: Current training step

        Returns:
            Current dominance coefficient (0 to lambda_dominance)
        """
        if self.excl_warmup_steps <= 0:
            return self.lambda_dominance
        progress = min(1.0, step / self.excl_warmup_steps)
        return progress * self.lambda_dominance

    def loss(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
        z_pred: Optional[torch.Tensor] = None,
        z_true: Optional[torch.Tensor] = None,
        reconstruction_error: Optional[Any] = None,
        sparsity_error: Optional[Any] = None,
        sparsity_latent: Optional[torch.Tensor] = None,
        homogeneous_loss: Optional[Any] = None,
        block_losses: Optional[Dict[str, Any]] = None,
        structured_latent: Optional[torch.Tensor] = None,
        temporal_latent_sequence: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Unified structured loss consuming explicit precomputed tensors only."""
        del block_losses
        if sparsity_error is None:
            sparsity_error = torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)
        device = x_pred.device
        dtype = x_pred.dtype
        horizon = x_pred.shape[1]
        sequence_term_scale = 1.0 / float(horizon)

        total_loss, metrics = self._aggregate_losses_from_tensors(
            x_pred=x_pred,
            x_true=x_true,
            x0=x0,
            z0=z0,
            z_pred=z_pred,
            z_true=z_true,
            reconstruction_error=reconstruction_error,
            sparsity_error=sparsity_error,
            sparsity_latent=sparsity_latent,
            step=step,
        )

        z_for_structured = structured_latent if structured_latent is not None else z_pred
        if z_for_structured is None:
            raise ValueError("StructuredLISTAKM.loss requires structured_latent (or z_pred).")
        if z_for_structured.ndim == 3:
            z_for_structured_flat = z_for_structured.reshape(-1, z_for_structured.shape[-1])
        elif z_for_structured.ndim == 2:
            z_for_structured_flat = z_for_structured
        else:
            raise ValueError("structured_latent must have shape [B, H, Dz] or [N, Dz]")

        global_sparsity_loss, local_sparsity_loss = self._structured_sparsity_from_z(z_for_structured_flat)
        excl_loss = self._exclusivity_from_z(z_for_structured_flat)
        entropy_loss = self._entropy_exclusivity_from_z(z_for_structured_flat)
        dominance_loss = self._dominance_loss_from_z(z_for_structured_flat)
        sparsity_loss = torch.norm(z_for_structured_flat, p=1, dim=-1).mean()

        z_for_temporal = temporal_latent_sequence
        if z_for_temporal is None and z0 is not None and z_pred is not None:
            z_for_temporal = torch.cat([z0.unsqueeze(1), z_pred], dim=1)
        if z_for_temporal is None:
            temporal_loss = torch.zeros((), device=device, dtype=dtype)
        else:
            temporal_loss = self._temporal_consistency_from_z_seq(z_for_temporal)

        global_sparsity_loss = global_sparsity_loss * sequence_term_scale
        local_sparsity_loss = local_sparsity_loss * sequence_term_scale
        excl_loss = excl_loss * sequence_term_scale
        entropy_loss = entropy_loss * sequence_term_scale
        dominance_loss = dominance_loss * sequence_term_scale
        sparsity_loss = sparsity_loss * sequence_term_scale
        temporal_loss = temporal_loss * sequence_term_scale

        loss_weights = loss_weights or {}
        excl_weight = float(loss_weights.get("exclusivity", self.get_exclusivity_weight(step)))
        entropy_weight = float(loss_weights.get("entropy", self.get_entropy_weight(step)))
        dominance_weight = float(loss_weights.get("dominance", self.get_dominance_weight(step)))
        sparsity_weight = float(loss_weights.get("sparsity", self.get_sparsity_weight(step)))
        temporal_weight = float(loss_weights.get("temporal", self.get_temporal_weight(step)))

        total_loss = (
            total_loss +
            self.lambda_global * global_sparsity_loss +
            self.lambda_local * local_sparsity_loss +
            excl_weight * excl_loss +
            entropy_weight * entropy_loss +
            dominance_weight * dominance_loss +
            sparsity_weight * sparsity_loss +
            temporal_weight * temporal_loss
        )

        if self.use_homogeneous:
            if homogeneous_loss is None and self.cfg.MODEL.HOMOGENEOUS_COEFF != 0.0:
                raise ValueError("homogeneous_loss must be provided when homogeneous coordinates are enabled.")
            if homogeneous_loss is not None:
                homog_loss = self._to_scalar_tensor(
                    homogeneous_loss,
                    device=device,
                    dtype=dtype,
                    name="homogeneous_loss",
                ) * sequence_term_scale
                total_loss = total_loss + self.cfg.MODEL.HOMOGENEOUS_COEFF * homog_loss
                metrics['homogeneous_loss'] = homog_loss.item()

        with torch.no_grad():
            _, z_basins = self._partition_latent(z_for_structured_flat)
            basin_norms = torch.norm(z_basins, p=2, dim=-1).mean(dim=0)
            active_basins = (basin_norms > 1e-4).sum().item()

        metrics.update({
            'global_sparsity_loss': global_sparsity_loss.item(),
            'local_sparsity_loss': local_sparsity_loss.item(),
            'exclusivity_loss': excl_loss.item(),
            'exclusivity_weight': excl_weight,
            'entropy_loss': entropy_loss.item(),
            'entropy_weight': entropy_weight,
            'dominance_loss': dominance_loss.item(),
            'dominance_weight': dominance_weight,
            'sparsity_loss': sparsity_loss.item(),
            'sparsity_weight': sparsity_weight,
            'temporal_loss': temporal_loss.item(),
            'temporal_weight': temporal_weight,
            'active_basins': active_basins,
        })
        metrics['loss'] = total_loss.item()
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------


_MODEL_REGISTRY = {
    "GenericKM": GenericKM,
    "SparseKM": GenericKM,  # Same as GenericKM, configured via sparsity coeff
    "LISTAKM": LISTAKM,
    "StructuredLISTAKM": StructuredLISTAKM,
}


def make_model(cfg: Config, observation_size: int) -> KoopmanMachine:
    """Factory function to create model from configuration.
    
    Args:
        cfg: Configuration object with MODEL.MODEL_NAME specifying the model type
        observation_size: Dimension of the observation space
        
    Returns:
        KoopmanMachine instance
        
    Raises:
        ValueError: If MODEL_NAME is not in registry
    """
    model_name = cfg.MODEL.MODEL_NAME
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_name](cfg, observation_size)
