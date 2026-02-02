"""
PyTorch implementation of Koopman Autoencoder models.

This module provides:
- MLPCoder: Multi-layer perceptron for encoding/decoding
- LISTA: Learned Iterative Soft-Thresholding Algorithm for sparse coding
- KoopmanMachine: Abstract base class for Koopman operator learning
- GenericKM: Standard Koopman autoencoder with MLP encoder
- LISTAKM: Koopman machine with LISTA sparse encoder
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from skae.config import Config

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


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
    
    def __init__(self, cfg: Config, xdim: int, Wd_init: torch.Tensor, L_override: Optional[float] = None):
        super().__init__()
        self.cfg = cfg
        self.xdim = xdim
        self.zdim = cfg.MODEL.TARGET_SIZE
        self.num_loops = cfg.MODEL.ENCODER.LISTA.NUM_LOOPS
        self.alpha = cfg.MODEL.ENCODER.LISTA.ALPHA
        self.L = L_override if L_override is not None else cfg.MODEL.ENCODER.LISTA.L
        self.use_linear_encode = cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER
        
        assert Wd_init.shape == (xdim, self.zdim), \
            f"Wd_init shape {Wd_init.shape} doesn't match expected ({xdim}, {self.zdim})"
        
        if self.use_linear_encode:
            use_bias = cfg.MODEL.ENCODER.USE_BIAS
            self.We = nn.Linear(xdim, self.zdim, bias=use_bias)
            # Initialize as (1/L) * Wd^T
            with torch.no_grad():
                self.We.weight.copy_((1.0 / self.L) * Wd_init.T)  # [zdim, xdim]
                if use_bias:
                    self.We.bias.zero_()
        else:
            self.We = MLPCoder(
                input_size=xdim,
                target_size=self.zdim,
                hidden_layers=cfg.MODEL.ENCODER.LAYERS,
                use_bias=cfg.MODEL.ENCODER.USE_BIAS,
                last_relu=cfg.MODEL.ENCODER.LAST_RELU,
                activation=cfg.MODEL.ENCODER.ACTIVATION,
            )
        
        S_init = torch.eye(self.zdim) - (1.0 / self.L) * (Wd_init.T @ Wd_init)
        self.S = nn.Parameter(S_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: iterative soft-thresholding.
        
        Args:
            x: Input tensor of shape [..., xdim]
            
        Returns:
            Sparse codes of shape [..., zdim]
        """
        # Initial encoding
        nonsparse_code = self.We(x)
        
        # Initialize with soft-thresholding of initial encoding
        z = shrink(nonsparse_code, self.alpha / self.L)
        
        # Iterative refinement
        for _ in range(self.num_loops):
            z = shrink(z @ self.S + nonsparse_code, self.alpha / self.L)
        
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
        if hypercfg.LEARN_HYPERPARAMS:
            self.c_theta = nn.Parameter(torch.tensor([hypercfg.C_THETA]))
            self.c_beta = nn.Parameter(torch.tensor([hypercfg.C_BETA]))
            self.c_ss = nn.Parameter(torch.tensor([hypercfg.C_SS]))
        else:
            self.register_buffer('c_theta', torch.tensor([hypercfg.C_THETA]))
            self.register_buffer('c_beta', torch.tensor([hypercfg.C_BETA]))
            self.register_buffer('c_ss', torch.tensor([hypercfg.C_SS]))
        
        self.use_ss = hypercfg.USE_SUPPORT_SELECTION
        self.use_momentum = hypercfg.USE_MOMENTUM
        self.mag_ratio = hypercfg.MAG_RATIO
        
        # Precompute pseudo-inverse for error approximation (updated when dict changes)
        # TODO: Consider more robust cache invalidation - data_ptr may give false positives
        # after optimizer steps if memory is reused. For now this is acceptable since we
        # detach the pinv anyway and it's only used for error estimation.
        self._cached_D_pinv = None
        self._cached_D_hash = None
    
    def _get_D_pinv(self, D: torch.Tensor) -> torch.Tensor:
        """Get pseudo-inverse of D, with caching for efficiency.
        
        The pseudo-inverse is detached to prevent gradients flowing through it,
        as it's only used for error estimation (not as part of the main computation).
        
        Args:
            D: Dictionary matrix [xdim, zdim]
            
        Returns:
            Pseudo-inverse of D [zdim, xdim]
        """
        D_hash = D.data_ptr()
        if self._cached_D_pinv is None or self._cached_D_hash != D_hash:
            self._cached_D_pinv = torch.linalg.pinv(D).detach()
            self._cached_D_hash = D_hash
        return self._cached_D_pinv
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        z = self._shrink(c, self.c_theta * gamma)  # Initial thresholding
        z_prev = torch.zeros_like(z)
        
        # Get pseudo-inverse for error approximation
        D_pinv = self._get_D_pinv(D)
        
        # Initial error estimate for support selection
        if self.use_ss:
            initial_error = torch.norm(x @ D_pinv.T, p=1, dim=-1, keepdim=True) + 1e-8
        
        # Unrolled iterations
        for k in range(self.num_loops):
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
    
    def koopman_ode_func(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ODE function for continuous-time Koopman dynamics: dz/dt = K @ z.
        
        Args:
            t: Time (scalar, unused but required by odeint)
            z: Latent state of shape [..., target_size]
            
        Returns:
            Time derivative dz/dt of shape [..., target_size]
        """
        kmat = self.kmatrix()
        # For linear dynamics: dz/dt = K @ z
        return z @ kmat
    
    def integrate_latent_ode(
        self, 
        z0: torch.Tensor, 
        t_span: torch.Tensor,
        method: str = 'dopri5'
    ) -> torch.Tensor:
        """Integrate Koopman dynamics from z0 over time points in t_span.
        
        Args:
            z0: Initial latent state of shape [batch_size, target_size]
            t_span: Time points of shape [num_steps+1] starting from 0
            method: Integration method ('dopri5' for adaptive RK, 'rk4' for fixed-step RK4)
            
        Returns:
            Latent trajectory of shape [num_steps+1, batch_size, target_size]
        """
        # Print integration method on first call
        if not hasattr(self, '_printed_ode_method'):
            if HAS_TORCHDIFFEQ:
                print(f"Using torchdiffeq with method '{method}' for ODE integration")
            else:
                print("Using manual RK4 for ODE integration (torchdiffeq not available)")
            self._printed_ode_method = True
        
        if HAS_TORCHDIFFEQ:
            # Use torchdiffeq for adaptive integration
            z_traj = odeint(
                self.koopman_ode_func,
                z0,
                t_span,
                method=method,
                rtol=1e-5,
                atol=1e-7,
            )
            return z_traj
        else:
            # Fallback: fixed-step RK4 (more accurate than Euler)
            return self._integrate_rk4_fallback(z0, t_span)
    
    def _integrate_rk4_fallback(
        self, 
        z0: torch.Tensor, 
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """Fallback RK4 integration when torchdiffeq is not available.
        
        Implements classic 4th-order Runge-Kutta method.
        
        Args:
            z0: Initial latent state [batch_size, target_size]
            t_span: Time points [num_steps+1]
            
        Returns:
            Latent trajectory [num_steps+1, batch_size, target_size]
        """
        z_list = [z0]
        z = z0
        for i in range(len(t_span) - 1):
            t = t_span[i]
            dt = t_span[i+1] - t_span[i]
            
            # RK4 stages
            k1 = self.koopman_ode_func(t, z)
            k2 = self.koopman_ode_func(t + 0.5 * dt, z + 0.5 * dt * k1)
            k3 = self.koopman_ode_func(t + 0.5 * dt, z + 0.5 * dt * k2)
            k4 = self.koopman_ode_func(t + dt, z + dt * k3)
            
            # Update state
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            z_list.append(z)
        
        return torch.stack(z_list, dim=0)
    
    def rollout_sequence_ode(
        self,
        x0: torch.Tensor,
        num_steps: int,
        dt: float,
    ) -> torch.Tensor:
        """Rollout a sequence using ODE integration of Koopman dynamics.
        
        Args:
            x0: Initial observations of shape [batch_size, observation_size]
            num_steps: Number of steps to roll out
            dt: Time step between observations
            
        Returns:
            Predicted trajectory of shape [num_steps+1, batch_size, observation_size]
            Includes x0 at index 0
        """
        # Encode initial state
        z0 = self.encode(x0)  # [batch_size, target_size]
        
        # Create time span
        t_span = torch.arange(num_steps + 1, dtype=torch.float32, device=x0.device) * dt
        
        # Integrate latent dynamics
        z_traj = self.integrate_latent_ode(z0, t_span)  # [num_steps+1, batch_size, target_size]
        
        # Decode all latent states
        # Need to reshape for decoding: [num_steps+1 * batch_size, target_size]
        num_times, batch_size, target_size = z_traj.shape
        z_flat = z_traj.reshape(num_times * batch_size, target_size)
        x_flat = self.decode(z_flat)  # [num_times * batch_size, observation_size]
        x_traj = x_flat.reshape(num_times, batch_size, self.observation_size)
        
        return x_traj
    
    def loss(
        self,
        x: torch.Tensor,
        nx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and metrics (single-step version, for backward compatibility).
        
        Args:
            x: Current states of shape [batch_size, observation_size]
            nx: Next states of shape [batch_size, observation_size]
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Linear prediction loss
        kmat = self.kmatrix()
        prediction = self.decode(self.encode(x) @ kmat)
        prediction_loss = torch.norm(prediction - nx, dim=-1).mean()
        
        # Linear dynamics alignment loss
        residual_loss = self.residual(x, nx).mean()
        
        # Reconstruction loss
        reconst_loss = torch.norm(x - self.reconstruction(x), dim=-1).mean()
        reconst_loss += torch.norm(nx - self.reconstruction(nx), dim=-1).mean()
        
        # Sparsity loss
        sparsity_loss = self.sparsity_loss(x)
        sparsity_loss += self.sparsity_loss(nx)
        sparsity_loss *= 0.5
        
        # Koopman matrix eigenvalues
        # MPS doesn't support eigvals, so move to CPU if needed
        with torch.no_grad():
            kmat_device = kmat.device
            if kmat_device.type == 'mps':
                kmat_cpu = kmat.cpu()
                eigvals = torch.linalg.eigvals(kmat_cpu)
            else:
                eigvals = torch.linalg.eigvals(kmat)
            max_eigenvalue = torch.max(eigvals.real)
        
        # Nonzero codes
        with torch.no_grad():
            z = self.encode(x)
            # Use epsilon threshold to count near-zero values as zero
            # (important for LISTA which can produce tiny non-zero values)
            num_nonzero_codes = (z.abs() > 1e-6).float().sum(dim=-1).mean()
            sparsity_ratio = 1.0 - num_nonzero_codes / self.target_size
        
        # Total weighted loss
        total_loss = (
            self.cfg.MODEL.RES_COEFF * residual_loss +
            self.cfg.MODEL.RECONST_COEFF * reconst_loss +
            self.cfg.MODEL.PRED_COEFF * prediction_loss +
            self.cfg.MODEL.SPARSITY_COEFF * sparsity_loss
        )
        
        metrics = {
            'loss': total_loss.item(),
            'residual_loss': residual_loss.item(),
            'reconst_loss': reconst_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'A_max_eigenvalue': max_eigenvalue.item(),
            'sparsity_ratio': sparsity_ratio.item(),
        }
        
        return total_loss, metrics
    
    def loss_sequence(
        self,
        x_seq: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute sequence-based loss using ODE integration.
        
        This implements the continuous-time training scheme:
        1. Encode all states in sequence: z_i = φ(x_i)
        2. Integrate dz/dt = Kz from z_0 to get predicted latents ẑ_i
        3. Decode both: x̃_i = ψ(z_i), x̂_i = ψ(ẑ_i)
        4. Compute alignment, reconstruction, and prediction losses
        
        Args:
            x_seq: Sequence of states with shape [batch_size, seq_len, observation_size]
                   Includes x_t, x_{t+1}, ..., x_{t+T}
            dt: Time step between consecutive observations
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        batch_size, seq_len, obs_size = x_seq.shape
        
        # 1. Encode each state in the sequence
        # Flatten for encoding: [batch_size * seq_len, obs_size]
        x_flat = x_seq.reshape(batch_size * seq_len, obs_size)
        z_flat = self.encode(x_flat)  # [batch_size * seq_len, target_size]
        z_seq = z_flat.reshape(batch_size, seq_len, self.target_size)
        
        # 2. Integrate Koopman dynamics from initial state
        x0 = x_seq[:, 0, :]  # [batch_size, obs_size]
        z0 = z_seq[:, 0, :]  # [batch_size, target_size]
        
        # Create time span for integration
        t_span = torch.arange(seq_len, dtype=torch.float32, device=x_seq.device) * dt
        
        # Integrate: z_hat has shape [seq_len, batch_size, target_size]
        z_hat_traj = self.integrate_latent_ode(z0, t_span)
        
        # Transpose to [batch_size, seq_len, target_size]
        z_hat_seq = z_hat_traj.transpose(0, 1)
        
        # 3. Decode both encoded and advanced latents
        # Reconstruction from encoded latents
        x_tilde = self.decode(z_flat).reshape(batch_size, seq_len, obs_size)
        
        # Prediction from ODE-advanced latents
        z_hat_flat = z_hat_seq.reshape(batch_size * seq_len, self.target_size)
        x_hat_flat = self.decode(z_hat_flat)
        x_hat_seq = x_hat_flat.reshape(batch_size, seq_len, obs_size)
        
        # 4. Compute losses
        
        # Alignment loss: sum over sequence (excluding initial state)
        # |ẑ_{t+i} - z_{t+i}|^2 for i = 1..T
        alignment_loss = torch.norm(
            z_hat_seq[:, 1:, :] - z_seq[:, 1:, :], 
            dim=-1
        ).pow(2).sum(dim=1).mean()
        
        # Reconstruction loss: sum over entire sequence including initial
        # |x_{t+i} - x̃_{t+i}|^2 for i = 0..T
        reconst_loss = torch.norm(
            x_seq - x_tilde,
            dim=-1
        ).pow(2).sum(dim=1).mean()
        
        # Prediction loss: sum over sequence (excluding initial state)
        # |x_{t+i} - x̂_{t+i}|^2 for i = 1..T
        prediction_loss = torch.norm(
            x_seq[:, 1:, :] - x_hat_seq[:, 1:, :],
            dim=-1
        ).pow(2).sum(dim=1).mean()
        
        # Sparsity loss: L1 on latents averaged over sequence
        sparsity_loss = torch.norm(z_seq, p=1, dim=-1).mean()
        
        # Metrics for monitoring
        with torch.no_grad():
            kmat = self.kmatrix()
            # MPS doesn't support eigvals, so move to CPU if needed
            kmat_device = kmat.device
            if kmat_device.type == 'mps':
                kmat_cpu = kmat.cpu()
                eigvals = torch.linalg.eigvals(kmat_cpu)
            else:
                eigvals = torch.linalg.eigvals(kmat)
            max_eigenvalue = torch.max(eigvals.real)
            
            # Use epsilon threshold to count near-zero values as zero
            # (important for LISTA which can produce tiny non-zero values)
            num_nonzero_codes = (z_seq.abs() > 1e-4).float().sum(dim=-1).mean()
            sparsity_ratio = 1.0 - num_nonzero_codes / self.target_size
        
        # Total weighted loss
        total_loss = (
            self.cfg.MODEL.RES_COEFF * alignment_loss +
            self.cfg.MODEL.RECONST_COEFF * reconst_loss +
            self.cfg.MODEL.PRED_COEFF * prediction_loss +
            self.cfg.MODEL.SPARSITY_COEFF * sparsity_loss
        )
        
        metrics = {
            'loss': total_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'reconst_loss': reconst_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'A_max_eigenvalue': max_eigenvalue.item(),
            'sparsity_ratio': sparsity_ratio.item(),
        }
        
        return total_loss, metrics


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
    """Koopman Machine with LISTA sparse encoder.
    
    Uses the Learned Iterative Soft-Thresholding Algorithm (LISTA) for sparse
    encoding. The decoder uses a normalized dictionary.
    
    Supports homogeneous coordinates: when enabled, input x is augmented to [x, 1],
    allowing the dictionary to learn an implicit bias through an extra dimension.
    The decoder outputs [x̂, ĉ] internally, with ĉ penalized to stay close to 1.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space (physical, without homogeneous)
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__(cfg, observation_size)
        
        # Homogeneous coordinates: augment input with constant 1
        self.use_homogeneous = cfg.MODEL.USE_HOMOGENEOUS
        self._internal_obs_size = observation_size + 1 if self.use_homogeneous else observation_size
        
        # Initialize dictionary with unit-norm columns, using orthogonal initialization
        # Shape [internal_obs_size, zdim]
        # Since zdim > internal_obs_size (overcomplete), we create a union of orthogonal bases
        Wd_init = torch.empty(self._internal_obs_size, cfg.MODEL.TARGET_SIZE)
        
        # Fill Wd_init with chunks of orthogonal matrices
        # We perform QR decomposition on random matrices to get orthogonal bases
        curr_idx = 0
        while curr_idx < cfg.MODEL.TARGET_SIZE:
            remaining = cfg.MODEL.TARGET_SIZE - curr_idx
            # Generate a square matrix of size max(dim, dim) to get a full basis
            dim = max(self._internal_obs_size, remaining)
            # Create random matrix
            mat = torch.randn(dim, dim)
            # QR decomposition gives orthogonal Q
            q, r = torch.linalg.qr(mat)
            # Take the first 'remaining' columns, or as many as we can fit
            chunk_size = min(remaining, self._internal_obs_size)
            
            # If internal_obs_size < remaining, we take the top internal_obs_size rows of Q
            # which are still orthogonal-ish but we need to select carefully.
            # Actually, standard practice for overcomplete dictionary:
            # Just fill with independent random vectors and normalize is 'ok',
            # BUT to be 'orthogonal', we want blocks of orthogonal bases.
            
            # Let's generate a random orthogonal matrix of size [internal_obs_size, internal_obs_size]
            # and append it. Repeat until full.
            if self._internal_obs_size <= remaining:
                # We can fit a full orthogonal basis
                mat = torch.randn(self._internal_obs_size, self._internal_obs_size)
                q, _ = torch.linalg.qr(mat)
                Wd_init[:, curr_idx:curr_idx+self._internal_obs_size] = q
                curr_idx += self._internal_obs_size
            else:
                # Fill the rest with part of an orthogonal basis
                mat = torch.randn(self._internal_obs_size, self._internal_obs_size)
                q, _ = torch.linalg.qr(mat)
                Wd_init[:, curr_idx:] = q[:, :remaining]
                curr_idx += remaining

        # Ensure unit norm (QR produces unit norm columns, but let's be safe)
        Wd_init = Wd_init / torch.norm(Wd_init, dim=0, keepdim=True)
        
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
        
        # Compute Lipschitz constant L for this dictionary
        # L >= max eigenvalue of (Wd^T @ Wd)
        with torch.no_grad():
            gram = Wd_init.T @ Wd_init
            # Power iteration to estimate spectral norm (largest eigenvalue)
            v = torch.randn(cfg.MODEL.TARGET_SIZE, 1)
            v = v / torch.norm(v)
            for _ in range(10):
                v = gram @ v
                v = v / torch.norm(v)
            L_computed = torch.norm(gram @ v) / torch.norm(v)
            # Add a small safety margin
            L = L_computed.item() * 1.05
            
        print(f"Initialized LISTA with computed Lipschitz constant L={L:.4f}")
        if self.use_homogeneous:
            print(f"  Using homogeneous coordinates: input {observation_size} -> internal {self._internal_obs_size}")
        
        # LISTA encoder (uses internal_obs_size)
        self.lista = LISTA(cfg, self._internal_obs_size, Wd_init, L_override=L)
        
        # Koopman matrix (learnable)
        self.kmat = nn.Parameter(torch.eye(cfg.MODEL.TARGET_SIZE))
    
    def _augment_homogeneous(self, x: torch.Tensor) -> torch.Tensor:
        """Augment input with homogeneous coordinate [x, 1]."""
        ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=-1)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations using LISTA.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Sparse latent codes of shape [..., target_size]
        """
        if self.use_homogeneous:
            x = self._augment_homogeneous(x)
        return self.lista(x)
    
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
        return self.kmat
    
    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity loss weighted by LISTA alpha.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Scalar sparsity loss
        """
        z = self.encode(x)
        return self.cfg.MODEL.ENCODER.LISTA.ALPHA * torch.norm(z, p=1, dim=-1).mean()
    
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
        x: torch.Tensor,
        nx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and metrics, including homogeneous consistency.
        
        Args:
            x: Current states of shape [batch_size, observation_size]
            nx: Next states of shape [batch_size, observation_size]
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Get base loss and metrics from parent class
        total_loss, metrics = super().loss(x, nx)
        
        # Add homogeneous consistency loss if enabled
        if self.use_homogeneous:
            homog_loss = self.homogeneous_loss(x) + self.homogeneous_loss(nx)
            homog_loss *= 0.5  # Average over x and nx
            
            total_loss = total_loss + self.cfg.MODEL.HOMOGENEOUS_COEFF * homog_loss
            metrics['homogeneous_loss'] = homog_loss.item()
            metrics['loss'] = total_loss.item()
        
        return total_loss, metrics


class HyperLISTAKM(KoopmanMachine):
    """Koopman Machine with HyperLISTA sparse encoder.
    
    Key differences from LISTAKM:
    1. Encoder weights are analytically derived from decoder dictionary
    2. Only 3 learnable hyperparameters (c_theta, c_beta, c_ss)
    3. Instance-adaptive threshold, momentum, and support selection
    4. Gradients flow from Koopman loss through encoder to dictionary
    
    This enables learning a dictionary that is jointly optimized for both
    sparse coding quality and Koopman dynamics prediction.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space (physical, without homogeneous)
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__(cfg, observation_size)
        
        # Homogeneous coordinates support
        self.use_homogeneous = cfg.MODEL.USE_HOMOGENEOUS
        self._internal_obs_size = observation_size + 1 if self.use_homogeneous else observation_size
        
        # Initialize dictionary with orthogonal columns
        Wd_init = self._init_dictionary(cfg.MODEL.TARGET_SIZE)
        
        # Dictionary parameter [zdim, internal_obs_size] - shared with encoder
        self.dict = nn.Parameter(Wd_init.T)
        
        # HyperLISTA encoder (references self.dict)
        self.hyperlista = HyperLISTA(cfg, self._internal_obs_size, self.dict)
        
        # Koopman matrix
        self.kmat = nn.Parameter(torch.eye(cfg.MODEL.TARGET_SIZE))
        
        print(f"Initialized HyperLISTAKM with {cfg.MODEL.TARGET_SIZE} latent dims")
        print(f"  HyperLISTA hyperparameters: c_theta={cfg.MODEL.ENCODER.HYPERLISTA.C_THETA:.4f}, "
              f"c_beta={cfg.MODEL.ENCODER.HYPERLISTA.C_BETA:.4f}, c_ss={cfg.MODEL.ENCODER.HYPERLISTA.C_SS:.4f}")
        print(f"  Learnable hyperparams: {cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS}")
        if self.use_homogeneous:
            print(f"  Using homogeneous coordinates: input {observation_size} -> internal {self._internal_obs_size}")
    
    def _init_dictionary(self, zdim: int) -> torch.Tensor:
        """Initialize dictionary with union of orthogonal bases.
        
        Since zdim > internal_obs_size (overcomplete), we create a dictionary
        by concatenating multiple orthogonal bases.
        
        Args:
            zdim: Target latent dimension
            
        Returns:
            Initialized dictionary [internal_obs_size, zdim] with unit-norm columns
        """
        Wd = torch.empty(self._internal_obs_size, zdim)
        curr = 0
        while curr < zdim:
            remaining = zdim - curr
            if self._internal_obs_size <= remaining:
                # We can fit a full orthogonal basis
                mat = torch.randn(self._internal_obs_size, self._internal_obs_size)
                q, _ = torch.linalg.qr(mat)
                Wd[:, curr:curr+self._internal_obs_size] = q
                curr += self._internal_obs_size
            else:
                # Fill the rest with part of an orthogonal basis
                mat = torch.randn(self._internal_obs_size, self._internal_obs_size)
                q, _ = torch.linalg.qr(mat)
                Wd[:, curr:] = q[:, :remaining]
                curr += remaining
        return Wd / torch.norm(Wd, dim=0, keepdim=True)
    
    def _augment_homogeneous(self, x: torch.Tensor) -> torch.Tensor:
        """Augment input with homogeneous coordinate [x, 1]."""
        ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=-1)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations using HyperLISTA.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Sparse latent codes of shape [..., target_size]
        """
        if self.use_homogeneous:
            x = self._augment_homogeneous(x)
        return self.hyperlista(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode using normalized dictionary.
        
        Args:
            z: Latent codes of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        D = self.dict / torch.norm(self.dict, dim=1, keepdim=True).clamp(min=1e-6)
        full_output = z @ D
        if self.use_homogeneous:
            return full_output[..., :-1]
        return full_output
    
    def _decode_full(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to full internal representation (includes homogeneous coord if enabled).
        
        Args:
            z: Latent codes of shape [..., target_size]
            
        Returns:
            Full decoded output of shape [..., internal_obs_size]
        """
        D = self.dict / torch.norm(self.dict, dim=1, keepdim=True).clamp(min=1e-6)
        return z @ D
    
    def get_homogeneous_coord(self, z: torch.Tensor) -> torch.Tensor:
        """Get the reconstructed homogeneous coordinate ĉ (should be close to 1).
        
        Args:
            z: Latent codes of shape [..., target_size]
            
        Returns:
            Homogeneous coordinate of shape [...] (scalar per sample)
        """
        if not self.use_homogeneous:
            raise ValueError("Model not using homogeneous coordinates")
        full_output = self._decode_full(z)
        return full_output[..., -1]
    
    def kmatrix(self) -> torch.Tensor:
        """Get the Koopman matrix.
        
        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        return self.kmat
    
    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity loss on latent codes.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Scalar sparsity loss
        """
        z = self.encode(x)
        return torch.norm(z, p=1, dim=-1).mean()
    
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
        x: torch.Tensor,
        nx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and metrics, including homogeneous consistency.
        
        Args:
            x: Current states of shape [batch_size, observation_size]
            nx: Next states of shape [batch_size, observation_size]
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Get base loss and metrics from parent class
        total_loss, metrics = super().loss(x, nx)
        
        # Add homogeneous consistency loss if enabled
        if self.use_homogeneous:
            homog_loss = self.homogeneous_loss(x) + self.homogeneous_loss(nx)
            homog_loss *= 0.5  # Average over x and nx
            
            total_loss = total_loss + self.cfg.MODEL.HOMOGENEOUS_COEFF * homog_loss
            metrics['homogeneous_loss'] = homog_loss.item()
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

        # Initialize parent (creates self.lista, self.dict, self.kmat)
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

    def koopman_ode_func(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ODE function for continuous-time Koopman dynamics: dz/dt = K @ z.

        Uses block-wise computation. Note: For ODE integration, prefer
        integrate_latent_ode() which stacks blocks once for efficiency.

        Args:
            t: Time (scalar, unused but required by odeint)
            z: Latent state of shape [..., target_size]

        Returns:
            Time derivative dz/dt of shape [..., target_size]
        """
        return self.step_latent(z)

    def integrate_latent_ode(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        method: str = 'dopri5'
    ) -> torch.Tensor:
        """Integrate Koopman dynamics using efficient block-wise computation.

        Stacks block parameters once at the start of integration, avoiding
        repeated stacking during ODE solver iterations.

        Args:
            z0: Initial latent state [batch_size, target_size]
            t_span: Time points [num_steps+1] starting from 0
            method: Integration method ('dopri5' for adaptive, 'rk4' for fixed-step)

        Returns:
            Latent trajectory [num_steps+1, batch_size, target_size]
        """
        # Print integration method on first call
        if not hasattr(self, '_printed_ode_method'):
            if HAS_TORCHDIFFEQ:
                print(f"Using torchdiffeq with method '{method}' for ODE integration")
            else:
                print("Using manual RK4 for ODE integration (torchdiffeq not available)")
            self._printed_ode_method = True

        # Stack blocks once for the entire integration
        K_coupling_T, K_basin_stack = self._stack_koopman_blocks()

        # Create closure with pre-stacked blocks
        def block_ode_func(t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            return self._step_latent_with_blocks(z, K_coupling_T, K_basin_stack)

        if HAS_TORCHDIFFEQ:
            z_traj = odeint(
                block_ode_func,
                z0,
                t_span,
                method=method,
                rtol=1e-5,
                atol=1e-7,
            )
            return z_traj
        else:
            # Fallback: fixed-step RK4 with block-wise computation
            return self._integrate_rk4_with_blocks(z0, t_span, block_ode_func)

    def _integrate_rk4_with_blocks(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        ode_func,
    ) -> torch.Tensor:
        """RK4 integration using pre-stacked block ODE function.

        Args:
            z0: Initial latent state [batch_size, target_size]
            t_span: Time points [num_steps+1]
            ode_func: ODE function that uses pre-stacked blocks

        Returns:
            Latent trajectory [num_steps+1, batch_size, target_size]
        """
        z_list = [z0]
        z = z0
        for i in range(len(t_span) - 1):
            t = t_span[i]
            dt = t_span[i+1] - t_span[i]

            # RK4 stages using block-wise ode_func
            k1 = ode_func(t, z)
            k2 = ode_func(t + 0.5 * dt, z + 0.5 * dt * k1)
            k3 = ode_func(t + 0.5 * dt, z + 0.5 * dt * k2)
            k4 = ode_func(t + dt, z + dt * k3)

            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            z_list.append(z)

        return torch.stack(z_list, dim=0)

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
        alpha = self.cfg.MODEL.ENCODER.LISTA.ALPHA

        # Global: near-zero penalty allows dense activation
        global_loss = alpha * torch.norm(z_global, p=1, dim=-1).mean()

        # Local: L1 norm over all basin dimensions (flatten B and d_b)
        # z_basins has shape [..., B, d_b], sum absolute values over last two dims
        local_loss = alpha * z_basins.abs().sum(dim=(-2, -1)).mean()

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
        x: torch.Tensor,
        nx: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss with structured sparsity and exclusivity.

        Uses block-wise Koopman computation to avoid assembling the full N×N matrix.

        Args:
            x: Current states of shape [batch_size, observation_size]
            nx: Next states of shape [batch_size, observation_size]
            step: Current training step (for exclusivity warmup)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Encode once for efficiency (LISTA forward pass is expensive)
        z = self.encode(x)
        z_nx = self.encode(nx)

        # Stack Koopman blocks once for this forward pass
        K_coupling_T, K_basin_stack = self._stack_koopman_blocks()

        # Compute z_next = z @ K using block-wise operation (no dense matrix)
        z_next = self._step_latent_with_blocks(z, K_coupling_T, K_basin_stack)

        # Decode for reconstruction and prediction
        x_recon = self.decode(z)
        nx_recon = self.decode(z_nx)
        prediction = self.decode(z_next)

        # Linear prediction loss
        prediction_loss = torch.norm(prediction - nx, dim=-1).mean()

        # Linear dynamics alignment loss: ||z @ K - z_nx||
        residual_loss = torch.norm(z_next - z_nx, dim=-1).mean()

        # Reconstruction loss
        reconst_loss = torch.norm(x - x_recon, dim=-1).mean()
        reconst_loss += torch.norm(nx - nx_recon, dim=-1).mean()

        # Structured sparsity from pre-encoded z (no redundant encoding)
        global_sparse, local_sparse = self._structured_sparsity_from_z(z)
        global_sparse_nx, local_sparse_nx = self._structured_sparsity_from_z(z_nx)
        global_sparsity_loss = 0.5 * (global_sparse + global_sparse_nx)
        local_sparsity_loss = 0.5 * (local_sparse + local_sparse_nx)

        # Exclusivity loss with warmup (from pre-encoded z)
        excl_weight = self.get_exclusivity_weight(step)
        excl_loss = 0.5 * (self._exclusivity_from_z(z) + self._exclusivity_from_z(z_nx))

        # Entropy-based exclusivity loss with warmup (encourages single dominant basin)
        entropy_weight = self.get_entropy_weight(step)
        entropy_loss = 0.5 * (self._entropy_exclusivity_from_z(z) + self._entropy_exclusivity_from_z(z_nx))

        # Top-1 dominance loss with warmup (penalizes non-max basins)
        dominance_weight = self.get_dominance_weight(step)
        dominance_loss = 0.5 * (self._dominance_loss_from_z(z) + self._dominance_loss_from_z(z_nx))

        # Explicit L1 sparsity loss on full z with warmup
        alpha = self.cfg.MODEL.ENCODER.LISTA.ALPHA
        sparsity_weight = self.get_sparsity_weight(step)
        sparsity_loss = 0.5 * alpha * (
            torch.norm(z, p=1, dim=-1).mean() + torch.norm(z_nx, p=1, dim=-1).mean()
        )

        # Koopman matrix eigenvalues (for monitoring only - assembles dense matrix in no_grad)
        with torch.no_grad():
            kmat = self.kmatrix()
            kmat_device = kmat.device
            if kmat_device.type == 'mps':
                kmat_cpu = kmat.cpu()
                eigvals = torch.linalg.eigvals(kmat_cpu)
            else:
                eigvals = torch.linalg.eigvals(kmat)
            max_eigenvalue = torch.max(eigvals.real)

        # Nonzero codes and per-basin activity (reuse z, no extra encoding)
        with torch.no_grad():
            num_nonzero_codes = (z.abs() > 1e-6).float().sum(dim=-1).mean()
            sparsity_ratio = 1.0 - num_nonzero_codes / self.target_size

            # Per-basin norms (for monitoring exclusivity) - vectorized
            _, z_basins = self._partition_latent(z)  # [batch, B, d_b]
            basin_norms = torch.norm(z_basins, p=2, dim=-1).mean(dim=0)  # [B]
            active_basins = (basin_norms > 1e-4).sum().item()

        # Total weighted loss
        total_loss = (
            self.cfg.MODEL.RES_COEFF * residual_loss +
            self.cfg.MODEL.RECONST_COEFF * reconst_loss +
            self.cfg.MODEL.PRED_COEFF * prediction_loss +
            self.lambda_global * global_sparsity_loss +
            self.lambda_local * local_sparsity_loss +
            excl_weight * excl_loss +
            entropy_weight * entropy_loss +
            dominance_weight * dominance_loss +
            sparsity_weight * sparsity_loss
        )

        # Homogeneous loss if enabled (reuse z, z_nx)
        if self.use_homogeneous:
            c_hat = self.get_homogeneous_coord(z)
            c_hat_nx = self.get_homogeneous_coord(z_nx)
            homog_loss = 0.5 * (torch.mean((c_hat - 1.0) ** 2) + torch.mean((c_hat_nx - 1.0) ** 2))
            total_loss = total_loss + self.cfg.MODEL.HOMOGENEOUS_COEFF * homog_loss

        metrics = {
            'loss': total_loss.item(),
            'residual_loss': residual_loss.item(),
            'reconst_loss': reconst_loss.item(),
            'prediction_loss': prediction_loss.item(),
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
            'A_max_eigenvalue': max_eigenvalue.item(),
            'sparsity_ratio': sparsity_ratio.item(),
            'active_basins': active_basins,
        }

        if self.use_homogeneous:
            metrics['homogeneous_loss'] = homog_loss.item()

        return total_loss, metrics

    def loss_sequence(
        self,
        x_seq: torch.Tensor,
        dt: float,
        step: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute sequence-based loss with structured sparsity and exclusivity.

        This implements sequence training for StructuredLISTAKM:
        1. Encode all states in sequence: z_i = φ(x_i)
        2. Integrate dz/dt = Kz from z_0 to get predicted latents ẑ_i
        3. Compute alignment, reconstruction, prediction, structured sparsity, exclusivity

        Args:
            x_seq: Sequence of states [batch_size, seq_len, observation_size]
            dt: Time step between consecutive observations
            step: Current training step (for exclusivity warmup)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        batch_size, seq_len, obs_size = x_seq.shape

        # 1. Encode each state in the sequence (single batched call)
        x_flat = x_seq.reshape(batch_size * seq_len, obs_size)
        z_flat = self.encode(x_flat)  # [batch_size * seq_len, target_size]
        z_seq = z_flat.reshape(batch_size, seq_len, self.target_size)

        # 2. Integrate Koopman dynamics from initial state
        z0 = z_seq[:, 0, :]  # [batch_size, target_size]
        t_span = torch.arange(seq_len, dtype=torch.float32, device=x_seq.device) * dt
        z_hat_traj = self.integrate_latent_ode(z0, t_span)  # [seq_len, batch_size, target_size]
        z_hat_seq = z_hat_traj.transpose(0, 1)  # [batch_size, seq_len, target_size]

        # 3. Decode both encoded and advanced latents
        x_tilde = self.decode(z_flat).reshape(batch_size, seq_len, obs_size)
        z_hat_flat = z_hat_seq.reshape(batch_size * seq_len, self.target_size)
        x_hat_seq = self.decode(z_hat_flat).reshape(batch_size, seq_len, obs_size)

        # 4. Compute losses

        # Alignment loss: |ẑ_{t+i} - z_{t+i}|^2 for i = 1..T
        alignment_loss = torch.norm(
            z_hat_seq[:, 1:, :] - z_seq[:, 1:, :],
            dim=-1
        ).pow(2).sum(dim=1).mean()

        # Reconstruction loss: |x_{t+i} - x̃_{t+i}|^2 for i = 0..T
        reconst_loss = torch.norm(x_seq - x_tilde, dim=-1).pow(2).sum(dim=1).mean()

        # Prediction loss: |x_{t+i} - x̂_{t+i}|^2 for i = 1..T
        prediction_loss = torch.norm(
            x_seq[:, 1:, :] - x_hat_seq[:, 1:, :],
            dim=-1
        ).pow(2).sum(dim=1).mean()

        # Structured sparsity: average over sequence
        global_sparsity_loss, local_sparsity_loss = self._structured_sparsity_from_z(z_flat)

        # Exclusivity loss with warmup: average over sequence
        excl_weight = self.get_exclusivity_weight(step)
        excl_loss = self._exclusivity_from_z(z_flat)

        # Explicit L1 sparsity loss on full z with warmup
        alpha = self.cfg.MODEL.ENCODER.LISTA.ALPHA
        sparsity_weight = self.get_sparsity_weight(step)
        sparsity_loss = alpha * torch.norm(z_flat, p=1, dim=-1).mean()

        # Temporal consistency loss: penalize basin activation changes within trajectory
        temporal_weight = self.get_temporal_weight(step)
        temporal_loss = self._temporal_consistency_from_z_seq(z_seq)

        # Metrics for monitoring
        with torch.no_grad():
            kmat = self.kmatrix()
            kmat_device = kmat.device
            if kmat_device.type == 'mps':
                kmat_cpu = kmat.cpu()
                eigvals = torch.linalg.eigvals(kmat_cpu)
            else:
                eigvals = torch.linalg.eigvals(kmat)
            max_eigenvalue = torch.max(eigvals.real)

            num_nonzero_codes = (z_seq.abs() > 1e-6).float().sum(dim=-1).mean()
            sparsity_ratio = 1.0 - num_nonzero_codes / self.target_size

            # Per-basin activity (from first timestep for efficiency) - vectorized
            _, z_basins = self._partition_latent(z_seq[:, 0, :])  # [batch, B, d_b]
            basin_norms = torch.norm(z_basins, p=2, dim=-1).mean(dim=0)  # [B]
            active_basins = (basin_norms > 1e-4).sum().item()

        # Total weighted loss
        total_loss = (
            self.cfg.MODEL.RES_COEFF * alignment_loss +
            self.cfg.MODEL.RECONST_COEFF * reconst_loss +
            self.cfg.MODEL.PRED_COEFF * prediction_loss +
            self.lambda_global * global_sparsity_loss +
            self.lambda_local * local_sparsity_loss +
            excl_weight * excl_loss +
            sparsity_weight * sparsity_loss +
            temporal_weight * temporal_loss
        )

        # Homogeneous loss if enabled
        if self.use_homogeneous:
            c_hat = self.get_homogeneous_coord(z_flat)
            homog_loss = torch.mean((c_hat - 1.0) ** 2)
            total_loss = total_loss + self.cfg.MODEL.HOMOGENEOUS_COEFF * homog_loss

        metrics = {
            'loss': total_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'reconst_loss': reconst_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'global_sparsity_loss': global_sparsity_loss.item(),
            'local_sparsity_loss': local_sparsity_loss.item(),
            'exclusivity_loss': excl_loss.item(),
            'exclusivity_weight': excl_weight,
            'sparsity_loss': sparsity_loss.item(),
            'sparsity_weight': sparsity_weight,
            'temporal_loss': temporal_loss.item(),
            'temporal_weight': temporal_weight,
            'A_max_eigenvalue': max_eigenvalue.item(),
            'sparsity_ratio': sparsity_ratio.item(),
            'active_basins': active_basins,
        }

        if self.use_homogeneous:
            metrics['homogeneous_loss'] = homog_loss.item()

        return total_loss, metrics


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------


_MODEL_REGISTRY = {
    "GenericKM": GenericKM,
    "SparseKM": GenericKM,  # Same as GenericKM, configured via sparsity coeff
    "LISTAKM": LISTAKM,
    "HyperLISTAKM": HyperLISTAKM,
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