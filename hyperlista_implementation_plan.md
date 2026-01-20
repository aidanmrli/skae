# HyperLISTA Integration Plan for Sparse Koopman Autoencoder

## Executive Summary

This document provides a detailed implementation plan for integrating HyperLISTA into the existing SKAE framework. The goal is to replace the current LISTA encoder with a HyperLISTA encoder that:
1. Uses analytically-derived weights from the decoder dictionary D
2. Reduces learnable parameters from O(n² × L) to **3 scalar hyperparameters** (c_beta, c_theta, c_ss)
3. Maintains end-to-end differentiability for joint training with Koopman dynamics

Note that no code should be deleted within the existing files. The existing files should be extended and modified to support the new HyperLISTA encoder.

---

## 1. Architecture Overview

### Current LISTA Architecture (`model.py:121-213`)
```
Input x → We(x) → [LISTA loops: z = shrink(S @ z + c, α/L)] → Sparse code z
                    ↑
            Learnable: We, S (n×n matrix)
```

### Proposed HyperLISTA Architecture
```
Input x → (1/L)D^T @ x → [HyperLISTA loops with momentum & adaptive threshold] → Sparse code z
                          ↑
            Instance-adaptive: θ(x), β(x), p(x) computed from 3 hyperparams (c_theta, c_beta, c_ss)
            Dictionary D is shared with decoder (tied weights)
```

---

## 2. Key Design Decisions

### 2.1 Weight Coupling Strategy
The encoder weight matrix W_e is **analytically derived** from the decoder dictionary D:
```python
W_e = (1/L) * D.T  # Shape: [zdim, xdim]
```

The mutual-inhibition matrix S is also derived:
```python
S = I - (1/L) * D.T @ D  # Shape: [zdim, zdim]
```

**Critical**: These are recomputed at each forward pass (not cached), ensuring gradients flow through D.

### 2.2 Three Hyperparameters

| Parameter | Symbol | Purpose | Typical Range |
|-----------|--------|---------|---------------|
| `c_theta` | c₁ | Threshold scaling | 1e-3 to 1e-2 |
| `c_beta` | c₂ | Momentum coefficient | 1e-3 to 1e-2 |
| `c_ss` | c₃ | Support selection ratio | 0.1 to 1.0 |

### 2.3 Instance-Adaptive Formulas

For each input x at iteration k:
```python
# Threshold: proportional to current residual error
residual = D @ z_k - x
approx_error = torch.norm(D_pinv @ residual, p=1, dim=-1, keepdim=True)
theta_k = c_theta * gamma * approx_error

# Momentum: proportional to estimated support size  
support_size = (z_k.abs() > mag_ratio * z_k.abs().max(dim=-1, keepdim=True)[0]).sum(dim=-1, keepdim=True)
beta_k = c_beta * support_size

# Support selection ratio: based on convergence progress
initial_error = torch.norm(D_pinv @ x, p=1, dim=-1, keepdim=True)
p_k = c_ss * torch.log(initial_error / approx_error).clamp(0, 1)
```

---

## 3. Implementation Plan

### Phase 1: Configuration Extension

**File: `config.py`**

Add new dataclass for HyperLISTA config:

```python
@dataclass
class HyperListaConfig:
    """HyperLISTA encoder-specific configuration."""
    NUM_LOOPS: int = 16              # Number of unrolled iterations
    C_THETA: float = 5e-3            # Threshold scaling hyperparameter
    C_BETA: float = 5e-3             # Momentum hyperparameter
    C_SS: float = 0.5                # Support selection hyperparameter
    USE_SUPPORT_SELECTION: bool = True
    USE_MOMENTUM: bool = True
    MAG_RATIO: float = 0.1           # Threshold for support approximation
    LEARN_HYPERPARAMS: bool = True   # Whether c_theta, c_beta, c_ss are learnable
```

Update `EncoderConfig`:
```python
@dataclass
class EncoderConfig:
    # ... existing fields ...
    LISTA: ListaConfig = field(default_factory=ListaConfig)
    HYPERLISTA: HyperListaConfig = field(default_factory=HyperListaConfig)  # NEW
```

Add config factory function:
```python
def get_train_hyperlista_config() -> Config:
    """Configuration for HyperLISTA-based Sparse KM."""
    cfg = Config()
    cfg.MODEL.MODEL_NAME = "HyperLISTAKM"
    cfg.MODEL.TARGET_SIZE = 2048
    cfg.MODEL.ENCODER.HYPERLISTA.NUM_LOOPS = 16
    cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = 5e-3
    cfg.MODEL.ENCODER.HYPERLISTA.C_BETA = 5e-3
    cfg.MODEL.ENCODER.HYPERLISTA.C_SS = 0.5
    cfg.MODEL.RES_COEFF = 1.0
    cfg.MODEL.RECONST_COEFF = 1.0
    cfg.MODEL.SPARSITY_COEFF = 0.1
    cfg.MODEL.USE_HOMOGENEOUS = True
    return cfg
```

---

### Phase 2: HyperLISTA Encoder Implementation

**File: `model.py`**

```python
class HyperLISTA(nn.Module):
    """HyperLISTA encoder with analytically-derived, instance-adaptive parameters.
    
    Unlike standard LISTA which learns W_e and S, HyperLISTA:
    1. Derives W_e = (1/L) * D.T from the decoder dictionary
    2. Computes S = I - (1/L) * D.T @ D on the fly
    3. Uses instance-adaptive threshold, momentum, and support selection
    4. Has only 3 learnable scalar hyperparameters (c_theta, c_beta, c_ss)
    
    This enables gradient flow from the Koopman loss back to D.
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
        self._cached_D_pinv = None
        self._cached_D_hash = None
    
    def _get_D_pinv(self, D: torch.Tensor) -> torch.Tensor:
        """Get pseudo-inverse of D, with caching for efficiency."""
        # D has shape [xdim, zdim]
        D_hash = D.data_ptr()
        if self._cached_D_pinv is None or self._cached_D_hash != D_hash:
            self._cached_D_pinv = torch.linalg.pinv(D).detach()
            self._cached_D_hash = D_hash
        return self._cached_D_pinv
    
    def _compute_L(self, D: torch.Tensor) -> torch.Tensor:
        """Compute Lipschitz constant L = spectral_norm(D.T @ D)."""
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
                # Estimate support size
                z_abs = z.abs()
                max_mag = z_abs.max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
                support = (z_abs > self.mag_ratio * max_mag).float().sum(dim=-1, keepdim=True)
                beta = self.c_beta * support
                momentum = beta * (z - z_prev)
            else:
                momentum = 0.0
            
            # Pre-threshold state
            z_tilde = z - gamma * grad + momentum
            
            # Adaptive threshold
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
        """Soft thresholding operator."""
        return torch.sign(x) * torch.relu(x.abs() - theta)
    
    def _shrink_ss(self, x: torch.Tensor, theta: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Soft thresholding with support selection."""
        x_abs = x.abs()
        # Find quantile threshold for top-p entries
        # p has shape [..., 1], need to handle per-sample
        batch_shape = x.shape[:-1]
        if len(batch_shape) == 0:
            threshold = torch.quantile(x_abs, 1 - p.item())
        else:
            # Vectorized quantile across batch
            threshold = torch.quantile(x_abs, (1 - p).squeeze(-1), dim=-1)
            if threshold.dim() == 0:
                threshold = threshold.unsqueeze(0)
            threshold = threshold.unsqueeze(-1)
        
        # Bypass shrinkage for entries above both thresholds
        bypass = (x_abs >= threshold) & (x_abs >= theta)
        return torch.where(bypass, x, self._shrink(x, theta))
```

---

### Phase 3: HyperLISTAKM Model Class

**File: `model.py`**

```python
class HyperLISTAKM(KoopmanMachine):
    """Koopman Machine with HyperLISTA sparse encoder.
    
    Key differences from LISTAKM:
    1. Encoder weights are analytically derived from decoder dictionary
    2. Only 3 learnable hyperparameters (c_theta, c_beta, c_ss)
    3. Instance-adaptive threshold, momentum, and support selection
    4. Gradients flow from Koopman loss through encoder to dictionary
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
    
    def _init_dictionary(self, zdim: int) -> torch.Tensor:
        """Initialize dictionary with union of orthogonal bases."""
        Wd = torch.empty(self._internal_obs_size, zdim)
        curr = 0
        while curr < zdim:
            remaining = zdim - curr
            chunk = min(remaining, self._internal_obs_size)
            mat = torch.randn(self._internal_obs_size, self._internal_obs_size)
            q, _ = torch.linalg.qr(mat)
            Wd[:, curr:curr+chunk] = q[:, :chunk]
            curr += chunk
        return Wd / torch.norm(Wd, dim=0, keepdim=True)
    
    def _augment_homogeneous(self, x: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=-1)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_homogeneous:
            x = self._augment_homogeneous(x)
        return self.hyperlista(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        D = self.dict / torch.norm(self.dict, dim=1, keepdim=True).clamp(min=1e-6)
        full_output = z @ D
        if self.use_homogeneous:
            return full_output[..., :-1]
        return full_output
    
    def kmatrix(self) -> torch.Tensor:
        return self.kmat
```

---

### Phase 4: Model Registry Update

**File: `model.py`**

```python
_MODEL_REGISTRY = {
    "GenericKM": GenericKM,
    "SparseKM": GenericKM,
    "LISTAKM": LISTAKM,
    "HyperLISTAKM": HyperLISTAKM,  # NEW
}
```

---

### Phase 5: Grid Search Tuning Script

**File: `tune_hyperlista.py`** (new file)

```python
"""Grid search for optimal HyperLISTA hyperparameters.

Usage:
    python tune_hyperlista.py --env duffing --target_size 1024
"""

import torch
import numpy as np
from config import get_config
from data import make_env, VectorWrapper
from model import make_model

def grid_search(cfg, device='cuda'):
    """Find optimal (c_theta, c_beta, c_ss) via grid search."""
    
    env = make_env(cfg)
    vec_env = VectorWrapper(env, batch_size=512)
    
    # Generate validation data
    rng = torch.Generator().manual_seed(42)
    x = vec_env.reset(rng).to(device)
    nx = vec_env.step(x).to(device)
    
    best_loss = float('inf')
    best_params = None
    
    # Define search grid
    c_theta_range = np.linspace(1e-3, 1e-2, 5)
    c_beta_range = np.linspace(1e-3, 1e-2, 5)
    c_ss_range = np.linspace(0.1, 1.0, 5)
    
    for c_theta in c_theta_range:
        for c_beta in c_beta_range:
            for c_ss in c_ss_range:
                # Set hyperparameters
                cfg.MODEL.ENCODER.HYPERLISTA.C_THETA = c_theta
                cfg.MODEL.ENCODER.HYPERLISTA.C_BETA = c_beta
                cfg.MODEL.ENCODER.HYPERLISTA.C_SS = c_ss
                cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = False
                
                model = make_model(cfg, env.observation_size).to(device)
                
                with torch.no_grad():
                    loss, _ = model.loss(x, nx)
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_params = (c_theta, c_beta, c_ss)
                    print(f"New best: c_theta={c_theta:.4f}, c_beta={c_beta:.4f}, "
                          f"c_ss={c_ss:.4f}, loss={loss.item():.6f}")
    
    return best_params
```

---

## 4. Testing Strategy

### Unit Tests (`tests/test_hyperlista.py`)

```python
def test_hyperlista_forward_shape():
    """Test output shape matches expected zdim."""
    cfg = get_config("hyperlista")
    model = make_model(cfg, observation_size=3)
    x = torch.randn(16, 3)
    z = model.encode(x)
    assert z.shape == (16, cfg.MODEL.TARGET_SIZE)

def test_hyperlista_gradient_flow():
    """Verify gradients flow from loss to dictionary."""
    cfg = get_config("hyperlista")
    model = make_model(cfg, observation_size=3)
    x = torch.randn(8, 3)
    nx = torch.randn(8, 3)
    loss, _ = model.loss(x, nx)
    loss.backward()
    assert model.dict.grad is not None
    assert torch.any(model.dict.grad != 0)

def test_hyperlista_sparsity():
    """Verify outputs are sparse."""
    cfg = get_config("hyperlista")
    model = make_model(cfg, observation_size=3)
    x = torch.randn(16, 3)
    z = model.encode(x)
    nonzero_ratio = (z.abs() > 1e-6).float().mean()
    assert nonzero_ratio < 0.5  # Expect <50% active
```

---

## 5. Training Guidelines

### Initial Training Command
```bash
python train.py \
    --config hyperlista \
    --env duffing \
    --num_steps 20000 \
    --target_size 1024 \
    --sparsity_coeff 0.1
```

### Two-Phase Training Strategy

**Phase 1: Hyperparameter Discovery (Grid Search)**
```bash
python tune_hyperlista.py --env duffing --target_size 1024
# Output: optimal (c_theta, c_beta, c_ss)
```

**Phase 2: End-to-End Training**
Use discovered hyperparameters as initialization, then enable learning:
```python
cfg.MODEL.ENCODER.HYPERLISTA.LEARN_HYPERPARAMS = True
```

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Dictionary collapses to low rank | Add orthogonality regularization: `loss += λ * ‖D^T D - I‖_F` |
| Gradients through D_pinv are unstable | Detach D_pinv computation (only used for error estimation) |
| Koopman dynamics don't transfer | Monitor residual loss; ensure it decreases with training |
| Sparse codes don't separate basins | Add basin-aware loss term based on active support overlap |

---

## 7. File Changes Summary

| File | Changes |
|------|---------|
| `config.py` | Add `HyperListaConfig`, `get_train_hyperlista_config()`, update registry |
| `model.py` | Add `HyperLISTA` class, `HyperLISTAKM` class, update `_MODEL_REGISTRY` |
| `tune_hyperlista.py` | New file for grid search hyperparameter tuning |
| `tests/test_hyperlista.py` | New test file |

---

## 8. Success Criteria

1. **Training Stability**: Loss decreases monotonically without divergence
2. **Sparsity**: Codes have <30% active entries on average
3. **Reconstruction**: MSE comparable to or better than LISTAKM
4. **Prediction**: Koopman dynamics achieve similar horizon performance
5. **Basin Separation**: Different basins activate distinct support patterns
6. **Gradient Flow**: `model.dict.grad` is non-zero after backward pass
