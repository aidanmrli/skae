# Structured LISTA Koopman Extension (Basin-Aware Dynamics)

This extension adds structured latent space partitioning and constrained Koopman dynamics to support multi-basin dynamical systems where:
- **Global dynamics** (e.g., gravity, conservation) are shared across all regimes
- **Local basin dynamics** are mutually exclusive linear subspaces
- Only one basin is active at any time (winner-take-all via exclusivity loss)

## Design Decisions (Confirmed)

| Decision | Resolution |
|----------|------------|
| Dimension allocation | Equal by default (`d_g = d_b = 8`), exposed as CLI flags |
| Exclusivity scaling | Normalize by `1/(B-1)` |
| Inheritance | Inherit from [LISTAKM](file:///Users/aidanli/Documents/skae/model.py#981-1214) |
| Gradient optimization | Separate `nn.Parameter`s for each Koopman block |
| Global sparsity | Small positive `λ_global = 1e-4`, CLI-configurable |
| Basin collapse prevention | Linear warmup schedule for exclusivity (0 → final) |
| Default basins | `NUM_BASINS = 20` |
| CLI exposure | All structured config params as command-line flags |
| Loss logging | Print all new loss terms (global/local sparsity, exclusivity) |

---

## Proposed Changes

### Configuration

#### [NEW] [config.py](file:///Users/aidanli/Documents/skae/config.py)

Add `StructuredLatentConfig` dataclass:

```python
@dataclass
class StructuredLatentConfig:
    """Configuration for structured latent space partitioning."""
    ENABLED: bool = False
    D_GLOBAL: int = 8              # Dimension of global dynamics block
    NUM_BASINS: int = 20           # Number of basin slots (B)
    D_BASIN: int = 8               # Dimension of each basin block
    LAMBDA_GLOBAL: float = 1e-4    # Sparsity weight for global block
    LAMBDA_LOCAL: float = 1e-3     # Sparsity weight for basin blocks
    LAMBDA_EXCLUSIVITY: float = 1e-2  # Final exclusivity penalty weight
    EXCL_WARMUP_STEPS: int = 1000  # Steps to ramp exclusivity from 0 to final
```

Update [ModelConfig](file:///Users/aidanli/Documents/skae/config.py#270-290) to include this as a nested field.

#### [MODIFY] CLI Arguments in [train.py](file:///Users/aidanli/Documents/skae/train.py)

Add command-line flags for all structured config parameters:

```python
# Structured latent space args
parser.add_argument('--structured', action='store_true', help='Enable structured latent space')
parser.add_argument('--d_global', type=int, default=8, help='Global block dimension')
parser.add_argument('--num_basins', type=int, default=20, help='Number of basin slots')
parser.add_argument('--d_basin', type=int, default=8, help='Per-basin block dimension')
parser.add_argument('--lambda_global', type=float, default=1e-4, help='Global sparsity weight')
parser.add_argument('--lambda_local', type=float, default=1e-3, help='Local sparsity weight')
parser.add_argument('--lambda_exclusivity', type=float, default=1e-2, help='Final exclusivity weight')
parser.add_argument('--excl_warmup_steps', type=int, default=1000, help='Exclusivity warmup steps')
```

---

### Model Architecture

#### [NEW] `StructuredLISTAKM` class in [model.py](file:///Users/aidanli/Documents/skae/model.py)

```python
class StructuredLISTAKM(LISTAKM):
    """LISTA Koopman Machine with structured latent space for multi-basin dynamics.
    
    Key differences from LISTAKM:
    - Latent z is partitioned: z^(g) [d_g] + z^(1)...z^(B) [d_b each]
    - Koopman uses separate nn.Parameters per block (no masked single matrix)
    - Block-weighted sparsity loss with near-zero global penalty
    - Exclusivity loss with linear warmup schedule
    """
```

**Optimized Koopman Blocks (separate `nn.Parameter`s):**

```python
def __init__(self, cfg, observation_size):
    super().__init__(cfg, observation_size)
    d_g, B, d_b = cfg.MODEL.STRUCTURED.D_GLOBAL, ...
    
    # Block-wise Koopman parameters (no unused entries)
    self.K_global = nn.Parameter(torch.eye(d_g))           # [d_g, d_g]
    self.K_coupling = nn.ParameterList([                    # B x [d_b, d_g]
        nn.Parameter(torch.zeros(d_b, d_g)) for _ in range(B)
    ])
    self.K_basin = nn.ParameterList([                       # B x [d_b, d_b]
        nn.Parameter(torch.eye(d_b)) for _ in range(B)
    ])

def kmatrix(self) -> torch.Tensor:
    """Assemble full Koopman matrix from block parameters."""
    N = self.d_global + self.num_basins * self.d_basin
    K = torch.zeros(N, N, device=self.K_global.device)
    
    # Global block
    K[:self.d_global, :self.d_global] = self.K_global
    
    # Basin blocks
    for k in range(self.num_basins):
        start = self.d_global + k * self.d_basin
        end = start + self.d_basin
        K[start:end, :self.d_global] = self.K_coupling[k]  # Coupling
        K[start:end, start:end] = self.K_basin[k]          # Self-dynamics
    
    return K
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| [kmatrix()](file:///Users/aidanli/Documents/skae/model.py#1348-1355) | Assemble full K from block parameters |
| `_partition_latent(z)` | Split z into [(z_global, [z_basin_1, ..., z_basin_B])](file:///Users/aidanli/Documents/skae/train.py#656-859) |
| `structured_sparsity_loss(x)` | Block-weighted L1: λ_g·‖z^(g)‖₁ + λ_l·Σ‖z^(k)‖₁ |
| `exclusivity_loss(x)` | (1/(B-1))·Σᵢ₍ⱼ ‖z^(i)‖₂·‖z^(j)‖₂ |
| [loss(x, nx, step)](file:///Users/aidanli/Documents/skae/model.py#1383-1410) | Override with warmup: λ_excl(step) linear from 0 |

---

### Loss Function Modifications

#### Block-Weighted Sparsity

Replace global L1 with block-aware weighting:

```python
def structured_sparsity_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute block-weighted sparsity loss.
    
    Returns:
        Tuple of (global_sparsity_loss, local_sparsity_loss)
    """
    z = self.encode(x)
    z_g, z_basins = self._partition_latent(z)
    
    # Global: near-zero penalty allows dense activation
    global_loss = torch.norm(z_g, p=1, dim=-1).mean()
    
    # Local: standard sparsity on concatenated basins
    z_local = torch.cat(z_basins, dim=-1)  # [batch, B * d_b]
    local_loss = torch.norm(z_local, p=1, dim=-1).mean()
    
    return global_loss, local_loss
```

#### Block Exclusivity Penalty

```python
def exclusivity_loss(self, x: torch.Tensor) -> torch.Tensor:
    """Compute mutual exclusivity penalty between basin blocks.
    
    Penalty = (1/(B-1)) * sum_{i<j} ||z^(i)||_2 * ||z^(j)||_2
    
    Normalized by 1/(B-1) for scale-independence.
    """
    z = self.encode(x)
    _, z_basins = self._partition_latent(z)
    B = self.num_basins
    
    # Compute L2 norms per basin: [batch, B]
    norms = torch.stack([torch.norm(zb, p=2, dim=-1) for zb in z_basins], dim=-1)
    
    # Efficient pairwise: sum_i sum_{j>i} = 0.5 * (sum_i sum_j - sum_i ||z_i||^2)
    # = 0.5 * ((sum norms)^2 - sum(norms^2))
    sum_norms = norms.sum(dim=-1)  # [batch]
    sum_sq_norms = (norms ** 2).sum(dim=-1)  # [batch]
    pairwise_sum = 0.5 * (sum_norms ** 2 - sum_sq_norms)
    
    # Normalize by 1/(B-1)
    return pairwise_sum.mean() / (B - 1)
```

---

### Factory Update

#### [MODIFY] [model.py](file:///Users/aidanli/Documents/skae/model.py) - [make_model()](file:///Users/aidanli/Documents/skae/model.py#1425-1445) function

Add new model type to factory:

```python
def make_model(cfg: Config, observation_size: int) -> KoopmanMachine:
    name = cfg.MODEL.MODEL_NAME.lower()
    if name in ('structuredlista', 'structured_lista', 'structuredlistakm'):
        return StructuredLISTAKM(cfg, observation_size)
    # ... existing cases
```

---

## Verification Plan

### Automated Tests

Add new test class to [test_model.py](file:///Users/aidanli/Documents/skae/tests/test_model.py):

```python
class TestStructuredLISTAKM:
    """Test structured LISTA Koopman machine."""
    
    def test_mask_shape_and_structure(self):
        """Verify arrowhead mask has correct structure."""
        
    def test_masked_kmatrix_zeros(self):
        """Verify kmatrix() returns zeros where mask is zero."""
        
    def test_latent_partition_shapes(self):
        """Verify _partition_latent returns correct shapes."""
        
    def test_structured_sparsity_loss_separation(self):
        """Verify global and local sparsity computed separately."""
        
    def test_exclusivity_loss_value(self):
        """Verify exclusivity is zero when only one basin active."""
        
    def test_exclusivity_loss_positive(self):
        """Verify exclusivity is positive when multiple basins active."""
        
    def test_gradient_flow_through_mask(self):
        """Verify gradients flow only through unmasked entries."""
        
    def test_full_loss_includes_new_terms(self):
        """Verify loss() includes structured sparsity and exclusivity."""
```

**Run command:**
```bash
cd /Users/aidanli/Documents/skae && python -m pytest tests/test_model.py::TestStructuredLISTAKM -v
```

### Integration Test

Smoke test training on Duffing environment:

```bash
cd /Users/aidanli/Documents/skae && python train.py \
    --model_name StructuredLISTA \
    --env_name duffing \
    --num_steps 500 \
    --target_size 88 \
    --d_global 8 \
    --num_basins 10 \
    --d_basin 8 \
    --lambda_exclusivity 1e-2
```

**Expected behavior:**
- No crashes
- Loss decreases over 500 steps
- Metrics include `global_sparsity_loss`, `local_sparsity_loss`, `exclusivity_loss`

### Manual Verification

1. **Mask visualization**: After training, extract and print `model._koopman_mask` to verify arrowhead structure
2. **Sparsity pattern**: Log per-basin activation norms during training to verify exclusivity emerges
