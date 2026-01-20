# Dysts Integration Plan for SKAE Benchmarks

## Executive Summary

This plan outlines the integration of the **dysts** library (135+ chaotic systems) into the SKAE (Sparse Koopman Autoencoder) codebase as comprehensive benchmarks for evaluating the LISTA-based Koopman forecasting model. The goal is to systematically test the hypothesis that **Structured Group Sparsity (Block Sparsity)** in the LISTA encoder enables the Koopman autoencoder to learn interpretable representations where each basin of attraction corresponds to its own high-dimensional subspace with approximately linear dynamics.

---

## 1. Motivation & Research Hypothesis

### 1.1 Core Hypothesis
The LISTA encoder, by enforcing sparsity through learned iterative soft-thresholding, should naturally discover:
1. **Block-sparse latent codes** where distinct groups of latent dimensions activate for different basins of attraction
2. **Linear dynamics within subspaces** — each basin maps to a high-dimensional subspace where Koopman dynamics are approximately linear
3. **Interpretable structure** — the dictionary atoms should correspond to meaningful dynamical features

### 1.2 Why Dysts?
The dysts library provides:
- **135 continuous-time chaotic systems** with diverse properties
- **Curated parameters** tuned for chaotic behavior
- **Metadata** including Lyapunov exponents, periods, and dimensions
- **Pre-computed trajectories** for fast iteration
- **Standardized API** for trajectory generation

### 1.3 Key Research Questions
1. **Multi-basin systems**: Does block sparsity emerge for systems with multiple attractors (e.g., Chua circuit, coupled oscillators)?
2. **High-dimensional systems**: How does the model scale to systems with d > 3 dimensions?
3. **Chaotic vs. quasi-periodic**: Does sparsity pattern differ for chaotic vs. periodic systems?
4. **Lyapunov spectrum**: Does prediction horizon correlate with maximum Lyapunov exponent?

---

## 2. Integration Architecture

### 2.1 File Structure

```
ksae/
├── dysts/                          # Submodule or installed as dependency
│   └── dysts/
│       ├── flows.py                # 135 dynamical systems
│       ├── base.py                 # DynSys base class
│       └── systems.py              # make_trajectory_ensemble()
├── benchmarks/                     # NEW: Benchmark module
│   ├── __init__.py
│   ├── dysts_adapter.py            # Adapter wrapping dysts for SKAE Env interface
│   ├── system_catalog.py           # System categorization & selection utilities
│   ├── curated_systems.py          # Hand-picked systems for different experiments
│   └── benchmark_runner.py         # Orchestrates large-scale benchmarking
├── config.py                       # Extended with DystsConfig
├── data.py                         # Extended with DystsEnv wrapper
├── evaluation.py                   # Extended benchmark evaluation
└── experiments/                    # NEW: Experiment scripts
    ├── run_dysts_sweep.py          # Sweep over all/subset of dysts systems
    ├── analyze_sparsity_patterns.py # Block sparsity analysis
    └── visualize_basins.py         # Basin/subspace visualization
```

### 2.2 Dependency Management

Add dysts as a dependency in `pyproject.toml`:
```toml
[project]
dependencies = [
    "torch>=2.0",
    "dysts>=0.0.1",  # or git+https://github.com/williamgilpin/dysts
    # ... existing deps
]
```

Or keep as a git submodule (current approach):
```bash
# Already present at dysts/
git submodule add https://github.com/williamgilpin/dysts dysts
```

---

## 3. Implementation Plan

### Phase 1: Core Adapter Layer ✱ PRIORITY

**Goal**: Create a seamless adapter that wraps dysts systems to match the SKAE `Env` interface.

#### 3.1.1 `benchmarks/dysts_adapter.py`

```python
"""Adapter to wrap dysts DynSys objects as SKAE Env instances."""

from typing import Optional
import torch
import numpy as np
from dysts import flows
from dysts.base import DynSys
from dysts.systems import get_attractor_list
from data import Env, integrate_rk4

class DystsEnv(Env):
    """Wrapper that adapts a dysts DynSys to the SKAE Env interface.
    
    Key adaptations:
    1. Uses dysts's default dt and initial conditions
    2. Converts between numpy (dysts) and torch (SKAE)
    3. Implements step() using dysts's rhs() method
    """
    
    def __init__(self, system_name: str, dt_override: Optional[float] = None):
        super().__init__(cfg=None)
        
        # Get the system class and instantiate
        system_class = getattr(flows, system_name)
        self.system: DynSys = system_class()
        self.system_name = system_name
        
        # Use dysts's calibrated dt or override
        self.dt = dt_override if dt_override else self.system.dt
        
        # Cache dimension
        self._dim = self.system.dimension
        
        # Store IC bounds from metadata (or derive from default IC)
        self._ic = torch.tensor(self.system.ic, dtype=torch.float32)
        self._mean = torch.tensor(self.system.mean, dtype=torch.float32)
        self._std = torch.tensor(self.system.std, dtype=torch.float32)
        
    @property
    def observation_size(self) -> int:
        return self._dim
    
    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reset to a random initial condition near the default IC."""
        if rng is None:
            rng = torch.Generator()
        
        # Sample perturbation around default IC
        # Scale by std for appropriate spread
        noise = torch.randn(self._dim, generator=rng) * self._std * 0.2
        return self._ic + noise
    
    def step(self, state: torch.Tensor, action=None) -> torch.Tensor:
        """Advance state by one timestep using dysts's RHS."""
        # Convert to numpy for dysts
        x_np = state.numpy()
        
        # Use dysts's RHS method
        def dynamics_fn(x, u):
            return torch.from_numpy(
                np.array(self.system.rhs(x.numpy(), t=0))
            ).float()
        
        # RK4 integration (reuse existing SKAE function)
        return integrate_rk4(state, None, self.dt, dynamics_fn)
```

#### 3.1.2 Config Extension

Extend `config.py` with dysts configuration:

```python
@dataclass
class DystsConfig:
    """Configuration for dysts-based environments."""
    SYSTEM_NAME: str = "Lorenz"  # Name matching dysts.flows class
    DT_OVERRIDE: Optional[float] = None  # If None, use dysts default
    STANDARDIZE: bool = False  # Whether to standardize trajectories
    RESAMPLE: bool = True  # Use dysts's period-based resampling
    PTS_PER_PERIOD: int = 100  # Points per period if resampling

@dataclass  
class EnvConfig:
    """Environment configuration."""
    ENV_NAME: str = "duffing"  # Legacy environments
    # ... existing fields ...
    DYSTS: DystsConfig = field(default_factory=DystsConfig)
```

#### 3.1.3 Data Module Extension

Extend `data.py`:

```python
# Add to _ENV_REGISTRY or create a factory
def make_dysts_env(cfg: Config) -> DystsEnv:
    """Create a dysts environment from config."""
    return DystsEnv(
        system_name=cfg.ENV.DYSTS.SYSTEM_NAME,
        dt_override=cfg.ENV.DYSTS.DT_OVERRIDE,
    )

def make_env(cfg: Config) -> Env:
    """Factory function extended for dysts support."""
    env_name = cfg.ENV.ENV_NAME
    
    # Check if it's a dysts system
    if env_name.startswith("dysts:"):
        system_name = env_name.split(":")[1]
        cfg.ENV.DYSTS.SYSTEM_NAME = system_name
        return make_dysts_env(cfg)
    
    # Fallback to existing registry
    if env_name not in _ENV_REGISTRY:
        # Try as dysts system name directly
        try:
            return DystsEnv(system_name=env_name)
        except AttributeError:
            raise ValueError(f"Unknown environment '{env_name}'")
    
    return _ENV_REGISTRY[env_name](cfg)
```

---

### Phase 2: System Catalog & Curation

**Goal**: Organize the 135 systems into meaningful categories for structured experimentation.

#### 3.2.1 `benchmarks/system_catalog.py`

```python
"""Categorization and metadata for dysts systems."""

from dysts.systems import get_attractor_list, get_system_data
from dysts import flows
from typing import Dict, List, Set
import json

def get_all_systems() -> List[str]:
    """Get list of all continuous dysts systems."""
    return get_attractor_list("continuous_no_delay")

def get_system_metadata() -> Dict:
    """Get metadata for all systems."""
    return get_system_data("continuous_no_delay")

# System categorization based on dynamics properties
MULTI_BASIN_SYSTEMS = [
    # Systems with multiple fixed points / attractors
    "Chua",           # Scroll attractor with multiple lobes
    "MultiChua",      # Multiple scroll attractors
    "DoubleScroll",   # Two scroll attractors
    "LorenzCoupled",  # Coupled Lorenz systems
    "Duffing",        # Already in SKAE - two centers
    "VanDerPolDuffing",
]

WELL_STUDIED_CHAOTIC = [
    # Canonical chaotic systems for benchmarking
    "Lorenz",
    "Rossler", 
    "Chen",
    "Lu",
    "Thomas",
    "Halvorsen",
    "Burke",
]

QUASI_PERIODIC = [
    # Systems with quasi-periodic behavior
    "Torus",
    "QuasiPeriodicFlow",
]

HIGH_DIMENSIONAL = [
    # Systems with dimension > 3
    "Lorenz96",       # Variable dimension
    "SprottLinz",     # 4D
    "HyperJerk",      # 4D
    "Laser",          # 4D
]

HAMILTONIAN_LIKE = [
    # Conservative or nearly-conservative
    "HenonHeiles",
    "DoublePendulum",
    "SwingingAtwood",
    "ThreeBodyCircular",
]

# Curated test sets
QUICK_TEST = ["Lorenz", "Rossler", "Chen", "Chua"]  # 4 systems for quick validation
STANDARD_BENCHMARK = WELL_STUDIED_CHAOTIC + MULTI_BASIN_SYSTEMS[:3]  # ~10 systems
FULL_BENCHMARK = get_all_systems()  # All 135 systems
```

#### 3.2.2 `benchmarks/curated_systems.py`

```python
"""Hand-picked system sets for specific experiments."""

# For testing block sparsity hypothesis
BLOCK_SPARSITY_EXPERIMENT = {
    "multi_basin": [
        "Chua",            # Multiple scroll regions
        "DoubleScroll",    # Two distinct basins  
        "LorenzCoupled",   # Two coupled attractors
    ],
    "single_basin_control": [
        "Lorenz",          # Single strange attractor
        "Rossler",         # Single attractor
        "Thomas",          # Single attractor
    ],
}

# For testing scaling with dimension
DIMENSION_SCALING_EXPERIMENT = {
    "2D": ["VanDerPol", "Duffing"],
    "3D": ["Lorenz", "Rossler", "Chen"],
    "4D": ["Laser", "HyperJerk"],
    # Note: Lorenz96 can be configured for any dimension
}

# For testing Lyapunov exponent correlation
LYAPUNOV_EXPERIMENT = {
    "high_lyapunov": [],   # Populate from metadata
    "medium_lyapunov": [],
    "low_lyapunov": [],
}
```

---

### Phase 3: Training Integration

**Goal**: Enable training on dysts systems via CLI and programmatic API.

#### 3.3.1 CLI Extension

Modify `train.py` argument parser:

```python
parser.add_argument('--env', type=str, default='duffing',
                    help='Environment name. Use "dysts:SystemName" for dysts systems, '
                         'e.g. "dysts:Lorenz" or "dysts:Chua"')
parser.add_argument('--list-dysts', action='store_true',
                    help='List all available dysts systems and exit')
```

#### 3.3.2 Batch Training Script

`experiments/run_dysts_sweep.py`:

```python
"""Sweep training across multiple dysts systems."""

import argparse
from pathlib import Path
from benchmarks.system_catalog import QUICK_TEST, STANDARD_BENCHMARK, FULL_BENCHMARK
from config import get_config
from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--systems', choices=['quick', 'standard', 'full'], default='quick')
    parser.add_argument('--config', default='lista')
    parser.add_argument('--num_steps', type=int, default=5000)
    parser.add_argument('--output_dir', type=Path, default=Path('runs/dysts_sweep'))
    args = parser.parse_args()
    
    systems = {
        'quick': QUICK_TEST,
        'standard': STANDARD_BENCHMARK,
        'full': FULL_BENCHMARK,
    }[args.systems]
    
    results = {}
    for system_name in systems:
        print(f"\n{'='*60}")
        print(f"Training on: {system_name}")
        print('='*60)
        
        cfg = get_config(args.config)
        cfg.ENV.ENV_NAME = f"dysts:{system_name}"
        cfg.TRAIN.NUM_STEPS = args.num_steps
        
        log_dir = args.output_dir / system_name
        try:
            model = train(cfg, log_dir=str(log_dir))
            results[system_name] = {"status": "success", "log_dir": str(log_dir)}
        except Exception as e:
            results[system_name] = {"status": "failed", "error": str(e)}
    
    # Save summary
    import json
    with open(args.output_dir / 'sweep_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
```

---

### Phase 4: Evaluation & Analysis

**Goal**: Extend evaluation framework for dysts benchmarks and sparsity analysis.

#### 3.4.1 Evaluation Extension

Extend `evaluation.py`:

```python
def evaluate_dysts_benchmark(
    model: KoopmanMachine,
    cfg: Config,
    systems: List[str],
    output_dir: Path,
    device: str = 'cuda',
) -> Dict[str, Dict]:
    """Evaluate model on multiple dysts systems.
    
    Returns:
        Dictionary mapping system_name -> evaluation_metrics
    """
    all_results = {}
    
    for system_name in tqdm(systems, desc="Evaluating systems"):
        # Temporarily modify config
        cfg.ENV.ENV_NAME = f"dysts:{system_name}"
        
        # Run standard evaluation
        eval_settings = EvaluationSettings()
        eval_settings.systems = [system_name]
        
        results = evaluate_model(
            model=model,
            cfg=cfg,
            device=device,
            settings=eval_settings,
            output_dir=output_dir / system_name,
        )
        all_results[system_name] = results
    
    return all_results
```

#### 3.4.2 Sparsity Pattern Analysis

`experiments/analyze_sparsity_patterns.py`:

```python
"""Analyze block sparsity patterns across different systems."""

import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def analyze_latent_codes(model, env, num_samples=1000, seed=42):
    """Analyze sparsity patterns in latent codes."""
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    
    # Sample initial conditions
    x = env.reset(rng)
    # ... generate trajectory
    
    # Encode to latent space
    with torch.no_grad():
        z = model.encode(x)
    
    # Compute sparsity metrics
    sparsity_ratio = (z.abs() < 1e-6).float().mean()
    
    # Identify active dimensions per sample
    active_dims = (z.abs() > 1e-4).float()
    
    # Cluster samples by their active dimension patterns
    # This reveals if different regions of state space activate different subspaces
    kmeans = KMeans(n_clusters=5, random_state=seed)
    clusters = kmeans.fit_predict(active_dims.numpy())
    
    return {
        'sparsity_ratio': sparsity_ratio.item(),
        'mean_active_dims': active_dims.sum(dim=-1).mean().item(),
        'cluster_centers': kmeans.cluster_centers_,
        'cluster_labels': clusters,
    }

def compare_multi_vs_single_basin(results_multi, results_single):
    """Compare sparsity patterns between multi-basin and single-basin systems."""
    # Hypothesis: multi-basin systems should show more distinct clusters
    # (different basins activate different latent subspaces)
    pass
```

#### 3.4.3 Basin/Subspace Visualization

`experiments/visualize_basins.py`:

```python
"""Visualize how different basins map to different latent subspaces."""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import numpy as np

def plot_latent_activation_grid(model, env, grid_lim=2.0, grid_n=20):
    """Plot which latent dimensions activate across state space."""
    # Create grid over state space
    x = torch.linspace(-grid_lim, grid_lim, grid_n)
    y = torch.linspace(-grid_lim, grid_lim, grid_n)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    
    with torch.no_grad():
        z = model.encode(grid)
        # Find top-k active dimensions per point
        top_dims = z.abs().argsort(dim=-1, descending=True)[:, :3]
    
    # Color by dominant active dimension
    colors = top_dims[:, 0].numpy()
    
    plt.figure(figsize=(10, 10))
    plt.scatter(grid[:, 0], grid[:, 1], c=colors, cmap='tab20', s=5)
    plt.colorbar(label='Dominant latent dimension')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Dominant Latent Dimension Across State Space')
    return plt.gcf()
```

---

### Phase 5: Benchmarking Infrastructure

**Goal**: Create infrastructure for reproducible, large-scale benchmarking.

#### 3.5.1 `benchmarks/benchmark_runner.py`

```python
"""Orchestrate large-scale benchmark experiments."""

import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import torch

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    systems: List[str]
    model_config: str  # 'lista', 'generic_sparse', etc.
    num_steps: int = 10000
    target_size: int = 512
    sparsity_coeff: float = 1.0
    seeds: List[int] = (0, 1, 2)  # Multiple seeds for statistics
    
    def hash(self) -> str:
        """Compute hash for caching."""
        return hashlib.md5(json.dumps(asdict(self)).encode()).hexdigest()[:8]

class BenchmarkRunner:
    """Run and manage benchmark experiments."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, config: BenchmarkConfig) -> Path:
        """Run a benchmark and return path to results."""
        run_id = f"{config.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{config.hash()}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Run experiments
        all_results = {}
        for system in config.systems:
            for seed in config.seeds:
                # ... training and evaluation
                pass
        
        # Save results
        with open(run_dir / 'results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return run_dir
```

#### 3.5.2 Aggregation & Reporting

```python
def aggregate_benchmark_results(run_dir: Path) -> pd.DataFrame:
    """Aggregate benchmark results into a DataFrame for analysis."""
    with open(run_dir / 'results.json') as f:
        results = json.load(f)
    
    rows = []
    for system, system_results in results.items():
        for seed, metrics in system_results.items():
            rows.append({
                'system': system,
                'seed': seed,
                'mse_100': metrics.get('mse_100'),
                'mse_1000': metrics.get('mse_1000'),
                'sparsity_ratio': metrics.get('sparsity_ratio'),
                'num_active_dims': metrics.get('num_active_dims'),
            })
    
    return pd.DataFrame(rows)
```

---

## 4. Prioritized Implementation Order

### Sprint 1: Foundation (Week 1)
1. ☐ Create `benchmarks/` directory structure
2. ☐ Implement `DystsEnv` adapter in `benchmarks/dysts_adapter.py`
3. ☐ Extend `config.py` with `DystsConfig`
4. ☐ Modify `data.py` to support dysts environments
5. ☐ Add `--list-dysts` flag to `train.py`
6. ☐ Validate: Train LISTA model on `dysts:Lorenz`

### Sprint 2: Catalog & CLI (Week 1-2)
7. ☐ Create `benchmarks/system_catalog.py` with categorizations
8. ☐ Create `benchmarks/curated_systems.py` for experiments
9. ☐ Extend `train.py` CLI for dysts system specification
10. ☐ Create `experiments/run_dysts_sweep.py`
11. ☐ Validate: Run quick sweep on 4 systems

### Sprint 3: Evaluation (Week 2)
12. ☐ Extend `evaluation.py` with `evaluate_dysts_benchmark()`
13. ☐ Add dysts-specific metrics (per-system Lyapunov, etc.)
14. ☐ Create unified output format for cross-system comparison
15. ☐ Validate: Evaluate sweep results

### Sprint 4: Analysis Tools (Week 2-3)
16. ☐ Implement `analyze_sparsity_patterns.py`
17. ☐ Implement `visualize_basins.py`
18. ☐ Add block sparsity detection metrics
19. ☐ Validate: Analyze multi-basin systems

### Sprint 5: Benchmarking Infrastructure (Week 3)
20. ☐ Implement `BenchmarkRunner` with reproducibility
21. ☐ Add result aggregation and reporting
22. ☐ Create LaTeX table generation for papers
23. ☐ Final validation: Full 135-system benchmark

---

## 5. Technical Considerations

### 5.1 Performance Optimizations

1. **Batched trajectory generation**: Use `make_trajectory_ensemble()` from dysts for parallel trajectory generation
2. **Caching**: Cache generated trajectories to disk for repeated experiments
3. **Mixed precision**: Use `torch.amp` for faster training on large sweeps

### 5.2 Numerical Stability

1. **Standardization**: Dysts provides `mean` and `std` for each system — use these for normalization
2. **dt calibration**: Dysts provides calibrated `dt` values — respect these or document overrides
3. **Trajectory validity**: Check for NaN/Inf during evaluation (some ICs may diverge)

### 5.3 Compatibility Concerns

1. **Delay systems**: Some dysts systems are DDEs (delay differential equations) — initially skip `DynSysDelay` subclasses
2. **Unbounded coordinates**: Some systems like `Lorenz84` have unbounded dimensions — use dysts's postprocessing
3. **Variable dimension**: `Lorenz96` has configurable dimension — handle specially

### 5.4 Suggested System Subsets for Different Goals

| Goal | Subset | # Systems |
|------|--------|-----------|
| Quick validation | `QUICK_TEST` | 4 |
| Paper benchmark | `STANDARD_BENCHMARK` | 10-15 |
| Block sparsity hypothesis | `MULTI_BASIN_SYSTEMS` | 6 |
| Full evaluation | All continuous, no-delay | ~120 |

---

## 6. Expected Outcomes

### 6.1 Quantitative Metrics
- **MSE @ horizon 100, 1000** for each system × rollout mode
- **Sparsity ratio** and **number of active dimensions** per system
- **Prediction horizon** before error exceeds threshold

### 6.2 Qualitative Insights
- **Block structure visualization** showing basin → subspace correspondence
- **Latent code clustering** revealing dynamical regimes
- **Koopman eigenvalue spectra** for learned dynamics

### 6.3 Research Deliverables
- Benchmark comparison table across 10-20 systems
- Analysis of block sparsity for multi-basin vs. single-basin systems
- Guidelines for LISTA hyperparameter selection per system type

---

## 7. Open Questions & Future Work

1. **Hyperparameter transfer**: Can LISTA alpha optimized on one system transfer to similar systems?
2. **Dictionary interpretation**: Can we interpret dictionary atoms as dynamical features?
3. **Continuous-time Koopman**: Should we use continuous-time ODE formulation vs discrete?
4. **Ensemble Koopman**: Can we train a single model across multiple systems?

---

## 8. Quick Start

After implementation, training on a dysts system should be as simple as:

```bash
# Train on Lorenz system
uv run python train.py --config lista --env dysts:Lorenz --num_steps 10000

# Train on Chua circuit (multi-basin)
uv run python train.py --config lista --env dysts:Chua --target_size 1024 --sparsity_coeff 2.0

# Sweep over all canonical chaotic systems
uv run python experiments/run_dysts_sweep.py --systems standard --config lista

# Analyze sparsity patterns
uv run python experiments/analyze_sparsity_patterns.py --run_dir runs/dysts_sweep/ --output analysis/
```

---

*Last updated: 2026-01-19*
*Author: SKAE Research Team*
