# Basin Structure Evaluation Plan

This document describes the experimental design for testing whether StructuredLISTAKM learns distinct Koopman dynamics for each basin of attraction.

## Hypothesis

The structured latent space partitioning in StructuredLISTAKM should learn to associate different basin blocks with different dynamical basins of attraction. Specifically:

1. **Support-Basin Correspondence**: The activated basin block(s) in latent space should correspond to the ground-truth basin of attraction
2. **Within-Basin Consistency**: Trajectories within the same basin should activate the same model basin block(s)
3. **Cross-Basin Distinctiveness**: Different ground-truth basins should map to different model basin blocks
4. **Temporal Stability**: The active basin block should remain stable as a trajectory evolves within a single basin

---

## Systems Under Evaluation

### Phase 1: Built-in Systems

| System | State Dim | # Basins | Basin Identification Method |
|--------|-----------|----------|----------------------------|
| **Duffing** | 2 | 2 | Sign of position x at equilibrium: basin A (x → +1), basin B (x → -1) |
| **Lyapunov** | 2 | Multiple | Nearest attractor after long rollout; cluster by converged fixed point |

### Phase 2: Dysts Multi-Basin Systems (Future Work)

| System | State Dim | # Basins | Notes |
|--------|-----------|----------|-------|
| **MultiChua** | 3+ | Multiple | Coupled Chua circuits with multiple scrolls |
| **LorenzCoupled** | 6 | Multiple | Two coupled Lorenz systems |
| **RikitakeDynamo** | 3 | 2 | Geomagnetic reversal model with two symmetric attractors |

---

## Data Generation

### Basin-Labeled Trajectory Dataset

For each system, generate a dataset of trajectories with ground-truth basin labels:

```python
@dataclass
class BasinLabeledTrajectory:
    """A trajectory with ground-truth basin labels."""
    states: torch.Tensor          # [T, state_dim] - trajectory states
    basin_labels: torch.Tensor    # [T] - ground-truth basin index at each timestep
    final_basin: int              # Basin determined by long-term convergence
    system_name: str              # e.g., "duffing", "lyapunov"
```

### Basin Identification Methods

#### Duffing Oscillator
```python
def identify_duffing_basin(trajectory: torch.Tensor, long_rollout_steps: int = 5000) -> int:
    """
    Duffing has two stable fixed points at x ≈ ±1.
    Run long rollout and check sign of final x position.

    Returns:
        0 if trajectory converges to x ≈ -1 (left well)
        1 if trajectory converges to x ≈ +1 (right well)
    """
    # Extend trajectory with long rollout
    final_state = rollout(trajectory[-1], steps=long_rollout_steps)
    return 0 if final_state[0] < 0 else 1
```

#### Lyapunov System
```python
def identify_lyapunov_basin(trajectory: torch.Tensor, attractor_positions: torch.Tensor) -> int:
    """
    Lyapunov system has multiple fixed-point attractors.
    Identify basin by nearest attractor after convergence.

    Args:
        trajectory: [T, 2] trajectory
        attractor_positions: [num_attractors, 2] known attractor locations

    Returns:
        Index of nearest attractor (0 to num_attractors-1)
    """
    final_state = rollout(trajectory[-1], steps=long_rollout_steps)
    distances = torch.norm(final_state - attractor_positions, dim=-1)
    return distances.argmin().item()
```

### Dataset Generation Pipeline

```python
def generate_basin_labeled_dataset(
    system: str,
    num_trajectories: int = 100,
    trajectory_length: int = 500,
    long_rollout_steps: int = 5000,
    seed: int = 42,
) -> List[BasinLabeledTrajectory]:
    """
    Generate trajectories with ground-truth basin labels.

    Strategy:
    1. Sample diverse initial conditions across state space
    2. Generate trajectory of length T
    3. Continue rollout to determine final basin
    4. Label entire trajectory with final basin (assumes no transitions)

    For systems where basin boundaries are known, can also compute
    instantaneous basin labels at each timestep.
    """
    pass
```

---

## Model Analysis

### Core Analysis Functions

#### 1. Compute Basin Activations

```python
def compute_basin_activations(
    model: StructuredLISTAKM,
    trajectory: torch.Tensor,  # [T, state_dim]
) -> Dict[str, torch.Tensor]:
    """
    Encode trajectory and compute basin block activations.

    Returns:
        {
            'z': [T, latent_dim],           # Full latent codes
            'z_global': [T, d_global],      # Global block
            'z_basins': [T, B, d_basin],    # Basin blocks
            'basin_norms': [T, B],          # L2 norm of each basin block
            'active_basin': [T],            # argmax basin at each timestep
            'activation_entropy': [T],      # Entropy of normalized basin norms
        }
    """
    with torch.no_grad():
        z = model.encode(trajectory)
        z_global, z_basins = model._partition_latent(z)
        basin_norms = torch.norm(z_basins, p=2, dim=-1)  # [T, B]

        # Softmax for entropy computation
        basin_probs = F.softmax(basin_norms, dim=-1)
        entropy = -torch.sum(basin_probs * torch.log(basin_probs + 1e-8), dim=-1)

        return {
            'z': z,
            'z_global': z_global,
            'z_basins': z_basins,
            'basin_norms': basin_norms,
            'active_basin': basin_norms.argmax(dim=-1),
            'activation_entropy': entropy,
        }
```

#### 2. Compute Consistency Metrics

```python
def compute_consistency_metrics(
    activations: List[Dict],           # Activations for each trajectory
    ground_truth_basins: List[int],    # Ground-truth basin for each trajectory
) -> Dict[str, float]:
    """
    Compute metrics measuring correspondence between model basins and ground-truth.

    Returns:
        {
            'basin_assignment_accuracy': float,  # Does active_basin match ground-truth?
            'temporal_consistency': float,       # % timesteps where active_basin = mode
            'cross_basin_separation': float,     # How distinct are activations across basins?
            'within_basin_similarity': float,    # How similar are activations within same basin?
        }
    """
    pass
```

#### 3. Build Confusion Matrix

```python
def build_basin_confusion_matrix(
    activations: List[Dict],
    ground_truth_basins: List[int],
    num_ground_truth_basins: int,
    num_model_basins: int,
) -> torch.Tensor:
    """
    Build confusion matrix: ground-truth basin vs. predicted model basin.

    Returns:
        [num_ground_truth_basins, num_model_basins] matrix where entry (i, j)
        is the fraction of timesteps from ground-truth basin i where model
        basin j was most active.
    """
    pass
```

---

## Visualizations

### 1. Phase Portrait with Basin Coloring

Plot trajectory in state space with two color channels:
- **Background/outline**: Ground-truth basin color
- **Fill/marker**: Predicted (active) model basin color

```python
def plot_phase_portrait_basin_comparison(
    trajectories: List[BasinLabeledTrajectory],
    activations: List[Dict],
    ground_truth_cmap: str = 'Set1',    # Colormap for ground-truth basins
    predicted_cmap: str = 'Set2',        # Colormap for predicted basins
    output_path: Path = None,
):
    """
    Create phase portrait showing ground-truth vs predicted basin assignments.

    Visual encoding:
    - Each trajectory point is a marker
    - Marker edge color = ground-truth basin
    - Marker face color = predicted (active) model basin
    - If colors match, the model correctly identified the basin

    Alternative: Side-by-side plots
    - Left: trajectory colored by ground-truth basin
    - Right: same trajectory colored by predicted basin
    """
    pass
```

### 2. Basin Norm Time Series

```python
def plot_basin_norm_timeseries(
    activations: Dict,
    ground_truth_basin: int,
    title: str = None,
    output_path: Path = None,
):
    """
    Plot L2 norms of all basin blocks over time for a single trajectory.

    Expected behavior for well-trained model:
    - One basin block should have consistently high norm
    - Other basin blocks should have near-zero norms
    - The dominant basin should correspond to ground-truth

    Visual elements:
    - Line plot with B lines (one per model basin block)
    - Horizontal band or annotation showing ground-truth basin
    - Shaded regions if basin transitions occur
    """
    pass
```

### 3. Confusion Matrix Heatmap

```python
def plot_basin_confusion_matrix(
    confusion_matrix: torch.Tensor,
    ground_truth_labels: List[str],   # e.g., ["Left Well", "Right Well"]
    model_basin_labels: List[str],    # e.g., ["Basin 0", "Basin 1", ...]
    output_path: Path = None,
):
    """
    Heatmap showing correspondence between ground-truth and model basins.

    Ideal result: Permutation matrix (each ground-truth basin maps to
    exactly one model basin, and vice versa for active basins).
    """
    pass
```

### 4. Activation Distribution per Ground-Truth Basin

```python
def plot_activation_distributions(
    activations: List[Dict],
    ground_truth_basins: List[int],
    output_path: Path = None,
):
    """
    For each ground-truth basin, show distribution of model basin activations.

    Format: Grid of histograms or violin plots
    - Rows: ground-truth basins
    - Columns: model basin blocks
    - Values: distribution of L2 norms
    """
    pass
```

---

## Metrics Summary

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **Basin Assignment Accuracy** | Fraction of timesteps where active model basin matches ground-truth | 1.0 |
| **Temporal Consistency** | Fraction of timesteps where active basin = trajectory mode | 1.0 |
| **Within-Basin Similarity** | Average cosine similarity of basin activations within same ground-truth basin | 1.0 |
| **Cross-Basin Separation** | Average L2 distance between basin activations from different ground-truth basins | High |
| **Activation Entropy** | Entropy of normalized basin norms (low = one basin dominates) | Low |
| **Confusion Matrix Sparsity** | Number of nonzero entries in confusion matrix | = num_ground_truth_basins |

---

## Implementation Structure

```
evaluate_basin_structure.py
├── BasinLabeledTrajectory (dataclass)
├── BasinLabeledDataset
│   ├── __init__(system, num_trajectories, ...)
│   ├── generate_trajectories()
│   ├── identify_basins()
│   └── __getitem__, __len__
├── BasinStructureAnalyzer
│   ├── __init__(model, dataset)
│   ├── compute_all_activations()
│   ├── compute_metrics() -> Dict[str, float]
│   ├── build_confusion_matrix() -> Tensor
│   └── run_full_analysis() -> AnalysisResults
├── Visualization functions
│   ├── plot_phase_portrait_basin_comparison()
│   ├── plot_basin_norm_timeseries()
│   ├── plot_basin_confusion_matrix()
│   └── plot_activation_distributions()
└── main() CLI
    ├── --checkpoint: path to trained StructuredLISTAKM
    ├── --system: duffing, lyapunov, or dysts:SystemName
    ├── --num_trajectories: number of test trajectories
    ├── --output_dir: where to save results
    └── --seed: random seed
```

---

## Usage Example

```bash
# Evaluate on Duffing system
python evaluate_basin_structure.py \
    --checkpoint runs/structured_lista/20240115-120000/checkpoint.pt \
    --system duffing \
    --num_trajectories 100 \
    --output_dir results/basin_analysis/duffing

# Evaluate on Lyapunov system
python evaluate_basin_structure.py \
    --checkpoint runs/structured_lista/20240115-120000/checkpoint.pt \
    --system lyapunov \
    --num_trajectories 100 \
    --output_dir results/basin_analysis/lyapunov
```

---

## Future Work

### Basin Transition Analysis

> **Note**: For initial evaluation, we assume trajectories stay within a single basin. Future work should analyze basin transitions.

Two approaches for generating transition data:

1. **Artificial Perturbation**: Add impulse noise to push trajectories across basin boundaries
   ```python
   def perturb_across_boundary(state, boundary_direction, magnitude):
       """Push state across known basin boundary."""
       return state + magnitude * boundary_direction
   ```

2. **Transition Oversampling**: For systems with natural transitions (e.g., stochastic forcing), oversample initial conditions near basin boundaries where transitions are more likely

Metrics for transition analysis:
- **Transition Detection Rate**: Does active basin change when ground-truth basin changes?
- **Transition Delay**: How many timesteps after ground-truth transition does model detect it?
- **False Transition Rate**: Does model incorrectly predict transitions?

### Training-Time Basin Metrics (Optional TODO)

> **Note**: Not implemented initially to avoid slowing training. Could be added as optional evaluation callback.

Track during training:
- Basin confusion matrix at each evaluation step
- When does basin structure emerge? (Early vs late training)
- Does exclusivity warmup schedule affect when basins specialize?

---

## Dependencies

- `torch`: Model inference and tensor operations
- `matplotlib`: All visualizations
- `seaborn`: Heatmaps and statistical plots (optional, enhances matplotlib)
- `numpy`: Numerical utilities
- `tqdm`: Progress bars for trajectory generation

---

## Success Criteria

The experiment is successful if:

1. **Duffing**: The 2 ground-truth basins map to 2 distinct (and only 2) active model basins with >90% accuracy
2. **Lyapunov**: Each ground-truth attractor maps to a distinct model basin with >80% accuracy
3. **Temporal Consistency**: >95% of timesteps within a trajectory have the same active basin
4. **Activation Entropy**: Mean entropy < 0.5 (indicating one dominant basin per timestep)

If these criteria are not met, investigate:
- Is the model undertrained?
- Are hyperparameters (lambda_exclusivity, num_basins) appropriate?
- Is the exclusivity loss actually encouraging basin specialization?
