# Basin Structure Evaluation Plan

This document describes the experimental design for testing whether StructuredLISTAKM learns distinct Koopman dynamics for each basin of attraction.

---

## Preliminary Findings (2026-01-25)

### GenericKM Baseline Analysis

We evaluated whether the baseline GenericKM (MLP encoder) already learns basin-distinguishing representations, before testing StructuredLISTAKM.

**System**: dysts:Duffing (2 basins)
**Model**: GenericKM with 64-dim latent space, trained with generic_sparse config

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Linear Classifier Accuracy** | **92-98%** | MLP latent space CAN distinguish basins |
| Silhouette Score | 0.30 | Moderate natural clustering |
| Adjusted Rand Index | 0.02-0.10 | K-means doesn't align well with GT |
| PCA Variance (2D) | 81% | Latent space is highly compressible |
| Sparsity (fraction < 0.01) | 18% | Lower than expected |

**Key Finding**: The GenericKM (MLP encoder) already learns basin-relevant structure. A simple logistic regression achieves ~92% accuracy classifying basins from latent codes. This sets a **baseline that StructuredLISTAKM must beat**.

### Issue: Basin Imbalance in dysts:Duffing

The dysts:Duffing initial condition distribution is heavily biased toward one basin:
- Observed distribution: 197:3 (basin 0 vs basin 1)
- This makes evaluation metrics unreliable

**Root cause**: The dysts library's default initial conditions don't span both basins equally.

**Workaround**: Use built-in `lyapunov` or `duffing` environments which have better IC coverage.

### Note: LISTA Threshold Sensitivity

LISTA models can achieve the same performance as MLP encoders, but are **highly sensitive to the threshold parameter** (`lista_alpha`). Key considerations:

- If `alpha` is too high: all activations are zeroed out, no signal
- If `alpha` is too low: no sparsity, behaves like dense network
- Optimal `alpha` depends on the data distribution and is system-specific

**Recommendation**: Always run an alpha sweep (0.1, 0.2, 0.3, 0.4, 0.5) when training LISTA on a new system:
```bash
sbatch sweep_lista_alpha.sh  # Sweeps alpha for lyapunov
```

### Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| `evaluate_basin_structure.py` | ✅ Complete | For StructuredLISTAKM with explicit basin blocks |
| `evaluate_latent_basin_clustering.py` | ✅ Complete | For ANY model (GenericKM, LISTAKM, etc.) |
| Basin identification (duffing) | ✅ Complete | Sign of x after long rollout |
| Basin identification (lyapunov) | ✅ Complete | Nearest attractor |
| Basin identification (dysts:Duffing) | ✅ Complete | Sign of x after long rollout |
| Visualizations (PCA, t-SNE, heatmaps) | ✅ Complete | In evaluate_latent_basin_clustering.py |

---

## Experimental Results (2026-01-25)

### Comparative Analysis on Lyapunov System (13 basins)

Three models were trained on the `lyapunov` environment with matched capacity (~60-64 dims):
- **GenericKM**: MLP encoder, 64-dim latent, `generic_sparse` config
- **LISTAKM**: LISTA encoder, 64-dim latent, `lista_nonlinear` config
- **StructuredLISTAKM**: LISTA encoder, 60-dim latent (8 global + 13×4 basin blocks)

#### Latent Space Basin Clustering Results

| Metric | GenericKM | LISTAKM | StructuredLISTAKM |
|--------|-----------|---------|-------------------|
| **Final Pred Error** | **0.149** | 0.522 | 0.298 |
| **Silhouette Score** | 0.992 | 0.990 | 0.990 |
| **Adjusted Rand Index** | 1.000 | 1.000 | 1.000 |
| **K-means Purity** | 1.000 | 1.000 | 1.000 |
| **Linear Classifier Acc** | **88.0%** | 82.0% | 70.0% |
| **Sparsity (< 0.01)** | 33.3% | **72.4%** | 69.3% |
| **L1 Norm** | 13.3 | 2.4 | 2.5 |
| **PCA Variance (2D)** | **96.8%** | 63.7% | 67.5% |

**Key Finding**: All three models achieve near-perfect basin clustering (ARI=1.0, purity=1.0). The lyapunov system has inherently separable basins that all architectures capture. GenericKM achieves lowest prediction error and highest linear classifier accuracy, while LISTA models achieve higher sparsity.

#### StructuredLISTAKM Basin Block Analysis

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Basin Assignment Accuracy** | 37.9% | >80% | ❌ Failed |
| **Temporal Consistency** | 99.7% | >95% | ✅ Pass |
| **Activation Entropy** | 2.56 | <0.5 | ❌ Failed |
| **Within-Basin Similarity** | 99.99% | High | ✅ Pass |
| **Cross-Basin Separation** | 0.62 | High | ⚠️ Moderate |

**Critical Issue: Basin Collapse**

The confusion matrix reveals that model basins are NOT mapping 1:1 to ground-truth basins:
- **Model Basin 6** handles 5 GT attractors (0, 2, 5, 7, 11)
- **Model Basin 0** handles 2 GT attractors (10, 12)
- **Model Basin 8** handles 2 GT attractors (1, 6)
- Only 8 of 13 model basins are actively used

The model learns deterministic basin assignments (each GT basin → one model basin), but multiple GT basins collapse to the same model basin. The exclusivity loss is not preventing this collapse.

#### Recommendations

1. **Increase `lambda_exclusivity`**: Current value (0.001) may be too weak
2. **Longer training**: Current 5000 steps may be insufficient for basin specialization
3. **Reduce `d_basin`**: 4 dims per basin may be too much capacity, allowing multiple basins to share a block
4. **Add basin-specific regularization**: Penalize model basins that handle multiple distinct attractor patterns

### Checkpoints

| Model | Checkpoint Location |
|-------|---------------------|
| GenericKM | `/network/scratch/l/lia/skae/basin_comparison_lyapunov/generic_km/20260125-123035/checkpoint.pt` |
| LISTAKM | `/network/scratch/l/lia/skae/basin_comparison_lyapunov/lista_km/20260125-123036/checkpoint.pt` |
| StructuredLISTAKM | `/network/scratch/l/lia/skae/basin_comparison_lyapunov/structured_lista_km/20260125-123330/checkpoint.pt` |

### Visualizations

All saved to `/network/scratch/l/lia/skae/basin_comparison_lyapunov/<model>/evaluation/`:
- `latent_pca.png` - PCA visualization colored by basin
- `latent_tsne.png` - t-SNE visualization
- `activation_heatmap.png` - Mean activations per basin
- `confusion_matrix.png` - GT basin vs model basin (StructuredLISTAKM only)

---

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

### Script 1: `evaluate_latent_basin_clustering.py` (General Purpose)

Works with ANY Koopman model (GenericKM, LISTAKM, StructuredLISTAKM). Analyzes whether latent space naturally clusters by basin.

```
evaluate_latent_basin_clustering.py
├── BasinLabeledTrajectory (dataclass)
├── BasinLabeledDataset
│   ├── Supports: duffing, lyapunov, dysts:Duffing
│   └── Basin identification via long rollout
├── Metrics
│   ├── compute_sparsity_metrics() - L1 norm, fraction near-zero
│   ├── compute_clustering_metrics() - Silhouette, ARI, K-means purity
│   ├── compute_separability_metrics() - Linear classifier accuracy
│   └── compute_pca_metrics() - Variance explained
├── Visualizations
│   ├── plot_latent_pca() - 2D PCA colored by basin
│   ├── plot_latent_tsne() - t-SNE visualization
│   ├── plot_latent_activation_heatmap() - Mean activations per basin
│   ├── plot_latent_sparsity_distribution() - Sparsity histograms
│   └── plot_phase_portrait_with_latent_coloring() - Phase space + PC1
└── main() CLI
    ├── --checkpoint: path to ANY trained model
    ├── --system: (optional) override system, else use checkpoint's
    └── --output_dir, --num_trajectories, --seed
```

### Script 2: `evaluate_basin_structure.py` (StructuredLISTAKM Only)

Specifically for StructuredLISTAKM with explicit basin blocks. Analyzes basin block activations.

```
evaluate_basin_structure.py
├── BasinActivations (dataclass) - z_global, z_basins, basin_norms
├── BasinStructureAnalyzer
│   ├── compute_basin_activations() - Per-block analysis
│   ├── compute_basin_assignment_accuracy() - With optimal mapping
│   ├── compute_temporal_consistency()
│   └── run_full_analysis() -> AnalysisResults
├── Visualizations
│   ├── plot_phase_portrait_basin_comparison()
│   ├── plot_basin_norm_timeseries() - Per-block norms over time
│   ├── plot_confusion_matrix() - GT basin vs model basin
│   └── plot_activation_distributions()
└── main() CLI (requires StructuredLISTAKM checkpoint)
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
- `scikit-learn`: Clustering metrics, PCA, t-SNE, logistic regression (for evaluate_latent_basin_clustering.py)

---

## Success Criteria

The experiment is successful if:

1. **Duffing**: The 2 ground-truth basins map to 2 distinct (and only 2) active model basins with >90% accuracy
2. **Lyapunov**: Each ground-truth attractor maps to a distinct model basin with >80% accuracy
3. **Temporal Consistency**: >95% of timesteps within a trajectory have the same active basin
4. **Activation Entropy**: Mean entropy < 0.5 (indicating one dominant basin per timestep)

**Updated baseline to beat** (from GenericKM analysis):
- Linear classifier accuracy on GenericKM: ~92%
- StructuredLISTAKM should achieve **significantly higher** accuracy with cleaner separation

If these criteria are not met, investigate:
- Is the model undertrained?
- Are hyperparameters (lambda_exclusivity, num_basins) appropriate?
- Is the exclusivity loss actually encouraging basin specialization?

---

## TODO List

### High Priority (COMPLETED ✅)

- [x] **Train GenericKM on built-in lyapunov** ✅ Complete
  - Checkpoint: `/network/scratch/l/lia/skae/basin_comparison_lyapunov/generic_km/20260125-123035/checkpoint.pt`

- [x] **Train LISTAKM on lyapunov** ✅ Complete
  - Checkpoint: `/network/scratch/l/lia/skae/basin_comparison_lyapunov/lista_km/20260125-123036/checkpoint.pt`

- [x] **Train StructuredLISTAKM on lyapunov** ✅ Complete
  - Checkpoint: `/network/scratch/l/lia/skae/basin_comparison_lyapunov/structured_lista_km/20260125-123330/checkpoint.pt`

- [x] **Run comparative analysis** on all three models ✅ Complete
  - Results in `/network/scratch/l/lia/skae/basin_comparison_lyapunov/<model>/evaluation/`

### High Priority (NEW - Address Basin Collapse)

- [ ] **Increase exclusivity loss** - Train StructuredLISTAKM with higher `lambda_exclusivity`:
  ```bash
  python train.py --config structured_lista --env lyapunov --num_steps 10000 \
      --d_global 8 --num_basins 13 --d_basin 4 \
      --lambda_exclusivity 0.01 --pairwise --device cuda  # 10x higher
  ```

- [ ] **Reduce basin capacity** - Try smaller `d_basin` to force basin specialization:
  ```bash
  python train.py --config structured_lista --env lyapunov --num_steps 10000 \
      --d_global 8 --num_basins 13 --d_basin 2 \
      --lambda_exclusivity 0.001 --pairwise --device cuda  # 2 dims per basin
  ```

- [ ] **Longer training** - Current 5000 steps may be insufficient:
  ```bash
  python train.py --config structured_lista --env lyapunov --num_steps 20000 \
      --d_global 8 --num_basins 13 --d_basin 4 \
      --lambda_exclusivity 0.001 --pairwise --device cuda
  ```

### Medium Priority

- [ ] **Fix basin imbalance for dysts systems** - Options:
  1. Stratified sampling by running long rollouts first to identify basins
  2. Rejection sampling to balance basins
  3. Use built-in environments instead

- [ ] **Add training-time basin metrics** - Track basin separation during training to understand when structure emerges

- [ ] **Test on simpler system (duffing)** - 2 basins should be easier to separate than 13

### Low Priority

- [ ] **Basin transition analysis** - Evaluate model behavior when trajectories cross basin boundaries

- [ ] **Ablation studies**:
  - Effect of `lambda_exclusivity` on basin separation (0.0001, 0.001, 0.01, 0.1)
  - Effect of `num_basins` (over/under-specified: 7 vs 13 vs 20)
  - Effect of warmup schedule (1000 vs 2000 vs 5000 steps)

---

## Quick Start Commands

### Evaluate existing checkpoint (any model)
```bash
python evaluate_latent_basin_clustering.py \
    --checkpoint <path_to_checkpoint.pt> \
    --num_trajectories 100 \
    --output_dir results/latent_clustering/<model_name>
```

### Evaluate StructuredLISTAKM specifically
```bash
python evaluate_basin_structure.py \
    --checkpoint <path_to_structured_checkpoint.pt> \
    --system lyapunov \
    --num_trajectories 100 \
    --output_dir results/basin_structure/<run_name>
```

### Compare outputs
Results are saved to:
- `analysis_results.json` - All metrics in JSON format
- `latent_pca.png` - PCA visualization colored by basin
- `latent_tsne.png` - t-SNE visualization
- `activation_heatmap.png` - Mean activations per basin
- `phase_portrait_latent.png` - Phase space colored by latent PC1
