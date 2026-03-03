# Sparse Koopman Autoencoder (SKAE)

PyTorch-based research codebase for learning Koopman operator representations of nonlinear dynamical systems using autoencoders with sparsity constraints.

## Overview

This repository implements several variants of Koopman autoencoders:

- **GenericKM**: Standard Koopman autoencoder with MLP encoder
- **SparseKM**: Koopman autoencoder with L1 sparsity regularization
- **LISTAKM**: Learned Iterative Soft-Thresholding Algorithm (LISTA) based sparse encoder, with configurable encoder mode (`lista` or `hyperlista`)
- **StructuredLISTAKM**: Basin-aware Koopman with structured latent space for multi-basin systems

## Quick Start

### Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable dependency management.

Installing uv on MacOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install uv for Windows, open PowerShell and run:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Install the project and dependencies**:
```bash
# Clone the repository
git clone <repository-url>
cd skae

# Install from lock file (reproducible, recommended)
uv sync

# Alternative: Install without lock file
uv pip install -e .
```

### Train a Model

```bash
# Train with defaults on the Duffing Oscillator (H=1, former pairwise behavior)
uv run python tools/train.py --config generic_sparse --env duffing --sequence_length 1 --num_steps 20000

# Custom learning rate and latent dimension
uv run python tools/train.py \
  --config generic_sparse \
  --env lyapunov \
  --num_steps 5000 \
  --batch_size 256 \
  --target_size 64 \
  --reconst_coeff 0.02 \
  --pred_coeff 1.0 \
  --sparsity_coeff 0.001 \
  --sequence_length 1 \
  --seed 0 \
  --device cuda

# LISTA with nonlinear pre-activation
uv run python tools/train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --num_steps 20000 \
  --batch_size 256 \
  --target_size 512 \
  --reconst_coeff 1.0 \
  --pred_coeff 10.0 \
  --sparsity_coeff 1.5 \
  --lista_alpha 0.3 \
  --sequence_length 1 \
  --seed 42 \
  --device cuda

# HyperLISTA with custom hyperparameters
uv run python tools/train.py \
  --config hyperlista \
  --env lyapunov \
  --num_steps 3000 \
  --batch_size 256 \
  --target_size 256 \
  --reconst_coeff 1.0 \
  --pred_coeff 1.0 \
  --sparsity_coeff 1.0 \
  --hyperlista_c_theta 0.01 \
  --hyperlista_c_beta 0.0 \
  --hyperlista_c_ss 0.5 \
  --sequence_length 1 \
  --lr 5e-5 \
  --seed 42 \
  --device cuda

# StructuredLISTAKM for multi-basin systems
uv run python tools/train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --structured \
  --num_steps 10000 \
  --batch_size 256 \
  --d_global 16 \
  --num_basins 20 \
  --d_basin 16 \
  --lambda_exclusivity 0.05 \
  --lambda_sparsity 0.3 \
  --sequence_length 1 \
  --seed 42 \
  --device cuda
```

### Evaluate a Trained Model

```bash
# Evaluate checkpoint on a specific system
uv run python tools/evaluate_checkpoints.py \
  --run_dir runs/lista/<timestamp> \
  --system lyapunov \
  --device cuda

# Evaluate basin structure correspondence (any model)
uv run python tools/evaluate_latent_basin_clustering.py \
  --checkpoint runs/<model>/<timestamp>/checkpoint.pt \
  --num_trajectories 100 \
  --output_dir results/latent_clustering/<model_name>

# Evaluate basin block specialization (StructuredLISTAKM only)
uv run python tools/evaluate_basin_structure.py \
  --checkpoint runs/structured_lista/<timestamp>/checkpoint.pt \
  --system lyapunov \
  --num_trajectories 100 \
  --output_dir results/basin_structure/<run_name>
```

## Repository Structure

```
skae/
├── skae/                  # Core library package
│   ├── __init__.py
│   ├── config.py          # Configuration system with presets
│   ├── model.py           # Koopman autoencoder models
│   ├── data.py            # Dynamical systems environments
│   ├── evaluation.py      # Model evaluation utilities
│   └── benchmarks/        # Benchmark system catalogs and adapters
├── tools/                 # CLI tools and scripts
│   ├── train.py           # Training script (CLI + API)
│   ├── evaluate_checkpoints.py
│   ├── evaluate_basin_structure.py
│   ├── evaluate_latent_basin_clustering.py
│   ├── plot_training_metrics.py
│   └── collect_sweep_results.py
├── scripts/               # Shell scripts for experiments (sbatch, sweeps)
├── experiments/           # Experiment-specific code
├── tests/                 # Unit tests
├── notebooks/             # Research notebooks
├── docs/                  # Documentation
│   ├── notes.tex          # Research paper draft
│   ├── figures/           # Visualizations
│   └── planning/          # Planning documents
└── runs/                  # Training outputs (gitignored)
```

## Available Configurations

### `generic` - Standard Koopman Autoencoder
```bash
uv run python tools/train.py --config generic --env duffing
```
- **Model**: GenericKM
- **Target size**: 64
- **Encoder**: [64, 64] MLP
- **Decoder**: Linear
- **Loss weights**: Residual (1.0), Reconstruction (0.02)

### `generic_sparse` - Sparse Koopman with L1 regularization
```bash
uv run python tools/train.py --config generic_sparse --env duffing --sparsity_coeff 0.01
```
- **Model**: GenericKM
- **Target size**: 64
- **Encoder**: [64, 64] MLP with ReLU + bias
- **Decoder**: Linear
- **Loss weights**: Residual (1.0), Reconstruction (0.5), Sparsity (0.01)

### `generic_prediction` - Prediction-focused
```bash
uv run python tools/train.py --config generic_prediction --env duffing
```
- **Loss weights**: Prediction (1.0), others disabled

### `lista` - LISTA Sparse Encoder
```bash
uv run python tools/train.py --config lista --env lotka_volterra --target_size 2048
```
- **Model**: LISTAKM
- **Target size**: 2048 (overcomplete)
- **Encoder**: LISTA with 5 iterations
- **Decoder**: Normalized dictionary
- **Loss weights**: Residual (1.0), Reconstruction (1.0), Sparsity (1.0)

### `lista_nonlinear` - LISTA with MLP
```bash
uv run python tools/train.py --config lista_nonlinear --env lorenz63
```
- **Model**: LISTAKM with nonlinear pre-activation
- **Encoder**: [16, 16] MLP → LISTA

### `hyperlista` - HyperLISTA sparse encoder
HyperLISTA exposes three scalar hyperparameters that control thresholding, momentum, and support selection:
- `--hyperlista_c_theta` (C_THETA): Threshold scaling
- `--hyperlista_c_beta` (C_BETA): Momentum coefficient
- `--hyperlista_c_ss` (C_SS): Support selection ratio

## Environments

| Environment | Dimension | Description |
|------------|-----------|-------------|
| `duffing` | 2D | Duffing oscillator with two stable centers |
| `pendulum` | 2D | Simple pendulum |
| `lotka_volterra` | 2D | Predator-prey dynamics |
| `lorenz63` | 3D | Chaotic Lorenz attractor |
| `parabolic` | 2D | Parabolic attractor (analytical Koopman) |
| `lyapunov` | Configurable | Multi-attractor system with Lyapunov dynamics |
| `blended` | 2D | 3 basins with genuinely different local dynamics |
| `dysts:*` | Various | 135+ chaotic systems from dysts library |

### Lyapunov Environment Options

The Lyapunov environment supports configurable dimensions and basin layouts:

```bash
uv run python tools/train.py \
  --env lyapunov \
  --lyapunov_dim 4 \
  --lyapunov_num_basins 8 \
  --lyapunov_points_mode random \
  --lyapunov_center_scale 3.0 \
  --lyapunov_extend_mode embed \
  ...
```

### Dysts Systems

Access 135+ chaotic systems from the [dysts](https://github.com/williamgilpin/dysts) library:

```bash
# List available systems
uv run python tools/train.py --list-dysts

# Train on a dysts system
uv run python tools/train.py \
  --config lista_nonlinear \
  --env "dysts:Lorenz" \
  --standardize \
  --dysts_ic_noise_scale 0.2 \
  ...
```

## Training Output

Each training run creates a timestamped directory:

```
runs/<model>/<timestamp>/
├── config.json              # Full configuration (reproducibility)
├── checkpoint.pt            # Best model (lowest validation error)
├── last.pt                  # Latest checkpoint
├── metrics_history.jsonl    # Time series of all metrics
└── evaluation_*/            # Evaluation results and plots
```

## Model Evaluation

The evaluation module provides comprehensive evaluation of trained Koopman models using multiple rollout strategies and horizon-wise metrics.

### Rollout Strategies

The evaluation protocol tests three rollout modes:

1. **No reencoding** (`no_reencode`): Evolves entirely in latent space using `step_latent()` without reencoding
2. **Every-step reencoding** (`every_step`): Reencodes at each step using `step_env()` (state-space evolution)
3. **Periodic reencoding** (`periodic_k`): Reencodes every k steps (default periods: 10, 25, 50, 100)

### Evaluation Metrics

For each system and rollout mode, the evaluation computes:

- **Horizon-wise MSE**: Mean ± std MSE aggregated across initial conditions
- **Cumulative MSE curve**: Time-averaged MSE vs. prediction horizon
- **Per-step L2 error**: Mean L2 error at each prediction step
- **Best periodic reencoding**: Automatically identifies optimal reencoding period per horizon

## Python API

```python
from skae.config import get_config
from skae.model import make_model
from skae.data import make_env, generate_trajectory
from skae.evaluation import evaluate_model, EvaluationSettings

# Create config and environment
cfg = get_config("lista_nonlinear")
cfg.ENV.ENV_NAME = "lyapunov"
cfg.MODEL.TARGET_SIZE = 512

env = make_env(cfg)
model = make_model(cfg, env.observation_size)

# Generate trajectories
trajectory = generate_trajectory(env, length=100, batch_size=32)

# Evaluate model
settings = EvaluationSettings(horizons=[100, 500, 1000])
metrics = evaluate_model(model, env, settings)
```

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_model.py -v

# Run with coverage
pytest --cov=skae --cov-report=html
```

## License

See `LICENSE` file for details.
