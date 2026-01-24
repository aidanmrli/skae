# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SKAE (Sparse Koopman Autoencoder) is a PyTorch research codebase for learning Koopman operator representations of nonlinear dynamical systems. The Koopman operator provides a linear representation of nonlinear dynamics in a lifted feature space, enabling linear prediction of nonlinear system evolution.

We are attempting to use a LISTA encoder instead of a MLP encoder for this Koopman autoencoder, which attempts to learn to make forecasts for nonlinear dynamical systems with multiple basins of attraction determined by having multiple fixed points. The idea is that the LISTA encoder should naturally enforce sparsity and allow for more simple and interpretable behavior. 

We want to induce structured sparsity on the Koopman matrix such that each basin of attraction corresponds to its own high dimensional subspace where the Koopman dynamics are approximately linear. Our key insight is to enforce a sparse latent representation, structured as a union of subspaces, where each active support approximately corresponds to a specific basin of attraction or dynamical regime. Ideally, we would like to see if we can isolate distinct Koopman linear dynamics for each basin. 

If this is true, we can isolate each basin and solve with LQR. Then, a nonlinear control problem over multiple basins of attraction reduces to solving linear Koopman dynamics within each basin using LQR, and modeling the changes between basins. We hypothesize that our periodic reencoding mechanism at inference time where we encode and immediately decode an input might be a good mechanism for modeling changes between basins..

## Common Commands

```bash
# Install dependencies (uses uv for package management)
uv sync
# Add a package
uv add <package_name>

# Train a model (basic examples)
uv run python train.py \
  --config generic_sparse \
  --env lyapunov \
  --num_steps 5000 \
  --batch_size 256 \
  --target_size 64 \
  --reconst_coeff 0.02 \
  --pred_coeff 1.0 \
  --sparsity_coeff 0.001 \
  --pairwise \
  --seed 0 \
  --device cuda

uv run python train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --num_steps 20000 \
  --batch_size 256 \
  --target_size 512 \
  --reconst_coeff 1.0 \
  --pred_coeff 10.0 \
  --sparsity_coeff 1.5 \
  --lista_alpha 0.3 \
  --pairwise \
  --seed 42 \
  --device cuda

# Example for one of the dysts systems
uv run python train.py \
  --config lista_nonlinear \
  --env "dysts:${SYSTEM}" \
  --num_steps 5000 \
  --batch_size 256 \
  --target_size 128 \
  --reconst_coeff 0.5 \
  --pred_coeff 1.0 \
  --sparsity_coeff 1.0 \
  --lista_alpha 0.30 \
  --lista_num_loops 5 \
  --pairwise \
  --standardize \
  --dysts_ic_noise_scale 0.2 \
  --dysts_native_cache \
  --dysts_cache_warmup 2000 \
  --seed 42 \
  --device cuda \
  --log_dir "/network/scratch/l/lia/skae/dysts_multi_basin_lista_nonlinear/dysts:${SYSTEM}"

# Evaluate a trained checkpoint
uv run python evaluate_checkpoints.py --run_dir runs/lista/<timestamp> --system lyapunov --device cuda

# Run sbatch sweep using lista_nonlinear config on dysts multi-basin environments. Note that hyperparameters should be set manually and carefully.
sbatch dysts_multi_basin_lista_nonlinear_long.sh

# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py -v

# List available dysts chaotic systems
uv run python train.py --list-dysts
```

## Architecture

### Core Components

- **`config.py`**: Dataclass-based configuration system with presets (`generic`, `generic_sparse`, `lista`, `lista_nonlinear`, `hyperlista`)
- **`model.py`**: Koopman machine implementations inheriting from `KoopmanMachine` base class
- **`data.py`**: Dynamical systems environments (Duffing, Pendulum, Lorenz63, Lyapunov, etc.) with `Env` base class
- **`train.py`**: Training loop with CLI interface and automatic evaluation
- **`evaluation.py`**: Comprehensive model evaluation with rollout strategies

### Model Hierarchy

```
KoopmanMachine (ABC)
├── GenericKM          # MLP encoder/decoder with learnable Koopman matrix
├── LISTAKM            # LISTA sparse encoder + dictionary decoder
├── HyperLISTAKM       # HyperLISTA (3 scalar hyperparams, gradient flow to dictionary)
└── StructuredLISTAKM  # Basin-aware Koopman with structured latent space
```

### Key Abstractions

- **Encoder**: Maps observations x → latent z (sparse for LISTA variants)
- **Decoder**: Maps latent z → reconstruction x̂ (dictionary-based for LISTA)
- **Koopman Matrix K**: Linear dynamics in latent space: z_{t+1} = K @ z_t
- **Loss Components**: residual (alignment), reconstruction, prediction, sparsity

### Environment System

Built-in environments: `duffing`, `pendulum`, `lotka_volterra`, `lorenz63`, `parabolic`, `lyapunov`

External dysts systems: Use `--env dysts:SystemName` (e.g., `dysts:Lorenz`, `dysts:Chua`)

## Configuration Patterns

Get a preset config and modify:
```python
from config import get_config
cfg = get_config("lista")
cfg.MODEL.TARGET_SIZE = 512
cfg.TRAIN.NUM_STEPS = 10000
```

Key config paths:
- `cfg.MODEL.TARGET_SIZE`: Latent dimension (zdim)
- `cfg.MODEL.SPARSITY_COEFF`: L1 sparsity weight
- `cfg.MODEL.ENCODER.LISTA.ALPHA`: LISTA soft-threshold
- `cfg.MODEL.ENCODER.HYPERLISTA.C_THETA/C_BETA/C_SS`: HyperLISTA hyperparams
- `cfg.TRAIN.USE_SEQUENCE_LOSS`: False for pairwise, True for sequence training

## Model Factory

```python
from model import make_model
model = make_model(cfg, observation_size)  # Uses cfg.MODEL.MODEL_NAME
```

## Training Output Structure

```
runs/<model>/<timestamp>/
├── config.json           # Full config for reproducibility
├── checkpoint.pt         # Best model (lowest validation error)
├── last.pt              # Latest checkpoint
├── metrics_history.jsonl # Training metrics time series
└── evaluation_*/        # Evaluation results and plots
```

## Important Implementation Details

- Homogeneous coordinates (`cfg.MODEL.USE_HOMOGENEOUS=True`): Appends 1 to input for implicit bias learning in LISTA/HyperLISTA
- Koopman matrix uses separate learning rate (`cfg.TRAIN.K_MATRIX_LR`, typically lower than encoder/decoder)
- LISTA/HyperLISTA dictionaries are column-normalized during forward pass
- StructuredLISTAKM uses block-wise Koopman parameters for basin-aware dynamics
