# Experiments Module

This directory contains experiment scripts for benchmarking the Koopman autoencoder 
on chaotic systems from the dysts library.

## Scripts

### `run_dysts_sweep.py`
Sweep training across multiple dysts systems.

```bash
# Quick test on 4 canonical systems (default hyperparameters)
uv run python dysts_experiments/run_dysts_sweep.py --systems quick --config lista --num_steps 5000

# Quick test with tuned hyperparameters (recommended)
uv run python dysts_experiments/run_dysts_sweep.py \
  --systems quick \
  --config lista \
  --num_steps 10000 \
  --target_size 512 \
  --pred_coeff 10.0 \
  --sparsity_coeff 1.5 \
  --lista_alpha 0.3 \
  --pairwise

# Standard benchmark on 12 systems
uv run python dysts_experiments/run_dysts_sweep.py --systems standard --config lista --num_steps 10000

# Multi-basin candidates (manual + metadata keywords)
uv run python dysts_experiments/run_dysts_sweep.py --systems multi_basin --config lista --num_steps 10000

# Full benchmark with multiple seeds
uv run python dysts_experiments/run_dysts_sweep.py --systems standard --config lista --seeds 0 1 2

# Custom system selection
uv run python dysts_experiments/run_dysts_sweep.py \
  --systems custom \
  --custom_systems Lorenz Rossler Chua Chen Thomas \
  --config lista \
  --num_steps 10000

# Dry run to see what would be trained
uv run python dysts_experiments/run_dysts_sweep.py --systems standard --dry_run
```

### Hyperparameter Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--target_size` | 2048 (from lista config) | Latent dimension |
| `--sparsity_coeff` | 1.0 | Sparsity loss weight |
| `--reconst_coeff` | 1.0 | Reconstruction loss weight |
| `--pred_coeff` | 0.0 | Prediction loss weight |
| `--lista_alpha` | 0.005 | LISTA soft-threshold alpha |
| `--pairwise` | False | Use single-step training instead of sequence |

## System Sets

- **quick**: 4 systems (Lorenz, Rossler, Chua, Chen) for rapid validation
- **standard**: 12 systems for paper benchmarks
- **extended**: 20+ systems for comprehensive evaluation
- **multi_basin**: union of multistable + multi-scroll candidates
- **multi_attractor**: multistable/multi-attractor systems
- **multi_scroll**: multi-scroll/double-wing systems (lobe switching)
- **full**: All 135+ dysts systems
- **custom**: User-specified list via `--custom_systems`

## Output Structure

```
runs/dysts_sweep/
└── lista_standard_20260119-220000/
    ├── sweep_config.json       # Sweep configuration
    ├── sweep_results.json      # Aggregated results
    └── seed_0/
        ├── Lorenz/
        │   ├── config.json
        │   ├── checkpoint.pt
        │   └── metrics_history.jsonl
        ├── Rossler/
        └── ...
```
