# Experiments Module

This directory contains experiment scripts for benchmarking the Koopman autoencoder 
on chaotic systems from the dysts library.

## Scripts

### `run_dysts_sweep.py`
Sweep training across multiple dysts systems.

```bash
# Quick test on 4 canonical systems
uv run python experiments/run_dysts_sweep.py --systems quick --config lista --num_steps 5000

# Standard benchmark on 12 systems
uv run python experiments/run_dysts_sweep.py --systems standard --config lista --num_steps 10000

# Full benchmark with multiple seeds
uv run python experiments/run_dysts_sweep.py --systems standard --config lista --seeds 0 1 2

# Custom system selection
uv run python experiments/run_dysts_sweep.py \
  --systems custom \
  --custom_systems Lorenz Rossler Chua Chen Thomas \
  --config lista \
  --num_steps 10000

# Dry run to see what would be trained
uv run python experiments/run_dysts_sweep.py --systems standard --dry_run
```

## System Sets

- **quick**: 4 systems (Lorenz, Rossler, Chua, Chen) for rapid validation
- **standard**: 12 systems for paper benchmarks
- **extended**: 20+ systems for comprehensive evaluation
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
