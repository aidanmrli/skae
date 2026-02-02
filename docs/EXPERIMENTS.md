# Experiments

Date: February 2, 2026

Goal: Achieve **unique support patterns for unique basins** (mechanistic interpretability), starting from the simplest setups and iterating on sparsity, target size, and Koopman structure.

## Notes
- **Default SLURM partition is `long`** for all sbatch scripts. (The `main` partition has GPU count restrictions.)
- Support uniqueness is measured with `tools/evaluate_support_uniqueness.py`.
- Threshold sweeps are **post‑hoc** and require a completed checkpoint.

## Queue Status
Submitted successfully (February 2, 2026):
- Lyapunov‑HD target size sweep: job `8601836`
- Duffing target size sweep: job `8601837`

## Queued Experiments (to submit)

### 1) Lyapunov‑HD target size sweep (simple LISTA baseline)
Script: `scripts/sweep_target_size_lyapunov_hd.sh`

Command:
```bash
sbatch scripts/sweep_target_size_lyapunov_hd.sh
```

Defaults:
- `DIM=8`, `NUM_BASINS=13`, `SPARSITY=1.0`
- Target sizes: 64, 128, 256, 512, 1024
- `lyapunov_extend_mode=embed` (2D Lyapunov + linear decay in extra dims)

Outputs:
- Training logs in `/network/scratch/l/lia/skae/lyapunov_hd_target_sweep/...`
- Support uniqueness evaluation in `support_eval/`

### 2) Duffing target size sweep (simple LISTA baseline)
Script: `scripts/sweep_target_size_duffing.sh`

Command:
```bash
sbatch scripts/sweep_target_size_duffing.sh
```

Defaults:
- Target sizes: 32, 64, 128, 256, 512
- `SPARSITY=1.0`

Outputs:
- Training logs in `/network/scratch/l/lia/skae/duffing_target_sweep/...`
- Support uniqueness evaluation in `support_eval/`

### 3) Support threshold sweep (post‑hoc eval)
Script: `scripts/sweep_support_threshold.sh`

Requires a checkpoint path:
```bash
sbatch --export=ALL,CKPT=/path/to/checkpoint.pt,OUT_BASE=/path/to/out \
  scripts/sweep_support_threshold.sh
```

Optional overrides:
- `SYSTEM=lyapunov|duffing`
- `SUPPORT_MODE=mean|last|median|majority`

Thresholds tested: `1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1`

## Iteration Plan (simple → structured)
1. **Simple baselines** (current): dense Koopman `K`, LISTA encoder, linear decoder, strong sparsity.
2. **Find stable unique supports** by sweeping target size and support threshold.
3. **Only after stable supports appear**, re‑introduce Koopman structure constraints (diagonal, block‑diagonal, arrowhead).

## Pending (needs code changes)
- Koopman structure restrictions beyond arrowhead (diagonal, block‑diagonal) are not wired into the model yet. Once added, we will create sbatch sweeps for those constraints.
