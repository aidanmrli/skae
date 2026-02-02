# Experiments

Date: February 2, 2026

Goal: Achieve **unique support patterns for unique basins** (mechanistic interpretability), starting from the simplest setups and iterating on sparsity, target size, and Koopman structure.

## Notes
- **Default SLURM partition is `long`** for all sbatch scripts. (The `main` partition has GPU count restrictions.)
- Support uniqueness is measured with `tools/evaluate_support_uniqueness.py`.
- Threshold sweeps are **post‑hoc** and require a completed checkpoint.
- **Training-time support monitoring** is now available via `--monitor_support` flag (logs `support/*` metrics every 500 steps).

## Queue Status
Submitted successfully (February 2, 2026, with support monitoring):
- Lyapunov‑HD target size sweep: job `8602046` (array 0-4)
- Duffing target size sweep: job `8602047` (array 0-4)

Previous jobs (8601923, 8601924) were cancelled and resubmitted with support monitoring enabled.

### Monitoring Commands
```bash
# Check job status
squeue -u $USER -a

# Watch Lyapunov-HD support metrics as they come in
tail -f /network/scratch/l/lia/skae/lyap_hd_tsize-8602046_*.out | grep -E "(Step|Support)"

# Watch Duffing support metrics
tail -f /network/scratch/l/lia/skae/duffing_tsize-8602047_*.out | grep -E "(Step|Support)"

# Check separation scores across all runs
grep "separation_score" /network/scratch/l/lia/skae/lyap_hd_tsize-8602046_*.out
```

### Interpreting Support Metrics
During training, you'll see lines like:
```
  Support[✓]: sep=0.72 cons=0.85 uniq=11 size=18.3
```
- **sep** (separation_score): Higher = less overlap between basins (target: >0.7)
- **cons** (intra_basin_consistency): Higher = stable supports within each basin (target: >0.8)
- **uniq** (unique_support_count): Should approach `num_basins` (13 for Lyapunov, 2 for Duffing)
- **size** (mean_support_size): Should be moderate (5-20% of `target_size`)

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
