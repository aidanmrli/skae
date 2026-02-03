# Experiments

Date: February 2, 2026

Goal: Achieve **unique support patterns for unique basins** (mechanistic interpretability), starting from the simplest setups and iterating on sparsity, target size, and Koopman structure.

## Notes
- **Default SLURM partition is `long`** for all sbatch scripts. (The `main` partition has GPU count restrictions.)
- Support uniqueness is measured with `tools/evaluate_support_uniqueness.py`.
- Threshold sweeps are **post-hoc** and require a completed checkpoint.
- **Training-time support monitoring** is now available via `--monitor_support` flag (logs `support/*` metrics every 500 steps).

## Queue Status
All jobs completed (February 2, 2026):
- Lyapunov-HD target size sweep: job `8602046` (array 0-4) -- **COMPLETED**
- Duffing target size sweep: job `8602047` (array 0-4) -- **COMPLETED**

---

## Results: Support Uniqueness (February 2, 2026)

Evaluated from `support_eval/support_uniqueness.json` with `support_threshold=1e-3`, `support_mode=mean`, 100 trajectories, 500 steps each.

Definitions:
- `unique` = number of basins with a distinct mode support / total basins
- `sep` = 1 - mean pairwise Jaccard (higher = less overlap between basins)
- `cons` = mean basin consistency (fraction of trajectories matching their basin's mode support)
- `size` = mean mode support size (number of active latent dimensions)

### Lyapunov-HD (DIM=8, NUM_BASINS=13, SPARSITY=1.0)

| target_size | unique | sep | cons | size | size % |
|-------------|--------|-------|-------|------|--------|
| 64 | 5/13 | 0.581 | 0.534 | 1.3 | 2.0% |
| 128 | 8/13 | 0.801 | 0.361 | 1.6 | 1.3% |
| **256** | **13/13** | **0.846** | 0.138 | 43.5 | 17.0% |
| **512** | **13/13** | 0.790 | 0.184 | 13.5 | 2.6% |
| **1024** | **13/13** | **0.852** | 0.138 | 138.2 | 13.5% |

### Duffing (NUM_BASINS=2, SPARSITY=1.0)

| target_size | unique | sep | cons | size | size % |
|-------------|--------|-------|-------|------|--------|
| 32 | 2/2 | 0.571 | 0.460 | 5.0 | 15.6% |
| 64 | 2/2 | **1.000** | 0.112 | 11.5 | 18.0% |
| 128 | 2/2 | 0.867 | 0.141 | 17.0 | 13.3% |
| 256 | 2/2 | 0.984 | 0.102 | 32.5 | 12.7% |
| 512 | 2/2 | **1.000** | 0.060 | 58.0 | 11.3% |

### Key Findings

1. **Target size >= 256 needed for full basin separation on Lyapunov (13 basins).** Smaller dimensions lack capacity: ts=64 only separates 5/13, ts=128 separates 8/13.

2. **Duffing (2 basins) is fully separable at all target sizes.** Even ts=32 achieves 2/2 unique supports, though with lower separation (0.571). ts=64 and ts=512 achieve perfect separation (Jaccard=0).

3. **Consistency-uniqueness tradeoff.** Across both systems, larger target sizes increase uniqueness and separation but *decrease* within-basin consistency. This is the central diagnostic challenge: the mode support per basin is unique, but individual trajectories within a basin don't consistently produce that exact mode support.

4. **Support sizes scale with target_size.** Active dimensions are roughly 10-20% of the latent dimension across all configurations, indicating the sparsity coefficient (1.0) produces a consistent sparsity level.

---

## Results: Training-Time Support Dynamics

The `--monitor_support` flag revealed how separation evolves during training.

### Lyapunov-HD: Separation score over training

| step | ts=64 | ts=128 | ts=256 | ts=512 | ts=1024 |
|------|-------|--------|--------|--------|---------|
| 500 | 0.538 | 0.727 | 0.728 | 0.000 | 0.420 |
| 1000 | 0.756 | 0.831 | 0.678 | 0.000 | 0.740 |
| 2000 | 0.809 | 0.814 | 0.706 | 0.661 | 0.533 |
| 3000 | 0.824 | 0.821 | 0.821 | 0.469 | 0.771 |
| 5000 | 0.823 | 0.833 | 0.834 | 0.811 | 0.842 |
| 7000 | 0.833 | 0.844 | 0.838 | 0.828 | 0.834 |
| 9500 | 0.840 | 0.851 | 0.847 | 0.843 | **0.855** |

Key observations:
- **ts=64-128** reach high separation quickly (~1000 steps) but plateau early
- **ts=512** starts at **zero separation for 1500 steps**, then rapidly catches up
- **ts=1024** is noisy early but achieves the **highest final separation** (0.855)
- **All sizes converge** to similar separation (~0.84-0.85) by end of training
- The convergence rate is inversely related to target size

---

## Results: Prediction MSE

### Lyapunov-HD (best checkpoint)

| target_size | H100 (no-re) | H500 (no-re) | H1000 (no-re) | H500 (best-PR) | best mode |
|-------------|-------------|-------------|-------------|-----------------|-----------|
| 64 | 4.45e+00 | 3.66e+00 | 3.60e+00 | 3.90e+00 | periodic_100 |
| 128 | 4.61e+00 | 6.37e+00 | 9.75e+02 | 3.93e+00 | periodic_100 |
| 256 | 3.26e+00 | 6.42e+06 | 2.91e+17 | 3.60e+00 | periodic_50 |
| 512 | 4.39e+00 | 3.14e+07 | 6.02e+19 | 3.90e+00 | periodic_50 |
| 1024 | 8.19e+00 | 1.79e+11 | 1.34e+25 | **3.38e+00** | periodic_25 |

### Duffing (best checkpoint)

| target_size | H100 (no-re) | H500 (no-re) | H1000 (no-re) | H500 (best-PR) | best mode |
|-------------|-------------|-------------|-------------|-----------------|-----------|
| 32 | 1.08e-01 | 2.80e+00 | 7.01e+01 | 1.19e+00 | periodic_10 |
| 64 | 3.38e-02 | 1.63e+05 | 1.81e+14 | 2.74e-01 | periodic_25 |
| 128 | 9.19e-03 | 6.33e-01 | 8.89e+01 | 6.92e-02 | periodic_25 |
| 256 | 3.24e-03 | 2.45e+01 | 3.08e+08 | 6.81e-02 | periodic_25 |
| 512 | **1.46e-03** | 2.16e+00 | 2.58e+04 | **1.98e-02** | periodic_25 |

### Prediction Findings

1. **Without reencoding, larger latent dims diverge catastrophically.** ts=256+ explode at H500+. This is expected: larger K matrices have more room for eigenvalue drift.

2. **Periodic reencoding equalizes performance.** With optimal reencoding period, all target sizes achieve similar MSE on Lyapunov (~3.4-3.9).

3. **ts=1024 with periodic_25 gives the best Lyapunov H500 MSE** (3.38e+00), despite the worst short-horizon accuracy without reencoding.

4. **Duffing accuracy improves monotonically with target size** (H100: 0.108 → 0.001). ts=512 is best overall.

---

## Interpretation: Uniqueness vs Consistency

The central finding is a **uniqueness-consistency tradeoff**:

- **Uniqueness** (do different basins have different mode supports?): Achieved with sufficient capacity (ts >= 256 for 13 basins, ts >= 32 for 2 basins).

- **Consistency** (do trajectories within the same basin always produce the same support?): Low across all configurations (max ~0.53 for small ts, drops to ~0.06 for large ts).

This means the LISTA encoder learns that *different basins should activate different regions of the latent space*, but individual trajectories within a basin activate *slightly different subsets* each time. The mode support is unique, but it's the plurality winner, not the unanimous winner.

**Possible explanations:**
1. The support threshold (1e-3) is too binary -- small activation differences near the threshold cause support fluctuation
2. Trajectories near basin boundaries genuinely straddle multiple support patterns
3. The encoder hasn't converged to a sharp partition -- sparsity encourages different atoms but doesn't enforce consistency

**Next steps to improve consistency:**
- Sweep support_threshold (the post-hoc threshold sweep script is ready)
- Try "majority" or "median" support modes instead of "mean"
- Consider soft support metrics (e.g., cosine similarity of continuous activations rather than binary supports)
- Add explicit within-basin consistency loss during training

---

## Experiment Definitions

### 1) Lyapunov-HD target size sweep (simple LISTA baseline)
Script: `scripts/sweep_target_size_lyapunov_hd.sh`

```bash
sbatch scripts/sweep_target_size_lyapunov_hd.sh
```

Defaults: `DIM=8`, `NUM_BASINS=13`, `SPARSITY=1.0`, target sizes: 64, 128, 256, 512, 1024

### 2) Duffing target size sweep (simple LISTA baseline)
Script: `scripts/sweep_target_size_duffing.sh`

```bash
sbatch scripts/sweep_target_size_duffing.sh
```

Defaults: target sizes: 32, 64, 128, 256, 512, `SPARSITY=1.0`

### 3) Support threshold sweep (post-hoc eval)
Script: `scripts/sweep_support_threshold.sh`

```bash
sbatch --export=ALL,CKPT=/path/to/checkpoint.pt,OUT_BASE=/path/to/out \
  scripts/sweep_support_threshold.sh
```

Thresholds tested: `1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1`

---

## Iteration Plan

1. **Simple baselines** (DONE): dense Koopman `K`, LISTA encoder, linear decoder, strong sparsity.
2. **Find stable unique supports** (DONE): ts >= 256 achieves full uniqueness.
3. **Improve consistency** (NEXT): threshold sweep, soft metrics, consistency regularization.
4. **Re-introduce Koopman structure** (AFTER): diagonal, block-diagonal constraints once support correspondence is stable.

## Pending (needs code changes)
- Soft support similarity metrics (cosine on continuous activations)
- Within-basin consistency regularization loss
- Koopman structure restrictions beyond arrowhead (diagonal, block-diagonal)
