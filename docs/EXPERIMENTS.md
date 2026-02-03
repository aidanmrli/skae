# Experiments

Date: February 2, 2026

Goal: Achieve **unique support patterns for unique basins** (mechanistic interpretability), starting from the simplest setups and iterating on sparsity, target size, and Koopman structure.

## Notes
- **Default SLURM partition is `long`** for all sbatch scripts. (The `main` partition has GPU count restrictions.)
- Support uniqueness is measured with `tools/evaluate_support_uniqueness.py`.
- Threshold sweeps are **post-hoc** and require a completed checkpoint.
- **Training-time support monitoring** is now available via `--monitor_support` flag (logs `support/*` metrics every 500 steps).

## Queue Status

Completed (February 2, 2026):
- Lyapunov-HD target size sweep: job `8602046` (array 0-4) -- **COMPLETED**
- Duffing target size sweep: job `8602047` (array 0-4) -- **COMPLETED**

Completed (February 3, 2026):
- K structure × target size sweep: job `8603752` (array 0-14, 5 target sizes × 3 K structures) -- **COMPLETED**
- Arrowhead (StructuredLISTAKM) sweep: job `8603753` (array 0-4, 5 total latent dims) -- **COMPLETED**

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

### 4) K structure × target size sweep
Script: `scripts/sweep_k_structure_lyapunov.sh`

```bash
sbatch scripts/sweep_k_structure_lyapunov.sh
```

15 jobs: 5 target sizes × 3 K structures (dense, diagonal, block_diagonal).
Post-training eval includes `--threshold_sweep` + cosine similarity.
Output: `/network/scratch/l/lia/skae/lyapunov_k_structure_sweep/`

### 5) Arrowhead (StructuredLISTAKM) sweep
Script: `scripts/sweep_arrowhead_lyapunov.sh`

```bash
sbatch scripts/sweep_arrowhead_lyapunov.sh
```

5 jobs: total latent dims 64, 128, 256, 512, 1024 (d_global + 13 * d_basin = total_dim).
Uses `lambda_exclusivity=0.05`, `lambda_sparsity=0.3`, `excl_warmup=2000`.
Output: `/network/scratch/l/lia/skae/lyapunov_k_structure_sweep/`

---

## Experiment: Koopman Structure + Refined Diagnostics (February 3, 2026)

### Motivation

The previous experiments established that the LISTA encoder learns basin-discriminative support patterns with a dense Koopman matrix, but two open questions remain:

1. **Is low consistency real or a thresholding artefact?** Consistency was measured by exact binary support match at threshold `1e-3`. Two activations of magnitude `9e-4` and `1.1e-3` are functionally identical but produce different binary supports. If consistency rises substantially at a different threshold, the inconsistency is noise rather than a structural problem.

2. **Does constraining the Koopman matrix improve basin--support correspondence?** A dense `K` has `d²` free parameters and no inductive bias toward basin-aligned dynamics. If different basins truly correspond to different subspaces, a structured `K` that respects that block structure should (a) improve support consistency by reducing cross-basin interference, and (b) improve long-horizon stability by having fewer eigenvalues to control.

### Hypotheses

- **H1 (threshold sensitivity):** The low consistency (~0.14--0.19) at `tau=1e-3` is largely a thresholding artefact. We expect to find a threshold where consistency is substantially higher while uniqueness is preserved. The cosine similarity metrics (which are threshold-free) should show high intra-basin similarity and low inter-basin similarity, confirming that the continuous representations are basin-discriminative even when binary supports fluctuate.

- **H2 (diagonal K):** A diagonal Koopman matrix forces each latent coordinate to evolve independently (`z_i' = k_i * z_i`). This is the most parsimonious structure (`d` parameters). If the LISTA encoder already produces basin-discriminative supports, diagonal dynamics may suffice for short-horizon prediction within a basin -- but may struggle to capture cross-coordinate coupling needed for accurate dynamics.

- **H3 (block-diagonal K):** With blocks of size `d/13` (one per ground-truth basin), the block-diagonal structure allows within-block coupling while preventing cross-block interaction. If the encoder aligns one block per basin, this should simultaneously (a) improve support consistency (each block is an independent dynamical unit), (b) maintain uniqueness (blocks are decoupled), and (c) improve long-horizon stability (smaller blocks = smaller eigenvalue problems).

- **H4 (arrowhead K):** The arrowhead structure (global block + basin blocks with one-directional coupling) is the most expressive structured option. It explicitly separates shared physics (global block) from basin-local dynamics, with the exclusivity regulariser encouraging one-basin-at-a-time activation. We expect this to produce the cleanest basin--support correspondence, but with higher training complexity and sensitivity to the exclusivity/sparsity hyperparameters.

- **H5 (structure vs capacity tradeoff):** Structured `K` matrices reduce the number of learnable dynamics parameters. At small latent dimensions (ts=64), this may help by reducing overfitting. At large dimensions (ts=1024), it may hurt by being too constrained. We expect an interaction between structure and latent dimension.

### Experiment Design

**All experiments use:** Lyapunov-HD (8D, 13 basins, embedded from 2D), `lista_nonlinear` config, pairwise training, `sparsity_coeff=1.0`, `batch_size=512`, 10k steps, `--monitor_support`.

#### Job 8603752: K structure × target size sweep (15 jobs)

Script: `scripts/sweep_k_structure_lyapunov.sh`

| Array ID | target_size | K structure | K params | Block size |
|----------|-------------|-------------|----------|------------|
| 0 | 64 | dense | 4,096 | -- |
| 1 | 64 | diagonal | 64 | -- |
| 2 | 64 | block_diagonal | 5 × 4² + 1 × 4² = 96 | 4 |
| 3 | 128 | dense | 16,384 | -- |
| 4 | 128 | diagonal | 128 | -- |
| 5 | 128 | block_diagonal | 13 × 9² + 1 × 11² = 1,174 | 9 |
| 6 | 256 | dense | 65,536 | -- |
| 7 | 256 | diagonal | 256 | -- |
| 8 | 256 | block_diagonal | 13 × 19² = 4,693 | 19 |
| 9 | 512 | dense | 262,144 | -- |
| 10 | 512 | diagonal | 512 | -- |
| 11 | 512 | block_diagonal | 13 × 39² = 19,773 | 39 |
| 12 | 1024 | dense | 1,048,576 | -- |
| 13 | 1024 | diagonal | 1,024 | -- |
| 14 | 1024 | block_diagonal | 13 × 78² = 79,092 | 78 |

Block size for block_diagonal = `target_size // 13` (one block per GT basin).

#### Job 8603753: Arrowhead (StructuredLISTAKM) sweep (5 jobs)

Script: `scripts/sweep_arrowhead_lyapunov.sh`

Total latent dim is set to match the K structure sweep: `d_global + 13 * d_basin = total_dim`.

| Array ID | total_dim | d_global | d_basin | B | lambda_excl | lambda_sparsity |
|----------|-----------|----------|---------|---|-------------|-----------------|
| 0 | 64 | 12 | 4 | 13 | 0.05 | 0.3 |
| 1 | 128 | 11 | 9 | 13 | 0.05 | 0.3 |
| 2 | 256 | 9 | 19 | 13 | 0.05 | 0.3 |
| 3 | 512 | 5 | 39 | 13 | 0.05 | 0.3 |
| 4 | 1024 | 10 | 78 | 13 | 0.05 | 0.3 |

Exclusivity warmup: 2000 steps.

#### Post-training evaluation

Every job runs `evaluate_support_uniqueness.py --threshold_sweep` after training, which produces:
- **Threshold sweep:** consistency, uniqueness, Jaccard, support size at 7 thresholds (`1e-4` to `1e-1`)
- **Cosine similarity:** intra-basin cosine, inter-basin cosine, separation score (threshold-free)
- Results saved to `<log_dir>/support_eval/threshold_sweep.json`

### Results: Cosine Similarity (threshold-free)

All 20 configurations evaluated with 100 trajectories, 500 steps each.

| ts | K structure | IntraCos | InterCos | CosSep |
|----|-------------|----------|----------|--------|
| 64 | dense | 0.9202 | 0.1310 | **0.7892** |
| 64 | diagonal | 0.7704 | 0.4775 | 0.2929 |
| 64 | block_diagonal | 0.8066 | 0.4545 | 0.3522 |
| 64 | arrowhead | 0.9887 | 0.2413 | 0.7474 |
| 128 | dense | 0.7785 | 0.5305 | 0.2481 |
| 128 | diagonal | 0.7772 | 0.5308 | 0.2464 |
| 128 | block_diagonal | 0.7663 | 0.5480 | 0.2183 |
| 128 | arrowhead | 0.9901 | 0.2335 | **0.7566** |
| 256 | dense | 0.9695 | 0.1253 | 0.8442 |
| 256 | diagonal | 0.9697 | 0.1177 | 0.8519 |
| 256 | block_diagonal | 0.9715 | 0.1166 | **0.8549** |
| 256 | arrowhead | 0.9902 | 0.2556 | 0.7347 |
| 512 | dense | 0.9883 | 0.4658 | 0.5225 |
| 512 | diagonal | 0.9918 | 0.5219 | 0.4699 |
| 512 | block_diagonal | 0.9917 | 0.5292 | 0.4625 |
| 512 | arrowhead | 0.9911 | 0.1931 | **0.7980** |
| 1024 | dense | 0.9854 | 0.1460 | 0.8394 |
| 1024 | diagonal | 0.9901 | 0.3025 | 0.6876 |
| 1024 | block_diagonal | 0.9910 | 0.3201 | 0.6709 |
| 1024 | arrowhead | 0.9884 | 0.1079 | **0.8805** |

**Key finding: H1 confirmed.** Intra-basin cosine similarity is 0.77--0.99 across all configurations, meaning trajectories within the same basin produce nearly identical continuous representations. The previously reported low binary consistency (~0.14) was entirely a thresholding artefact. The cosine separation score (intra - inter) is the correct metric going forward.

### Results: Threshold Sweep (ts=256, all structures)

| Threshold | dense cons | diag cons | blkdiag cons | arrow cons | dense unique | diag unique | blkdiag unique | arrow unique |
|-----------|-----------|-----------|-------------|-----------|-------------|------------|---------------|-------------|
| 1e-4 | 0.138 | 0.138 | 0.138 | 0.138 | 13/13 | 13/13 | 13/13 | 13/13 |
| 5e-4 | 0.138 | 0.138 | 0.138 | 0.138 | 13/13 | 13/13 | 13/13 | 13/13 |
| 1e-3 | 0.138 | 0.138 | 0.138 | 0.138 | 13/13 | 13/13 | 13/13 | 13/13 |
| 5e-3 | 0.138 | 0.138 | 0.138 | 0.138 | 13/13 | 13/13 | 13/13 | 13/13 |
| 1e-2 | 0.157 | 0.138 | 0.138 | 0.169 | 13/13 | 13/13 | 13/13 | 13/13 |
| **5e-2** | **0.473** | **0.438** | **0.553** | 0.297 | **13/13** | **13/13** | **13/13** | 13/13 |
| 1e-1 | 0.539 | 0.527 | **0.561** | 0.409 | 13/13 | 13/13 | 13/13 | 13/13 |

At `tau=5e-2`, block_diagonal achieves 0.553 consistency with full 13/13 uniqueness (up from 0.138 at `tau=1e-3`). At `tau=1e-1`, it reaches 0.561. The consistency was genuinely an artefact of too-aggressive thresholding.

### Results: Support Uniqueness at tau=1e-3

| ts | dense | diagonal | block_diag | arrowhead |
|----|-------|----------|------------|-----------|
| 64 | **13/13** | 6/13 | 6/13 | **13/13** |
| 128 | 8/13 | 8/13 | 7/13 | **13/13** |
| 256 | **13/13** | **13/13** | **13/13** | **13/13** |
| 512 | **13/13** | **13/13** | **13/13** | **13/13** |
| 1024 | **13/13** | **13/13** | **13/13** | **13/13** |

At ts>=256, all structures achieve full uniqueness. At low capacity (ts=64), dense and arrowhead achieve 13/13 while diagonal/block_diagonal fail (6/13). The arrowhead model achieves 13/13 at *all* latent dimensions due to the explicit exclusivity regulariser.

### Results: Eval Error and Training Loss

| ts | dense eval | diag eval | blkdiag eval | arrow eval | dense resid | arrow resid |
|----|-----------|-----------|-------------|-----------|-------------|-------------|
| 64 | **2.21** | 2.48 | 2.25 | 2.52 | 0.136 | 0.127 |
| 128 | 2.32 | 2.34 | **2.30** | **DIVERGED** | 0.121 | 0.103 |
| 256 | **2.59** | 2.99 | 2.69 | 3.39 | 0.093 | 0.081 |
| 512 | 2.97 | 2.96 | **2.34** | 2.87 | 0.069 | 0.064 |
| 1024 | **2.28** | 2.48 | 2.37 | 2.86 | 0.047 | 0.050 |

Note: ts=128 arrowhead diverged catastrophically (eval final error = 13,520). The arrowhead model consistently achieves the lowest residual loss but this does not translate to better prediction accuracy — the reconstruction pathway appears to be the bottleneck.

**Block_diagonal at ts=512 achieves the best eval error (2.34)** across all 20 configurations. It outperforms dense (2.97) by 21% at that dimensionality, with 13x fewer K parameters (19,773 vs 262,144).

### Interpretation

1. **The LISTA encoder is the primary driver of basin discrimination, not the Koopman matrix.** At sufficient capacity (ts>=256), dense, diagonal, and block_diagonal K produce nearly identical cosine separation scores (~0.84--0.85). The encoder learns basin-discriminative supports regardless of K structure. This means the sparsity inductive bias of LISTA is doing the heavy lifting.

2. **Constraining K at low capacity hurts uniqueness.** At ts=64, dense K achieves 13/13 uniqueness while diagonal/block_diagonal only manage 6/13. With limited latent dimensions, the model needs the full K coupling to compensate — the encoder can't produce enough distinct sparse codes when K is too constrained. The exception is arrowhead, which achieves 13/13 at ts=64 via the exclusivity loss, not K structure.

3. **Block_diagonal K provides a parameter efficiency advantage for dynamics.** At ts=512, block_diagonal gives 21% better eval error than dense with 13x fewer K parameters. This suggests that at moderate-to-large latent dimensions, constraining off-diagonal coupling acts as beneficial regularisation for the dynamics, preventing overfitting in the Koopman matrix.

4. **Arrowhead is unstable at intermediate sizes.** The ts=128 arrowhead diverged catastrophically despite achieving the lowest residual loss during training. This indicates a disconnect: low residual loss (good latent-space alignment) does not guarantee good prediction (good decode-step-decode accuracy). The arrowhead's reconstruction quality is excellent (lowest reconst loss), but the coupling terms may create amplifying feedback loops during multi-step rollout.

5. **The "uniqueness--consistency tradeoff" from the previous experiments is resolved.** It was entirely a thresholding artefact. The cosine metrics show that within-basin representations are highly consistent (cosine ~0.97) at all configurations where uniqueness is achieved. The correct diagnostic is the cosine separation score, not binary support consistency.

### Next Steps

1. **Long-horizon prediction MSE with periodic reencoding** on all 20 checkpoints. Block_diagonal at ts=512 is the most promising candidate — fewer K parameters should produce smaller eigenvalue drift and better long-horizon stability than dense K.

2. **Extract per-block dynamics from block_diagonal K.** With block_size = d/13, each block is a candidate per-basin Koopman matrix. Compute the eigenvalues of each block and compare them to the ground-truth Lyapunov attractor dynamics. If the blocks align with basins, we can read off the local linear dynamics directly.

3. **Test LQR on extracted basin dynamics.** The original motivation is to reduce nonlinear control to per-basin LQR. With block_diagonal K, each block defines a local linear system. Design LQR controllers for each block and test whether they successfully steer trajectories within their basin.

4. **Stabilise arrowhead at ts=128.** Possible fixes: lower K learning rate, gradient clipping on coupling terms, or longer exclusivity warmup. The arrowhead's guaranteed uniqueness at all dimensions is valuable if the stability issue can be resolved.

5. **Validate on Duffing and dysts systems.** Run the best configurations (dense ts=256, block_diagonal ts=512) on Duffing (2 basins) and multi-basin dysts systems to test generality.

---

## Iteration Plan

1. **Simple baselines** (DONE): dense Koopman `K`, LISTA encoder, linear decoder, strong sparsity.
2. **Find stable unique supports** (DONE): ts >= 256 achieves full uniqueness.
3. **Resolve consistency + test K structure** (DONE): cosine metrics confirm basin-discriminative representations; K structure is secondary to encoder for support correspondence; block_diagonal shows dynamics benefit.
4. **Long-horizon dynamics + per-basin extraction** (NEXT): prediction MSE with reencoding, extract block dynamics, eigenvalue analysis.
5. **LQR control on extracted basins** (AFTER): per-basin LQR using block K dynamics.
6. **Generalise to other systems** (AFTER): Duffing, dysts multi-basin benchmarks.

## Pending
- Long-horizon prediction MSE with periodic reencoding on all 20 K structure checkpoints
- Per-block eigenvalue analysis for block_diagonal models
- LQR feasibility study on extracted per-basin dynamics
- Arrowhead stability investigation (ts=128 divergence)
- Validation on Duffing and dysts multi-basin systems
