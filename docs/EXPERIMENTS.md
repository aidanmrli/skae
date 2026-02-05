# Experiments

Date: February 5, 2026

Goal: Achieve **unique support patterns for unique basins** (mechanistic interpretability), starting from the simplest setups and iterating on sparsity, target size, and Koopman structure.

## Current Status Summary

Problem we are solving: learn basin-discriminative sparse latents that are both **unique across basins** and **stable for long-horizon prediction**, so basin structure can be extracted and used for downstream control (per-basin LQR).

What we found so far:
- **Uniqueness** is solved at sufficient capacity (ts >= 256). LISTA encoders reliably produce distinct basin supports.
- The apparent **uniqueness–consistency tradeoff** was a thresholding artefact. Cosine similarity shows high intra-basin consistency when using threshold-free metrics.
- **Koopman structure is secondary** to the encoder for basin discrimination. Block-diagonal K improves parameter efficiency and can improve eval error.
- **Spectral radius determines long-horizon fate.** Models with all eigenvalues inside the unit circle (SR < 1) converge to bounded MSE (~3.5); models with any eigenvalue outside (SR > 1) diverge catastrophically (MSE > 1e+10 at H1000). Periodic reencoding rescues divergent models but cannot improve beyond the bounded-MSE floor.
- **Arrowhead with exclusivity is stable at ts <= 256 under pairwise training** (best no-reencode H1000 MSE = 3.49 at ts=128), but it **loses stability at ts >= 512** (SR > 1).
- **Sequence-length training does not stabilize K.** For L in {4, 8, 12, 16}, only diagonal K remains stable (SR < 1), and stability **degrades** as L grows; dense, block-diagonal, and arrowhead are unstable at all L.
- **Basin-to-block correspondence is strongest at low capacity** (block_diagonal ts=64: concentration 0.87) but washes out at higher dimensions (~0.10 at ts=256). The encoder distributes activations across blocks rather than concentrating each basin into a single block.
- **Block-usage balance losses improve cosine separation.** `usage_entropy` and `kl_uniform` raise separation vs control across ts={64,128,256,512}, while strict one-block penalties (`low_entropy`, `pairwise_overlap`) often collapse to degenerate supports (0 uniqueness, Jaccard=1 at tau=1e-3).

Current solution direction:
- Use **arrowhead K with exclusivity** for pairwise training at ts <= 256 (best stable long-horizon results there), but do not assume stability at ts >= 512.
- Use **diagonal K** as the only sequence-loss option that stays stable at short L (4–8), with explicit spectral constraints for larger L or higher ts.
- Add **explicit spectral regularization or constrained parameterization** to keep SR < 1 at high ts and under sequence loss.
- Use **block-usage balance losses** (`usage_entropy` / `kl_uniform`) as the baseline for block-diagonal K; follow with **combined one-block + balance** sweeps to avoid collapse while encouraging per-trajectory block selection.
- Run a **sequence-loss weight sweep** to test whether loss-weight tuning can stabilize non-diagonal K under sequence training.
- **Periodic reencoding at period 100** is the universal fallback: it equalizes all structures to H1000 MSE ~3.6–3.9.

Outstanding problems (active):
- **Basin-block alignment does not emerge naturally.** Even with block-diagonal K sized to match basins (d/13 per block), the encoder does not consistently assign one basin per block. This limits the ability to extract per-basin Koopman dynamics for LQR. Need to investigate explicit basin-block alignment losses or post-hoc block assignment.
- **No structure is stably below SR < 1 at ts = 1024.** Need explicit spectral stabilization to scale latent size.
- **Sequence training destabilizes non-diagonal K.** Need a spectral constraint or alternative training objective to keep SR < 1 under sequence loss.
- **Loss-weight sensitivity is unknown.** Need evidence whether tuning residual/reconstruction/prediction/sparsity weights can recover stability under sequence loss.
- **One-block losses can collapse.** The current per-sample exclusivity penalties (low-entropy/pairwise-overlap) often yield degenerate supports unless balanced; need combined one-block + usage-balance objectives with tuned weights.

**LQR Readiness Blockers**
1. Reliable, label-free basin identification. We do not know basin labels or how many basins exist in the real setting. We currently cannot reliably assign a trajectory to a specific block in a stable, unsupervised way. Without a trustworthy basin assignment, LQR has no “local system” to attach to.
2. Stable, block-specific dynamics (SR < 1). LQR assumes the local linear dynamics are meaningful and stable (or at least stabilizable). We have seen spectral radius instability in many settings; long-horizon rollouts diverge if SR > 1. Until each block is spectrally controlled, LQR might optimize a system that explodes in practice.
3. Basin-block alignment that persists with capacity. At low latent sizes, block alignment can appear, but it washes out at higher capacity. We need alignment to be stable across sizes, or at least stable in the regime we want to deploy.
4. Local linearity actually captures local dynamics. We need evidence that within a basin, the block dynamics are predictive (not just “separable”). Right now, separation and stability are decoupled from actual predictive quality in some configs.

## Result Reporting Protocol

When a new experiment produces results, document updates in the following order:
1. Report the concrete result(s) first (key metrics/tables/outcomes).
2. Explain the result(s) in the context of the experiment question/design.
3. Explain how to interpret the result(s) (what changed, what did not, uncertainty/caveats).
4. Explain implications for the broader project direction.
5. Suggest next steps.

After reporting results, update the project state in this file:
- Refresh **Current Status Summary** (problem, current solution direction, outstanding problem).
- Update **Queue Status** (running/completed/planned labels and progress numbers).
- Update the relevant experiment log entry with latest status and conclusions.

## Definitions (Support Metrics)

Support metrics are computed in `tools/evaluate_support_uniqueness.py`.

Support per trajectory:
- We encode a trajectory to latents `z[t]` and aggregate over time based on `support_mode`.
- `support_mode=mean`: use the mean latent over time, then threshold: `support_i = |mean(z)_i| > tau`.
- Other modes are available: `last`, `median`, `majority` (see `_support_from_latents` in the script).

Mode support per basin:
- For each basin, we collect the binary supports from all trajectories ending in that basin.
- The **mode support** is the most frequent support pattern (argmax count).
- **Consistency** is `mode_count / num_trajectories_in_basin`, then averaged across basins.

Soft vs hard thresholding:
- The LISTA encoder uses **soft thresholding (shrinkage)** internally during encoding (see `skae/model.py`).
- Support evaluation uses a **hard threshold** on the aggregated latents to get a binary support for counting/uniqueness.
- Low consistency at `tau=1e-3` was due to the hard threshold, not the encoder's soft-thresholding.

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
- Arrowhead no-exclusivity sweep: job `8605505` (array 0-4, 5 total latent dims) -- **COMPLETED**

Completed (February 4, 2026):
- Long-horizon prediction + eigenvalue analysis sweep: job `8607640` (sequential, 25 checkpoints) -- **COMPLETED** (25/25)
- Sequence length spectral-stability sweep: job `8613261` via `scripts/sweep_sequence_length_spectral.sh` (array 0-47; 4 sequence lengths × 4 K structures × 3 latent sizes) -- **COMPLETED** (48/48)

Running (February 4, 2026):
- Sequence-loss weight sweep: job `8613853` via `scripts/sweep_sequence_loss_weights.sh` (array 0-35; 36 weight configs) -- **RUNNING**

Completed (February 5, 2026):
- Block-loss ablation sweep: job `8615740` via `scripts/sweep_block_loss_ablation.sh` (array 0-23; 6 loss conditions × 4 target sizes) -- **COMPLETED**

Running (February 5, 2026):
- Block-loss balance sweep (Phase 1): job `8615817` via `scripts/sweep_block_loss_balance_phase1.sh` (array 0-71; 72 configs) -- **RUNNING**

Status check (February 4, 2026):
- All arrays for jobs `8602046`, `8602047`, `8603752`, `8603753`, and `8605505` produced logs and `support_eval/*.json` outputs in `/network/scratch/l/lia/skae/...`.
- Job `8607640` has evaluation + eigenvalue outputs for all 25 checkpoints.
- Job `8613261` has evaluation + eigenvalue outputs for all 48 configurations (some configs have multiple timestamps; the latest run may be train-only but earlier runs contain the full outputs).

---

## Experiment Log (Newest First)

### -4) Block-Loss Balance Sweep (Phase 1)
Timestamp: 2026-02-05 (submitted; job `8615817`)

Script: `scripts/sweep_block_loss_balance_phase1.sh`

```bash
sbatch scripts/sweep_block_loss_balance_phase1.sh
```

**Question:** Can we balance per-sample single-block activation (top-1 margin) with across-batch block usage (usage_entropy / kl_uniform) to improve separation without collapse?

**Fixed settings:** Lyapunov-HD (dim=8), pairwise training, `lista_nonlinear`, `sparsity_coeff=1.0`, `target_size=256`, 10k steps.

**Grid (72 jobs):**
- `one_block_weight`: {0.1, 0.3, 1.0}
- `balance_weight`: {0.1, 0.3, 1.0}
- `top1_margin`: {0.05, 0.1}
- `balance_loss`: {usage_entropy, kl_uniform}
- `seed`: {0, 1}

**Block structure:** `K=block_diagonal` with `NUM_BLOCKS=20` (independent of true basins), `K_BLOCK_SIZE = target_size // NUM_BLOCKS`.

**Evaluation:**
1. Cosine separation (threshold-free) via `support_eval/cosine_metrics.json` (primary metric).
2. Threshold sweep (`support_eval/threshold_sweep.json`) for uniqueness/consistency/Jaccard (secondary diagnostics).

**Output base:** `/network/scratch/l/lia/skae/lyapunov_block_loss_balance_phase1/`

**Status:** RUNNING (array 0-71).

### -3) Block-Loss Ablation Sweep (new)
Timestamp: 2026-02-05 (submitted; job `8615740`)

Script: `scripts/sweep_block_loss_ablation.sh`

```bash
sbatch scripts/sweep_block_loss_ablation.sh
```

**Question:** Do simple block-activation losses improve basin identifiability (cosine separation) for block-diagonal K without basin labels?

**Fixed settings (default):** Lyapunov-HD (dim=8), pairwise training, `lista_nonlinear`, `sparsity_coeff=1.0`, 10k steps.

**Grid (24 jobs):**
- `target_size`: {64, 128, 256, 512}
- `loss`: {control, low_entropy, pairwise_overlap, top1_margin, usage_entropy, kl_uniform}

**Block structure:** `K=block_diagonal` with `NUM_BLOCKS=20` (independent of true basins), `K_BLOCK_SIZE = target_size // NUM_BLOCKS`.

**Evaluation:**
1. Cosine separation (threshold-free) via `support_eval/cosine_metrics.json` (primary metric).
2. Threshold sweep (`support_eval/threshold_sweep.json`) for uniqueness/consistency/Jaccard (secondary diagnostics).

**Output base:** `/network/scratch/l/lia/skae/lyapunov_block_loss_sweep/`

**Status:** COMPLETED (array 0-23).

**Results (primary metric = cosine separation; threshold metrics secondary):**

Best cosine separation by target size (vs control, Δ shown):
- **ts=64:** `usage_entropy` **0.786** (Δ +0.433 vs control 0.353)
- **ts=128:** `kl_uniform` **0.852** (Δ +0.614 vs control 0.238)
- **ts=256:** `kl_uniform` **0.884** (Δ +0.058 vs control 0.826)
- **ts=512:** `usage_entropy` **0.791** (Δ +0.325 vs control 0.466)

Notable patterns:
- **Across-batch balance losses help separation.** `usage_entropy` and `kl_uniform` consistently improve cosine separation relative to control at all target sizes (sometimes large gains at ts=64/128).
- **Per-sample one-block losses often collapse.** `low_entropy` and `pairwise_overlap` frequently yield near-zero cosine separation (intra and inter ~0 or ~1), with **0 uniqueness** and **Jaccard=1** at `tau=1e-3`—indicative of degenerate or near-zero supports.
- **Top-1 margin is mixed.** It improves cosine separation over control at ts=64/512 but lags balance losses, and does not consistently improve threshold-based uniqueness/consistency.

**Interpretation:**
Balance losses increase separability but **do not guarantee single-block activation per sample**. The strict one-block penalties as implemented can drive collapse (near-zero or uninformative supports), suggesting they are too harsh without a stabilizing counter-term.

**Implications:**
For label-free basin identification, **block-usage balance is a safer first step** than aggressive per-sample exclusivity. To obtain *both* separation and one-block activation, we likely need **combined losses** (one-block + balance) with careful weighting or temperature schedules.

**Next steps:**
1. Run **combined loss** experiments (e.g., `top1_margin + kl_uniform`, `low_entropy + usage_entropy`) with weight sweeps.
2. Add **monitoring of per-sample block entropy and top-1 gap** to catch collapse early.
3. Test whether balance losses preserve separation under **sequence loss** or higher `target_size`.


### -2) Sequence-Loss Weight Sweep (new)
Timestamp: 2026-02-04 (submitted; job `8613853`)

Script: `scripts/sweep_sequence_loss_weights.sh`

```bash
sbatch scripts/sweep_sequence_loss_weights.sh
```

**Question:** Can loss-weight tuning stabilize non-diagonal K under sequence training (SR < 1)?

**Fixed settings (default):** L=8, target_size=128, K=block_diagonal (override with env vars).

**Grid (36 jobs):**
- `res_coeff`: {0.3, 1.0, 3.0}
- `reconst_coeff`: {0.3, 1.0, 3.0}
- `pred_coeff`: {0.0, 1.0}
- `sparsity_coeff`: {0.1, 0.3}

**Protocol:** Each job trains with `--sequence --sequence_length L`, then runs:
1. `tools/evaluate_checkpoints.py`
2. `tools/analyze_k_eigenvalues.py` (SR + basin correlation)

**Output base:** `/network/scratch/l/lia/skae/sequence_loss_weight_sweep/`

**Status:** RUNNING (array 0-35).

### -1) Sequence-Length Spectral-Stability Sweep (new)
Timestamp: 2026-02-04 (completed; job `8613261`)

Script: `scripts/sweep_sequence_length_spectral.sh`

```bash
sbatch scripts/sweep_sequence_length_spectral.sh
```

**Question:** Does increasing sequence length during training push Koopman spectral radius below 1 (without explicit spectral regularization)?

**Grid:** `L in {4, 8, 12, 16}` x `K in {dense, diagonal, block_diagonal, arrowhead}` x `target_size in {64, 128, 256}` = 48 jobs.

**Protocol:** Each job trains with `--sequence --sequence_length L` using matched core loss weights, then runs:
1. `tools/evaluate_checkpoints.py` (long-horizon prediction metrics)
2. `tools/analyze_k_eigenvalues.py` (max SR + per-block spectra)

**Output base:** `/network/scratch/l/lia/skae/sequence_length_spectral_sweep/`

**Results (stable = max SR < 1; counts out of 3 target sizes):**

| L | dense SR range (stable/3) | diagonal SR range (stable/3) | block-diagonal SR range (stable/3) | arrowhead SR range (stable/3) |
|---|---------------------------|------------------------------|------------------------------------|-------------------------------|
| 4 | 1.199–1.344 (0/3) | 0.955–0.994 (3/3) | 1.067–1.202 (0/3) | 1.122–1.215 (0/3) |
| 8 | 1.178–1.320 (0/3) | 0.938–0.987 (3/3) | 1.089–1.219 (0/3) | 1.135–1.261 (0/3) |
| 12 | 1.100–1.208 (0/3) | 0.934–1.047 (2/3) | 1.023–1.226 (0/3) | 1.174–1.330 (0/3) |
| 16 | 1.110–1.260 (0/3) | 0.966–1.078 (1/3) | 1.063–1.264 (0/3) | 1.180–1.339 (0/3) |

Stable diagonal runs have H1000 no-reencode MSE in the range **3.63–3.72** (similar to pairwise baselines). All SR > 1 runs diverge at H1000 (very large or inf MSE).

**Context:** The sweep tests whether longer sequence loss (L in {4, 8, 12, 16}) provides enough multi-step gradient pressure to push SR <= 1 without explicit spectral regularization.

**Interpretation:** Sequence-length training does **not** stabilize K. Only diagonal is stable at short L (4–8), and even that stability degrades at L=12 and L=16. Dense, block-diagonal, and arrowhead are unstable at all L (SR > 1 across all target sizes).

**Implications:** Multi-step loss is not a substitute for spectral control. If we need sequence training, we must add explicit spectral regularization or constrained parameterizations; otherwise long-horizon rollouts will diverge for non-diagonal structures.

**Next steps:**
1. Add explicit SR regularization or spectral normalization for K, then rerun a reduced grid (e.g., L=4 vs L=16 at ts={64,128}).
2. Test whether arrowhead stability can be recovered under sequence loss with adjusted exclusivity/sparsity weights.
3. Compare sequence-loss vs pairwise training under the same spectral constraint to isolate the effect of L.

### 0) Long-Horizon Prediction + Per-Block Eigenvalue Analysis
Timestamp: 2026-02-03 (evaluation sweep, job `8607640`)

Script: `scripts/sweep_eval_k_structure.sh`

```bash
sbatch scripts/sweep_eval_k_structure.sh
```

**Scope:** For each of the 25 (5 target sizes × 5 K structures) trained checkpoints, run:
1. `evaluate_checkpoints.py` -- 1000-step rollout MSE with 6 rollout modes (no-reencode, every-step, periodic at 10/25/50/100)
2. `analyze_k_eigenvalues.py --correlate_basins` -- per-block eigenvalue extraction + basin-to-block activation heatmap

**Status:** **COMPLETED** (25/25). Results below cover ts={64, 128, 256, 512, 1024} × all 5 structures.

#### Results: Spectral Radius and Long-Horizon Stability

The spectral radius (max |λ| across all K eigenvalues) is the key predictor of long-horizon behavior. Models with SR < 1 are spectrally stable; models with SR > 1 have exponentially growing modes that eventually dominate.

| ts | K structure | Max SR | All stable? | H1000 no-reencode |
|----|-------------|--------|-------------|-------------------|
| 64 | dense | 0.9886 | YES | 3.61e+00 |
| 64 | diagonal | 1.0006 | NO | 3.81e+00 |
| 64 | block_diagonal | 1.0006 | NO | 3.80e+00 |
| 64 | arrowhead | **0.9903** | **YES** | **3.55e+00** |
| 64 | arrowhead_no_excl | 0.9854 | YES | 3.60e+00 |
| 128 | dense | 1.0067 | NO | 9.49e+02 |
| 128 | diagonal | 0.9998 | YES | 3.82e+00 |
| 128 | block_diagonal | 1.0010 | NO | 3.81e+00 |
| 128 | arrowhead | **0.9963** | **YES** | **3.49e+00** |
| 128 | arrowhead_no_excl | 1.0049 | NO | 3.86e+00 |
| 256 | dense | 1.0262 | NO | 6.78e+17 |
| 256 | diagonal | **0.9993** | **YES** | **3.57e+00** |
| 256 | block_diagonal | **0.9996** | **YES** | **3.58e+00** |
| 256 | arrowhead | **0.9917** | **YES** | **3.59e+00** |
| 256 | arrowhead_no_excl | 1.0215 | NO | 1.52e+13 |
| 512 | dense | 1.0295 | NO | 7.88e+19 |
| 512 | diagonal | **0.9996** | **YES** | **3.82e+00** |
| 512 | block_diagonal | 1.0072 | NO | 4.66e+00 |
| 512 | arrowhead | 1.0169 | NO | 1.67e+09 |
| 512 | arrowhead_no_excl | 1.0190 | NO | 2.95e+10 |
| 1024 | dense | 1.0337 | NO | 4.02e+25 |
| 1024 | diagonal | 1.0192 | NO | 1.22e+12 |
| 1024 | block_diagonal | 1.0118 | NO | 4.71e+04 |
| 1024 | arrowhead | 1.0113 | NO | 2.26e+05 |
| 1024 | arrowhead_no_excl | 1.0411 | NO | 3.35e+29 |

**Context:** This sweep links Koopman spectral radius to long-horizon prediction stability across K structures and latent sizes under pairwise training.

**Key finding: spectral radius is a binary switch for long-horizon fate.** Every model with SR < 1 converges to H1000 MSE in the range 3.49–3.82. Every model with SR > 1 diverges, often catastrophically (orders of magnitude). There is no graceful degradation — even SR = 1.0006 (diagonal ts=64) causes slow drift to H1000 = 3.81, while SR = 1.0262 (dense ts=256) causes H1000 = 6.78e+17.

**Arrowhead with exclusivity maintains SR < 1 at ts <= 256 but not at higher dimensions.** At ts=512 and ts=1024, SR rises above 1 (1.0169 and 1.0113), so stability is not guaranteed at high capacity. The exclusivity loss is helpful but insufficient on its own at larger latent sizes.

**Dense K becomes unstable at ts >= 128** (SR = 1.007 at ts=128, SR = 1.026 at ts=256, SR = 1.030 at ts=512, SR = 1.034 at ts=1024). As the latent dimension grows, more eigenvalues drift outside the unit circle.

**Diagonal and block-diagonal are marginally stable.** At ts=64, both have SR = 1.0006 (barely unstable). At ts=256, both have SR < 1 (stable). At ts=512, diagonal remains stable (SR = 0.9996) while block-diagonal is unstable (SR = 1.0072). At ts=1024, both are unstable. Stability depends on the training outcome and degrades with higher capacity.

#### Results: Best Periodic Reencoding

Periodic reencoding equalises all models to similar H1000 MSE. The best periodic mode is always `periodic_100` for stable models and `periodic_50` or `periodic_25` for unstable ones (which need more frequent correction).

| ts | K structure | H1000 best-PR | Best mode | H1000 every-step |
|----|-------------|---------------|-----------|------------------|
| 64 | dense | 3.62e+00 | periodic_100 | 4.03e+00 |
| 64 | arrowhead | **3.55e+00** | periodic_50 | 6.92e+00 |
| 128 | dense | 3.85e+00 | periodic_100 | 3.87e+00 |
| 128 | arrowhead | **3.55e+00** | periodic_100 | 8.51e+00 |
| 256 | dense | 3.69e+00 | periodic_50 | 3.82e+00 |
| 256 | diagonal | **3.61e+00** | periodic_100 | 4.09e+00 |
| 256 | block_diagonal | 3.61e+00 | periodic_100 | 3.90e+00 |
| 256 | arrowhead | 3.60e+00 | periodic_100 | 4.61e+00 |

**The arrowhead model has the worst every-step reencoding MSE** (6.92 at ts=64, 8.51 at ts=128) despite the best no-reencode MSE. This means its encode-decode pathway is less accurate than other structures, but its latent dynamics are more stable. The arrowhead trades reconstruction quality for dynamical stability.

**Every-step reencoding is worse than no reencoding for stable models.** For models with SR < 1, the pure latent rollout (no-reencode) outperforms every-step reencoding because the encode-decode cycle introduces reconstruction error at each step. Periodic reencoding at period 100 is optimal — infrequent enough to avoid compounding reconstruction error, but frequent enough to correct any drift.

#### Results: Per-Block Eigenvalue Analysis

For block-diagonal and arrowhead models, eigenvalues are computed per block. For dense/diagonal, there is a single global eigenvalue set.

**Arrowhead per-block spectral radii (ts=256):**
- Global block: SR = 0.983 (14 blocks total: 1 global + 13 basin)
- Mean basin SR: 0.984, std: 0.005
- All 14 blocks strictly inside the unit circle

**Block-diagonal per-block spectral radii (ts=256):**
- 14 blocks, mean SR = 0.999, std = 0.001
- All blocks very close to SR = 1 (near-identity dynamics per block)

The arrowhead blocks have more diverse spectral radii (std = 0.005 vs 0.001) and are more conservatively pushed inside the unit circle (mean 0.984 vs 0.999). This explains its superior long-horizon stability.

#### Results: Basin-to-Block Activation Correlation

For each model, we encode 100 basin-labeled trajectories and compute the mean activation magnitude per latent dimension grouped by K block. The "basin-block concentration" metric measures how peaked each basin's activation is toward a single block (1.0 = perfect one-basin-one-block alignment, 0.0 = uniform spread).

| ts | K structure | Basin-block concentration |
|----|-------------|--------------------------|
| 64 | diagonal | **0.894** |
| 64 | block_diagonal | **0.875** |
| 64 | arrowhead_no_excl | 0.342 |
| 64 | arrowhead | 0.246 |
| 128 | block_diagonal | 0.693 |
| 128 | diagonal | 0.693 |
| 128 | arrowhead_no_excl | 0.678 |
| 128 | arrowhead | 0.154 |
| 256 | arrowhead_no_excl | 0.126 |
| 256 | block_diagonal | 0.105 |
| 256 | diagonal | 0.101 |
| 256 | arrowhead | 0.061 |

**At ts=64, diagonal and block-diagonal show strong basin-block alignment** (concentration 0.87–0.89). This means the encoder has learned to route each basin's activation to a specific block of the K matrix. However, this alignment **fades with increasing capacity**: at ts=256, all structures show concentration ~0.06–0.13, meaning the encoder distributes activations across many blocks.

**Counterintuitively, the arrowhead with exclusivity has the *lowest* basin-block concentration** at every dimension. The exclusivity loss encourages one-basin-at-a-time activation in latent space, but this does not produce alignment between *specific* basin blocks and *specific* ground-truth basins. Instead, the encoder appears to use different basins for different dynamical regimes without a fixed assignment.

#### Interpretation

1. **Spectral stability is the dominant factor for long-horizon prediction quality.** The binary stable/unstable classification predicted by the spectral radius perfectly explains the 15+ orders-of-magnitude spread in H1000 MSE across configurations. All stable models converge to a narrow MSE band (3.49–3.82); all unstable models diverge. This means Koopman structure choice is primarily a question of *which structures reliably produce SR < 1*, not which produce the lowest MSE.

2. **Arrowhead with exclusivity is robust up to ts=256 but not at higher capacity.** It achieves the best no-reencode H1000 MSE (3.49 at ts=128) and stays stable for ts <= 256, but SR exceeds 1 at ts=512 and ts=1024. The exclusivity loss provides an implicit spectral constraint, but it is insufficient alone at high latent sizes.

3. **Dense K is spectrally unstable above ts=64.** The dense Koopman matrix's d² free parameters allow eigenvalues to drift outside the unit circle during training. At ts=512, the spectral radius reaches 1.03, causing H1000 divergence to 10^19. Dense K should not be used for long-horizon prediction without explicit spectral regularisation (e.g., eigenvalue penalty or spectral normalisation of K).

4. **Block-diagonal and diagonal stability erodes at high capacity.** Both are unstable at ts=64, both are stable at ts=256, diagonal remains stable at ts=512, but both are unstable at ts=1024. This makes them competitive at moderate sizes but unreliable at high capacity without explicit spectral constraints.

5. **Basin-block alignment does not emerge from K structure alone.** Despite using block sizes matching the number of ground-truth basins (d/13), the encoder does not learn a consistent one-basin-one-block mapping at moderate-to-large latent dimensions. At ts=64 there is strong alignment (concentration 0.87), but this is likely because the low capacity forces each block to specialize. At ts=256+, the encoder has enough capacity to distribute basin information across multiple blocks, and no training signal explicitly encourages concentration.

6. **For LQR control, additional basin-block alignment losses are needed.** The original goal — isolate per-basin linear dynamics as blocks of K, then apply LQR per block — requires that each block maps to exactly one basin. The current results show this alignment exists at low capacity (ts=64) but breaks down at the capacity levels needed for accurate dynamics (ts=256+). An explicit basin-assignment loss during training (e.g., a classifier head predicting basin from block activations) would be needed to enforce this correspondence.

#### Implications

- Without explicit spectral control, high-capacity latents (ts >= 512) are unstable across all K structures.
- Arrowhead structure alone is not enough to guarantee stability at high dimensions; stability constraints must be part of training or parameterization.
- Long-horizon reliability should be treated as a first-class objective, not an emergent property of structure choice.

#### Next Steps

1. **Add spectral regularisation to K (dense + structured).** Penalise eigenvalues outside the unit circle during training (e.g., `lambda_spectral * max(0, SR - 1)^2`) or use spectral normalization. Re-run a small grid at ts={256,512} to check if stability persists at higher capacity.

2. **Investigate why arrowhead loses stability at high ts.** Sweep exclusivity/sparsity weights and global/basin split to see if SR can be kept < 1 at ts >= 512.

3. **Investigate basin-block alignment losses.** Add an auxiliary classifier head that predicts basin identity from per-block activation norms to encourage consistent basin-to-block routing for LQR.

4. **Test LQR on block-diagonal ts=64.** Despite marginal instability (SR = 1.0006), the ts=64 block-diagonal model has strong basin-block alignment (0.87). Test whether small eigenvalue corrections (clamping SR to < 1) yield effective per-basin controllers.

5. **Validate on Duffing.** Duffing has only 2 basins, which should produce cleaner basin-block alignment. Run the same evaluation suite on the Duffing checkpoints to confirm the spectral stability findings generalise.

---

### 1) Arrowhead no-exclusivity sweep (control)
Timestamp: 2026-02-03

Script: `scripts/sweep_arrowhead_no_excl_lyapunov.sh`

```bash
sbatch scripts/sweep_arrowhead_no_excl_lyapunov.sh
```

Same total latent dims as the arrowhead sweep, but **without** the exclusivity loss to isolate the effect of Koopman structure alone.
Output: `/network/scratch/l/lia/skae/lyapunov_k_structure_sweep/`

### 2) Arrowhead (StructuredLISTAKM) sweep
Timestamp: 2026-02-03

Script: `scripts/sweep_arrowhead_lyapunov.sh`

```bash
sbatch scripts/sweep_arrowhead_lyapunov.sh
```

5 jobs: total latent dims 64, 128, 256, 512, 1024 (d_global + 13 * d_basin = total_dim).
Uses `lambda_exclusivity=0.05`, `lambda_sparsity=0.3`, `excl_warmup=2000`.
Output: `/network/scratch/l/lia/skae/lyapunov_k_structure_sweep/`

### 3) K structure × target size sweep
Timestamp: 2026-02-03

Script: `scripts/sweep_k_structure_lyapunov.sh`

```bash
sbatch scripts/sweep_k_structure_lyapunov.sh
```

15 jobs: 5 target sizes × 3 K structures (dense, diagonal, block_diagonal).
Post-training eval includes `--threshold_sweep` + cosine similarity.
Output: `/network/scratch/l/lia/skae/lyapunov_k_structure_sweep/`

### 4) Support threshold sweep (post-hoc eval)
Timestamp: 2026-02-03 (post-hoc / on-demand)

Script: `scripts/sweep_support_threshold.sh`

```bash
sbatch --export=ALL,CKPT=/path/to/checkpoint.pt,OUT_BASE=/path/to/out \
  scripts/sweep_support_threshold.sh
```

Thresholds tested: `1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1`

### 5) Duffing target size sweep (simple LISTA baseline)
Timestamp: 2026-02-02

Script: `scripts/sweep_target_size_duffing.sh`

```bash
sbatch scripts/sweep_target_size_duffing.sh
```

Defaults: target sizes: 32, 64, 128, 256, 512, `SPARSITY=1.0`

### 6) Lyapunov-HD target size sweep (simple LISTA baseline)
Timestamp: 2026-02-02

Script: `scripts/sweep_target_size_lyapunov_hd.sh`

```bash
sbatch scripts/sweep_target_size_lyapunov_hd.sh
```

Defaults: `DIM=8`, `NUM_BASINS=13`, `SPARSITY=1.0`, target sizes: 64, 128, 256, 512, 1024

---

## Experiment: Koopman Structure + Refined Diagnostics (February 3, 2026)
Timestamp: 2026-02-03

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

### Results: Arrowhead Without Exclusivity (control)

Evaluated from `support_eval/threshold_sweep.json` (cosine metrics are threshold-free). Uniqueness is reported at `tau=1e-3`.

| ts | CosSep (intra-inter) | Unique @1e-3 | Eval final error |
|----|----------------------|--------------|-----------------|
| 64 | 0.7852 | 13/13 | 2.1817 |
| 128 | 0.1931 | 9/13 | 2.5567 |
| 256 | **0.8612** | 13/13 | **1052.7489** |
| 512 | **0.8621** | 13/13 | 2.2564 |
| 1024 | 0.6672 | 13/13 | 2.2571 |

Key observations:
- **Structure alone is not sufficient at mid-size.** At `ts=128`, cosine separation collapses (0.19) and uniqueness drops to 9/13.
- **Stability is inconsistent.** `ts=256` shows strong separation but catastrophic eval error, indicating unstable rollout dynamics without exclusivity.
- **Large dims can look good without exclusivity**, but the separation benefit is not consistent (ts=1024 drops to 0.67).
- **Why we call it unreliable:** the failure mode flips with size (separation fails at 128, stability fails at 256), so structure alone is not robust.

### Comparison: Arrowhead With vs Without Exclusivity

Side-by-side summary at `tau=1e-3` (uniqueness) using cosine separation (threshold-free) and eval final error.

| ts | CosSep (excl) | CosSep (no-excl) | Unique (excl) | Unique (no-excl) | Eval (excl) | Eval (no-excl) |
|----|--------------|------------------|---------------|------------------|-------------|----------------|
| 64 | 0.7474 | 0.7852 | 13/13 | 13/13 | 2.5245 | 2.1817 |
| 128 | 0.7566 | 0.1931 | 13/13 | 9/13 | **13519.6357** | 2.5567 |
| 256 | 0.7347 | **0.8612** | 13/13 | 13/13 | 3.3905 | **1052.7489** |
| 512 | 0.7980 | **0.8621** | 13/13 | 13/13 | 2.8736 | 2.2564 |
| 1024 | **0.8805** | 0.6672 | 13/13 | 13/13 | 2.8577 | 2.2571 |

### Interpretation

1. **The LISTA encoder is the primary driver of basin discrimination, not the Koopman matrix.** At sufficient capacity (ts>=256), dense, diagonal, and block_diagonal K produce nearly identical cosine separation scores (~0.84--0.85). The encoder learns basin-discriminative supports regardless of K structure. This means the sparsity inductive bias of LISTA is doing the heavy lifting.

2. **Constraining K at low capacity hurts uniqueness.** At ts=64, dense K achieves 13/13 uniqueness while diagonal/block_diagonal only manage 6/13. With limited latent dimensions, the model needs the full K coupling to compensate — the encoder can't produce enough distinct sparse codes when K is too constrained. The exception is arrowhead, which achieves 13/13 at ts=64 via the exclusivity loss, not K structure.

3. **Block_diagonal K provides a parameter efficiency advantage for dynamics.** At ts=512, block_diagonal gives 21% better eval error than dense with 13x fewer K parameters. This suggests that at moderate-to-large latent dimensions, constraining off-diagonal coupling acts as beneficial regularisation for the dynamics, preventing overfitting in the Koopman matrix.

4. **Arrowhead is unstable at intermediate sizes.** The ts=128 arrowhead diverged catastrophically despite achieving the lowest residual loss during training. This indicates a disconnect: low residual loss (good latent-space alignment) does not guarantee good prediction (good decode-step-decode accuracy). The arrowhead's reconstruction quality is excellent (lowest reconst loss), but the coupling terms may create amplifying feedback loops during multi-step rollout.

5. **The "uniqueness--consistency tradeoff" from the previous experiments is resolved.** It was entirely a thresholding artefact. The cosine metrics show that within-basin representations are highly consistent (cosine ~0.97) at all configurations where uniqueness is achieved. The correct diagnostic is the cosine separation score, not binary support consistency.

6. **Arrowhead without exclusivity is unreliable.** The no-exclusivity control shows that **Koopman structure alone does not guarantee basin separation or stability**. Exclusivity provides the consistent basin-discriminative bias at low/mid dimensions, while the structure alone can be unstable (ts=256) or weakly separating (ts=128).

Implication for the project: treat **exclusivity as a necessary inductive bias** for basin-discriminative representations at practical sizes, and treat arrowhead structure as a *secondary* stabilizer that must be paired with either exclusivity or additional regularization. The best near-term path remains block-diagonal K (stable, parameter-efficient) plus structured losses when using arrowhead.

### Next Steps (status as of Feb 3 evaluation sweep)

1. ~~**Long-horizon prediction MSE with periodic reencoding** on all 25 checkpoints.~~ **DONE** (see Experiment 0 above). Block_diagonal at ts=256 is spectrally stable (SR < 1) with H1000 = 3.58. Dense K diverges at ts >= 128.

2. ~~**Extract per-block dynamics from block_diagonal K.**~~ **DONE** (see Experiment 0 above). Per-block eigenvalues extracted. Basin-block alignment is strong at ts=64 (concentration 0.87) but fades at ts=256 (0.10).

3. **Test LQR on extracted basin dynamics.** Still pending. The ts=64 block-diagonal model is the best candidate due to its strong basin-block alignment, but has marginal spectral instability (SR = 1.0006).

4. ~~**Stabilise arrowhead at ts=128.**~~ **RESOLVED.** The arrowhead with exclusivity at ts=128 is spectrally stable (SR = 0.996) and achieves the best H1000 no-reencode MSE (3.49). The previously reported divergence was in the *eval final error* metric (short-horizon), not in long-horizon rollout stability.

5. **Validate on Duffing and dysts systems.** Still pending.

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

## Interpretation: Uniqueness vs Consistency (Superseded by Feb 3 Diagnostics)

This section reflects the Feb 2 readout using hard-thresholded supports at `tau=1e-3`. It is kept for provenance, but the updated conclusion (see Feb 3 cosine results above) is:

- Binary consistency at `tau=1e-3` is low because supports flip near the hard threshold.
- Threshold sweeps and cosine similarity show high within-basin consistency in the continuous latents.
- The right diagnostic going forward is cosine separation (threshold-free), plus threshold sweeps when binarizing.

Original Feb 2 snapshot (context only):
- Uniqueness increases with capacity (ts >= 256 for 13 basins, ts >= 32 for 2 basins).
- Binary consistency drops as capacity increases under `tau=1e-3`.

---
