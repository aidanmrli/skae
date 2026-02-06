# Experiments

Date: February 7, 2026

Goal: Achieve **unique support patterns for unique basins** (mechanistic interpretability), starting from the simplest setups and iterating on sparsity, target size, and Koopman structure.

## Current Status Summary

Problem we are solving: learn basin-discriminative sparse latents that are both **unique across basins** and **stable for long-horizon prediction**, so basin structure can be extracted and used for downstream control (per-basin LQR).

Assumption split:
- **Training/deployment target:** basin count and basin labels are unknown.
- **Benchmark evaluation:** using known basin counts/labels is acceptable for diagnostics.

What we found so far:
- **Uniqueness** is solved at sufficient capacity (ts >= 256). LISTA encoders reliably produce distinct basin supports.
- The apparent **uniqueness–consistency tradeoff** was a thresholding artefact. Cosine similarity shows high intra-basin consistency when using threshold-free metrics.
- **Koopman structure is secondary** to the encoder for basin discrimination. Block-diagonal K improves parameter efficiency and can improve eval error.
- **Spectral radius determines long-horizon fate.** Models with all eigenvalues inside the unit circle (SR < 1) converge to bounded MSE (~3.5); models with any eigenvalue outside (SR > 1) diverge catastrophically (MSE > 1e+10 at H1000). Periodic reencoding rescues divergent models but cannot improve beyond the bounded-MSE floor.
- **Arrowhead with exclusivity is stable at ts <= 256 under pairwise training** (best no-reencode H1000 MSE = 3.49 at ts=128), but it **loses stability at ts >= 512** (SR > 1).
- **Sequence-length training does not stabilize K.** For L in {4, 8, 12, 16}, only diagonal K remains stable (SR < 1), and stability **degrades** as L grows; dense, block-diagonal, and arrowhead are unstable at all L.
- **Diagonal K Stage-2 add-on is complete (72 runs).** `diag_c1` reached `M2=M3=1.0` and had highest mean `M4` (`0.8618`) vs `ah_prag` (`0.8285`) and `bd_c2` (`-10.7638`, heavy-tailed failures), but the pre-registered rule still returned **no clear winner** because `M2/M3` remained saturated and the practical threshold gate was not met.
- **Sequence-loss weight tuning alone does not stabilize non-diagonal K.** In the L=8 block-diagonal weight sweep (36 configs), **0/36** reached SR < 1 (best SR = 1.0014), so coefficient tuning is insufficient without explicit spectral control.
- **Basin-to-block concentration is a secondary diagnostic, not the objective.** It is strongest at low capacity (block_diagonal ts=64: concentration 0.87) and washes out at higher dimensions (~0.10 at ts=256), while basin-support uniqueness can still be strong.
- **Block-usage balance losses improve cosine separation.** `usage_entropy` and `kl_uniform` raise separation vs control across ts={64,128,256,512}, while strict one-block penalties (`low_entropy`, `pairwise_overlap`) often collapse to degenerate supports (0 uniqueness, Jaccard=1 at tau=1e-3).
- **Combined top-1 + balance losses can improve separability at ts=256, but are sensitive.** Phase 1 reached best CosSep **0.8826** (vs ts=256 control 0.8256, +0.057), but some weight settings still collapsed (worst CosSep 0.222, uniqueness 0.5 at tau=1e-3).
- **LQR-readiness decision pipeline is now complete (Stages 0-3).** On the Lyapunov Stage 2 decision subset (144 runs), both finalists (`bd_c2`, `ah_prag`) achieved `M2=M3=1.0`; pairwise `M4` difference was not significant (`bd_c2-ah_prag=-11.593`, 95% CI `[-34.750, 0.001]`), so the pre-registered rule returned **no clear winner** and selected fallback `bd_c2`.
- **Structure-only arrowhead vs block-diagonal (Stage 1) was also a tie on primary decision metrics.** `ah_iso`, `bd_c1`, and `bd_c2` all reached `M2=M3=1.0`; `M4` means were close (`ah_iso=0.728`, `bd_c2=0.735`) with overlapping CIs.
- **Duffing transfer (Stage 3) favored arrowhead on `M4`, but below the practical threshold.** Pairwise `M4` difference (`bd_c2-ah_prag=-0.0849`, 95% CI `[-0.1100, -0.0632]`) favored `ah_prag`, but the absolute gain (<0.10) did not meet the pre-registered practical threshold, so the final decision remained fallback `bd_c2`.
- **Exclusivity regularization does not improve LQR control metrics and significantly hurts basin separation.** In a 144-run ablation (129 completed), `ah_prag` vs `ah_prag_no_excl` showed no significant `M4` difference (Cohen's d = 0.07), while no-exclusivity achieved significantly better CosSep (0.80 vs 0.74, Cohen's d = −0.97, p < 0.001) and better recovery `M5` (0.55 vs 0.48, Cohen's d = −0.82, p < 0.001). Exclusivity's only advantage is a modest spectral-stability edge (45% vs 61% unstable runs).

Current solution direction:
- Use **`bd_c2` as the default simpler arm** for downstream label-free control experiments (current fallback winner under the pre-registered rule), while keeping `ah_prag` as the primary comparator.
- Keep **`diag_c1` as a stability-first comparator** in LQR-readiness studies; it is competitive and slightly stronger on Stage-2 `M4`, but not yet a pre-registered-rule winner.
- Use **arrowhead K without exclusivity** as a strong pairwise candidate at low-to-mid size; the exclusivity ablation shows that exclusivity hurts CosSep and M5 without improving M4, so the default arrowhead arm should disable exclusivity (`ah_prag_no_excl`) unless spectral stability is the primary concern.
- Use **diagonal K** as the only sequence-loss option that stays stable at short L (4–8), with explicit spectral constraints for larger L or higher ts.
- Add **explicit spectral regularization or constrained parameterization** to keep SR < 1 at high ts and under sequence loss.
- Use **block-usage balance losses** (`usage_entropy` / `kl_uniform`) as the baseline for block-diagonal K; follow with **combined one-block + balance** sweeps to avoid collapse while encouraging per-trajectory block selection.
- Keep **combined one-block + balance** in the robust region (small top-1 margin, moderate one-block weight) and avoid aggressive settings that reduce uniqueness.
- Treat **sequence-loss weight tuning as insufficient by itself**; move to explicit spectral constraints/parameterizations to enforce SR < 1 for non-diagonal K.
- **Periodic reencoding** is the fallback: `periodic_100` is usually best for spectrally stable models, while unstable models often need shorter periods (`periodic_50`/`periodic_25`) to remain bounded.

Outstanding problems (active):
- **Reliable label-free regime assignment from sparse supports is still open.** We need a robust unsupervised mapping from support patterns to local control models when basin count/labels are unknown at training/deployment time.
- **No clear LQR winner between arrowhead and block-diagonal under the pre-registered threshold.** Current decisions on Lyapunov and Duffing both fall back to the simpler arm (`bd_c2`) because metric gains do not pass the practical threshold gate.
- **Metric/rule mismatch remains after adding diagonal.** Even when `diag_c1` improves `M4` vs `ah_prag`, `M2/M3` saturation plus practical-threshold gating still prevents a decisive rule-based winner.
- **Closed-loop cost reduction (`M4`) is heavy-tailed for some block-diagonal settings.** Stage 2 includes catastrophic outlier regimes (notably `bd_c2`, `ts=512`, `B_proxy=13`), which widen CIs and prevent confident selection.
- **No structure is stably below SR < 1 at ts = 1024.** Need explicit spectral stabilization to scale latent size.
- **Sequence training destabilizes non-diagonal K.** Need a spectral constraint or alternative training objective to keep SR < 1 under sequence loss.
- **Combined one-block + balance remains brittle.** Even with top1+balance, some settings lose uniqueness and separation; need a robust objective region that improves separability without collapse.
- **Exclusivity regularization is net-negative for arrowhead.** It worsens basin separation and recovery success without improving closed-loop control, suggesting the exclusivity loss interferes with the encoder's ability to produce orthogonal basin representations.

**LQR Readiness Blockers**
1. Reliable, label-free basin/regime identification. We do not know basin labels or how many basins exist in the real setting. We currently cannot reliably assign a trajectory to a support-defined regime in a stable, unsupervised way. Without a trustworthy assignment, LQR has no “local system” to attach to.
2. Discriminative power of current control metrics. In completed Stage 1/2/3 runs, `M2` and `M3` saturated at 1.0 for all finalists, so they did not separate architectures; we need harder stress tests/perturbations.
3. Local linearity actually captures local dynamics. We need evidence that support-conditioned local dynamics are predictive (not just “separable”), especially under misspecified `B_proxy`.
4. Robustness across capacity/seeds. We still see large variance/outliers in `M4` for some settings, so the pipeline is not yet stable enough for a confident architecture call.
5. Decision-rule discriminativeness after diagonal inclusion. We now have diagonal comparisons, but the current pre-registered gate still cannot make a decisive architecture call when `M2/M3` saturate.

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

Completed (February 5, 2026):
- Block-loss ablation sweep: job `8615740` via `scripts/sweep_block_loss_ablation.sh` (array 0-23; 6 loss conditions × 4 target sizes) -- **COMPLETED**
- Sequence-loss weight sweep: job `8613853` via `scripts/sweep_sequence_loss_weights.sh` (array 0-35; 36 weight configs) -- **COMPLETED**
- Block-loss balance sweep (Phase 1): job `8615817` via `scripts/sweep_block_loss_balance_phase1.sh` (array 0-71; 72 configs) -- **COMPLETED**

Completed (February 6, 2026):
- LQR decision Stage 0 script: `scripts/sweep_lqr_decision_stage0.sh` -- job `8620570` (array 0-3; 4 jobs) -- **COMPLETED** (4/4)
- LQR decision Stage 1 script: `scripts/sweep_lqr_decision_stage1.sh` -- job `8620569` (array 0-35; 36 jobs) -- **COMPLETED** (36/36)
- LQR decision Stage 2 script: `scripts/sweep_lqr_decision_stage2.sh` -- job `8621121` (array 0-143; 144 jobs; `BD_STAR=bd_c2`) -- **COMPLETED** (144/144)
- LQR decision Stage 3 script: `scripts/sweep_lqr_decision_stage3.sh` -- job `8621236` (array 0-71; 72 jobs; dependency `afterany:8621121`) -- **COMPLETED** (72/72)
- LQR decision final collection: `scripts/collect_lqr_decision_after_stage3.sh` -- job `8621237` (dependency `afterany:8621236`) -- **COMPLETED**
- LQR decision Stage 2 diagonal-only training sweep: `scripts/sweep_lqr_decision_stage2_diag_only.sh` -- job `8622447` (array 0-71; 72 jobs) -- **COMPLETED** (72/72 checkpoints; post-train eval interrupted by runner shell parse error)
- LQR decision Stage 2 diagonal-only LQR eval recovery: `scripts/sweep_lqr_decision_stage2_diag_lqr_eval_only.sh` -- job `8622706` (array 0-71; 72 jobs) -- **COMPLETED** (72/72)

Completed (February 7, 2026):
- Stage 2 arrowhead exclusivity attribution sweep: `scripts/sweep_lqr_decision_stage2_excl_ablation.sh` -- job `8622549` (array 0-143; 144 jobs; arms `{ah_prag, ah_prag_no_excl}`) -- **COMPLETED** (129/144; 15 failed due to shell parse error at LQR eval stage, training + eigenvalue analysis intact for all 144)

Submitted (February 7, 2026):
- Exclusivity ablation recovery (support_eval + cosine_diag + LQR eval for 15 failed runs): `scripts/sweep_lqr_decision_stage2_excl_ablation_recovery.sh` -- job `8622997` (array 0-14)

Status check (February 6, 2026):
- All arrays for jobs `8602046`, `8602047`, `8603752`, `8603753`, and `8605505` produced logs and `support_eval/*.json` outputs in `/network/scratch/l/lia/skae/...`.
- Job `8607640` has evaluation + eigenvalue outputs for all 25 checkpoints.
- Job `8613261` has evaluation + eigenvalue outputs for all 48 configurations (some configs have multiple timestamps; the latest run may be train-only but earlier runs contain the full outputs).
- Job `8613853` has training + checkpoint evaluation + eigenvalue outputs for all 36 configs.
- Job `8615817` has training + support-eval outputs (`cosine_metrics.json`, `threshold_sweep.json`) for all 72 configs.
- LQR decision pipeline jobs `8620570`, `8620569`, `8621121`, `8621236`, and `8621237` are complete; collection produced:
  - `results/lqr_decision/{summary_decision_table.csv,lqr_readiness_summary.json,final_decision.md}` (Stage 2 Lyapunov decision),
  - `results/lqr_decision/stage1_lyapunov/*` (Stage 1 structure isolation),
  - `results/lqr_decision/stage3_duffing/*` (Stage 3 Duffing transfer).
- Diagonal Stage-2 add-on (`diag_c1`) is complete: job `8622447` produced 72 checkpoints and job `8622706` produced 72 LQR-readiness summaries under `/network/scratch/l/lia/skae/lqr_decision/stage2_lyapunov_diag_c1_*`.
- Diagonal pairwise decision artifacts were generated at:
  - `results/lqr_decision_stage2_diag_vs_bd/{summary_decision_table.csv,lqr_readiness_summary.json,final_decision.md}`
  - `results/lqr_decision_stage2_diag_vs_ah/{summary_decision_table.csv,lqr_readiness_summary.json,final_decision.md}`
- Exclusivity ablation sweep (job `8622549`) completed 129/144 runs. All 144 output directories populated with training artifacts. 15 runs missing post-train eval (support_eval, cosine_diag, lqr_readiness) due to shell parse error; recovery submitted as job `8622997`.

---

## Experiment Log (Newest First)

### -8) Stage-2 Arrowhead Exclusivity Attribution Sweep (completed)
Timestamp: 2026-02-06

Scripts:
- Main sweep: `scripts/sweep_lqr_decision_stage2_excl_ablation.sh` (job `8622549`, array 0-143)
- Recovery (15 failed post-train evals): `scripts/sweep_lqr_decision_stage2_excl_ablation_recovery.sh` (job `8622997`, array 0-14)

**Question**
- In the practical Stage-2 grid, how much of arrowhead control-readiness behavior is due to exclusivity regularization versus structure+sparsity alone?

**Design**
- Arms: `ah_prag` (with exclusivity, `lambda_excl=0.05`) vs `ah_prag_no_excl` (same structured layout and sparsity settings, exclusivity disabled, `lambda_excl=0.0`).
- System: Lyapunov.
- Grid: `target_size in {128,256,512}` x `B_proxy in {8,13,20}` x `seed in {0..7}`.
- Total runs: 144 (72 per arm).
- Output base: `/network/scratch/l/lia/skae/lqr_decision_excl_ablation`.

**Status**
- 129/144 completed; 15 failed at LQR eval stage due to shell parse error (training + eigenvalue analysis intact for all 144). Recovery job `8622997` submitted for the 15 failed runs.
- Failed indices concentrated in `ts=128, bp=8` (6 `ah_prag` seeds, all 8 `ah_prag_no_excl` seeds) plus `ah_prag ts=256 bp=8 seed=5`.

**1) Concrete results (129 completed runs: 65 ah_prag, 64 ah_prag_no_excl)**

Overall per-arm LQR readiness metrics:

| Metric | ah_prag (excl) | ah_prag_no_excl | Diff (excl − no) | Significance |
|--------|---------------|-----------------|-------------------|-------------|
| M2 (feasibility) | 1.0000 | 1.0000 | 0.0000 | — |
| M3 (CL stability) | 1.0000 | 1.0000 | 0.0000 | — |
| M4 (CL cost reduction) | 0.8264 ± 0.027 | 0.8233 ± 0.028 | +0.0023 | n.s. (d = 0.07) |
| M5 (recovery success) | 0.4826 ± 0.158 | 0.5493 ± 0.159 | −0.0657 | p < 0.001 (d = −0.82) |

Overall cosine separation metrics:

| Metric | ah_prag (excl) | ah_prag_no_excl | Diff |
|--------|---------------|-----------------|------|
| CosSep | 0.7404 ± 0.064 | 0.8011 ± 0.036 | −0.0564 (p < 0.001, d = −0.97) |
| IntraCos | 0.9901 ± 0.005 | 0.9796 ± 0.012 | +0.0105 |
| InterCos | 0.2497 ± 0.062 | 0.1785 ± 0.031 | +0.0712 |

Spectral stability:

| Metric | ah_prag (excl) | ah_prag_no_excl |
|--------|---------------|-----------------|
| Max SR (mean) | 1.0007 ± 0.010 | 1.0033 ± 0.008 |
| Max SR (median) | 0.9982 | 1.0019 |
| Runs with SR > 1 | 29/65 (45%) | 39/64 (61%) |

M4 by target_size:

| ts | ah_prag | ah_prag_no_excl |
|----|---------|-----------------|
| 128 | 0.8350 ± 0.027 (n=18) | 0.8165 ± 0.039 (n=16) |
| 256 | 0.8231 ± 0.033 (n=23) | 0.8284 ± 0.024 (n=24) |
| 512 | 0.8230 ± 0.020 (n=24) | 0.8226 ± 0.023 (n=24) |

CosSep by target_size:

| ts | ah_prag | ah_prag_no_excl |
|----|---------|-----------------|
| 128 | 0.6770 ± 0.068 (n=18) | 0.7858 ± 0.034 (n=16) |
| 256 | 0.7358 ± 0.036 (n=23) | 0.7951 ± 0.044 (n=24) |
| 512 | 0.7922 ± 0.028 (n=24) | 0.8172 ± 0.018 (n=24) |

M5 by B_proxy:

| bp | ah_prag | ah_prag_no_excl |
|----|---------|-----------------|
| 8 | 0.3457 ± 0.050 | 0.3732 ± 0.057 |
| 13 | 0.3892 ± 0.072 | 0.4894 ± 0.083 |
| 20 | 0.6728 ± 0.046 | 0.7268 ± 0.041 |

Pairwise matched-pair tests (63 pairs):
- **M4:** Paired t = 0.573 (p > 0.05), Cohen's d = 0.07, win rate 33/63 for excl. **No significant difference.**
- **CosSep:** Paired t = −7.661 (p < 0.001), Cohen's d = −0.97, no-excl wins 54/63. **Large significant advantage for no-exclusivity.**
- **M5:** Paired t = −6.498 (p < 0.001), Cohen's d = −0.82, no-excl wins 50/63. **Large significant advantage for no-exclusivity.**

**2) Context in experiment design**

This ablation directly isolates the effect of exclusivity regularization within the arrowhead architecture. Both arms share the same structured layout (B_proxy basin blocks), sparsity settings (`lambda_sparsity=0.3`, `lambda_global=1e-4`, `lambda_local=1e-3`), and training protocol. The only difference is `lambda_excl=0.05` (ah_prag) vs `lambda_excl=0.0` (ah_prag_no_excl).

**3) Interpretation**

- **Exclusivity does not help closed-loop control performance (M4).** Both arms achieve nearly identical cost reduction (~82–83%), with no statistically significant difference. This means exclusivity is not contributing to the quality of support-conditioned local linear models used by LQR.
- **Exclusivity significantly hurts basin separation (CosSep).** The mechanism is increased inter-basin cosine similarity: exclusivity raises InterCos from 0.18 to 0.25, making basin representations less orthogonal. The intra-basin cohesion is slightly better with exclusivity (0.99 vs 0.98), but this small gain is overwhelmed by the loss of inter-basin separation.
- **Exclusivity significantly hurts state recovery (M5).** The recovery success rate drops from 55% to 48% with exclusivity, consistent across all B_proxy values.
- **Exclusivity provides a modest spectral stability advantage.** Fewer runs have SR > 1 (45% vs 61%), and the median spectral radius is just below 1.0 for exclusivity vs just above for no-exclusivity. However, this stability advantage does not translate to any downstream control benefit.
- **B_proxy = 20 is the strongest operating point** for both arms across M4 and M5, suggesting that finer-grained block decomposition improves LQR outcomes.

**4) Project implications**

- **Exclusivity should be removed from the default arrowhead configuration.** It introduces a net cost (worse separation, worse recovery) with no control benefit. The arrowhead structure + sparsity alone is sufficient for competitive LQR performance.
- **The prior `ah_prag` results in Stage 2 and Stage 3 likely underestimate arrowhead's potential.** The comparison against `bd_c2` was conducted with exclusivity enabled; without it, arrowhead would have stronger basin separation and recovery.
- **Re-running the `bd_c2` vs arrowhead decision with `ah_prag_no_excl` may change the outcome**, since CosSep and M5 are substantially better without exclusivity.

**5) Re-run: `bd_c2` vs `ah_prag_no_excl` pairwise decision**

After recovery job `8622997` completed all 15 failed runs, we re-ran the Stage-2 pairwise decision tool comparing `bd_c2` (72 runs from original pipeline) vs `ah_prag_no_excl` (72 runs from this ablation, now all complete).

Artifacts: `results/lqr_decision_no_excl_vs_bd/{final_decision.md,summary_decision_table.csv,lqr_readiness_summary.json}`

| Metric (bd_c2 − ah_prag_no_excl) | Mean Diff | 95% CI | CI excludes 0 |
|---|---:|---:|:---:|
| M2 (feasibility) | 0.0000 | [0.0000, 0.0000] | no |
| M3 (CL stability) | 0.0000 | [0.0000, 0.0000] | no |
| M4 (cost reduction) | −11.585 | [−34.739, 0.009] | no |

Arm-level M4 means: `ah_prag_no_excl = 0.823`, `bd_c2 = −10.764`.

**Decision: no clear winner; fallback `bd_c2`** (unchanged from the prior `ah_prag` comparison).

The result is nearly identical to the original `bd_c2 vs ah_prag` comparison (`diff = −11.593`, CI `[-34.750, 0.001]`). The formal decision does not change because `bd_c2`'s catastrophic M4 outliers (ts=512, bp=13: mean M4 = −116.8) dominate the bootstrap CI, making it impossible to exclude 0. The arrowhead arm (with or without exclusivity) is far more reliable on M4 (all runs in the 0.79–0.85 range), but the current CI-based decision rule cannot distinguish "arm B is much more reliable" from "no difference."

**Implications:**
- Removing exclusivity did not change the formal architecture decision because the bottleneck is the `bd_c2` outlier problem, not the arrowhead arm's quality.
- The practical takeaway is that `ah_prag_no_excl` is strictly more reliable than `bd_c2` on M4 (no catastrophic failures, tight variance) and also has better CosSep and M5, but the pre-registered decision rule cannot express this.
- The decision rule needs to be updated to handle heavy-tailed M4 (e.g., median-based comparison, trimmed means, or a tail-risk penalty) before it can make a decisive call.

**6) Next steps**

1. Update the decision rule to use robust M4 summaries (median or trimmed mean) and re-apply to the same data.
2. Consider adding explicit spectral regularization to `ah_prag_no_excl` to recover the modest stability advantage without the basin-separation penalty.
3. Investigate why exclusivity raises inter-basin cosine — does it force the encoder to reuse similar activation patterns across basins?
4. Diagnose the `bd_c2` outlier regimes (ts=512, bp=13) to determine if they are fixable or should be excluded.

### -7) Diagonal Koopman Add-On for LQR Readiness (completed)
Timestamp: 2026-02-06

Scripts:
- Training sweep: `scripts/sweep_lqr_decision_stage2_diag_only.sh` (job `8622447`, array 0-71)
- LQR-only recovery sweep: `scripts/sweep_lqr_decision_stage2_diag_lqr_eval_only.sh` (job `8622706`, array 0-71)
- Aggregation: `tools/collect_lqr_decision_results.py`

Artifacts:
- `results/lqr_decision_stage2_diag_vs_bd/{summary_decision_table.csv,lqr_readiness_summary.json,final_decision.md}`
- `results/lqr_decision_stage2_diag_vs_ah/{summary_decision_table.csv,lqr_readiness_summary.json,final_decision.md}`

**1) Concrete results**

- Diagonal arm sweep (`diag_c1`) completed on the Stage-2 Lyapunov grid (72 runs: `target_size in {128,256,512}`, `B_proxy in {8,13,20}`, seeds `0..7`).
- Arm-level means (Stage-2 subset):
  - `diag_c1`: `M2=1.0`, `M3=1.0`, `M4=0.8618`
  - `ah_prag`: `M2=1.0`, `M3=1.0`, `M4=0.8285`
  - `bd_c2`: `M2=1.0`, `M3=1.0`, `M4=-10.7638` (heavy-tailed failures)
- Pairwise (`diag_c1-ah_prag`):
  - `M2` diff = `0.0000`, 95% CI `[0.0000, 0.0000]`
  - `M3` diff = `0.0000`, 95% CI `[0.0000, 0.0000]`
  - `M4` diff = `+0.0319`, 95% CI `[0.0077, 0.0542]` (favors `diag_c1`)
  - Decision output: **no clear winner**, fallback `diag_c1` (no practical-threshold pass).
- Pairwise (`diag_c1-bd_c2`):
  - `M2` diff = `0.0000`, 95% CI `[0.0000, 0.0000]`
  - `M3` diff = `0.0000`, 95% CI `[0.0000, 0.0000]`
  - `M4` diff = `+13.1299`, 95% CI `[0.0303, 39.2959]` (favors `diag_c1`)
  - Decision output: **no clear winner**, fallback `bd_c2` (current fallback rule prefers `bd*` arms when present).
- Robustness metric `M5`: `0.0` for all compared arms in both pairwise summaries.
- Robust `M4` snapshot from run-level aggregates:
  - `diag_c1`: finite `63/72`, median `0.8788`, min `0.5169`, max `0.9698`
  - `ah_prag`: finite `72/72`, median `0.8291`, min `0.7856`, max `0.8720`
  - `bd_c2`: finite `61/72`, median `0.8285`, min `-704.9678`, max `0.9573`

**2) Context in experiment design**

- This add-on reused the same Stage-2 grid and label-free evaluator as the prior LQR decision pipeline so diagonal can be compared directly to existing finalists.
- Training completed for all 72 runs under job `8622447`; post-training evaluation in that script failed due a runner shell parse issue, then was recovered by running LQR-readiness-only evaluation on existing checkpoints (job `8622706`).

**3) Interpretation**

- Diagonal is **not under-expressive in this evaluator regime**: it matched `M2/M3` saturation and improved `M4` relative to `ah_prag`.
- The large `diag_c1-bd_c2` `M4` gap is strongly influenced by catastrophic `bd_c2` outliers; median `M4` for `bd_c2` is close to `ah_prag`.
- The current pre-registered decision gate still cannot make a decisive call because `M2/M3` saturate and practical-threshold criteria are not met.

**4) Project implications**

- `diag_c1` is now a credible LQR-readiness baseline and should remain in architecture comparisons.
- Heavy-tail handling for `M4` is now more urgent than before, since mean-based differences can be dominated by outlier regimes.
- The existing fallback logic is biased toward `bd*` when present and is not suitable as a final architecture selector once diagonal is included.

**5) Next steps**

1. Add robust primary summaries for `M4` (median/trimmed mean/tail-risk) to the decision rule and re-run diagonal vs finalists under the same Stage-2 grid.
2. Update fallback semantics for multi-architecture comparisons (`diag`, `bd`, `ah`) so the fallback is not hard-coded toward `bd*`.
3. Carry the same diagonal comparison to Duffing transfer after robust-`M4` decision updates.

### -6) LQR-Readiness + Arrowhead vs Block-Diagonal Koopman Decision (completed)
Timestamp: 2026-02-06

Plan: `docs/planning/block_diagonal_vs_arrowhead_lqr_decision_plan.md`

Artifacts:
- Stage 2 (Lyapunov decision): `results/lqr_decision/final_decision.md`, `results/lqr_decision/summary_decision_table.csv`, `results/lqr_decision/lqr_readiness_summary.json`
- Stage 1 (Lyapunov structure isolation): `results/lqr_decision/stage1_lyapunov/final_decision.md`, `results/lqr_decision/stage1_lyapunov/summary_decision_table.csv`
- Stage 3 (Duffing transfer): `results/lqr_decision/stage3_duffing/final_decision.md`, `results/lqr_decision/stage3_duffing/summary_decision_table.csv`

**1) Concrete results**

- Stage 1 structure isolation (`AH-ISO` vs `BD-C1`/`BD-C2`, 36 runs):
  - All arms: `M2=1.0`, `M3=1.0`.
  - Arm-level `M4` means: `ah_iso=0.7279`, `bd_c1=0.5939`, `bd_c2=0.7351`.
  - Pairwise (`bd_c2-ah_iso`) `M4` diff = `+0.0407`, 95% CI `[-0.0888, 0.1650]` (CI includes 0).
- Stage 2 practical head-to-head (`BD*=bd_c2` vs `AH-PRAG`, Lyapunov, 144 runs):
  - Both arms: `M2=1.0`, `M3=1.0` across 72 runs/arm.
  - Pairwise (`bd_c2-ah_prag`) `M4` diff = `-11.5930`, 95% CI `[-34.7496, 0.0006]` (CI includes 0).
  - Arm-level summary: `ah_prag M4=0.8285`, `bd_c2 M4=-10.7638`.
  - Decision output: **no clear winner**, fallback simpler arm `bd_c2`.
- Stage 3 transfer (Duffing, 78 runs):
  - Both arms: `M2=1.0`, `M3=1.0`.
  - Pairwise (`bd_c2-ah_prag`) `M4` diff = `-0.0849`, 95% CI `[-0.1100, -0.0632]` (favors `ah_prag` on M4).
  - Pre-registered practical threshold (`+0.10` on M2/M3) not met; decision remains fallback `bd_c2`.
- Robustness metric `M5` (B_proxy sensitivity): `0.0` for both finalist arms in all decision summaries.

**2) Context in experiment design**

The staged design separated confounds:
- Stage 1 tested **Koopman structure effect** directly (`AH-ISO` removes exclusivity regularization confound).
- Stage 2 compared practical finalists under the main Lyapunov decision protocol.
- Stage 3 tested transfer on Duffing with misspecified `B_proxy`.

**3) Interpretation**

- In this evaluator regime, `M2` and `M3` are saturated and do not discriminate architectures.
- Arrowhead variants (`ah_iso`/`ah_prag`) tend to show stronger local-fit and/or `M4` behavior, but uncertainty and practical-threshold rules prevent a decisive win on primary criteria.
- Block-diagonal (`bd_c2`) remains competitive on control-readiness rates and is selected by the pre-registered fallback rule because there is no clear practical winner.

**4) Project implications**

- For current downstream LQR workflow, keep `bd_c2` as the default baseline.
- Treat `ah_prag` as the strongest alternative when optimizing cost reduction behavior.
- Primary bottleneck shifts from feasibility/stability to **metric discriminativeness and robustness** (especially heavy-tailed `M4` cases).

**5) Next steps**

1. Make `M2/M3` harder to saturate (stronger perturbations, harder regimes, stricter feasibility/stability checks).
2. Add robust `M4` summaries (median, trimmed mean, tail-risk metrics) and explicitly diagnose outlier settings (notably `bd_c2`, `ts=512`, `B_proxy=13` on Lyapunov Stage 2).
3. Re-run a reduced head-to-head after adding explicit spectral constraints, then re-apply the same pre-registered decision rule.

### -5) LQR Decision Pipeline Implementation (completed)
Timestamp: 2026-02-05

Plan: `docs/planning/block_diagonal_vs_arrowhead_lqr_decision_plan.md`

Implemented components:
- Label-free LQR-readiness evaluator: `tools/evaluate_lqr_readiness.py`
- Stage runner: `scripts/run_lqr_decision_trial.sh`
- Stage sweeps: `scripts/sweep_lqr_decision_stage0.sh`, `scripts/sweep_lqr_decision_stage1.sh`, `scripts/sweep_lqr_decision_stage2.sh`, `scripts/sweep_lqr_decision_stage3.sh`
- Aggregation and final decision tool: `tools/collect_lqr_decision_results.py`

`AH-PRAG` lock for reproducibility:
- `lambda_global = 1e-4`
- `lambda_local = 1e-3`

Suggested launch order:
1. `sbatch scripts/sweep_lqr_decision_stage0.sh`
2. `sbatch scripts/sweep_lqr_decision_stage1.sh`
3. Pick `BD*` and run `sbatch scripts/sweep_lqr_decision_stage2.sh`
4. Run `sbatch scripts/sweep_lqr_decision_stage3.sh`
5. Aggregate with `uv run python tools/collect_lqr_decision_results.py --base_dir /network/scratch/l/lia/skae/lqr_decision --output_dir results/lqr_decision --decision_stage 2 --decision_system lyapunov --arms <BD_STAR>,ah_prag`.

Submission status:
- Stage 1 submitted: job `8620569` (`scripts/sweep_lqr_decision_stage1.sh`)
- Stage 0 submitted: job `8620570` (`scripts/sweep_lqr_decision_stage0.sh`)

What to do next:
1. Wait for Stage 0/1 completion and review Stage 1 arm metrics to choose `BD*`.
2. Launch Stage 2 with selected baseline using `BD_STAR=<bd_c1_or_bd_c2>`.
3. Launch Stage 3 only after Stage 2.
4. Run `tools/collect_lqr_decision_results.py` to produce final decision artifacts.

Update (February 5, 2026, later):
- Stage 1 decision completed from observed metrics: `BD* = bd_c2` (decided by M4).
- Stage 2 submitted with `BD_STAR=bd_c2` as job `8621121`.
- Stage 3 submitted as dependent job `8621236` (`afterany:8621121`).
- Final collection submitted as dependent job `8621237` (`afterany:8621236`).

### -4) Block-Loss Balance Sweep (Phase 1)
Timestamp: 2026-02-05 (completed; job `8615817`)

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

**Status:** COMPLETED (72/72).

**Results (primary metric = cosine separation; threshold metrics secondary):**

- Best seed-averaged config: `kl_uniform`, `top1_margin=0.05`, `one_block_weight=0.3`, `balance_weight=1.0`
  - **CosSep = 0.8826** (`intra=0.9676`, `inter=0.0851`)
  - At `tau=1e-3`: `uniqueness=1.0`, `consistency=0.138`, `Jaccard=0.142`
- Relative to the ts=256 control from the ablation sweep (`CosSep=0.8256`), the best Phase 1 config improves by **+0.0570**.
- Robustness across the 36 seed-averaged configs:
  - **23/36** configs beat the ts=256 control separation.
  - **28/36** retain full uniqueness at `tau=1e-3`; **8/36** lose full uniqueness.
  - Worst config (`usage_entropy`, `top1_margin=0.1`, `one_block_weight=0.3`, `balance_weight=0.1`) collapses to **CosSep=0.2225**, `uniqueness=0.5`.
- Aggregate trends:
  - `kl_uniform` is stronger on average than `usage_entropy` (mean CosSep **0.776** vs **0.700**).
  - Smaller margin is safer (`top1_margin=0.05`: mean CosSep **0.776** vs **0.701** at `0.1`).

**Context:** This sweep directly tests whether combining per-sample one-block pressure (`top1_margin`) with across-batch usage balancing can improve basin separability without the collapse seen in strict one-block-only losses.

**Interpretation:** The combination can improve basin separability, but only in part of the weight space. A moderate one-block pressure plus balancing gives stronger intra-vs-inter basin separation, while aggressive settings still produce degenerate supports and lost uniqueness.

**Implications for basin separability goal:** This is progress toward label-free basin discrimination: combined losses can push separation above control at ts=256. But separability is not yet robust enough across hyperparameters to treat basin assignment as reliable by default.

**Next steps:**
1. Center Phase 2 on the robust region (`top1_margin=0.05`, moderate `one_block_weight`, `kl_uniform`-heavy balancing) with more seeds.
2. Add early-stop guards using cosine separation + uniqueness to terminate collapsing runs.
3. Validate whether the robust region transfers to larger `target_size` and to sequence-training settings.

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
Timestamp: 2026-02-04 (completed readout on 2026-02-05; job `8613853`)

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

**Status:** COMPLETED (36/36).

**Results (stability metric = max spectral radius):**

- **0/36** configs reached SR < 1.0 (strict stability).
- SR range across sweep: **1.0014 to 1.2540**.
- Near-stable region exists but does not cross stability boundary:
  - Best SR: `res=1.0, reconst=0.3, pred=1.0, sparsity=0.1` with **SR=1.0014** and `H1000 no-reencode=3.77`.
  - Best H1000 among finite no-reencode runs: `res=3.0, reconst=0.3, pred=1.0, sparsity=0.3` with **H1000=3.05**, **SR=1.0017**.
- Long-horizon no-reencode outcomes remain fragile:
  - **22/36** runs produce non-finite (`nan/inf`) H1000 no-reencode.
  - Among finite runs (14), median H1000 no-reencode is still very large (`~7.6e3`).
- Coefficient pattern:
  - `pred_coeff=1.0` substantially helps (mean SR **1.064** vs **1.176** when `pred_coeff=0.0`), but still not enough to get SR < 1.

**Context:** This experiment isolates whether reweighting residual/reconstruction/prediction/sparsity terms can stabilize non-diagonal Koopman dynamics under sequence training, without adding explicit spectral constraints.

**Interpretation:** Weight tuning can move models closer to stability (near-SR=1 region) but does not solve the core instability. The stabilizing signal from prediction loss is useful yet insufficient when unconstrained K dynamics remain slightly expansive.

**Implications for basin separability goal:** Basin separability in latent space is not enough for usable basin-wise dynamics. If SR remains above 1, long-horizon rollouts are unreliable, so separable basins cannot be converted into dependable local linear models for downstream control.

**Next steps:**
1. Add explicit spectral control (e.g., SR penalty, projection, or constrained K parameterization) and rerun a reduced grid around the near-stable settings.
2. Keep `pred_coeff=1.0` in the follow-up baseline since it consistently reduces SR.
3. Re-check separability metrics (cosine separation + uniqueness) after adding spectral constraints to ensure stability gains do not erase basin discrimination.

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

#### Results: Secondary Diagnostic - Basin-to-Block Activation Correlation

For each model, we encode 100 basin-labeled trajectories and compute the mean activation magnitude per latent dimension grouped by K block. The "basin-block concentration" metric measures how peaked each basin's activation is toward a single block (1.0 = perfect one-basin-one-block alignment, 0.0 = uniform spread). This is a benchmark-only diagnostic and not the primary objective.

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

**At ts=64, diagonal and block-diagonal show strong basin-block alignment** (concentration 0.87–0.89), but this alignment **fades with increasing capacity**: at ts=256, all structures show concentration ~0.06–0.13, meaning activations are distributed across many blocks.

**Counterintuitively, the arrowhead with exclusivity has the *lowest* basin-block concentration** at every dimension. The exclusivity loss encourages one-basin-at-a-time activation in latent space, but this does not enforce alignment between specific basin blocks and specific ground-truth basins.

#### Interpretation

1. **Spectral stability is the dominant factor for long-horizon prediction quality.** The binary stable/unstable classification predicted by the spectral radius perfectly explains the 15+ orders-of-magnitude spread in H1000 MSE across configurations. All stable models converge to a narrow MSE band (3.49–3.82); all unstable models diverge. This means Koopman structure choice is primarily a question of *which structures reliably produce SR < 1*, not which produce the lowest MSE.

2. **Arrowhead with exclusivity is robust up to ts=256 but not at higher capacity.** It achieves the best no-reencode H1000 MSE (3.49 at ts=128) and stays stable for ts <= 256, but SR exceeds 1 at ts=512 and ts=1024. The exclusivity loss provides an implicit spectral constraint, but it is insufficient alone at high latent sizes.

3. **Dense K is spectrally unstable above ts=64.** The dense Koopman matrix's d² free parameters allow eigenvalues to drift outside the unit circle during training. At ts=512, the spectral radius reaches 1.03, causing H1000 divergence to 10^19. Dense K should not be used for long-horizon prediction without explicit spectral regularisation (e.g., eigenvalue penalty or spectral normalisation of K).

4. **Block-diagonal and diagonal stability erodes at high capacity.** Both are unstable at ts=64, both are stable at ts=256, diagonal remains stable at ts=512, but both are unstable at ts=1024. This makes them competitive at moderate sizes but unreliable at high capacity without explicit spectral constraints.

5. **Basin-block alignment does not emerge from K structure alone.** Despite using block sizes matching the number of ground-truth basins (d/13), the encoder does not learn a consistent one-basin-one-block mapping at moderate-to-large latent dimensions.

6. **For LQR control, the primary requirement is reliable support-conditioned regime models, not one-basin-one-block mapping.** Basin-support uniqueness can be strong even when basin-block concentration is low, so we should prioritize label-free support/regime assignment and local predictive/stability checks.

#### Implications

- Without explicit spectral control, high-capacity latents (ts >= 512) are unstable across all K structures.
- Arrowhead structure alone is not enough to guarantee stability at high dimensions; stability constraints must be part of training or parameterization.
- Long-horizon reliability should be treated as a first-class objective, not an emergent property of structure choice.

#### Next Steps

1. **Add spectral regularisation to K (dense + structured).** Penalise eigenvalues outside the unit circle during training (e.g., `lambda_spectral * max(0, SR - 1)^2`) or use spectral normalization. Re-run a small grid at ts={256,512} to check if stability persists at higher capacity.

2. **Investigate why arrowhead loses stability at high ts.** Sweep exclusivity/sparsity weights and global/basin split to see if SR can be kept < 1 at ts >= 512.

3. **Develop label-free support-to-regime assignment diagnostics.** Evaluate clustering/merging of support signatures and latent trajectories under unknown basin count, and measure assignment robustness across seeds.

4. **Test LQR on stable, high-separation checkpoints first.** Prioritize checkpoints with strong support separation and SR < 1, then evaluate local controller feasibility and closed-loop gains under the label-free regime assignments.

5. **Validate on Duffing.** Keep known-basin evaluation for benchmarking, but report results in terms of basin-support uniqueness and local predictive/stability metrics.

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

2. ~~**Extract per-block dynamics from block_diagonal K.**~~ **DONE** (see Experiment 0 above). Per-block eigenvalues extracted. Basin-block concentration is strong at ts=64 (0.87) but fades at ts=256 (0.10); this is a secondary diagnostic.

3. **Test LQR on extracted local dynamics.** Still pending. Prioritize stable, high-separation checkpoints and evaluate controllers with label-free regime assignment rather than requiring one-basin-one-block alignment.

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
