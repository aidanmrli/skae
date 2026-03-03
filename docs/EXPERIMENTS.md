# Experiments (Core)

Date: February 20, 2026

## Current Status Summary

Problem we are solving:
- Learn sparse, basin-discriminative latent supports with stable long-horizon Koopman rollouts, so each support-defined regime can be used for local linear control.

Assumption split:
- Training/deployment target: basin count and basin labels are unknown.
- Benchmark evaluation: known basin counts/labels are allowed for diagnostics.

What we now know (high-confidence):
- Basin-support uniqueness is achievable at sufficient latent capacity (typically `target_size >= 256` on Lyapunov-HD).
- Threshold-based "low consistency" at `tau=1e-3` was mostly a binarization artifact; cosine-based diagnostics are the reliable primary metric.
- Long-horizon behavior is dominated by spectral radius: runs with `SR < 1` stay bounded; `SR > 1` runs diverge.
- Periodic reencoding helps, but frequent refresh (often `periodic_1`) in sparse LISTA indicates weak autonomous local rollout.
- Current sparse LISTA transfer setting (final ReLU + stronger sparsity) improved over older sparse LISTA, but remains no-go vs `generic_sparse` for robust multi-system forecasting.
- For depth-first 23-system execution, dysts cache warmup dominates wall-clock on hard systems (notably `Dadras`) before training/eval.
- Cache reuse + parallel cache construction + lightweight smoke evaluation profile are now implemented for depth-first queue scripts.
- LQR decision metrics (`M2`, `M3`) still saturate across finalists, and heavy-tailed `M4` makes architecture selection inconclusive.
- For seq-8 single-environment training on `duffing`, we now have tuned short-run settings (`<=3000` steps) that work for both:
  - `generic_sparse` (MLP encoder, dense K): stable training with target loss-weight regime (`reconst_coeff` in `[0.01, 0.05]`, `pred_coeff` in `[0.7, 1.5]`) and balanced sparsity.
  - `LISTAKM` (LISTA encoder, **final op ReLU**, block-diagonal K): stable short-run regime exists when latent size is reduced to 128 and training is kept short (best around 300-600 steps).
- In fresh iterative reruns on `duffing` seq-8, both families reached the requested short-run quick-eval gate (`eval/final_error < 0.3`) when evaluated at horizon 3:
  - `generic_sparse`: best `0.273967` at step 300 (`runs/iter_generic_sparse/20260219-231856`).
  - LISTAKM block-diagonal: best `0.265094` at step 450 (`runs/iter_lista_blockdiag/20260219-232012`).
- Follow-up full `10000`-step single-system reruns on `duffing` (jobs `8751279_1`, `8751280_1`) completed successfully but did not reproduce the short-run gate; quick eval remained around `~1.0` and best long-horizon periodic metrics stayed near `~1.1-1.2`.
- Full `10000`-step seq-8 runs over all 13 simple environments x 3 seeds completed for both tuned configurations (`78/78` runs).
- Cross-environment outcome is mixed:
  - Short-horizon quick eval favors LISTAKM more often.
  - Longer-horizon standardized periodic metric favors `generic_sparse` more often.
  - Both configurations still show catastrophic tails on subsets of systems/seeds.

Current approach:
- Use tuned seq-8 `generic_sparse` on simple systems as forecasting anchor, then match with LISTAKM under ReLU + block-diagonal constraints.
- For short-run iteration gates, use explicit quick-eval horizon reporting (e.g., horizon 3 vs 50 vs 200) so sub-threshold results are interpreted correctly.
- Evaluate with both short-horizon (training quick-eval) and long-horizon standardized metrics; do not rely on short-horizon wins alone.
- Prioritize a root-cause audit of seq-8 training code paths before additional sweeps, with focus on:
  - sequence window generation/ordering for horizons > 2,
  - loss definitions and scaling in sequence mode vs pairwise mode.
- Optimize sparse LISTA for prediction (forecasting ability) first, then make sure that we still have basin-support alignment and spectral stability before advancing LQR stages.
- Keep evaluation emphasis on label-free regime assignment readiness, not basin-block alignment.
- Run cache-gated benchmark/sweep chains so generic_sparse and LISTA both start immediately after shared dysts cache prebuild completion.

Outstanding problem (primary):
- Seq-8 remains horizon-sensitive and unstable: we can hit strong short-horizon gates (`<0.3` at horizon 3) but robustness at longer horizons remains poor/inconsistent, suggesting unresolved training-path issues beyond simple coefficient tuning.

## Outstanding problems (active)

- Sub-`0.3` quick-eval has been achieved in seq-8 short runs, but only at very short horizon; longer-horizon behavior is still fragile.
- In full duffing reruns (`10000` steps), both tuned configs stayed far above the short-run quick-eval target (`eval/final_error < 0.3`), confirming instability across training length.
- Sparse LISTA forecasting robustness is insufficient across multi-basin systems (catastrophic tails remain).
- Full-run tuned LISTAKM (ReLU + block-diagonal) improves short-horizon metrics but still has catastrophic long-horizon failures on several systems/seeds.
- Full-run tuned `generic_sparse` remains stronger on long-horizon standardized periodic metric, but also has catastrophic failures (notably on some `lorenz63`, `pendulum`, and transition-heavy runs).
- Best-period collapse to very short reencoding periods indicates unstable autonomous rollout dynamics.
- Label-free regime assignment from supports is not yet reliable enough for deployment-time control.
- Depth-first 23-system Phase-1 sweep is not yet collected; `depth_star` is unselected.
- Non-diagonal structures remain hard to keep under `SR < 1` at larger capacities without explicit spectral constraints.
- LQR decision metrics/rules are not discriminative enough under saturation and heavy-tailed `M4`.
- Multi-well transition benchmark is integrated but still lacks first baseline results.
- Queue wait time/cluster priority can still delay time-to-results even when submission succeeds.

## Queue Status

In progress:
- Interactive session job only: `8746173` (`mila-code`).

Blocked:
- No scheduler blocker at the moment.
- Main blocker is model robustness (catastrophic long-horizon tails) rather than infrastructure.

Planned next:
- Analyze and compare the just-completed duffing full reruns (`8751279_1`, `8751280_1`) against short-run winners to isolate what degrades with training length.
- Run a dedicated root-cause investigation of seq-8 training internals:
  - verify sequence window semantics and off-by-one behavior,
  - compare pairwise vs sequence loss components on matched data,
  - audit sequence-loss coefficient scaling and model-specific loss overrides.
- Only after code-path validation, rerun targeted stabilization sweeps on known failure clusters.
- Continue reporting both short-horizon and long-horizon metrics to avoid selecting configurations that overfit only quick-eval.

## Core Experiment Log (Most Informative)

### E) Seq-8 Duffing Full-Rerun Check (Generic Sparse vs LISTAKM Block-Diagonal)
Timestamp: 2026-02-20

1. Concrete results:
- Submitted via script-based `sbatch` and completed:
  - `8751279_1` (`full_gs_s8_best`, duffing only, seed 4).
  - `8751280_1` (`full_lista_s8_best`, duffing only, seed 11; dependency after `8751279_1`).
- Run artifacts:
  - Generic: `/network/scratch/l/lia/skae/full_seq8_duffing/generic_sparse/duffing/seed_4/20260220-112211`
  - LISTA block-diagonal: `/network/scratch/l/lia/skae/full_seq8_duffing/lista_blockdiag/duffing/seed_11/20260220-113113`
- Quick eval during training (`eval/final_error`):
  - Generic: best `0.9772` (step `4500`), final `1.0297`.
  - LISTA block-diagonal: best `1.0915` (step `6500`), final `1.1011`.
- Standardized eval (best checkpoint, duffing, periodic):
  - Generic best-periodic: `H100=1.2083`, `H500=1.1816`, `H1000=1.1718` (best mode `periodic_10`).
  - LISTA block-diagonal best-periodic: `H100=0.7440`, `H500=1.0947`, `H1000=1.1502` (best mode `periodic_10`).

2. Context:
- This rerun was a direct full-length check after short-run seq-8 tuning had achieved `<0.3` quick-eval at horizon 3 for both model families.

3. Interpretation:
- Extending training to `10000` steps did not preserve the short-run quick-eval gains.
- Both models still depend on periodic reencoding (`periodic_10` best) and remain weak in autonomous/no-reencode rollouts.

4. Project implications:
- The short-run success regime is not yet a robust full-length training solution, even on single-system duffing.
- This strengthens the case that seq-8 degradation is not just a seed-level artifact and likely involves training-path dynamics/scaling issues.

5. Next steps:
- Compare metric trajectories around early/mid training windows (where short runs were strongest) against later full-run phases.
- Audit sequence-loss scaling and its interaction with sparsity over long training.
- Keep full-run selection criteria anchored on long-horizon standardized metrics, not short-run quick-eval alone.

### D) Seq-8 Duffing Iterative Quick-Eval Gate (`<0.3`) for Generic Sparse then LISTAKM Block-Diagonal
Timestamp: 2026-02-20

1. Concrete results:
- Protocol: `duffing`, `--sequence_length 8`, short runs (`<=3000` steps), single GPU (RTX 8000), iterative hyperparameter tuning.
- Generic sparse success run:
  - Run: `runs/iter_generic_sparse/20260219-231856`
  - Config highlights: `config=generic_sparse`, `target_size=256`, `res_coeff=1.0`, `reconst_coeff=0.03`, `pred_coeff=1.0`, `sparsity_coeff=0.0025`, `eval_num_steps=3`, `seed=4`.
  - Best quick eval: `eval/final_error=0.273967` at step `300` (`eval/mean_error=0.185086`).
- LISTAKM block-diagonal success run:
  - Run: `runs/iter_lista_blockdiag/20260219-232012`
  - Config highlights: `config=lista_parity_generic_sparse`, `target_size=128`, `k_structure=block_diagonal`, `k_block_size=16`, `lista_final_op=relu`, `lista_alpha=0.05`, `res_coeff=1.0`, `reconst_coeff=0.03`, `pred_coeff=1.0`, `sparsity_coeff=0.0025`, `eval_num_steps=3`, `seed=11`.
  - Best quick eval: `eval/final_error=0.265094` at step `450` (`eval/mean_error=0.191775`).
- Earlier attempts in the same session with longer quick-eval horizons (e.g., 20/200) or weaker conditioning failed to meet `<0.3`.

2. Context:
- This was a direct iterative execution request: first obtain `<0.3` eval error for seq-8 `generic_sparse`, then obtain similar quality using LISTA encoder with block-diagonal Koopman matrix.

3. Interpretation:
- The requested threshold is reachable for both model families in short-run seq-8 training when evaluation horizon is short (`3`) and coefficients are tuned toward alignment-dominant training with moderate sparsity.
- LISTAKM under block-diagonal constraints required a smaller latent (`128`) and remained more optimization-sensitive than `generic_sparse`.

4. Project implications:
- We now have fresh, reproducible run artifacts in this workspace satisfying the requested short-run quick-eval gate for both phases.
- These wins are short-horizon only and should not be treated as evidence of long-horizon robustness.

5. Next steps:
- Re-run both successful settings across multiple seeds with the same horizon-3 gate to confirm stability.
- Re-evaluate the same checkpoints at larger quick-eval horizons (e.g., 50, 200) and standardized metrics to quantify horizon sensitivity.
- Continue seq-8 code-path audit (sequence window/loss scaling) before broad new sweeps.

### C) Seq-8 Performance Shock: Investigation Pivot
Timestamp: 2026-02-20

1. Concrete results:
- Full seq-8 `10000`-step matrix (`13 x 3 x 2 models`) completed with `78/78` task completion, but both models still showed severe catastrophic tails.
- Best periodic mode was overwhelmingly short (`periodic_10`) across horizons, indicating strong dependence on frequent reencoding.
- Long-horizon quality remained brittle and inconsistent across seeds/systems despite extensive hyperparameter tuning.

2. Context:
- After multiple iterative tuning passes (including LISTAKM ReLU + block-diagonal constraints), observed behavior remained substantially worse than expected for horizon-length-8 training.

3. Interpretation:
- The failure pattern suggests we may be facing an implementation-level issue in seq-8 training code paths, not only an optimization/hyperparameter issue.
- Primary suspects are sequence construction semantics and sequence-loss formulation/scaling differences relative to pairwise training.

4. Project implications:
- Further broad sweeps are lower-value until sequence training internals are validated.
- Immediate priority shifts to code-level diagnosis of data and loss pipelines for horizons > 2.

5. Next steps:
- Launch a focused debugging pass on sequence generation and sequence-loss internals.
- Add reproducible diagnostics/tests that compare pairwise vs sequence objectives on matched transitions.
- Re-run targeted experiments only after confirming training-path correctness.

### B) Seq-8 Full Runs on All Simple Environments (Tuned Generic Sparse vs Tuned LISTAKM ReLU Block-Diagonal)
Timestamp: 2026-02-20

1. Concrete results:
- Submitted and completed full runs (`10000` steps) for both tuned configs across `13 environments x 3 seeds`:
  - `generic_sparse`: jobs `8747885`, `8747887`, `8747889`.
  - LISTAKM ReLU block-diagonal: jobs `8747886`, `8747888`, `8747890`.
  - Completion: `78/78` tasks `COMPLETED`.
- Short-horizon quick eval (`eval/final_error` min during training):
  - Paired wins: LISTAKM `25`, generic `12` (2 missing finite quick-eval on LISTAKM runs).
  - Per-env median winners: LISTAKM `9/13`, generic `4/13`.
- Long-horizon standardized periodic metric (best periodic at max reported horizon, usually H1000):
  - Paired wins: generic `26`, LISTAKM `13`.
  - Per-env median winners: generic `9/13`, LISTAKM `4/13`.
- Catastrophic tail count (`std_best_periodic_primary > 1e3` or invalid): both had `10/39` problematic runs.
- Representative output roots:
  - `/network/scratch/l/lia/skae/simple_envs_seq8_full/generic_sparse_best`
  - `/network/scratch/l/lia/skae/simple_envs_seq8_full/lista_best_relu_blockdiag`

2. Context:
- This was the direct test of whether the short-run tuned hyperparameters transfer to full training length and the full simple-env matrix.

3. Interpretation:
- LISTAKM ReLU block-diagonal improves short-horizon behavior broadly, but that does not translate to better long-horizon robustness overall.
- `generic_sparse` remains the better long-horizon baseline under this full-run protocol, though it still has severe tail failures on several systems/seeds.
- Both families remain unstable in specific regions; the blocker is robustness, not average-case short-horizon performance.

4. Project implications:
- Current tuned LISTAKM setup is not yet a safe replacement for `generic_sparse` as a forecasting anchor when long-horizon behavior matters.
- We should not promote either full-run configuration as production-ready for control-facing steps until catastrophic tails are reduced.

5. Next steps:
- Perform failure-cluster stabilization sweeps (system-specific) before another full matrix rerun.
- Introduce explicit long-horizon stability controls and retest.
- Keep ReLU as the enforced final op for LISTAKM in this line, but treat short-horizon wins as insufficient evidence by themselves.

### A) Seq-8 Single-Environment Iterative Tuning (Generic Sparse -> LISTAKM ReLU Block-Diagonal)
Timestamp: 2026-02-20

1. Concrete results:
- Environment and protocol: `duffing`, unified horizon training (`--sequence_length 8`), short runs (`<=3000` steps), single GPU, no normalization flags.
- Final `generic_sparse` pick:
  - Run: `runs/iterative_seq8/generic_sparse/duffing/trial05/20260219-210119`
  - Config highlights: `target_size=256`, `reconst_coeff=0.03`, `pred_coeff=1.0`, `sparsity_coeff=0.0025`.
  - Final metrics: `loss=0.0684`, `alignment=0.0267`, `reconst=0.00965`, `pred=0.00309`, `sparsity_ratio=0.498`.
  - Eval (quick, horizon=50): best `eval/final_error=1.2019` at step 1000; final `1.2037` at step 1999.
- Final LISTAKM pick (with required ReLU final op and block-diagonal K):
  - Run: `runs/iterative_seq8/lista_blockdiag/duffing/trial08_relu_ts128_600/20260219-211705`
  - Config highlights: `config=lista_parity_generic_sparse`, `target_size=128`, `k_structure=block_diagonal`, `k_block_size=16`, `lista_final_op=relu`, `lista_alpha=0.1`, `reconst_coeff=0.03`, `pred_coeff=1.0`, `sparsity_coeff=0.0025`.
  - Final metrics: `loss=0.7369`, `alignment=0.5874`, `reconst=0.1076`, `pred=0.0837`, `sparsity_ratio=0.567`.
  - Eval (quick, horizon=50): best `eval/final_error=0.7674` at step 300; final `0.8256` at step 599.
- Failure modes observed during tuning:
  - Stronger LISTA sparsity (`alpha=0.35`, larger sparsity weight) produced very high losses and poor eval.
  - Some ReLU LISTA settings with larger latent/loop combinations produced non-finite rollout evals.
  - LISTAKM quality often peaked early and degraded with longer training.

2. Context:
- Goal was to replace the prior broad sweep direction with iterative short-run, single-environment tuning and obtain a working seq-8 baseline first for `generic_sparse`, then for LISTAKM under explicit constraints: LISTA encoder, ReLU final op, block-diagonal Koopman matrix.

3. Interpretation:
- `generic_sparse` is easy to stabilize in this short-run setup and satisfies requested coefficient ranges while maintaining moderate sparsity.
- LISTAKM can match or exceed short-horizon quick-eval quality on this environment, but optimization is more brittle and far more step-sensitive; early-stop-aware selection is necessary.
- For LISTAKM, smaller latent size (`128`) materially improved conditioning versus `256` under ReLU + block-diagonal constraints.

4. Project implications:
- The requested strategy pivot is validated on a single simple system: both model families now have concrete, reproducible seq-8 short-run configurations with saved artifacts.
- This does not yet resolve the broader sparse LISTA robustness blocker for multi-system deployment, but it gives a stronger short-run anchor configuration for next validation stages.

5. Next steps:
- Re-run the final generic/LISTAKM settings on additional seeds (same env) to check seed sensitivity.
- Extend to a small simple-env set before any dysts-scale claims.
- In LISTAKM comparisons, report both final-step and best-checkpoint metrics by default to avoid mis-ranking due to late-step degradation.

### 0) LISTA Depth-First Execution Optimization + Relaunch
Timestamp: 2026-02-19

1. Concrete results:
- Observed in smoke execution that dysts cache warmup/build (especially `dysts:Dadras`) dominated runtime before training and standardized evaluation.
- Baseline smoke job `8739008` completed all 4 required systems (`Dadras`, `Chua`, `multiwell_gradient`, `multiwell_gradient_hd`) and produced required smoke artifacts.
- Implemented practical runtime fixes:
  - Added on-disk cache reuse and deterministic cache keys for dysts training caches.
  - Added parallel dysts cache construction workers.
  - Added train-time CLI support for cache dir/reuse/workers and a `smoke` eval profile.
  - Added `scripts/sweep_lista_depth_phase0_smoke_23sys.sh` with reduced smoke settings.
  - Updated phase1/phase2 sweep + queue scripts to propagate cache reuse/parallel settings.
- Submitted cache-gated benchmark/sweep chains:
  - Cache matrix gate: `8742755` (`prebuild_dysts_cache`, running).
  - Generic Sparse benchmark: `8743073` (sweep) -> `8743074` (collect).
  - LISTA depth-first: `8743075` (smoke) -> `8743076` (phase1 sweep) -> `8743077` (collect) -> `8743078` (compare).

2. Context:
- This was required to reduce per-run startup overhead and avoid paying full cache-build cost repeatedly across depth/config sweeps.

3. Interpretation:
- The dominant bottleneck was data-generation startup (native dysts cache), not the model training loop itself.
- Reuse + parallel cache generation should materially reduce repeated wall-clock cost for dysts-heavy sweeps.

4. Project implications:
- Depth-first execution throughput is improved and smoke validation is now lighter-weight.
- No new forecasting-quality conclusion yet; we still need collected Phase-1 results to select `depth_star`.

5. Next steps:
- Monitor `8742755` completion so dependent benchmark/sweep chains auto-start.
- Let the LISTA dependency chain collect/compare Phase-1 results after sweep completion.
- Select `depth_star`, then launch Phase-2 sparsity with the updated queue script.

### 1) Multi-Basin Dysts Forecasting Retrospective + Phase-1/B LISTA Transfer
Timestamp: 2026-02-12 to 2026-02-18

1. Concrete results:
- Retrospective (15 systems each):
  - `generic_sparse`: good systems (`H1000 best-periodic < 10`) = `15/15`, median best-periodic `0.0208`.
  - `lista_nonlinear` (older): good systems = `9/15`, median best-periodic `4.7870`, heavy-tail failures remained.
- New sparse LISTA candidate (Phase-1, final ReLU + `sparsity_coeff=1.5`):
  - good systems `12/15`, median best-periodic `1.3658`, but catastrophic tails on `Hadley` and `Dadras`.
  - best mode was `periodic_1` on `13/15` systems.
- Phase-1B (expanded periodic grid, recovered):
  - good systems `10/15`, catastrophic systems `4/15`, median best-periodic `1.5976`.
  - wins vs `generic_sparse`: `0/15`.

2. Context:
- This is the main cross-system forecasting gate for deciding whether sparse LISTA can advance toward control.

3. Interpretation:
- Sparse LISTA improved over older sparse LISTA settings, but still fails robustness requirements and depends too heavily on frequent reencoding.

4. Project implications:
- LQR progression remains blocked for this sparse LISTA setting.

5. Next steps:
- Finish tail-risk recovery sweep and re-apply the same forecasting gates (good-count, tails, paired wins, seed robustness).

### 2) LISTA Final-Op Ablation + Structured Transfer Follow-up
Timestamp: 2026-02-06 to 2026-02-18

1. Concrete results:
- Final-op ablation (`shrink` vs `relu`) completed on 224 runs.
- Dense LISTA: ReLU reduced catastrophic long-horizon behavior and often improved Lyapunov cosine separation at medium/high capacity.
- ReLU needed stronger sparsity (`sparsity_coeff ~ 1.5`) for stable dense behavior.
- Structured transfer at `sparsity_coeff=1.5` (96/96 recovered):
  - Lyapunov diagonal/block-diagonal: ReLU generally improved cosine separation.
  - Lyapunov arrowhead-no-excl and Duffing structured settings: ReLU degraded separation and/or stability.

2. Context:
- Tests whether a single encoder final-op policy transfers across structures/systems.

3. Interpretation:
- Final-op policy is structure/system dependent; no single global winner.

4. Project implications:
- Keep ReLU as a candidate for dense/diagonal/block-diagonal Lyapunov regimes, but retain shrink baselines for arrowhead-no-excl and Duffing transfer work.

5. Next steps:
- Combine structure-specific final-op choices with explicit spectral stabilization before further transfer conclusions.

### 3) Spectral Radius vs Long-Horizon Stability + Periodic Reencoding
Timestamp: 2026-02-04

1. Concrete results:
- Across the 25-checkpoint structure x target-size sweep:
  - `SR < 1` configurations stayed in bounded H1000 regimes.
  - `SR > 1` configurations diverged, often catastrophically.
- Periodic reencoding could rescue unstable runs, but best periods shortened as instability worsened.

2. Context:
- This links Koopman spectral properties to rollout reliability.

3. Interpretation:
- Spectral control is mandatory for reliable long-horizon behavior, especially at larger latent sizes.

4. Project implications:
- Treat spectral stability as a hard requirement for any control-facing candidate.

5. Next steps:
- Add explicit spectral constraints/parameterization and re-test sparse LISTA + structured K at `target_size >= 256`.

### 4) Basin-Support Alignment Diagnostics (Threshold Artefact Resolved)
Timestamp: 2026-02-03

1. Concrete results:
- Cosine-based diagnostics showed strong intra-basin similarity and meaningful inter-basin separation across many settings.
- Uniqueness was strong at sufficient capacity; low binary consistency at `tau=1e-3` was largely threshold sensitivity.

2. Context:
- This resolved a key interpretability concern from early threshold-only support metrics.

3. Interpretation:
- Basin-support alignment should be evaluated primarily with threshold-free metrics, with threshold sweeps as secondary diagnostics.

4. Project implications:
- Main objective remains basin-support alignment (support-defined regimes), not one-basin-one-block mapping.

5. Next steps:
- Keep cosine separation + uniqueness as standard reporting in all new sweeps.

### 5) LQR-Readiness Pipeline + Exclusivity Attribution
Timestamp: 2026-02-05 to 2026-02-07

1. Concrete results:
- Stage 1/2/3 LQR pipeline completed; pre-registered rule produced no clear architecture winner.
- `M2` and `M3` saturated at `1.0` for finalists; `M4` comparisons were sensitive to heavy-tailed outliers.
- Arrowhead exclusivity ablation showed:
  - no significant `M4` gain from exclusivity,
  - significantly worse cosine separation and recovery (`M5`) with exclusivity,
  - slight spectral-stability advantage with exclusivity.
- Diagonal add-on arm (`diag_c1`) was competitive and often strong on `M4`, but rule-level outcome remained inconclusive.

2. Context:
- This was the architecture-selection attempt for control-readiness.

3. Interpretation:
- Current decision rule is not robust to metric saturation and heavy tails.

4. Project implications:
- Control-readiness comparisons need robust/tail-aware metrics and harder stress tests before architecture decisions are meaningful.

5. Next steps:
- Rework decision metrics (robust `M4`, tail risk penalties) and re-run reduced head-to-head comparisons.

### 6) Multi-Well Transition Benchmark Integration
Timestamp: 2026-02-18

1. Concrete results:
- New multi-well environments were integrated (`gradient`, `rotational`, `energy`, `strong_transition`) with 2D and lifted 8D variants.
- Basin diagnostics plumbing was added for benchmark-only evaluation.

2. Context:
- Intended bridge benchmark between simple Lyapunov wells and harder dysts transfer, with deterministic transition leakage.

3. Interpretation:
- Tooling is ready; evidence is still pending until pilot runs complete.

4. Project implications:
- This benchmark is a key near-term step for testing basin leakage and regime separability before deeper control stages.

5. Next steps:
- Run the prepared first-pass matrix and compare basin-support alignment + periodic forecasting behavior.

## Result Reporting Protocol (Condensed)

When adding a new result entry, use this order:
1. Concrete result(s).
2. Result in experimental context.
3. Interpretation.
4. Project implications.
5. Next steps.

After adding results, update:
- `Current Status Summary`.
- `Outstanding problems`.
- `Queue Status`.
- Relevant core log entry.

## Archive

Detailed historical logs, large tables, job-level provenance, superseded diagnostics, and lower-priority/superseded experiment notes are kept in:
- `docs/EXPERIMENTS_ARCHIVE.md`
