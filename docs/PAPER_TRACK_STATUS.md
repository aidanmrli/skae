# Paper Track Status

Date: March 11, 2026

## Goal

The paper target is now explicit:

- maximize **periodic-reencoding forecasting** quality and basin-discriminative latent structure on nonlinear dynamical systems with multiple basins of attraction
- treat **step size** and **dimension** as the primary current bottlenecks
- when default-`dt` performance is clearly bad, prefer **reducing `dt`** before spending more budget on architecture churn
- optimize for forecasting-first evidence; sparsity and support structure matter, but only after the forecasting stack is competitive

Active execution note:
- The immediate paper-strengthening experiment program is coordinated in [docs/planning/paper_parallel_workstreams_20260309.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309.md). Use that file for current workstream ownership, pre-queue QA requirements, and per-agent run logs.

Immediate fairness blocker (March 11):
- We **must** add a matched `generic_sparse + block_diagonal K` control everywhere a paper-facing experiment currently uses `lista_blockdiag`.
- Until that control exists, current `lista_blockdiag` results are valid as end-to-end model comparisons but **not** as clean evidence that block-diagonal Koopman structure itself is the cause of the gain, because `GenericKM` still uses a dense `K`.
- Execution plan: [docs/planning/generic_sparse_blockdiag_fairness_plan_20260311.md](/home/mila/l/lia/skae/docs/planning/generic_sparse_blockdiag_fairness_plan_20260311.md)
- Diagonal policy (March 11): retire `lista_diagonal` from active paper scope. Keep the existing diagonal-K results only as historical context; do not spend more queue budget on diagonal reruns or include diagonal in future paper-facing rebuilds.

## Current Best Evidence

### 1. Canonical `v4` paper benchmark is complete: `generic_sparse` remains best overall, dense LISTA is the strongest LISTA-family competitor

- Completed `v4` full-matrix result (`29` systems, `4` baselines, `3` seeds) under the repaired `dt`-rescue chain:
  - `generic_sparse` is best by cross-system median `H1000` best-periodic (`0.0328`)
  - dense LISTA is second (`0.0388`)
  - block-diagonal LISTA is third (`0.1508`)
  - diagonal-K LISTA is worst (`1.2110`)
- `H1000` good-system counts (`best-periodic < 10`):
  - `generic_sparse`: `25/29`
  - dense LISTA: `24/29`
  - block-diagonal LISTA: `24/29`
  - diagonal-K LISTA: `24/29`
- Shared-system comparison against `generic_sparse`:
  - dense LISTA wins `15/29`
  - block-diagonal LISTA wins `3/29`
  - diagonal-K LISTA wins `3/29`
- Interpretation note:
  - `lista_diagonal` is now retired from active paper scope. Keep the completed diagonal numbers only as historical context; do not allocate new experiment budget to diagonal reruns.
- Primary audit files:
  - [v4 paper summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/paper_benchmark_summary.md)
  - [v4 final forecasting summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_summary.md)
  - [dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_dense_vs_generic_sparse/forecasting_comparison.md)
  - [block-diagonal vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - `generic_sparse` is still the strongest overall paper baseline.
  - Dense LISTA is now the only LISTA-family variant close enough to matter as a cross-system paper comparator.
  - Block-diagonal LISTA is not competitive as a global paper baseline, even though it still matters on some intrinsic-HD systems.
  - The current block-diagonal evidence still lacks the matched MLP-blockdiag control, so keep it out of any causal claim about Koopman structure until that fairness fix is complete.

### 2. Fair `200k` follow-up is complete: the fairness hole is closed, `generic_sparse` regains best median, dense LISTA still wins more systems, and block-diagonal dense-opt transfer is globally negative

- The full benchmark-wide `200k` follow-up is complete under [follow-up paper summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/paper_benchmark_summary.md):
  - same `29` systems, `3` seeds, and benchmark-selected pass-2 `dt`
  - new training arms: `generic_sparse_ns200k_best`, `lista_blockdiag_ns200k_denseopt_sc3em3`, `lista_blockdiag_ns200k_denseopt_sc6em3`
  - existing comparison-only root: promoted dense Stage-4 root `lista_dense_promoted_stage4`
- `generic_sparse_ns200k_best` closes the anchor fairness hole:
  - cross-system median `H1000` best-periodic is `0.0208`, ahead of promoted dense Stage 4 (`0.0232`) and canonical `v4` `generic_sparse` (`0.0328`)
  - versus canonical `generic_sparse`, it wins `19/29` systems with median ratio `0.8323`
  - good-system count stays `25/29`, but catastrophic systems rise from `2` to `3`
  - the most important regression is `kuramoto` (`65.70 -> 8566.42`), while notable gains include `hopfield` (`199.50 -> 121.98`) and `dysts:LorenzCoupled` (`0.0453 -> 0.0208`)
- Promoted dense Stage 4 remains the strongest cross-system LISTA read in pairwise terms:
  - dense still wins `18/29` systems versus `generic_sparse_ns200k_best`
  - dense keeps `26/29` good systems versus `25/29` for `generic_sparse_ns200k_best`
  - there are `0` systems where dense fails while the fair `200k` `generic_sparse` rerun passes
  - the remaining dense failures are still concentrated on `kuramoto`, `hopfield`, and `multiwell_strong_transition_hd`
  - **Fixed-cadence ablation (completed):** under a single global `periodic_100`, promoted dense Stage 4 still beats `generic_sparse_ns200k_best` on `17/29` systems overall (`13/23` non-intrinsic-HD), but the good-system count falls to a tie (`22/29` vs `22/29`) and dense newly fails `blended` while the fair anchor passes. The win-count advantage survives but the safety margin does not.
- The dense-optimizer block-diagonal transfer does not rescue block-diagonal LISTA globally:
  - `lista_blockdiag_ns200k_denseopt_sc3em3`: median `0.0555`, wins `7/29` vs `generic_sparse_ns200k_best`
  - `lista_blockdiag_ns200k_denseopt_sc6em3`: median `0.0477`, wins `5/29` vs `generic_sparse_ns200k_best`
  - both arms still provide targeted positives on `multiwell_strong_transition` and `multiwell_strong_transition_hd`, but neither is globally competitive against the fair `200k` MLP rerun or the promoted dense root
- Primary audit files:
  - [follow-up paper summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/paper_benchmark_summary.md)
  - [follow-up forecasting summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_summary.md)
  - [fair `200k` `generic_sparse` vs canonical `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_canonical_generic_sparse/generic_sparse_ns200k_best_vs_generic_sparse/forecasting_comparison.md)
  - [promoted dense Stage 4 vs fair `200k` `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_generic_sparse_ns200k_best/lista_dense_promoted_stage4_vs_generic_sparse_ns200k_best/forecasting_comparison.md)
  - [block-diagonal `sc=0.006` vs fair `200k` `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_generic_sparse_ns200k_best/lista_blockdiag_ns200k_denseopt_sc6em3_vs_generic_sparse_ns200k_best/forecasting_comparison.md)
- Interpretation:
  - the paper no longer has a `50k` vs `200k` fairness hole
  - `generic_sparse` remains the overall paper anchor once matched at `200k`
  - dense LISTA remains the only LISTA-family result that is globally competitive, but the paper narrative must now distinguish “best overall median” from “wins more systems / keeps more systems in-band”
  - block-diagonal dense-opt transfer is a negative full-benchmark result; keep block-diagonal claims targeted to smaller-`dt` hard-system evidence and the isolated multiwell transition repairs

### 3. Architecture-fixed dense LISTA tuning is now decision-grade: one fair fixed recipe beats the fixed `generic_sparse` anchor on most systems, but the hard cases still define the limitation story

- The dense-LISTA easy-system parity Stage 1 is complete under [dense-LISTA easy-system Stage 1 summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md):
  - same dense LISTA architecture and benchmark-selected `dt` on all `8` target systems
  - Stage 1 changed only `num_steps`, `lr`, and `k_matrix_lr`
- Best Stage-1 recipes against the fixed `generic_sparse` anchor:
  - `lista_dense_ns100k_lr5em5_klr5em6_wd1em4` wins `6/8` target systems with median dense/generic ratio `0.8699`
  - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4` wins `5/8` target systems with median ratio `0.7888`
- Best per-system dense recipe still loses on:
  - `competitive_lv` (`1.764x` vs `generic_sparse`)
  - `duffing` (`1.041x` vs `generic_sparse`)
- Positive details:
  - all `9` Stage-1 dense-LISTA recipes keep `8/8` target systems under the good-forecast band
  - no Stage-1 dense recipe is catastrophic on the target set
- Primary audit files:
  - [Stage-1 paper summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md)
  - [Stage-1 forecasting summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/forecasting_summary.md)
  - [best win-count comparison](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/compare/lista_dense_ns100k_lr5em5_klr5em6_wd1em4_vs_generic_sparse/forecasting_comparison.md)
  - [best median-ratio comparison](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - The dense-LISTA gap on the easier accepted-default systems is now clearly partly optimization-limited, not purely architectural.
  - This is strong support for a fairness-preserving dense-LISTA recovery story.
  - It is still not enough to claim dense LISTA is better than `generic_sparse` on most systems overall, because the result is limited to the targeted `8`-system subset and the holdouts remain real.
- The coefficient-only Stage 2 holdout sweep is now complete under [Stage-2 forecasting summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage2_20260308/collect/forecasting_summary.md):
  - `duffing` is flipped only by the specialized `100k, sc=0.012` recipe (`0.0182` vs `0.0309`, `0.590x`)
  - `competitive_lv` is not flipped by any coefficient-only recipe
  - the best global-compromise holdout recipe is `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
- The exact `8`-system validation Stage 3 is now complete under [Stage-3 paper summary](/home/mila/l/lia/skae/results/dense_lista_recipe_validation_stage3_20260309/collect/paper_benchmark_summary.md):
  - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` wins `6/8` shared systems vs `generic_sparse`
  - shared-system median `H1000` best-periodic ratio is `0.6928`
  - `8/8` systems stay under the good-forecast band with `0` catastrophic systems
  - all seeds are good on all `8` systems
  - the cheaper `100k, sc=0.003` recipe reaches `3/8` wins
  - the Duffing-fixing `100k, sc=0.012` recipe falls to `2/8` wins overall
- Interpretation:
  - the dense-LISTA fairness question is now resolved: promote `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` as the single fair dense-LISTA recipe
  - the paper story is no longer “LISTA almost catches up if tuned enough”; it is “a fixed dense LISTA architecture recovers most easy-system near-misses with one fair external recipe, but still leaves a persistent `competitive_lv`-style holdout and does not overturn the global `generic_sparse` ranking”
- The promoted dense-LISTA full `29`-system rerun is now complete under [Stage-4 paper summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md):
  - same dense LISTA architecture, promoted Stage-3 recipe, and benchmark-selected pass-2 `dt` table
  - compared against the fixed `generic_sparse` `v4` anchor, dense LISTA wins `21/29` shared systems with median dense/generic ratio `0.6455`
  - cross-system median `H1000` best-periodic improves from `0.0328` to `0.0232`
  - good-system count improves from `25/29` to `26/29`
  - there are `0` systems where the promoted dense recipe fails the good-forecast band while `generic_sparse` passes
  - the remaining dense failures are still concentrated on the hard systems, especially `kuramoto` (`48.50`), `hopfield` (`1.578e+06`), and `multiwell_strong_transition_hd` (`4.533e+04`)
- Primary audit files:
  - [Stage-4 paper summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md)
  - [Stage-4 forecasting summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/forecasting_summary.md)
  - [Stage-4 dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - the dense-LISTA parity story is now stronger than the Stage-3 subset result alone: one fixed dense recipe beats the fixed `generic_sparse` anchor on most benchmark systems overall
  - this does not replace the symmetric `v4` matrix as the canonical all-model paper benchmark, because only dense LISTA was rerun
  - the remaining dense-LISTA paper risk is no longer “can it catch up on the easier systems?”; it is whether the paper cleanly separates the cross-system parity win from the unresolved hard-system failures

### 4. `dt` rescue finished operationally, but step size remains a real paper bottleneck

- The repaired `dt` resolution completed through pass `2`:
  - `15/29` systems accept default `dt`
  - `4/29` systems accept after at least one halving
  - `10/29` systems remain `integration_hard`
- The most important remaining `integration_hard` systems are:
  - `kuramoto` (`selected dt = 0.0125`)
  - `hopfield` (`selected dt = 0.0125`)
  - `lotka_volterra`
  - `multiwell_strong_transition`
  - `multiwell_gradient_hd`
  - `multiwell_rotational_hd`
  - `multiwell_strong_transition_hd`
  - `dysts:DequanLi`
  - `dysts:WangSun`
  - `dysts:LorenzCoupled`
- Current high-dimensional bottlenecks at the selected smaller `dt` are still bad:
  - `kuramoto`, `generic_sparse`, system-median `H1000` best-periodic: `65.7014`
  - `hopfield`, `generic_sparse`, system-median `H1000` best-periodic: `199.4978`
  - `kuramoto`, `lista_blockdiag`, system-median `H1000` best-periodic: `14.2618`
  - `hopfield`, dense LISTA, system-median `H1000` best-periodic: `7.241e+09`
- Primary audit file:
  - [v4 pass-2 `dt` resolution summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/dt_resolution.md)
- Interpretation:
  - the benchmark is no longer blocked by queue completion
  - the open paper problem shifted from "finish the rerun" to "what to do with systems that stay hard even after the allowed `dt` rescue"

### 5. Intrinsic-HD follow-up is now decision-grade: smaller `dt` plus longer training rescues Kuramoto through `N=32` and Hopfield at `N=16` under periodic reencoding, but not autonomous rollouts

- The repaired focused intrinsic-HD rerun is complete under [intrinsic-HD `dt` rescue rerun summary](/home/mila/l/lia/skae/results/intrinsic_hd_dt_rescue_20260308_rerun1/forecasting_summary.md):
  - all `48` rows are collected
  - official selection is still based on `evaluation_results_best.json`
- Best current intrinsic-HD arms at `H1000` best-periodic:
  - `kuramoto`:
    - `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`: `14.36`
    - matched `generic_sparse`, `dt=0.0125`, `sp=0.0005`: `25.93`
  - `hopfield`:
    - `generic_sparse`, `dt=0.0125`, `sp=0.0005`: `71.02`
    - best `lista_blockdiag`, `dt=0.0125`, `sp=0.0010`: `80.54`
- Smaller `dt=0.0125` beats `dt=0.025` for both systems in both model families, so the step-size hypothesis is now directly supported by a full focused rerun.
- A diagnostic recollection from `evaluation_results_last.json` shows checkpoint-selection mismatch on Kuramoto:
  - `lista_blockdiag`, `dt=0.0125`, system-median `H1000` improves from `23.40` to `14.64` across the focused pilot grid
  - on the winning `lista_blockdiag`, `dt=0.0125`, `sp=0.0005` arm, the last-checkpoint median is `13.91`
  - this is a diagnostic, not yet the official paper metric, but it shows late training still matters on Kuramoto
- Completed focused Kuramoto `dt=0.00625`, `200k`, `5`-seed comparison under [Kuramoto comparison](/home/mila/l/lia/skae/results/kuramoto_dt00625_200k_compare_20260308/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `27.02`
  - dense LISTA: seed-median `H1000` best-periodic `13.84`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.98`
  - all five `lista_blockdiag` seeds are good and tightly clustered in `6.89-7.13`
  - **Fixed-cadence ablation (completed):** `periodic_100` exactly reproduces the official `best_periodic` `H1000` ranking for all three roots (`6.98`, `13.84`, `27.02`), so the long-horizon Kuramoto block-diagonal win is already a fixed-cadence result, not a `best_periodic` oracle artifact. At `H500`, `generic_sparse` still has lower error than `lista_blockdiag` under the fixed cadence.
  - **Checkpoint-selection ablation (completed):** switching from `evaluation_results_best.json` to `evaluation_results_last.json` on the current `dt=0.00625`, `200k` comparison does not change the model ranking or good-band membership (`lista_blockdiag`: `6.98 -> 7.00`, `lista_dense`: `13.84 -> 17.63`, `generic_sparse`: `27.02 -> 29.43`). The older `dt=0.0125` pilot mismatch (`23.40 -> 14.64`) is real but tied to the earlier, superseded setting. Keep `evaluation_results_best.json` as the official paper rule.
- Completed Hopfield `dt=0.00625`, `200k` follow-up under [Hopfield follow-up summary](/home/mila/l/lia/skae/results/hopfield_dt00625_200k_compare_20260309/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `3.36`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `8.82`
  - both are inside the good-forecast band on the system median, but every-step errors are still enormous for both
- Completed Kuramoto `N=32`, `dt=0.00625`, `200k`, `3`-seed confirmation under [Kuramoto `N=32` summary](/home/mila/l/lia/skae/results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `6.65` (all seeds good)
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.00` (all seeds good, std `0.33`)
- Completed Kuramoto dimension sweep under [Kuramoto dimension summary](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md):
  - dimensions: `N={8,16,24,32,64}`
  - models: `generic_sparse`, promoted dense LISTA, `lista_blockdiag`
  - fixed setting: `dt=0.00625`, `200k`, `5` seeds
  - `H1000` seed-median best-periodic by dimension:
    - `generic_sparse`: `813.57`, `30.18`, `6.71`, `6.68`, `208.93`
    - promoted dense LISTA: `495.07`, `13.44`, `15.01`, `92.28`, `208.71`
    - `lista_blockdiag`: `8.11`, `7.07`, `6.57`, `5.92`, `23.27`
  - seed robustness:
    - `lista_blockdiag` is all-seeds-good at `N=16/24/32`
    - `lista_blockdiag` is median-good but not fully robust at `N=8` (`4/5` good seeds, worst seed `10.89`)
    - `lista_blockdiag` falls out of band at `N=64` (`2/5` good seeds, worst seed `209.20`)
- Interpretation:
  - smaller `dt` is the dominant hard-system lever in the current regime
  - `lista_blockdiag` is the strongest hard-system LISTA result on Kuramoto and the only model that cleanly wins the `N=16` three-way comparison there
  - the completed Kuramoto sweep resolves the scaling story: the smaller-`dt` rescue is strong through `N=32`, but it is not dimension-robust through `N=64`
  - promoted dense LISTA does not transfer as a robust Kuramoto solution under this sweep
  - Hopfield is no longer a catastrophic boundary case in the targeted `N=16`, `dt=0.00625`, `200k` setting, but it is still not a structured-LISTA success story because `generic_sparse` remains better
  - the remaining scientific limitation is autonomous rollout stability, not whether periodic reencoding can rescue the hard systems at all
  - the remaining Kuramoto paper question is no longer whether scaling works at all; it is how to present a moderate-dimension success with an explicit `N=64` limit and nontrivial seed-instability at `N=8`
- Completed Kuramoto robustness evaluation (uniform frequency spread, `N=16`, `dt=0.00625`, `200k`, `5` seeds) under [Kuramoto robustness comparison](/home/mila/l/lia/skae/results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/compare/lista_blockdiag_uniform_spread_vs_generic_sparse_uniform_spread/forecasting_comparison.md):
  - `lista_blockdiag`: seed-median `H1000` best-periodic `9.53` (4/5 seeds good, std `0.64`)
  - `generic_sparse`: seed-median `H1000` best-periodic `44.46` (0/5 seeds good)
  - `4.7x` improvement; the Kuramoto block-diagonal positive is not a single-regime artifact
  - every-step errors remain catastrophic for both models under heterogeneity; periodic re-encoding is essential

## Recent Queue Activity

### Active now (paper-parallel workstreams, March 10)

The immediate paper-strengthening program is coordinated in `docs/planning/paper_parallel_workstreams_20260309/`.

- **Block-diagonal fairness-control implementation/re-run:** not started. This is now required before freezing any paper claim that compares `lista_blockdiag` against `generic_sparse`. Tracking plan: [docs/planning/generic_sparse_blockdiag_fairness_plan_20260311.md](/home/mila/l/lia/skae/docs/planning/generic_sparse_blockdiag_fairness_plan_20260311.md).
- **A - Competitive LV support alignment:** invalidated. The March 9 evaluation ran on the old 1-basin `competitive_lv` setup, so none of the prior `competitive_lv` forecasting, support-alignment, or label-free clustering results should be reused.
- **Competitive LV multi-basin retrain:** the first array attempt `8922033` finished in a mixed state under `results/competitive_lv_multibas_retrain_20260310/`: `17/28` rows completed and `11/28` rows failed. `generic_sparse` and block-diagonal rows ran, while dense and diagonal rows failed because empty optional TSV fields shift arguments in `scripts/run_competitive_lv_retrain_array.sh`. No clean rerun is queued yet. The clean rerun should drop diagonal and target `generic_sparse`, `lista_dense`, `lista_blockdiag`, and the matched `generic_sparse + block_diagonal K` control once available.
- **Hopfield basin-count mechanism sweep:** live on SLURM under `results/hopfield_basin_sweep_n64_dt00625_200k_20260310/` with wrapper `8922497`, array `8922498_[0-44]`, collector `8922499`, and comparison `8922500`.
  - setup: Hopfield only, `N=64`, `P in {8,10,12,14,16}`, `dt=0.00625`, `200k`, `seeds={0,1,2}`
  - roots: `generic_sparse`, `lista_dense_promoted_stage4`, `lista_blockdiag_targeted`
  - purpose: test whether increasing Hopfield basin count can create a regime where both LISTA baselines beat the MLP on periodic-reencoding forecasting
  - interpretation: this is a mechanistic system-design probe, not a replacement for the canonical paper benchmark, because the environment itself is being altered
  - current status: as of the latest SLURM check, all `45/45` array tasks are running and the collector/comparison remain pending on dependency
- **Kuramoto unique mode-support audit:** complete. All `30/30` array tasks and collector finished. The strong negative claim is confirmed: mode-support uniqueness is trivially degenerate across all 3 model families, all 5 seeds, both sampling protocols, all support definitions, and all thresholds. Every trajectory has its own unique support (`traj_unique=1.0`), mode supports are singletons, basin consistency is negligible, and Hamming geometry is flat (`ratio≈1.0`). Balanced probe `8922718_1` timed out (non-critical). Full results at `/network/scratch/l/lia/skae/kuramoto_mode_support_audit_20260310/summary/kuramoto_mode_support_audit_summary.md`.
- **B - Kuramoto support alignment:** closed as redundant. Label-free clustering v2 already showed the current Kuramoto checkpoints are non-separable.
- **C - Fixed-cadence re-encoding ablation:** complete. The dense win-count story survives `periodic_100`, and the Kuramoto `H1000` block-diagonal win is already a fixed-cadence result.
- **D - More seeds on headline positives:** complete. The low-dimensional `5`-seed extension leaves the dense-vs-MLP narrative unchanged, and the Kuramoto `N=16` and `N=32` confirmations both support the block-diagonal headline.
- **E - Kuramoto robustness:** complete. The uniform-spread heterogeneity check remains positive for `lista_blockdiag`.
- **F - Support transition dynamics:** not started and lower priority than closing `competitive_lv`.
- **G - Kuramoto checkpoint selection:** complete. The current Kuramoto headline is selection-stable.
- **H - Label-free clustering quality:** complete for all non-`competitive_lv` systems. v1 is retained only as a methodology limitation; v2 is final for multiwell, Duffing, and Kuramoto.

### Recently completed

- **Kuramoto unique mode-support audit (completed March 10):** All `30/30` array tasks and collector finished under `results/kuramoto_mode_support_audit_20260310/`. The strong negative claim is confirmed: Kuramoto winding-number basins do not have meaningful basin-specific support patterns. Mode-support uniqueness is trivially degenerate — every trajectory has its own unique support (`traj_unique=1.0`), mode supports are singletons, basin consistency is negligible (`0.0625` balanced / `0.309` random), and Hamming geometry is flat (`ratio≈1.0`). This holds identically across all 3 model families (`generic_sparse`, `lista_dense`, `lista_blockdiag`), all 5 seeds, both sampling protocols (`random`, `balanced`), all support modes (`mean`, `majority`, `modal`), and all threshold values (`1e-4` to `1e-1`). This closes the gap left by label-free clustering v2 and directly confirms both claims: basins are not recoverable from latent features, and basins do not have literal reusable mode supports.
- **Broad support-alignment audit on labelable `v4` systems:** complete under [support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md). Across `11` valid labelable systems (`132` checkpoints), binary `mode_uniqueness_rate` saturated at `1.0` on all `44/44` system-root medians, while cosine separation still cleanly split the systems: multiwell positive, Duffing negative, Kuramoto negative, and Hopfield mixed.
- **Fair `200k` follow-up benchmark:** complete. `generic_sparse_ns200k_best` is the best full-benchmark root by cross-system median `H1000` best-periodic (`0.0208`), while promoted dense Stage 4 still wins `18/29` shared systems and keeps `26/29` good systems.
- **Dense LISTA promoted Stage 4 rerun:** complete. One fixed fair dense recipe wins `21/29` systems against the fixed `generic_sparse` anchor and improves the dense median to `0.0232`.
- **Kuramoto dimension sweep:** complete. `lista_blockdiag` is robust through `N=32`, not fully robust at `N=8`, and no longer in-band at `N=64`.
- **Focused Kuramoto and Hopfield smaller-`dt` follow-ups:** complete. `lista_blockdiag` wins the Kuramoto `dt=0.00625`, `200k` comparison; Hopfield enters the good band for both models under the smaller `dt`, but `generic_sparse` remains better.
- **Label-free clustering v2:** complete on array `8919951` with collector `8919952`. Multiwell systems are strongly positive, Duffing is weakly positive, and Kuramoto is negative.

### 6. Broad support-alignment audit: binary mode uniqueness saturates, cosine separation carries the real signal

- **Result (COMPLETE, March 11, local audit):** Evaluated the canonical `v4` checkpoints on all currently valid labelable benchmark systems under [support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md).
  - scope: `11` systems (`duffing`, `8` `multiwell*` variants, `kuramoto`, `hopfield`) x `4` roots x `3` seeds = `132` checkpoints
  - excluded: `competitive_lv`, because the canonical `v4` checkpoints used the invalidated 1-basin configuration
  - settings: `100` trajectories, length `500`, `5000`-step basin rollout, `support_threshold=1e-3`, `support_mode=mean`
- **Concrete result:**
  - binary mode-support uniqueness saturates completely: **all `44/44` system-root medians have `mode_uniqueness_rate=1.0`**
  - support reuse is weak almost everywhere: **`40/44` system-root medians have `mean_basin_consistency < 0.2`**
  - trajectory-level supports are often unique: **`24/44` system-root medians have `trajectory_unique_support_rate = 1.0`**
  - all multiwell system-root medians are positive by cosine separation (`0.250` to `0.706`)
  - Duffing is negative across all roots (`-0.129` to `-0.084`) despite perfect mode uniqueness
  - Kuramoto is negative across all roots (`-0.307` to `-0.264`) despite perfect mode uniqueness; the random `100`-trajectory audit again produces singleton `q=±2` basins, so the apparent `mean_basin_consistency≈0.424` is inflated and not evidence of reusable basin supports
  - Hopfield is mixed: cosine separation is positive across all roots (`0.459` to `0.607`), but `mean_basin_consistency` is only `0.043` for every root and `trajectory_unique_support_rate=1.0` throughout
- **Interpretation:**
  - the literal binary question "does each basin have a unique mode support?" is too weak as a broad benchmark diagnostic, because it returns a perfect score even on known negatives like Duffing and Kuramoto
  - cosine separation reproduces the known qualitative split and should remain the primary support-alignment metric
  - Hopfield currently shows continuous basin separation without reusable sparse support signatures: basin centroids separate, but trajectories do not reuse a stable support within each basin
- **Paper implication:** do not make a benchmark-wide paper claim based on `mode_uniqueness_rate`. Keep the support story tied to multiwell cosine/clustering positives, scope Duffing and Kuramoto as negatives, and treat Hopfield as a mixed continuous-separation-only case.

### 7. Label-free basin recovery: v2 validates label-free clustering on potential-well systems

- **v1 result (methodology limitation, March 10):** The initial label-free clustering evaluation used trajectory-mean cosine k-means on 128 trajectories in 256 dimensions. Results:
  - Duffing (2 basins): ARI=`0.134` — all three models produce **identical** scores, confirming the feature extraction protocol (not the encoder) is the bottleneck
  - Kuramoto (5 basins): ARI≈`0` for all models
  - Competitive LV: only 1 basin observed (trivial, now fixed — see competitive_lv retrain below)
- **Root cause:** v1 protocol destroyed per-timestep support signal via trajectory averaging, suffered from concentration of measure (no PCA), and tested only the cosine feature view. The identical Duffing scores across all encoder families confirmed this was a feature-extraction issue.
- **v2 result (COMPLETE, March 10, array `8919951`, collector `8919952`):** Revised evaluation with 6 feature views, PCA to 20d, 256 trajectories, 10 systems.
  - **Multiwell systems (8 variants, 5 basins each): strong positive.**
    - `multiwell_gradient/gradient_hd`: mean ARI `0.976/0.991`, near-perfect recovery (max `1.000`)
    - `multiwell_rotational/rotational_hd`: mean ARI `0.963/0.971`
    - `multiwell_energy/energy_hd`: mean ARI `0.794/0.916`
    - `multiwell_strong_transition/strong_transition_hd`: mean ARI `0.931/0.918`
    - `generic_sparse` tends to have highest ARI; LISTA families close behind
  - **Duffing (2 basins): weak positive.** Mean ARI `0.19–0.24` across all views. Root cause: within-basin support consistency is only ~10% (basin 0: 12.8%, basin 1: 7.5%), so ~90% of trajectories activate a different support than their basin's mode. The encoder learned basin-discriminative continuous representations but not basin-aligned sparse supports on this system.
  - **Kuramoto (5 basins): negative.** Mean ARI ~`0` across all views. Supports are genuinely non-separable: within-basin vs between-basin Hamming distance ratio is 1.004 (flat). Winding-number basin distribution is highly imbalanced (q=0: 59%, q=±2: <1%). **Bug fix:** the v2 linear accuracy (~0.92–0.99 on support views) was a measurement artifact — singleton basins caused a fallback to train accuracy with no CV; corrected 3-class CV gives `0.427` (below majority baseline). Fix applied in `evaluate_label_free_clustering_v2.py`: singleton classes are now dropped before CV.
  - **Direct uniqueness update:** this v2 negative is strong evidence against basin recoverability, and the completed Kuramoto unique mode-support audit now directly confirms that Kuramoto also lacks meaningful literal basin-specific mode supports. Uniqueness is trivially degenerate (every trajectory has its own singleton support), so the Kuramoto negative is established on both the clustering and literal-support-uniqueness fronts.
  - **Feature view comparison:** `last_step_cosine` is strongest on multiwell; discrete support views (`majority_support`, `modal_support`) are competitive but do not clearly outperform PCA'd cosine features; `traj_mean_cosine` (v1 baseline) is comparable after PCA, suggesting v1 failure was primarily concentration-of-measure rather than averaging.
- **Interpretation:**
  - The multiwell positives (8/8 systems, ARI 0.71–1.00) upgrade the basin-support claim from "per-timestep uniqueness" to **"label-free basin recovery is possible"** without training-time basin labels. This is a key paper claim.
  - The kuramoto negative is genuine — supports carry zero basin-discriminative signal (flat Hamming geometry, every trajectory unique, ~93 active dims in all basins). This limits the label-free claim to potential-well systems.
  - The duffing weak result demonstrates that per-timestep support uniqueness (2/2) does not guarantee trajectory-level basin-support alignment when within-basin consistency is low (~10%). This is an honest limitation worth reporting.
- **Competitive LV multi-basin retrain (March 10):** The previous `competitive_lv` benchmark was trivial (1 observed basin at `INTERACTION_SCALE=0.35`). The config is now `0.70`, producing 4 major basins. The first retrain attempt `8922033` finished in a mixed state: `17/28` rows completed and `11/28` rows failed because the array runner shifts empty optional TSV columns on dense/diagonal rows. No clean rerun is queued yet. The clean rerun should drop diagonal and focus on `generic_sparse`, `lista_dense`, `lista_blockdiag`, and the matched `generic_sparse + block_diagonal K` control once available. After that rerun: (1) collect forecasting, (2) re-run support alignment (Subagent A), and (3) run label-free clustering v2 on the new 4-basin checkpoints. Do NOT re-use any old `competitive_lv` checkpoints or evaluation results.

## Decision Rules

- If default `dt` is poor and the benchmark rescue chain requests halving, prefer **smaller `dt`** before broader model changes or `10x` longer training.
- **The `200k` results are now the primary paper evidence.** The `v4` `50k` matrix is retained as a historical audit and for the symmetric four-model comparison, but all headline paper claims, cross-system rankings, and model comparisons should be drawn from the `200k` runs in `results/paper_followup_recipes_200k_20260309`.
- Use `generic_sparse_ns200k_best` as the primary paper anchor (not the `50k` `generic_sparse`).
- Use the promoted dense Stage-4 root (`lista_dense_promoted_stage4`) as the primary dense LISTA comparator.
- Use dense LISTA as the cross-system LISTA reference, but keep `lista_blockdiag` as the only LISTA-family candidate for intrinsic-HD follow-up unless new evidence clearly overturns that ranking.
- Treat the completed dense-LISTA easy-system Stage-1 sweep as evidence that external optimization alone can recover most easy dense-LISTA near-misses without changing architecture or `dt`.
- Treat the completed dense-LISTA Stage-2 / Stage-4 chain as the parity decision point:
  - stop coefficient-only holdout tuning
  - promote `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` as the single fair dense recipe
  - use the completed Stage-4 rerun as the current dense parity evidence (`21/29` wins, `26/29` good systems, `0` dense-fails-anchor-passes systems)
  - when discussing the fair `200k` comparison, make the split explicit: dense wins more systems and keeps more systems good, while `generic_sparse_ns200k_best` has the best overall median
  - `v4` remains a useful symmetric four-model audit; the `200k` follow-up is the primary paper-facing comparison
- Treat the full-benchmark block-diagonal dense-opt transfer as a negative result for global parity:
  - do not promote `lista_blockdiag_ns200k_denseopt_sc3em3` or `lista_blockdiag_ns200k_denseopt_sc6em3` as paper baselines
  - mention them only as targeted positives on `multiwell_strong_transition` / `multiwell_strong_transition_hd` or as evidence that the dense optimizer does not transfer cleanly to block-diagonal LISTA
- Treat the missing `generic_sparse + block_diagonal K` control as a paper-blocking fairness gap for every `lista_blockdiag` claim:
  - every paper-facing experiment that includes `lista_blockdiag*` must gain a matched `generic_sparse_blockdiag*` control with the same non-encoder recipe
  - until those controls exist, frame block-diagonal results as end-to-end variant comparisons, not isolated structure effects
- Use the repaired intrinsic-HD rerun as the current decision-grade targeted evidence.
- Keep `evaluation_results_best.json` as the official checkpoint-selection rule for now, but treat `evaluation_results_last.json` as an important diagnostic on Kuramoto when discussing model-selection limits.
- Use the completed `dt=0.00625`, `200k` follow-ups and Kuramoto dimension sweep as the hard-system evidence:
  - on Kuramoto, emphasize that `lista_blockdiag` is robustly in-band at `N=16/24/32`, is not fully robust at `N=8`, and fails by `N=64`
  - on Hopfield, emphasize that smaller `dt` rescues periodic-reencoding forecasts for both models, but `generic_sparse` remains better
  - do not claim autonomous stability on the hard systems; every-step rollout errors remain the main limitation
- Treat the active Hopfield basin-count sweep as mechanism mapping only:
  - use it to test whether higher basin count changes the architecture ordering on Hopfield
  - do not use it to overwrite the canonical Hopfield paper claim unless the modified environment is explicitly framed as a new benchmark setting
- Use the completed Kuramoto dimension sweep to frame the hard-system claim:
  - claim a smaller-`dt` Kuramoto rescue for `lista_blockdiag` through `N=32`, not through `N=64`
  - make the `N=64` failure (`23.27`, `2/5` good seeds) and the non-robust `N=8` result (`8.11`, `4/5` good seeds) explicit
  - do not present promoted dense LISTA as a positive Kuramoto transfer result

## Highest-Value Audit Files

- [Current experiment log](/home/mila/l/lia/skae/docs/EXPERIMENTS.md)
- [Broad labelable-system support-alignment summary](/home/mila/l/lia/skae/results/paper_benchmark_support_alignment_20260311_v4_labelable/summary.md)
- [v4 paper summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/paper_benchmark_summary.md)
- [v4 final forecasting summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_summary.md)
- [follow-up `200k` paper summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/paper_benchmark_summary.md)
- [follow-up `200k` forecasting summary](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/collect/forecasting_summary.md)
- [fair `200k` `generic_sparse` vs canonical `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_canonical_generic_sparse/generic_sparse_ns200k_best_vs_generic_sparse/forecasting_comparison.md)
- [promoted dense Stage 4 vs fair `200k` `generic_sparse`](/home/mila/l/lia/skae/results/paper_followup_recipes_200k_20260309/compare/vs_generic_sparse_ns200k_best/lista_dense_promoted_stage4_vs_generic_sparse_ns200k_best/forecasting_comparison.md)
- [Stage-4 dense rerun summary](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/collect/paper_benchmark_summary.md)
- [Stage-4 dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/dense_lista_paper_rerun_stage4_20260309/compare/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3_vs_generic_sparse/forecasting_comparison.md)
- [Kuramoto dimension summary](/home/mila/l/lia/skae/results/kuramoto_dimension_sweep_dt00625_200k_20260309/collect/kuramoto_dimension_summary.md)
- [v4 pass-2 `dt` resolution summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/dt_resolution.md)
- [dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_dense_vs_generic_sparse/forecasting_comparison.md)
- [block-diagonal vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md)
- [Stage-1 dense-LISTA easy-system summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md)
- [Kuramoto recovery summary](/home/mila/l/lia/skae/results/kuramoto_recovery_seq8_20260305/forecasting_summary.md)
- [intrinsic-HD `dt` rescue rerun summary](/home/mila/l/lia/skae/results/intrinsic_hd_dt_rescue_20260308_rerun1/forecasting_summary.md)
