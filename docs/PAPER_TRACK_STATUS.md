# Paper Track Status

Date: March 9, 2026

## Goal

The paper target is now explicit:

- maximize **in-time prediction** and **periodic-reencoding forecasting** quality on nonlinear dynamical systems with multiple basins of attraction
- treat **step size** and **dimension** as the primary current bottlenecks
- when default-`dt` performance is clearly bad, prefer **reducing `dt`** before spending more budget on architecture churn
- optimize for forecasting-first evidence; sparsity and support structure matter, but only after the forecasting stack is competitive

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
- Primary audit files:
  - [v4 paper summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/paper_benchmark_summary.md)
  - [v4 final forecasting summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_summary.md)
  - [dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_dense_vs_generic_sparse/forecasting_comparison.md)
  - [block-diagonal vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md)
- Interpretation:
  - `generic_sparse` is still the strongest overall paper baseline.
  - Dense LISTA is now the only LISTA-family variant close enough to matter as a cross-system paper comparator.
  - Block-diagonal LISTA is not competitive as a global paper baseline, even though it still matters on some intrinsic-HD systems.

### 2. Architecture-fixed dense LISTA tuning now produces one promotable fair recipe on the targeted easy-system subset

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

### 3. `dt` rescue finished operationally, but step size remains a real paper bottleneck

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

### 4. Intrinsic-HD follow-up is now decision-grade: smaller `dt` plus longer training rescues the targeted hard systems under periodic reencoding, but not autonomous rollouts

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
- Completed Hopfield `dt=0.00625`, `200k` follow-up under [Hopfield follow-up summary](/home/mila/l/lia/skae/results/hopfield_dt00625_200k_compare_20260309/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `3.36`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `8.82`
  - both are inside the good-forecast band on the system median, but every-step errors are still enormous for both
- Completed Kuramoto `N=32`, `dt=0.00625`, `200k` confirmation under [Kuramoto `N=32` summary](/home/mila/l/lia/skae/results/kuramoto_n32_dt00625_200k_confirm_20260309/forecasting_summary.md):
  - `generic_sparse`: seed-median `H1000` best-periodic `6.65`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.00`
- Active Kuramoto dimension sweep under `results/kuramoto_dimension_sweep_dt00625_200k_20260309`:
  - dimensions: `N={8,16,24,32,64}`
  - models: `generic_sparse`, promoted dense LISTA, `lista_blockdiag`
  - fixed setting: `dt=0.00625`, `200k`, `5` seeds
- Interpretation:
  - smaller `dt` is the dominant hard-system lever in the current regime
  - `lista_blockdiag` is the strongest hard-system LISTA result on Kuramoto and the only model that cleanly wins the `N=16` three-way comparison there
  - Hopfield is no longer a catastrophic boundary case in the targeted `N=16`, `dt=0.00625`, `200k` setting, but it is still not a structured-LISTA success story because `generic_sparse` remains better
  - the remaining scientific limitation is autonomous rollout stability, not whether periodic reencoding can rescue the hard systems at all
  - the remaining Kuramoto paper question is now whether that rescue scales cleanly with dimension and whether dense LISTA transfers at all beyond the original `N=16` case

## Recent Queue Activity

### Active now

- Promoted dense-LISTA full `29`-system rerun:
  - queue launcher / array: `8909900_[0-86]`
  - collector: `8909901`
  - comparison: `8909902`
  - output roots:
    - `/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309`
    - `results/dense_lista_paper_rerun_stage4_20260309`
  - recipe:
    - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
  - fairness constraints:
    - fixed dense LISTA architecture
    - benchmark-selected pass-2 `dt` table from the completed `v4` run
    - comparison against the existing `generic_sparse` `v4` anchor
  - current live queue state:
    - the array is broadly running across the cluster
    - collect / compare are waiting on dependencies
- Kuramoto dimension sweep:
  - queue launcher / array: `8910056_[0-74]`
  - collector: `8910057`
  - comparison: `8910061`
  - output roots:
    - `/network/scratch/l/lia/skae/kuramoto_dimension_sweep_dt00625_200k_20260309`
    - `results/kuramoto_dimension_sweep_dt00625_200k_20260309`
  - experimental scope:
    - `N={8,16,24,32,64}`
    - `generic_sparse`, promoted dense LISTA, `lista_blockdiag`
    - `dt=0.00625`, `200k`, `5` seeds
  - current live queue state:
    - the array is broadly running across the cluster
    - collect / compare are waiting on dependencies

### Completed in the last 24 hours

- Dense LISTA easy-system parity Stage 2:
  - launcher: `8907833`
  - array: `8907833_[0-83]`
  - collector: `8907834`
  - comparison: `8907835`
  - result:
    - `duffing` flips only under the specialized `100k, sc=0.012` recipe
    - `competitive_lv` does not flip
    - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` is the best fair holdout recipe
- Dense LISTA exact recipe validation:
  - launcher: `8908839`
  - result:
    - the promoted dense recipe is `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
    - it wins `6/8` shared systems with `0` catastrophic runs
- Focused Kuramoto `dt=0.00625`, `200k` comparison:
  - launcher: `8907758`
  - array: `8907759_[0-14]`
  - collector: `8907760`
  - comparison: `8907761`
  - result:
    - `lista_blockdiag` wins the direct Kuramoto comparison (`6.98` vs `27.02` for `generic_sparse`, `13.84` for dense LISTA)
- Hopfield `dt=0.00625`, `200k` follow-up:
  - launcher: `8908838`
  - result:
    - both `generic_sparse` and `lista_blockdiag` enter the good-forecast band on Hopfield, but `generic_sparse` remains clearly better
- Kuramoto `N=32`, `dt=0.00625`, `200k` confirmation:
  - launcher: `8908842`
  - result:
    - the positive Kuramoto result survives at `N=32`, with a small block-diagonal edge (`6.00` vs `6.65`)
- Dense LISTA easy-system parity Stage 1:
  - launcher: `8906725`
  - array: `8906726_[0-215]`
  - collector: `8906727`
  - comparison: `8906728`
  - result:
    - external optimization alone flips `6/8` targeted easy near-misses at best
    - the strongest win-count recipe is `lista_dense_ns100k_lr5em5_klr5em6_wd1em4`
    - the strongest median-ratio recipe is `lista_dense_ns200k_lr5em5_klr5em6_wd1em4`
    - `competitive_lv` and `duffing` remain dense-LISTA holdouts
- Canonical paper benchmark `v4` chain:
  - rescue pass `1` array: `8903420_[0-41]`
  - rescue pass `1` collector: `8903421`
  - resolver after pass `1`: `8903422`
  - rescue pass `2` array: `8903423_[0-86]`
  - rescue pass `2` collector: `8903424`
  - resolver after pass `2`: `8903425`
  - full matrix array: `8903426_[0-347]`
  - final collect: `8903427`
  - final compare: `8903428`
- Focused intrinsic-HD `dt` rescue pilot:
  - launcher: `8903785`
  - `generic_sparse` array: `8903787_[0-23]`
  - `lista_blockdiag` array: `8903788_[0-23]`
  - collector: `8903789`
  - audit result: the launcher queued `0-23` correctly, but the child jobs saw `TOTAL_JOBS=1` because comma-separated CSV values were passed through `sbatch --export`
- Repaired intrinsic-HD `dt` rescue rerun:
  - launcher: `8906425`
  - `generic_sparse` array: `8906426_[0-23]`
  - `lista_blockdiag` array: `8906427_[0-23]`
  - collector: `8906428`
  - result:
    - full `48`-row collection completed under [intrinsic-HD `dt` rescue rerun summary](/home/mila/l/lia/skae/results/intrinsic_hd_dt_rescue_20260308_rerun1/forecasting_summary.md)
    - smaller `dt` improves both `kuramoto` and `hopfield`
    - `lista_blockdiag` wins the focused Kuramoto comparison, but Hopfield remains unsolved
- Current cluster snapshot:
  - the targeted March 9 follow-up queues are complete
  - the promoted dense-LISTA full rerun is now the only active paper-track submission

## Decision Rules

- If default `dt` is poor and the benchmark rescue chain requests halving, prefer **smaller `dt`** before broader model changes or `10x` longer training.
- Treat the completed `v4` `final_collect` artifacts as the current canonical paper matrix for cross-system claims.
- Keep `generic_sparse` as the overall paper anchor.
- Use dense LISTA as the cross-system LISTA reference, but keep `lista_blockdiag` as the only LISTA-family candidate for intrinsic-HD follow-up unless new evidence clearly overturns that ranking.
- Treat the completed dense-LISTA easy-system Stage-1 sweep as evidence that external optimization alone can recover most easy dense-LISTA near-misses without changing architecture or `dt`.
- Treat the completed dense-LISTA Stage-2 / Stage-3 chain as the parity decision point:
  - stop coefficient-only holdout tuning
  - promote `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` as the single fair dense recipe
  - if more benchmark budget is approved, spend it on a full `29`-system rerun of that recipe
- Use the repaired intrinsic-HD rerun as the current decision-grade targeted evidence.
- Keep `evaluation_results_best.json` as the official checkpoint-selection rule for now, but treat `evaluation_results_last.json` as an important diagnostic on Kuramoto when discussing model-selection limits.
- Use the completed `dt=0.00625`, `200k` follow-ups as the hard-system evidence:
  - on Kuramoto, emphasize that `lista_blockdiag` wins cleanly at `N=16` and still holds a slight edge at `N=32`
  - on Hopfield, emphasize that smaller `dt` rescues periodic-reencoding forecasts for both models, but `generic_sparse` remains better
  - do not claim autonomous stability on the hard systems; every-step rollout errors remain the main limitation
- Use the active Kuramoto dimension sweep to decide how strong the hard-system claim can be:
  - if `lista_blockdiag` stays in-band through `N=64`, claim a meaningful dimension-robust Kuramoto rescue under smaller `dt`
  - if performance degrades materially with `N`, present the result as a moderate-dimension success with explicit scaling limits
  - treat any promoted-dense success on Kuramoto as a secondary transfer story, not the primary intrinsic-HD claim

## Highest-Value Audit Files

- [Current experiment log](/home/mila/l/lia/skae/docs/EXPERIMENTS.md)
- [v4 paper summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/paper_benchmark_summary.md)
- [v4 final forecasting summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/forecasting_summary.md)
- [v4 pass-2 `dt` resolution summary](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/dt_resolution.md)
- [dense vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_dense_vs_generic_sparse/forecasting_comparison.md)
- [block-diagonal vs `generic_sparse` comparison](/home/mila/l/lia/skae/results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md)
- [Stage-1 dense-LISTA easy-system summary](/home/mila/l/lia/skae/results/dense_lista_easy_parity_stage1_20260308/collect/paper_benchmark_summary.md)
- [Kuramoto recovery summary](/home/mila/l/lia/skae/results/kuramoto_recovery_seq8_20260305/forecasting_summary.md)
- [intrinsic-HD `dt` rescue summary](/home/mila/l/lia/skae/results/intrinsic_hd_dt_rescue_20260308/forecasting_summary.md)
