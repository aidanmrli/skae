# Experiments (Core)

Date: March 9, 2026

## Current Status Summary

Paper-track directive:
- `docs/PAPER_TRACK_STATUS.md` is the high-level source of truth for paper-facing claims, wrap-up priorities, and remaining blockers.
- This file remains the detailed experiment ledger that backs the paper-track view.

Wrap-up objective:
- We are actively pushing to wrap up the project and convert the current evidence into a publishable top-tier machine learning conference paper, with NeurIPS as the default target venue.

Problem we are solving:
- Learn sparse, basin-discriminative latent supports with stable long-horizon Koopman rollouts, so each support-defined regime can be used for local linear control.

Assumption split:
- Training/deployment target: basin count and basin labels are unknown.
- Benchmark evaluation: known basin counts/labels are allowed for diagnostics.

What we now know (high-confidence):
- Basin-support uniqueness is achievable at sufficient latent capacity (typically `target_size >= 256`), and cosine-based diagnostics are the reliable primary metric.
- Long-horizon behavior tracks spectral radius: `SR < 1` is generally bounded; `SR > 1` is generally unstable.
- The canonical **research-paper benchmark** is now locked to `target_size=256`, `sequence_length=8`, `batch_size=256`, `num_steps=50000`, `seeds={0,1,2}` across `29` systems and `4` baselines (`generic_sparse`, dense LISTA, diagonal-K LISTA, block-diagonal-K LISTA).
- The paper benchmark infrastructure is now reproducible end-to-end: per-system `dt` overrides are supported in config/training, mixed-root forecasting collection records `env_dt`, and the `generic_sparse` anchor can drive `dt` rescue using the `H1000` every-step per-dim gate.
- The first fully collected paper-benchmark matrix (`v3`) finished `347/348` rows at default `dt`, and it remains a useful provisional default-`dt` audit only: `generic_sparse` is strongest at `H1000` (`0.0251` vs `0.0648` dense LISTA, `0.1619` block LISTA, `1.1417` diagonal-K LISTA).
- The corrected paper-benchmark rerun (`v4`) is now complete under `results/paper_benchmark_20260307_paper_final_ts256_50k_v4`; rescue passes `1/2`, the final `348`-task full matrix, final collection, and final comparison all finished overnight on March 8, 2026.
- On the completed `v4` canonical matrix, `generic_sparse` remains best by cross-system median `H1000` best-periodic (`0.0328`), dense LISTA is now the strongest LISTA-family competitor (`0.0388`), block-diagonal LISTA is third (`0.1508`), and diagonal-K LISTA remains worst (`1.2110`).
- The completed `v4` comparisons narrow the cross-system LISTA gap without overturning the ranking:
  - dense LISTA wins `15/29` systems vs `generic_sparse`
  - `generic_sparse` still wins `26/29` vs `lista_blockdiag`
  - `generic_sparse` still wins `26/29` vs `lista_diagonal`
- The architecture-fixed dense-LISTA easy-system parity Stage 1 is now complete under `results/dense_lista_easy_parity_stage1_20260308`:
  - target systems: `blended`, `competitive_lv`, `duffing`, `dysts:Dadras`, `dysts:Hadley`, `dysts:LuChenCheng`, `dysts:SanUmSrisuchinwong`, `multiwell_gradient`
  - same dense LISTA architecture and benchmark-selected `dt` on every system
  - Stage 1 changed only external knobs: `num_steps in {50k,100k,200k}` and `(lr, k_matrix_lr) in {(1e-4,1e-5),(3e-4,3e-5),(5e-5,5e-6)}`
  - all comparisons stayed anchored to the fixed `generic_sparse` `v4` runs
  - best win-count recipe: `lista_dense_ns100k_lr5em5_klr5em6_wd1em4` wins `6/8` target systems with median dense/generic ratio `0.8699`
  - best median-ratio recipe: `lista_dense_ns200k_lr5em5_klr5em6_wd1em4` wins `5/8` target systems with median ratio `0.7888`
  - holdouts: `competitive_lv` and `duffing` remain `generic_sparse`-better under all Stage-1 recipes
- The coefficient-only dense-LISTA easy-system parity Stage 2 is now complete under `results/dense_lista_easy_parity_stage2_20260308`:
  - holdout systems only: `competitive_lv`, `duffing`
  - same dense LISTA architecture, same benchmark-selected `dt`, same low-LR winner recipes from Stage 1
  - base optimizer recipes:
    - `100k`, `lr=5e-5`, `k_matrix_lr=5e-6`, `weight_decay=1e-4`
    - `200k`, `lr=5e-5`, `k_matrix_lr=5e-6`, `weight_decay=1e-4`
  - coefficient-only one-axis sweep:
    - `sparsity_coeff in {0.003, 0.006, 0.012}`
    - `reconst_coeff in {0.01, 0.03, 0.1}`
    - `pred_coeff in {0.5, 1.0, 2.0}`
  - concrete result:
    - `duffing` is flipped by the Duffing-specialized recipe `lista_dense_ns100k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc1em2` with seed-median `H1000` best-periodic `0.0182` vs `generic_sparse=0.0309` (ratio `0.590`)
    - `competitive_lv` is not flipped by any Stage-2 coefficient-only recipe
    - the best global-compromise holdout recipe is `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`, which keeps both holdouts good and minimizes the remaining gap (`competitive_lv=0.0373` vs `0.0276`, `duffing=0.0332` vs `0.0187`)
  - interpretation:
    - coefficient-only tuning can repair `duffing` but does not recover `competitive_lv` under the fixed dense-LISTA architecture
    - the Duffing-fixing recipe is too specialized to serve as the fair dense-LISTA paper default
    - the real paper question after Stage 2 is recipe promotion, not more holdout-only tuning
- The dense-LISTA exact `8`-system recipe validation Stage 3 is now complete under `results/dense_lista_recipe_validation_stage3_20260309`:
  - exact recipes:
    - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
    - `lista_dense_ns100k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
    - `lista_dense_ns100k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc1em2`
  - concrete result:
    - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` is the clear promoted global recipe: it wins `6/8` shared systems vs `generic_sparse`, reaches median shared-system `H1000` best-periodic ratio `0.6928`, keeps `8/8` systems under the good-forecast band, and has `0` catastrophic runs with all seeds good on all `8` systems
    - the Duffing-fixing `100k, sc=0.012` recipe falls to `2/8` shared-system wins overall
    - the cheaper `100k, sc=0.003` recipe reaches `3/8` wins and is dominated by the `200k, sc=0.003` recipe on `H500/H1000`
  - interpretation:
    - a single fair dense-LISTA external recipe is now promotable without changing architecture or `dt`
    - dense LISTA still does not dominate `generic_sparse`, but it now has a publication-grade parity narrative on the targeted easy subset
  - project implication:
    - the dense thread is no longer blocked by recipe ambiguity; the full `29`-system rerun of the promoted `200k, sc=0.003` recipe is now queued under `results/dense_lista_paper_rerun_stage4_20260309`
- The repaired `dt` resolution is now complete through pass `2`: `15/29` systems accept default `dt`, `4/29` accept after at least one halving, and `10/29` remain `integration_hard`. The queueing blocker is gone, but the step-size/integration bottleneck remains real.
- The hardest intrinsic-HD systems are still the main blockers even at the selected smaller `dt=0.0125`:
  - `kuramoto`: `generic_sparse=65.70`, `lista_blockdiag=14.26`, dense LISTA=`35.23` (`H1000` system-median best-periodic)
  - `hopfield`: `generic_sparse=199.50`, `lista_blockdiag=280.42`, dense LISTA=`7.241e+09`
- For the current LISTA-family recovery phase (`lista_dense`, `lista_blockdiag`, HyperLISTA), the primary optimization target is long-horizon forecasting robustness; sparsity/support calibration is a secondary diagnostic until forecasting improves.
- Dense LISTA is now the strongest cross-system LISTA baseline, but `lista_blockdiag` remains the only LISTA-family candidate worth carrying forward on the hardest intrinsic-HD systems.
- On `duffing`, `L=8` training is superior for both `generic_sparse` and LISTA given that the model trains long enough. Use at least 20000 steps for training to draw definitive conclusions.
- On `duffing`, controlled 50k training with coefficient-matched arms (Experiment K) did **not** close the LISTA-vs-generic_sparse gap; `generic_sparse` remains best across all seeds and reported horizons.
- The intrinsic-HD `L=8`, `TARGET_SIZE=256` baseline with current env defaults (`kuramoto=16`, `hopfield=16`, `competitive_lv=10`) is now complete for `generic_sparse`, dense LISTA, and block-diagonal LISTA:
  - `competitive_lv` is solved by all three models; seed-median `H1000` best-periodic is `0.0651` (`generic_sparse`), `0.1192` (`lista_blockdiag`), `0.1654` (`lista_dense`).
  - `kuramoto` favors `generic_sparse` (`199.86`) over `lista_blockdiag` (`258.50`); dense LISTA is catastrophic (`6.636e8` median `H1000` best-periodic).
  - `hopfield` is the current blocker: `generic_sparse` is least bad (`5436.67` median `H1000` best-periodic), while `lista_blockdiag` (`3.599e15`) and dense LISTA (`3.045e33`, with one seed lacking a finite best-periodic score) are catastrophic.
- The dedicated current-default Kuramoto recovery pilot is now complete, and the summary artifacts have been regenerated with the repaired sweep collection (`36` rows, `H100/H500/H1000`). The best block-diagonal LISTA arm (`sp_0p0010`, `loops_1`) materially improves Kuramoto long-horizon error over matched `generic_sparse` (`H100/H500/H1000` seed-median best-periodic `5.47/24.83/48.64` vs `6.24/48.30/136.60` for `generic_sparse sp_0p0005`).
- Despite that improvement, Kuramoto is still not solved under the good-forecast band (`H1000 < 10`), and several block-diagonal arms still show catastrophic seed outliers.
- The repaired focused intrinsic-HD `dt` rescue rerun is now complete under `results/intrinsic_hd_dt_rescue_20260308_rerun1` with all `48` rows collected.
- Smaller `dt=0.0125` materially improves both intrinsic-HD systems at `20k`, but only Kuramoto approaches the good-forecast band:
  - `kuramoto` best arm: `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`, seed-median `H1000` best-periodic `14.36`; matched `generic_sparse` best arm is `25.93`.
  - `hopfield` best arm: `generic_sparse`, `dt=0.0125`, `sp=0.0005`, seed-median `H1000` best-periodic `71.02`; best `lista_blockdiag` arm is `80.54`.
- The narrow Kuramoto-only `200k` continuation is now complete under `results/intrinsic_hd_kuramoto_dt00125_200k`:
  - `generic_sparse`, `dt=0.0125`, `sp=0.0005`: `H1000=26.36`
  - `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`: `H1000=13.77`
  - longer training helps the block-diagonal arm modestly (`14.36 -> 13.77`) but still does not cross the good-forecast band (`< 10`)
- The focused Kuramoto `dt=0.00625`, `200k`, `5`-seed three-way comparison is now complete under `results/kuramoto_dt00625_200k_compare_20260308`:
  - `generic_sparse`: seed-median `H1000` best-periodic `27.02`
  - dense LISTA: seed-median `H1000` best-periodic `13.84`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.98`
  - all five `lista_blockdiag` seeds are good and tightly clustered at `{6.89, 6.97, 6.98, 7.04, 7.13}`
  - interpretation:
    - smaller `dt=0.00625` plus longer training materially rescues Kuramoto long-horizon forecasting
    - `lista_blockdiag` is the only model that cleanly enters the good-forecast band in the direct `N=16` three-way comparison
    - dense LISTA improves a lot over `generic_sparse` on Kuramoto but still misses the good band
- The Hopfield `dt=0.00625`, `200k` boundary-case follow-up is now complete under `results/hopfield_dt00625_200k_compare_20260309`:
  - `generic_sparse`: seed-median `H1000` best-periodic `3.36`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `8.82`
  - both models are now inside the good-forecast band on the system median, but every-step rollouts remain extremely unstable for both and `generic_sparse` is still clearly better
  - interpretation:
    - smaller `dt` plus longer training removes Hopfield from the catastrophic regime in this milder `N=16` evaluation
    - Hopfield remains a limitation for autonomous rollouts and for any claim that structured LISTA beats the MLP anchor there
- The Kuramoto `N=32`, `dt=0.00625`, `200k` confirmation is now complete under `results/kuramoto_n32_dt00625_200k_confirm_20260309`:
  - `generic_sparse`: seed-median `H1000` best-periodic `6.65`
  - `lista_blockdiag`: seed-median `H1000` best-periodic `6.00`
  - both models stay inside the good-forecast band, with a small block-diagonal edge on the system median
  - interpretation:
    - the stronger Kuramoto result survives a stricter `N=32` confirmation
    - the hard-system story is now more about step size and periodic reencoding than about a universal architecture win
- A dedicated Kuramoto dimension sweep is now queued under `results/kuramoto_dimension_sweep_dt00625_200k_20260309`:
  - dimensions: `N in {8,16,24,32,64}`
  - models: `generic_sparse`, promoted dense LISTA, `lista_blockdiag`
  - fixed setting: `dt=0.00625`, `num_steps=200000`, `seeds={0,1,2,3,4}`
  - purpose:
    - test whether the smaller-`dt`, longer-training Kuramoto rescue actually scales with dimension
    - measure whether the promoted dense-LISTA recipe transfers to the oscillator setting as dimension grows
- A diagnostic recollection from `evaluation_results_last.json` shows checkpoint-selection misalignment on Kuramoto:
  - for `lista_blockdiag`, `dt=0.0125`, system-median `H1000` best-periodic improves from `23.40` to `14.64` when switching from the validation-selected checkpoint to the last checkpoint across the focused pilot grid
  - on the winning `lista_blockdiag`, `dt=0.0125`, `sp=0.0005` arm, the last-checkpoint median is `13.91`
  - this is a diagnostic, not yet the official paper metric, but it shows late training can still help on Kuramoto
- Queue `0–1` on `duffing` is complete under ReLU-final LISTA; `lista_alpha=0.15` is the best long-horizon alpha among tested values.
- Queue `2` (`SPARSITY_COEFF` sweep at fixed `lista_alpha=0.15`) is complete:
  - `sp_0p0005`: sparsity `0.7708`, `H500=0.0191`, `H1000=0.1324` (best in-band candidate, closest to target `~0.8`).
  - `sp_0p0010`: sparsity `0.8744`, `H500=0.0217`, `H1000=0.1938` (in-band but weaker long horizon).
  - `sp_0p0020/0p0040/0p0060`: stronger sparsity pressure (`~0.92–0.95`) outside target band.
- Independent QA pass confirms Queue-2 summary rows match raw per-run artifacts (`15/15`).
- Queue `3` (`lista_num_loops` sweep at fixed `alpha=0.15`, `sp=0.0005`) is complete:
  - best forecast arm is `loops_1` (`quick=0.4164`, `H500=0.0114`, `H1000=0.0889`) but under-sparse (`0.6644`, below target band).
  - `loops_7` preserves target sparsity (`0.8040`) but with weaker forecast means (`H1000=0.1317`).
  - `loops_3` shows catastrophic instability (`H1000 mean=643.10`) from one severe outlier seed.
- Queue `4` (joint `SPARSITY_COEFF × lista_num_loops` Pareto sweep) is complete:
  - In-band Pareto frontier includes:
    - `sp_0p0060_loops_1`: quick `0.3162`, `H500=0.0119`, `H1000=0.0919`, sparsity `0.8490`
    - `sp_0p0040_loops_1`: quick `0.3108`, `H500=0.0131`, `H1000=0.0996`, sparsity `0.8190`
    - `sp_0p0005_loops_7`: quick `0.5802`, `H500=0.0351`, `H1000=0.1890`, sparsity `0.8043`
  - Best in-band long horizon is `sp_0p0060_loops_1`; best in-band quick-best is `sp_0p0040_loops_1`.
  - Several arms remain heavy-tailed/unstable (`sp_0p0010_loops_7`, `sp_0p0060_loops_7`, `sp_0p0005_loops_5`).
- Relative to the `generic_sparse` anchor (`quick=0.1115`, `H500=0.00309`, `H1000=0.0294`), the best Queue-4 in-band arm (`sp_0p0060_loops_1`) is still worse by roughly `2.84x` (quick), `3.85x` (`H500`), and `3.12x` (`H1000`).
- Queue `5` adaptive-threshold intervention (HyperLISTA) is complete:
  - launcher `8876267`, sweep `8876268_[0-23]`, and collector `8876269` all completed successfully.
  - best Queue-5 long-horizon arm is `sp_0p0040_loops_1_ct_0p0040`: `quick=0.7751`, `H500=1.5481`, `H1000=2.2958`, `sparsity=0.00010`.
  - compared with Queue-4 anchor `sp_0p0060_loops_1`, this best Queue-5 arm is worse by about `2.45x` (quick), `130x` (`H500`), and `24.98x` (`H1000`).
  - compared with `generic_sparse`, it is worse by about `6.95x` (quick), `501x` (`H500`), and `78.09x` (`H1000`).
- Queue `5b` HyperLISTA stabilization smoke is complete:
  - launcher `8883005`, sweep `8883008_[0-23]`, and collector `8883009` all completed successfully with `use_ss=false`, `use_momentum=false`, and `constrain_c_theta=true`.
  - best smoke long-horizon arm is `sp_0p0060_loops_1_ct_0p0040`: `quick=0.9535`, `H100=0.3372`, `H500=52.3228`, `H1000=4223.8420`, `sparsity=0.00166`.
  - the stabilization patch repaired the code path, but the smoke run remained far behind both Queue-4 LISTA anchors and `generic_sparse`; increasing `c_theta` above `0.0040` monotonically worsened long-horizon error in this smoke grid.
- Cross-system LISTA robustness improved materially for dense LISTA on the completed `v4` matrix, but the hardest intrinsic-HD systems still break badly and block-diagonal LISTA still has heavy tails on several systems.
- LQR decision metrics remain non-discriminative due metric saturation (`M2`, `M3`) and heavy-tailed `M4`.

Current approach:
- Treat the completed canonical paper-benchmark rerun (`v4`) as the primary evaluation program and current paper evidence base, not the earlier subset sweeps.
- Keep `L=8` sequence training as the default training mode for long-horizon forecasting experiments.
- Use `generic_sparse` as the overall performance anchor while LISTA architecture/capacity changes are evaluated.
- Use dense LISTA as the cross-system LISTA reference, but keep `lista_blockdiag` as the only LISTA-family candidate for intrinsic-HD follow-up unless new evidence clearly overturns that ranking.
- Use the current-default intrinsic-HD sweep as a baseline only; do not yet treat it as the final stress test for the plan because only the `N=32` Kuramoto confirmation has results in hand and the broader `N={8,16,24,32,64}` scaling sweep is still running.
- For intrinsic-HD follow-up, prioritize autonomous long-horizon stability on `hopfield` and robustness on `kuramoto`; `competitive_lv` is not the bottleneck at `TARGET_SIZE=256`.
- Treat `results/intrinsic_hd_dt_rescue_20260308_rerun1` as the decision-grade focused intrinsic-HD source of truth for the current-default `N=16` setting.
- Use the best-checkpoint collection as the official paper-facing metric for now, and use `evaluation_results_last.json` only as a diagnostic for checkpoint-selection mismatch.
- Replace the old current-default Kuramoto LISTA anchor with the completed smaller-`dt`, longer-training winner: carry forward `lista_blockdiag`, `dt=0.00625`, `num_steps=200000` as the Kuramoto intrinsic-HD anchor, with the `N=32` confirmation as the stricter supporting result.
- Use the queued Kuramoto dimension sweep as the next decision point for the hard-system narrative:
  - if `lista_blockdiag` stays in-band through `N=64`, the paper can claim a real oscillator-scaling rescue under smaller `dt`
  - if the rescue degrades sharply with `N`, position the result as a smaller-`dt`, moderate-dimension success with explicit scaling limits
  - if promoted dense LISTA closes the gap at some `N`, keep that as a secondary transfer result rather than the primary hard-system claim
- Carry forward `generic_sparse`, `dt=0.00625`, `num_steps=200000` as the Hopfield intrinsic-HD anchor; keep `lista_blockdiag` as the structured-encoder comparator there, but not as the preferred model.
- Treat the completed Kuramoto-only `200k` continuation as a modest positive result rather than a full rescue:
  - `lista_blockdiag` improves from `H1000=14.36` to `13.77`
  - matched `generic_sparse` does not improve over the `20k` rerun
  - Kuramoto remains above the good-forecast band, so the paper narrative should not claim that longer training alone solves the intrinsic-HD oscillator case
- Use the completed architecture-fixed dense-LISTA easy-system Stage-1 sweep in [docs/planning/dense_lista_easy_system_parity_plan_20260308.md](/home/mila/l/lia/skae/docs/planning/dense_lista_easy_system_parity_plan_20260308.md) as the cross-system parity reference point:
  - Stage 1 shows that external optimization alone can flip a majority of the targeted easy near-misses without changing architecture or `dt`
  - carry forward the low-LR longer-training winners (`100k, 5e-5/5e-6` and `200k, 5e-5/5e-6`) as the only sensible Stage-2 anchors
  - use the completed Stage-2 / Stage-3 chain as the current decision point:
    - `duffing` can be fixed by a specialized coefficient recipe, but `competitive_lv` remains MLP-favored
    - the promoted fair dense recipe is `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
    - if more benchmark budget is approved, spend it on the full `29`-system rerun of that one promoted recipe
- Treat the old `50k` coefficient-matching result on `duffing` as superseded by the completed low-LR Stage-2 / Stage-3 read: coefficient balance does matter once training length and optimizer scale are in the right regime, but the effect is recipe-specific rather than a global dense-LISTA win.
- Enforce **ReLU as the final operation** in LISTA going forward, and rank LISTA-family models by long-horizon forecasting first; track `sparsity_ratio` / support structure as diagnostics rather than promotion gates for now (`docs/LISTA_STACK_FORECASTING_PLAN.md`).
- Keep `lista_alpha=0.15` fixed and treat Queue-2/3 outcomes as anchors for joint tuning.
- Promote Queue-4 winners as forecasting-first LISTA anchors:
  - robustness-first anchor: `sp_0p0060_loops_1`
  - quick-best tie-break anchor: `sp_0p0040_loops_1`
- Treat Queue-5 HyperLISTA adaptive-threshold results as a negative result for the tested `c_theta` grid; do not promote Queue-5 arms.
- Treat HyperLISTA stabilization as a validated code-path repair: keep constrained positive `c_theta` and safe `pinv(D)` recomputation, and only rank future HyperLISTA runs by long-horizon forecasting under those safe settings.
- Use Queue-4 instability flags to exclude heavy-tail arms from further shortlist.
- Validate current `duffing` conclusions on additional systems/seeds before promoting defaults.
- Prioritize forecasting robustness first across all LISTA-family models; only then advance basin-support deployment checks and control-facing stages.
- For intrinsic-HD blockers, treat smaller-`dt` rescue as the next cheapest honest test before broader representation changes or `10x` longer training; if the benchmark-selected smaller `dt` still leaves a system `integration_hard`, promote targeted representation or optimization changes rather than assuming more blind halving or more steps will be enough.
- For the current intrinsic-HD follow-up, use the completed Hopfield `dt=0.00625`, `200k` result and the completed Kuramoto `N=32` confirmation as the decision-grade evidence for the paper narrative.

Outstanding problem:
- The wrap-up problem is no longer broad exploration; it is to close, or clearly delimit, the remaining paper blockers identified in `docs/PAPER_TRACK_STATUS.md` so the final NeurIPS-facing story is crisp and defensible.
- The canonical `v4` paper benchmark is now complete, but `10/29` systems remain `integration_hard` after two rescue passes and the hardest intrinsic-HD systems (`kuramoto`, `hopfield`) still sit well above the good-forecast band.
- The repaired focused intrinsic-HD rerun and the completed smaller-`dt`, longer-training follow-ups are now jointly interpretable: `kuramoto` is rescued under `dt=0.00625`, `200k`, and `hopfield` is no longer catastrophic on the same lever at `N=16`, but every-step autonomous rollouts remain unstable and `generic_sparse` still wins Hopfield.
- Checkpoint selection is now part of the scientific problem: on Kuramoto, the validation-selected checkpoint can be materially worse than the last checkpoint for the paper metric, so the model-selection narrative must be explicit.
- The remaining Kuramoto paper question is now a scaling question, not a one-off rescue question: `N=16` and `N=32` are positive, but the full `N={8,16,24,32,64}` sweep is still in flight and the promoted dense-LISTA transfer story is still unmeasured beyond `N=16`.
- The dense-LISTA easy-system parity story is now narrower and cleaner after the completed Stage-2/Stage-3 chain:
  - `duffing` is flipped by coefficient-only tuning under the fixed architecture, but only by a specialized recipe
  - `competitive_lv` remains the only accepted-default easy-system holdout that is still clearly `generic_sparse`-better
  - the fairness question is no longer recipe selection among three arms; the promoted global dense recipe is now `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`


## Outstanding problems (active)

These are the active blockers for the paper-track plan in `docs/PAPER_TRACK_STATUS.md`.

- The completed `v4` paper-benchmark matrix is now the canonical paper audit, but `10/29` systems remain `integration_hard` after two rescue passes (`lotka_volterra`, `multiwell_strong_transition`, `multiwell_gradient_hd`, `multiwell_rotational_hd`, `multiwell_strong_transition_hd`, `kuramoto`, `hopfield`, `dysts:DequanLi`, `dysts:WangSun`, `dysts:LorenzCoupled`).
- **LISTA-vs-generic_sparse quality gap persists after controlled matching**: at 50k steps with matched coefficients, LISTA remains substantially worse than `generic_sparse` on `duffing` (quick-best and long horizons), indicating the main gap is not resolved by loss coefficient matching alone.
- Queue `2` identified better LISTA operating points, but forecasting remains clearly behind the `generic_sparse` anchor.
- Queue `3` improved forecast means at low loop counts (`loops_1`), but the best settings still trail `generic_sparse` and seed robustness is uneven.
- Queue `4` identified in-band non-dominated settings, but the best one (`sp_0p0060_loops_1`) is still ~`3x` behind `generic_sparse` on long horizons.
- Queue `4` exposed unstable heavy-tail regimes for some arms; robust candidate filtering is still needed before promotion.
- Queue `5` adaptive-threshold sweep failed the primary objective in this regime: long-horizon errors stayed far above Queue-4 anchors.
- Queue `5b` smoke validated the safe HyperLISTA code path, but its best aggregate arm still has very poor long-horizon error (`H1000 ~ 4.2e3`), so HyperLISTA remains uncompetitive on forecasting.
- Unstructured LISTA parity is validated only on `duffing`; generalization to other systems/seeds is unverified.
- Dense LISTA can now close most of the targeted easy accepted-default near-misses by changing only external optimization knobs, and the official Stage-2/Stage-3 read narrows the remaining issue substantially:
  - Stage 1 flips `6/8` systems at best with no catastrophic runs
  - Stage 2 shows `duffing` can be flipped but `competitive_lv` does not close under coefficient-only tuning
  - Stage 3 promotes a single global recipe, `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`, which wins `6/8` shared systems with `0` catastrophic runs
- The next dense-LISTA paper decision is no longer recipe selection; it is whether to spend the remaining benchmark budget on a full `29`-system rerun of the promoted `200k, sc=0.003` recipe.
- Intrinsic-HD `hopfield` is no longer catastrophic under the targeted `dt=0.00625`, `200k` follow-up, but it remains unresolved as a structured-LISTA success story because `generic_sparse` is still better and every-step autonomous errors remain enormous.
- Dense LISTA is the strongest cross-system LISTA variant on the completed `v4` benchmark, but it is still not viable on the hardest intrinsic-HD systems because `kuramoto` and especially `hopfield` remain badly unstable.
- Current-default Kuramoto recovery is only partial: block-diagonal LISTA can beat matched `generic_sparse` on `H1000`, but the best arm is still `48.64` and several other settings retain catastrophic tails.
- The repaired intrinsic-HD `dt` rerun plus the completed `dt=0.00625`, `200k` follow-ups show that smaller `dt` helps materially on both `kuramoto` and `hopfield`, and that long-horizon periodic-reencoding forecasts can enter the good band on both systems under the milder targeted setting.
- Kuramoto dimension scaling is now the main unresolved hard-system question: the direct `N=16` comparison and the stricter `N=32` confirmation are positive, but we do not yet know whether the smaller-`dt` rescue and the block-diagonal edge persist through `N=64`, or whether promoted dense LISTA transfers beyond the original `N=16` case.
- Current focused-pilot best arms are still unsatisfactory for a final paper claim:
  - `kuramoto`: `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`, `H1000=14.36`
  - `hopfield`: `generic_sparse`, `dt=0.0125`, `sp=0.0005`, `H1000=71.02`
- The completed Kuramoto `200k` continuation improves the winning block-diagonal arm only modestly (`H1000=13.77`) and still leaves one clear long-horizon outlier seed; longer training alone is therefore not enough for a clean intrinsic-HD rescue claim.
- Checkpoint-selection mismatch is now a live issue on Kuramoto: at least one strong seed has a much better last-checkpoint `H1000` than its validation-selected checkpoint.
- Cross-system LISTA performance improved materially for dense LISTA on the completed `v4` matrix, but that improvement has not translated into reliable intrinsic-HD transfer and block-diagonal LISTA still has catastrophic long-horizon tails on some systems.
- Best-period collapse toward short reencoding periods indicates weak autonomous rollout stability.
- Label-free regime assignment from supports is not yet reliable enough for deployment-time control.
- Non-diagonal sequence training remains difficult to keep spectrally stable at larger capacities without explicit spectral constraints.
- Only part of the stricter intrinsic-HD scaling plan has produced results so far: `kuramoto N=32` now has a successful confirmation, the full `N={8,16,24,32,64}` Kuramoto sweep is running, and stricter Hopfield scaling remains untested.
- LQR decision metrics/rules are still weakly discriminative under saturation and heavy-tailed outcomes.

## Queue Status

Queue work should be justified against `docs/PAPER_TRACK_STATUS.md`; prefer runs that directly retire a paper blocker or sharpen the final paper narrative.

In progress:
- **Dense LISTA promoted full `29`-system rerun (`TS=256`, `L=8`, `200k`, promoted fair recipe, seeds `0,1,2`)**:
  - queue launcher: `8909900`
  - array: `8909900_[0-86]`
  - collector: `8909901`
  - comparison: `8909902`
  - output roots:
    - `/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309`
    - `results/dense_lista_paper_rerun_stage4_20260309`
  - recipe:
    - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
  - fairness constraints:
    - same dense LISTA architecture as the canonical benchmark
    - benchmark-selected `dt` from `results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/selected_dt.tsv`
    - fixed comparison against the existing `generic_sparse` `v4` anchor
  - rationale:
    - Stage 2 and Stage 3 resolved the dense recipe question
    - this is the next honest paper-grade spend for the dense thread
  - current cluster snapshot:
    - the array is broadly running across the cluster
    - collect / compare are waiting on dependency completion
- **Kuramoto dimension sweep (`dt=0.00625`, `200k`, `5` seeds, `N={8,16,24,32,64}`)**:
  - queue launcher / array: `8910056_[0-74]`
  - collector: `8910057`
  - comparison: `8910061`
  - output roots:
    - `/network/scratch/l/lia/skae/kuramoto_dimension_sweep_dt00625_200k_20260309`
    - `results/kuramoto_dimension_sweep_dt00625_200k_20260309`
  - models:
    - `generic_sparse`
    - promoted dense LISTA: `lista_dense_promoted` = `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
    - `lista_blockdiag`
  - purpose:
    - settle the Kuramoto scaling story with a paper-usable `N` sweep
    - test whether the promoted dense recipe transfers to Kuramoto as dimension grows
    - measure whether the `lista_blockdiag` smaller-`dt` rescue persists through `N=64`
  - current cluster snapshot:
    - the array is broadly running across the cluster
    - collect / compare are waiting on dependency completion

Most recent completed paper-track chains:
- **Dense LISTA easy-system parity Stage 2 (`competitive_lv` + `duffing`, coefficient-only, architecture fixed, `84` tasks)**:
  - Planning document:
    - [docs/planning/dense_lista_easy_system_parity_plan_20260308.md](/home/mila/l/lia/skae/docs/planning/dense_lista_easy_system_parity_plan_20260308.md)
  - Completed chain:
    - launcher: `8907833`
    - dense LISTA array: `8907833_[0-83]`
    - collector: `8907834`
    - comparison vs fixed `generic_sparse` anchor: `8907835`
  - Output roots:
    - `/network/scratch/l/lia/skae/dense_lista_easy_parity_stage2_20260308`
    - `results/dense_lista_easy_parity_stage2_20260308`
  - Concrete result:
    - `duffing` is flipped only by the specialized `100k, sc=0.012` recipe
    - `competitive_lv` is not flipped by any coefficient-only recipe
    - the best non-specialized holdout recipe is `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3`
  - Current interpretation:
    - Stage 2 finished the holdout question cleanly; more coefficient-only tuning is not the right next spend
- **Dense LISTA exact recipe validation (`8` target systems, `3` shortlisted recipes, architecture fixed)**:
  - Completed chain:
    - queue launcher: `8908839`
  - Output roots:
    - `/network/scratch/l/lia/skae/dense_lista_recipe_validation_stage3_20260309`
    - `results/dense_lista_recipe_validation_stage3_20260309`
  - Concrete result:
    - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` is the promoted dense-LISTA recipe (`6/8` wins, median shared-system ratio `0.6928`, `8/8` good systems, `0` catastrophic systems)
  - Current interpretation:
    - the dense thread is now promotion-ready; the remaining question is whether to rerun this one recipe on the full `29`-system benchmark
- **Focused Kuramoto `dt=0.00625`, `200k` comparison (`generic_sparse`, dense LISTA, block-diagonal LISTA; seeds `0,1,2,3,4`)**:
  - Completed chain:
    - queue launcher: `8907758`
    - array: `8907759_[0-14]`
    - collector: `8907760`
    - comparison vs `generic_sparse`: `8907761`
  - Output roots:
    - `/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308`
    - `results/kuramoto_dt00625_200k_compare_20260308`
  - Concrete result:
    - `generic_sparse=27.02`, dense LISTA=`13.84`, `lista_blockdiag=6.98` on seed-median `H1000` best-periodic
  - Current interpretation:
    - `lista_blockdiag` cleanly rescues Kuramoto at `N=16` under the smaller-`dt`, longer-training setting
- **Hopfield `dt=0.00625`, `200k` boundary-case follow-up (`generic_sparse` + `lista_blockdiag`, seeds `0,1,2`)**:
  - Completed chain:
    - queue launcher: `8908838`
  - Output roots:
    - `/network/scratch/l/lia/skae/hopfield_dt00625_200k_compare_20260309`
    - `results/hopfield_dt00625_200k_compare_20260309`
  - Concrete result:
    - `generic_sparse=3.36`, `lista_blockdiag=8.82` on seed-median `H1000` best-periodic
  - Current interpretation:
    - Hopfield is no longer catastrophic in this targeted setting, but `generic_sparse` remains the preferred model and autonomous every-step rollouts are still extremely unstable
- **Kuramoto `N=32`, `dt=0.00625`, `200k` confirmation (`generic_sparse` + `lista_blockdiag`, seeds `0,1,2`)**:
  - Completed chain:
    - queue launcher: `8908842`
  - Output roots:
    - `/network/scratch/l/lia/skae/kuramoto_n32_dt00625_200k_confirm_20260309`
    - `results/kuramoto_n32_dt00625_200k_confirm_20260309`
  - Concrete result:
    - `generic_sparse=6.65`, `lista_blockdiag=6.00` on seed-median `H1000` best-periodic
  - Current interpretation:
    - the positive Kuramoto result survives the stricter `N=32` confirmation, though the architectural gap narrows
- Recommended next queue if more paper-track budget is approved:
  - let the active dense full rerun and active Kuramoto dimension sweep finish before queuing more hard-system work
  - avoid more coefficient-only holdout sweeps

Completed in the last 24 hours:
- **Dense LISTA easy-system parity Stage 1 (`8` accepted-default systems, dense LISTA only, architecture fixed, `216` tasks)**:
  - Planning document:
    - [docs/planning/dense_lista_easy_system_parity_plan_20260308.md](/home/mila/l/lia/skae/docs/planning/dense_lista_easy_system_parity_plan_20260308.md)
  - Completion chain:
    - launcher: `8906725` (**completed**)
    - dense LISTA array (`216` tasks): `8906726_[0-215]` (**completed**)
    - collector: `8906727` (**completed**)
    - comparison vs fixed `generic_sparse` anchor: `8906728` (**completed**)
  - Output roots:
    - `/network/scratch/l/lia/skae/dense_lista_easy_parity_stage1_20260308`
    - `results/dense_lista_easy_parity_stage1_20260308`
  - Concrete result:
    - best win-count recipe: `lista_dense_ns100k_lr5em5_klr5em6_wd1em4` wins `6/8` target systems with median dense/generic ratio `0.8699`
    - best median-ratio recipe: `lista_dense_ns200k_lr5em5_klr5em6_wd1em4` wins `5/8` target systems with median ratio `0.7888`
    - all `9` dense-LISTA recipes keep `8/8` good systems with `0` catastrophic systems on the target set
    - best per-system dense recipe still loses on `competitive_lv` (`1.764x`) and `duffing` (`1.041x`)
  - Current queue interpretation:
    - external optimization alone is enough to flip a majority of the targeted easy dense-LISTA near-misses
    - the easy-system dense-LISTA gap is therefore not purely architectural
    - the result is still not strong enough to claim dense LISTA beats `generic_sparse` on most systems globally
- **Targeted Kuramoto `200k` continuation (`dt=0.0125`, `TS=256`, `L=8`, `sp=0.0005`, seeds `0,1,2`)**:
  - Queue launcher: `8906546` (**completed**)
  - `generic_sparse` sweep array (`3` tasks): `8906547_[0-2]` (**completed**)
  - `lista_blockdiag` sweep array (`3` tasks): `8906548_[0-2]` (**completed**)
  - Collector: `8906549` (**completed**)
  - Output roots:
    - `/network/scratch/l/lia/skae/intrinsic_hd_dt_rescue_20260308_kuramoto_200k`
    - `results/intrinsic_hd_kuramoto_dt00125_200k`
  - Concrete result:
    - `generic_sparse`, `dt=0.0125`, `sp=0.0005`: `H1000=26.36`
    - `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`: `H1000=13.77`
    - longer training helps the block-diagonal arm modestly relative to the repaired `20k` rerun (`14.36 -> 13.77`) but does not cross the good-forecast band
  - Current queue interpretation:
    - this is a modest positive optimization result for Kuramoto, not a full rescue
    - the next intrinsic-HD paper decision is now about narrative honesty, not whether the `200k` continuation still needs to finish
- **Canonical research-paper benchmark `v4` dt-rescue rerun (`29 systems x 4 baselines x 3 seeds`, `TS=256`, `L=8`, `50k`)**:
  - Output roots:
    - `/network/scratch/l/lia/skae/paper_benchmark_20260307_paper_final_ts256_50k_v4`
    - `results/paper_benchmark_20260307_paper_final_ts256_50k_v4`
  - Overnight completion chain:
    - rescue pass `1` array (`42` tasks): `8903420_[0-41]` (**completed**)
    - rescue pass `1` collector: `8903421` (**completed**)
    - resolver after pass `1`: `8903422` (**completed**)
    - rescue pass `2` array (`87` tasks): `8903423_[0-86]` (**completed**)
    - rescue pass `2` collector: `8903424` (**completed**)
    - resolver after pass `2`: `8903425` (**completed**)
    - full matrix array (`348` tasks): `8903426_[0-347]` (**completed**)
    - final collect: `8903427` (**completed**)
    - final compare: `8903428` (**completed**)
  - Current benchmark result:
    - `generic_sparse` is best by cross-system median `H1000` best-periodic (`0.0328`)
    - dense LISTA is the strongest LISTA-family competitor (`0.0388`)
    - pass-2 `dt` resolution leaves `10/29` systems `integration_hard`
  - Current queue interpretation:
    - the canonical paper benchmark chain is complete
    - the remaining work is analysis and hard-system follow-up, not more queue completion
- **Intrinsic-HD `dt` rescue pilot (`kuramoto` + `hopfield`, `L=8`, `TARGET_SIZE=256`, `20k`, `generic_sparse` + `lista_blockdiag`)**:
  - Queue launcher: `8903785` (**completed**)
  - `generic_sparse` sweep array (`24` tasks): `8903787_[0-23]` (**completed**)
  - `lista_blockdiag` sweep array (`24` tasks): `8903788_[0-23]` (**completed**)
  - Collector: `8903789` (**completed**)
  - Output roots:
    - `/network/scratch/l/lia/skae/intrinsic_hd_dt_rescue_20260308`
    - `results/intrinsic_hd_dt_rescue_20260308`
  - Launch-path audit:
    - the queue launcher computed the intended `0-23` arrays
    - the child sweep jobs saw `TOTAL_JOBS=1` because comma-separated grid values were passed through `sbatch --export`
    - only task `0` ran per model, which matches the two realized run directories and the two collected rows
  - Current queue interpretation:
    - the scheduler work finished, but the launch path collapsed the grid
    - the experiment is not decision-grade until the repaired launcher is rerun and recollected
- **Repaired intrinsic-HD `dt` rescue rerun (`kuramoto` + `hopfield`, `L=8`, `TARGET_SIZE=256`, `20k`, `generic_sparse` + `lista_blockdiag`)**:
  - Queue launcher: `8906425` (**completed**)
  - `generic_sparse` sweep array (`24` tasks): `8906426_[0-23]` (**completed**)
  - `lista_blockdiag` sweep array (`24` tasks): `8906427_[0-23]` (**completed**)
  - Collector: `8906428` (**completed**)
  - Output roots:
    - `/network/scratch/l/lia/skae/intrinsic_hd_dt_rescue_20260308_rerun1`
    - `results/intrinsic_hd_dt_rescue_20260308_rerun1`
  - Concrete result:
    - full `48`-row collection is complete
    - best Kuramoto arm at official best-checkpoint selection: `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`, `H1000=14.36`
    - best Hopfield arm at official best-checkpoint selection: `generic_sparse`, `dt=0.0125`, `sp=0.0005`, `H1000=71.02`
    - smaller `dt` materially improves both systems relative to `dt=0.025`, but neither is fully solved at `20k`
  - Diagnostic addendum:
    - a last-checkpoint recollection shows checkpoint-selection mismatch on Kuramoto; for `lista_blockdiag`, `dt=0.0125`, the system-median `H1000` improves from `23.40` to `14.64`
  - Current queue interpretation:
    - the launcher bug is fully resolved
    - the repaired rerun is now the correct targeted intrinsic-HD reference point
    - the next live question is whether a narrow `200k` Kuramoto continuation can push the winning arm below the good-forecast band

Completed:
- **Kuramoto recovery pilot (`L=8`, `TARGET_SIZE=256`, current defaults, `20k`, `generic_sparse` + `lista_blockdiag`)**:
  - Queue launcher: `8882966` (**completed**)
  - `generic_sparse` sweep array (`9` tasks): `8882967_[0-8]` (**completed**)
  - `lista_blockdiag` sweep array (`27` tasks): `8882968_[0-26]` (**completed**)
  - Collector: `8882969` (**completed**)
  - Output roots:
    - `/network/scratch/l/lia/skae/kuramoto_recovery_seq8_20260305`
    - `results/kuramoto_recovery_seq8_20260305`
  - Artifacts:
    - `results/kuramoto_recovery_seq8_20260305/forecasting_summary.{json,md}`
    - `results/kuramoto_recovery_seq8_20260305/forecasting_rows.{csv,json}`
  - Artifact note:
    - the summary artifacts were regenerated on `2026-03-08` with repaired sweep collection (`--select all`, `H100/H500/H1000`), so the standard results files now contain the full `36`-row sweep
  - Key queue result:
    - best block-diagonal arm is `sp_0p0010`, `loops_1`: seed-median `H100/H500/H1000` best-periodic `5.47 / 24.83 / 48.64`
    - best matched `generic_sparse` arm is `sp_0p0005`: seed-median `H100/H500/H1000` best-periodic `6.24 / 48.30 / 136.60`
    - interpretation: current-default block-diagonal retuning materially improves Kuramoto versus matched `generic_sparse`, but remains outside the good-forecast band and some other block-diagonal arms have catastrophic tails
- **Canonical research-paper benchmark `v3` default-`dt` pass (`29 systems x 4 baselines x 3 seeds`, `TS=256`, `L=8`, `50k`)**:
  - Launch/debug history:
    - first chain `8897520`-`8897533` failed due Slurm repo-root resolution
    - second chain `8897551`-`8897564` failed because shell TSV parsing corrupted empty fields / `env_dt`
    - final corrected chain `8897639`-`8897652` completed training/collection
  - Output roots:
    - `/network/scratch/l/lia/skae/paper_benchmark_20260306_paper_final_ts256_50k_v3`
    - `results/paper_benchmark_20260306_paper_final_ts256_50k_v3`
  - Post-hoc repair:
    - the original in-chain collector missed the `system/dt_tag/seed/run` layout, so final collection and comparison were rerun locally after fixing `tools/collect_forecasting_roots.py`
    - repaired final collection found `347` completed rows out of the expected `348`
    - repaired final comparison was rerun locally after collection was fixed
  - Provisional default-`dt` result:
    - `generic_sparse` is best by cross-system median `H1000` best-periodic (`0.0251`)
    - dense LISTA is second (`0.0648`) but has `4` catastrophic systems and one missing run
    - block LISTA is third (`0.1619`)
    - diagonal-K LISTA is worst (`1.1417`)
  - Important validity note:
    - after recomputing anchor pass `0` with the repaired collector, only `13/29` systems accept the default `dt`
    - `16/29` systems require at least one halving, so this completed `v3` matrix is **provisional** and must be rerun before being treated as the paper-final benchmark
- **Queue 5b (HyperLISTA stabilization smoke @ `L=8`, `10k`, 3 seeds, safe settings)**:
  - Launcher: `8883005` (**completed**)
  - Sweep array (`2 coeffs × 1 loop × 4 c_theta × 3 seeds = 24`): `8883008_[0-23]` (**completed**)
  - Collector: `8883009` (**completed**)
  - Output roots:
    - `/network/scratch/l/lia/skae/duffing_hyperlista_q05b_smoke_10k_20260305`
    - `results/duffing_hyperlista_q05b_smoke_10k_20260305`
  - Artifacts:
    - `results/duffing_hyperlista_q05b_smoke_10k_20260305/duffing_hyperlista_q05b_smoke_10k_summary.{json,md}`
    - `results/duffing_hyperlista_q05b_smoke_10k_20260305/duffing_hyperlista_q05b_smoke_10k_pareto_frontier.{json,md}`
  - Safe settings:
    - `use_ss=false`, `use_momentum=false`, `constrain_c_theta=true`, `eval_profile=smoke`
  - Key queue result:
    - Best aggregate long-horizon arm: `sp_0p0060_loops_1_ct_0p0040` (`quick=0.9535`, `H100=0.3372`, `H500=52.3228`, `H1000=4223.8420`)
    - Lower `c_theta=0.0040` is best in this smoke grid; higher `c_theta` worsens long-horizon forecasting.
- **Intrinsic-HD benchmark baseline (`L=8`, `TARGET_SIZE=256`, current env defaults, `3 systems × 3 seeds × 3 model variants`)**:
  - `generic_sparse` array: `8877775_[0-8]` (**completed**)
  - dense LISTA array: `8877776_[0-8]` (**completed**)
  - block-diagonal LISTA array: `8877777_[0-8]` (**completed**)
  - Output roots:
    - `/network/scratch/l/lia/skae/high_dim_benchmarks_plan_seq8_20260305`
    - `results/high_dim_benchmarks_plan_seq8_20260305`
  - Artifacts:
    - `results/high_dim_benchmarks_plan_seq8_20260305/forecasting_summary.{json,md}`
    - `results/high_dim_benchmarks_plan_seq8_20260305/system_medians_h1000.md`
  - Key queue result:
    - `competitive_lv`: all three models are stable at `H1000`; `generic_sparse` is best.
    - `kuramoto`: `generic_sparse` is best; `lista_blockdiag` is second-best but has one catastrophic seed; dense LISTA is catastrophic.
    - `hopfield`: all models fail; `generic_sparse` is least bad and both LISTA variants are catastrophically unstable.
- **Queue 5 (HyperLISTA adaptive-threshold sweep @ `L=8`, `50k`, 3 seeds)**:
  - Launcher: `8876267` (**completed**)
  - Sweep array (`2 coeffs × 1 loop × 4 c_theta × 3 seeds = 24`): `8876268_[0-23]` (**completed**)
  - Collector: `8876269` (**completed**)
  - Output roots:
    - `/network/scratch/l/lia/skae/duffing_hyperlista_q05_adaptive_50k_20260304`
    - `results/duffing_hyperlista_q05_adaptive_50k_20260304`
  - Artifacts:
    - `results/duffing_hyperlista_q05_adaptive_50k_20260304/duffing_hyperlista_q05_adaptive_50k_summary.{json,md}`
    - `results/duffing_hyperlista_q05_adaptive_50k_20260304/duffing_hyperlista_q05_adaptive_50k_pareto_frontier.{json,md}`
  - Key queue result:
    - No aggregate arm satisfies sparsity band `0.7–0.9` (all `~1e-4`).
    - Best Queue-5 long-horizon arm `sp_0p0040_loops_1_ct_0p0040` is still far behind Queue-4 anchors and `generic_sparse`.
- **Queue 4 (Pareto sweep: `SPARSITY_COEFF × lista_num_loops` @ `lista_alpha=0.15`, 50k, 3 seeds)**:
  - Sweep array (`5 coeffs × 4 loops × 3 seeds = 60`): `8875708_[0-59]` (**completed**)
  - Collector: `8875709` (**completed**)
  - Output roots:
    - `/network/scratch/l/lia/skae/duffing_lista_q04_pareto_50k_20260304`
    - `results/duffing_lista_q04_pareto_50k_20260304`
  - Artifacts:
    - `results/duffing_lista_q04_pareto_50k_20260304/duffing_lista_q04_pareto_50k_summary.{json,md}`
    - `results/duffing_lista_q04_pareto_50k_20260304/duffing_lista_q04_pareto_50k_pareto_frontier.{json,md}`
  - Key queue result:
    - Forecast-robust in-band anchor: `sp_0p0060_loops_1` (`quick=0.3162`, `H500=0.0119`, `H1000=0.0919`, sparsity `0.8490`)
    - Quick-favoring in-band anchor: `sp_0p0040_loops_1` (`quick=0.3108`, `H500=0.0131`, `H1000=0.0996`, sparsity `0.8190`)
- **Queue 3 (LISTA depth/capacity sweep @ `lista_alpha=0.15`, `sparsity_coeff=0.0005`, 50k, 3 seeds)**:
  - Sweep array: `8874672_[0-14]` (**completed**)
  - Collector: `8874673` (**completed**)
  - Summary artifacts:
    - `results/duffing_lista_q03_loops_50k_20260304/duffing_lista_q03_loops_50k_summary.json`
    - `results/duffing_lista_q03_loops_50k_20260304/duffing_lista_q03_loops_50k_summary.md`
  - Key result:
    - Forecast best: `loops_1` (`quick=0.4164`, `H500=0.0114`, `H1000=0.0889`) but under target sparsity (`0.6644`).
    - In-band candidate: `loops_7` (`sparsity=0.8040`) with weaker forecast (`H1000=0.1317`).
- **Queue 2 (SPARSITY_COEFF sweep @ `lista_alpha=0.15`, 50k, 3 seeds)**:
  - Sweep array: `8874221_[0-14]` (**completed**)
  - Collector: `8874222` (**completed**)
  - Summary artifacts:
    - `results/duffing_lista_q02_spcoeff_50k_20260304/duffing_lista_q02_spcoeff_50k_summary.json`
    - `results/duffing_lista_q02_spcoeff_50k_20260304/duffing_lista_q02_spcoeff_50k_summary.md`
  - Ranked result:
    - Winner for sparsity-band objective (`target ~0.8`): `sp_0p0005` (`sparsity=0.7708`, `H500=0.0191`, `H1000=0.1324`)
- **Queue 0 (post-change baseline, 50k, 3 seeds)**:
  - Sweep array: `8873286_[0-2]` (**completed**)
  - Scheduled collector: `8873287` (canceled due queue delay); collector command executed locally with identical args.
  - Summary artifacts:
    - `results/duffing_lista_q01_20260304/queue0_baseline/duffing_lista_relu_baseline_50k_summary.json`
    - `results/duffing_lista_q01_20260304/queue0_baseline/duffing_lista_relu_baseline_50k_summary.md`
- **Queue 1 (alpha gate/full, 3 seeds)**:
  - Initial (bugged) chain canceled: gate `8873288_[0-14]`, collector `8873289`, selector `8873290`, full launcher `8873291`.
  - Corrected gate chain completed: gate `8873328_[0-14]`, collector `8873329`.
  - Scheduled selector: `8873330` (canceled due queue delay); selector command executed locally with identical args.
  - Selected survivors: `0.15,0.10,0.20`.
  - Full launcher: `8873462` (**completed**), full sweep: `8873469_[0-8]` (**completed**).
  - Scheduled full collector: `8873470` (canceled due queue delay); collector command executed locally with identical args.
  - Summary artifacts:
    - `results/duffing_lista_q01_20260304/queue1_gate/duffing_lista_alpha_gate_10k_summary.json`
    - `results/duffing_lista_q01_20260304/queue1_gate/duffing_lista_alpha_gate_10k_summary.md`
    - `results/duffing_lista_q01_20260304/queue1_gate/selection/lista_alpha_survivors.csv`
    - `results/duffing_lista_q01_20260304/queue1_full/duffing_lista_alpha_full_50k_summary.json`
    - `results/duffing_lista_q01_20260304/queue1_full/duffing_lista_alpha_full_50k_summary.md`
- **Experiment K**: 50k encoder comparison (LISTA-current vs LISTA-matched vs generic_sparse), `L=8`, 3 seeds each. Sweep: `8866049_[0-8]`, collector: `8866050`.
- Consolidated outputs: `results/duffing_encoder_50k_20260303/duffing_encoder_50k_summary.{json,md}`.

Blocked:
- No scheduler blocker at the moment.
- Queue-1 launch bug (CSV env values split by `sbatch --export`) is resolved and validated by successful 15/15 corrected gate-task mapping.
- Queue-4 direct manual submission from interactive shell intermittently hit SLURM controller connectivity errors, but queue launcher `8875686` completed and successfully submitted the chain `8875708/8875709`, which has now completed.
- The intrinsic-HD `dt` rescue root cause is now known: `scripts/queue_intrinsic_hd_dt_rescue.sh` passed comma-separated grids through `sbatch --export`, so the child jobs saw singleton values and only task `0` ran per model.
- The dense parity queue launcher root-path bug is resolved: `scripts/queue_dense_lista_easy_parity_stage1.sh` now prefers `SLURM_SUBMIT_DIR`, matching the other queue scripts.
- The immediate paper-track blockers are scientific rather than scheduler-level:
  - which single fair dense-LISTA external recipe should be promoted beyond the targeted easy-system subset
  - whether the final two `lista_blockdiag` seeds keep the `dt=0.00625`, `200k` Kuramoto result clean enough to serve as the main positive intrinsic-HD follow-up
  - whether Hopfield must remain an explicit boundary-case limitation

Planned next:
- Promote `v4` `final_collect` / `final_compare` as the paper benchmark source of truth and stop citing the provisional `v3` matrix or anchor-only `v4` stage for paper-ranking claims.
- Do not queue further architecture-changing LISTA experiments while the architecture-fixed dense-LISTA parity story is still being resolved with external knobs alone.
- Keep Queue-5 as a completed negative control for the tested adaptive-threshold region.
- Re-evaluate LISTA-family candidates against `generic_sparse` using forecasting-first criteria (`H1000`, `H500`, `H100`, `quick-best`, robustness across seeds, best-period mode distribution); keep sparsity as a diagnostic column only.
- Separate LISTA follow-up into two tracks:
  - dense LISTA as the cross-system paper comparator
  - `lista_blockdiag` as the only LISTA-family intrinsic-HD candidate
- Let the queued dense-LISTA easy-system Stage-2 holdout sweep finish and use the formal collector only as confirmation of the live partial read:
  - `duffing` is already flipped in the live partial read
  - `competitive_lv` is not going to flip under this coefficient-only Stage-2 grid
- After the Stage-2 collector lands, stop coefficient-only holdout tuning and run a small `8`-system confirmation on exactly two single global dense-LISTA recipes:
  - safe Stage-1 anchor: `lista_dense_ns100k_lr5em5_klr5em6_wd1em4`
  - Duffing-fixing Stage-2 candidate: `lista_dense_ns100k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc1em2`
- Use that `8`-system confirmation to choose one fair dense-LISTA recipe for the full `29`-system rerun; do not add more `competitive_lv`-specific coefficient sweeps before that rerun.
- Use the completed repaired intrinsic-HD `dt` rerun as the current decision-grade targeted evidence for `kuramoto` / `hopfield`.
- Use the completed Kuramoto `200k` continuation as evidence that longer training helps Kuramoto only modestly under the current architecture and `dt`; do not claim that longer training alone solves the intrinsic-HD oscillator case.
- Replace the old current-default Kuramoto LISTA anchor with the repaired smaller-`dt` winner: `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`, `alpha=0.15`, `loops=1`, `block_16`.
- Use the completed `dt=0.00625`, `20k` Kuramoto diagnostic as the next honest Kuramoto setup:
  - `lista_blockdiag` is already below the good band there (`H1000=7.37`)
  - matched `generic_sparse` remains above threshold (`H1000=15.61`)
  - treat that setting as a more forgiving discretization analysis, not as a silent replacement for the main benchmark
- Run the focused `dt=0.00625`, `200k` Kuramoto comparison across `generic_sparse`, dense LISTA, and block-diagonal LISTA with seeds `0,1,2,3,4`:
  - use it to test whether the easier-discretization advantage for `lista_blockdiag` survives a stronger seed audit and a direct dense-LISTA comparison
  - if it does, this becomes the main positive Kuramoto follow-up result for the paper’s limitation-aware narrative
- If the final two `lista_blockdiag` seeds stay in band, do not keep halving `dt`; instead consider a single stricter intrinsic-HD scaling confirmation (`N=32`) at the winning `dt=0.00625`, `200k` setting.
- Do not queue the stricter intrinsic-HD scaling check (`N=32/64`) until the current `5`-seed `dt=0.00625`, `200k` collector confirms the live partial read.
- Make the paper narrative explicit that step size, checkpoint selection, and longer training help substantially on Kuramoto but still do not fully solve the intrinsic-HD oscillator regime, and keep Hopfield as the clearest boundary case.
- Use the validated HyperLISTA stabilization patch set for any follow-up, but only queue runs that explicitly target better long-horizon forecasting.
- Queue the stricter intrinsic-HD plan variants (`N=32/64`) once the current-default baseline and the repaired smaller-`dt` pilot are understood well enough to serve as references.

## Core Experiment Log (Most Informative)

### ZF) Late-Partial Audit: `duffing` Likely Flips, `competitive_lv` Likely Does Not, and `lista_blockdiag` Likely Solves Kuramoto at `dt=0.00625`
Timestamp: 2026-03-08
Status: **running, late partial read**

1. Concrete results:
- A live partial recollection from `results/dense_lista_easy_parity_stage2_20260308` shows:
  - `80/84` Stage-2 tasks complete, `4/84` still running, collector `8907834` still pending
  - `duffing` is flipped by `lista_dense_ns100k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc1em2` with seed-median `H1000` best-periodic `0.0182` vs `generic_sparse=0.0309`
  - `competitive_lv` is not flipped; the best incomplete arm is `lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3` with current two-seed median `0.0395` vs `generic_sparse=0.0276`, and the missing third seed cannot lower the final three-seed median below the anchor
- A live partial recollection from `results/kuramoto_dt00625_200k_compare_20260308` shows:
  - `13/15` training runs complete, `2/15` still running, both remaining runs are `lista_blockdiag`
  - `generic_sparse` (`5/5` seeds): `H1000=27.02`
  - dense LISTA (`5/5` seeds): `H1000=13.84`
  - `lista_blockdiag` (`3/5` seeds so far): `H1000=6.98`, with completed-seed values `{6.89, 6.98, 7.13}`

2. Context:
- Stage 2 was designed as the minimal coefficient-only follow-up after Stage 1 left `duffing` and `competitive_lv` as the only easy-system holdouts.
- The `dt=0.00625`, `200k` Kuramoto comparison was the direct test of whether smaller `dt` helps more than longer training alone and whether block-diagonal LISTA still wins after adding seeds and a dense-LISTA comparator.

3. Interpretation:
- `duffing` now looks coefficient-limited under the fixed dense-LISTA architecture, but `competitive_lv` still does not.
- The current Stage-2 grid has effectively answered the holdout question:
  - `duffing`: yes
  - `competitive_lv`: no
- On Kuramoto, smaller `dt=0.00625` is a stronger lever than the earlier `dt=0.0125`, `200k` continuation, and block-diagonal LISTA is now very likely to stay below the good-forecast band once the last two seeds finish.

4. Project implications:
- The dense-LISTA paper story should now move from “can coefficient-only tuning fix the holdouts?” to “which single fair global recipe should be promoted?”
- The Kuramoto paper story is now close to a clean limitation-aware positive result:
  - under a more forgiving discretization, `lista_blockdiag` appears to solve Kuramoto while dense LISTA and `generic_sparse` do not
  - Hopfield remains the real intrinsic-HD limitation

5. Next steps:
- Let `8907834/8907835` and `8907760/8907761` finish, and treat the formal collectors as confirmation of the current live read.
- Stop coefficient-only holdout exploration after Stage 2 and run a small `8`-system confirmation on the two single global dense-LISTA recipes already shortlisted in the summary above.
- If the final two Kuramoto `lista_blockdiag` seeds stay in band, promote `dt=0.00625`, `200k`, block-diagonal LISTA as the main positive Kuramoto follow-up and consider one stricter `N=32` confirmation instead of more `dt` halving.

### ZE) Dense LISTA Easy-System Parity Stage 2 Queued: Coefficient-Only Holdout Sweep on `competitive_lv` and `duffing`
Timestamp: 2026-03-08
Status: **running**

1. Concrete results:
- The coefficient-only dense-LISTA Stage 2 holdout sweep is now running under `results/dense_lista_easy_parity_stage2_20260308`:
  - array: `8907833_[0-83]`
  - collector: `8907834`
  - comparison: `8907835`
- Scope:
  - systems: `competitive_lv`, `duffing`
  - base optimizer recipes:
    - `100k`, `lr=5e-5`, `k_matrix_lr=5e-6`, `weight_decay=1e-4`
    - `200k`, `lr=5e-5`, `k_matrix_lr=5e-6`, `weight_decay=1e-4`
  - coefficient variants per base recipe:
    - baseline
    - `sparsity_coeff in {0.003, 0.012}`
    - `reconst_coeff in {0.01, 0.1}`
    - `pred_coeff in {0.5, 2.0}`
  - total jobs: `84`

2. Context:
- Stage 1 already answered the first fairness question positively: external optimization alone can flip `6/8` targeted easy near-misses.
- The only unresolved easy systems are now `competitive_lv` and `duffing`.
- This sweep keeps architecture and `dt` fixed, and changes only loss balance around the two low-LR Stage-1 winners.

3. Interpretation:
- This is the minimal next test before spending full-benchmark compute:
  - same architecture
  - same benchmark-selected task difficulty
  - same optimizer family that already worked best
  - only coefficient balance changes

4. Project implications:
- If either holdout flips here, the dense-LISTA paper story gets stronger without sacrificing fairness.
- If both holdouts stay MLP-better, we should stop trying to make the easy-system story perfect and move to a full-benchmark rerun of the best fair dense-LISTA recipe.

5. Next steps:
- Let `8907833_[0-83]` finish and collect under `results/dense_lista_easy_parity_stage2_20260308/collect`.
- Compare each candidate root against the fixed `generic_sparse` anchor.
- Decide whether to promote one dense-LISTA recipe to a full `29`-system rerun or to stop the easy-system tuning track.

### ZD) Dense LISTA Easy-System Parity Stage 1 Completed: External Optimization Alone Flips 6/8 Targeted Near-Misses
Timestamp: 2026-03-08
Status: **completed**

1. Concrete results:
- The architecture-fixed dense-LISTA easy-system Stage 1 finished and collected under `results/dense_lista_easy_parity_stage1_20260308`:
  - launcher: `8906725`
  - array: `8906726_[0-215]`
  - collector: `8906727`
  - comparison: `8906728`
- Best Stage-1 recipes at `H1000` system-median best-periodic against the fixed `generic_sparse` anchor:
  - best win-count recipe: `lista_dense_ns100k_lr5em5_klr5em6_wd1em4` wins `6/8` target systems with median dense/generic ratio `0.8699`
  - best median-ratio recipe: `lista_dense_ns200k_lr5em5_klr5em6_wd1em4` wins `5/8` target systems with median ratio `0.7888`
- Best per-system dense recipes beat `generic_sparse` on:
  - `blended` (`0.530x`)
  - `dysts:Dadras` (`0.660x`)
  - `dysts:Hadley` (`0.724x`)
  - `dysts:LuChenCheng` (`0.216x`)
  - `dysts:SanUmSrisuchinwong` (`0.436x`)
  - `multiwell_gradient` (`0.483x`)
- Holdouts:
  - `competitive_lv` (`1.764x`)
  - `duffing` (`1.041x`)
- All `9` dense-LISTA Stage-1 recipes keep `8/8` target systems under the good-forecast band with `0` catastrophic systems.

2. Context:
- Stage 1 was the clean fairness-constrained test of whether dense LISTA could close easy accepted-default near-misses without changing architecture or making the task easier.
- The sweep changed only `num_steps`, `lr`, and `k_matrix_lr`; it kept dense LISTA architecture, benchmark-selected `dt`, and loss coefficients fixed across all `8` target systems.

3. Interpretation:
- The answer is now yes for the targeted easy subset: external optimization alone is enough to flip a majority of these near-miss systems.
- The strongest recipe family is clearly the lower-learning-rate, longer-training corner of the grid.
- The answer is still not yes globally: `competitive_lv` and `duffing` remain MLP-better even after the full Stage-1 sweep.

4. Project implications:
- This materially strengthens the dense-LISTA paper narrative. The remaining cross-system gap is not purely architectural; part of it was optimization-limited.
- It is now defensible to say that dense LISTA can recover most easy accepted-default losses using architecture-fixed external tuning.
- It is still not defensible to claim that dense LISTA beats `generic_sparse` on most systems overall, because the result is limited to the targeted `8`-system subset and the winning recipe has not yet been validated on the full `29`-system benchmark.

5. Next steps:
- If we continue the easy-system parity campaign, run a coefficient-only Stage 2 around:
  - `lista_dense_ns100k_lr5em5_klr5em6_wd1em4`
  - `lista_dense_ns200k_lr5em5_klr5em6_wd1em4`
- Keep architecture and `dt` fixed.
- Prioritize `competitive_lv` and `duffing`, then rerun a single winning dense-LISTA external recipe on the full `29`-system benchmark before changing the paper-level parity claim.

### ZC) Focused Kuramoto `dt=0.00625`, `200k`, 5-Seed Comparison Running: Direct Three-Way Check of `generic_sparse` vs Dense LISTA vs Block-Diagonal LISTA
Timestamp: 2026-03-08
Status: **running**

1. Concrete results:
- A focused Kuramoto follow-up is now running under `results/kuramoto_dt00625_200k_compare_20260308`:
  - launcher: `8907758`
  - array: `8907759_[0-14]`
  - collector: `8907760`
  - comparison: `8907761`
- The run matrix is:
  - system: `kuramoto`
  - `env_dt=0.00625`
  - `num_steps=200000`
  - models: `generic_sparse`, `lista_dense`, `lista_blockdiag`
  - seeds: `0,1,2,3,4`

2. Context:
- The completed Kuramoto follow-ups now say:
  - `dt=0.0125`, `20k`: block-diagonal LISTA reaches `H1000=14.36`
  - `dt=0.0125`, `200k`: block-diagonal LISTA reaches `H1000=13.77`
  - `dt=0.00625`, `20k`: block-diagonal LISTA reaches `H1000=7.37`
- That makes `dt=0.00625` the strongest Kuramoto setup observed so far, but it has only been checked at `20k` and only against `generic_sparse`.

3. Interpretation:
- This is the correct next Kuramoto test if we want a stronger paper-facing result without changing architecture:
  - more seeds for robustness
  - direct dense-LISTA comparison instead of only generic-vs-blockdiag
  - longer training on the smaller-`dt` setup that already looked best

4. Project implications:
- If block-diagonal LISTA stays below the good band with `5` seeds at `200k`, then the paper can make a cleaner positive Kuramoto claim under an explicitly easier discretization setting.
- If dense LISTA closes the gap here, then the cross-system near-parity story may extend further into the intrinsic-HD oscillator regime than the current canonical benchmark suggests.

5. Next steps:
- Let `8907759_[0-14]` finish and collect under `results/kuramoto_dt00625_200k_compare_20260308/collect`.
- Compare block-diagonal LISTA not only against `generic_sparse`, but also against dense LISTA on the same smaller-`dt`, longer-training, higher-seed-budget setup.

### ZB) Kuramoto `200k` Continuation Completed: Longer Training Helps Modestly but Does Not Reach the Good-Forecast Band
Timestamp: 2026-03-08
Status: **completed**

1. Concrete results:
- The narrow Kuramoto-only `200k` continuation finished and collected under `results/intrinsic_hd_kuramoto_dt00125_200k`:
  - launcher: `8906546`
  - `generic_sparse` array: `8906547_[0-2]`
  - `lista_blockdiag` array: `8906548_[0-2]`
  - collector: `8906549`
- Official best-checkpoint `H1000` system-median best-periodic scores:
  - `generic_sparse`, `dt=0.0125`, `sp=0.0005`: `26.36`
  - `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`: `13.77`
- Relative to the repaired `20k` rerun:
  - `lista_blockdiag` improves from `14.36` to `13.77`
  - matched `generic_sparse` changes from `25.93` to `26.36`
- Supporting horizons:
  - `lista_blockdiag` reaches `H500=8.62` and `H100=0.60`
  - `generic_sparse` reaches `H500=11.13` and `H100=0.92`

2. Context:
- This was the narrowest fair “train longer” test after the repaired `20k` smaller-`dt` rerun showed Kuramoto was close enough to justify more optimization budget.
- The question was whether longer training alone, without changing architecture or `dt`, could push the winning smaller-`dt` block-diagonal arm below the good-forecast band.

3. Interpretation:
- Longer training helps Kuramoto, but only modestly.
- The gain is real for `lista_blockdiag`, especially at `H500`, but `H1000` remains above the paper-ready threshold (`13.77 > 10`).
- The result is also still not fully robust: one block-diagonal seed remains a clear long-horizon outlier (`H1000=28.60`).

4. Project implications:
- This is not a negative result, but it is not the publishable “longer training solved Kuramoto” story either.
- The defensible paper claim is now:
  - smaller `dt` changes the intrinsic-HD ranking
  - longer training gives additional but limited improvement
  - block-diagonal structure helps on Kuramoto, but the regime is still not fully solved

5. Next steps:
- Keep this `200k` result as the current Kuramoto follow-up reference point.
- Do not queue a broader “train longer everywhere” campaign based on this alone.
- Let the active dense-LISTA Stage-2 holdout sweep answer the separate cross-system parity follow-up while Hopfield remains the clearest unresolved intrinsic-HD boundary case.

### ZA) Dense LISTA Easy-System Parity Stage 1 Queued: External Optimization-Only Sweep on the 8 Accepted-Default Near-Miss Systems
Timestamp: 2026-03-08
Status: **completed; superseded by ZD**

1. Concrete results:
- The architecture-fixed dense-LISTA parity plan is now documented in:
  - [docs/planning/dense_lista_easy_system_parity_plan_20260308.md](/home/mila/l/lia/skae/docs/planning/dense_lista_easy_system_parity_plan_20260308.md)
- Stage-1 dense-LISTA sweep was queued under `results/dense_lista_easy_parity_stage1_20260308`:
  - launcher: `8906725`
  - array: `8906726_[0-215]`
  - collector: `8906727`
  - comparison: `8906728`
- The Stage-1 target systems are the accepted-default dense-LISTA near-misses from `v4`:
  - `blended`, `competitive_lv`, `duffing`, `dysts:Dadras`, `dysts:Hadley`, `dysts:LuChenCheng`, `dysts:SanUmSrisuchinwong`, `multiwell_gradient`
- The active arm grid is:
  - `num_steps in {50000,100000,200000}`
  - `(lr, k_matrix_lr) in {(1e-4,1e-5),(3e-4,3e-5),(5e-5,5e-6)}`
  - fixed coefficients: `res=1.0`, `reconst=0.03`, `pred=1.0`, `sparsity=0.006`, `weight_decay=1e-4`
- Operational note:
  - initial launcher `8906721` failed because the queue script resolved `ROOT_DIR` from the SLURM spool copy instead of `SLURM_SUBMIT_DIR`
  - a later duplicate retry submitted `8906756/8906757/8906758`, which was canceled after confirming the first fixed chain was already live

2. Context:
- On the canonical `v4` benchmark, dense LISTA is already close to `generic_sparse` overall (`15/29` wins, median paired ratio `0.9588`), but the remaining accepted-default losses are still important for the paper narrative.
- The goal here is not to change architecture or task difficulty. It is to test whether dense LISTA can close the easy-system gap using only fair external optimization changes.

3. Interpretation:
- This is the cleanest cross-system parity test available:
  - same architecture on every system
  - same benchmark-selected `dt`
  - same loss coefficients as the `v4` dense-LISTA baseline
  - only `num_steps`, `lr`, and `k_matrix_lr` change
- If Stage 1 materially improves several of these easy near-miss systems, then the remaining dense-LISTA gap is at least partly optimization-limited rather than purely representational.

4. Project implications:
- The paper can now make a stronger fairness claim:
  - we did not need to change architecture to test whether dense LISTA could improve
  - we isolated the first parity push to external knobs only
- This also keeps the narrative cleanly separated from the intrinsic-HD `dt` story:
  - easy-system parity work keeps `dt` fixed
  - hard intrinsic-HD work still uses smaller-`dt` diagnostics where numerics are the actual issue

5. Next steps:
- Let `8906726_[0-215]` progress to the first collected summary under `results/dense_lista_easy_parity_stage1_20260308/collect`.
- Rank the `9` Stage-1 dense-LISTA arms by target-set median `H1000` ratio vs the fixed `generic_sparse` anchor.
- Only queue Stage 2 coefficient sweeps if at least one Stage-1 arm materially improves the easy near-miss set without creating new catastrophic tails.

### Z) Repaired Intrinsic-HD DT Rescue Completed: Smaller-`dt` Changes the Intrinsic-HD Ranking, and a Narrow Kuramoto `200k` Continuation Is Now Justified
Timestamp: 2026-03-08
Status: **completed, follow-up queued**

1. Concrete results:
- The repaired focused intrinsic-HD rerun is complete under `results/intrinsic_hd_dt_rescue_20260308_rerun1` with all `48` rows collected.
- Official best-checkpoint ranking at `H1000`:
  - `kuramoto` best arm: `lista_blockdiag`, `dt=0.0125`, `sp=0.0005`, `H1000=14.36`
  - matched `generic_sparse` Kuramoto arm: `dt=0.0125`, `sp=0.0005`, `H1000=25.93`
  - `hopfield` best arm: `generic_sparse`, `dt=0.0125`, `sp=0.0005`, `H1000=71.02`
  - best `lista_blockdiag` Hopfield arm: `dt=0.0125`, `sp=0.0010`, `H1000=80.54`
- Smaller `dt=0.0125` beats `dt=0.025` for both systems in both model families.
- A diagnostic recollection from `evaluation_results_last.json` shows checkpoint-selection mismatch on Kuramoto:
  - `lista_blockdiag`, `dt=0.0125`, system-median `H1000` improves from `23.40` to `14.64` across the focused pilot grid
  - on the winning `sp=0.0005` Kuramoto arm, the last-checkpoint median is `13.91`
- Based on that result, the next follow-up has been queued:
  - launcher: `8906546`
  - `generic_sparse` array: `8906547_[0-2]`
  - `lista_blockdiag` array: `8906548_[0-2]`
  - collector: `8906549`

2. Context:
- This is the first decision-grade collection from the focused intrinsic-HD pilot after repairing the `sbatch --export` CSV bug.
- The scientific question was whether smaller `dt` alone could rescue the hardest current-default intrinsic-HD systems under the existing `L=8`, `TARGET_SIZE=256`, `N=16` setup.

3. Interpretation:
- Smaller `dt` matters substantially, but it is not a complete rescue by itself at `20k`.
- On Kuramoto, the intrinsic-HD ranking changes in favor of `lista_blockdiag`; this is now the only LISTA-family arm close enough to justify more optimization budget.
- On Hopfield, smaller `dt` helps but does not make the system competitive; the best current arm is still `generic_sparse` and it remains far above the good-forecast band.
- Checkpoint selection is now part of the scientific story: validation-selected checkpoints are not always the best long-horizon periodic-forecast checkpoints on Kuramoto.

4. Project implications:
- The paper narrative is now stronger and more defensible:
  - cross-system story: `generic_sparse` still wins overall and dense LISTA is the only serious global LISTA comparator
  - intrinsic-HD story: step size and checkpoint selection materially change conclusions, and structure helps specifically on oscillator-like Kuramoto
  - limitation story: Hopfield remains a real hard boundary case
- The next paper-risk item is no longer “fix the pilot”; it is “determine whether the near-threshold Kuramoto arm crosses into publishable territory with a narrow optimization push.”

5. Next steps:
- Monitor the queued Kuramoto-only `200k` continuation under `results/intrinsic_hd_kuramoto_dt00125_200k`.
- If the winning block-diagonal arm crosses `H1000 < 10`, use that result to justify an `N=32/64` scaling check on the same settings.
- If it stays above threshold, keep the step-size and checkpoint-selection results as a strong mechanistic narrative and frame Hopfield as the clearest unresolved intrinsic-HD limitation.

### Y) Repaired Intrinsic-HD DT Rescue Rerun: Full Grid Active, Early Partial Results Show Smaller-`dt` Gains but Not Yet a Full Rescue
Timestamp: 2026-03-08
Status: **running, early partial read only**

1. Concrete results:
- The repaired rerun launcher completed successfully:
  - launcher: `8906425`
  - `generic_sparse` array: `8906426_[0-23]`
  - `lista_blockdiag` array: `8906427_[0-23]`
  - collector: `8906428`
- The child sweep jobs now decode the intended full grid:
  - representative logs show `TOTAL_JOBS=24`
  - run directories have been created under `/network/scratch/l/lia/skae/intrinsic_hd_dt_rescue_20260308_rerun1`
- Early partial collection from completed rows (`10` rows so far) shows:
  - `generic_sparse`, `kuramoto`, `dt=0.0125`, `sp=0.0005`, `H1000` best-periodic:
    - seed `0`: `25.93`
    - seed `1`: `25.87`
    - seed `2`: `36.16`
  - `generic_sparse`, `hopfield`, `dt=0.0125`, `sp=0.0025`, `H1000` best-periodic:
    - seed `1`: `130.89`
    - seed `2`: `197.34`
- Relative to existing anchors:
  - `generic_sparse` Kuramoto at the paper-default sparsity in `v4` had system-median `H1000` best-periodic `65.70`
  - the current-default Kuramoto recovery at `dt=0.05` had `generic_sparse` seed-median `H1000` best-periodic `136.60`

2. Context:
- This rerun is the direct repair of the collapsed singleton pilot.
- The scientific question is whether smaller `dt` alone makes `kuramoto` / `hopfield` paper-worthy, or whether a longer-training or representation change is still needed.

3. Interpretation:
- The rerun is now scientifically valid at the launch level.
- Early generic-sparse results confirm the main numeric hypothesis: smaller `dt` materially improves both Kuramoto and Hopfield.
- The gains are not yet sufficient to claim a full rescue:
  - Kuramoto is improved to the `~26–36` range on the first completed `dt=0.0125`, `sp=0.0005` rows, still above the good-forecast band (`< 10`)
  - Hopfield remains much worse than acceptable on the first completed rows, even though it is far better than the old catastrophic baseline

4. Project implications:
- The paper narrative is strengthening around step size as a hidden first-order confound, not just around encoder ranking.
- The next experiment decision should be based on the completed repaired pilot:
  - if the best Kuramoto / Hopfield arms are still improving materially but stay above threshold, a targeted `200k` continuation is justified
  - if the best arms plateau far above threshold, the paper should frame Hopfield as a boundary case and prioritize a stronger mechanistic limitation story

5. Next steps:
- Let the repaired rerun finish and collect all rows under `results/intrinsic_hd_dt_rescue_20260308_rerun1`.
- Re-rank the best arms by `H100/H500/H1000` best-periodic and robustness across seeds.
- Use that completed ranking to decide whether the next queued run is:
  - a narrow `200k` continuation on the best smaller-`dt` arms
  - or the `N=32/64` scaling check on the winning intrinsic-HD settings
  - or a boundary-case follow-up centered on Hopfield
### X) Intrinsic-HD DT Rescue Pilot Root-Cause Audit: `sbatch --export` Collapsed the Grid to Singleton Child Jobs
Timestamp: 2026-03-08
Status: **completed, launcher repaired, rerun required**

1. Concrete results:
- The focused intrinsic-HD `dt` rescue launcher (`8903785`) computed the intended arrays correctly:
  - `generic_sparse`: `0-23`
  - `lista_blockdiag`: `0-23`
- The child sweep jobs did not receive the full grid. Representative task logs show:
  - queue log `queue-hd-dt-8903785.out`: `SYSTEMS_CSV=kuramoto,hopfield`, `ENV_DTS_CSV=0.025,0.0125`, arrays `0-23`
  - sweep task logs `hd-dt-rescue-8903787_1.out` and `hd-dt-rescue-8903788_1.out`: `Task 1 out of range for TOTAL_JOBS=1. Exiting.`
  - only task `0` ran per model (`8903787_0`, `8903788_0`), which matches the two collected rows in `results/intrinsic_hd_dt_rescue_20260308/forecasting_rows.csv`
- Root cause:
  - `scripts/queue_intrinsic_hd_dt_rescue.sh` passed comma-separated values like `SYSTEMS_CSV=kuramoto,hopfield` through `sbatch --export=...`
  - SLURM uses commas as separators in `--export`, so the child jobs saw only the first value of each CSV and recomputed `TOTAL_JOBS=1`
- The launcher script has been repaired to pass grid variables through the process environment and use `--export=ALL`, which preserves comma-containing values.

2. Context:
- This audit was needed because the collector itself looked suspicious, but the scratch root contained only two real training runs. That pointed to a launch-path failure rather than a collection-only bug.
- The pilot was intended to be the cheapest direct test of whether smaller `dt` helps the two hardest intrinsic-HD systems (`kuramoto`, `hopfield`) under the current seq8 / `TARGET_SIZE=256` setup.

3. Interpretation:
- The current two-row summary is not a partial sample of a finished grid; it is the exact output of a collapsed singleton launch.
- There is still no clean focused-pilot evidence on whether smaller `dt` alone resolves `kuramoto` or `hopfield`.

4. Project implications:
- The right next action is a narrow rerun of the same pilot with the repaired launcher, not more collector work.
- Because step size itself may be making optimization harder on these systems, smaller `dt` remains the first intervention to test before spending `10x` more budget on longer training.

5. Next steps:
- Rerun the exact same focused pilot (`kuramoto`, `hopfield`, `dt in {0.025, 0.0125}`, `20k`, current grids) with the repaired launcher.
- Re-collect and inspect whether the smaller-`dt` runs produce clear monotonic gains at `H500/H1000` or merely shift catastrophic tails.
- Only consider a targeted `200000`-step follow-up after that rerun if the repaired `20k` smaller-`dt` results look optimization-limited rather than obviously integration-limited.

### W) Canonical Paper Benchmark Completion: Repaired `dt`-Rescue Matrix Collected and Compared (`29` Systems, `4` Baselines, `TS=256`, `L=8`, `50k`)
Timestamp: 2026-03-08
Status: **completed**

1. Concrete results:
- The corrected `v4` paper-benchmark chain completed end-to-end:
  - rescue pass `1` array: `8903420_[0-41]`
  - rescue pass `1` collector: `8903421`
  - resolver after pass `1`: `8903422`
  - rescue pass `2` array: `8903423_[0-86]`
  - rescue pass `2` collector: `8903424`
  - resolver after pass `2`: `8903425`
  - full matrix array: `8903426_[0-347]`
  - final collect: `8903427`
  - final compare: `8903428`
- Final paper-facing artifacts now exist under:
  - `results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_collect/`
  - `results/paper_benchmark_20260307_paper_final_ts256_50k_v4/final_compare/`
- Cross-system `H1000` system-median best-periodic ranking:
  - `generic_sparse`: `0.0328`
  - `lista_dense`: `0.0388`
  - `lista_blockdiag`: `0.1508`
  - `lista_diagonal`: `1.2110`
- Shared-system comparison vs `generic_sparse`:
  - dense LISTA wins `15/29`
  - block-diagonal LISTA wins `3/29`
  - diagonal-K LISTA wins `3/29`
- Pass-2 `dt` resolution outcome:
  - `15/29` systems accepted default `dt`
  - `4/29` systems accepted after at least one halving
  - `10/29` systems remained `integration_hard`

2. Context:
- This is the completed rerun that supersedes the provisional default-`dt` `v3` matrix as the canonical paper benchmark.
- The point of the rerun was not architecture churn; it was to complete the benchmark under the repaired collector and the intended `generic_sparse`-driven `dt` rescue policy.

3. Interpretation:
- `generic_sparse` remains the strongest overall paper baseline.
- Dense LISTA is now the only LISTA-family model that is close enough to matter as a cross-system comparator, but it still does not overturn the global ranking.
- Queue completion is no longer the blocker. The paper bottleneck has shifted to the `integration_hard` systems that remain bad even after the allowed rescue passes.

4. Project implications:
- Paper tables, summaries, and model-ranking claims should now point to `v4` `final_collect` / `final_compare`, not the provisional `v3` matrix or the `v4` anchor-only stage.
- Cross-system and intrinsic-HD conclusions should now be separated:
  - dense LISTA is the strongest cross-system LISTA comparator
  - `lista_blockdiag` remains the only LISTA-family intrinsic-HD candidate worth carrying forward

5. Next steps:
- Update all paper-track notes and figure references to cite the completed `v4` artifacts.
- Prioritize targeted follow-up on the `10` `integration_hard` systems, especially `kuramoto` and `hopfield`.
- Use the completed benchmark to decide whether the next paper-facing intrinsic-HD work should be step-size repair, optimization changes, or representation changes on the unresolved systems.

### V) Intrinsic-HD DT Rescue Pilot Collection Audit: Scheduler Finished, but the Collected Summary Is Incomplete (`kuramoto` + `hopfield`, Seq8, `TARGET_SIZE=256`, `20k`)
Timestamp: 2026-03-08
Status: **completed, follow-up required**

1. Concrete results:
- The focused intrinsic-HD `dt` rescue scheduler chain completed:
  - launcher: `8903785`
  - `generic_sparse` array: `8903787_[0-23]`
  - `lista_blockdiag` array: `8903788_[0-23]`
  - collector: `8903789`
- Output roots:
  - `/network/scratch/l/lia/skae/intrinsic_hd_dt_rescue_20260308`
  - `results/intrinsic_hd_dt_rescue_20260308`
- Intended grid:
  - systems: `kuramoto`, `hopfield`
  - `dt`: `0.025`, `0.0125`
  - seeds: `0,1,2`
  - `generic_sparse`: `sp in {0.0005, 0.0025}`
  - `lista_blockdiag`: `sp in {0.0005, 0.0010}`, `alpha=0.15`, `loops=1`, `block=16`
  - `num_steps=20000`
- Current collected artifact gap:
  - `results/intrinsic_hd_dt_rescue_20260308/forecasting_rows.csv` contains only `2` rows total
  - both rows are `kuramoto`, `seed_0`, `dt=0.025`
  - collected `H1000` best-periodic:
    - `generic_sparse`: `52.8679`
    - `lista_blockdiag`: `28.2276`
  - no `hopfield` rows and no `dt=0.0125` rows appear in the current summary

2. Context:
- This pilot was supposed to be the cheapest direct test of the paper-track step-size hypothesis on the two intrinsic-HD systems that stayed obviously bad at default `dt=0.05`.
- The current-default Kuramoto recovery sweep already suggested that `lista_blockdiag` was the only LISTA-family intrinsic-HD variant worth retesting under smaller `dt`.

3. Interpretation:
- The scheduler did its job, but the collected result is not decision-grade.
- The currently visible rows do suggest that `dt=0.025` alone does not make Kuramoto paper-worthy, but that is too thin to support a broader claim because the summary is missing most of the intended grid.

4. Project implications:
- We still do not have clean focused-pilot evidence on whether smaller `dt` alone resolves `kuramoto` / `hopfield`.
- Intrinsic-HD follow-up is therefore blocked on rerunning the repaired launcher rather than on more interpretation of the current two-row summary.

5. Next steps:
- Keep this entry as the scheduler/collection audit record, but use `### X` as the launch root-cause source of truth.
- Rerun the same narrow pilot with the repaired launcher before making any paper-facing step-size conclusion for `kuramoto` / `hopfield`.
- Continue using the current-default Kuramoto recovery result (`lista_blockdiag sp_0p0010 loops_1`) as the intrinsic-HD LISTA anchor until the smaller-`dt` pilot is repaired.

### U) Kuramoto Recovery Completion + Reporting Repair: Current-Default Retuning Beats Matched `generic_sparse` but Remains Above Good-Forecast Band
Timestamp: 2026-03-08
Status: **completed**

1. Concrete results:
- The Kuramoto recovery chain completed cleanly:
  - launcher: `8882966`
  - `generic_sparse` array: `8882967_[0-8]` (`9/9` completed)
  - `lista_blockdiag` array: `8882968_[0-26]` (`27/27` completed)
  - collector: `8882969` (`COMPLETED`, `0:0`)
- The standard summary artifacts were regenerated with the repaired sweep collection (`--select all`, `H100/H500/H1000`):
  - `results/kuramoto_recovery_seq8_20260305/forecasting_summary.{json,md}`
  - `results/kuramoto_recovery_seq8_20260305/forecasting_rows.{csv,json}`
- The repaired artifacts now expose the full `36`-row sweep and confirm the actual forecasting result:
  - best block-diagonal arm: `sp_0p0010`, `loops_1` with seed-median best-periodic `H100=5.47`, `H500=24.83`, `H1000=48.64`
  - second-best block-diagonal arm: `sp_0p0005`, `loops_1` with seed-median `H1000=52.09`
  - best matched `generic_sparse` arm: `sp_0p0005` with seed-median best-periodic `H100=6.24`, `H500=48.30`, `H1000=136.60`
- Heavy-tail note:
  - several block-diagonal settings still have catastrophic worst-seed `H1000` outliers (for example `3.12e4` and `1.10e5`)

2. Context:
- This was the staged current-default Kuramoto follow-up proposed after the intrinsic-HD baseline showed `generic_sparse` ahead, `lista_blockdiag` second-best, and dense LISTA unusable.
- The intent was to test whether modest block-diagonal retuning on the current environment (`20k` steps, lower sparsity pressure, low loop counts, `alpha=0.15`) could recover long-horizon performance before changing the representation.

3. Interpretation:
- Partial recovery exists: low-loop block-diagonal LISTA can beat matched `generic_sparse` on Kuramoto at `H500/H1000`.
- Kuramoto is still not solved: even the best seed-median `H1000` remains `48.64`, still far above the good-forecast band `< 10`, and robustness across seeds/settings is uneven.
- The reporting problem is fixed: the standard summary files now contain the full sweep and can be used directly for audit / comparison.

4. Project implications:
- On current defaults, `lista_blockdiag` remains the only LISTA-family Kuramoto candidate worth carrying forward.
- Kuramoto no longer looks like a simple `generic_sparse`-dominant case; retuned block-diagonal LISTA can be materially better, but the gain is not yet sufficient for promotion as a stable cross-system default.
- Further Kuramoto decision-making should now focus on smaller `dt`, since the default-`dt` reporting path is repaired and the remaining bottleneck is forecasting quality rather than missing artifacts.

5. Next steps:
- Promote `lista_blockdiag sp_0p0010 loops_1` as the current-default Kuramoto anchor for any near-term follow-up.
- Use the repaired or rerun intrinsic-HD `dt` rescue pilot to determine whether smaller `dt` closes the remaining Kuramoto gap before changing the representation.
- Only escalate to phase-centered / sinusoidal input changes if smaller-`dt` block-diagonal retuning still plateaus above the good-forecast band.

### T) Queue 5b Completion: HyperLISTA Stabilization Smoke (Duffing 2D, `L=8`, `10k`, 3 Seeds)
Timestamp: 2026-03-05
Status: **completed**

1. Concrete results:
- Queue-5b smoke chain completed cleanly:
  - launcher: `8883005`
  - sweep array: `8883008_[0-23]` (`24/24` completed, no failures)
  - collector: `8883009` (`COMPLETED`, `0:0`)
- Final artifacts:
  - `results/duffing_hyperlista_q05b_smoke_10k_20260305/duffing_hyperlista_q05b_smoke_10k_summary.{json,md}`
  - `results/duffing_hyperlista_q05b_smoke_10k_20260305/duffing_hyperlista_q05b_smoke_10k_pareto_frontier.{json,md}`
- Safe settings used throughout:
  - `use_ss=false`
  - `use_momentum=false`
  - `constrain_c_theta=true`
  - `eval_profile=smoke`
- Best aggregate long-horizon arm:
  - `sp_0p0060_loops_1_ct_0p0040`: `quick=0.9535`, `H100=0.3372`, `H500=52.3228`, `H1000=4223.8420`, `sparsity=0.00166`
- Aggregate trend:
  - for both tested sparsity coefficients, `c_theta=0.0040` was best and higher `c_theta` values monotonically worsened `H100/H500/H1000`.

2. Context:
- This was the first post-repair HyperLISTA smoke queue after patching the stabilization path:
  - constrained positive `c_theta`
  - safe `pinv(D)` recomputation
  - backward-compatible loading for legacy `c_theta` checkpoints
- The queue intentionally used a short `10k` smoke profile and the conservative settings above to verify that the repaired code path runs end-to-end before broader HyperLISTA search.

3. Interpretation:
- The repaired HyperLISTA code path is operational: all `24/24` tasks and the collector finished successfully with no queue failures.
- Forecast quality is still very poor relative to the current LISTA and `generic_sparse` anchors. Even the best smoke arm remains orders of magnitude worse than the Queue-4 LISTA anchors at `H500/H1000`.
- The dominant signal in this smoke grid is not “find the right sparsity band”; it is that lower `c_theta` is materially safer for long-horizon forecasting than higher `c_theta` under the repaired setup.

4. Project implications:
- HyperLISTA is now safe to iterate on technically, but it is not yet competitive on the actual forecasting objective.
- Future HyperLISTA ranking should be forecasting-first, consistent with the current LISTA-family policy; sparsity should be recorded but should not block promotion at this stage.
- Queue-4 LISTA anchors and `generic_sparse` remain the forecasting references for any HyperLISTA follow-up.

5. Next steps:
- Keep the repaired HyperLISTA defaults (`use_ss=false`, `use_momentum=false`, constrained `c_theta`) for any near-term follow-up.
- If HyperLISTA is retried, search around the lower-`c_theta` region first and rank candidates by `H500/H1000` robustness rather than sparsity-band compliance.
- Fold the smoke summary into supervisor-facing status notes only as a code-path validation plus forecasting-negative result, not as a sparsity study.

### S) Intrinsic-HD Baseline Completion: Seq8, `TARGET_SIZE=256` on `kuramoto` / `hopfield` / `competitive_lv`
Timestamp: 2026-03-05
Status: **completed**

1. Concrete results:
- All `27/27` array tasks completed cleanly with `0:0` exit codes:
  - `generic_sparse`: `8877775_[0-8]`
  - dense LISTA: `8877776_[0-8]`
  - block-diagonal LISTA: `8877777_[0-8]`
- Output roots:
  - `/network/scratch/l/lia/skae/high_dim_benchmarks_plan_seq8_20260305`
  - `results/high_dim_benchmarks_plan_seq8_20260305`
- Summary artifacts:
  - `results/high_dim_benchmarks_plan_seq8_20260305/forecasting_summary.{json,md}`
  - `results/high_dim_benchmarks_plan_seq8_20260305/system_medians_h1000.md`
- Seed-median `H1000` best-periodic forecasting by system:
  - `competitive_lv`: `generic_sparse=0.0651`, `lista_blockdiag=0.1192`, `lista_dense=0.1654`
  - `kuramoto`: `generic_sparse=199.86`, `lista_blockdiag=258.50`, `lista_dense=6.636e8`
  - `hopfield`: `generic_sparse=5436.67`, `lista_blockdiag=3.599e15`, `lista_dense=3.045e33`

2. Context:
- This was the first completed intrinsic-HD `L=8` baseline from `docs/planning/high_dim_benchmarks_plan.md` at matched `TARGET_SIZE=256`.
- These runs used the current environment defaults rather than the stricter plan sizes:
  - `kuramoto=16` oscillators
  - `hopfield=16` neurons
  - `competitive_lv=10` species
- The purpose was to compare `generic_sparse`, dense LISTA, and block-diagonal LISTA before queueing the stricter `N=32/64` variants.

3. Interpretation:
- `competitive_lv` is not the bottleneck: all three models remain accurate at `H1000`, and periodic reencoding strongly rescues long-horizon error.
- `kuramoto` cleanly separates architectures: `generic_sparse` is best, `lista_blockdiag` is viable but not robust, and dense LISTA collapses.
- `hopfield` is the blocking environment. Best-periodic selection collapses to short periods (`periodic_10`), but that does not rescue forecasting; LISTA variants show catastrophic tails, and even `generic_sparse` remains far above the good-forecast band.
- On `hopfield`, `every_step` medians stay around `1e2`, while best-periodic often explodes, so periodic reencoding is not fixing the autonomous-stability issue in this regime.

4. Project implications:
- `generic_sparse` remains the strongest baseline across the current-default intrinsic-HD suite.
- Dense LISTA should not be promoted as the intrinsic-HD baseline; it is already catastrophically unstable on `kuramoto` and `hopfield`.
- `lista_blockdiag` is the only LISTA-family variant worth carrying forward to stricter intrinsic-HD tests, but only as a secondary baseline behind `generic_sparse`.
- The main unresolved issue is long-horizon autonomous stability on `hopfield`, not forecasting on `competitive_lv`.
- Because these were milder-than-plan defaults, the current ranking should be treated as a baseline rather than the final verdict for the intrinsic-HD plan.

5. Next steps:
- Queue the stricter intrinsic-HD plan variants (`N=32/64`) to test whether the current ranking persists under the intended scale.
- For `hopfield`, inspect whether stability control is the main missing ingredient before investing further in dense LISTA variants.
- Carry forward `generic_sparse` and `lista_blockdiag` as the baseline pair for the stricter sweep; drop dense LISTA unless there is a specific ablation reason to keep it.

### R) Queue 5 Completion: HyperLISTA Adaptive-Threshold Sweep (Duffing 2D, L=8, 3 Seeds)
Timestamp: 2026-03-04
Status: **completed**

1. Concrete results:
- Queue-5 chain completed cleanly:
  - launcher: `8876267`
  - sweep array: `8876268_[0-23]` (`24/24` completed, no failures)
  - collector: `8876269` (`COMPLETED`, `0:0`)
- Final artifacts:
  - `results/duffing_hyperlista_q05_adaptive_50k_20260304/duffing_hyperlista_q05_adaptive_50k_summary.{json,md}`
  - `results/duffing_hyperlista_q05_adaptive_50k_20260304/duffing_hyperlista_q05_adaptive_50k_pareto_frontier.{json,md}`
- Best aggregate long-horizon arm:
  - `sp_0p0040_loops_1_ct_0p0040`: `quick=0.7751`, `H500=1.5481`, `H1000=2.2958`, `sparsity=0.00010`
- Sparsity target result:
  - in-band (`0.7–0.9`) aggregate arms: `0/8`
  - all arms are near-zero sparsity ratio (`~1e-4`)

2. Context:
- Queue-4 identified the best in-band LISTA operating points (`sp_0p0060_loops_1`, `sp_0p0040_loops_1`) but still left a substantial gap vs `generic_sparse`.
- Queue-5 tested whether adaptive thresholding (HyperLISTA) could close that gap while preserving the target sparsity band.

3. Interpretation:
- In this tested region, HyperLISTA thresholds are effectively too permissive: supports become dense (`sparsity ~ 1e-4`) rather than target-sparse.
- Forecast quality also degrades severely: compared with Queue-4 robustness anchor `sp_0p0060_loops_1`, the best Queue-5 arm is worse by `~2.45x` (quick), `~130x` (`H500`), `~24.98x` (`H1000`).
- Versus `generic_sparse`, the same arm is worse by `~6.95x` (quick), `~501x` (`H500`), `~78.09x` (`H1000`).
- Best-period modes remain collapsed to `periodic_10` across arms/horizons, consistent with weak autonomous rollout stability.

4. Project implications:
- HyperLISTA adaptive-thresholding is not a viable replacement for Queue-4 anchors under the current `c_theta` grid and coefficient settings.
- Queue-4 in-band anchors remain the only credible LISTA-family baselines for next-stage comparisons.
- The core unresolved issue remains LISTA rollout robustness while preserving target sparsity.

5. Next steps:
- Move to Queue `6` (encoder input-map variants), benchmarked directly against Queue-4 anchors and `generic_sparse`.
- Keep Queue-5 as a completed negative control to avoid retesting this threshold region.
- If HyperLISTA is revisited later, test a materially different threshold regime (stronger shrinkage schedule / different parameterization), not minor `c_theta` perturbations around this grid.

### Q) Queue 4 Completion: Joint Pareto Sweep (`SPARSITY_COEFF × lista_num_loops`) at Fixed `lista_alpha=0.15` (Duffing 2D, L=8, 3 Seeds)
Timestamp: 2026-03-04
Status: **completed**

1. Concrete results:
- Queue-4 jobs completed:
  - Sweep array: `8875708_[0-59]`
  - Collector: `8875709`
- Summary artifacts:
  - `results/duffing_lista_q04_pareto_50k_20260304/duffing_lista_q04_pareto_50k_summary.json`
  - `results/duffing_lista_q04_pareto_50k_20260304/duffing_lista_q04_pareto_50k_summary.md`
  - `results/duffing_lista_q04_pareto_50k_20260304/duffing_lista_q04_pareto_50k_pareto_frontier.json`
  - `results/duffing_lista_q04_pareto_50k_20260304/duffing_lista_q04_pareto_50k_pareto_frontier.md`
- Pareto frontier arms:
  - `sp_0p0010_loops_1`: quick `0.4146`, `H500` `0.0127`, `H1000` `0.0911`, sparsity `0.6865` (**out of band**)
  - `sp_0p0060_loops_1`: quick `0.3162`, `H500` `0.0119`, `H1000` `0.0919`, sparsity `0.8490` (**in band**)
  - `sp_0p0040_loops_1`: quick `0.3108`, `H500` `0.0131`, `H1000` `0.0996`, sparsity `0.8190` (**in band**)
  - `sp_0p0005_loops_7`: quick `0.5802`, `H500` `0.0351`, `H1000` `0.1890`, sparsity `0.8043` (**in band**)
- Unstable mean outliers observed:
  - `sp_0p0010_loops_7` (`H1000 mean=1.343e8`)
  - `sp_0p0060_loops_7` (`H500 mean=2.871e5`)
  - `sp_0p0005_loops_5` (`H500 mean=5.39`)

2. Context:
- Queue-2 and Queue-3 showed conflicting single-knob optima (forecasting vs sparsity-band), so Queue-4 jointly swept coefficient and LISTA depth to identify non-dominated operating points.
- User objective remained to keep sparsity near `~0.8` while preserving long-horizon forecasting quality.

3. Interpretation:
- Queue-4 found a better in-band tradeoff than prior single-knob anchors:
  - `sp_0p0060_loops_1` is the strongest in-band long-horizon point.
  - `sp_0p0040_loops_1` is the strongest in-band quick-best point with only modest long-horizon degradation relative to `sp_0p0060_loops_1`.
- Mean-based ranking can be badly distorted by heavy-tail seeds for some arms; those settings should be treated as unstable even when one horizon metric appears good.

4. Project implications:
- The forecast-vs-sparsity tradeoff is now mapped directly; we no longer need to tune coefficient and depth independently for this regime.
- Despite Queue-4 improvement, LISTA still trails `generic_sparse` meaningfully:
  - Versus `generic_sparse` (`quick=0.1115`, `H500=0.00309`, `H1000=0.0294`), the best in-band Queue-4 arm (`sp_0p0060_loops_1`) is about `2.84x`/`3.85x`/`3.12x` worse on quick/`H500`/`H1000`.
- Main unresolved issue is now robustness/expressivity of LISTA dynamics, not just sparsity calibration.

5. Next steps:
- Launch Queue `5` adaptive-threshold intervention (HyperLISTA / learnable thresholds), seeded from:
  - robustness-first: `sp_0p0060_loops_1`
  - quick-best tie-break: `sp_0p0040_loops_1`
- Exclude heavy-tail Queue-4 arms from promotion unless robustified.
- Re-evaluate against `generic_sparse` using the same metrics and sparsity-band objective.

### P) Queue 4 Launch: Joint Pareto Sweep (`SPARSITY_COEFF × lista_num_loops`) at Fixed `lista_alpha=0.15` (Duffing 2D, L=8, 3 Seeds)
Timestamp: 2026-03-04
Status: **completed (launch record; results in Entry Q)**

1. Concrete results:
- Added Queue-4 scripts:
  - `scripts/sweep_duffing_lista_pareto_q4.sh`
  - `scripts/collect_duffing_lista_pareto_q4.sh`
  - `scripts/queue_duffing_lista_queue04_pareto.sh`
- Added Pareto utility:
  - `tools/compute_pareto_frontier.py`
- Submitted Queue-4 chain via queue launcher:
  - Queue launcher: `8875686` (**completed**)
  - Sweep array: `8875708_[0-59]`
  - Collector: `8875709` (`afterany:8875708`)
- Sweep grid:
  - `sparsity_coeff ∈ {0.0005, 0.0010, 0.0020, 0.0040, 0.0060}`
  - `lista_num_loops ∈ {1,2,5,7}`
  - `seed ∈ {0,1,2}`
- Collector outputs both aggregate and Pareto artifacts:
  - `duffing_lista_q04_pareto_50k_summary.{json,md}`
  - `duffing_lista_q04_pareto_50k_pareto_frontier.{json,md}`

2. Context:
- Queue-2 and Queue-3 each varied a single knob and produced conflicting optima (forecast vs sparsity-band compliance).
- A joint sweep is required to identify non-dominated settings rather than forcing a single scalar objective too early.

3. Interpretation:
- Queue-4 is the correct next step to map the true tradeoff surface and avoid locking into an artifact of single-knob tuning.
- Pareto reporting makes the branch decision explicit and reproducible.

4. Project implications:
- If Queue-4 yields an in-band point with materially better `H500/H1000`, that becomes the new LISTA anchor.
- If all in-band Pareto points remain far from `generic_sparse`, we should transition from coefficient/depth tuning to threshold adaptivity/encoder-method changes.

5. Next steps:
- Wait for Queue-4 completion and inspect non-dominated arms.
- Compare frontier candidates against `generic_sparse` benchmark.
- Select Queue-5 branch based on remaining gap and robustness.

### O) Queue 3 Completion: `lista_num_loops` Capacity Sweep at Fixed (`lista_alpha=0.15`, `sparsity_coeff=0.0005`) (Duffing 2D, L=8, 3 Seeds)
Timestamp: 2026-03-04
Status: **completed**

1. Concrete results:
- Queue-3 jobs completed:
  - Sweep array: `8874672_[0-14]`
  - Collector: `8874673`
- Summary artifacts:
  - `results/duffing_lista_q03_loops_50k_20260304/duffing_lista_q03_loops_50k_summary.json`
  - `results/duffing_lista_q03_loops_50k_20260304/duffing_lista_q03_loops_50k_summary.md`
- Aggregate means:
  - `loops_1`: quick `0.4164`, `H500` `0.0114`, `H1000` `0.0889`, sparsity `0.6644`
  - `loops_2`: quick `0.6919`, `H500` `0.0133`, `H1000` `0.0969`, sparsity `0.6944`
  - `loops_3`: quick `0.6585`, `H500` `0.0571`, `H1000` `643.10`, sparsity `0.7255`
  - `loops_5`: quick `0.6653`, `H500` `0.0185`, `H1000` `0.1616`, sparsity `0.7735`
  - `loops_7`: quick `0.5915`, `H500` `0.0239`, `H1000` `0.1317`, sparsity `0.8040`

2. Context:
- Queue-3 tested whether increasing/decreasing LISTA depth could improve forecast robustness while holding the Queue-2 winner (`sp=0.0005`) fixed.

3. Interpretation:
- Forecast-only best is `loops_1`, but it under-shoots sparsity target (`0.6644`, below band).
- `loops_7` stays in-band (`0.8040`) but gives weaker forecast means than `loops_1`.
- `loops_3` shows severe instability from a catastrophic outlier seed, indicating depth increases are not uniformly safe.

4. Project implications:
- Depth alone does not resolve the forecast-vs-sparsity tradeoff cleanly.
- This supports moving to a joint selection strategy (Queue-4 Pareto) instead of committing to a single depth at fixed sparsity.

5. Next steps:
- Run Queue-4 joint sweep (`SPARSITY_COEFF × num_loops`) and select a non-dominated operating point.
- Use Pareto frontier plus `generic_sparse` gap to choose the next intervention family.

### N) Queue 2 Completion: `SPARSITY_COEFF` Sweep at Fixed `lista_alpha=0.15` (Duffing 2D, L=8, 3 Seeds)
Timestamp: 2026-03-04
Status: **completed**

1. Concrete results:
- Queue-2 jobs completed:
  - Sweep array: `8874221_[0-14]`
  - Collector: `8874222`
- Summary artifacts:
  - `results/duffing_lista_q02_spcoeff_50k_20260304/duffing_lista_q02_spcoeff_50k_summary.json`
  - `results/duffing_lista_q02_spcoeff_50k_20260304/duffing_lista_q02_spcoeff_50k_summary.md`
- Aggregate means (lower is better for errors):
  - `sp_0p0005`: quick-best `0.6510`, `H500` `0.0191`, `H1000` `0.1324`, sparsity `0.7708`
  - `sp_0p0010`: quick-best `0.5615`, `H500` `0.0217`, `H1000` `0.1938`, sparsity `0.8744`
  - `sp_0p0020`: quick-best `0.6125`, `H500` `0.0195`, `H1000` `0.1158`, sparsity `0.9215`
  - `sp_0p0040`: quick-best `0.4584`, `H500` `0.0266`, `H1000` `0.1973`, sparsity `0.9391`
  - `sp_0p0060`: quick-best `0.4291`, `H500` `0.0156`, `H1000` `0.0987`, sparsity `0.9461`
- In-band candidates (`0.7–0.9`): `sp_0p0005`, `sp_0p0010`.
- Independent QA pass (`15/15` rows) validated that summary rows exactly match raw run artifacts and fixed-config invariants (`alpha=0.15`, `num_loops=5`, `final_op=relu`).

2. Context:
- Queue-1 winner (`alpha=0.15`) was over-sparse (`~0.95`), so Queue-2 isolated sparsity calibration by sweeping `SPARSITY_COEFF`.
- User objective was explicit: keep alpha moderate and reduce sparsity toward `~0.8`.

3. Interpretation:
- `sp_0p0005` is the best coefficient under the target objective:
  - within target band, closest to `0.8`,
  - best long-horizon metrics among in-band candidates (`H500/H1000`),
  - and improved long-horizon performance vs Queue-1 winner (`alpha_0p15`, `H1000=0.1709`).
- Higher coefficients (`>=0.0020`) can reduce some error metrics but push sparsity back outside target band (`>0.9`), conflicting with the design target.

4. Project implications:
- Sparsity-target recovery is now validated at fixed alpha (`0.15`) using `sparsity_coeff=0.0005`.
- The remaining performance gap is no longer “wrong sparsity level”; it is now likely tied to LISTA capacity/optimization behavior.

5. Next steps:
- Lock `sp_0p0005` as the new anchor and run Queue-3 depth/capacity sweep.
- Select depth by long-horizon robustness first (`H500/H1000`), while keeping sparsity in-band.
- If depth tuning fails to close robustness gaps, move to adaptive-threshold and encoder-variant interventions.

### M) Queue 2 Launch: `SPARSITY_COEFF` Sweep at Fixed `lista_alpha=0.15` (Duffing 2D, L=8, 3 Seeds)
Timestamp: 2026-03-04
Status: **completed (launch record; results in Entry N)**

1. Concrete results:
- Launched Queue-2 sparsity sweep to reduce `sparsity_ratio` toward `~0.8` while keeping `lista_alpha=0.15` fixed:
  - Sweep array: `8874221_[0-14]`
  - Collector: `8874222` (afterany dependency on sweep)
- Sweep grid:
  - `sparsity_coeff ∈ {0.0005, 0.0010, 0.0020, 0.0040, 0.0060}`
  - `seed ∈ {0,1,2}`
- Added reproducible Queue-2 scripts:
  - `scripts/sweep_duffing_lista_spcoeff_q2.sh`
  - `scripts/collect_duffing_lista_spcoeff_q2.sh`
  - `scripts/queue_duffing_lista_queue02_spcoeff.sh`

2. Context:
- Queue-1 full results selected `alpha=0.15` as the strongest long-horizon setting (`H500/H1000`) among tested alphas.
- Queue-1 also showed all tested alphas were over-sparse (`sparsity_ratio ~ 0.95`), missing the target `0.7–0.9`.
- This sweep isolates sparsity-pressure tuning at fixed alpha to test whether the target band can be reached without degrading long-horizon forecasting.

3. Interpretation:
- This is the correct immediate follow-up to user-priority criteria: keep alpha moderate (`0.15`) and reduce shrinkage pressure through lower `SPARSITY_COEFF`.
- Metrics are not available yet; no branch decision is changed until Queue-2 summary is collected.

4. Project implications:
- If one or more coefficients move sparsity into band (`~0.8`) while preserving `H500/H1000`, we can lock a stronger LISTA config for the next stage.
- If all coefficients remain too sparse or collapse forecasting, this supports moving to Queue `3` (capacity/depth-focused changes).

5. Next steps:
- Wait for Queue-2 sweep/collector completion and summarize metrics from:
  - `results/duffing_lista_q02_spcoeff_50k_20260304/duffing_lista_q02_spcoeff_50k_summary.{json,md}`
- Rank coefficients by `H500/H1000`, `quick-best`, sparsity-band match (`0.7–0.9`, target near `0.8`), and mode distribution.
- Decide Queue `2` winner or escalate to Queue `3`.

### L) Queue 0-1 Launch: ReLU-Baseline + Focused `lista_alpha` Sweep (Duffing 2D, L=8, 3 Seeds)
Timestamp: 2026-03-04
Status: **completed**

1. Concrete results:
- Implemented Queue `0–1` launch scripts and submitted the dependency chain.
- Detected Queue-1 launch bug: comma-delimited `LISTA_ALPHAS`/`SEEDS_CSV` were truncated by `sbatch --export`, so initial Queue-1 run only executed task `0` and marked tasks `1–14` out-of-range (`TOTAL_JOBS=1`).
- Patched launch scripts to pass CSV values via environment prefix (instead of inline `--export` values), then canceled and relaunched Queue-1.
- Queue 0 completed:
  - Sweep: `8873286_[0-2]` (**completed**)
  - Scheduled collector `8873287` was canceled due queue delay; equivalent collector command was run locally.
  - Baseline aggregate: quick-best `0.4527`, `H500` `0.0506`, `H1000` `0.3992`, sparsity median `0.9571`.
- Queue 1 gate completed:
  - Initial (canceled): Queue 1 gate sweep `8873288_[0-14]`, collector `8873289`, selector `8873290`, full launcher `8873291`
  - Corrected gate chain: sweep `8873328_[0-14]`, collector `8873329`
  - Gate summary artifacts: `results/duffing_lista_q01_20260304/queue1_gate/duffing_lista_alpha_gate_10k_summary.{json,md}`
- Gate aggregate means (10k stage, lower better for errors):
  - `alpha_0p15`: quick-best `1.2095`, `H500` `0.1082`, `H1000` `0.3736`, sparsity median `0.9501`
  - `alpha_0p10`: quick-best `0.9123`, `H500` `0.1201`, `H1000` `0.4532`, sparsity median `0.9484`
  - `alpha_0p20`: quick-best `0.9936`, `H500` `0.1739`, `H1000` `0.5712`, sparsity median `0.9491`
  - `alpha_0p30`: quick-best `0.8347`, `H500` `0.1953`, `H1000` `0.5733`, sparsity median `0.9467`
  - `alpha_0p40`: quick-best `0.9646`, `H500` `0.1452`, `H1000` `0.5772`, sparsity median `0.9473`
- Selector job `8873330` remained queued (`PD`) with dependencies satisfied, so selection was run locally using `tools/select_lista_alpha_survivors.py`; survivors were `alpha_0p15`, `alpha_0p10`, `alpha_0p20`.
- Full stage completed:
  - Launcher `8873462`, sweep `8873469_[0-8]` (**completed**)
  - Scheduled collector `8873470` was canceled due queue delay; equivalent collector command was run locally.
  - Full summary artifacts: `results/duffing_lista_q01_20260304/queue1_full/duffing_lista_alpha_full_50k_summary.{json,md}`
- Full-stage aggregate means (50k survivors):
  - `alpha_0p15`: quick-best `0.3918`, `H500` `0.0205`, `H1000` `0.1709`, sparsity median `0.9549`
  - `alpha_0p20`: quick-best `0.4581`, `H500` `0.0269`, `H1000` `0.2060`, sparsity median `0.9552`
  - `alpha_0p10`: quick-best `0.4672`, `H500` `0.0521`, `H1000` `0.4166`, sparsity median `0.9571`
- Full ranking (primary `H1000`, tie-break `H500`, then quick-best): `alpha_0p15` > `alpha_0p20` > `alpha_0p10`.
- Best-period mode distributions at full stage:
  - `alpha_0p15`: `H500` mostly `periodic_25` (67%), `H1000` mostly `periodic_50` (67%).
  - `alpha_0p20`: `H500/H1000` both `periodic_25` (100%).
  - `alpha_0p10`: mixed `periodic_10/25/50`, with one severe long-horizon outlier (`seed_1`, `H1000=0.9833`).

2. Context:
- This is the first execution of `docs/LISTA_STACK_FORECASTING_PLAN.md` Queue `0–1` after enforcing ReLU-final in LISTA.
- Queue 0 re-baselines the coefficient-matched LISTA setup from Experiment K (`reconst=0.5`, `sparsity=0.01`) under the new final-op behavior.
- Queue 1 uses a short gate (`10k`) before full runs (`50k`) to reduce wasted long jobs on clearly weak alphas.

3. Interpretation:
- Queue `0–1` orchestration is valid after the CSV export fix; corrected gate execution covered all 15 alpha-seed tasks.
- Full stage confirms a clear winner among tested alphas: `alpha_0p15` has the best `H500/H1000` means and best quick-best mean.
- `alpha_0p20` is second-best and shows the cleanest mode concentration (`periodic_25`), but with weaker long-horizon means than `alpha_0p15`.
- `alpha_0p10` is unstable across seeds (one large `H1000` outlier), making it a weak candidate despite one strong seed.
- No tested alpha satisfies sparsity target (`0.7–0.9`): all remain over-sparse near `0.95`.
- Versus Experiment K, `alpha_0p15` improves LISTA (`quick-best` and `H1000`) but remains far behind `generic_sparse` on long horizon.

4. Project implications:
- ReLU-final + alpha selection materially improved LISTA within this family, but alpha tuning alone does not solve the sparsity-target mismatch.
- The dominant remaining control knob is sparsity calibration at fixed strong alpha, not broader alpha exploration.
- Queue `2` is the appropriate next branch: keep `alpha=0.15` and sweep `SPARSITY_COEFF` to target `sparsity_ratio 0.7–0.9` without sacrificing `H500/H1000`.

5. Next steps:
- Launch Queue `2` (`SPARSITY_COEFF` sweep) at fixed `lista_alpha=0.15`, `L=8`, `50k`, 3 seeds.
- Use the same ranking criteria (`H500/H1000`, quick-best, sparsity band, mode distribution) to choose the best Queue-2 setting.
- If Queue-2 cannot bring sparsity into `0.7–0.9` without long-horizon collapse, move to Queue `3` (`lista_num_loops`/capacity or refined alpha neighborhood).

### M) Canonical Paper Benchmark Progress Check (`v3` Collected, Provisional, Rerun Required)
Timestamp: 2026-03-07
Status: **completed once, rerun required**

1. Concrete results:
- The corrected `v3` paper benchmark chain (`8897639`-`8897652`) completed smoke, anchor, rescue, full training, and final collection.
- After repairing `tools/collect_forecasting_roots.py` for the `system/dt_tag/seed/run` layout and rerunning collection locally:
  - repaired final collection found `347` rows out of the expected `348`
  - the missing row is `lista_dense / multiwell_strong_transition / seed_2` because the run lacks `evaluation_results_best.json`
  - repaired pairwise comparisons against `generic_sparse` were generated under:
    - `results/paper_benchmark_20260306_paper_final_ts256_50k_v3/final_compare/`
- Provisional cross-system `H1000` best-periodic medians from the repaired default-`dt` full matrix:
  - `generic_sparse`: `0.0251`
  - `lista_dense`: `0.0648`
  - `lista_blockdiag`: `0.1619`
  - `lista_diagonal`: `1.1417`
- Provisional catastrophic-system counts at `H1000`:
  - `generic_sparse`: `2`
  - `lista_dense`: `4`
  - `lista_blockdiag`: `2`
  - `lista_diagonal`: `2`
- Candidate-vs-anchor comparisons on the repaired default-`dt` matrix:
  - dense LISTA vs `generic_sparse`: candidate wins `5/29`, anchor wins `24/29`
  - block LISTA vs `generic_sparse`: candidate wins `4/29`, anchor wins `25/29`
  - diagonal-K LISTA vs `generic_sparse`: candidate wins `6/29`, anchor wins `23/29`
- After recomputing anchor pass `0` with the repaired collector:
  - `13/29` systems accept default `dt`
  - `16/29` systems require at least one halving before the benchmark matches the intended protocol

2. Context:
- The full `v3` matrix did finish, but the original in-chain collector had been empty during queue execution.
- Because of that, the in-chain `dt` resolver never saw the anchor results and the queued rescue/full stages effectively stayed at default `dt`.
- The repaired collection therefore tells us two things simultaneously:
  - what the default-`dt` matrix looks like
  - whether that matrix is actually valid as the intended paper benchmark

3. Interpretation:
- On the provisional default-`dt` matrix, `generic_sparse` is clearly the strongest overall baseline.
- Dense LISTA is the closest LISTA-family competitor by median `H1000`, but it is less robust and has more catastrophic systems.
- The benchmark is **not yet paper-final** because more than half the systems (`16/29`) should have been rerun at smaller `dt` according to the agreed rescue rule.

4. Project implications:
- We now have a useful default-difficulty audit, but not the final paper benchmark.
- The repaired collector confirms that step size really is a major difficulty knob across this system set; leaving everything at default `dt` overstates task difficulty for many systems.
- The current `v3` full matrix should be treated as provisional evidence only and should not be the version cited in the paper.

5. Next steps:
- Relaunch the canonical paper benchmark with the repaired collector in the loop so `dt` rescue uses real anchor data.
- Use the recomputed anchor/default collect as the source of truth for the pass-1 rescue task table.
- Patch or backfill the missing dense-LISTA `multiwell_strong_transition / seed_2` evaluation artifact if we want the provisional `v3` matrix to remain analyzable while the corrected rerun is executing.

### L) Canonical Research-Paper Benchmark Lock-In + Queue Launch (29 Systems, 4 Baselines, `TS=256`, `L=8`, `50k`)
Timestamp: 2026-03-06
Status: **in progress**

1. Concrete results:
- Implemented the canonical paper-benchmark scaffold:
  - manifest module: `29` systems, `4` baselines
  - locked training defaults: `target_size=256`, `sequence_length=8`, `batch_size=256`, `num_steps=50000`, `seeds={0,1,2}`
  - locked baselines: `generic_sparse`, `lista_dense`, `lista_diagonal`, `lista_blockdiag`
- Added explicit environment-step control for the benchmark:
  - `tools/train.py` now accepts `--env_dt`
  - built-in envs and `dysts` persist the resolved `dt` into `config.json`
  - `blended` now reads `cfg.ENV.BLENDED.DT` instead of a hard-coded step size
- Implemented benchmark automation:
  - `tools/build_paper_benchmark_tasks.py`
  - `tools/resolve_paper_benchmark_dt.py`
  - `tools/summarize_paper_benchmark_results.py`
  - `scripts/run_paper_benchmark_array.sh`
  - `scripts/collect_paper_benchmark.sh`
  - `scripts/resolve_paper_benchmark_dt.sh`
  - `scripts/compare_paper_benchmark.sh`
  - `scripts/queue_paper_benchmark_chain.sh`
- Validation results:
  - focused validation suite passed: `28 passed in 139.88s`
  - smoke task table count: `16`
  - anchor task table count: `87`
  - full benchmark training count (after dt resolution): `348`
  - mixed-root collector validation on existing high-dimensional roots produced `18` rows and recorded `env_dt`
  - paper-summary writer emitted paper-ready markdown/json on existing benchmark rows
  - the `dt` resolver initially failed when unresolved systems had `selected_dt=None`; that markdown bug was fixed and the resolver then wrote all expected artifacts successfully
- First queue submission (`8897520`-`8897533`) failed immediately due a Slurm path bug: the batch scripts resolved `.venv` from the temporary spool copy instead of the repo root.
- Patched all batch scripts to use `SLURM_SUBMIT_DIR`, canceled the broken chain, and resubmitted a second chain (`8897551`-`8897564`).
- The second chain reached live execution and confirmed the path fix:
  - task `8897551_15` started on `cn-c008`
  - the run executed from `/home/mila/l/lia/skae`
  - `dysts:LorenzCoupled` received the resolved `dt=0.0003241940323387382`
  - Dysts train/validation caches were both hit successfully
- Built-in smoke tasks on the second chain then failed because shell TSV parsing collapsed empty fields and corrupted `env_dt` (`'\r'` reached `train.py` for built-in rows).
- Replaced shell `read` parsing with Python-backed TSV parsing in `scripts/run_paper_benchmark_array.sh`, canceled the second chain, and resubmitted the final clean canonical chain:
  - smoke array: `8897639`
  - smoke collect: `8897640`
  - anchor array: `8897641`
  - anchor collect: `8897642`
  - resolve/rescue/full/final chain: `8897643`-`8897652`
- First live smoke results from the final clean chain:
  - `generic_sparse / duffing / seed_0 / dt=0.01` (last checkpoint):
    - `H100` best-periodic `2.5783e-04`
    - `H500` best-periodic `4.6018e-03`
    - `H1000` best-periodic `1.5665e-02`
    - `H1000` every-step `1.2278`
  - `lista_blockdiag / multiwell_rotational / seed_0 / dt=0.02` (last checkpoint):
    - `H100` best-periodic `3.0264e-02`
    - `H500` best-periodic `1.7808e-01`
    - `H1000` best-periodic `2.3307e-01`
    - `H1000` every-step `8.0212e-01`
- Canonical output roots for the paper benchmark:
  - `/network/scratch/l/lia/skae/paper_benchmark_20260306_paper_final_ts256_50k_v3`
  - `results/paper_benchmark_20260306_paper_final_ts256_50k_v3`

2. Context:
- This is the benchmark suite intended for the actual research paper. It replaces the earlier subset-only view (`duffing`, intrinsic-HD only, or ad hoc Dysts subsets) with a single reproducible queue that covers:
  - low-dimensional built-ins
  - high-dimensional built-ins
  - chaotic multi-basin Dysts systems
- The difficulty knob is now explicit and controlled:
  - keep architecture/training recipe fixed
  - reduce environment integration `dt` only when the anchor `generic_sparse` median `H1000` every-step per-dim is poor
  - use at most two halvings (`dt`, `dt/2`, `dt/4`)
- The benchmark reports both requested evaluation views:
  - in-time prediction: every-step MSE
  - forecasting: best-periodic / no-reencode evaluation from the standardized suite

3. Interpretation:
- The paper benchmark definition is now fixed and reproducible; we no longer need to manually stitch together separate launchers, collectors, or per-system `dt` overrides.
- The first two failures were purely infrastructure-related and are now corrected; they do not invalidate the benchmark specification itself.
- The final clean chain is now producing valid smoke-stage training and standardized evaluation artifacts on both built-in and Dysts-backed systems.
- The remaining blocker is benchmark wall-clock time on cluster, not missing experimental machinery or launcher correctness.

4. Project implications:
- These are now the experiments that should go into the actual research paper.
- Earlier subset benchmarks remain useful for diagnosis, but they should not be treated as the final paper evidence once this canonical run completes.
- All future cross-system model claims should be grounded in the artifacts produced under:
  - `/network/scratch/l/lia/skae/paper_benchmark_20260306_paper_final_ts256_50k_v3`
  - `results/paper_benchmark_20260306_paper_final_ts256_50k_v3`

5. Next steps:
- Let the corrected smoke stage run and verify the first collected artifacts under `results/paper_benchmark_20260306_paper_final_ts256_50k_v3/smoke_collect`.
- Let the anchor stage resolve per-system `dt` under `results/paper_benchmark_20260306_paper_final_ts256_50k_v3/dt_resolution`.
- When the full chain completes, report the final paper tables from:
  - `results/paper_benchmark_20260306_paper_final_ts256_50k_v3/final_collect/forecasting_summary.md`
  - `results/paper_benchmark_20260306_paper_final_ts256_50k_v3/final_collect/paper_benchmark_summary.md`
  - `results/paper_benchmark_20260306_paper_final_ts256_50k_v3/final_compare/`

### K) 50k Encoder Comparison: LISTA-current vs LISTA-matched vs generic_sparse (Duffing 2D, L=8, 3 Seeds)
Timestamp: 2026-03-03
Status: **completed**

1. Concrete results:
- All 9 seed runs produced complete outputs (latest eval step `49999`) under `/network/scratch/l/lia/skae/duffing_encoder_50k_20260303`.
- Consolidated summary artifacts:
  - `results/duffing_encoder_50k_20260303/duffing_encoder_50k_summary.json`
  - `results/duffing_encoder_50k_20260303/duffing_encoder_50k_summary.md`
- Aggregate mean metrics (lower is better):
  - `lista_current`: quick-best `0.4320`, `H100` `8.30e-04`, `H500` `3.13e-02`, `H1000` `1.17e-01`
  - `lista_matched`: quick-best `0.4608`, `H100` `7.83e-04`, `H500` `3.08e-02`, `H1000` `1.84e-01`
  - `generic_sparse`: quick-best `0.1115`, `H100` `7.43e-05`, `H500` `3.09e-03`, `H1000` `2.94e-02`
- `generic_sparse` is best on every seed for all reported metrics (`quick-best`, `H100`, `H500`, `H1000`).

2. Context:
- This experiment controlled two confounders from prior comparisons: longer training (`50k`) and coefficient matching (`reconst`, `sparsity`) across encoder families.
- Arms were identical except encoder family and LISTA-specific parameters (for LISTA arms), so this isolates whether loss-coefficient mismatch explained the earlier gap.

3. Interpretation:
- Matching LISTA coefficients to `generic_sparse` did not close the gap.
- Relative to `generic_sparse`, LISTA remains worse by roughly:
  - `quick-best`: `3.87x` (`lista_current`) and `4.13x` (`lista_matched`)
  - `H100`: `11.18x` and `10.55x`
  - `H500`: `10.12x` and `9.96x`
  - `H1000`: `3.97x` and `6.25x`
- `lista_matched` is not a consistent improvement over `lista_current`; it is slightly better on `H100/H500` means and clearly worse on `H1000`.

4. Project implications:
- For `duffing` under this protocol, the LISTA-vs-generic gap is not explained by the previous coefficient mismatch.
- Coefficient matching alone should not be considered a viable path to LISTA parity at 50k.
- `generic_sparse` remains the reliable long-horizon forecasting anchor while LISTA architecture/capacity changes are pursued.

5. Next steps:
- Run a controlled LISTA capacity/optimization sweep (e.g., `lista_num_loops`, hidden width, `lista_alpha` schedule) with the same `L=8`, `50k`, 3-seed protocol.
- Keep benchmark reporting in the same summary format to compare directly against this K baseline.
- After identifying a stronger LISTA candidate on `duffing`, replicate the comparison on at least one additional benchmark system.

### J) Full-Run Unstructured LISTA Pairwise-vs-Sequence Parity (Duffing 2D, 3 Seeds, 20000 Steps)
Timestamp: 2026-03-03

1. Concrete results:
- Executed full parity with `NUM_STEPS=20000` for unstructured LISTA (`config=lista_parity_generic_sparse`, `k_structure=dense`) on `duffing`, `3` seeds each for `L=1` and `L=8`.
- SLURM jobs completed with exit code `0`:
  - Sweep array: `8864953_[0-5]`
  - Dependent collector: `8864954`
- Output root:
  - `/network/scratch/l/lia/skae/duffing_lista_pairseq_full_20k_20260303`
- Consolidated summary artifacts:
  - `results/duffing_lista_pairseq_full_20k_20260303/duffing_lista_pairseq_full_20k_summary.md`
  - `results/duffing_lista_pairseq_full_20k_20260303/duffing_lista_pairseq_full_20k_summary.json`
- Aggregate metrics (mean across seeds):
  - Quick eval best (`eval/final_error`): `L=1` `0.6468`, `L=8` `0.7194`
  - H100 best-periodic: `L=1` `4.344e-03`, `L=8` `1.288e-03`
  - H500 best-periodic: `L=1` `1.115e-01`, `L=8` `4.890e-02`
  - H1000 best-periodic: `L=1` `4.825e-01`, `L=8` `2.220e-01`

2. Context:
- This run tests whether training longer than `10000` steps materially improves unstructured LISTA parity behavior and robustness.

3. Interpretation:
- Longer training improved both `L=1` and `L=8` substantially.
- `L=8` remains the stronger long-horizon choice, while `L=1` still has better quick-best short-horizon values.
- No catastrophic long-horizon seed appeared in this 20k parity run.

4. Project implications:
- Extending to `20000` steps is beneficial on duffing for this setup.
- The short-horizon vs long-horizon tradeoff remains: pairwise leads quick-best, sequence leads long-horizon forecasting.

5. Next steps:
- Tune seq-8 coefficients to improve quick-best while preserving `H500/H1000` gains.
- Validate this behavior on additional benchmark systems beyond duffing.

### I) Full-Run Unstructured LISTA Pairwise-vs-Sequence Parity (Duffing 2D, 3 Seeds, 10000 Steps)
Timestamp: 2026-03-03

1. Concrete results:
- Executed full parity with `NUM_STEPS=10000` for unstructured LISTA (`config=lista_parity_generic_sparse`, `k_structure=dense`) on `duffing`, `3` seeds each for `L=1` and `L=8`.
- SLURM jobs completed with exit code `0`:
  - Sweep array: `8864937_[0-5]`
  - Dependent collector: `8864938`
- Output root:
  - `/network/scratch/l/lia/skae/duffing_lista_pairseq_full_20260303`
- Consolidated summary artifacts:
  - `results/duffing_lista_pairseq_full_20260303/duffing_lista_pairseq_full_summary.md`
  - `results/duffing_lista_pairseq_full_20260303/duffing_lista_pairseq_full_summary.json`
- Aggregate metrics (mean across seeds):
  - Quick eval best (`eval/final_error`): `L=1` `1.0058`, `L=8` `1.1251`
  - H100 best-periodic: `L=1` `2.227e-02`, `L=8` `1.554e-02`
  - H500 best-periodic: `L=1` `2.9644`, `L=8` `3.971e-01`
  - H1000 best-periodic: `L=1` `5.616e+04`, `L=8` `8.172e-01`

2. Context:
- This is the full-length confirmation step after the quick (`3000`-step) unstructured LISTA parity gate.

3. Interpretation:
- `L=8` keeps the long-horizon advantage and has better seed robustness at H1000.
- `L=1` still has better quick-best on average and produced one catastrophic long-horizon outlier.

4. Project implications:
- Sequence-training benefit transfers to unstructured LISTA for long-horizon forecasting at full length on duffing.
- Quick-best short-horizon parity is still unresolved for `L=8`.

5. Next steps:
- Test whether longer training helps both modes (done in 20k follow-up).
- Continue with seq-8 coefficient tuning for short-horizon improvement.

### G) Full-Run Generic Sparse Pairwise-vs-Sequence Parity (Duffing 2D, 3 Seeds)
Timestamp: 2026-03-03

1. Concrete results:
- Executed full `10000`-step training + standardized eval for `generic_sparse` on `duffing`, `3` seeds each for:
  - Pairwise-equivalent: `L=1`
  - Sequence: `L=8`
- SLURM jobs (duffing-only task index) completed with exit code `0`:
  - `L=1`: `8864584` (`seed=0`), `8864585` (`seed=1`), `8864586` (`seed=2`)
  - `L=8`: `8864587` (`seed=0`), `8864588` (`seed=1`), `8864589` (`seed=2`)
- Output root:
  - `/network/scratch/l/lia/skae/duffing_generic_pairseq_full_20260303`
- Consolidated summary artifacts:
  - `results/duffing_generic_pairseq_full_20260303/duffing_generic_pairseq_full_summary.json`
  - `results/duffing_generic_pairseq_full_20260303/duffing_generic_pairseq_full_summary.md`
- Aggregate metrics (mean across seeds):
  - Quick eval best (`eval/final_error`): `L=1` `0.1223`, `L=8` `0.1309`
  - H100 best-periodic: `L=1` `4.73e-04`, `L=8` `1.21e-04`
  - H500 best-periodic: `L=1` `1.49e-02`, `L=8` `3.72e-03`
  - H1000 best-periodic: `L=1` `1.23e-01`, `L=8` `4.09e-02`

2. Context:
- This is the requested full-run follow-up to validate whether sequence-mode behavior after the loss refactor matches or diverges from pairwise on the same model family (`generic_sparse`).

3. Interpretation:
- Sequence training (`L=8`) is not collapsing on this benchmark; it is substantially better for long-horizon periodic forecasting on all three seeds.
- Pairwise (`L=1`) remains slightly better on short-horizon quick eval on average, with seed-dependent variation.

4. Project implications:
- For `generic_sparse` on duffing, the dominant gap is now short-horizon parity, not long-horizon instability.
- Sequence mode appears viable as the long-horizon setting if we can tighten short-horizon quick-eval behavior.

5. Next steps:
- Run a small seq-8 coefficient sweep around the current anchor (`reconst=0.03`, `pred=1.0`, `sparsity=0.0025`) to improve quick-eval while monitoring H500/H1000.
- Keep `L=1` as the short-horizon anchor and reject seq-8 settings that regress long-horizon best-periodic.

## Archived Core Logs

Detailed logs for superseded or intermediate milestones were moved from this core file to the archive on March 3, 2026:
- `H` (quick LISTA parity gate),
- `F` through `A` (post-refactor and iterative seq-8 tuning history),
- `0` through `6` (infrastructure, transfer, spectral, support, LQR, and benchmark-integration logs).

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
