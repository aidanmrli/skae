# Experiments (Core)

Date: March 11, 2026

## Current Status Summary

Paper-track directive:
- `docs/PAPER_TRACK_STATUS.md` is the high-level source of truth for paper-facing claims, wrap-up priorities, and remaining blockers.
- This file remains the detailed experiment ledger that backs the paper-track view.
- The active paper follow-up execution plan is now coordinated in [docs/planning/paper_parallel_workstreams_20260309.md](/home/mila/l/lia/skae/docs/planning/paper_parallel_workstreams_20260309.md). Treat that file as the current source of truth for the next batch of paper-strengthening experiments and per-agent status logs.

Wrap-up objective:
- We are actively pushing to wrap up the project and convert the current evidence into a publishable top-tier machine learning conference paper, with NeurIPS as the default target venue.

Problem we are solving:
- Learn sparse, basin-discriminative latent supports with stable long-horizon Koopman rollouts, so each support-defined regime can be used for local linear control.

Assumption split:
- Training/deployment target: basin count and basin labels are unknown.
- Benchmark evaluation: known basin counts/labels are allowed for diagnostics.

What we now know (high-confidence):
- Basin-support uniqueness is achievable at sufficient latent capacity (typically `target_size >= 256`), and cosine-based diagnostics are the reliable primary metric.
- The broad labelable-system `v4` support audit is now complete under `results/paper_benchmark_support_alignment_20260311_v4_labelable`: binary mode-support uniqueness saturates (`44/44` system-root medians have `mode_uniqueness_rate=1.0`), so it is not a useful cross-system paper metric. Cosine separation remains informative: multiwell systems are positive, Duffing and Kuramoto are negative, and Hopfield shows continuous separation without stable discrete support reuse.
- The direct Kuramoto unique-mode-support audit is now **complete** under `results/kuramoto_mode_support_audit_20260310/` with summary at `/network/scratch/l/lia/skae/kuramoto_mode_support_audit_20260310/summary/kuramoto_mode_support_audit_summary.md`. All `30/30` array tasks completed, collector `8922717` completed, balanced probe `8922718_1` timed out (non-critical: the main array covers both `random` and `balanced` sampling). **Result: the strong negative claim holds.** Mode-support uniqueness is trivially degenerate across all 3 model families, all 5 seeds, both sampling protocols, all support definitions, and all thresholds tested. Every trajectory has its own unique support (`traj_unique_rate=1.0`), mode supports are singletons (`all_mode_counts_ge_2=no`), basin consistency is negligible (`0.0625` balanced / `0.309` random), and Hamming geometry is flat (`ratio≈1.0`). Kuramoto winding-number basins do not have meaningful basin-specific support patterns under any tested model family or evaluation protocol.
- Long-horizon behavior tracks spectral radius: `SR < 1` is generally bounded; `SR > 1` is generally unstable.
- New paper-blocking fairness requirement (March 11): every paper-facing experiment that uses `lista_blockdiag` must also include a matched `generic_sparse + block_diagonal K` control. Current block-diagonal comparisons are still useful as end-to-end model results, but they do not isolate Koopman structure because `GenericKM` currently forces dense `K`. Execution plan: [docs/planning/generic_sparse_blockdiag_fairness_plan_20260311.md](/home/mila/l/lia/skae/docs/planning/generic_sparse_blockdiag_fairness_plan_20260311.md).
- Diagonal policy (March 11): retire `lista_diagonal` from active paper scope. Keep the existing diagonal-K results only as historical context; do not spend more queue budget on diagonal reruns, and do not include diagonal in future paper-facing rebuilds.
- A targeted Hopfield mechanism sweep is now live on SLURM under `results/hopfield_basin_sweep_n64_dt00625_200k_20260310/` with wrapper `8922497`, array `8922498_[0-44]`, collector `8922499`, and comparison `8922500`; as of the latest SLURM check, all `45/45` array tasks are running. This sweep tests whether increasing stored-pattern count can create a regime where dense LISTA and block-diagonal LISTA beat the MLP baseline on periodic-reencoding forecasting. This is a system-design probe, not a replacement for the canonical paper benchmark.
- The canonical **research-paper benchmark** uses `target_size=256`, `sequence_length=8`, `batch_size=256`, `seeds={0,1,2}` across `29` systems. The **primary paper results** are at `num_steps=200000` (`results/paper_followup_recipes_200k_20260309`), with `generic_sparse_ns200k_best` as the anchor. The earlier `50k` `v4` matrix is retained as a historical symmetric four-model audit.
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
- The promoted dense-LISTA full `29`-system rerun is now complete under `results/dense_lista_paper_rerun_stage4_20260309`:
  - same dense LISTA architecture, promoted Stage-3 recipe, and benchmark-selected pass-2 `dt`
  - compared against the fixed `generic_sparse` `v4` anchor, dense LISTA wins `21/29` shared systems with median dense/generic ratio `0.6455`
  - cross-system median `H1000` best-periodic improves from `0.0328` to `0.0232`
  - good-system count improves from `25/29` to `26/29`
  - there are `0` systems where the promoted dense recipe fails the good-forecast band while `generic_sparse` passes
  - the remaining dense failures are still concentrated on the hard systems, especially `kuramoto` (`48.50`), `hopfield` (`1.578e+06`), and `multiwell_strong_transition_hd` (`4.533e+04`)
  - interpretation:
    - the dense thread is no longer blocked by pending reruns; the fixed-recipe parity claim is now backed by a full-benchmark follow-up
    - this result materially strengthens the dense-LISTA paper story, but it does not replace the symmetric `v4` matrix as the canonical all-model benchmark because only the dense arm changed
- The paper follow-up recipe rerun is now complete under `results/paper_followup_recipes_200k_20260309`:
  - concrete result:
    - the full `200k` collector and all comparison artifacts were generated on March 9, 2026, so the `50k` vs `200k` fairness hole is now closed
    - `generic_sparse_ns200k_best` is the best full-benchmark root by cross-system median `H1000` best-periodic (`0.0208`), ahead of promoted dense Stage 4 (`0.0232`) and canonical `v4` `generic_sparse` (`0.0328`)
    - versus canonical `generic_sparse`, `generic_sparse_ns200k_best` wins `19/29` systems with median ratio `0.8323`, but the good-system count stays `25/29` and catastrophic systems rise from `2` to `3`
    - promoted dense Stage 4 still wins `18/29` systems versus `generic_sparse_ns200k_best`, keeps the better good-system count (`26/29` vs `25/29`), and has `0` systems where dense fails while the fair `200k` `generic_sparse` rerun passes
    - the dense-optimizer block-diagonal follow-ups do not recover global parity: `lista_blockdiag_ns200k_denseopt_sc3em3` wins `7/29` vs `generic_sparse_ns200k_best` with median ratio `4.0114`, and `lista_blockdiag_ns200k_denseopt_sc6em3` wins `5/29` with median ratio `3.0644`
  - interpretation:
    - `generic_sparse` remains the overall paper anchor once matched at `200k`
    - dense LISTA remains the strongest cross-system LISTA result, but the paper story now has to separate “best overall median” from “wins more systems / keeps more systems in-band”
    - dense-optimizer transfer to block-diagonal LISTA is a negative global result even though it repairs a small number of hard transition systems
- The repaired `dt` resolution is now complete through pass `2`: `15/29` systems accept default `dt`, `4/29` accept after at least one halving, and `10/29` remain `integration_hard`. The queueing blocker is gone, but the step-size/integration bottleneck remains real.
- The hardest intrinsic-HD systems are still the main blockers even at the selected smaller `dt=0.0125`:
  - `kuramoto`: `generic_sparse=65.70`, `lista_blockdiag=14.26`, dense LISTA=`35.23` (`H1000` system-median best-periodic)
  - `hopfield`: `generic_sparse=199.50`, `lista_blockdiag=280.42`, dense LISTA=`7.241e+09`
- For the current LISTA-family recovery phase (`lista_dense`, `lista_blockdiag`, HyperLISTA), the primary optimization target is long-horizon forecasting robustness; sparsity/support calibration is a secondary diagnostic until forecasting improves.
- Dense LISTA remains the strongest cross-system LISTA reference, but the fair `200k` rerun shows it no longer has the best overall median once `generic_sparse` is matched at `200k`; `lista_blockdiag` remains the only LISTA-family candidate worth carrying forward on the hardest intrinsic-HD systems.
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
- The Kuramoto dimension sweep is now complete under `results/kuramoto_dimension_sweep_dt00625_200k_20260309`:
  - dimensions: `N in {8,16,24,32,64}`
  - models: `generic_sparse`, promoted dense LISTA, `lista_blockdiag`
  - fixed setting: `dt=0.00625`, `num_steps=200000`, `seeds={0,1,2,3,4}`
  - `H1000` seed-median best-periodic by `N`:
    - `generic_sparse`: `813.57`, `30.18`, `6.71`, `6.68`, `208.93`
    - promoted dense LISTA: `495.07`, `13.44`, `15.01`, `92.28`, `208.71`
    - `lista_blockdiag`: `8.11`, `7.07`, `6.57`, `5.92`, `23.27`
  - `lista_blockdiag` is all-seeds-good at `N=16/24/32`, median-good but not fully robust at `N=8` (`4/5` good seeds), and no longer in-band at `N=64` (`2/5` good seeds)
  - interpretation:
    - the smaller-`dt`, longer-training Kuramoto rescue is now a moderate-dimension block-diagonal result with an explicit scaling limit at `N=64`
    - the promoted dense-LISTA recipe does not transfer as a robust Kuramoto solution
- A diagnostic recollection from `evaluation_results_last.json` shows checkpoint-selection misalignment on Kuramoto:
  - for `lista_blockdiag`, `dt=0.0125`, system-median `H1000` best-periodic improves from `23.40` to `14.64` when switching from the validation-selected checkpoint to the last checkpoint across the focused pilot grid
  - on the winning `lista_blockdiag`, `dt=0.0125`, `sp=0.0005` arm, the last-checkpoint median is `13.91`
  - this is a diagnostic, not yet the official paper metric, but it shows late training can still help on Kuramoto
- **More seeds on headline positives (Subagent D, completed March 10):**
  - low-dimensional `5`-seed extension confirms the prior `3`-seed headline: `generic_sparse_ns200k_best` median `H1000` best-periodic `0.0233`, `lista_dense_promoted_stage4` median `0.0232`, dense wins `18/29` systems, dense keeps `26/29` good vs `25/29` for generic
  - Kuramoto `N=16` `7`-seed extension strengthens the block-diagonal headline: `lista_blockdiag` seed-median `6.98` with all `7` seeds good (range `6.82–7.20`), `generic_sparse` `30.18`, `lista_dense` `16.39`
  - Kuramoto `N=32` `5`-seed confirmation completed; see full entry below
- **Kuramoto N=32 more seeds (Subagent D, completed March 10):**
  - seeds `3,4` for `generic_sparse` and `lista_blockdiag` completed
  - `3`-seed result: `lista_blockdiag` median `H1000` best-periodic `6.00` (range `5.79–6.58`, std `0.33`), `generic_sparse` median `6.65` (range `6.62–8.09`)
  - both models have all seeds good; marginal block-diagonal edge (ratio `0.90`)
  - primary audit files: `results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/compare/lista_blockdiag_vs_generic_sparse/forecasting_comparison.md`
- **Kuramoto robustness — uniform frequency spread (Subagent E, completed March 10):**
  - Kuramoto `N=16`, `dt=0.00625`, `200k`, `5` seeds with uniform frequency spread (heterogeneous natural frequencies)
  - `lista_blockdiag`: seed-median `H1000` best-periodic `9.53` (4/5 seeds good, std `0.64`)
  - `generic_sparse`: seed-median `H1000` best-periodic `44.46` (0/5 seeds good)
  - `lista_blockdiag` achieves `4.7x` improvement (ratio `0.214`) and is the only model that passes the good-forecast band
  - every-step errors remain catastrophic for `lista_blockdiag` (`1.5e34`) — periodic re-encoding is essential
  - interpretation: the Kuramoto block-diagonal win is not a single-regime artifact; it survives frequency heterogeneity
  - primary audit files: `results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/compare/lista_blockdiag_uniform_spread_vs_generic_sparse_uniform_spread/forecasting_comparison.md`
- **Label-free clustering v1 (Subagent H, completed March 10) — methodology limitation, not a negative result:**
  - protocol: encode each trajectory → average over all timesteps → cosine normalize → k-means with oracle cluster count → score vs basin labels
  - systems evaluated: `duffing` (2 basins), `competitive_lv` (1 observed basin), `kuramoto` (5 basins)
  - results:
    - `duffing`: ARI=`0.134`, NMI=`0.104`, purity=`0.688`, linear_acc=`0.679` — all three models give **exactly identical** scores (to 15+ decimal places)
    - `competitive_lv`: ARI=`1.0` trivially (only 1 basin observed)
    - `kuramoto`: ARI≈`0` for all models (generic `0.009`, dense `-0.016`, blockdiag `-0.001`); purity=`0.60`, linear_acc=`0.625`
  - interpretation:
    - the v1 clustering protocol has three methodology limitations: (1) trajectory-mean averaging washes out per-timestep support structure that is already confirmed to exist (Duffing 2/2 unique supports, Lyapunov 13/13 at TS>=256), (2) cosine k-means in 256 dimensions with 128 points suffers from concentration of measure, and (3) only the `cosine` feature view was tested while the code supports `support` views
    - the identical Duffing scores across all three architectures confirm that the feature extraction protocol (not the encoder) is the bottleneck
    - this is a methodology limitation, not a negative result on basin-support separability — per-timestep support uniqueness is well-established by `evaluate_support_uniqueness.py`
    - **follow-up:** label-free clustering v2 (below) addresses all three limitations
  - primary audit files: `/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/summary/label_free_clustering_summary.md`
- **Label-free clustering v2 (completed March 10):**
  - revised evaluation addressing all v1 methodology limitations
  - completed on array `8919951` with collector `8919952`
  - systems: `duffing` (2 basins), `multiwell_energy/gradient/rotational` + HD variants + `strong_transition` variants (5 basins each), `kuramoto` (winding-number basins)
  - models: `generic_sparse`, `lista_dense`, `lista_blockdiag` (3 seeds each for low-dim from 50k benchmark, 5 seeds for kuramoto from 200k fine-dt runs)
  - key protocol improvements over v1:
    1. six feature views tested per checkpoint: `modal_support` (per-timestep binary support → most common per trajectory), `majority_support` (per-dim majority vote), `last_step_support` (binary support of last timestep), `last_step_cosine` (cosine-normalised last timestep), `traj_mean_support` (binarised trajectory mean), `traj_mean_cosine` (original v1 approach for comparison)
    2. `256` trajectories (up from `128`)
    3. PCA to `20` dims for continuous features before k-means (addresses concentration of measure)
    4. binary support views preserve discrete structure that trajectory-mean averaging destroys
  - hypothesis: `modal_support` and `majority_support` should substantially outperform `traj_mean_cosine` on systems where per-timestep support uniqueness is already demonstrated (Duffing: 2/2, multiwell variants: 5/5)
  - results directory: `/network/scratch/l/lia/skae/label_free_clustering_v2_20260310/`
  - task specs: `results/label_free_clustering_v2_20260310/task_specs.tsv`
  - **v2 results (completed March 10, array `8919951`, collector `8919952`):**
    - **multiwell systems (all 8 variants, 5 basins each): strong positive.** Label-free k-means on latent features recovers basin structure with high fidelity:
      - `multiwell_gradient`: mean ARI `0.976` (best view `last_step_cosine`), max `0.990`
      - `multiwell_gradient_hd`: mean ARI `0.991`, max `1.000` (perfect recovery on multiple seeds)
      - `multiwell_rotational`: mean ARI `0.963`, max `0.981`
      - `multiwell_rotational_hd`: mean ARI `0.971`, max `0.981`
      - `multiwell_energy`: mean ARI `0.794`, max `0.960`
      - `multiwell_energy_hd`: mean ARI `0.916`, max `0.990`
      - `multiwell_strong_transition`: mean ARI `0.931`, max `0.982`
      - `multiwell_strong_transition_hd`: mean ARI `0.918`, max `0.992`
      - `generic_sparse` tends to have the highest ARI on most multiwell systems; LISTA families are close behind
    - **duffing (2 basins): weak positive.** ARI ~`0.19–0.24` across all views, all families similar. Above chance but not strong. `majority_support` view is slightly best (mean `0.239`). Root cause: the 2/2 unique per-timestep mode supports have very low within-basin consistency (basin 0: 12.8%, basin 1: 7.5%), meaning ~90% of trajectories in each basin activate a different support than their basin's mode. The encoder learned basin-discriminative continuous representations (linear accuracy ~0.84) but not basin-aligned sparse supports on this system.
    - **kuramoto (5 basins): negative.** ARI ~`0` across all views and families (mean `-0.001` to `+0.009`). Kuramoto basin structure is not recoverable from latent features under any tested feature extraction protocol. Deeper analysis:
      - winding-number basin distribution is extremely imbalanced: q=0 has 150/256 (58.6%), q=±2 have only 1–2 samples
      - supports are genuinely non-separable: within-basin Hamming distance (99.5 bits) ≈ between-basin distance (100.0 bits), ratio 1.004
      - every trajectory has a unique support pattern (256/256 unique), with ~93–94 active dims per basin (identical across all 5 basins)
      - purity = 0.5859 = 150/256 (majority class fraction), constant across all evaluations — k-means assigns clusters randomly w.r.t. basin structure
      - **linear accuracy bug (fixed):** the v2 `compute_linear_accuracy` reported ~0.92–0.99 on support views for Kuramoto, but this was a measurement artifact — singleton basins (q=±2) caused a fallback to train accuracy with no CV, and logistic regression memorizes when p=n=256. Corrected 3-class CV on well-populated basins (q=-1,0,+1) gives accuracy `0.427`, below majority baseline `0.593`. Fix applied: singleton classes are now dropped before CV in `evaluate_label_free_clustering_v2.py`.
    - **best feature view by system class:** `last_step_cosine` is strongest on multiwell systems; `majority_support` and `modal_support` are competitive but not clearly better than continuous cosine views after PCA reduction; `traj_mean_cosine` (v1 baseline) is comparable to discrete views on most systems, suggesting the v1 failure was primarily driven by concentration of measure (no PCA) rather than the averaging itself
    - **interpretation:**
      - the core basin-support story is validated on multiwell systems (8/8 strong, ARI 0.71–1.00) — label-free basin recovery from latent features is possible without training-time basin labels
      - duffing's weak result reflects low within-basin support consistency (~10%), not a feature extraction issue — the encoder has not achieved basin-support alignment on this system despite having 2/2 unique mode supports
      - the kuramoto negative is genuine: supports carry zero basin-discriminative signal (flat Hamming geometry), the encoder produces unique supports per trajectory with ~90 active dims, and the high linear accuracy previously reported was a measurement artifact
      - **paper impact:** the multiwell positives upgrade the basin-support claim from "per-timestep support uniqueness" to "label-free basin recovery." The kuramoto and duffing negatives limit this claim to potential-well systems with well-separated energy landscapes. The duffing result honestly demonstrates that per-timestep support uniqueness does not guarantee trajectory-level basin-support alignment.
    - summary audit: `/network/scratch/l/lia/skae/label_free_clustering_v2_20260310/summary/label_free_clustering_v2_summary.md`
- **Broad support-alignment audit on labelable `v4` benchmark systems (completed March 11, local):**
  - output root: `results/paper_benchmark_support_alignment_20260311_v4_labelable`
  - protocol:
    - canonical `v4` benchmark checkpoints only (`11` currently valid labelable systems x `4` roots x `3` seeds = `132` checkpoints)
    - systems: `duffing`, the `8` `multiwell*` variants, `kuramoto`, `hopfield`
    - excluded: `competitive_lv` from canonical `v4`, because those checkpoints used the invalidated 1-basin configuration
    - evaluation settings: `100` trajectories, length `500`, `5000`-step post-rollout basin identification, `support_threshold=1e-3`, `support_mode=mean`, cosine aggregation=`mean`
  - concrete result:
    - binary mode-support uniqueness saturates completely: **all `44/44` system-root medians have `mode_uniqueness_rate=1.0`**
    - support repetition is weak almost everywhere: **`40/44` system-root medians have `mean_basin_consistency < 0.2`**
    - trajectory-level supports are usually unique: **`24/44` system-root medians have `trajectory_unique_support_rate = 1.0`**
    - all `8` multiwell variants are positive by cosine separation across all roots (system-root medians `0.250` to `0.706`)
    - Duffing is negative across all roots (`-0.129` to `-0.084`) despite `mode_uniqueness_rate=1.0`
    - Kuramoto is negative across all roots (`-0.307` to `-0.264`) despite `mode_uniqueness_rate=1.0`; the random `100`-trajectory audit again hits the same singleton-basin issue (`q=±2` appear once each), so the apparent `mean_basin_consistency≈0.424` is inflated and not evidence of reusable basin supports
    - Hopfield is mixed: cosine separation is positive across all roots (`0.459` to `0.607`), but `mean_basin_consistency` is only `0.043` for every root and `trajectory_unique_support_rate=1.0` throughout
  - interpretation:
    - the literal binary question "does each basin have a unique mode support?" is **too weak** as a broad benchmark diagnostic, because it returns a perfect score even on known negatives like Kuramoto and Duffing
    - cosine separation reproduces the real system split and should remain the primary support-alignment metric
    - Hopfield currently has **continuous basin separation without reusable discrete support signatures**: the centroids separate, but trajectories do not reuse a stable sparse support per basin
  - project implications:
    - do **not** make a paper claim based on benchmark-wide `mode_uniqueness_rate`
    - keep the strong support story tied to multiwell cosine/clustering positives, not to a global "unique supports everywhere" statement
    - treat Duffing and Kuramoto as genuine support-alignment negatives under the current encoder settings
  - suggested next steps:
    - keep the queued rescued-Kuramoto mode-support audit as the direct test for the stronger claim on the `dt=0.00625`, `200k` checkpoints
    - after the `competitive_lv` retrain is clean, rerun this same audit on the new 4-basin checkpoints
- **Kuramoto unique mode-support audit (completed March 10):**
  - output roots: `results/kuramoto_mode_support_audit_20260310/` and `/network/scratch/l/lia/skae/kuramoto_mode_support_audit_20260310/`
  - jobs: full array `8922716_[0-29]` (all `30/30` completed), collector `8922717` (completed), balanced probe `8922718_1` (timed out at 20 min wall-clock — non-critical)
  - pre-queue QA: three infrastructure bugs were fixed before clean submission (negative raw-label CSV parsing, empty-field TSV parsing, CRLF leakage)
  - experiment setup:
    - checkpoints: rescued Kuramoto `N=16`, `dt=0.00625`, `200k`
    - roots: `generic_sparse`, `lista_dense`, `lista_blockdiag`
    - seeds `0..4`, sampling `{random, balanced}`, support modes `{mean, majority, modal}`, threshold sweeps on `{mean, modal}`
  - concrete result:
    - **strong negative — degenerate uniqueness across all conditions:**
    - `mode_uniqueness = 5/5` for every family × sampling × seed → technically every basin has a "unique" mode support, but this is vacuous
    - `traj_unique_rate = 1.0` everywhere → every single trajectory has its own unique support pattern
    - `all_mode_counts_ge_2 = no` everywhere → mode supports are singletons, not reused within basins
    - `hamming_ratio ≈ 1.0` everywhere → within-basin vs between-basin Hamming distance is flat (no geometric structure)
    - `basin_consistency = 0.0625` (balanced) / `0.309` (random) → negligible reuse
    - `full_unique_nontrivial_seeds = 0` everywhere → zero seeds achieve nontrivial (non-singleton) unique supports
    - results are **identical across all 3 model families** — the encoder architecture makes no difference
    - threshold sweeps (mean & modal modes, thresholds `1e-4` to `1e-1`) do not change the conclusion
  - interpretation:
    - the strong negative claim is now directly confirmed: Kuramoto winding-number basins do not have meaningful basin-specific support patterns under any tested model family, sampling protocol, support definition, or threshold
    - uniqueness is trivially degenerate — each basin has a "unique" mode support only because essentially every trajectory activates a different support pattern; supports are not reusable within basins
    - this closes the gap left by label-free clustering v2, which only established non-recoverability but did not directly test literal mode-support uniqueness
    - the balanced probe timed out but is non-critical: the main array already includes both `random` and `balanced` sampling conditions with identical conclusions
  - primary audit file: `/network/scratch/l/lia/skae/kuramoto_mode_support_audit_20260310/summary/kuramoto_mode_support_audit_summary.md`
- **Competitive LV multi-basin retrain (in progress March 10):**
  - default `INTERACTION_SCALE` changed from `0.35` to `0.70` in `skae/config.py` to produce 4 major basins instead of 1 trivial basin
  - basin structure at scale=0.70: species 1, 5, 6 are fragile under competition; basins differ by which subset goes extinct
  - retraining ALL paper-facing competitive_lv experiments (28 tasks):
    - Group A: v4 benchmark (generic_sparse, lista_dense, lista_blockdiag, lista_diagonal × seeds 0-2 × 50k)
    - Group B: 200k followup (generic_sparse_ns200k_best, lista_blockdiag_ns200k_denseopt_sc3em3, lista_blockdiag_ns200k_denseopt_sc6em3 × seeds 0-2 × 200k)
    - Group C: promoted dense Stage 4 (lista_dense_promoted_stage4 × seeds 0-2 × 200k)
    - Group D: more seeds (generic_sparse_ns200k_best + lista_dense_promoted_stage4 × seeds 3,4 × 200k)
  - forward-looking scope change: `lista_diagonal` is now retired from active paper scope. The clean rerun should exclude diagonal and focus on `generic_sparse`, `lista_dense`, `lista_blockdiag`, and the matched `generic_sparse + block_diagonal K` control once it exists.
  - first retrain attempt array: `8922033`
  - current status: the first attempt finished in a mixed state. `17/28` rows completed and `11/28` rows failed. `generic_sparse` and `lista_blockdiag` rows ran, while `lista_dense` and `lista_diagonal` rows failed because empty optional TSV columns in `scripts/run_competitive_lv_retrain_array.sh` are shifting arguments (for example `--k_block_size` receives `0.15` on dense/diagonal rows). No clean rerun is queued yet.
  - task specs: `results/competitive_lv_multibas_retrain_20260310/task_specs.tsv`
  - results: `/network/scratch/l/lia/skae/competitive_lv_multibas_retrain_20260310/`
  - scripts: `scripts/queue_competitive_lv_retrain.sh`, `scripts/run_competitive_lv_retrain_array.sh`
  - **After a clean retrain completes:** re-run support alignment (Subagent A) and label-free clustering on the new 4-basin checkpoints. The old competitive_lv evaluations are ALL INVALID (ran on 1-basin data).
- **Checkpoint-selection ablation (Subagent G, completed offline March 9):**
  - the current `dt=0.00625`, `200k` Kuramoto comparison is checkpoint-selection-stable: switching to `evaluation_results_last.json` changes `lista_blockdiag` from `6.98 -> 7.00`, `lista_dense` from `13.84 -> 17.63`, `generic_sparse` from `27.02 -> 29.43`
  - ranking and good-band membership are unchanged under both selection rules
  - the older `dt=0.0125`, `20k` pilot mismatch reproduces (`lista_blockdiag` aggregated: `23.40 -> 14.64`; winning arm: `14.36 -> 13.91`) but is tied to the earlier setting, not the current headline result
  - decision: keep `evaluation_results_best.json` as the official paper rule; mention the older pilot mismatch as a diagnostic limitation
- **Fixed-cadence re-encoding ablation (Subagent C, completed offline March 9):**
  - dense benchmark under fixed `periodic_100`: wins `17/29` overall (`13/23` non-intrinsic-HD), but good-system count falls to tie (`22/29` vs `22/29`) and dense newly fails `blended` while the fair anchor passes; the win-count advantage survives but the safety margin does not
  - Kuramoto under fixed `periodic_100`: exactly reproduces the official `best_periodic` `H1000` ranking (`6.98`, `13.84`, `27.02`) with `periodic_100/best_periodic` ratio of `1.0` for all three roots; the long-horizon Kuramoto claim does not depend on horizon-wise cadence selection
  - at `H500`, `generic_sparse` still has lower error than `lista_blockdiag` under the fixed cadence; the Kuramoto positive is specifically an `H1000` forecasting result
  - sensitivity: `periodic_100` is the strongest of the saved fixed cadences for both stories; shorter cadences (`periodic_50/25/10`) progressively weaken or break both the dense benchmark and Kuramoto claims
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
- **The `200k` follow-up (`results/paper_followup_recipes_200k_20260309`) is the primary paper evidence base.** Use `generic_sparse_ns200k_best` as the main anchor and `lista_dense_promoted_stage4` as the main dense LISTA comparator. Treat the `v4` `50k` matrix as a symmetric historical audit, not the source of headline claims.
- Keep `L=8` as the default training mode for forecasting-facing comparisons.
- Use dense LISTA as the cross-system LISTA reference, and reserve `lista_blockdiag` for targeted hard-system claims rather than global baseline claims.
- Retire `lista_diagonal` from active paper scope. Keep its existing benchmark numbers only as historical negatives; do not schedule new diagonal reruns.
- For hard systems, prefer smaller `dt` before more architecture churn. The current decision-grade evidence is the repaired focused intrinsic-HD rerun plus the completed `dt=0.00625`, `200k` Kuramoto/Hopfield follow-ups and the Kuramoto dimension sweep.
- Present the dense result as a split-metric story: `generic_sparse_ns200k_best` has the best full-benchmark median, while promoted dense Stage 4 wins more systems and keeps more systems in-band.
- Present the basin-support story as system-dependent: strong label-free recovery on multiwell, weak on Duffing, negative on Kuramoto, and currently invalidated on `competitive_lv` until the new 4-basin retrain is clean.
- Keep `evaluation_results_best.json` as the official checkpoint rule; use `evaluation_results_last.json` only as a diagnostic.
- Do not spend more queue budget on broad reruns by default, except for the new `generic_sparse + block_diagonal K` fairness control. The remaining paper-track work is to close the `competitive_lv` blocker, land the block-diagonal fairness fix, and sharpen the limitation framing around hard-system autonomous rollout stability.

Outstanding problem:
- The remaining wrap-up task is to present a clean paper story that separates the real positives from the real limits: dense LISTA is now a credible cross-system comparator, `lista_blockdiag` has a targeted Kuramoto rescue through `N=32`, and label-free basin recovery works well on multiwell systems, but `generic_sparse` still owns the best median and in-time accuracy, Hopfield remains MLP-better, Kuramoto breaks by `N=64`, autonomous rollouts remain unstable, the `competitive_lv` paper evidence is currently being rebuilt, and the missing `generic_sparse + block_diagonal K` control still blocks clean causal claims about block-diagonal gains.

## Outstanding problems (active)

These are the active blockers for the paper-track plan in `docs/PAPER_TRACK_STATUS.md`.

- `competitive_lv` must be rebuilt on `INTERACTION_SCALE=0.70`; all old 1-basin forecasting, support-alignment, and label-free clustering results are invalid.
- The first `competitive_lv` retrain attempt `8922033` is not yet clean enough to use: `generic_sparse` and block-diagonal rows ran, dense rows failed because empty optional TSV fields shift arguments in `scripts/run_competitive_lv_retrain_array.sh`, and diagonal is now retired from active scope rather than something to rescue.
- Paper-blocking fairness gap: every paper-facing `lista_blockdiag` experiment still lacks the matched `generic_sparse + block_diagonal K` control. Implement structured `K` for `GenericKM`, mirror the block-diagonal task variants, and rerun the affected paper studies before locking causal language around block-diagonal wins.
- `generic_sparse_ns200k_best` has the best full-benchmark median (`0.0208`), while promoted dense Stage 4 wins more shared systems (`18/29`) and keeps a better good-system count (`26/29` vs `25/29`). The paper needs a crisp split-metric presentation.
- `lista_blockdiag` is a targeted Kuramoto positive, not a global paper baseline: it is robust through `N=32`, not fully robust at `N=8`, and broken by `N=64`.
- Hopfield is rescued under smaller `dt` for periodic-reencoding forecasts, but `generic_sparse` remains better and every-step autonomous errors are still enormous.
- It remains unknown whether a higher-basin Hopfield variant can flip the dense/block-diagonal LISTA ordering against the MLP without turning the task into an overloaded-memory artifact; the active `N=64`, `P in {8,10,12,14,16}` sweep is meant to answer that mechanistic question.
- Label-free basin recovery is strong on multiwell, weak on Duffing, and negative on Kuramoto. The basin-support claim must be scoped accordingly.
- Binary `mode_uniqueness_rate` should not be used as a primary cross-system paper metric. The broad labelable-system audit saturates at `1.0` even on Duffing and Kuramoto, so the paper should anchor basin-support claims on cosine separation and label-free recovery instead.
- The stronger Kuramoto claim "no unique basin mode support per basin" is now **resolved (confirmed negative)**. The completed mode-support audit directly shows that Kuramoto supports are trivially degenerate: every trajectory has its own support, mode supports are singletons, and Hamming geometry is flat. This holds across all model families, seeds, sampling protocols, support modes, and thresholds.
- Autonomous rollout stability remains the main scientific limitation on the hard systems.
- No additional broad benchmark rerun is justified by current evidence; remaining work should either retire the `competitive_lv` blocker or sharpen limitation framing.

## Queue Status

Queue work should be justified against `docs/PAPER_TRACK_STATUS.md`; prefer runs that directly retire a paper blocker or sharpen the final paper narrative.

In progress (paper-parallel workstreams, March 11):
- The immediate paper-strengthening program is coordinated in `docs/planning/paper_parallel_workstreams_20260309/`.
- **Block-diagonal fairness-control implementation/re-run:** not started. This is now required before freezing any paper claim that compares `lista_blockdiag` against `generic_sparse`. Tracking plan: [docs/planning/generic_sparse_blockdiag_fairness_plan_20260311.md](/home/mila/l/lia/skae/docs/planning/generic_sparse_blockdiag_fairness_plan_20260311.md).
- **A - Competitive LV support alignment:** invalidated. The March 9 evaluation ran on the old 1-basin `competitive_lv` setup, so all prior `competitive_lv` support-alignment, forecasting, and label-free clustering results must be replaced after a clean multi-basin retrain. Do not re-use the old checkpoints or metrics.
- **Competitive LV multi-basin retrain:** the first array attempt `8922033` finished in a mixed state under `results/competitive_lv_multibas_retrain_20260310/`: `17/28` rows completed and `11/28` rows failed. The current runner mis-parses empty optional TSV columns, so dense/diagonal rows fail while `generic_sparse` and block-diagonal rows run. No clean rerun is queued yet. The clean rerun should drop diagonal and cover `generic_sparse`, `lista_dense`, `lista_blockdiag`, and the matched `generic_sparse + block_diagonal K` control once available. After that rerun completes, re-run support alignment (Subagent A) and label-free clustering v2 on the new 4-basin checkpoints.
- **Hopfield basin-count mechanism sweep:** wrapper `8922497`, array `8922498_[0-44]`, collector `8922499`, comparison `8922500`.
  - Output roots:
    - `/network/scratch/l/lia/skae/hopfield_basin_sweep_n64_dt00625_200k_20260310`
    - `results/hopfield_basin_sweep_n64_dt00625_200k_20260310`
  - Setup:
    - Hopfield only, `N=64`, `P in {8,10,12,14,16}`, `dt=0.00625`, `200k`, `seeds={0,1,2}`
    - roots: `generic_sparse`, `lista_dense_promoted_stage4`, `lista_blockdiag_targeted`
    - dense recipe is the promoted Stage-4 optimizer setting; block-diagonal recipe is the targeted smaller-`dt` Hopfield/Kuramoto-style setting (`sp=0.001`, `alpha=0.15`, `loops=1`, block size `16`)
  - Purpose:
    - test whether increasing the number of stored Hopfield patterns can create a high-basin regime where both LISTA baselines beat the MLP anchor on periodic-reencoding forecasting
  - Interpretation rule:
    - treat this as a mechanistic environment-design follow-up, not as a new canonical benchmark result, because the system itself is being changed to probe the architecture ordering
  - Current status:
    - as of the latest SLURM check, all `45/45` array tasks are running and the collector/comparison jobs remain pending on dependency
- **Kuramoto unique mode-support audit:** complete. All `30/30` array tasks and collector finished. The strong negative claim is confirmed: mode-support uniqueness is trivially degenerate across all model families, seeds, sampling protocols, support definitions, and thresholds. Balanced probe `8922718_1` timed out but is non-critical. See experiment entry above for full results.
- **B - Kuramoto support alignment:** closed as redundant. Label-free clustering v2 already showed the same Kuramoto checkpoints are non-separable.
- **C - Fixed-cadence re-encoding ablation:** complete. The dense win-count story survives `periodic_100`, and the Kuramoto `H1000` block-diagonal win is already a fixed-cadence result.
- **D - More seeds on headline positives:** complete. The low-dimensional `5`-seed extension leaves the dense-vs-MLP narrative unchanged, and the Kuramoto `N=16` and `N=32` confirmations both support the block-diagonal headline.
- **E - Kuramoto robustness:** complete. The uniform-spread heterogeneity check remains positive for `lista_blockdiag`.
- **F - Support transition dynamics:** not started and lower priority than closing `competitive_lv`.
- **G - Kuramoto checkpoint selection:** complete. The current headline result is selection-stable.
- **H - Label-free clustering quality:** complete for all non-`competitive_lv` systems. v1 is retained only as a methodology limitation; v2 is final for multiwell, Duffing, and Kuramoto.

Most recent completed paper-track chains:
- **Local broad support-alignment audit on labelable `v4` systems (completed March 11, no queue):**
  - Output root:
    - `results/paper_benchmark_support_alignment_20260311_v4_labelable`
  - Setup:
    - canonical `v4` checkpoints for `11` currently valid labelable systems (`duffing`, `8` `multiwell*` variants, `kuramoto`, `hopfield`)
    - roots: `generic_sparse`, `lista_dense`, `lista_diagonal`, `lista_blockdiag` (historical `v4` set; diagonal is now retired from active scope)
    - `3` seeds each (`132` checkpoints total)
    - evaluation settings: `100` trajectories, length `500`, `5000` rollout steps, `support_threshold=1e-3`, `support_mode=mean`
  - Concrete result:
    - all `44/44` system-root medians have `mode_uniqueness_rate=1.0`
    - `40/44` have `mean_basin_consistency < 0.2`
    - all multiwell system-root medians have positive cosine separation (`0.250` to `0.706`)
    - Duffing is negative across all roots (`-0.129` to `-0.084`)
    - Kuramoto is negative across all roots (`-0.307` to `-0.264`)
    - Hopfield is mixed: positive cosine separation (`0.459` to `0.607`) but uniformly tiny support consistency (`0.043`)
  - Interpretation:
    - benchmark-wide binary mode-support uniqueness saturates and is not a useful discriminative paper metric
    - cosine separation matches the known qualitative story, so it remains the correct primary support-alignment diagnostic
    - Hopfield currently shows continuous basin separation without reusable sparse support signatures
- **Paper follow-up recipe rerun (`29` systems, pass-2 `dt`, `200k`, seeds `0,1,2`)**:
  - Completed submission chain:
    - queue launcher / array: `8911901_[0-260]`
    - collector: `8911902`
    - comparison vs canonical `generic_sparse`: `8911903`
    - comparison vs `generic_sparse_ns200k_best`: `8911904`
    - comparison vs promoted dense Stage-4 root: `8911905`
  - Output roots:
    - `/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309`
    - `results/paper_followup_recipes_200k_20260309`
  - Training arms:
    - `generic_sparse_ns200k_best`
    - `lista_blockdiag_ns200k_denseopt_sc6em3`
    - `lista_blockdiag_ns200k_denseopt_sc3em3`
  - Existing comparison-only root:
    - promoted dense Stage-4 root `lista_dense_promoted_stage4`
  - Concrete result:
    - all training, collection, and comparison stages are complete; the output root now contains the final summary and comparison artifacts
    - `generic_sparse_ns200k_best` is the best full-benchmark root by cross-system median `H1000` best-periodic (`0.0208`)
    - promoted dense Stage 4 still wins `18/29` shared systems against `generic_sparse_ns200k_best` and keeps a better good-system count (`26/29` vs `25/29`)
    - `lista_blockdiag_ns200k_denseopt_sc3em3` and `lista_blockdiag_ns200k_denseopt_sc6em3` remain globally behind the fair `200k` `generic_sparse` rerun (`7/29` and `5/29` wins, respectively)
  - Context:
    - this chain was designed to close the `50k` vs `200k` fairness hole for `generic_sparse` and test whether the promoted dense optimizer transfers to block-diagonal LISTA at full benchmark scale
  - Interpretation:
    - the fairness hole is now closed
    - the ranking story is now split across metrics: `generic_sparse_ns200k_best` has the best overall median, while promoted dense Stage 4 has better win-count and good-system count
    - dense-optimizer transfer to block-diagonal LISTA is a negative full-benchmark result
  - Project implications:
    - use `generic_sparse_ns200k_best` as the fair `200k` anchor in any asymmetric `200k` paper comparisons
    - do not claim that promoted dense LISTA remains better than a fair `200k` MLP anchor on the single summary statistic of cross-system median
    - keep the block-diagonal full-benchmark `200k` reruns only as targeted evidence for `multiwell_strong_transition` / `multiwell_strong_transition_hd`, not as global parity evidence
  - Suggested next steps:
    - do not queue another broad benchmark rerun by default
    - if more paper-track budget is approved, spend it on targeted hard-system clarification (`N=64` Kuramoto robustness or autonomous-rollout diagnostics), not another global parity sweep
- **Kuramoto dimension sweep (`dt=0.00625`, `200k`, `5` seeds, `N={8,16,24,32,64}`)**:
  - Completed chain:
    - queue launcher / array: `8910056_[0-74]`
    - collector: `8910057`
    - comparison: `8910061`
  - Output roots:
    - `/network/scratch/l/lia/skae/kuramoto_dimension_sweep_dt00625_200k_20260309`
    - `results/kuramoto_dimension_sweep_dt00625_200k_20260309`
  - Concrete result:
    - `lista_blockdiag` wins the Kuramoto `H1000` median against `generic_sparse` at every tested `N`
    - the rescue is robust at `N=16/24/32` (`7.07`, `6.57`, `5.92`, all seeds good)
    - `N=8` is median-good but not fully robust (`8.11`, `4/5` good seeds)
    - `N=64` breaks the rescue (`23.27`, `2/5` good seeds)
    - promoted dense LISTA does not enter the good band at any tested `N`
  - Current interpretation:
    - the Kuramoto paper story is now a moderate-dimension smaller-`dt` success with explicit scaling limits, not an open scaling question
- **Dense LISTA promoted full `29`-system rerun (`TS=256`, `L=8`, `200k`, promoted fair recipe, seeds `0,1,2`)**:
  - Completed chain:
    - queue launcher: `8909900`
    - dense LISTA array: `8909900_[0-86]`
    - collector: `8909901`
    - comparison vs fixed `generic_sparse` anchor: `8909902`
  - Output roots:
    - `/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309`
    - `results/dense_lista_paper_rerun_stage4_20260309`
  - Concrete result:
    - all `87` dense runs were collected
    - the promoted dense recipe wins `21/29` shared systems vs the fixed `generic_sparse` anchor
    - median shared-system `H1000` best-periodic ratio is `0.6455`
    - cross-system median `H1000` best-periodic improves from `0.0328` to `0.0232`
    - good-system count improves from `25/29` to `26/29`
    - there are `0` systems where dense fails while `generic_sparse` passes
    - the remaining dense failures are concentrated on `kuramoto`, `hopfield`, and `multiwell_strong_transition_hd`
  - Current interpretation:
    - the dense thread is now full-benchmark positive against the fixed `generic_sparse` anchor
    - the dense paper question is no longer whether to run this rerun; it is how to present the result without confusing it with the symmetric `v4` matrix
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
    - the dense thread is no longer blocked on recipe validation; Stage 4 has already supplied the full `29`-system follow-up, so the remaining question is paper positioning rather than more dense queue work
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
  - do not queue another broad paper-benchmark rerun unless a specific paper-positioning question cannot be answered from the existing artifacts
  - if more hard-system budget is needed after that, target `N=64` robustness or autonomous-rollout diagnostics directly instead of another broad sweep
  - avoid more coefficient-only holdout sweeps or another dense full-benchmark rerun for now

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
  - Completed queue chain: `8897639`-`8897652`
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

Current paper-facing blockers:
- `competitive_lv` is the only live queue blocker: the old 1-basin results are invalid, and the multi-basin retrain still needs a clean rerun for dense plus the new `generic_sparse + block_diagonal K` control. Diagonal is no longer in scope.
- The paper still needs a crisp split-metric presentation of `generic_sparse_ns200k_best` vs promoted dense Stage 4.
- Autonomous rollout instability remains the limiting story on the hard systems, even where periodic re-encoding succeeds.

Planned next:
- Repair or replace the current `competitive_lv` retrain attempt, then re-run support alignment and label-free clustering on the new 4-basin checkpoints.
- Keep the hard-system narrative tied to the completed smaller-`dt` Kuramoto/Hopfield follow-ups and the completed Kuramoto dimension sweep; do not open another broad rerun by default.
- Keep forecasting-first evaluation as the promotion rule for any further LISTA-family follow-up.

## Core Experiment Log (Most Informative)

### ZH) Kuramoto Dimension Sweep Completed: `lista_blockdiag` Rescues Kuramoto Through `N=32`, But the Rescue Breaks at `N=64`
Timestamp: 2026-03-09
Status: **completed**

1. Concrete results:
- The full Kuramoto dimension sweep finished, collected, and compared under `results/kuramoto_dimension_sweep_dt00625_200k_20260309`:
  - queue launcher / array: `8910056_[0-74]`
  - collector: `8910057`
  - comparison: `8910061`
- `H1000` seed-median best-periodic by dimension:
  - `generic_sparse`: `N=8` `813.5733`, `N=16` `30.1799`, `N=24` `6.7146`, `N=32` `6.6781`, `N=64` `208.9328`
  - promoted dense LISTA: `N=8` `495.0707`, `N=16` `13.4408`, `N=24` `15.0073`, `N=32` `92.2826`, `N=64` `208.7072`
  - `lista_blockdiag`: `N=8` `8.1126`, `N=16` `7.0714`, `N=24` `6.5693`, `N=32` `5.9158`, `N=64` `23.2681`
- Seed robustness:
  - `lista_blockdiag` is all-seeds-good at `N=16/24/32`
  - `lista_blockdiag` is median-good but not fully robust at `N=8` (`4/5` good seeds, worst seed `10.8898`)
  - `lista_blockdiag` is no longer in-band at `N=64` (`2/5` good seeds, worst seed `209.2029`)
  - promoted dense LISTA is out of band at every tested `N`

2. Context:
- This was the direct scaling test of the stronger Kuramoto setting that had already looked positive at `N=16` and `N=32`: `dt=0.00625`, `200k` steps, `5` seeds, and a three-way comparison among `generic_sparse`, promoted dense LISTA, and `lista_blockdiag`.
- The goal was to decide whether the smaller-`dt`, longer-training Kuramoto rescue was a genuine scaling story and whether the promoted dense-LISTA recipe transferred to the oscillator setting.

3. Interpretation:
- The Kuramoto rescue is real for `lista_blockdiag`, but only through moderate dimensions.
- The clean positive part of the story is `N=16/24/32`, where `lista_blockdiag` stays in-band and all seeds are good.
- `N=8` is not a clean robustness win despite an in-band median, and `N=64` is the clear scaling failure point.
- The promoted dense recipe does not transfer as a robust Kuramoto solution; its `N=16` improvement over `generic_sparse` is still out of band and does not persist.

4. Project implications:
- The paper no longer has to treat Kuramoto scaling as an open question.
- The defensible claim is now: smaller `dt` plus longer training gives a moderate-dimension Kuramoto rescue for block-diagonal LISTA under periodic reencoding, with explicit scaling failure by `N=64`.
- This also removes promoted dense LISTA from the hard-system positive narrative; dense remains a cross-system parity story, not the Kuramoto rescue story.

5. Next steps:
- Update the paper-track docs and queue ledger so they reflect the completed sweep and its moderate-dimension conclusion.
- Keep the active `29`-system paper follow-up rerun as the only paper-track queue priority.
- If additional hard-system work is justified later, focus on `N=64` robustness or autonomous-rollout stability rather than another broad Kuramoto sweep.

### ZG) Dense LISTA Promoted Full Rerun Completed: One Fixed Fair Recipe Beats the `generic_sparse` Anchor on `21/29` Systems
Timestamp: 2026-03-09
Status: **completed**

1. Concrete results:
- The promoted dense-LISTA full `29`-system rerun finished and collected under `results/dense_lista_paper_rerun_stage4_20260309`:
  - queue launcher: `8909900`
  - dense LISTA array: `8909900_[0-86]`
  - collector: `8909901`
  - comparison vs fixed `generic_sparse` anchor: `8909902`
- Compared against the fixed `generic_sparse` `v4` anchor:
  - dense LISTA wins `21/29` shared systems and loses `8/29`
  - median shared-system `H1000` best-periodic ratio is `0.6455`
  - cross-system median `H1000` best-periodic improves from `0.0328` to `0.0232`
  - good-system count improves from `25/29` to `26/29`
  - there are `0` systems where dense LISTA fails while `generic_sparse` passes
- The remaining dense failures are still concentrated on the hard systems:
  - `hopfield`: `1.578e+06`
  - `kuramoto`: `48.50`
  - `multiwell_strong_transition_hd`: `4.533e+04`

2. Context:
- Stage 4 was the full-benchmark follow-up after Stage 2 and Stage 3 resolved the dense-LISTA recipe question.
- The rerun kept the dense LISTA architecture fixed, used the promoted Stage-3 external recipe, reused the benchmark-selected pass-2 `dt` table, and compared only against the existing `generic_sparse` `v4` anchor.

3. Interpretation:
- The dense-LISTA parity story is now full-benchmark positive rather than only a targeted easy-system subset result.
- One fixed fair dense recipe now beats the fixed `generic_sparse` anchor on most benchmark systems overall.
- This still does not solve the hard-system story: the same intrinsic-HD failures continue to dominate the tail risk.
- Because only dense LISTA was rerun, this result should be treated as a dense-specific follow-up and not as a replacement for the symmetric `v4` all-model matrix.

4. Project implications:
- The dense thread is no longer blocked by pending reruns or by recipe-selection ambiguity.
- The paper can now make a substantially stronger dense-LISTA claim: fixed-architecture external tuning is enough to beat the fixed `generic_sparse` anchor on most systems overall.
- The remaining dense-LISTA paper risk is now narrative discipline, not missing experiments:
  - keep the Stage-4 result separate from the canonical `v4` benchmark
  - do not oversell dense LISTA on the unresolved hard systems

5. Next steps:
- Update the paper-track docs and queue ledger so they reflect that Stage 4 is complete and the paper follow-up recipe rerun is now the only active paper-track queue.
- Do not queue more dense coefficient-only sweeps or another dense full-benchmark rerun for now.
- Keep the hard-system narrative tied to the completed Kuramoto scaling read rather than opening another dense-specific branch.

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

### L) Queue 0-1 Completion: ReLU Baseline + Focused `lista_alpha` Sweep (Duffing 2D, L=8, 3 Seeds)
Timestamp: 2026-03-04
Status: **completed**

1. Concrete results:
- Queue 0 completed:
  - Sweep: `8873286_[0-2]` (**completed**)
  - Collector output was produced locally from the completed sweep artifacts.
  - Baseline aggregate: quick-best `0.4527`, `H500` `0.0506`, `H1000` `0.3992`, sparsity median `0.9571`.
- Queue 1 gate completed:
  - Sweep: `8873328_[0-14]`
  - Collector: `8873329`
  - Gate summary artifacts: `results/duffing_lista_q01_20260304/queue1_gate/duffing_lista_alpha_gate_10k_summary.{json,md}`
- Gate aggregate means (10k stage, lower better for errors):
  - `alpha_0p15`: quick-best `1.2095`, `H500` `0.1082`, `H1000` `0.3736`, sparsity median `0.9501`
  - `alpha_0p10`: quick-best `0.9123`, `H500` `0.1201`, `H1000` `0.4532`, sparsity median `0.9484`
  - `alpha_0p20`: quick-best `0.9936`, `H500` `0.1739`, `H1000` `0.5712`, sparsity median `0.9491`
  - `alpha_0p30`: quick-best `0.8347`, `H500` `0.1953`, `H1000` `0.5733`, sparsity median `0.9467`
  - `alpha_0p40`: quick-best `0.9646`, `H500` `0.1452`, `H1000` `0.5772`, sparsity median `0.9473`
- Selection used `tools/select_lista_alpha_survivors.py`; survivors were `alpha_0p15`, `alpha_0p10`, `alpha_0p20`.
- Full stage completed:
  - Launcher `8873462`, sweep `8873469_[0-8]` (**completed**)
  - Collector output was produced locally from the completed sweep artifacts.
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
