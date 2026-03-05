# Experiments (Core)

Date: March 5, 2026

## Current Status Summary

Problem we are solving:
- Learn sparse, basin-discriminative latent supports with stable long-horizon Koopman rollouts, so each support-defined regime can be used for local linear control.

Assumption split:
- Training/deployment target: basin count and basin labels are unknown.
- Benchmark evaluation: known basin counts/labels are allowed for diagnostics.

What we now know (high-confidence):
- Basin-support uniqueness is achievable at sufficient latent capacity (typically `target_size >= 256`), and cosine-based diagnostics are the reliable primary metric.
- Long-horizon behavior tracks spectral radius: `SR < 1` is generally bounded; `SR > 1` is generally unstable.
- On `duffing`, `L=8` training is superior for both `generic_sparse` and LISTA given that the model trains long enough. Use at least 20000 steps for training to draw definitive conclusions.
- On `duffing`, controlled 50k training with coefficient-matched arms (Experiment K) did **not** close the LISTA-vs-generic_sparse gap; `generic_sparse` remains best across all seeds and reported horizons.
- The intrinsic-HD `L=8`, `TARGET_SIZE=256` baseline with current env defaults (`kuramoto=16`, `hopfield=16`, `competitive_lv=10`) is now complete for `generic_sparse`, dense LISTA, and block-diagonal LISTA:
  - `competitive_lv` is solved by all three models; seed-median `H1000` best-periodic is `0.0651` (`generic_sparse`), `0.1192` (`lista_blockdiag`), `0.1654` (`lista_dense`).
  - `kuramoto` favors `generic_sparse` (`199.86`) over `lista_blockdiag` (`258.50`); dense LISTA is catastrophic (`6.636e8` median `H1000` best-periodic).
  - `hopfield` is the current blocker: `generic_sparse` is least bad (`5436.67` median `H1000` best-periodic), while `lista_blockdiag` (`3.599e15`) and dense LISTA (`3.045e33`, with one seed lacking a finite best-periodic score) are catastrophic.
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
  - no arm reached the target sparsity band `0.7–0.9` (`0/8` aggregate arms in-band; all `~1e-4` sparsity).
  - compared with Queue-4 anchor `sp_0p0060_loops_1`, this best Queue-5 arm is worse by about `2.45x` (quick), `130x` (`H500`), and `24.98x` (`H1000`).
  - compared with `generic_sparse`, it is worse by about `6.95x` (quick), `501x` (`H500`), and `78.09x` (`H1000`).
- Cross-system sparse LISTA robustness is still insufficient relative to `generic_sparse`, with catastrophic tails and frequent reliance on short-period reencoding.
- LQR decision metrics remain non-discriminative due metric saturation (`M2`, `M3`) and heavy-tailed `M4`.

Current approach:
- Keep `L=8` sequence training as the default training mode for long-horizon forecasting experiments.
- Use `generic_sparse` as the performance anchor on `duffing` while LISTA architecture/capacity changes are evaluated.
- Use the current-default intrinsic-HD sweep as a baseline only; do not yet treat it as the final stress test for the plan because the stricter `N=32/64` variants have not run.
- For intrinsic-HD follow-up, prioritize autonomous long-horizon stability on `hopfield` and robustness on `kuramoto`; `competitive_lv` is not the bottleneck at `TARGET_SIZE=256`.
- Treat coefficient matching (`reconst`, `sparsity`) as insufficient by itself for LISTA parity on `duffing`; prioritize LISTA capacity/optimization changes (e.g., loops/width/alpha schedule) for follow-up.
- Enforce **ReLU as the final operation** in LISTA going forward; re-baseline and target `sparsity_ratio ~ 0.7–0.9` as part of the LISTA forecasting improvement plan (`docs/LISTA_STACK_FORECASTING_PLAN.md`).
- Keep `lista_alpha=0.15` fixed and treat Queue-2/3 outcomes as anchors for joint tuning.
- Promote Queue-4 Pareto winners as LISTA anchors:
  - robustness-first anchor: `sp_0p0060_loops_1`
  - quick-best tie-break anchor: `sp_0p0040_loops_1`
- Treat Queue-5 HyperLISTA adaptive-threshold results as a negative result for the tested `c_theta` grid; do not promote Queue-5 arms.
- Use Queue-4 instability flags to exclude heavy-tail arms from further shortlist.
- Validate current `duffing` conclusions on additional systems/seeds before promoting defaults.
- Prioritize forecasting robustness first; only then advance basin-support deployment checks and control-facing stages.

Outstanding problem:
- Robust long-horizon transfer is still failing on harder non-diagonal systems, now concretely on intrinsic-HD `hopfield` and dense-LISTA `kuramoto`; we do not yet have a sparse model family that is both stable and broadly transferable across these regimes.


## Outstanding problems (active)

- **LISTA-vs-generic_sparse quality gap persists after controlled matching**: at 50k steps with matched coefficients, LISTA remains substantially worse than `generic_sparse` on `duffing` (quick-best and long horizons), indicating the main gap is not resolved by loss coefficient matching alone.
- Queue `2` recovered target sparsity band with `sparsity_coeff=0.0005`/`0.0010`, but forecasting remains clearly behind the `generic_sparse` anchor.
- Queue `3` improved forecast means at low loop counts (`loops_1`) but pulled sparsity below target band; in-band loop settings remain weaker.
- Queue `4` identified in-band non-dominated settings, but the best one (`sp_0p0060_loops_1`) is still ~`3x` behind `generic_sparse` on long horizons.
- Queue `4` exposed unstable heavy-tail regimes for some arms; robust candidate filtering is still needed before promotion.
- Queue `5` adaptive-threshold sweep failed both objectives in this regime: no in-band sparsity candidates and long-horizon errors far above Queue-4 anchors.
- Unstructured LISTA parity is validated only on `duffing`; generalization to other systems/seeds is unverified.
- Intrinsic-HD `hopfield` remains unresolved at `L=8`, `TARGET_SIZE=256`: all three models fail at `H1000`, and LISTA variants show catastrophic tails.
- Dense LISTA is not currently viable on the intrinsic-HD baseline because `kuramoto` and `hopfield` rollouts are catastrophically unstable.
- Cross-system sparse LISTA transfer still has catastrophic long-horizon tails.
- Best-period collapse toward short reencoding periods indicates weak autonomous rollout stability.
- Label-free regime assignment from supports is not yet reliable enough for deployment-time control.
- Non-diagonal sequence training remains difficult to keep spectrally stable at larger capacities without explicit spectral constraints.
- The stricter intrinsic-HD plan sizes (`N=32/64`) are still untested; current results use milder defaults (`kuramoto=16`, `hopfield=16`, `competitive_lv=10`).
- LQR decision metrics/rules are still weakly discriminative under saturation and heavy-tailed outcomes.

## Queue Status

In progress:
- None.

Completed:
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

Planned next:
- Proceed to Queue `6` (encoder input-map variants), using Queue-4 in-band anchors as baselines.
- Keep Queue-5 as a completed negative control for the tested adaptive-threshold region.
- Re-evaluate against `generic_sparse` using the same Pareto criteria (`H1000`, `H500`, `quick-best`, `|sparsity-0.8|`).
- Queue the stricter intrinsic-HD plan variants (`N=32/64`) once the current-default baseline is accepted as the reference point.

## Core Experiment Log (Most Informative)

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
